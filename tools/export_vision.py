#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_PROFILE, PROJECT_ROOT, checkpoint_path, load_profile, project_path, validate_checkpoint


VENDOR = PROJECT_ROOT / "python/fastvlm_vendor"


def can_import(python: Path, module: str) -> bool:
    return subprocess.run([str(python), "-c", "import {}".format(module)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def toolkit_python(value: str | None) -> Path:
    candidates = [value, os.environ.get("RKNN_TOOLKIT2_PYTHON"), sys.executable]
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if path.is_file() and can_import(path, "rknn.api"):
                return path
    raise RuntimeError("RKNN-Toolkit2 Python not found; pass --toolkit-python or set RKNN_TOOLKIT2_PYTHON")


def export_onnx(model_path: Path, output: Path, resolution: int, force: bool) -> None:
    if output.exists() and not force:
        print("reusing ONNX: {}".format(output))
        return
    sys.path.insert(0, str(VENDOR))
    import torch
    from transformers import AutoConfig
    from llava.model import LlavaQwen2ForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    model = LlavaQwen2ForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, low_cpu_mem_usage=True
    ).eval()

    class VisionProjector(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.vision_tower = source.get_vision_tower()
            self.projector = source.get_model().mm_projector

        def forward(self, pixel):
            return self.projector(self.vision_tower(pixel))

    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = VisionProjector(model).eval()
    sample = torch.randn(1, 3, resolution, resolution)
    torch.onnx.export(wrapper, sample, output, input_names=["pixel"], output_names=["image_embeddings"],
                      opset_version=17)
    print("ONNX created: {}".format(output))


def build_rknn(onnx: Path, output: Path, force: bool, verbose: bool) -> None:
    if output.exists() and not force:
        raise FileExistsError("RKNN exists; use --force: {}".format(output))
    from rknn.api import RKNN

    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".build.log")
    rknn = RKNN(verbose=verbose, verbose_file=str(log) if verbose else None)
    try:
        ret = rknn.config(target_platform="rk3588", mean_values=[[0.0, 0.0, 0.0]],
                          std_values=[[255.0, 255.0, 255.0]], float_dtype="float16",
                          optimization_level=3)
        if ret != 0: raise RuntimeError("rknn.config failed: {}".format(ret))
        ret = rknn.load_onnx(model=str(onnx))
        if ret != 0: raise RuntimeError("rknn.load_onnx failed: {}".format(ret))
        ret = rknn.build(do_quantization=False)
        if ret != 0: raise RuntimeError("rknn.build failed: {}".format(ret))
        ret = rknn.export_rknn(str(output))
        if ret != 0: raise RuntimeError("rknn.export_rknn failed: {}".format(ret))
    finally:
        rknn.release()
    print("RKNN created: {}".format(output))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export standalone FastVLM Vision + projector for RK3588.")
    parser.add_argument("stage", choices=("check", "onnx", "rknn", "all"), nargs="?", default="all")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--resolution", type=int, choices=(512, 1024))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--toolkit-python")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    resolution = args.resolution or profile["vision"]["default_resolution"]
    variant = profile["vision"]["resolutions"][str(resolution)]
    model_path = args.model_path.expanduser().resolve() if args.model_path else checkpoint_path(profile)
    onnx = project_path(variant["onnx"])
    rknn = project_path(variant["rknn"])
    validate_checkpoint(profile, model_path)
    print("profile={} resolution={} tokens={} embedding={}".format(
        profile["model_id"], resolution, variant["image_tokens"], profile["embedding_size"]))
    if args.stage == "check":
        print("checkpoint=OK toolkit_python={}".format(toolkit_python(args.toolkit_python)))
        return 0
    if args.stage in ("onnx", "all"):
        export_onnx(model_path, onnx, resolution, args.force)
    if args.stage in ("rknn", "all"):
        python = toolkit_python(args.toolkit_python)
        if Path(sys.executable).resolve() != python:
            command = [str(python), str(Path(__file__).resolve()), "rknn", "--profile", str(args.profile),
                       "--resolution", str(resolution), "--model-path", str(model_path),
                       "--toolkit-python", str(python)]
            if args.force: command.append("--force")
            if args.verbose: command.append("--verbose")
            subprocess.run(command, check=True)
        else:
            if not onnx.is_file(): raise FileNotFoundError("ONNX not found: {}".format(onnx))
            build_rknn(onnx, rknn, args.force, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
