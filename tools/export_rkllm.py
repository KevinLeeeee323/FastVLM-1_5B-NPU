#!/usr/bin/env python3
"""Convert the FastVLM Qwen2 language component to an RKLLM model."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
import time
from pathlib import Path

from common import (DEFAULT_MANIFEST, DEFAULT_PROFILE, PROJECT_ROOT, checkpoint_path, load_profile,
                    project_path, sha256, write_current_manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--quantized-dtype", choices=("w8a8", "w4a16"))
    parser.add_argument("--target-platform")
    parser.add_argument("--num-npu-core", type=int)
    parser.add_argument("--max-context", type=int)
    parser.add_argument("--check-only", action="store_true", help="Run load_huggingface(load_weight=False) only.")
    parser.add_argument(
        "--allow-fp32-fallback", action="store_true",
        help="Allow the CPU Toolkit to fall back from requested float16 loading to float32.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output model.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else project_path(path)


def main() -> int:
    args = parse_args()
    profile = load_profile(args.profile)
    model_path = resolve(args.model_path) if args.model_path else checkpoint_path(profile)
    dataset = resolve(args.dataset) if args.dataset else project_path(profile["llm"]["calibration_directory"]) / "inputs.json"
    output = resolve(args.output) if args.output else project_path(profile["llm"]["output"])
    args.quantized_dtype = args.quantized_dtype or profile["llm"]["quantization"]
    args.target_platform = args.target_platform or profile["target_platform"]
    args.num_npu_core = args.num_npu_core or profile["llm"]["build_npu_cores"]
    args.max_context = args.max_context or profile["llm"]["build_max_context"]
    if output != project_path(profile["llm"]["output"]):
        try:
            profile["llm"]["output"] = str(output.relative_to(PROJECT_ROOT))
        except ValueError as exc:
            raise ValueError("--output must stay inside the standalone project so it can be manifested") from exc
    profile["llm"]["quantization"] = args.quantized_dtype
    profile["llm"]["build_npu_cores"] = args.num_npu_core
    profile["llm"]["build_max_context"] = args.max_context
    profile["target_platform"] = args.target_platform
    if not (model_path / "config.json").is_file() or not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"FastVLM checkpoint is incomplete: {model_path}")
    if not args.check_only and not dataset.is_file():
        raise FileNotFoundError(f"Calibration index not found: {dataset}; run tools/generate_calibration.py first")
    if not args.check_only:
        missing_vision = [
            project_path(variant["rknn"])
            for variant in profile["vision"]["resolutions"].values()
            if not project_path(variant["rknn"]).is_file()
        ]
        if missing_vision:
            missing = "\n".join(f"  {path}" for path in missing_vision)
            raise FileNotFoundError(
                f"Build every configured Vision RKNN before the RKLLM so the final manifest is complete:\n{missing}"
            )
    if not args.check_only and output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; use --force")
    if args.max_context <= 0 or args.max_context % 32:
        raise ValueError("--max-context must be a positive multiple of 32")
    if not 1 <= args.num_npu_core <= 3:
        raise ValueError("RK3588 supports 1 to 3 NPU cores")

    try:
        toolkit_version = importlib.metadata.version("rkllm-toolkit")
    except importlib.metadata.PackageNotFoundError:
        toolkit_version = "unknown"
    print(f"python={sys.executable}")
    print(f"python_version={platform.python_version()}")
    print(f"rkllm-toolkit={toolkit_version}")
    print(f"load_dtype={args.dtype}; quantization={args.quantized_dtype}; target={args.target_platform}")
    print(f"model={model_path}; dataset={dataset}; output={output}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("warning: PYTHONNOUSERSITE is not 1; ROS/user packages may contaminate imports")

    from rkllm.api import RKLLM

    llm = RKLLM()
    started = time.monotonic()
    print("--> load_huggingface")
    ret = llm.load_huggingface(
        model=str(model_path), device="cpu", dtype=args.dtype, load_weight=not args.check_only
    )
    print(f"load_huggingface return={ret} elapsed={time.monotonic() - started:.1f}s")
    if ret != 0:
        return int(ret)
    # RKLLM-Toolkit 1.3.0 emits this fallback on the CPU-only WSL host. It is
    # not represented by a distinct return code, so do not silently claim an
    # FP16 source load or start a high-memory build without explicit consent.
    effective_dtype = args.dtype
    if args.dtype == "float16":
        effective_dtype = "float32 (Toolkit CPU fallback)"
        print("WARNING: RKLLM-Toolkit CPU does not support float16 and fell back to float32")
        if not args.allow_fp32_fallback:
            print("ERROR: refusing to continue; use --dtype float32 or explicitly pass --allow-fp32-fallback")
            return 3
    print(f"effective_load_dtype={effective_dtype}")
    if args.check_only:
        print("check-only passed")
        return 0

    started = time.monotonic()
    print("--> build")
    ret = llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype=args.quantized_dtype,
        quantized_algorithm="normal",
        target_platform=args.target_platform,
        num_npu_core=args.num_npu_core,
        dataset=str(dataset),
        max_context=args.max_context,
    )
    print(f"build return={ret} elapsed={time.monotonic() - started:.1f}s")
    if ret != 0:
        return int(ret)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    # Toolkit 1.3.0 may append `.rkllm` to either a temporary name or an
    # already suffixed destination. Account for both observed behaviours.
    appended_temporary = temporary.with_name(temporary.name + ".rkllm")
    normalized_output = output.with_name(output.name + ".rkllm")
    staging_paths = (temporary, appended_temporary, normalized_output)
    stale = next((path for path in staging_paths if path.exists()), None)
    if stale is not None and not args.force:
        raise FileExistsError(f"Stale RKLLM export path exists: {stale}; use --force to replace it")
    if args.force:
        for path in staging_paths:
            path.unlink(missing_ok=True)
    started = time.monotonic()
    print(f"--> export_rkllm {temporary}")
    ret = llm.export_rkllm(str(temporary))
    print(f"export_rkllm return={ret} elapsed={time.monotonic() - started:.1f}s")
    if ret != 0:
        temporary.unlink(missing_ok=True)
        return int(ret)
    produced = next(
        (path for path in staging_paths
         if path.is_file() and path.stat().st_size > 0),
        None,
    )
    if produced is None:
        for path in staging_paths:
            path.unlink(missing_ok=True)
        raise RuntimeError("Toolkit reported success but produced no non-empty RKLLM file")
    if produced != output:
        os.replace(produced, output)
    print(f"output_size={output.stat().st_size} sha256={sha256(output)}")
    write_current_manifest(profile, DEFAULT_MANIFEST)
    print(f"manifest={DEFAULT_MANIFEST}")
    print(f"completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
