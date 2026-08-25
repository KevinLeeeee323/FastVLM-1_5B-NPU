#!/usr/bin/env python3
"""Generate RKLLM calibration inputs for the local FastVLM checkpoint.

The RKLLM toolkit expects a JSON index and one pickle per sample.  FastVLM is
LLaVA-style: the ``<image>`` placeholder is replaced by the vision/projector
embeddings before the Qwen2 language model is called.  A pre-hook on the Qwen2
base model captures that final ``inputs_embeds`` tensor without running the
expensive language-model layers.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any


from common import DEFAULT_PROFILE, PROJECT_ROOT, checkpoint_path, load_profile, project_path


DEFAULT_DATASET = PROJECT_ROOT / "calibration/dataset.json"
FASTVLM_PYTHON = PROJECT_ROOT / "python/fastvlm_vendor"


class StopForward(RuntimeError):
    """Stop after the language-model input has been captured."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--sizes", type=int, nargs="+", choices=(512, 1024),
        help="Vision resolutions to include in one mixed calibration set.",
    )
    parser.add_argument(
        "--vision-compute-dtype", choices=("float32", "float16"), default="float32",
        help="CPU dtype for the temporary calibration vision forward (default: float32).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit source images (0 means all).")
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else project_path(path)


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def build_pixel_values(image_path: Path, processor: Any, size: int):
    """Apply FastVLM's RGB + black square padding at the selected resolution."""
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    side = max(width, height)
    padded = Image.new("RGB", (side, side), (0, 0, 0))
    padded.paste(image, ((side - width) // 2, (side - height) // 2))
    padded = padded.resize((size, size), Image.Resampling.BICUBIC)

    # MobileCLIP's processor is mean=0/std=1.  Temporarily changing its size
    # keeps preprocessing identical to the board path for both 512 and 1024.
    old_size = dict(processor.size)
    old_crop = dict(processor.crop_size)
    try:
        processor.size = {"shortest_edge": size}
        processor.crop_size = {"height": size, "width": size}
        values = processor.preprocess(padded, return_tensors="pt")["pixel_values"]
    finally:
        processor.size = old_size
        processor.crop_size = old_crop
    if tuple(values.shape) != (1, 3, size, size):
        raise RuntimeError(f"Unexpected preprocessed image shape {tuple(values.shape)} for {size}")
    return values.to(dtype=torch.float16)


def make_prompt(tokenizer: Any, question: str):
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token

    content = "<image>\n" + question
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    return prompt, ids.unsqueeze(0)


def tensor_for_pickle(value: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    value = value.detach().cpu()
    # RKLLM calibration examples use regular CPU tensors.  Keep masks and
    # positions intact, while making embeddings unambiguously float32.
    return value.float() if value.is_floating_point() else value


def main() -> int:
    args = parse_args()
    profile = load_profile(args.profile)
    model_path = resolve_path(args.model_path) if args.model_path else checkpoint_path(profile)
    dataset_path = resolve_path(args.dataset)
    output_dir = resolve_path(args.output_dir) if args.output_dir else project_path(profile["llm"]["calibration_directory"])
    sizes = args.sizes or profile["llm"]["calibration_resolutions"]
    require_file(model_path / "config.json", "FastVLM config")
    require_file(model_path / "model.safetensors", "FastVLM checkpoint")
    require_file(model_path / "tokenizer_config.json", "FastVLM tokenizer")
    require_file(dataset_path, "calibration dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"Output directory is not empty: {output_dir}; use --force to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = output_dir / "llm_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    if str(FASTVLM_PYTHON) not in sys.path:
        sys.path.insert(0, str(FASTVLM_PYTHON))
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from llava.model import LlavaQwen2ForCausalLM

    print(f"python={sys.executable}")
    print(f"torch={torch.__version__}")
    print(f"model={model_path}")
    print(f"profile={profile['model_id']}; sizes={sizes}")
    dtype = torch.float16
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    print("loading FastVLM with dtype=float16 on CPU")
    started = time.monotonic()
    model = LlavaQwen2ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map={"": "cpu"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    vision_tower = model.get_vision_tower()
    if args.vision_compute_dtype == "float32":
        # CPU FP16 convolution kernels can be unavailable or extremely slow.
        # This changes only the temporary calibration forward; the checkpoint
        # and RKLLM source loading contract remains dtype=float16.
        vision_tower.float()
        print("vision_compute_dtype=float32 (temporary CPU calibration path)")
    processor = vision_tower.image_processor
    print(f"model loaded in {time.monotonic() - started:.1f}s; hidden_size={model.config.hidden_size}")

    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("Calibration dataset has no samples")

    captured: list[dict[str, Any]] = []

    def capture(_module: Any, hook_args: tuple[Any, ...], hook_kwargs: dict[str, Any]) -> None:
        values = dict(hook_kwargs)
        if not values and hook_args:
            # Transformers may pass the first argument positionally on older
            # versions.  The current RKLLM environment uses keyword arguments.
            values["input_ids"] = hook_args[0]
        captured.append(values)
        raise StopForward

    handle = model.model.register_forward_pre_hook(capture, with_kwargs=True)
    index: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    try:
        for size in sizes:
            for idx, item in enumerate(records):
                image_path = Path(item["image_path"])
                if not image_path.is_absolute():
                    # The official manifest stores paths relative to the
                    # multimodal demo directory/repository (``data/datasets``),
                    # not relative to the JSON file itself.
                    candidates = (
                        dataset_path.parent / "images" / item["image"],
                        dataset_path.parent.parent / image_path,
                        PROJECT_ROOT / image_path,
                        dataset_path.parent / image_path,
                    )
                    direct = next((candidate for candidate in candidates if candidate.is_file()), None)
                    image_path = direct if direct is not None else (candidates[1] / item["image"])
                else:
                    image_path = image_path / item["image"]
                image_path = image_path.resolve()
                require_file(image_path, "calibration image")
                prompt, input_ids = make_prompt(tokenizer, item["input"])
                pixel_values = build_pixel_values(image_path, processor, size)
                captured.clear()
                print(f"capturing {size}px sample {idx}: {item['image']}", flush=True)
                try:
                    with torch.inference_mode():
                        model(
                            input_ids=input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            images=pixel_values,
                            image_sizes=[(size, size)],
                            use_cache=False,
                            return_dict=False,
                        )
                except StopForward:
                    pass
                if len(captured) != 1:
                    raise RuntimeError(f"Could not capture Qwen2 inputs for {size}/{idx}")
                sample = {key: tensor_for_pickle(value) for key, value in captured[0].items() if value is not None}
                embeds = sample.get("inputs_embeds")
                if embeds is None or embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[2] != 1536:
                    raise RuntimeError(f"Invalid inputs_embeds shape for {size}/{idx}: {getattr(embeds, 'shape', None)}")
                expected_image_tokens = (size // 64) ** 2
                text_token_count = int(input_ids.shape[1]) - 1
                actual_image_tokens = int(embeds.shape[1]) - text_token_count
                if actual_image_tokens != expected_image_tokens:
                    raise RuntimeError(
                        f"Image token mismatch for {size}/{idx}: expected {expected_image_tokens}, "
                        f"captured {actual_image_tokens} (sequence={embeds.shape[1]}, text={text_token_count})"
                    )
                sample_name = f"sample_{size}_{idx}"
                sample_path = inputs_dir / sample_name
                with sample_path.open("wb") as stream:
                    pickle.dump(sample, stream, protocol=pickle.HIGHEST_PROTOCOL)
                index.append({"sample": f"llm_inputs/{sample_name}", "token_nums": int(embeds.shape[1])})
                metadata.append({
                    "sample": sample_name,
                    "image": str(image_path),
                    "resolution": size,
                    "image_tokens": expected_image_tokens,
                    "sequence_tokens": int(embeds.shape[1]),
                    "prompt": prompt,
                })
                print(f"[{len(index):02d}] {size}px {item['image']} -> image_tokens={actual_image_tokens}, sequence_tokens={embeds.shape[1]}")
                captured.clear()
                del pixel_values, input_ids, sample
                gc.collect()
    finally:
        handle.remove()
        del model
        gc.collect()

    (output_dir / "inputs.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(index)} samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
