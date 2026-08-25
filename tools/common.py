from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = PROJECT_ROOT / "configs/fastvlm-1.5b-rk3588.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "models/manifest.json"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_profile(path: Path = DEFAULT_PROFILE) -> dict[str, Any]:
    profile = load_json(path.resolve())
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported profile schema: {}".format(profile.get("schema_version")))
    if profile.get("embedding_size") != 1536:
        raise ValueError("FastVLM Qwen2 embedding_size must be 1536")
    return profile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(profile: dict[str, Any]) -> Path:
    return project_path(profile["checkpoint"]["directory"])


def validate_checkpoint(profile: dict[str, Any], model_path: Path, verify_hashes: bool = False) -> None:
    if not model_path.is_dir():
        raise FileNotFoundError("FastVLM checkpoint directory not found: {}".format(model_path))
    required = profile["checkpoint"]["required_files"]
    for name, expected_hash in required.items():
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError("Checkpoint file not found: {}".format(path))
        if verify_hashes and sha256(path) != expected_hash:
            raise RuntimeError("Checkpoint checksum mismatch: {}".format(path))


def artifact_record(path: Path, **metadata: Any) -> dict[str, Any]:
    record = {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }
    record.update(metadata)
    return record


def collect_artifacts(profile: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"vision": [], "llm": None}
    for resolution, variant in profile["vision"]["resolutions"].items():
        path = project_path(variant["rknn"])
        if path.is_file():
            artifacts["vision"].append(artifact_record(
                path, resolution=int(resolution), image_tokens=variant["image_tokens"],
                embedding_size=profile["embedding_size"], precision=profile["vision"]["precision"]))
    llm = project_path(profile["llm"]["output"])
    if llm.is_file():
        artifacts["llm"] = artifact_record(
            llm, quantization=profile["llm"]["quantization"],
            build_max_context=profile["llm"]["build_max_context"],
            build_npu_cores=profile["llm"]["build_npu_cores"])
    return artifacts


def write_manifest(profile: dict[str, Any], artifacts: dict[str, Any], path: Path = DEFAULT_MANIFEST) -> None:
    manifest = {
        "schema_version": 1,
        "model_id": profile["model_id"],
        "target_platform": profile["target_platform"],
        "embedding_size": profile["embedding_size"],
        "multimodal": profile["multimodal"],
        "runtime": profile["runtime"],
        "llm_contract": {
            "quantization": profile["llm"]["quantization"],
            "build_max_context": profile["llm"]["build_max_context"],
            "build_npu_cores": profile["llm"]["build_npu_cores"],
        },
        "artifacts": artifacts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_current_manifest(profile: dict[str, Any], path: Path = DEFAULT_MANIFEST) -> None:
    artifacts = collect_artifacts(profile)
    expected_vision = len(profile["vision"]["resolutions"])
    if len(artifacts["vision"]) != expected_vision or artifacts["llm"] is None:
        raise RuntimeError("Every configured Vision RKNN and the RKLLM are required for a deployable manifest")
    write_manifest(profile, artifacts, path)
