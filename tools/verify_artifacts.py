#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import (DEFAULT_MANIFEST, DEFAULT_PROFILE, load_json, load_profile,
                    project_path, sha256, write_current_manifest)


def verify(manifest_path: Path) -> None:
    manifest = load_json(manifest_path)
    for record in manifest["artifacts"]["vision"] + [manifest["artifacts"]["llm"]]:
        if record is None:
            raise RuntimeError("Manifest has no RKLLM artifact")
        path = project_path(record["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["size"]:
            raise RuntimeError("Artifact size mismatch: {}".format(path))
        actual = sha256(path)
        if actual != record["sha256"]:
            raise RuntimeError("Artifact checksum mismatch: {}".format(path))
        print("OK {} {}".format(record["sha256"], path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or verify the versioned FastVLM artifact manifest.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    manifest = args.manifest.expanduser().resolve()
    if args.write:
        write_current_manifest(profile, manifest)
        print("manifest written: {}".format(manifest))
    verify(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
