#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from common import DEFAULT_PROFILE, PROJECT_ROOT, checkpoint_path, load_profile, validate_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Apple's original FastVLM-1.5B checkpoint.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--archive", type=Path, default=PROJECT_ROOT / "artifacts/downloads/llava-fastvithd_1.5b_stage3.zip")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-large-hash", action="store_true", help="Also hash the 3.8 GB safetensors file.")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    destination = checkpoint_path(profile)
    if args.verify_only:
        validate_checkpoint(profile, destination, verify_hashes=args.verify_large_hash)
        print("checkpoint=OK path={}".format(destination))
        return 0

    archive = args.archive.expanduser().resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        curl = shutil.which("curl")
        if curl is None:
            raise RuntimeError("curl is required for the resumable checkpoint download")
        subprocess.run([curl, "-fL", "--retry", "5", "-C", "-", "-o", str(archive),
                        profile["checkpoint"]["url"]], check=True)
    if destination.exists():
        validate_checkpoint(profile, destination, verify_hashes=args.verify_large_hash)
        print("checkpoint already extracted: {}".format(destination))
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if not name.startswith("__MACOSX/")]
        bundle.extractall(destination.parent, members=members)
    validate_checkpoint(profile, destination, verify_hashes=args.verify_large_hash)
    print("checkpoint ready: {}".format(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
