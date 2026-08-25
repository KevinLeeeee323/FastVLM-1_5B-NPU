# Locally generated models

This source repository does not distribute prebuilt RKNN or RKLLM files. Follow the root
README on x86_64 Linux/WSL2 to download Apple's checkpoint and build all model artifacts
locally. Model conversion is not performed on RK3588.

The committed `manifest.json` describes the maintainer's verified reference build. After a
successful local conversion, `tools/export_rkllm.py` rewrites it with the sizes and SHA-256
hashes of the local model files. Copy that generated manifest and all three models to RK3588
as one set; do not rename or replace one artifact independently.
