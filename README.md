# FastVLM-1.5B NPU for RK3588

Standalone conversion and board-side inference for Apple's FastVLM-1.5B on RK3588. The Vision tower and multimodal projector run as an FP16 RKNN model; the Qwen2 1.5B language component runs as a W8A8 RKLLM model.

This repository does not require a checkout of `rknn3-model-zoo`. It vendors only the small FastVLM/LLaVA/MobileCLIP Python subset needed during conversion. Rockchip's x86_64 conversion toolkits still have to be installed from their official releases.

This is a source-only project. It does not distribute prebuilt `.rknn` or `.rkllm` files in
Git or as release assets. Building the models locally on x86_64 Linux/WSL2, as documented
below, is the only supported way to obtain them. RK3588 is the deployment target, not the
model-conversion host.

## Tested contract

| Component | Contract |
| --- | --- |
| Target | RK3588, Linux aarch64 |
| Vision | FP16 RKNN, 1024 -> 256x1536; 512 -> 64x1536 |
| LLM | Qwen2 1.5B, W8A8 RKLLM |
| RKLLM build | Toolkit 1.3.0, `max_context=4096`, `num_npu_core=3` |
| Runtime | RKLLM 1.3.0 plus the paired RKNN runtime in `runtime/lib` |
| Image marker | `image_start=""`, `image_end=""`, `image_content="<image>"` |
| Preprocessing | OpenCV BGR -> RGB, black square padding, resize; UINT8 NHWC input, RKNN applies `/255` |

The source configuration is [configs/fastvlm-1.5b-rk3588.json](configs/fastvlm-1.5b-rk3588.json).
The committed [models/manifest.json](models/manifest.json) records the maintainer's verified
build as a reference contract. A successful local RKLLM export rewrites it with the sizes and
SHA-256 hashes of the locally generated models. Keep the profile, resulting manifest and model
files together when deploying to a board.

## Repository layout

```text
configs/       single conversion/runtime profile
tools/         download, conversion and verification commands
python/        minimal vendored FastVLM model implementation
calibration/   RKLLM multimodal calibration manifest and source images
src, include/  RK3588 C++ application
runtime/       matched RKLLM/RKNN 1.3.0 aarch64 headers and libraries
models/        reference manifest in Git; locally generated models are ignored
Pic/           test images
artifacts/     ignored local checkpoint, ONNX and generated calibration data
```

## Build the models on x86_64 Linux or WSL2

Complete all five steps in order. The process downloads Apple's original checkpoint, builds
both Vision variants, generates multimodal calibration inputs and finally creates the W8A8
RKLLM. Do not copy the repository to RK3588 until this section has completed successfully.

Plan for at least 12 GB of free disk space. The verified WSL2 build used a 10 GB RAM limit,
8 GB swap and approximately 9.29 GiB peak resident memory during RKLLM conversion. Close
memory-heavy Windows and WSL applications before that stage; a host with more RAM is preferable.

Use three separate Conda environments. Match each Toolkit environment to the CPython ABI in its official wheel; the verified setup uses Python 3.8 for RKNN-Toolkit2 2.3.2 and Python 3.10 for Vision export and RKLLM-Toolkit 1.3.0. Do not combine these dependency sets: RKNN-Toolkit2 and RKLLM-Toolkit pin incompatible versions of several packages.

### 1. Install the environments

Vision/ONNX export:

```bash
conda create -n FastVLM-Vision python=3.10 -y
conda activate FastVLM-Vision
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/vision-export.txt
```

RKNN-Toolkit2 2.3.2:

```bash
conda create -n RKNN-Toolkit2 python=3.8 -y
conda activate RKNN-Toolkit2
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/rknn-toolkit2.txt
python -m pip install /path/to/rknn_toolkit2-2.3.2-cp38-*-linux_x86_64.whl
export RKNN_TOOLKIT2_PYTHON="$CONDA_PREFIX/bin/python"
```

RKLLM-Toolkit 1.3.0:

```bash
conda create -n RKLLM-Toolkit-1.3 python=3.10 -y
conda activate RKLLM-Toolkit-1.3
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/rkllm-toolkit.txt
python -m pip install /path/to/rkllm_toolkit-1.3.0-cp310-cp310-linux_x86_64.whl
```

Obtain the wheels from the official projects:

- RKNN-Toolkit2: <https://github.com/airockchip/rknn-toolkit2>
- RKLLM 1.3.0: <https://github.com/airockchip/rknn-llm/releases/tag/release-v1.3.0>

### 2. Download the original checkpoint

Apple publishes the checkpoint as a zip file. The helper resumes interrupted downloads and validates the extracted config; add `--verify-large-hash` to hash the 3.8 GB safetensors file.

```bash
conda activate FastVLM-Vision
python tools/download_model.py
python tools/download_model.py --verify-only --verify-large-hash
```

Default checkpoint directory:

```text
artifacts/checkpoint/llava-fastvithd_1.5b_stage3/
```

### 3. Convert Vision to RKNN

Run the ONNX stage in the Vision environment and the RKNN stage in RKNN-Toolkit2. The `all` command switches to `$RKNN_TOOLKIT2_PYTHON` for the compiler subprocess.

```bash
conda activate FastVLM-Vision
export RKNN_TOOLKIT2_PYTHON=/path/to/conda/envs/RKNN-Toolkit2/bin/python

python tools/export_vision.py all --resolution 1024 --verbose
python tools/export_vision.py all --resolution 512 --verbose
```

The default outputs come from the profile. Existing RKNN files are protected; pass `--force` only for an intentional rebuild.

### 4. Generate RKLLM calibration data

This creates 40 samples: the 20 included source examples at both 1024/256-token and 512/64-token resolutions. Generated embeddings occupy about 65 MB under `artifacts/` and are not committed.

```bash
conda activate RKLLM-Toolkit-1.3
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python tools/generate_calibration.py
```

### 5. Convert the language component to RKLLM

First run the structure check. On a CPU-only host, Toolkit 1.3.0 falls back from requested FP16 source loading to FP32. The explicit flag acknowledges its higher memory cost; the final artifact is still W8A8.

```bash
python tools/export_rkllm.py --check-only --allow-fp32-fallback

/usr/bin/time -v python -u tools/export_rkllm.py \
  --allow-fp32-fallback
```

The verified build used approximately 9.29 GiB peak RSS on WSL2. The exporter writes
`models/manifest.json` only after the RKLLM has been exported successfully and all required
model files are present. Verify the complete set with:

```bash
python tools/verify_artifacts.py
```

The completed local model set is:

```text
models/FastVLM-vision-1024-fp16-rk3588.rknn
models/FastVLM-vision-rk3588.rknn
models/FastVLM-1.5B-w8a8-rk3588.rkllm
models/manifest.json
```

Do not run `verify_artifacts.py` immediately after cloning: the committed manifest describes
the reference build, while the corresponding binaries are deliberately absent. Run it after
the conversion sequence above, when the exporter has updated the manifest for local outputs.

## Build and run on RK3588

Copy the repository together with the four generated files listed above to the board. Install
CMake, a C++ compiler and OpenCV development files, then build natively:

```bash
cd FastVLM-1_5B-NPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
cp build/FastVLM_NPU .
```

For cross compilation, pass an aarch64 toolchain through `CMAKE_C_COMPILER` and
`CMAKE_CXX_COMPILER`. Use `-DFASTVLM_OPENCV_DIR=/path/to/OpenCV` when supplying Rockchip's aarch64 OpenCV package. An x86_64 WSL compiler cannot link the included aarch64 runtime `.so`.

Run the default 1024 model:

```bash
export LD_LIBRARY_PATH="$PWD/runtime/lib:$LD_LIBRARY_PATH"

./FastVLM_NPU \
  --manifest models/manifest.json \
  --image Pic/Pizza.jpg \
  --vision models/FastVLM-vision-1024-fp16-rk3588.rknn \
  --llm models/FastVLM-1.5B-w8a8-rk3588.rkllm \
  --max-new-tokens 256 \
  --max-context 4096 \
  --vision-cores 3
```

Use `models/FastVLM-vision-rk3588.rknn` for the 512 fallback. A pure-text smoke test is:

```bash
./FastVLM_NPU \
  --manifest models/manifest.json \
  --text-only \
  --llm models/FastVLM-1.5B-w8a8-rk3588.rkllm \
  --prompt '用一句话介绍你自己。' \
  --once
```

The executable prefixes `<image>` to the first visual prompt when it is omitted. Later text-only prompts retain history. Type `clear` to call `rkllm_clear_kv_cache`; type `exit` to quit.

## Parameter compatibility

| Parameter | Set when | Rule |
| --- | --- | --- |
| `llm.build_max_context` | RKLLM conversion | Hard maximum embedded in the RKLLM artifact |
| `--max-context` | Inference | May be lower, but must not exceed the manifest's build maximum |
| `--max-new-tokens` | Inference | May change; prompt, image tokens, history and output must fit the context |
| `llm.build_npu_cores` | RKLLM conversion | Build property of the RKLLM artifact |
| `--vision-cores` | Vision inference | RKNN scheduling choice 1/2/3; does not change model precision |
| `vision.resolutions` | Vision conversion | Must agree with RKNN shape and 64/256 image-token contract |
| W8A8 / FP16 | Conversion | Artifact property; cannot be changed at inference time |

The executable reads `models/manifest.json`. It rejects an unknown model filename, an embedding width other than 1536, a runtime contract other than RKLLM 1.3.0/RK3588, and a `--max-context` larger than the RKLLM build maximum. If CLI values are omitted, defaults come from the manifest.

To change context, quantization, core count or output names reproducibly, edit a copy of the
profile before conversion and keep the resulting model files with its generated manifest. Do
not edit only the README or rename model files after manifest generation.

## Publishing the source repository

Commit the source, calibration source data, runtime headers/libraries and the reference
`models/manifest.json`. The repository `.gitignore` deliberately excludes:

```text
artifacts/                downloaded checkpoint, ONNX and generated calibration data
models/*.rknn             generated Vision models
models/*.rkllm            generated language model
*.build.log               local Toolkit logs
```

Do not attach those generated models to a GitHub Release for this project. Developers obtain
them through the documented build process. Preserve [LICENSE](LICENSE) and
[LICENSES/THIRD_PARTY.md](LICENSES/THIRD_PARTY.md) when redistributing the runtime or vendored
conversion code.
