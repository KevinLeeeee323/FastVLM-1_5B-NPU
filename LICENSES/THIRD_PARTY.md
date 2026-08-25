# Third-party sources

- The minimal `python/fastvlm_vendor/llava` conversion code comes from the FastVLM example
  in `airockchip/rknn3-model-zoo` and retains its Apple/LLaVA copyright headers. The source
  repository is Apache-2.0; see the project `LICENSE`.
- `runtime/include` and `runtime/lib` come from the RKLLM SDK 1.3.0 Linux aarch64 release.
  See `RKLLM-LICENSE` and do not mix them with RKLLM 1.2.x artifacts.
- `calibration/dataset.json` and `calibration/images` come from the RKLLM SDK 1.3.0 official
  `multimodal_model_demo` data directory and are used only to create quantization inputs.
- The original checkpoint is downloaded from Apple's public FastVLM dataset URL and is not
  stored in this Git repository. Users remain responsible for its applicable model terms.
