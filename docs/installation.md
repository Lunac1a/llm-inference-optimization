# Installation

The API package requires Python3.12+. Its CPU-only HTTP layer can be installed and tested independently:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

If an existing GPU environment was created with uv and has no pip module, use `uv pip install --python /path/to/that/environment/bin/python -e .` instead. This installs the API package into that chosen environment; it does not require installing pip or replacing the GPU runtime.

The tested GPU route uses Ubuntu24.04 under WSL2, NVIDIA GPU support, Python3.12, vLLM0.23.0, PyTorch2.11.0+cu130 and Transformers4.57.6. Keep the Linux GPU environment separate from a Windows virtual environment. In an existing pinned GPU environment, install this package without changing its runtime dependencies. For a new compatible Linux GPU environment, `python -m pip install -e '.[runtime]'` requests the pinned vLLM/Transformers versions; the matching CUDA/PyTorch build and driver must be selected for the host. See [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) for platform requirements.

Run `inference-api serve` for the default mixed-attention service. Use `--profile document` for the BF16 comparison. `--profile hybrid` selects the earlier full-prefill comparison; `--profile chunked-hybrid` explicitly selects the same mixed implementation as the default. The backend logs to `.local/logs/` and never replaces an occupied port. The supported profile list contains only these three attention paths.

Version 0.1.1 adds `inference-api serve --profile chunked-hybrid` as a separate
opt-in attention profile: historical FP8 KV, current BF16 chunks, fixed 4 GiB KV,
2,048-token chunks and 16,640-token maximum context. It retains native Decode and
uses the validated Prefill query tile of 64. It requires the pinned vLLM 0.23.0
environment and is not combined with CPU offload or batch invariance. See
[measurements and scope](attention-opt.md). This describes version 0.1.1; version 0.1.3 makes mixed attention the default.

By default the backend loads a pinned Hugging Face model revision and may download weights. For a previously cached local snapshot, export `INFERENCE_MODEL=/path/to/Qwen3-4B/snapshot`. No user-specific cache path is embedded in the package.

If a WSL Python installation requires separately supplied C headers for runtime compilation, set `PYTHON_DEV_HEADERS` to the directory containing `Python.h`. This affects only the owned backend environment; it does not modify system headers. An existing compatible backend can instead be used with `inference-api api --backend-url http://127.0.0.1:8001`.

Version 0.1.3 integrates the accepted single-loop source-branching kernel without new tuning. Version 0.1.2 removed budget routing and CPU-KV service extensions. See [archive and migration](archive.md).
