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

Run `inference-api serve --profile document` for the default BF16 document route, or `--profile adaptive` for budget routing. `--profile hybrid` and `--profile cpu-kv` explicitly opt into the constrained extensions described in the architecture document. `--batch-invariant` is available only with adaptive mode. The backend logs to `.local/logs/` and never replaces an occupied port.

By default the backend loads a pinned Hugging Face model revision and may download weights. For a previously cached local snapshot, export `INFERENCE_MODEL=/path/to/Qwen3-4B/snapshot`. No user-specific cache path is embedded in the package.

If a WSL Python installation requires separately supplied C headers for runtime compilation, set `PYTHON_DEV_HEADERS` to the directory containing `Python.h`. This affects only the owned backend environment; it does not modify system headers. An existing compatible backend can instead be used with `inference-api api --backend-url http://127.0.0.1:8001`.
