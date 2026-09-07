# Local vLLM environment

The user installed Ubuntu-24.04 under WSL2 and the base development tools.
This task added the isolated inference runtime, using prebuilt packages rather
than a system-wide installation or a source build.

## Verified configuration

- Ubuntu 24.04.4 LTS, x86_64, WSL2 kernel 6.18.33.2.
- Python 3.12.3; uv 0.12.10; Git 2.43.0; GCC/G++ 13.3 and Make available.
- RTX 4090 Laptop GPU, 16,376 MiB VRAM, driver 596.49, compute capability 8.9.
- WSL has about 16 GiB RAM and 4 GiB swap. Its memory limit was not changed.
- Environment: `/home/lunacia/.venvs/inference-vllm`, stored on the Linux filesystem.
- vLLM 0.23.0, PyTorch 2.11.0+cu130, Transformers 4.57.6, Triton 3.6.0.
- Ninja 1.13.2, FastAPI, API clients, model-loading libraries, and the required
  NVIDIA runtime packages were installed through vLLM's dependency resolution.

vLLM 0.23.0 was selected as a fixed release with versioned documentation and
matching PyTorch/torchaudio/torchvision dependencies. It is not claimed to be the
latest release. The explicitly selected PyTorch backend is CUDA 13.0; successful
GPU/kernel checks below provide local compatibility evidence beyond metadata.

## Use and reproduce

Run in Ubuntu, not Windows PowerShell:

```sh
source ~/.venvs/inference-vllm/bin/activate
vllm --help
cd /mnt/d/Lunacia/Inference
python scripts/check-vllm-environment.py
```

Full resolved versions are in `locks/requirements-wsl-cu130.txt`. With Python 3.12
and uv installed, recreate the project environment using:

```sh
cd /mnt/d/Lunacia/Inference
bash scripts/setup-vllm-env.sh
```

The script synchronizes its target environment to the lock, removing packages
outside that lock. Use only the dedicated project environment. To use a different
dedicated directory, set `INFERENCE_VENV` before running it. Package versions are
pinned; the file is not a hash-verified artifact lock or a cloud validation.

The system Python does not have pip; uv manages the project packages directly,
so installing system pip is unnecessary. Docker and a system CUDA toolkit are
not required for this prebuilt runtime checkpoint and were not added. A future
CUDA source build may require additional compiler/header tooling. No Linux GPU
driver was installed; GPU access uses the Windows host driver through WSL.

## Checks and limits

Passed: dependency consistency (193 packages), CUDA discovery, BF16 support,
deterministic GPU matrix multiplication, vLLM's compiled RMSNorm against a
PyTorch reference, and `vllm --help`. The setup script was syntax checked.

The CLI emitted two non-fatal warnings: the resolved Transformers v4 path is
deprecated, and WSL causes vLLM to disable pinned host memory. Preserve the
resolved stack for this checkpoint; reassess Transformers compatibility before
an engine upgrade. WSL measurements must not be assumed equivalent to native
Linux/cloud measurements because the host-memory behavior differs.

No model weights were downloaded, no model was loaded, no API generation was
run, and no performance baseline was measured in this task. Model loading will
also test whether the current WSL RAM allocation is sufficient.

Evidence is under `artifacts/stage1/`: `wsl-install.log`, `dependency-check.log`,
`wsl-runtime-check.json`, `wsl-runtime-check.stderr.log`, and `vllm-help.log`.

Sources: [versioned vLLM installation documentation](https://docs.vllm.ai/en/v0.23.0/getting_started/installation/gpu/)
and [PyPI release metadata](https://pypi.org/pypi/vllm/0.23.0/json).
