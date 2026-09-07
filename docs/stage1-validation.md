# Stage 1: local vLLM baseline

Status on 2026-09-07: **incomplete; preflight checked, Linux setup pending**.
This is a checkpoint, not a passed baseline or performance result.

## Observed environment

- Windows 11 Pro, version 10.0.26200.
- Intel Core i9-13980HX; firmware virtualization and SLAT reported enabled.
- Approximately 31.62 GiB usable system RAM.
- NVIDIA RTX 4090 Laptop GPU, 16,376 MiB VRAM, compute capability 8.9.
- Driver 596.49. NVIDIA-SMI advertises CUDA 13.2 driver compatibility; this is
  not an installed Linux CUDA runtime or PyTorch version.
- `wsl --status` exits 50 and reports that WSL is not installed.
- Docker was not found on PATH; this does not prove no installation exists.
- Windows optional-feature inspection requires an elevated administrator process;
  VirtualMachinePlatform and WSL feature states remain unverified.

The raw command outcomes are recorded in
[preflight evidence](../artifacts/stage1/preflight.json).

## Feasibility and proposed configuration

The current official vLLM GPU requirements specify Linux and NVIDIA compute
capability >= 7.5. The GPU clears this hardware check, but actual Linux GPU
access, package compatibility, and model fit still require verification.
Official vLLM does not support native Windows and identifies WSL as an option.

Proposed route: WSL2 with Ubuntu 24.04, a separate Linux Python 3.12 environment,
and an exact stable vLLM package version selected after checking its release
dependencies against the available driver. No package version is locked yet.
Do not reuse the Windows virtual environment.

Candidate baseline: Qwen/Qwen3-4B in BF16, initially capped at 4,096 context tokens
and low concurrency. Rough weight memory alone is about 8 GB (4 billion parameters
times two bytes); runtime, activations, and KV cache add to that. This is a sizing
estimate, not a successful load. Model revision and final parameters are pending.
Keep existing GGUF weights; they are not the proposed BF16 baseline.

## Next steps

1. Install WSL2 and Ubuntu 24.04 in an administrator PowerShell session:

   ```powershell
   wsl --install -d Ubuntu-24.04 --no-launch
   ```

   This changes Windows system components and may require a restart. Let the
   user schedule the restart and complete the Ubuntu account setup. No automatic
   restart or system installation was performed in this checkpoint.
2. Verify `wsl --list --verbose` reports version 2 and `nvidia-smi` works inside
   Ubuntu before installing vLLM. Verify the distribution name if installation
   reports it unavailable; use `wsl --list --online`.
3. Pin the software/model revisions, create the isolated environment, and record
   the resolved dependencies. Keep the API bound to localhost for the baseline.
4. Run real API generation and a small repeated baseline, preserving requests,
   outputs, configuration, and timing. Only then evaluate Stage 1 acceptance.

No GPU rental, model download, dependency installation, optimization code, or
real-model benchmark was performed. Cloud spending for this checkpoint: AUD 0.

## Official sources checked

- [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Microsoft WSL commands](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)
- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
