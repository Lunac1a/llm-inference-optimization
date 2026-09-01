# Qwen3 Cache-Specific INT8 KV Cache

A reproducible LLM inference optimization project targeting KV-cache memory
traffic and capacity on an NVIDIA RTX 4090 Laptop GPU.

The planned custom optimization is `q8_kv_128`: one asymmetric INT8 block per
128-value Qwen3 KV head. It will be compared against F16 KV cache as the primary
baseline and upstream `q8_0` KV cache as a control. The custom cache format is
**planned but not implemented yet**.

## Current status

| Milestone | Status | Meaning |
| --- | --- | --- |
| Stage 0: reproducible environment | Passed | Unmodified upstream CUDA build and baseline smoke tests are verified |
| Stage 1: bottleneck confirmation | Not started | No formal bottleneck or performance claim yet |
| Stage 2: CPU reference format | Not started | `q8_kv_128` is not implemented |
| Stage 3: CUDA integration | Not started | No custom CUDA path exists yet |
| Stages 4-6: benchmark and replay | Not started | No resume metric has been validated |

See [the Stage 0 validation report](docs/stage0-validation.md) for the locked
inputs, commands, test results, and raw-evidence references.

## Stage 0 result

The unmodified `llama.cpp` baseline was built with CUDA for Ada SM 89 and tested
with the official Qwen3-8B Q4_K_M GGUF model. Both F16 and upstream `q8_0` KV
caches completed full-GPU-offload generation with Flash Attention enabled.

At a 2,048-token cache allocation:

| Cache type | KV allocation | Role |
| --- | ---: | --- |
| F16 | 288.00 MiB | Primary baseline |
| upstream q8_0 | 153.00 MiB | Existing control |

The observed 46.875% cache reduction belongs to the **upstream q8_0 control**;
it is not a result of this project's planned custom implementation.

## Reproducibility locks

- `llama.cpp`: `f8dbcd61893702976f9ab03be89c2b9f436d532c`
- Model revision: `7c41481f57cb95916b40956ab2f0b139b296d974`
- Model SHA-256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`
- Platform: Windows AMD64, CUDA 13.3, SM 89

Machine-readable details are stored in [`locks/`](locks/). The 5 GB model,
local build outputs, virtual environments, and IDE files are intentionally not
tracked.

## Repository layout

```text
artifacts/stage0/   Raw Stage 0 validation logs
docs/               Validation and design documentation
engine/llama.cpp/   Pinned upstream Git submodule
locks/              Machine-readable source, model, and environment locks
```

Clone with the pinned engine source:

```powershell
git clone --recurse-submodules <repository-url>
```

## Project boundary

The custom work is limited to a new KV data format, CPU reference routines,
CUDA quantize-on-write, and the matching Flash Attention K/V paths. It will not
change Q4_K_M model weights, the Qwen3 architecture, general GEMM kernels,
RoPE, or the scheduler.

Benchmark claims will only be added after the preregistered protocol and an
independent clean-build rerun pass their acceptance criteria.
