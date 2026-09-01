# Stage 0: Environment and upstream baseline validation

Status: **passed** on 2026-09-01 (Australia/Sydney).

This stage validates only the unmodified upstream engine and the reproducible
environment. It does not claim that the custom KV cache is implemented or that
any benchmark has passed.

## Locked inputs

- Upstream: `ggml-org/llama.cpp` at
  `f8dbcd61893702976f9ab03be89c2b9f436d532c` (build 10720).
- Model: official `Qwen/Qwen3-8B-GGUF`, revision
  `7c41481f57cb95916b40956ab2f0b139b296d974`.
- Model file: `Qwen3-8B-Q4_K_M.gguf`, 5,027,783,488 bytes.
- Model SHA-256:
  `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.

Machine-readable values are recorded in `locks/`.

## Baseline build

The unmodified upstream source was configured as a Release CUDA build for Ada
SM 89:

```text
cmake -S . -B build-baseline -G Ninja -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build-baseline --target llama-cli --parallel 8
```

The resulting executable reports commit `f8dbcd618`, Windows AMD64, CUDA
architecture 890, and detects the RTX 4090 Laptop GPU.

## Acceptance evidence

Both smoke tests used the same model, executable, prompt, context size, seed,
sampling temperature, full GPU offload, and Flash Attention. Only the KV cache
type changed.

| Check | F16 KV | Upstream q8_0 KV |
| --- | ---: | ---: |
| Full GPU offload | 37/37 layers | 37/37 layers |
| Flash Attention | enabled | enabled |
| Context cells | 2,048 | 2,048 |
| K allocation | 144.00 MiB | 76.50 MiB |
| V allocation | 144.00 MiB | 76.50 MiB |
| Total KV allocation | 288.00 MiB | 153.00 MiB |
| Generation completed | yes | yes |

The observed q8_0 cache reduction is exactly 46.875% versus F16:

```text
1 - 153 / 288 = 0.46875
```

This is an upstream control result, not a custom optimization result.

Raw evidence:

- `artifacts/stage0/stage0-f16-verbose.log`
- `artifacts/stage0/stage0-q8_0-verbose.log`

## Upstream tests

- `test-quantize-fns`: passed with exit code 0, including the upstream `q8_0`
  quantize/dequantize functions used by the control cache.
- `test-backend-ops test -b CUDA0 -j 4`: 14,595/14,595 supported CUDA
  operation cases passed; backend `CUDA0: OK`; process exit code 0.
- Cases reported as `not supported` are capability probes skipped by the
  upstream harness and are not failed correctness tests.

Test logs:

- `artifacts/stage0/test-quantize-fns.log`
- `artifacts/stage0/test-backend-ops-cuda.log`

## Stage boundary

Stage 0 is complete. The following remain explicitly unverified:

- KV cache bottleneck profiling and formal baseline benchmark (Stage 1).
- `q8_kv_128` reference format and tests (Stage 2).
- Custom CUDA implementation (Stage 3).
- Formal benchmark and reproducibility claims (Stages 4-6).
