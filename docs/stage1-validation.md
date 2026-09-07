# Stage 1: local vLLM baseline

Status: **passed on 2026-09-08**. Stage 2 bottleneck investigation and all later
stages remain unstarted.

## Scope and fixed configuration

- Host: Windows 11 with Ubuntu-24.04 WSL2; one RTX 4090 Laptop GPU, 16,376 MiB
  VRAM, driver 596.49.
- Runtime: Python 3.12.3, vLLM 0.23.0, PyTorch 2.11.0+cu130, Transformers 4.57.6,
  Triton 3.6.0, isolated at `/home/lunacia/.venvs/inference-vllm`.
- Model: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`; the downloaded snapshot and SHA256
  manifest are in [model-lock.json](../artifacts/stage1/model-lock.json).
- Service: `127.0.0.1:8000`, served name `qwen3-4b-baseline`, BF16, tensor parallel
  size 1, maximum context 4096, GPU memory utilization 0.80, maximum sequences 4,
  maximum batched tokens 2048.
- Prefix caching was explicitly disabled; chunked prefill was enabled; speculative
  decoding and CPU offload were not used. `--generation-config vllm` makes the
  benchmark's temperature 0 explicit instead of inheriting the model card defaults.
- Actual attention backend: FlashAttention 2. Startup reported 5.02 GiB available
  KV-cache memory, 36,528 GPU KV-cache tokens, and 8.92x maximum concurrency for
  4096-token requests.

The service also required two WSL compatibility switches: the V2 model runner was
disabled because WSL reported that UVA was unavailable, and the FlashInfer sampler
was disabled because its JIT path required an unavailable `nvcc`. `--enforce-eager`
was used because the WSL image did not contain Python development headers for the
torch compiler helper. User-space Ubuntu `.deb` contents were extracted under the
ignored `.tmp/python-dev/`; no system-wide package installation or sudo change was
made.

## Acceptance evidence

- Environment and GPU checks: [wsl-runtime-check.json](../artifacts/stage1/wsl-runtime-check.json)
  and [environment.md](environment.md).
- Model snapshot manifest: [model-lock.json](../artifacts/stage1/model-lock.json).
- Final API run: [api-check-20260907T141629Z.json](../artifacts/stage1/api/api-check-20260907T141629Z.json),
  17/17 checks passed. This covered health, model discovery, Chinese/English/
  arithmetic prompts, normal and streaming chat completions, invalid-model and
  oversized-context errors, post-error health, and generation after restart.
- Final merged benchmark report:
  [final-benchmark-summary.json](../artifacts/stage1/final-benchmark-summary.json).
  Every selected configuration completed 32/32 requests with zero failed or
  detailed-error requests. The selected three-round CV values were:

  | Random input | Concurrency | Output throughput mean | CV |
  | ---: | ---: | ---: | ---: |
  | 512 | 1 | 37.32 tok/s | 4.13% |
  | 512 | 4 | 141.15 tok/s | 4.51% |
  | 2048 | 1 | 36.41 tok/s | 1.08% |
  | 2048 | 4 | 109.28 tok/s | 2.55% |

  All CV values are at or below the 5% acceptance threshold. The report retains
  p50/p95 TTFT, TPOT, and end-to-end latency for each selected raw row.
- Formal benchmark protocol: random inputs 512/2048, concurrency 1/4, 4 warmups,
  32 measured requests, 3 rounds, 128 output tokens, request rate `inf`,
  `ignore-eos`, temperature 0, seed 42. GPU telemetry was sampled once per second.

The first complete run exposed a host-state issue: GPU power changed from roughly
130--160W to a 55W limit while Windows was using its only available Balanced power
scheme. That run is preserved at
`artifacts/stage1/benchmarks/20260907T113324Z` and is not used as the final result.
The failed groups were rerun only as needed; their raw evidence is preserved at
`20260907T115614Z`, `20260907T135907Z`, and `20260907T140950Z`. This is why the
final report records source provenance per configuration instead of silently
averaging incompatible runs.

## Reproduction commands

From Ubuntu-24.04 WSL2:

```sh
cd /mnt/d/Lunacia/Inference
bash scripts/download-qwen3-baseline.sh
bash scripts/start-vllm-baseline.sh
python scripts/check-vllm-api.py --base-url http://127.0.0.1:8000 --model qwen3-4b-baseline
bash scripts/run-stage1-benchmark.sh
bash scripts/stop-vllm-baseline.sh
```

The checked-in scripts keep the service on localhost, preserve raw results, and
support `--resume` and `--only INPUT:CONCURRENCY,...` for an explicitly recorded
stability rerun. The service must remain attached to a persistent WSL process on
this host; a short-lived `wsl bash -lc` caller can reap background processes.

No Docker image, cloud GPU, optimization implementation, or Stage 2 experiment was
created by this stage.
