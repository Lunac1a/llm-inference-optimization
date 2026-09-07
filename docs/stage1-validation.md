# Stage 1: local vLLM baseline

Status on 2026-09-08: **Stage 1 passed under acceptance protocol v2**.
Stage 2 has not started.

## Scope and reproducibility

This stage establishes local serving and measurement; it does not implement an
inference optimization, Docker deployment, or cloud deployment.

- Ubuntu 24.04 WSL2; RTX 4090 Laptop, 16,376 MiB; driver 596.49.
- Python 3.12.3; vLLM 0.23.0; PyTorch 2.11.0+cu130; Transformers 4.57.6.
- Qwen/Qwen3-4B BF16, snapshot `1cfa9a7208912126459214e8b04321603b3df60c`.
- Localhost:8000, context 4096, GPU memory fraction 0.80, four sequences,
  2048 batched tokens, prefix caching off, chunked prefill on.
- Eager execution, V2 runner off, FlashInfer sampler off. These are fixed WSL
  compatibility settings, not a stock optimized-vLLM performance claim.
- SHA256-verified Python header packages are provisioned without sudo by
  `setup-wsl-headers.sh`, also invoked by the environment setup. The start script
  checks for them. Shell scripts and shell-sourced .env files use LF in Git.
- Dependency versions, model hashes, runtime settings, and collection-time source
  hashes are retained. No clean-machine reinstall or cloud reproduction is claimed.

See [environment.md](environment.md), [the frozen protocol](stage1-protocol-v2.md),
and [collection hashes](../artifacts/stage1/acceptance-v2/source-sha256.txt).

## Correctness and restart evidence

The corrected checker disables thinking using top-level
`chat_template_kwargs.enable_thinking=false`. Responses must be nonempty,
without thinking output, and finish with `stop`; streams must end cleanly with
DONE. Invalid requests are followed by a successful generation check.

- Before restart: **18/18**, `api-before-restart/api-check-20260907T143943Z.json`.
- After restart: **18/18**, `api-after-restart/api-check-20260907T144107Z.json`.
- Separate startup logs and PIDs **399 -> 722** are in
  [acceptance-v2](../artifacts/stage1/acceptance-v2/).
- Five regression tests passed in WSL: truncated/thinking answers, incomplete
  streams, missing/error request arrays, missing rounds, and retaining all 12 rows
  while rejecting excessive reruns. See `regression.log` there. An earlier Windows
  invocation hit temporary-directory permission errors; the WSL run is the
  successful automated acceptance evidence.

## Performance protocol and evidence

Fixed random input lengths 512/2048, output 128, concurrency 1/4, seed 42,
temperature 0, ignore EOS, no shared prefix. Each group has four warmups and
32 measured requests, repeated three times. Three preparation batches preceded
formal measurement. Client and server both ran in WSL.

A preparation command failed before sending requests because vLLM 0.23.0 uses
range ratio **0** for fixed lengths. Installed source was checked, scripts
corrected, and hashes refreshed before formal measurement. The failure log is
retained; preparation results are not part of acceptance.

First run: `20260907T144656Z`, **384/384 requests passed**.

| Input tokens | Concurrency | Mean output tokens/s | Sample CV |
| --- | --- | --- | --- |
| 512 | 1 | 45.60 | 14.07% |
| 512 | 4 | 168.39 | 11.97% |
| 2048 | 1 | 39.86 | 3.69% |
| 2048 | 4 | 117.37 | 4.33% |

Two groups failed the <=5% stability gate. Telemetry was inspected before the
single permitted full rerun `20260907T150215Z`. Active telemetry samples ranged
from 74 to 90 C, with varying power readings; this does not establish a single
causal mechanism. The entire rerun is selected; groups are not mixed across runs.

The full rerun passed: **384/384 requests**, zero errors, exact token lengths,
all 12 rows preserved, and all four CVs <=5%. The single-rerun policy passed.

| Input tokens | Concurrency | Mean output tokens/s | Sample CV |
| --- | --- | --- | --- |
| 512 | 1 | 41.87 | 3.15% |
| 512 | 4 | 151.21 | 4.19% |
| 2048 | 1 | 39.09 | 4.29% |
| 2048 | 4 | 115.48 | 2.70% |

See [benchmark summary](../artifacts/stage1/acceptance-v2/benchmark-summary.json)
and [overall acceptance](../artifacts/stage1/acceptance-v2/acceptance.json).
The benchmark-only report intentionally leaves `stage1_passed=false` because it
cannot certify API or restart checks. The overall report combines those gates.
Collection-time source/config hashes verified unchanged. Final server logs show
no OOM; service PID 722 was stopped, port 8000 was closed, and no GPU compute
processes or model workers remained. See `cleanup.log` and `server-final.log`.

## Interpretation and historical evidence

Per-request details, p50/p95 TTFT, TPOT, end-to-end latency, throughput, and
one-second GPU telemetry are retained. Throughput uses raw completions and fixed
work; it does not measure answer quality. API smoke uses non-thinking chat.
Three rounds and 32 requests per group are small samples; no production SLO,
custom speedup, or cross-hardware comparison is established.

The old v1 pass remains withdrawn. Its API outputs were truncated reasoning,
its parser ignored errors, and its selected groups exceeded the rerun allowance.
All original evidence remains preserved. See [review history](stage1-review-history.md).
The root `final-benchmark-summary.json` is a historical rejected report; current
v2 evidence lives under `artifacts/stage1/acceptance-v2/`.

## Reproduction (WSL)

```sh
bash scripts/setup-vllm-env.sh
bash scripts/download-qwen3-baseline.sh
bash scripts/start-vllm-baseline.sh
# Wait for http://127.0.0.1:8000/health to succeed.
python scripts/check-vllm-api.py --output-dir artifacts/stage1/new-api-check
python -m unittest discover -s tests -v
# Follow the frozen protocol with fresh evidence paths; preserve existing runs.
bash scripts/run-stage1-benchmark.sh
bash scripts/stop-vllm-baseline.sh
```

Keep a WSL shell open while serving. Do not invoke `collect-stage1-v2.sh` again
against the existing acceptance folder: it overwrites preparation/hash logs.
