# Mixed-load diagnostic: stopped before measurement

**No valid interference result and no scheduling comparison were obtained.**
The first baseline trial stopped at prefix-cache reset, before warming the long
documents or submitting A/B/C. This is a collector/API integration failure, not
a slow request, quality failure or evidence that interference is absent.

The frozen [plan](mixed-load-plan.md) and collector were committed as `741823f`
before starting the GPU. The plan explicitly stops on reset failure and prohibits
GPU retries. No thresholds, workload or repetition count were changed; the
conditional chunked arm was not launched. No additional experiment is justified
by this failed attempt's results.

## What happened and why

The fixed environment started successfully: vLLM 0.23.0, Qwen3-4B BF16, RTX 4090
Laptop, hybrid Triton, full prefill, 16640 model/batched token limits, 8 sequences,
55,680 KV cache tokens. Two short 16-output-token compilation warmups completed.
The first `POST /reset_prefix_cache` then returned **404 Not Found**.

The installed `entrypoints/openai/api_server.py` registers development routes only
when `VLLM_SERVER_DEV_MODE` is true. `envs.py` defaults this to false; the reset
handler is in `entrypoints/serve/dev/cache/api_router.py`. The experiment launcher
omitted that flag. Reusing the historical reset helper without checking its route
registration was the collector integration mistake. The historical full-wave
hybrid collector did not require this endpoint. Saved installed source and raw
HTTP response establish the cause; this is not evidence of a GPU/model defect.

After stopping, the experiment-only entry adapter was repaired offline to set
`VLLM_SERVER_DEV_MODE=1` inside its owned subprocess and record it. No default
service, hybrid backend, installed package or historical launcher was changed.
The repair has offline launch-argument coverage only; **it has not been validated
with a live GPU server**. The failed run's launch evidence and source commit remain
unchanged. A future run requires a fresh evidence directory and newly committed
budget/plan; this run must never be overwritten or restarted.

## Measured, inferred and unverified

Measured facts: CPU token preflight found 16446 performance input tokens for each
document, 16430 own-document shared tokens, only 3 common tokens across documents;
all fit 128 output tokens within 16640. One server started; two short warmups passed;
reset returned 404; server exited 0. No valid A/B/C trial, no fact-answer run and
no chunked comparison exist. The derived measurement table is intentionally empty.

Source-established behavior: V1 already performs continuous batching, schedules
running requests before waiting ones and bounds admitted prefill by remaining
token budget. It reads `long_prefill_token_threshold` in running and waiting
paths. Merely exposing partial-prefill settings does not prove that this V1
scheduler uses those settings. Async scheduling was enabled in the actual startup.
Full-prefill batches can include running decode when the long input fits the
remaining budget. Whether that noticeably stalls this workload remains unmeasured.

Inference: splitting a long prefill may shorten individual decode stalls while
increasing long-request work/latency. The unchanged hybrid routing would use BF16
for an initial zero-context chunk and FP8 cached attention for later chunks;
any future comparison must attribute results to that combined intervention,
not pure scheduling or a complete original-BF16 first pass.

Still unverified: A output pauses; B/C first-content latency; A/B/C E2E and
throughput; interference queueing, preemption and cache occupancy; mixed-load
prefix hits; mitigation effectiveness and cost; cross-document correctness.
No numerical values for these can be inferred from the short warmups.

The collector timestamps SSE events and asks for output token IDs, enabling a
fixed output-progress trigger. Offline tests cover coalesced chunks: 128 output
tokens need not produce 128 content events. Gaps remain client-observed **SSE
content-block intervals**, not precise token-generation latency. Request-level
queue latency is not exposed by this collector; retained queue histograms would
be aggregate only. Neither capability was validated on the intended mixed load.

## Cleanup and preservation

GPU attempt including cleanup: **29.26 seconds** of the 30-minute ceiling.
Port 8000 is closed, no GPU compute applications remained, and recorded GPU
memory was 916 MiB. The owned server exited 0. See
[cleanup](../artifacts/mixed-load/cleanup.json),
[failure](../artifacts/mixed-load/failure.json), and
[raw reset response](../artifacts/mixed-load/full/r1-control/reset.json).

Preservation and offline test evidence are recorded in
`artifacts/mixed-load/closure.json`: **1,755 old artifacts, 46 frozen source/config
files and 6 installed source files match**, and **10/10 offline tests pass**
(4 mixed-load collector/launcher checks and 6 historical hybrid checks).
The BF16 default and historical
hybrid performance/quality conclusions remain exactly as previously validated.
