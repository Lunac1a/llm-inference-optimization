# CPU KV API integration: delivered as experimental, acceptance failed

The optional CPU KV profile is integrated into the existing local API. Two
complete paired rounds establish substantial revisit gains with matched GPU
prefix retention. It is **not accepted as a recommended/default profile**:
round 2 exceeds the cold-latency gate, and round 3 stops on the host-swap gate.
No thresholds, cache sizes or runtime settings were changed after measurement.

## Frozen scope and controls

[Plan](kv-offload-integration-plan.md), API implementation, collector and CPU
preflight were committed as `2ffdde1` before GPU use. The previous incomplete
[recovery attempt](kv-offload-recovery-validation.md) remains separate; this new
protocol explicitly accepts partial GPU eviction and requires paired retained
prefix counts within one 16-token block. Original latency gates are unchanged.

Qwen3-4B BF16 weights/KV, vLLM 0.23.0, RTX 4090 Laptop 16GB, WSL Ubuntu-24.04,
FlashAttention, eager, chunking and prefix caching retain the baseline settings.
Every started server reported 36,528 GPU KV tokens. CPU treatment adds only the
8 GiB native offload pool and existing scoped single-copy adapter; no FP8 hybrid,
driver/package change, custom kernel or scheduler. WSL pinning remains disabled.
Two multi-block copy layouts passed both directions before serving.

Three independent 16K documents, 16,446-token performance prompts, fixed 128-token
outputs, serial cold A/B/C then revisit A/B/C. Completed order: offload/baseline,
baseline/offload; the third offload instance stopped before performance requests.
There are **24/36 performance requests, 10/12 short warmups, 6/6 fact answers and
2/2 nonstream API checks**. Third baseline was never started. No GPU retries.

Both complete pairs had exactly matching 3,360 GPU prefix hit tokens for every
revisit; offload additionally restored 13,072 external tokens and 1,927,544,832
bytes per request. Baseline computed 13,086 prefill tokens; offload computed 14.
Cold prefix hits were zero. All measured performance requests completed 128
tokens with zero preemptions. Pair controls pass, so the observed gains are not
explained by unequal GPU prefix retention.

## Results and costs

Each p95 below is an interpolated statistic of three requests, not a population SLA.

| Metric | Pair 1 GPU only | Pair 1 CPU | Pair 2 GPU only | Pair 2 CPU |
|---|---:|---:|---:|---:|
| Revisit p95 first-content latency (s) | 2.938 | 0.326 | 3.061 | 0.309 |
| Revisit p95 E2E (s) | 5.963 | 3.288 | 6.166 | 3.400 |
| Cold p95 first-content latency (s) | 3.420 | 3.668 | 3.217 | 4.200 |
| Cold p95 E2E (s) | 6.400 | 6.913 | 6.117 | 7.247 |
| Revisit output token/s, sum output / sum request time | 21.75 | 39.27 | 21.22 | 39.62 |
| Revisit output token/s, observed wave including internal gaps | 19.09 | 32.05 | 18.78 | 32.09 |

Revisit p95 TTFT falls **88.9% / 89.9%**, and E2E falls about **44.9% / 44.9%**.
Both pairs meet revisit gates. Cold p95 E2E rises **8.0% / 18.5%**; the second
pair fails the <=15% limit. It remains a valid negative sample, not an environment
error to discard or rerun. First-content speedup is not decode-only acceleration.

Each revisit transfers about 1.795 GiB from CPU. Saving and restoring KV adds
host memory, Python descriptor/launch overhead and GPU/CPU traffic. Per-request
queue, computed tokens, transfers in both directions, CUDA-event time, effective
bytes/event-second, SSE gaps and throughput are in
[analysis.json](../artifacts/kv-offload-integration/analysis.json). Raw metrics
and telemetry remain available. Event times do not measure PCIe saturation or
isolate the adapter's CPU overhead. SSE content-arrival gaps are not exact token
generation intervals, even though returned token IDs validate output counts.

Complete offload runs sampled roughly 10.61–10.83 GiB summed owned-process RSS,
versus 2.81 GiB baseline. Summed RSS may double-count shared pages. Configured
CPU pool size is not measured occupancy. The active GPU cache gauge similarly
does not describe all reusable inactive prefix blocks.

## Host-memory stop and API evidence

Before round 3's first performance request, WSL reported 698,957,824 bytes of
swap use (666.6 MiB), against zero at preflight and a maximum allowed growth of
256 MiB. MemAvailable was still about 4.89 GiB. The collector stopped on swap
growth, not OOM, timeout or an incorrect answer. Third-round performance is
unmeasured and must not be imputed from the first two rounds.

This is observed host memory pressure. The precise contribution of CPU buffers,
process startup and WSL/host memory behavior was not isolated. A 12 GiB startup
MemAvailable check alone did not ensure the later swap gate; do not represent
that startup check as a runtime resource guarantee. No swap clearing, WSL tuning
or CPU pool reduction was used to pursue acceptance.

The actual offload servers use the same `CpuKvServer` launch method as
`local_document_qa.py serve --cpu-kv-cache`, preserving endpoint and alias.
First and restarted third instances both passed nonstream normal-EOS API checks;
performance and fact checks use SSE. Offline CLI tests cover selection, rejection
of prefix-off mode, environment restoration and cleanup. The full interactive
CLI process was not separately launched on GPU; serving-class execution was.

All six normal-EOS fact answers were inspected against document bodies. Both
arms correctly answered A: 林若安 / 480万元; B: 周明川 / 315万元;
C: 顾清禾 / 260万元. Each response states both facts without contradiction or
foreign document values. This is a small source-grounded check, not general
coding-agent quality evaluation.

## Delivery and limitations

[Usage](cpu-kv-api.md) documents the explicit optional flag, memory requirements,
OpenAI-compatible requests, logs and shutdown. The existing default configuration,
historical hybrid prototype and raw evidence remain intact. CPU mode is exposed
for experiments, not recommended as the default after these failures.

The complete attempt consumed 336.70 seconds of the 30-minute budget including
cleanup. Port 8000 is closed; no GPU compute processes remain. Cleanup reported
825 MiB GPU memory and about 70.4 MiB residual WSL swap; no attempt was made to
clear unrelated system swap. See [cleanup](../artifacts/kv-offload-integration/cleanup.json)
and [preservation/tests](../artifacts/kv-offload-integration/closure.json).
All 1,990 pre-run artifact files, 56 frozen source/config files and nine installed
vLLM source files retain their hashes. All 22 offline tests pass. The additive CLI
change was committed before preflight freezing; the BF16 configuration file and
historical hybrid implementation were not edited.

Measured: substantial repeated-prefix gains in two valid pairs, cold-request cost,
real CPU recovery, correct small fact checks and working API profile. Inference:
avoided prefill accounts for much of the first-content improvement, supported by
computed-token counters. Unverified: third pair, reliable operation under broader
memory pressure/concurrency, hybrid FP8 combination, and active contexts beyond
GPU capacity. No general stable-speedup or full-acceptance claim is made.
