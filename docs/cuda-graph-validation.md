# Decode-only CUDA Graph: reproducible concurrency-4 gain

All three fixed paired rounds completed. At concurrency 4, every pair meets the
preregistered speed, E2E and TTFT sub-gates. At concurrency 1, rate gains remain
below 10%; therefore **the combined acceptance rule fails and the default remains
unchanged**. This is useful bounded concurrency evidence, not a failed runtime or
a reason to discard the single-request measurements.

## Scope and execution evidence

[Plan](cuda-graph-plan.md), launch adapter, collector, tests and CPU preflight were
committed as `89677aa` before GPU. Independent evidence is in `artifacts/cuda-graph`.
Historical documentation identifies eager as a WSL compatibility baseline and
the graph path as a hypothesis, not a previously completed graph experiment.

Pinned Qwen3-4B BF16, vLLM 0.23.0, RTX4090 Laptop16GB, WSL Ubuntu-24.04, BF16 KV,
FlashAttention, V1 runner, FlashInfer sampler off, prefix caching on, context32768,
token budget2048, sequence cap8 and GPU utilization .8 are held fixed. No CPU
offload, hybrid FP8, external profiler, new kernel, package or driver changes.

Graph removes `--enforce-eager` and selects compilation mode 0, graph mode
`FULL_DECODE_ONLY`, capture sizes [1,2,4], maximum4. Eager explicitly selects
compilation mode0 and graph NONE. Full graphs without compilation are supported
by the saved installed `config/compilation.py` source. Both arms use native graph
metrics. This isolates upstream graph execution, rather than torch compilation
or a custom implementation. Native startup memory estimation is retained; no
additional profiling experiment was run.

All graph runs report successful full-decode capture and native runtime FULL
stats for token batches1/4. Eager reports NONE. Prefill/mixed work remains outside
graphs. Logs and `runtime-evidence.json` establish actual runtime selection, not
just accepted CLI flags. The saved wrapper source calls upstream graph replay;
this is not an independent profiler measurement of launch counts or saved cycles.

Performance body2048 tokens, chat prompt2110 tokens, identical repeated question,
output512, ignore_eos, temperature0, seed42, thinking off. Each fresh server has
two32-token warmups, three serial requests, then a simultaneous four-request wave.
Orders graph/eager, eager/graph, graph/eager. **42/42 performance requests, 12/12
warmups and 6/6 normal-EOS fact answers complete**. All cache/preemption controls
pass; no request failure, GPU retry, extra load point or configuration scan.

## Paired results

Rates are client-observed `(output_tokens-1)/(last_content-first_content)` and
exclude first-content wait. They are not exact device token-generation intervals.
Each p95 covers three serial requests or four concurrent requests, not an SLA.

| Pair | C1 rate gain | C1 p95 E2E reduction | C4 rate gain | C4 p95 E2E reduction |
|---|---:|---:|---:|---:|
| 1 | 8.35% | 1.88% | 15.84% | 12.50% |
| 2 | 6.00% | 7.53% | 14.13% | 11.99% |
| 3 | 4.75% | 5.38% | 17.48% | 14.70% |

At C1, eager median observed rate is 49.26–50.58 token/s versus graph
52.81–53.61. At C4, per-request median is 41.92–43.32 versus 49.24–49.45 token/s.
C4 aggregate wave throughput rises from 165.04–171.25 to 192.69–194.57 token/s
(paired +13.6–17.3%). C4 p95 E2E falls from 11.95–12.40s to 10.52–10.62s.

All TTFT sub-gates pass. C1 p95 TTFT is around .05s eager / .04s graph; C4 is
.10–.12s / .08–.09s. This workload has warm prefixes and does not establish a
long cold-prefill improvement. The graphs optimize decode execution, and cannot
be represented as shrinking the model's mathematical compute or weights.

The original rule requires >=10% median post-first rate gain and >=5% p95 E2E
reduction in both load levels in every pair. All C1 rate sub-gates fail; pair1 C1
E2E also fails. All C4 sub-gates pass. These outcomes remain separate; no threshold
was lowered and no single-request point was removed to claim full acceptance.

## Costs and interpretation

GPU KV capacity is 36,528 tokens eager and 36,208 graph in every run, a reduction
of320 tokens (0.88%). All capacity gates pass. Native logs estimate .04GiB graph
memory and report .03GiB captured; these are rounded runtime reports, not isolated
high-precision allocation measurements. Startup takes20.25–22.29s graph versus
20.16–22.04s eager; capture overhead is included. No graph-size scan was made.

Per-request TTFT/E2E/rates/content-gap statistics and telemetry summaries are in
[analysis.json](../artifacts/cuda-graph/analysis.json). Wave snapshots preserve
queue, preemption and cache metrics; telemetry records GPU samples, owned RSS,
host memory and cache gauges. All performance cache reuse is >=90%; zero
preemptions. CPU RSS is a sum across processes and may double-count shared pages.
SSE timestamps include transport/client effects; returned token IDs verify counts,
not precise per-token generation timestamps.

Measured fact: native graph execution is compatible here and gives repeated C4
improvement with small capacity cost; C1 improvement is smaller. Plausible
mechanism: less host launch/dispatch overhead, as expected from replay. Its exact
share was not profiled; model memory bandwidth, kernel time and laptop power/
thermal variation remain alternative contributors to effect size. These results
do not establish general workload performance, cold long-context behavior,
concurrency beyond4, or combination with CPU KV/FP8 hybrid profiles.

## Correctness, cleanup and use boundary

Six separate normal-EOS body-checkable questions ask owner/budget. Both arms
correctly state A: 林若安 / 480万元; B: 周明川 / 315万元;
C: 顾清禾 / 260万元. Human inspection found no omission, contradiction or
cross-document values. Forced-length performance prose is not scored as quality.
This small correctness check does not prove broad output equivalence.

The run finishes in402.08 seconds of its20-minute budget including cleanup. Port
8000 is closed, no GPU compute process remains, swap is zero and GPU memory is
834MiB at cleanup. Some raw logs contain EngineDeadError during owned SIGTERM
shutdown; they occur after completed requests and the shutdown signal, and the
server parents exit0. These are retained shutdown diagnostics, not discarded
measurement failures. See [cleanup](../artifacts/cuda-graph/cleanup.json) and
[preservation/tests](../artifacts/cuda-graph/closure.json).
All 2,095 historical evidence files, 80 frozen source/config files and six audited
installed vLLM sources retain their hashes. All 18 offline tests pass.

No new API flag or default change is made by this experiment. Existing BF16 and
experimental CPU profiles retain their behavior. `scripts/cuda_graph_entry.py` is
an experimental launch adapter used by the fixed collector, not a recommended
replacement launch command. Reproduction needs newly authorized fresh evidence
paths and a fixed budget; never overwrite this run. The useful delivery candidate
is a decode graph option for the tested small concurrent workload, subject to a
separate API-integration decision rather than a claim of universal acceleration.
