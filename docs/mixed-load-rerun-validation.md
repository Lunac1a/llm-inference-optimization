# Interleaved mixed-load validation

## Outcome: interference confirmed; tested chunking is not adopted

**Cold C noticeably stalls both active A and cached B in all three full-prefill
pairs.** The sole chunking alternative improves the longest stall and B's first
content but fails the preregistered A/B E2E guardrail. Keep the BF16 default and
existing full-prefill hybrid prototype unchanged; no new profile is shipped.

The following are medians of three trial measurements, in seconds except throughput:

| Metric | Full, A/B only | Full, insert C | Chunked, A/B only | Chunked, insert C |
| --- | ---: | ---: | ---: | ---: |
| A maximum content gap after insertion | 0.084 | 3.031 | 0.084 | 1.722 |
| B first-content latency | 0.119 | 3.448 | 0.111 | 0.395 |
| A E2E | 3.516 | 6.995 | 3.535 | 12.138 |
| B E2E | 3.497 | 7.129 | 3.575 | 12.242 |
| C first-content latency | absent | 3.448 | absent | 8.530 |
| C E2E | absent | 7.108 | absent | 12.451 |
| Trial output tok/s | 65.57 | 51.01 | 64.33 | 29.80 |

Every full-prefill pair exceeds both fixed interference tests: A gap >=2x control
and >=.250 s increase, B TTFT >=2x and >=.500 s increase. Relative to mixed full
prefill, chunking reduces median A maximum gap **43.18%** and B TTFT **88.54%**.
However A/B median E2E increases **73.54% / 71.72%**, far above the permitted 15%.
C TTFT increases **147.41%**, C E2E **75.18%**, and throughput falls **41.58%**.
The fixed mitigation decision is therefore **fail**, despite its latency subgates
passing. Residual A interference still exceeds the gate in all three chunked pairs.

The cost is visible in the stream: full prefill produces 1/2 gaps >=250 ms per
mixed trial; chunking produces **8 in every trial**. Median summed time inside
those gaps is 3.355 versus 8.338 s; median per-trial gap p95 is .033 versus .679 s.
These are descriptive derived measurements, not extra selection criteria or a
pooled p95 estimate. Shortening the single worst pause does not ensure smoother
overall generation or faster completion.

See the [12-trial table](../artifacts/mixed-load-rerun/measurement-table.md),
[raw analysis](../artifacts/mixed-load-rerun/analysis.json), and
[gate/progress assessment](../artifacts/mixed-load-rerun/assessment.json).

## Validity, cache and actual routes

All **30/30** performance requests terminate correctly with exactly 128 output
tokens. At every dispatch the client has received exactly 16 A tokens, followed
by another 112; C submission occurs 0.492–0.876 ms after the trigger. Engine route
logs independently show a one-token A decode, B's 30-token cached suffix and C's
cold prefill in the same batch. This supports actual ongoing generation, beyond
merely seeing a socket that has not yet closed. Actual HTTP arrival/admission
timestamps per request are not exposed; dispatch timestamps are client-side.

A/B each reuse **16,416 / 16,446 prompt tokens (99.82%)** in every measured
request; C has zero cached tokens. Engine hit/query is 32,832/32,892 in controls
and 32,832/49,338 with C. The latter's 66.55% aggregate is expected from adding
an entirely cold third document, not loss of A/B reuse. Both configurations report
**55,680 cache tokens**; mixed active KV peaks are about 89.1–89.3%.
Sampled running peaks are two without C and three with C; waiting peaks are zero.
All trials have **zero preemptions**. Aggregate queue sums are only 14–28
microseconds, while observed stalls take seconds. This supports an in-batch
execution-stall interpretation rather than long admission queues; it does not
identify separate kernel, CPU, quantization or transport contributions.

Full C is a complete 16,446-token direct-BF16 prefill. Chunked C first runs 2048
tokens through original BF16, then seven 2048-token chunks plus a 62-token suffix
through cached FP8 attention (14,398 prompt tokens after the initial chunk).
Actual logs show A/B one-token decode sharing those later C prefill batches.
Matching cache capacity and stable no-C controls strengthen the comparison, but
the scheduling/attention-path changes cannot be disentangled by these two modes.

Normal-EOS quality is separate: **6/6** answers (three documents per mode) correctly
give the owner AND official budget, with no missing requested fact, contradiction,
cross-document value or training-budget confusion. All finish with `stop`.
Equivalent phrasing is accepted; fixed-length performance text is not quality
scored. See [source-aligned answer inspection](../artifacts/mixed-load-rerun/answer-review.md).

## Protocol and evidence

This is the newly authorized execution of the [unchanged workload and gates](mixed-load-plan.md),
with [fresh budget and evidence](mixed-load-rerun-plan.md), preregistered in `8e7d728`.
The earlier reset-API failure remains under `artifacts/mixed-load/`; this run uses
`artifacts/mixed-load-rerun/`. No failed performance sample was replaced.

Pinned Qwen3-4B BF16 revision `1cfa9a7208912126459214e8b04321603b3df60c`,
vLLM 0.23.0, torch 2.11.0+cu130, RTX 4090 Laptop 16GB, Ubuntu-24.04 WSL.
Both modes use the unchanged experimental hybrid backend, Triton FP8 persistent
KV, prefix caching, eager execution, memory utilization .8, 8 maximum sequences,
and model/batched token limits 16640. Continuous and async batching already exist;
this experiment does not implement or newly enable batching.

Each trial resets cache, warms independent A/B document prefixes, starts A, then
submits B and optionally cold C when A reports exactly 16 output token IDs.
Each document has 16384 tokens; each performance prompt has 16446 tokens and
requests exactly 128 output tokens, ignore EOS, temperature 0, seed 42, thinking off.
The paired order is control/mixed, mixed/control, control/mixed. Inputs, timing
trigger and output length are identical between modes. One full-prefill server
runs first; only its interference gate authorizes the one chunked server.

The sole alternative enables chunking and caps each long prefill at 2048 tokens.
The total batched token budget remains 16640. This avoids deliberately changing
activation profiling capacity, but does not isolate computation precision/kernel
effects from scheduling. FCFS and all other workload settings stay fixed.

## Measurement meaning and limits

TTFT here means client time to first nonempty SSE content. Every raw SSE event
has a monotonic receipt timestamp; token IDs establish output progress and final
usage establishes length. Content events can include zero/multiple tokens, so A
gaps are **SSE content-block intervals, not exact per-token generation latency**.
The A maximum includes a gap straddling B/C dispatch. Per-request output tok/s
uses output count divided by E2E; trial throughput uses all outputs divided by
the A-start-to-last-finish window. Both are retained, not interchangeable.

Queue histograms are aggregate across the trial, not individual A/B/C queue times.
Running/waiting/KV gauges are sampled at one-second intervals and may miss short
events. KV usage is active block occupancy, not inactive-prefix residency or total
reserved VRAM. Prefix hit/query is an engine token-counter ratio, not a request-hit
percentage; request-specific cached token details provide separate reuse evidence.

The unchanged hybrid backend routes a zero-context first chunk to original BF16
attention when query length equals available sequence length; later chunks read
the already quantized FP8 cache. Thus full cold prefill and chunked cold prefill
are different compute paths. A reduction in the longest observed stall cannot be
attributed solely to scheduler mechanics, nor described as a complete BF16 first
pass under chunking. No profiling or kernel-level causal isolation was performed.

Three small repetitions describe this specific synthetic arrival pattern, not a
production tail-latency distribution or SLA. Different chunk sizes, delayed C
admission, output lengths, models and real-document workloads are not tested;
the frozen plan authorizes no search over these alternatives.

Full mode necessarily precedes the conditional chunk mode. Thermal/order effects
are therefore not independently controlled: sampled GPU temperatures are 69–86 C
versus 82–87 C and SM clocks 2085–2355 versus 1590–2340 MHz. Telemetry is retained;
no thermal reruns or corrections were added. The approximately unchanged no-C
medians do not prove absence of thermal effects on long prefill. Report the whole
observed intervention, not a universal 2048-token chunking cost.

## Closure

Two owned servers completed and exited 0; elapsed GPU budget including cleanup
was **319.39 seconds (5.32 minutes)** of 30. Port 8000 is closed, no GPU compute
applications remain, and cleanup reports 971 MiB GPU memory. No errors, retries,
profiling or additional configurations occurred in this execution. The prior 404
attempt remains separate and unchanged.

Old evidence/source verification and offline regression checks are recorded in
`artifacts/mixed-load-rerun/closure.json`: **1,779 old artifacts, 46 frozen source/
config files and 6 installed source files are unchanged; 10/10 offline tests pass**.
Raw evidence has a SHA-256 manifest.
