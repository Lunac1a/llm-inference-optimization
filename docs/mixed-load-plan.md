# Mixed arrivals: frozen bounded diagnostic

Commit this plan and collector before GPU use. Evidence: artifacts/mixed-load/;
refuse overwrite. Preserve every older artifact, BF16 default and hybrid backend.
Pinned existing Qwen3-4B BF16 revision, WSL Ubuntu-24.04, RTX 4090 Laptop,
vLLM 0.23.0; no profiling, cloud, multi-GPU or scheduler implementation.

## Source audit and one intervention

Installed v1/core/sched/scheduler.py schedules running requests before waiting
requests on every iteration: continuous batching already exists. Without chunking,
a waiting prefill exceeding remaining token budget waits; a fitting long prefill
can share an iteration with decode. FCFS remains unchanged. Existing configuration
exposes chunking, max_num_batched_tokens, long_prefill_token_threshold and partial
prefill limits; do not assume every exposed limit is used by this V1 scheduler.
Save installed scheduler/config/API source and hashes in preflight.
Reference: https://raw.githubusercontent.com/vllm-project/vllm/v0.23.0/vllm/v1/core/sched/scheduler.py

Baseline uses existing hybrid config: model limit/batched budget 16640, 8 sequences,
memory utilization .8, prefix on, eager Triton FP8 persistent cache, chunking off.
Only conditional alternative: chunking on plus long_prefill_token_threshold=2048.
Keep batched budget 16640 to avoid intentionally changing activation profiling.
Record actual capacity regardless. No other tuning. The hybrid route uses original
BF16 only where query length equals sequence length and >1: first cold chunk may
use BF16, later chunks necessarily use cached FP8. This is a scheduling AND
computation-path intervention, not an all-BF16 cold pass or pure scheduling effect.

## Frozen workload and decision

Use unchanged three 16384-token source documents and existing hybrid prompt
layout. Performance question: '概括本文档的项目情况。' for A/B/C; warm A/B with
'文档编号是什么？'. Save exact tokenized lengths before GPU; each prompt plus
128 tokens must fit 16640; cross-document prefix <16, own-document prefix >99%.
All performance requests: exactly 128 output tokens, ignore_eos, temperature 0,
seed 42, thinking off. Warmups A/B output 16 fixed tokens, preceded by two short
16-token compilation warmups per server. Reset prefix cache before each trial;
reset failure stops. Warm A then B, start A, at first SSE reporting >=16 output
token IDs submit B then C immediately (no deliberate gap). Control submits B only
at identical progress. Record actual dispatch times/order and trigger count.
Require A unfinished at C dispatch AND subsequent A output token IDs after dispatch;
require exact 16-token trigger (otherwise invalidate and stop, no retry).

Three paired repetitions: control/mixed, mixed/control, control/mixed; six trials
on one baseline server, cache reset and A/B rewarm every trial. Fixed gate:
meaningful interference when at least 2 of 3 pairs show either A post-trigger
maximum SSE content gap >=2x control AND >=0.250 s absolute increase, or B first
content latency >=2x control AND >=0.500 s increase. Include the gap straddling
dispatch. If gate false, finish baseline quality/cleanup and STOP, no alternative.
If true, repeat exactly those six trials on one alternative server. No extra points.
Effective mitigation requires >=30% reduction in the median affected metric(s)
versus mixed baseline, no >15% regression in A/B median E2E, all valid streams,
zero preemptions and correct fact answers. Report C TTFT/E2E and throughput cost
even if mitigation fails. No threshold changes or extra reruns.

## Evidence, correctness and stops

Use monotonic per-SSE timestamps, token IDs for progress and usage for exact final
length. Content gaps are client-observed SSE content-block gaps, NOT exact per-token
latency (detokenization, batching and transport can coalesce events). Save raw events,
TTFT/E2E/output throughput per request and trial, A post-trigger gaps, dispatch proof,
request cache tokens (API prompt details), metrics before/after and 1-second telemetry
for queue histograms, preemptions, active KV occupancy, running/waiting and prefix
hit/query. Queue histograms are aggregate, not per-request queue estimates; explicitly
mark individual queue time unavailable. Record real route logs and cache capacity.

Separate quality: after trials, reset and ask each document the normal-EOS question
'项目负责人是谁，正式预算金额是多少？', max 128, once per config, sequentially.
Inspect source-aligned answers for omissions/contradictions/cross-document facts;
accept equivalent wording. No quality scoring of forced-length performance output.

One 30-minute wall budget from first server start, including compilation, diagnostics,
conditional alternative and cleanup; reserve 120 seconds; startup <=180 s,
each request <=60 s absolute, each trial <=180 s. At most two server starts,
30 measured performance requests, 24 document warmups, 4 short warmups, 6 quality
requests. Stop on occupied port, environment/startup error, OOM, timeout, malformed
stream, wrong length, missing route/metrics/cache/progress proof, preemption or
budget exhaustion; retain failures separately, no GPU retry. Latency gate failure
does not authorize extra trials. CPU fixes before GPU are allowed. Finally verify
all old evidence/source hashes, clean owned services, document limits, commit/push
current branch. No service or permanent plugin is left running.
