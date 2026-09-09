# Prefix cache + FP8 KV capacity acceptance

## Decision

**Do not adopt an FP8 capacity-first profile under the frozen usability gates.**
FP8 demonstrably sustains three independent cached document requests with fast
follow-ups, whereas BF16 only sustains two at that follow-up latency. However,
FP8 C=3 cold p95 TTFT is **24.54 / 24.86 seconds**, exceeding the preregistered
20-second bound in both repetitions. Therefore usable capacity including cold
requests did not increase from two to three under the complete acceptance rule.
Keep the validated FlashAttention/BF16 default; no FP8 CLI option is shipped.

Plan commit: `6280fc0`; pre-measurement collector/preflight commit: `7c8e8a8`.
See [frozen plan](fp8-capacity-plan.md). No post-result threshold change, extra
point, rerun, profiling or cloud work. The eight planned fresh servers completed.

## Verified controls and workload

RTX 4090 Laptop 16,376 MiB, SM 8.9; Ubuntu-24.04 in WSL; vLLM 0.23.0,
torch 2.11.0+cu130, pinned Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c`, BF16 model weights.
Installed backend source is retained in preflight.json. Both actual startup
logs confirm TRITON_ATTN, prefix caching and the intended KV dtype. All other
settings and API model alias match, including calculate-kv-scales. FP8 here means
upstream `fp8_per_token_head`, not weight quantization or custom kernel work.
The [upstream backend matrix](https://docs.vllm.ai/en/stable/design/attention_backends/)
is supporting documentation; the installed source and actual starts establish
support on this fixed environment. No comparison against FlashAttention is made.

Three different synthetic 16,384-token documents have only 35 common initial
prompt tokens across documents. Each document's own questions share 16,422
initial tokens. Full prompts contain 16,435–16,444 tokens. Each concurrent wave
contains exactly one request per document. Each point has one cold wave and two
follow-up waves, with a fresh server and two short warmups. Two paired repetitions
use BF16/FP8 then FP8/BF16 order; C=2 then C=3 in each arm.

## Measured performance

All **60/60** performance requests completed with exactly 128 output tokens
(temperature 0, thinking off, ignore EOS). All measured waves had zero preemptions.
Values below are ranges across the two repetitions and, for follow-ups, both
waves; they are not pooled percentiles or production SLAs.

| Metric | BF16 C=2 | FP8 C=2 | BF16 C=3 | FP8 C=3 |
| --- | --- | --- | --- | --- |
| Cold p95 TTFT, s | 9.41–9.64 | 14.74–15.77 | 16.37–17.52 | 24.54–24.86 |
| Cold p95 E2E, s | 13.54–13.80 | 18.75–20.25 | 19.94–21.23 | 29.75–29.92 |
| Follow-up p95 TTFT, s | 0.105–0.149 | 0.143–0.212 | 8.18–16.88 | 0.218–0.249 |
| Follow-up p95 E2E, s | 3.49–4.12 | 3.54–4.28 | 11.74–20.85 | 4.21–4.75 |
| Follow-up output tok/s | 59.36–71.66 | 57.22–69.55 | 17.88–30.18 | 77.59–87.98 |
| Complete point gate, both rounds | Pass | Pass | Fail | Fail |

FP8 C=2 paired follow-up E2E changes were -5.36%, +1.67%, +5.43%, -2.14%,
all within the 15% regression bound. FP8 C=3 passes the follow-up gates, but
fails the cold TTFT gate. BF16 C=3 fails follow-up TTFT, E2E and reuse gates.
The two-point experiment does not determine either configuration's global maximum.

See [all 24 wave measurements](../artifacts/fp8-capacity/measurement-table.md)
and measurements.json for throughput, p50/p95, queue totals and cache statistics.
Raw summaries retain labelled queue histograms before/after, and one-second
telemetry records running/waiting requests and KV usage. Throughput uses batch
wall time including HTTP worker overhead; TTFT/E2E start inside each HTTP worker.
Startup, warmups and measurement bookkeeping are excluded from wave throughput.

## Capacity and queue evidence

- At C=3, BF16 sampled running max is 2; FP8 is 3. All requests succeed in both
  arms, so HTTP success alone does not establish useful simultaneous capacity.
- BF16 C=3 follow-up total queue time per three-request wave is 3.96–15.82 s,
  with 1–2 waiting requests observed. FP8 follow-up queue totals are only
  0.000023–0.000041 s, with no waiting request observed in the samples.
- Cold C=3 queue totals are 16.17–16.95 s BF16 and 20.55–21.05 s FP8. FP8 does
  not remove initial prefill/admission delay despite the larger KV cache.
- Active KV usage peaks near 90.5–90.6% for BF16 C=3 and 70.0–70.1% for FP8 C=3.
  This gauge is active block occupancy, not total reserved GPU memory or the
  quantity of reusable inactive prefixes; it returns to zero between waves.
- FP8 C=3 follow-up hit/query is 99.85–99.88%; BF16 is 20.88–22.29%.
  These are raw engine token-counter ratios, **not unique-input or request hit
  percentages**. BF16 C=3 query deltas exceed submitted input tokens substantially;
  scheduler accounting can query prefixes repeatedly. Retain raw counters rather
  than claiming that exactly 21% of user input was reused.
- Engine cache sizes are 36,528 versus 70,848 tokens. These numbers are context,
  not the proof of usable concurrency. The observed requests, queueing, reuse and
  latency provide the additional evidence—and also expose the cold latency cost.

**Inference:** the consistent residency/reuse/queue pattern supports a cache
capacity advantage for three warm document clients. Precision is the only arm
setting changed. The cause of slower FP8 cold prefill is not isolated by this
experiment; no kernel profiling or universal performance attribution is claimed.

## Correctness, limits and closure

All **18/18** normal-EOS fact answers (nine per arm) were inspected against the
source: identifier; owner plus official budget; region plus risk. No truncation,
wrong-document facts, omissions or contradictions were found. Equivalent framing
was accepted, including the shorter FP8 identifier response. The explicit training
budgets were not confused with official budgets. See
[answer review](../artifacts/fp8-capacity/answer-review.md) and retained raw answers.
Performance outputs were not quality-scored. No historical Stage 3 score changed.

This is a small synthetic local workload, not general model correctness, an
arrival-rate capacity test or a production SLA. Two repeated small-wave p95 values
do not estimate tail risk reliably. Behavior with real heterogeneous documents,
different output lengths or backends remains unresolved. A warmed-only application
could value the measured benefit, but changing the acceptance scope after seeing
results would not validate this delivery. No additional experiments were run.

Collection and owned-service cleanup took **7.879 minutes of 30**. Eight owned
servers exited; port 8000 closed, no GPU compute applications remained and GPU
memory was 965 MiB at the recorded cleanup. No selected-profile CLI smoke was
run because the selection gate failed; the existing default launcher is unchanged.
Five targeted offline tests cover precision isolation/default preservation,
labelled counters, stable prefixes and rejection of successful-but-slow/missing-reuse
capacity claims. These checks are separate from real-model acceptance.

All **1,284** pre-existing artifact files and seven frozen source/config hashes
remain unchanged. New raw evidence, analysis, queue telemetry, review and hash
manifest are isolated under artifacts/fp8-capacity/. The scripts refuse to rerun
or overwrite collection evidence. Future reproduction requires a fresh evidence
location and separately authorized budget; do not delete these results.
