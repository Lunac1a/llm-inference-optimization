# BF16 cold attention + FP8 persistent cache: prototype validation

## Outcome

**The bounded prototype passes the preregistered performance and fact gates.**
For three independent 16K document clients, hybrid cold p95 TTFT is
**10.74 / 9.47 s**, versus **20.86 / 21.41 s** for native FP8 under identical
full-prefill settings: **48.53% / 55.78% lower**. Follow-up latency remains close
to native FP8 and all three document prefixes remain reusable.

This delivers an experimental backend and reproducible evidence, not a replacement
for the validated FlashAttention/BF16 default. The successful result is conditional
on full prefill, the shorter context limit and the pinned model/runtime below.
See [prototype use and boundaries](hybrid-prefill.md).

## What was actually implemented

The local general plugin subclasses the fixed vLLM Triton backend. When a sequence
has **zero cached context and the entire query is present**, attention reads the
original BF16 Q/K/V via upstream causal `context_attention_fwd`. Upstream
`do_kv_cache_update` still writes `fp8_per_token_head` KV. Decode and prefix hits
read that persistent FP8 cache through the original paged attention implementation.
Mixed decode/prefill batches are sliced per sequence; a partial prefix hit never
uses incomplete original K/V as if it were a complete document.

The first attention computation and persistent storage format are separated.
There is no all-layer BF16 duplicate, asynchronous cache migration, or pretending
that dequantized FP8 recovers original BF16 values. The FP8 conversion/write still
occurs on the first request's critical path. Computation uses an existing upstream
kernel; the project contribution is routing, integration and controlled validation.

Runtime evidence verifies the direct long BF16 route on all **36 layers**, exactly
three full cold 16,446-token prompts per hybrid run, and the cached FP8 route on all
36 layers. Both FP8 arms report **55,680** cache tokens. The controls have no plugin
registration. No installed vLLM source file was changed; plugin selection is scoped
to owned subprocesses through PYTHONPATH and VLLM_PLUGINS.

## Controls and measured results

Pinned Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, BF16 weights;
vLLM 0.23.0, torch 2.11.0+cu130, RTX 4090 Laptop 16GB, WSL Ubuntu-24.04.
All arms use Triton, prefix on, eager execution, memory utilization 0.8, eight
maximum sequences, **max_model_len=16640**, **max_num_batched_tokens=16640**,
and **chunked_prefill=false**. The native FP8 and hybrid configs otherwise match.
Hybrid's kernel/layout route is the intervention; this is not a precision-only
kernel benchmark against native BF16.

Two rounds: BF16/FP8/hybrid, then hybrid/FP8/BF16. One cold and two follow-up
three-request waves per fresh server, with two short warmups. The unchanged
doc-a/b/c source texts have a stable document-specific title before the instructions.
Across documents only 3 initial prompt tokens match; own-document questions share
16,430 tokens. Full prompts are 16,443–16,452 tokens, plus 128 output tokens.

| Metric | Native BF16 | Native FP8 | Hybrid BF16 prefill / FP8 cache |
| --- | --- | --- | --- |
| Cold p95 TTFT, round 1 / 2 | 17.14 / 18.69 s | 20.86 / 21.41 s | **10.74 / 9.47 s** |
| Cold p95 E2E, round 1 / 2 | 19.95 / 21.59 s | 25.32 / 26.11 s | **15.07 / 13.85 s** |
| Follow-up p95 TTFT range | 13.02–17.49 s | 0.209–0.267 s | **0.195–0.249 s** |
| Follow-up p95 E2E range | 15.99–20.78 s | 3.96–4.34 s | **4.10–4.33 s** |
| Follow-up output throughput | 17.66–22.79 tok/s | 85.93–94.22 tok/s | **86.53–90.15 tok/s** |
| Observed running requests max | 1 | 3 | **3** |
| Engine cache tokens | 28,704 | 55,680 | **55,680** |
| Full gate, both rounds | Fail: follow-ups | Fail: cold TTFT | **Pass** |

Ranges include both repetitions and both follow-up waves, not pooled population
percentiles. Hybrid follow-up E2E relative changes versus paired native FP8 are
**+4.82%, +3.55%, +0.13%, -0.98%**, within the fixed 15% bound. Both hybrid rounds
pass cold TTFT <=20 s, cold E2E <=30 s, >=20% cold TTFT improvement, follow-up
TTFT <=2 s/E2E <=8 s, >=90% hit/query, zero preemptions and complete responses.

All **54/54** measured performance responses contain exactly 128 output tokens
with ignore EOS. Quality is separate: **27/27** normal-EOS fact answers inspected
correct (9 per arm), without omissions, contradictions, cross-document answers,
or confusing training-example budgets with official budgets. Equivalent phrasing
was accepted. See [answer review](../artifacts/hybrid-prefill/answer-review.md).

## Queueing, reuse and memory tradeoffs

Hybrid cold total queue time per three-request wave is **3.78 / 3.31 s**, versus
native FP8 **7.19 / 7.30 s**, and BF16 **20.48 / 22.69 s**. Hybrid follow-up queue
totals are 0.000027–0.000066 s; no waiting request was observed in its follow-up
samples. FP8 and hybrid follow-up hit/query ratios are **99.81–99.84%**.
All measured waves have zero preemptions. Full request, queue histogram, gauge,
TTFT/E2E p50/p95 and throughput evidence is retained in the
[18-wave table](../artifacts/hybrid-prefill/measurement-table.md), raw summaries
and one-second telemetry. Engine query counters may count repeated scheduler
lookups; they are not unique user-input token counts or request hit percentages.

Full prefill leaves **3.94 GiB** for KV in each new arm. Compared with the old
chunked setup, reported BF16 capacity falls from 36,528 to 28,704 tokens and FP8
from 70,848 to 55,680 (about 21.4% lower). Those historical numbers are context,
not a controlled isolation of the scheduling change. The new paired comparison
does control that change across all three arms. Active KV peak is about 89% for
both FP8 arms and about 58% for BF16, where one full request runs and others wait.
This gauge is active block occupancy, not total reserved VRAM or inactive-prefix
residency. The remaining BF16 space cannot hold another complete 16K request.

The prototype thus gives up some cache budget and maximum context length to make
complete original-BF16 prefill possible, but still holds three tested documents
in FP8. It does not establish capacity at C=4, longer contexts, or a benefit for
short/no-reuse workloads. Avoid interpreting the new BF16 control as a regression
in the unchanged delivered default.

## Facts, inferences and unresolved questions

Follow-up: the [completed mixed-load diagnostic](mixed-load-rerun-validation.md)
confirms cold-arrival interference. The single chunking contrast improves A's
worst pause and B TTFT but fails the A/B E2E gate, so it is not adopted. This is
separate from the historical full-wave results below; the earlier API failure
remains archived with its own evidence.

**Facts:** six successful server starts and clean stops; matching native FP8/hybrid
cache capacity; correct cold/cached routes; two paired cold-latency improvements;
three running long clients; retained prefix reuse; bounded warm-latency changes;
complete performance streams and source-aligned answers.

**Inference:** bypassing the native paged FP8 attention route for full cold queries
improves this workload while keeping FP8 storage capacity. Because quantization
writes remain, the earlier cold slowdown cannot simply be attributed entirely to
the one-time conversion cost. Kernel/layout, repeated cache reads/dequantization,
scheduling and CPU overhead are not separately profiled here. The observed benefit
belongs to the complete routing intervention, not a claimed universal FP8 defect.

**Unresolved:** arbitrary real documents, C=4+, 32K contexts, chunked prefill,
partial-prefix-heavy arrivals, other models, output quantization, CUDA graphs,
sliding windows, ALiBi/sinks/softcaps and production tail latency. The plugin rejects
several unsupported attention options and is fixed to vLLM 0.23.0. Two small waves
per condition are descriptive evidence, not a production SLA or general QA score.

## Verification and preserved failure

Plan committed as `3635738`; implementation/numerical gates as `4566164`.
The first numerical subprocess stopped before executing attention because it lacked
the existing Python.h include path. Its logs/protocol/budget/cleanup are retained
under environment-attempt/ and original collector.log. A pre-measurement amendment
was committed as `0625759`: one narrowly guarded environment recovery, with the
**original start and deadline**, no algorithm/workload/threshold change. There was
no failed numerical score or performance request to rerun. The normal comparison
started only after this repair; no later repairs or additional measured points.

All four GPU numerical comparisons passed: BF16 causal GQA at 64/256 tokens versus
SDPA, plus mixed-batch cold versus SDPA and cached versus native FP8. Maximum
absolute difference <=0.00390625 and RMS <=0.00022219; mixed comparisons matched
exactly and attention reads left cache bytes unchanged. Six offline tests cover
route boundaries, invalid lengths, matched configurations/default preservation,
subprocess environment restoration and rejection of cold gains hiding warm latency
regression. These tests are distinct from real-model acceptance.

Elapsed budget through cleanup: **9.521 minutes of 30**, including the initial
environment failure and repair. Port 8000 closed, no GPU compute applications,
984 MiB GPU memory at cleanup. All six owned servers exited 0. All **1,531** old
artifact files, **14** frozen source/config files and **5** installed vLLM sources
verify unchanged. New evidence has its own hash manifest. No service or installed
plugin is left active; the original local API configuration remains unchanged.
