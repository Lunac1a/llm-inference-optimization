# Local document QA prefix-cache validation

## Outcome

**Passed for the bounded sequential shared-document workload.** The deliverable
is `configs/local-document-qa.json` plus `scripts/local_document_qa.py`: a local
vLLM launcher and document-aware client for its existing OpenAI-compatible API.
See [usage](local-document-qa.md). This integrates upstream prefix caching; it
does not introduce a custom cache algorithm or a new server API layer.

The user explicitly redirected the project from six-way configuration selection
to this focused delivery. The [new protocol](prefix-cache-plan.md), code and
configuration were committed as `0b34ee1` before GPU measurement. The old Stage 3
quality scores remain unchanged and do not gate or substantiate these new claims.
No FP8, alternate backend, cloud, profiling or additional performance reruns.

## Measured performance

Pinned Qwen3-4B BF16 weights and BF16 KV, FLASH_ATTN, vLLM 0.23.0, WSL Ubuntu,
RTX 4090 Laptop GPU. Only prefix caching differs; both arms use the same model
alias and all other server settings. Client concurrency is **1**. Each arm has
a fresh server, two unrelated short warmups, one cold document request and eight
distinct follow-up questions. Output is fixed to 128 tokens for performance.
Three paired rounds run off/on, on/off, off/on without reruns.

| Round | Cache off follow-up p95 TTFT | Cache on follow-up p95 TTFT | Reduction |
| --- | --- | --- | --- |
| 1 | 3.563 s | 0.095 s | 97.33% |
| 2 | 4.010 s | 0.112 s | 97.21% |
| 3 | 4.199 s | 0.105 s | 97.50% |

| Measure | Cache off | Cache on |
| --- | --- | --- |
| Mean of three follow-up p95 TTFT values | 3.924 s | 0.104 s |
| Mean of three follow-up p95 E2E values | 7.017 s | 3.287 s |
| Mean follow-up output throughput | 18.93 tokens/s | 37.93 tokens/s |
| Mean first-plus-eight-follow-up session throughput | 19.05 tokens/s | 34.17 tokens/s |
| Three-round follow-up throughput CV | 4.68% | 1.78% |

Follow-up throughput increased **100.4%**; whole-session throughput, including
the cold first request, increased **79.4%**. Throughput includes the client's
HTTP-worker startup overhead. TTFT/E2E measure time inside each HTTP worker.
Session throughput uses the summed first/follow-up active batch times; it
excludes server startup, warmups and inter-batch measurement bookkeeping.
The table averages round-level p95 values; it is not a pooled population p95.
All **54 measured performance responses** completed at exactly 128 tokens.

Cold first-request TTFT remained **3.17–3.32 s off**, **3.41–3.49 s on**. It did
not improve. The gain is in repeated document processing, not faster decoding:
mean round p95 TPOT was about 24.55 ms off and 25.10 ms on. Per-round follow-up
p95 E2E decreased 52.5–54.3%, satisfying the preregistered regression bound.

## Evidence that the prefix was actually reused

The document contains 16,384 tokens; the first full chat prompt contains 16,438.
Pinned-tokenizer preflight verified all follow-ups share at least 16,422 initial
tokens with the first request. Instructions/document stay fixed; only the
question suffix changes. No timestamp or request ID precedes the document.

Each enabled arm's first request recorded **0 / 16,438** hit/query tokens.
Each enabled follow-up batch recorded **131,328 / 131,492**, or **99.875%**.
Disabled arms recorded zero hits. Raw labelled Prometheus counters, not the
older unlabelled-only delta helper, support these values. The comparison passes
the >=90% hit-ratio gate and >=20% TTFT improvement gate in every paired round.

**Inference:** the single configuration change, observed token reuse and cold
versus warm timing support attributing this workload's benefit to avoided
repeated prefill computation. This does not show a faster decode kernel, reduced
weight traffic, larger context limit, or a general solution to KV memory pressure.

## Answers and real API delivery

- Both arms answered six source-aligned facts correctly with normal EOS:
  document identifier, date, owner, budget, region and risk. Retained answers
  are identical between arms. Equivalent natural wording is allowed; this is
  an inspected fact check, not a semantic benchmark score.
- Both arms returned document B's own identifier and owner after switching
  from A. The identifier used nonstreaming; the owner used streaming. All
  **16 fact/document-switch checks** completed without truncation or contradiction.
- After the comparison, the actual `serve` and `ask` CLI commands were run.
  Streaming A and nonstreaming B returned their correct identifiers. Terminating
  the launcher exercised its cleanup handler; the launcher exited 0 and port
  8000 closed. This verifies the shipped commands, not just an internal helper.
- Three targeted offline tests passed: only the cache flag differs, question
  suffixes preserve the document prefix, and labelled token counters are handled
  correctly without mixing in creation-time gauges.

## Limits and closure

These are three small local paired rounds on a synthetic document with one
serial client and fixed performance output length. They establish a useful
local shared-document configuration, not general model accuracy, production
p95/SLA, multi-user capacity or behavior after eviction under memory pressure.
Independent-document workloads and FP8 tradeoffs remain unmeasured here.
The six factual questions do not repair or replace the old 48-question fixture.

Budget through CLI cleanup: **8.438 minutes of 30**. No GPU compute processes
remained; port 8000 was closed and GPU memory was 625 MiB at the recorded check.
No service is intentionally left running; use the documented command when needed.

Evidence lives under `artifacts/prefix-cache/`: frozen protocol/config hashes,
documents, raw responses, counters/telemetry, per-round summaries, answer review,
actual CLI commands/responses/lifecycle logs and final hash manifest. Historical
Stage 1/2/3 evidence is preserved. README and project plan link this completed
slice separately from the failed historical six-way study.
