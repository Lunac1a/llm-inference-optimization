# Local document QA: prefix-cache delivery

## Authorized scope change

The user approved replacing the six-way Stage 3 selection with one useful
local API slice: repeated questions over one document, using upstream vLLM
prefix caching. This protocol supersedes the six-way quality gate for this
new comparison; it does not revise the historical v1/v2 scores or evidence.
No FP8, alternate backend, cloud, profiling, gateway or new server API.

## Frozen comparison

- Use the pinned Qwen3-4B BF16 model, FLASH_ATTN, BF16 KV and existing WSL
  switches. Only prefix caching differs. Use the same served model name.
- Reuse the existing 16,384-token performance document A read-only, with
  stable instructions + document before a changing question suffix. Document
  B is the document-switch control. Record full prompt token counts/hashes.
- Client concurrency **1**, representing sequential questions from one user.
  This does not estimate multi-user concurrency capacity.
- Exactly **three paired rounds**, ordered off/on, on/off, off/on. Every arm
  starts a fresh owned server; two unrelated short warmups precede the cold
  document request. No cache-reset endpoint or cross-arm retained cache.
- Each arm: one first-document request, then eight different follow-up
  questions. Performance uses temperature 0, thinking off, exactly 128 output
  tokens (ignore EOS). Report cold TTFT separately from follow-up p50/p95 TTFT,
  E2E, output throughput and the complete first-plus-follow-up session.
- Save raw streams, counters before/after, GPU telemetry, config and source
  hashes. Sum correctly labelled prefix query/hit token counters. Show first
  versus follow-up deltas; baseline must have zero prefix hits, optimized
  follow-up hit/query ratio must be >=90%.

## Answer checks and delivery

In the first pair only, ask six source-aligned fact questions with normal EOS,
max 128 output tokens: document identifier, publication date, owner, official
budget, region and documented risk. Inspect retained answers for required facts,
wrong-document values, contradiction and truncation; equivalent wording is
allowed. Preserve the review separately from raw output. No broad quality score.
Then switch to document B and ask its identifier (nonstreaming), followed by
its owner (streaming); neither may return A's corresponding value.

The same launcher and prompt builder used by the comparison are the delivery:
foreground local serve with cleanup on Ctrl+C, and an ask command using the
existing `/v1/chat/completions`. The optimized arm's fresh starts verify restart
availability. Do not leave a server running after acceptance.
After the comparison and answer review pass, allow one final CLI smoke within
the same budget: start `serve`, query document A via streaming and document B
via nonstreaming using the delivered `ask` command, then terminate the owned
launcher and verify cleanup. These two requests are functional checks only.

## Decision, limits and budget

Deliver the enabled default as verified for this workload only if all requests
complete, all fact/switch checks pass, cache counters support actual reuse,
and each paired round reduces follow-up p95 TTFT by >=20% without >15% follow-up
p95 E2E regression. Report throughput and CV; three small rounds are descriptive,
not a production SLA or a general quality guarantee. Cold-request results are
reported separately and must not be presented as cache-hit acceleration.

This is an explicitly changed, focused protocol: the old six-way 46/48 and
throughput-CV selection gates are not transferred. No reruns, threshold changes
or extra profiling after observing results. If evidence fails, retain it and
do not label the configuration validated.

One independent **30-minute** wall budget includes startup, warmups, collection,
answer checks and cleanup. Per request <=60 s; batch <=180 s; startup <=180 s;
reserve 120 s for cleanup. Stop on OOM, crash, invalid streams/output lengths,
timeout or occupied port. New artifacts live only in `artifacts/prefix-cache/`.
Finish with a validation report, preserved old hashes, commit and push.
