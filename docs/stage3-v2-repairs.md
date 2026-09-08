# Stage 3 v2: review repairs, offline acceptance only

The user requested fixes after review of commit `d461b12`. This repair does
not start another GPU investigation or claim a selected configuration.
The original `artifacts/stage3/` materials, scores, raw responses and hashes
remain immutable, including the historical 41/48 output. That output cannot
be used as a validated model-quality finding because the oracle was defective.

## Corrected behavior

- The budget question explicitly asks for the official amount and unit. The
  last-step question asks for the final documented step, with an answer specific
  to that document. Quality and performance documents have different identities
  and hashes. Process steps are distributed across the beginning, middle and end.
- Version 2 questions use `answer_groups`: every group is required; alternatives
  within a group are synonyms. The prompt requests answer values only, separated
  by semicolons. The scorer compares complete normalized values (Unicode NFKC,
  ignoring punctuation/whitespace, allowing field reordering). Partial fields,
  another document's value, contradictions and extra prose do not pass. This is
  a bounded exact-answer fixture, not a general semantic answer evaluator.
- Quality requires HTTP 200, a complete stream, a nonempty answer and `stop`.
  Streaming requires `[DONE]`, a valid finish reason and token usage. Missing
  completion signals fail; `length` is acceptable for fixed-length performance
  collection but fails quality. Invalid SSE remains an error.
- First visits and follow-ups share the identical instruction/document prefix;
  only the suffix question changes. Independent requests retain distinct leading
  identifiers. The pinned tokenizer verified all 32 first/follow-up pairs at
  both document lengths share at least the full document's token-block count.
- Failed reset (including HTTP 200 with an explicit failure body) stops the owned
  server and starts a fresh healthy process in a new log directory. Startup
  failure cleans up and aborts the branch; no silent continuation occurs.
- HTTP requests execute in disposable subprocesses. Their parent enforces an
  absolute deadline and kills/reaps timed-out workers, including connections that
  keep sending data. Batch workers share one deadline; queued work cannot extend
  it. Every request ID receives a retained success/failure record. Failed warmups,
  incomplete requests and invalid fixed token lengths stop the branch after saving
  evidence. The budget is checked before every batch; server cleanup is separate.
- Worker startup overhead is included in batch throughput; API TTFT/E2E are
  measured inside the worker. Future configurations must all use this collector;
  old and new collector throughput must not be mixed into a controlled comparison.
- Formal eligibility now includes the baseline-minus-one quality constraint;
  stability requires exactly three rounds. The throughput tie-break uses shared
  follow-up TTFT, then cache capacity and the simpler backend/precision.

## Evidence boundary and future execution

New default output is `artifacts/stage3-v2/`. Shared writers reject the old
Stage 1/2/bandwidth/Stage 3 evidence roots. Existing phase/batch outputs cannot
be silently rerun or overwritten; the v2 collector rejects v1 material schemas.
The v1 committed plan and GPU artifacts remain available unchanged in history.

The existing six configurations and 46/48, baseline-minus-one, latency, CV and
90-minute protocol limits remain in force for a future separately started v2
experiment. This repair does not initialize or resume a GPU budget. No prior
scores are reinterpreted under the new rubric, and no threshold was lowered.

Prepared materials contain eight performance documents (four each at exactly
8192/16384 tokens) and 48 questions over four independent 16384-token quality
documents. Document size is not full chat input size: future collection must
retain API-reported token counts for the complete prompt.

## Offline verification

- Full pinned-WSL regression suite: **21 tests passed**. Stage 3-specific Windows
  regression also passed; see the retained logs under the v2 evidence root.
- Tests cover partial/wrong-document/contradictory answers, malformed or incomplete
  streams, trickling HTTP deadlines, batch cancellation records, reset/restart
  success/failure, shared prefixes and protected evidence paths.
- CPU-only pinned tokenizer verification checks exact document lengths, complete
  shared-prefix blocks, disjoint material hashes and all 48 answer oracles.
- All 140 original Stage 3 manifest entries match. Older stage evidence must also
  remain unchanged at commit verification.
- On Windows, sandbox-created temporary directory ACLs prevented one filesystem
  test from running; the same suite passed with normal local permissions and
  inside the pinned WSL environment. This was not a GPU/runtime failure.

Reproduce without model inference:

```sh
python -m unittest discover -s tests
python scripts/verify-stage3.py --tokenizer-path /path/to/pinned/model/snapshot
```

Real-model quality, formal throughput/latency, cache-hit behavior after actual
restart, and selected-configuration API acceptance remain **unverified for v2**.
