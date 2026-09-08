# Stage 3 v2 local acceptance

## Decision

The real-model run stops at the unchanged quality gate: baseline A scored
40/48 against the required 46/48, with 48 complete responses. No formal
three-round performance comparison, configuration recommendation, or selected
configuration API acceptance is justified. Stage 4 and cloud work remain unstarted.

The raw score is not a validated model-quality finding. Inspection found two
remaining fixture/scorer defects, described below. Scores are retained exactly
as collected; there is no post-hoc rescoring, threshold change, or GPU rerun.

## Protocol and measured facts

- Frozen implementation: `bfa1b92f2cfdfcbe4408e0f1ee90ed623514e723`.
  `artifacts/stage3-v2/run-protocol.json` records source/material hashes before
  inference. The original Stage 3 plan plus v2 repair amendment sets the gates.
- Pinned Qwen3-4B BF16 weights, vLLM 0.23.0, local WSL Ubuntu and RTX 4090
  Laptop GPU. A separate 90-minute budget began at 2026-09-08 14:42:23 UTC.
- All six configurations completed the compatibility request. A/B use
  FlashAttention BF16 KV, C/D Triton BF16 KV, E/F Triton per-token-head FP8 KV;
  B/D/F enable prefix caching.
- Engine-reported cache capacity was 36,528 tokens for A-D and 70,848 for E/F
  (about 1.94 times). These are token-block capacities, not measured reductions
  in total reserved GPU memory or proof of throughput improvement.

Baseline screening used four warmups and eight measured requests per point,
fixed 128-token outputs. Document lengths exclude the surrounding chat prompt;
raw API usage retains full input counts.

| Document tokens | Client concurrency | Successes | p95 E2E (s) | Screening gate |
| --- | --- | --- | --- | --- |
| 8,192 | 4 | 8/8 | 10.90 | Pass |
| 8,192 | 8 | 8/8 | 20.96 | Pass |
| 16,384 | 4 | 8/8 | 29.30 | Pass, selected |
| 16,384 | 8 | 8/8 | 48.35 | Fail: above 30 seconds |

All four measured screening batches had zero preemption-counter increments.
This A-only screening selects a workload; it is not a comparison of optimization
candidates. Its proximity to the 30-second gate is not evidence of repeatability.

## Quality evidence and remaining defects

Each configuration receives the same 48 fixed questions over four independent
16,384-token documents. Quality uses normal EOS, at most 256 output tokens,
and requires complete streams with a `stop` finish reason.

| Configuration | Correct under frozen scorer | Complete | Quality gate |
| --- | --- | --- | --- |
| A: FlashAttention, BF16, prefix off | 40/48 | 48/48 | Fail |
| B: FlashAttention, BF16, prefix on | 40/48 | 48/48 | Fail |
| C: Triton, BF16, prefix off | 41/48 | 48/48 | Fail |
| D: Triton, BF16, prefix on | 41/48 | 48/48 | Fail |
| E: Triton, FP8, prefix off | 41/48 | 48/48 | Fail |
| F: Triton, FP8, prefix on | 41/48 | 48/48 | Fail |

All 288 quality responses completed. The collector exited with code 1 only
after saving all six summaries, explicitly reporting the baseline quality gate
failure. `analysis.json` and `selection.json` record no selected candidate.
All six cache-reset attempts returned 404 and successfully fell back to an
actual owned-service restart; the retained reset/start/stop records verify that
repair in real execution. This does not certify the unrun API acceptance suite.

The baseline's eight failed items are four `direct-1` identifier questions and
four `cross-1` process-order questions:

1. The question asks for the **project identifier**, but the source explicitly
   labels the oracle value as the **document identifier**. A answers "资料未提供"
   for three documents and gives the project title for the fourth. The fixture
   does not establish that the two identifiers are interchangeable; these
   failures cannot establish a retrieval failure.
2. Process answers use lists or paraphrases, while the oracle demands one exact
   sentence including connective words. For example, A returns
   "校验数据层；执行模型评估；提交服务报告", but the expected value is
   "先校验数据层，再执行模型评估，最后提交服务报告". Punctuation normalization
   does not remove the connective-word mismatch. This example preserves the
   required sequence but is rejected. Other paraphrases require review; no
   replacement quality score is asserted.

The previous offline tests validated exact-oracle examples, transport controls,
and material separation. They did not establish that every natural-language
question aligned with its source label or that valid process paraphrases would
pass. A passing offline suite is therefore not a passing real-model acceptance.

## Inferences and unresolved questions

- FP8 compatibility and larger reported KV capacity are established on this
  pinned setup. Its quality tradeoff and service performance benefit remain
  unresolved because the quality gate cannot support candidate selection.
- Prefix caching, backend choice, and KV dtype cannot be ranked from these
  quality runs or from A-only screening. No formal throughput claim is made.
- A future separately authorized run needs a source-aligned identifier question
  and an order-sensitive process rubric that accepts validated equivalent
  answers while rejecting missing, reordered, contradictory, and wrong-document
  steps. It needs new versioned materials and evidence, not edits to this run.
- These synthetic Chinese documents do not establish general long-document QA
  quality or production/native-Linux behavior.

## Evidence and closure

Raw evidence is under `artifacts/stage3-v2/`: compatibility, screening, quality
responses/scores, server commands/logs, telemetry, budget and frozen protocol.
The original `evidence-sha256.json` remains the offline evidence anchor; the
separate run manifest covers the completed real-model run.

- Budget elapsed through verified cleanup: **30.834 minutes of 90**. No reruns,
  formal rounds, profiling, selected-configuration API tests or cloud activity.
- Cleanup verified at 2026-09-08 15:13:13 UTC: port 8000 closed, no vLLM service
  or request workers found, no GPU compute applications; GPU memory 729 MiB.
- Final hash checks: Stage 1 **206**, Stage 2 **545**, bandwidth supplement
  **66**, Stage 3 v1 **140**, and v2 offline **6** entries unchanged. All **10**
  frozen protocol source/material hashes also match.
- The offline verifier passed again; its retained output explicitly does not
  certify GPU acceptance. No implementation changes were made during this run.
- `quality-failure-audit.json` provides compact failed-answer evidence without
  rescoring; `run-verification.json` records the decision and closure checks;
  `run-evidence-sha256.json` is the separate real-model evidence manifest.
