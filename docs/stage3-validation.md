# Stage 3: local KV-cache combination validation

Review amendment: the historical run below is preserved, but its quality
materials/scorer were found defective. Treat 41/48 as the archived scorer's
output, not a validated model-quality conclusion. Tooling has been repaired
and checked offline; see [the v2 repair record](stage3-v2-repairs.md).
No corrected GPU run or final configuration acceptance has occurred.

Status on 2026-09-08: **stopped before formal throughput comparison and
configuration recommendation.** All six declared candidates passed a short
compatibility request, but the required A quality baseline scored 41/48 rather
than the preregistered minimum 46/48. The plan therefore forbids a formal
combination recommendation. This is a valid bounded no-go result, not a
throughput comparison.

## Scope and provenance

The repository copy of the authorized plan was committed first as `c123d46`
(`docs: authorize Stage 3 KV combination validation`). Stage 1 and Stage 2
tracked evidence was not rewritten. This task used Qwen3-4B BF16 revision
`1cfa9a7208912126459214e8b04321603b3df60c`, vLLM 0.23.0, Ubuntu 24.04.4 LTS
under WSL2, one RTX 4090 Laptop GPU with 16,376 MiB VRAM, and the existing WSL
compatibility switches. No dependency, model, weight, kernel, gateway, Docker,
or cloud resource was added.

The independent collector is `scripts/stage3_collect.py`; its shared service,
streaming client, Prometheus parser, telemetry sampler, and budget guard are in
`scripts/stage3_lib.py`. Existing Stage 1/2 defaults remain unchanged.

## Materials and compatibility

The fixed materials contain four deterministic Chinese documents at each target
length. The pinned tokenizer measured all documents exactly at 8,192 or 16,384
tokens. There are 32 performance prompts per length and 48 quality questions:
12 direct retrieval, 12 cross-paragraph, 12 distractor, and 12 missing-
information questions. Complete materials and hashes are under
`artifacts/stage3/materials.json` and `materials-manifest.json`.

| Candidate | Backend observed | KV dtype observed | Prefix setting | GPU KV blocks × block size |
| --- | --- | --- | --- | ---: |
| A | `FLASH_ATTN` / FlashAttention | `bfloat16` | off | 2,283 × 16 = 36,528 tokens |
| B | `FLASH_ATTN` / FlashAttention | `bfloat16` | on | 2,283 × 16 = 36,528 tokens |
| C | `TRITON_ATTN` | `bfloat16` | off | 2,283 × 16 = 36,528 tokens |
| D | `TRITON_ATTN` | `bfloat16` | on | 2,283 × 16 = 36,528 tokens |
| E | `TRITON_ATTN` | `fp8_per_token_head` | off | 4,428 × 16 = 70,848 tokens |
| F | `TRITON_ATTN` | `fp8_per_token_head` | on | 4,428 × 16 = 70,848 tokens |

Each candidate started, served `/v1/chat/completions`, returned HTTP 200, and
produced valid text. The startup log and raw short request for every candidate
are under `artifacts/stage3/runs/compatibility-*` and
`artifacts/stage3/compatibility/`.

The optional `/reset_prefix_cache` route returned 404 in this vLLM server
configuration. No formal shared/no-shared group was started after that finding;
the plan's fallback restart rule remains recorded in the collector boundary for
future authorized work.

## A-only screening

The screening used four warmups and eight measured streaming requests per point,
with performance generation fixed at 128 tokens and `ignore_eos`. All 32
measured requests completed with no output-length anomaly.

| Input target | Client concurrency | Successful requests | p95 E2E | Output throughput |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 4 | 8/8 | 12.21 s | 54.35 tok/s |
| 8,192 | 8 | 8/8 | 19.82 s | 51.58 tok/s |
| 16,384 | 4 | 8/8 | 25.67 s | 21.61 tok/s |
| 16,384 | 8 | 8/8 | 43.61 s | 23.45 tok/s |

The selected screening load would have been 16,384 tokens at client concurrency
4: it is the longest input under the 30-second p95 E2E gate. The c8 point was
excluded by that gate.

## Quality gate and stop decision

All six candidates used the same 48 questions and completed all 48 responses.
The fixed scorer required a non-empty non-truncated answer containing the
predeclared expected marker(s); the missing-information questions accepted the
predeclared “not provided/unknown” markers.

| Candidate | Correct | Complete | Quality gate |
| --- | ---: | ---: | --- |
| A | 41/48 | 48/48 | **fail** |
| B | 41/48 | 48/48 | fail |
| C | 41/48 | 48/48 | fail |
| D | 41/48 | 48/48 | fail |
| E | 41/48 | 48/48 | fail |
| F | 41/48 | 48/48 | fail |

The seven A failures repeat in the same question categories across documents:
the training-example budget discrimination and the “what follows the process”
question. The raw scored answers are retained in each candidate's
`scored.jsonl`. This pattern may represent model retrieval difficulty or
question wording sensitivity; the current evidence does not distinguish them.
The questions were not changed after seeing the result.

Because A did not reach 46/48, the protocol stopped before formal three-round
throughput comparison. There is therefore no measured claim here about which
candidate is faster, whether FP8 improves throughput, or whether prefix caching
improves shared-load service metrics. `selection.json` records no selected
configuration, and Stage 4 API acceptance was not attempted because there was
no authorized selected configuration to deliver.

## Evidence and verification

The evidence classes are intentionally separate:

- Automated: Windows unit checks `5/5`, Python syntax/static verifier passed;
  see `tests/test_stage3_collection.py` and `scripts/verify-stage3.py`.
- Real-model compatibility/screening/quality: retained under
  `artifacts/stage3/`, with raw streaming chunks, metrics snapshots, telemetry,
  server logs, and material hashes.
- Formal comparison: **not run** due to the A quality stop gate.
- API delivery: **not run** because no candidate was selected.

The 90-minute GPU budget started at 2026-09-08T08:19:00Z. The collector stopped
after the quality gate with about 3,809 seconds remaining in its conservative
budget record. The vLLM service was stopped after every candidate; the final
health check could not connect to port 8000 and no compute process remained in
the final cleanup check. `artifacts/stage3/budget.json` and
`artifacts/stage3/evidence-sha256.json` are the audit anchors.

This fixed 48-question set is not a general model-quality guarantee. A future
authorized attempt would need a separately approved quality-material revision,
new plan/commit, and a fresh budget; it must not silently reinterpret this
no-go as a performance result.
