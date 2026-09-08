# Stage 3: local KV-cache combination validation

Status: plan committed before experiment startup. The execution reached the
declared quality stop gate; see [stage3-validation.md](stage3-validation.md).
This document is the repository copy of the authorized next-stage boundary
supplied for this task.

## Goal and fixed boundary

For shared long-document question answering on the local RTX 4090 Laptop 16 GB,
select a KV-cache configuration with acceptable quality, latency, and throughput
using the existing vLLM API. Stage 3 is local combination validation; Stage 4 is
local API delivery around the selected configuration. There is no cloud phase or
cloud budget in this route.

Fixed inputs are Qwen3-4B BF16, the existing model revision, vLLM 0.23.0, the
existing WSL compatibility switches, one local GPU, and a 90-minute GPU budget
including startup, warmup, experiment, and cleanup. Do not change weights, model,
dependencies, CUDA kernels, or KV eviction behavior. Do not add a gateway,
frontend, Docker, or authentication system.

## Six candidate configurations

| ID | Attention backend | KV dtype | Prefix caching |
| --- | --- | --- | --- |
| A | `FLASH_ATTN` | BF16 | off |
| B | `FLASH_ATTN` | BF16 | on |
| C | `TRITON_ATTN` | BF16 | off |
| D | `TRITON_ATTN` | BF16 | on |
| E | `TRITON_ATTN` | `fp8_per_token_head` | off |
| F | `TRITON_ATTN` | `fp8_per_token_head` | on |

All configurations use max context 32768, max sequences 8, max batched tokens
2048, GPU utilization 0.80, eager execution, and the existing WSL settings.
`fp8_per_token_head` must be validated by actual startup and request evidence;
source declarations or CLI help are not acceptance evidence. If Triton or FP8 is
incompatible, retain the failure evidence and stop that branch without changing
the dependency or dtype.

## Workload and collection order

The new collector uses `/v1/chat/completions`, streaming timing, fixed request
materials, cache-state controls, quality scoring, raw responses, telemetry, and
budget enforcement. Existing Stage 1/2 collection defaults remain unchanged.

- Generate deterministic Chinese performance and quality documents from fixed
  seeds. The documents contain explicit facts, distractors, cross-paragraph
  relations, and missing-information cases.
- Use target input lengths 8192 and 16384 tokens; record actual token counts,
  complete request hashes, and identical requests across configurations.
- Put the document before the question. Disable thinking and use temperature 0.
- Performance requests generate 128 tokens and ignore EOS. Quality requests allow
  normal termination up to 256 tokens.

1. Start each candidate once and issue a short request. Record actual backend,
   KV dtype, cache-token capacity, cache state, and output validity.
2. Use A only for screening: 8K/16K input by client concurrency 4/8, four
   warmups and eight measurements per point. Choose the longest input with no
   request errors and p95 E2E <= 30 seconds, then the highest passing concurrency.
   If no point passes, stop before formal comparison.
3. Run the same 48-question quality set for every compatible candidate: 12 each
   for direct retrieval, cross-paragraph relation, distractor discrimination, and
   missing information, with answer locations balanced across the document.
4. Run three formal rounds for compatible candidates. Rotate A-F by two positions
   each round. Each round contains four first visits plus 16 shared-document
   follow-up questions, and 16 independent no-shared-prefix requests.
5. Validate the selected configuration through the existing API: normal and
   streaming responses, consecutive different questions, document switching,
   cold cache after restart, and oversized-input rejection.

Reset or restart between groups when prefix-cache state cannot be verified.
Record actual hits; do not assume four documents fit simultaneously. Compilation
warmup uses separate material.

## Metrics, gates, and stop conditions

Record correctness, completeness, failures, output throughput, p50/p95 TTFT,
TPOT, E2E, prefix hits, queueing, preemptions/recomputation, cache-token
capacity/use, GPU temperature/power/frequency/memory, and host resource samples.
Reserved VRAM alone is not evidence of KV savings.

- A must score at least 46/48. Otherwise stop configuration recommendation.
- A candidate must score at least 46/48 and no more than one correct answer below A.
  Empty, truncated, and incorrect answers fail.
- Relative to A, p95 TPOT regression must be <=10%, p95 E2E regression <=15%, and
  no-shared-prefix p95 TTFT regression <=15%.
- Among passing candidates, choose highest shared-load mean throughput. Within
  5%, prefer lower p95 TTFT, larger cache capacity, then simpler
  FlashAttention/BF16.
- Claim a clear service benefit only for >=10% shared throughput improvement or
  >=20% shared p95 TTFT reduction versus A. Compare quantized candidates with
  the best BF16 prefix-cache candidate before attributing any gain to FP8.
- Per-request timeout is 120 seconds and batch timeout is 10 minutes. Do not
  start a new batch without enough budget for its maximum duration and cleanup.
- Stop the current branch on OOM, crash, transport error, or invalid performance
  token lengths. Stop concurrency escalation after two consecutive screening
  batches with more than one preemption per request.
- Formal three-round throughput CV must be <=5%. Do not rerun an unstable
  selection result or add profiling rounds.

## Required delivery

Deliver the independent collector, configs, generated request materials, raw
responses, telemetry, analysis, selection decision, acceptance report, and
SHA-256 manifests under `artifacts/stage3/`. Update README and this plan while
preserving old evidence. Automated collector tests must cover streaming timing,
error/truncation handling, scoring, cache state, and budget enforcement.

The report must label measured facts, inferences, and unresolved questions
separately. The small fixed QA set is not a general model-quality guarantee.
Stop the service, verify port/GPU cleanup and old-evidence integrity, commit and
push the private repository. Stage 4 must only serve the selected configuration.
