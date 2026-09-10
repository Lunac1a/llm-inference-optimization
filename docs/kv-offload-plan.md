# Bounded GPU/CPU KV reuse experiment

User authorized trying hierarchical KV reuse. Commit this plan and collector before
GPU. Use fresh `artifacts/kv-offload/`, refuse overwrite, preserve all older evidence,
default BF16 API and hybrid prototype. No cloud, multi-GPU, profiling, custom kernels,
package changes, clock tuning or LMCache installation.

## Existing capability and fixed choice

Pinned installed vLLM 0.23.0 supports `--kv-offloading-size` (GiB) and
`--kv-offloading-backend native`. With `VLLM_USE_SIMPLE_KV_OFFLOAD=0`, this selects
upstream OffloadingConnector and CPUOffloadingSpec with LRU and default store
threshold zero. It saves KV blocks to CPU and loads external cached prefixes.
This is upstream capability integration, not a new tiering algorithm. It is KV
offload, NOT weight offload (`--cpu-offload-gb`). Save actual config/source hashes.
Reference: https://docs.vllm.ai/en/v0.23.0/api/vllm/config/vllm/

Use the unchanged local-document-qa model/config: Qwen3-4B BF16 revision
1cfa9a7208912126459214e8b04321603b3df60c, BF16 KV, FlashAttention, prefix on,
chunking on, eager, max_model_len 32768, batched token budget 2048, seq cap 8,
memory utilization .8, WSL Ubuntu-24.04 on RTX 4090 Laptop 16GB. No hybrid plugin.
Only treatment: native CPU KV offload, 8 GiB CPU budget. Baseline: GPU prefix only.
Both keep native GPU cache capacity; require capacities equal and greater than one
full document but smaller than all three full documents. Do not shrink GPU memory
to manufacture eviction. Three 16K BF16 caches require approximately 6.75 GiB,
so the CPU pool is intentionally sufficient while the expected GPU pool is not.

WSL normally disables pin_memory. Preserve that behavior, document it; do not
force pinned memory or change drivers. Before model startup run ONE 4 MiB
GPU->CPU->GPU roundtrip using the installed native swap_blocks_batch DMA operator
with its actual pin-memory setting. Verify byte equality, record timings as a
compatibility check only. Stop on failure; no workaround or GPU retry this run.
This test is covered by the same 30-minute wall budget as servers and cleanup.

## Requests, repetitions and acceptance

Use unchanged doc-a/b/c from hybrid-prefill materials and its independent-title
prompt layout. Exact token lengths, cross-prefix <16, same-doc >99% verified on
CPU. All performance asks: 概括本文档的项目情况。, max_tokens=128, ignore_eos=true,
temperature=0, seed=42, thinking off. Two short 16-token compile warmups/server.
Then serial A,B,C cold visits, followed by serial A,B,C revisits; no cache reset,
explicit CPU purge, concurrent requests or additional eviction workload.
Use a fixed one-second bookkeeping gap after each response in both arms; exclude
the gap from per-request latency but retain it in session wall time.

Exactly 3 pairs / 6 fresh servers, order offload/baseline, baseline/offload,
offload/baseline. 36 measured requests, 12 short warmups maximum. First pair only,
append three sequential normal-EOS questions (one per document):
项目负责人是谁，正式预算金额是多少？, max128. Inspect all six answers against the
body, accepting equivalent wording; reject omission, cross-document values,
contradictions or truncated/empty responses. Do not score forced-length text.

Require real eviction/recovery evidence: every baseline revisit has <10% GPU
prefix hits and no external transfer; every offload revisit has >=90% combined
cached tokens, positive CPU->GPU byte counters, and externally restored prefix
tokens. First visits must have no substantial cache reuse. Save local and external
prefix counters separately; API cached token count alone does not prove CPU reuse.
No performance claim if these controls fail. Finish only the current planned
collection if a latency criterion fails; do not append new points or repeats.

Promising: in each pair, offload revisit p95 TTFT <=2s AND >=30% lower than baseline;
revisit p95 E2E <=baseline (no regression), cold p95 E2E regression <=15%; complete
responses, zero preemptions, correct facts and verified transfer/cache controls.
Report actual medians and p95 of three small waves separately, not a population SLA.
Failure is a valid no-adoption result; never change thresholds after results.

## Evidence, limits, budget and stop

Record raw SSE timestamps, usage and fixed output count; per-request first-content
latency/E2E/output tok/s, cold/revisit throughput, queue counters, preemptions,
GPU active KV occupancy, GPU/cache capacities, local/connector prefix hits/queries,
bidirectional transfer bytes/time/count and effective bytes per second. Transfer
time is connector CUDA-event accumulated time, not necessarily exposed wall-clock
latency or PCIe saturation. CPU memory: sampled owned-process RSS, WSL MemAvailable,
swap usage, configured CPU KV bytes; no CPU occupancy percentage unless exposed.
SSE content-block gaps are not exact token-generation intervals. Record actual
startup flags, selected connector, WSL pinning and transfer implementation.

Fresh 30-minute wall budget from roundtrip probe; cleanup reserve120s, probe<=60s,
startup<=180s, each request<=60s absolute. Require preflight MemAvailable>=12GiB;
stop before subsequent requests if MemAvailable<1GiB or swap use grows >256MiB
from preflight. Stop on OOM, occupied port, failed copy/startup/response, missing
required transfer/reuse evidence, preemption, malformed stream or exhausted budget.
No GPU retry, environment replacement or alternate connector in this run. Preserve
environment errors separately from effective measurements. Cleanup all owned
processes, verify old evidence and frozen sources, write validation/use guidance,
commit and push current private branch. Default service remains unchanged.
