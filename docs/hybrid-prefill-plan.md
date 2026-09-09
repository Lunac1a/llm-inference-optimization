# BF16 cold prefill with FP8 persistent KV: bounded prototype

## Scope and implementation choice

User authorizes trying BF16 first computation plus FP8 prefix reuse and measuring
the benefit/tradeoff. Preserve all previous evidence and the delivered default.
This is a new experiment, not an extension/rescore of fp8-capacity.

Use a local vLLM general plugin and backend subclass; do not alter site-packages.
Reuse upstream Triton causal context_attention_fwd on original BF16 Q/K/V when
the complete sequence has no cached context. Upstream cache update still writes
FP8 per-token-head KV for later decode/reuse. On a cached prefix or decode use
the original paged FP8 attention. This is per-layer BF16 computation + immediate
FP8 storage, not retaining a BF16 cache then asynchronously migrating it.
The quantization write cost remains on the first-request critical path.

Full cold prefill is necessary: old chunked prefill would read prior quantized
chunks and would not be an all-BF16 first attention pass. All new arms therefore
use max_model_len=16640, max_num_batched_tokens=16640, chunked_prefill=false.
This keeps one complete 16K document within the budget. The reduced context limit
and larger activation peak are explicit tradeoffs, not hidden tuning. Independent
document identifiers precede constant instructions, preventing the common 32-token
instruction block from turning cold document B/C into partial-prefix requests.
Preflight verifies cross-document prefix <16 tokens and same-document reuse >99%.

Three arms: native Triton BF16 KV; native Triton FP8 KV; hybrid Triton FP8 KV.
All use identical pinned Qwen3-4B BF16 weights, vLLM 0.23.0, existing GPU,
prefix caching on, memory utilization 0.8, eager mode and remaining existing
settings. Hybrid changes only the no-context attention route from native FP8.
The BF16 control uses the original paged BF16 kernel; different kernel/layout in
hybrid is part of the intervention, not a precision-only comparison. Old chunked
results are contextual only and cannot isolate this new intervention.

## Fixed sequence, budget and gates

Before GPU: commit plan; implement and test CPU routing/config controls; retain
package/backend hashes and materials under artifacts/hybrid-prefill/.
One 30-minute wall budget starts with numerical GPU checks, includes compilation,
servers, measurements and cleanup; reserve 120 seconds, startup <=180 seconds,
request <=60 seconds, wave <=180 seconds. No profiling or cloud work.

GPU numerical gate: seeded BF16 causal GQA, head dimension 128, Q heads 32,
KV heads 8, lengths 64 and 256. Compare direct upstream Triton prefill against
torch SDPA with max absolute error <=0.04 and RMS <=0.005. One run only.
Include one mixed batch (64 cold query tokens plus 4 query tokens over a 32-token
cached prefix): cold output versus SDPA, cached output versus native FP8, same
tolerances, and cache bytes unchanged by attention reads. This is a numerical
integration check, not another performance point.
Also verify FP8 write/cache read path via actual model answers and route logs.

Exactly two rounds, C=3 only, independent doc-a/b/c from fp8-capacity materials:
round 1 BF16 / FP8 / hybrid; round 2 hybrid / FP8 / BF16. Six fresh starts.
Each arm: two short warmups; one concurrent cold wave; two concurrent follow-up
waves. Performance output exactly 128 tokens, ignore EOS, temperature zero,
thinking off. 54 measured performance requests + 12 short warmups maximum.
Round 1 each arm additionally has three concurrent normal-EOS fact waves,
max 128 tokens: identifier; owner+official budget; region+risk. Inspect all 27
answers for source alignment, omissions, contradictions and cross-document values;
equivalent wording accepted, no old exact-match scorer.

Save raw streams, actual final startup arguments, backend/route logs, GPU telemetry,
queue histograms, running/waiting gauges, preemptions, KV occupancy, prefix counters,
allocated cache capacity, TTFT/E2E p50/p95 and throughput. Explicitly report the
cost of full-prefill memory profiling and whether three warm prefixes still fit.

Promising hybrid requires every response complete/correct, zero preemptions,
both rounds cold p95 TTFT <=20 s and E2E <=30 s, >=20% lower cold TTFT than the
paired native FP8 control, each follow-up p95 TTFT <=2 s/E2E <=8 s, >=90% hit/query
and <=15% follow-up E2E regression versus FP8. Report the BF16 tradeoff separately.
Otherwise do not adopt. Keep the default unchanged even if promising: this task
delivers an experimental prototype, not generic support for other models/backends.

Stop on failed numerical check, OOM, startup failure, missing/wrong route, request
timeout, malformed/truncated response, occupied port or budget exhaustion. A
normal latency/reuse gate failure is not an execution error; finish only the
predeclared paired rounds. No post-result fixes or extra performance runs to seek
a positive result. If an implementation failure occurs, preserve it and report
the prototype as not validated. At most offline repair/inspection, no GPU rerun.

Closure: document facts/inferences/unknowns, verify old artifact hashes, clean owned
processes, commit and push current private branch. No permanent plugin installation.

References: installed vllm/v1/attention/backends/triton_attn.py and
vllm/v1/attention/ops/triton_prefill_attention.py; upstream
https://docs.vllm.ai/en/latest/design/plugin_system/ and
https://docs.vllm.ai/en/stable/api/vllm/v1/attention/backends/registry/ .
