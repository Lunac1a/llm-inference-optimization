# Prefix cache plus FP8 KV: frozen local capacity experiment

## Question and controls

Can FP8 sustain three distinct 16,384-token document clients where BF16
sustains two, with useful follow-up reuse, acceptable latency and correct facts?
This is a bounded service capacity definition, not a claim about maximum capacity.
Historical cache token counts only motivate C=2 and C=3; they are not results.

Current preflight: RTX 4090 Laptop, 16,376 MiB, compute capability 8.9;
WSL distribution Ubuntu-24.04, vLLM 0.23.0, torch 2.11.0+cu130.
Installed triton_attn.py explicitly supports fp8_per_token_head. Use TRITON_ATTN
for BOTH arms, BF16 Qwen3-4B weights at the existing pinned revision, prefix on.
Derive both configs from local-document-qa.json; ONLY kv_cache_dtype differs
(bfloat16 versus fp8_per_token_head). Preserve FlashAttention BF16 default.
Record installed backend source hashes, package versions, GPU, actual startup
logs and short successful API calls. No version, weight, memory or scheduler tuning.
Upstream reference: https://docs.vllm.ai/en/stable/design/attention_backends/ ;
installed fixed-version source and real startup are authoritative for this run.

## Fixed workload and budget

- Independent evidence: artifacts/fp8-capacity/. Never overwrite old evidence.
- Reuse doc-a, doc-b, doc-c from stage3-v2 materials read-only. Preflight tokenizer
  verifies distinct documents, full prompt counts and cross-document common prefix
  under 1% of document length; same-document question prefixes >=99%.
- Exactly two paired repetitions per point, no additional reruns: round 1 BF16/FP8,
  round 2 FP8/BF16. In each arm run C=2 then C=3 with a fresh server at EACH point.
  Eight server starts total. No other concurrency points or length search.
- Each start: two unrelated short warmups, one concurrent cold wave (one request
  per document), then two concurrent follow-up waves, preserving each document's
  bytes and changing only its question. No shared-document concurrent duplicates.
  Performance: temperature zero, thinking off, 128 tokens, ignore EOS.
  Total 60 measured performance requests and 16 short warmups maximum.
- Round 1 at C=3 only: three concurrent fact waves, normal EOS, max 128 tokens:
  document identifier; owner and official budget; region and documented risk.
  Nine answers per arm (18 total). Human-readable review against source excerpts
  must allow equivalent wording and reject missing, contradictory or other-document
  facts. No old exact-match scorer and no aggregate general-quality claim.
- One 30-minute wall budget starts before first server launch; startup <=180 s,
  each request <=60 s, each batch <=180 s, reserve 120 s for cleanup.
  At most one final CLI startup and two functional calls if benefit is established.

## Measurements and decision fixed before GPU serving

Every wave must have 100% complete responses and exact performance token length.
Report cold and follow-up p50/p95 TTFT/E2E separately, output tokens/wall-second,
raw labelled queue-time histogram deltas, running/waiting gauges, preemptions,
cache occupancy and prefix hit/query counters, sampled telemetry and raw streams.
Report observed queueing even if zero; missing required metrics makes evidence
inconclusive. Do not infer successful concurrent service from allocated KV tokens.

A point passes in BOTH repetitions if cold p95 TTFT <=20 s and E2E <=30 s;
EACH follow-up wave p95 TTFT <=2 s and E2E <=8 s, hit/query >=90%; zero preemptions
and all requests successful. These are workload usability gates, not production SLAs.
FP8 benefit requires BF16 C=2 pass, BF16 C=3 fail the point gates in both repetitions,
FP8 C=2 and C=3 pass, and all 18 fact answers correct. Also FP8 C=2 follow-up p95
E2E must not exceed paired BF16 by >15% (each wave). All other outcomes mean do not
adopt under this budget, including both arms passing C=3 or inconsistent repetitions.
Queueing/reuse evidence must support interpreting a BF16 failure as capacity pressure;
if attribution is unclear report inconclusive rather than a precision benefit.

Stop immediately on OOM, crash, timeout, incomplete stream or wrong output length,
occupied port, unsupported configuration, failed material checks or budget exhaustion.
Ordinary latency/reuse gate failures are retained and the fixed pairs continue;
no thresholds or workload changes after results. No profiling, cloud or extra scans.

If all gates pass, add an optional capacity-first profile to the existing launcher,
verify real serve/ask/termination cleanup within remaining budget. Otherwise retain
the default without exposing an unvalidated FP8 option. Update validation and usage,
preserve old evidence hashes, clean only owned services, commit and push this branch.
