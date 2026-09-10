# Bounded CUDA Graph decode comparison

Commit this plan, collector and CPU preflight before GPU. Fresh evidence:
artifacts/cuda-graph. Preserve all earlier evidence and the default/API profiles.
No CPU offload or FP8 hybrid, profiling, compilation tuning, dependency changes,
new kernels/scheduler, cloud or multi-GPU work. No retries or configuration scan.

Historical eager is a WSL compatibility baseline; no earlier graph comparison
is recorded. Installed vLLM 0.23.0 CompilationConfig explicitly allows full CUDA
graphs without compilation. Compare eager against FULL_DECODE_ONLY with mode=0
(no torch compilation), capture sizes [1,2,4], maximum capture size 4. Mixed/prefill
stays outside graphs. Both arms enable native cudagraph metrics. Require actual
FULL runtime stats and successful capture for graph, NONE runtime stats for eager;
otherwise compatibility/evidence stop. Retain actual launch/config and source hashes.

Use existing Qwen3-4B BF16 revision, BF16 KV, FlashAttention, V1 runner, disabled
FlashInfer sampler, prefix caching, eager baseline, context32768, token budget2048,
sequence cap8, GPU utilization .8 on the existing RTX4090 Laptop WSL system.
Only graph treatment removes --enforce-eager and sets the above compilation config.
No GPU capacity shrinking; record any graph memory / KV capacity difference.

Fixed performance document: first 2048 tokenizer tokens of existing doc-a body,
roundtrip token length asserted; fixed question from prior experiments. Record
exact chat prompt length before GPU. Output512, ignore_eos=true, temperature0,
seed42, thinking off. Per server two 32-token warmups on this same document;
then three serial requests, then one simultaneous batch of four requests.
All performance prompts must have >=90% GPU prefix reuse; zero preemptions.
Three fresh paired orders graph/eager, eager/graph, graph/eager: 42 performance
requests, 12 warmups. First pair only: three short body-checkable owner/budget
questions per arm, normal EOS max128, six answers total; equivalent wording
accepted, no omission/contradiction/cross-document values. No extra load points.

Measurements: first-content TTFT, E2E, output/E2E throughput, observed client
post-first-content rate=(output_tokens-1)/(last_content-first_content), max/p95
SSE content gaps, wave throughput, queue/preempt/cache metrics, native graph
runtime stats, startup/capture time, GPU memory and KV capacity, CPU RSS/memory.
Client content intervals are NOT exact token-generation intervals. Native graph
stats establish selected runtime path; do not claim profiler-level launch counts.

Promising only if EACH pair at BOTH concurrency1/4: median client post-first rate
improves >=10%, p95 E2E falls >=5%, p95 TTFT <=max(baseline*1.2, baseline+.050s).
All responses/quality correct; graph KV capacity must remain >=90% of eager.
These are small fixed-wave statistics, not population SLAs. Complete only the
planned collection if latency gates fail; never append points to obtain a pass.

Budget20min including six startups and cleanup; startup<=180s, each request<=60s,
reserve120s, >=300s left before each server. Stop on compatibility/startup/capture/
response/control failure, OOM, missing runtime-mode evidence, occupied port,
MemAvailable<1GiB or swap growth>256MiB. Preflight MemAvailable>=8GiB. No GPU
retry, graph-mode fallback or environment repair within this run. Cleanup owned
services, audit old evidence and installed/default sources, update validation/use
guidance, commit/push current private branch. Graph remains an experiment until
the complete gates pass; no automatic default/API change in this task.
