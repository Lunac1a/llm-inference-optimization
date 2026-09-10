# CPU KV recovery works; paired acceptance stopped

The scoped native-copy adapter restored real model KV blocks from CPU. The
performance experiment stopped on its **unchanged baseline cache-control gate**,
not a transfer or model error. There are 10 completed fixed-128-token performance
requests, four short warmups and three normal-EOS fact answers, but **zero complete
pairs of the planned three**. No adopted profile or validated repeated speedup.

## Compatibility diagnosis and implementation

The [continuation plan](kv-offload-compat-plan.md) was committed as `f877e24`, with
explicit destination-pinning instrumentation in `c130e57`, before the nine fixed
4 MiB diagnostic probes. They took 29.25 seconds including cleanup:

| Native operation | Pageable D2H / H2D | Pinned D2H / H2D | GPU to GPU |
|---|---|---|---|
| `swap_blocks_batch` | Both fail at submission, error 1 | Both fail, same error | Fails, same error |
| `swap_blocks` | Both pass byte equality | Not tested | Not tested |
| Torch `copy_` control | Both pass byte equality | Not tested | Not tested |

[Raw diagnosis](../artifacts/kv-offload-compat/results.json) isolates a failure of
the batch primitive in this installed environment. Pinned memory alone does not
resolve it. The exact driver/API/ABI cause and native-Linux behavior remain unknown.
The [v0.23.0 native source](https://github.com/vllm-project/vllm/blob/v0.23.0/csrc/libtorch_stable/cache_kernels.cu)
shows batch copying selects the resolved `cuMemcpyBatchAsync` symbol; the existing
single-copy primitive uses explicit-direction `cudaMemcpyAsync` calls.

Commit `f3554f4` adds an experiment-only general plugin. It validates and groups
the connector's source/destination descriptors, then calls upstream `swap_blocks`.
It preserves the connector's LRU, allocation, streams/events and scheduling,
introduces Python mapping/per-block launch overhead, and changes no installed
package or CUDA kernel. WSL `pin_memory=False` remains in force. Both directions
appear as `KV_OFFLOAD_COMPAT native_single` in the actual model startup log.
Two multi-block layouts, with permutations and 64 KiB / 2.25 MiB pages, passed
byte equality in both directions before model launch. No hybrid FP8 plugin used.

## Measurements and stop reason

The unchanged Qwen3-4B BF16 configuration has 36,528 GPU KV tokens in both arms;
only offload adds an 8 GiB CPU pool and the scoped transport plugin. Independent
documents have 16,384 body tokens, 16,446 performance prompt tokens, and only
three common leading tokens across documents. Fixed serial order: cold A/B/C,
then revisit A/B/C. Actual flags, token preflight, raw SSE, metrics and telemetry
are retained in [recovery evidence](../artifacts/kv-offload-recovery/analysis.json).

| First-round observation | GPU only | CPU offload + adapter |
|---|---:|---:|
| Cold A/B/C TTFT p95 (s), n=3 each | 3.330 | 3.603 |
| Cold E2E p95 (s), n=3 each | 6.434 | 6.486 |
| Revisit A TTFT (s), n=1 each | 2.852 | 0.300 |
| Revisit A E2E (s), n=1 each | 5.914 | 3.307 |
| Revisit A GPU prefix hits (tokens) | 3,360 | 3,360 |
| Revisit A external hits (tokens) | 0 | 13,072 |
| Revisit A computed prefill tokens | 13,086 | 14 |

Offload revisit B/C TTFT was 0.272/0.286 s; each also restored 13,072 tokens and
1,927,544,832 bytes (1.795 GiB). Combined reuse is 16,432/16,446 = 99.91%.
All cold requests had zero prefix hits. Every collected request had zero
preemptions and completed its fixed output length. The isolated A comparison is
an observation, **not a substitute for the missing paired waves or repetitions**.
Cold p95 E2E rose 0.81% in this first wave; repeatability is not established.

The first baseline revisit retained 3,360/16,446 = 20.43% GPU prefix hits, violating
the preregistered <10% requirement. Collection immediately stopped, before baseline
B/C revisits, baseline fact checks or rounds 2/3. The capacity condition "less than
three documents" does not imply near-total eviction of A's prefix: partial blocks
can remain. This was an overly strong workload-control assumption, not evidence
that CPU restoration failed. Thresholds, capacity and workload were not changed;
there was no GPU retry. A future separately planned comparison should explicitly
control/measure partial GPU retention instead of assuming complete eviction.

## Costs, timing scope and correctness

Offload writes about 2.26 GiB per cold request; recorded CUDA-event transfer sums
are 0.30–0.34 s. Revisit restores cost 0.20–0.21 s of event time, with server queue
time 0.21–0.23 s. These counters can include restoration wait; they are not a pure
admission-delay decomposition. Transfer bandwidth is bytes / accumulated event
time, not measured PCIe saturation. Per-request bidirectional bytes, counts, time,
queue, SSE maximum content gaps and throughput are in `analysis.json`; raw metrics
remain authoritative. No profiling or adapter-overhead isolation was performed.

Cold output throughput excluding bookkeeping gaps was 19.97 / 20.15 token/s
(baseline / offload); observed-wave throughput including internal gaps was
17.87 / 18.10 token/s. Offload revisit values were 39.79 / 32.53 token/s.
Missing baseline revisit B/C prevents a complete-wave throughput comparison.
Throughput here includes first-content wait; it is not decode-only speed.
SSE content gaps ranged up to about 0.09 s. Token IDs establish output counts,
but client arrival timestamps do not measure exact token-generation intervals.

Sampled peak summed owned-process RSS was 2.80 GiB baseline / 10.84 GiB offload;
summed RSS can double-count shared pages. Offload's minimum WSL MemAvailable was
4.42 GiB; swap remained 48 KiB. Both sampled active GPU cache gauges peaked near
45.4%; that gauge does not represent all reusable inactive prefix blocks. The CPU
pool is configured at 8 GiB, not a measured occupancy percentage. Recovery trades
host memory, CPU dispatch work and GPU/CPU transfers for avoided recomputation.

Human inspection against each document body found all three collected offload
answers correct, with no omission, contradiction or cross-document values:
A: 林若安 / 480万元; B: 周明川 / 315万元; C: 顾清禾 / 260万元. Each response stated
exactly the requested owner and budget and stopped normally after 17 tokens.
Baseline quality checks were not reached. This small fact check does not establish
broad quality, concurrent operation or combination with the hybrid FP8 prototype.

## Closure and claim boundary

Model-run budget consumed 129.86 seconds of 30 minutes including cleanup. Port
8000 is closed and no GPU compute processes remain. All 1,936 pre-run evidence
files, 55 frozen source/config files and nine installed vLLM files retain their
hashes; the separate 1,923-file pre-diagnosis snapshot also passes. Sixteen offline
tests pass. See [closure](../artifacts/kv-offload-recovery/closure.json).

Measured fact: a scoped compatibility adapter enabled real CPU KV restoration and
three fast offload revisits. Inference: avoided prefill explains much of that
latency reduction, supported by computed-token and transfer counters. Unverified:
three-pair acceptance, stable speedup, full quality comparison, concurrency,
broader workload behavior and driver root cause. Keep BF16 defaults unchanged.
Do not put a repeatable percentage speedup on the resume from this partial run.
