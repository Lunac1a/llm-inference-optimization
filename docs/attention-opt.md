# Optimized chunked attention (0.1.1)

The optional `chunked-hybrid` profile combines historical per-token-head FP8 KV
with original BF16 current chunks. Prefill now uses `BLOCK_M=64` (16 query tokens
per block for Qwen3-4B's GQA layout), up from 16 (4 query tokens). Cold initial
chunks retain the existing BF16 path and Decode retains the upstream path.
This is a scoped extension of vLLM 0.23.0, with original author/license notices.

Run `inference-api serve --profile chunked-hybrid` in the pinned WSL environment.
The profile fixes 2,048-token chunks, 4 GiB FP8 KV, NHD layout, eager execution,
eight sequence slots and a 16,640-token total context window. It does not combine
with CPU offloading or batch invariance. The default BF16 profile stays unchanged.

## Numerical and microbenchmark evidence

All three tiles (16, 64, 128) passed eight fixed FP32-reference cases: 24 checks,
including reversed pages, partial pages, zero/long history, GQA, unequal current
K/V strides and current lengths 1 through 2048. The reference uses dequantized
historical FP8 and original BF16 current values, not all-BF16 historical values.
Selected tile 64: worst max-absolute error 0.0106981 and RMS 0.001077, within the
unchanged limits 0.02 and 0.002. All outputs were finite.

Thirty CUDA-event samples per fixed shape after ten warmups, median milliseconds:

| History/current tokens | Tile16 | Tile64 | Tile128 |
|---|---:|---:|---:|
| 2048/2048 | 6.681 | 3.110 | 3.874 |
| 8192/2048 | 20.646 | 9.212 | 11.304 |
| 12288/2048 | 30.423 | 13.195 | 16.357 |

Tile64's geometric-mean ratio is 0.4483 of tile16. Compiled resources for these
shapes were 145/255/255 registers, 0/6/82 reported spills, and
37,248/49,536/65,920 shared bytes for tiles 16/64/128. Tile128 was slower and had
more spills; no hardware-counter evidence attributes the full difference to
spilling or establishes occupancy. No later tile tuning followed formal results.

## Two unprofiled service rounds

Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, RTX 4090 Laptop,
vLLM 0.23.0, identical 4 GiB KV capacity and the profile settings above. Inputs
are fixed token IDs, seed42, temperature0; cold and partial cases emit one token.
Partial means a 12K prefix warmup followed by a 16K request. Mixed means two
warmed 12K A/B requests and a distinct cold 12K C dispatched after both A/B emit
content and remain active; each emits 256 fixed tokens. Fixed outputs are for
timing, not answer-quality evaluation. Cache hits are verified with engine counters.

Times in seconds; each formal round reverses the four-arm order:

| Round | Arm | Cold8K | Cold16K | Partial16K | Mixed batch |
|---|---|---:|---:|---:|---:|
| 1 | Native FP8 chunked | 2.097 | 6.587 | 2.557 | 11.154 |
| 1 | Old full BF16 Prefill | 1.205 | 2.993 | 2.600 | 8.818 |
| 1 | Original dual-source tile16 | 2.181 | 7.275 | 3.089 | 11.860 |
| 1 | Optimized tile64 | 1.468 | 4.169 | 1.597 | 9.948 |
| 2 | Native FP8 chunked | 2.154 | 7.268 | 3.002 | 11.825 |
| 2 | Old full BF16 Prefill | 1.186 | 2.974 | 2.761 | 8.996 |
| 2 | Original dual-source tile16 | 2.243 | 7.568 | 3.327 | 11.989 |
| 2 | Optimized tile64 | 1.499 | 4.142 | 1.584 | 9.781 |

Against the original dual-source implementation, cold8K time falls 32.7–33.2%,
cold16K 42.7–45.3%, partial16K 48.3–52.4% and mixed-batch time 16.1–18.4%.
Both rounds pass every preregistered gate: each cold/partial ratio <=0.90,
mixed batch <=0.95, each A/B E2E <=1.05, each A/B maximum content gap <=1.10.

Maximum A/B **SSE content-block gap**, not exact token latency:

| Arm | Round1 seconds | Round2 seconds |
|---|---:|---:|
| Native FP8 chunked | 1.103 | 1.301 |
| Old full BF16 Prefill | 2.025 | 2.136 |
| Original dual-source | 1.351 | 1.351 |
| Optimized | 0.733 | 0.704 |

The old full-Prefill implementation still has lower cold latency and mixed batch
time in these observations, but longer streaming pauses. The optimized profile
is therefore optional, not a universal replacement or the default configuration.
No production p95/SLA or broad task-quality claim follows from these workloads.

## Integration and reproduction

The installed 0.1.1 profile passed 22 CPU regression tests and two real-model API
checks on `/v1/document/qa`: the 7,268-token cold question answered `violet`;
the streamed follow-up also answered `violet`, reported 7,248 cached tokens and
completed its SSE stream. Actual registration logged `prefill_block_m=64`.
The product kernel has the same function bodies as the measured candidate, with
a fixed tile constant and no research environment-variable dependency.

The 0.1.1 wheel was also installed in a clean API-only environment without Torch
or vLLM; all 22 regression tests passed there. Its 19 packaged source/data files
match the working source byte-for-byte, including upstream notices. GPU attempts
totalled 715.05 seconds (11.92 minutes), including startup and cleanup, within the
30-minute allowance; all attempts in this optimization round completed successfully.

Original baseline: `67071e9`. Frozen protocols, input token IDs, all numerical
results, microbenchmark samples, launch arguments, raw responses, resource
records and formal timings are under
`.local/research/attention-opt-2026-09-13/`. They are not runtime dependencies.
This round's collector preflight covered all arms and null cache fields before
formal collection. Prior diagnostic failures remain in their original directory.
No previously rolled-back versioned-document API or semantic router was restored.
