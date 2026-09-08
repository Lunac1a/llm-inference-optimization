# Supplemental Stage 2: test the weight-bandwidth hypothesis

Authorized by the user after the initial investigation: test whether small-batch
decode is limited by reading weights. This is continued diagnosis, not Stage 3
optimization. Original Stage 1 and Stage 2 artifacts remain unchanged.

Predeclared boundary: local only; at most 20 GPU-active minutes, preserving the
initial investigation's remaining budget (45.69 of 90 minutes previously used).
No model, engine, precision, clock or power setting changes. The user explicitly
enabled driver counter access after the initial permission failure.
Use the existing Nsight Compute 2026.2.1 Linux target shipped with the installed
Windows profiler. Do not install a CUDA stack.

1. One tiny PyTorch kernel probes counter access. It is not model evidence.
2. If available, run pinned Qwen3-4B BF16, A (512 input / 128 output), client c1,
   cap8 to remove the previous admission ceiling; eager/WSL settings unchanged.
   Warm up before activating profiling. Collect at most eight matching decode
   GEMV kernels per capture, up to three captures; no parameter sweep.
3. Record kernel time, DRAM read/write bytes and throughput relative to sustained
   peak, and SM throughput. Use clock-control none. Document cache/replay effects.
   High DRAM throughput with materially lower compute throughput supports a
   bandwidth-limited kernel; low DRAM throughput does not by itself establish
   CPU limitation. Kernel evidence must be distinguished from application latency.
4. Stop on counter-permission failure, OOM, crash, request error, 120s request
   timeout, 10-minute batch timeout, or the GPU budget. Retain failures. Do not
   relabel missing counters as confirmed bandwidth pressure.

Readiness: initial launch rejected whitespace in the installed path before GPU
work; existing CLI binaries were copied to an ignored workspace directory.
The actual probe then returned ERR_NVGPUCTRPERM. The user enabled Windows GPU
counter access and confirmed this explicitly. The next probe successfully
returned all five metrics. Probe results are not model evidence.

Before model capture, use DRAM >=80% of sustained peak with SM throughput <=60%
as a conservative strong-support indicator for a sampled kernel; 60-80% DRAM
is suggestive rather than decisive. Report all samples, not just threshold hits.
These are diagnostic heuristics, not a universal classifier. The SM throughput
metric is not a pure FLOPS utilization measurement. Cache flushing/replay and
unlocked clocks limit transfer to an unprofiled full-request bottleneck claim.

Execution amendments (no experimental parameters changed):

- The first wrapper invocation mistook `capture-protocol.json` for an existing
  capture. It stopped before model launch. The guard now checks directories.
- Capture 1 produced eight valid kernels in its binary report. `--export`
  suppresses metric printing to the launch log, which the wrapper mistakenly
  treated as failed collection. A read-only report import recovered all metrics;
  capture 1 was not repeated. Continue only captures 2 and 3, charging capture 1
  against the same cumulative budget. All failure logs remain retained.
- The copied Linux CLI lacks relative section/rule resources when importing.
  The original installed Linux CLI can import without launching a target, so its
  path containing spaces is safe for this read-only step. The installed Windows
  reader independently exported capture 1 as well.
- The filter selects GEMV, not a verified decode phase. The first kernel is
  consistent with the vocabulary projection following prefill; the next seven
  match early decode projections. Report all eight and keep this distinction.

Status: all three declared captures completed; no capture repeated. The evidence
strongly supports bandwidth limitation of sampled small-batch weight GEMV
kernels. The end-to-end service bottleneck and achievable speedup remain open.

Metric interpretation follows the [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure)
and its [cache/replay limitations](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control).
Throughput percentages summarize constituent hardware counters and are not
fractions of request latency. Default kernel replay flushes caches; unlocked
clocks and serialized collection limit cross-tool timing comparisons.

## Measurement facts

The pinned Qwen3-4B BF16 server ran on the local RTX 4090 Laptop GPU with client
concurrency 1 and server cap 8. Each fresh server handled eight preparation
requests and two profiled requests, all fixed at 512 input / 128 output tokens.
All **30/30 requests** passed; **24/24 kernel samples** contain all five requested
metrics. There were no additional model captures or parameter sweeps.

| Capture | DRAM sustained peak % | SM throughput % | Predeclared strong indicator |
|---|---:|---:|---:|
| 1 | 72.66–95.23 | 24.22–61.12 | 5/8 |
| 2 | 75.22–97.63 | 25.17–46.80 | 7/8 |
| 3 | 77.01–96.03 | 21.62–56.97 | 7/8 |

The indicator remains DRAM >=80% and SM <=60%; it was not relaxed after seeing
the data. Five samples miss it. In particular, one vocabulary projection has
94.68% DRAM but 61.12% SM; four attention-output samples have DRAM below 80%.
These are mixed findings, not discarded observations.

The original Torch low-concurrency trace contains 15 decode execution windows.
GEMV accounts for **253.437/270.452 ms = 93.71% of summed kernel time** inside
those windows. This is a historical cap4, instrumented kernel-time fraction,
not an unprofiled request-latency fraction.

Raw Nsight binaries, exported CSV, request logs, telemetry and failed tooling
attempts are retained under [the supplemental evidence directory](../artifacts/stage2-bandwidth/).
The [analysis](../artifacts/stage2-bandwidth/analysis.json) contains every sample,
including bytes, duration, percentages, and decimal GB/s computed as bytes/ns.
Capture 1's independent Windows and Linux report exports agree exactly after
numeric parsing. The [trace context](../artifacts/stage2-bandwidth/trace-context.json)
records original CPU matrix dimensions linked to kernels by External id.

## Inferences

The first eight GEMV samples in each capture have the same launch grids and
blocks as the original trace's first eight GEMVs. Their DRAM read sizes match
the corresponding BF16 weight sizes closely. This supports the following
cross-run mapping; it is not a direct layer-name annotation from Nsight.

| Inferred projection | Samples | DRAM % range | SM % range | DRAM read GB/s | Strong indicator |
|---|---:|---:|---:|---:|---:|
| Vocabulary after prefill | 3 | 94.68–97.63 | 36.37–61.12 | 541.60–557.04 | 2/3 |
| Decode QKV | 6 | 80.62–87.58 | 33.56–52.07 | 463.42–503.50 | 6/6 |
| Decode attention output | 6 | 72.66–87.52 | 21.62–44.78 | 417.47–503.27 | 2/6 |
| Decode gate/up | 6 | 90.47–95.28 | 24.22–56.97 | 506.80–533.23 | 6/6 |
| Decode down | 3 | 84.40–92.18 | 27.32–53.97 | 485.18–530.03 | 3/3 |

**The weight-read bandwidth hypothesis has strong kernel-level support.**
The gate/up group is particularly clear: all six samples approach sustained
DRAM peak while the SM throughput metric is materially lower. Each reads about
99.6 MB, consistent with a 2560 x 19456 BF16 weight matrix. Overall, 17/21
inferred decode samples meet the predeclared indicator. Together with the old
trace's large GEMV share, this makes weight movement a credible major component
of single-request decode cost, beyond the previously measured cap4 queueing
effect. This remains an inference about normal serving, not a causal A/B result.

## Unresolved questions and limits

- Kernel replay clears caches and serializes collection. Normal execution may
  reuse caches differently. The experiment demonstrates pressure under these
  measurement conditions; it does not isolate normal-run bandwidth sensitivity.
- Clocks were not locked or changed. SM throughput is a hardware summary metric,
  not pure FLOPS utilization; thresholds are diagnostic heuristics.
- Samples are the first eight matching GEMVs, including one likely post-prefill
  vocabulary projection, followed by only early decode layers. Three restarted
  captures check repetition, not broad coverage of tokens, layers or workloads.
- Normal request-time shares of weight reads, CPU launch gaps, attention and
  other work are not separately identified. This does not prove bandwidth is
  the only service bottleneck or explain higher-concurrency saturation.
- No precision change, custom kernel, clock perturbation or optimization A/B was
  performed. The achievable end-to-end speedup is unknown. Quantization and
  other Stage 3 implementation work have not started.

## Acceptance and cleanup

[Verification](../artifacts/stage2-bandwidth/verification.json) passed: 206 original
Stage 1 hashes and 545 original Stage 2 manifest entries match; raw evidence was
not edited. The two existing Stage 2 collector regression checks passed.
The analyzer additionally validates metric units, fixed request lengths,
sample counts, grid/block correspondence and independent report-reader agreement.

Measured capture wall windows total **9.704 minutes**, including server startup,
warmup, collection and shutdown. Charging an additional **2-minute allowance**
for tiny permission probes and cleanup yields **11.704/20 minutes** supplementary
budget, or **57.395/90 minutes** including the prior Stage 2 investigation. The
allowance is conservative accounting, not separately measured GPU execution.
Time waiting for the user's driver setting and offline report analysis is
excluded. Cloud spend remains AUD 0.

Port 8000 is closed, no vLLM/Nsight processes or GPU compute processes remain.
The user's Windows counter-access setting remains as they configured it.
No Stage 3 or cloud work was performed.

Offline reproduction, from the repository root in the pinned WSL environment:

```sh
python scripts/bandwidth-trace-context.py
python scripts/analyze-bandwidth-check.py
python scripts/verify-bandwidth-check.py
```

The GPU orchestration script refuses to repeat completed captures. Its initial
hash is retained in `capture-protocol.json`; the execution amendments above
explain the wrapper changes, and final script hashes are in `verification.json`.
