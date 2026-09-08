# Stage 2: bounded bottleneck investigation

Status: planned, not started. Planning date: 2026-09-08.
Baseline: Stage 1 acceptance v2, commit c2786e8.

## Objective

Identify the dominant limitation for one explicitly defined workload, supported
by load-response measurements and a short execution trace. Deliver one falsifiable
optimization hypothesis, or an evidence-backed no-go. This stage does not promise
a speedup or implement the optimization.

Keep Qwen3-4B BF16, model revision, dependency lock, single local GPU, and WSL
compatibility settings. Preserve Stage 1 files; use separate Stage 2 configs and
artifacts. No cloud rental, Docker, API wrapper, engine upgrade, or kernel work.

## 1. Instrumentation and measurement readiness

- Read the installed vLLM 0.23.0 CLI/source to verify benchmark arguments,
  available /metrics names, and profiler support. Do not assume current online
  documentation exactly matches this pinned release.
- Extend the existing collector/analyzer rather than duplicating it. Parameterize
  request count, output length, client concurrency, config and output directory.
  Stage 1 acceptance semantics must remain unchanged.
- Save per-request successes/errors, actual token lengths, TTFT, TPOT, end-to-end
  latency and throughput. Capture server running/waiting requests, queue/prefill/
  decode timing and KV usage/preemptions where available; mark unavailable fields
  explicitly. Capture host CPU/RAM and GPU temperature, power, clocks, utilization
  and memory at roughly one-second intervals where supported.
- Distinguish client wall-clock latency from server timing; TTFT is not pure
  prefill time. Record histogram deltas rather than treating cumulative metrics
  as per-run samples. VRAM allocation alone does not measure live KV pressure.
- Record power mode and competing GPU activity. Do not change machine-wide power
  settings. Use a bounded warmup (maximum 10 minutes) and two identical anchor
  batches; if throughput differs by >5%, stop and investigate device state before
  interpreting engine behavior. This is a readiness check, not accepted results.
- Freeze commands, configs, seeds, timeouts, run order and evidence hashes before
  formal collection. Add focused checks only for changed collection semantics.

## 2. Small exploratory load sweep

Use raw completions, fixed token lengths (range ratio 0), temperature 0, ignore
EOS, seed 42 and zero shared prefix. Lengths stay below context 4096.

| Workload | Input tokens | Output tokens | Purpose |
| --- | --- | --- | --- |
| A | 512 | 128 | Connect to Stage 1; locate concurrency saturation |
| B | 2048 | 128 | Increase prefill work relative to A |
| C | 512 | 512 | Increase decode work relative to A |

For each workload sweep client concurrency 1, 2, 4, 8, 16. Use 4 warmups and
16 measured requests per point, one exploratory pass (15 points maximum).
Use bounded concurrent outstanding requests, not an unbounded arrival-rate test.
These small-sample percentiles are screening evidence, not production tail SLOs.

Keep server max_num_seqs=4 for this sweep. Concurrency above 4 can expose queueing
under that cap; it does not by itself establish the GPU's capacity limit.
Use a predeclared heuristic to flag a possible knee: doubling client concurrency
gains <10% output throughput while p95 TTFT or E2E rises >25%. If no point meets
it, report no observed knee in the tested range instead of forcing one.

Select one workload and at most three points: below the knee, at it, and above
it (or low/mid/high if no knee). Explain selection before collecting confirmation.

## 3. Confirm and separate configuration effects

- Repeat selected points for three rounds, 4 warmups and 32 measured requests
  per point. Rotate point order between rounds to reduce order/thermal confounding.
  Preserve all raw data; throughput sample CV <=5% supports a stable comparison.
- If stability fails, inspect telemetry and allow at most one complete repeat of
  this selected set with unchanged settings. Keep both sets; select the entire
  repeat, never favorable rows. If still unstable, stop with an environment blocker.
- If queueing correlates with the four-sequence cap, run one diagnostic contrast:
  max_num_seqs 4 versus 8 at the same selected workload and client concurrency 8.
  All other settings stay fixed. Use three paired rounds of 32 requests per
  setting, alternating order. Reuse same-session confirmation data only when the
  pairing and controls are preserved. Do not sweep many tuning parameters.
- Treat the contrast as attribution evidence. A configuration gain is not a
  custom optimization contribution. If it only removes the imposed cap, report
  that finding and do not label the cap a new engine bottleneck.

## 4. Short profiling and causal checks

Profile only the selected workload at low load and near its knee, with 2-4
requests or a bounded iteration window after warmup. Prefer the installed vLLM
PyTorch profiler; verify that useful CPU/CUDA events are actually captured.
Keep profiling separate from throughput runs because it changes execution cost.
Do not install a full CUDA development stack just to complete this stage.

Inspect CPU scheduling/tokenization, prefill/decode kernels, kernel launch gaps,
and transfers where visible. Use these categories as hypotheses, not predetermined
answers. A busy GPU does not distinguish compute from memory bandwidth limits;
make either claim only with suitable counters or controlled supporting evidence.
Kernel-time sums are not necessarily wall-clock time when execution overlaps.

Connect each proposed bottleneck to: load response -> timing/trace evidence ->
one controlled contrast -> remaining alternatives. Inspect the relevant pinned
upstream implementation and existing feature before proposing a custom change.
If GPU profiling is unavailable in WSL, document exactly what can and cannot be
attributed; do not substitute utilization screenshots for execution evidence.

## Stop conditions and budget

- Local only; cloud spend AUD 0. Initial GPU-active investigation budget: 90
  minutes including warmup and profiling. Stop and report partial evidence if
  exhausted; do not silently expand the matrix or continue repeated attempts.
- Set 120-second request timeout and 10-minute batch timeout. On OOM, server
  crash, invalid lengths, request errors or sustained timeout, preserve evidence,
  stop escalation and check recovery. Do not include failed requests as zero latency.
- If thermal/power drift prevents stable comparisons, resolve that measurement
  problem before asserting a software bottleneck. No promise of finding one.

## Deliverables and completion

- Separate Stage 2 configuration, reproducible commands, source/config hashes,
  raw request data, telemetry and short traces under artifacts/stage2/.
- Throughput and p50/p95 latency versus concurrency plots, annotated with server
  sequence cap and configuration identity; selected-point stability summaries.
- docs/stage2-validation.md: observation, interpretation, alternatives, evidence
  paths and limits. State a workload-specific conclusion, not a universal bottleneck.
- One candidate for Stage 3: avoidable work/waiting, relevant source boundary,
  existing upstream solution, expected mechanism, and an experiment that could
  refute it. Detailed optimization design and implementation remain Stages 3-4.
- Completion can be supported bottleneck attribution or a justified no-go. An
  unresolved profiler/environment failure is recorded as blocked, not passed.
- Stop project processes, verify cleanup, commit and push evidence to the private
  repository. Run this stage in its separately authorized task.

Reference: [vLLM profiling guide](https://docs.vllm.ai/en/stable/contributing/profiling/).
It documents profiler use and significant profiling overhead; exact commands
must be checked against the locally pinned 0.23.0 installation before execution.
