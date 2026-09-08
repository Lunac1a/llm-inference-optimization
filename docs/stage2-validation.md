# Stage 2: bounded bottleneck investigation

Status on 2026-09-08: **Stage 2 completed: workload-specific attribution supported.**
For A at client concurrency 8, the imposed four-sequence admission cap causes
avoidable queueing and limits throughput. Raising the existing cap to 8 improves
the three-round mean from 184.22 to 327.46 output tokens/s (+77.75%). This is
an upstream configuration effect, **not a custom optimization**. No custom
engine change is justified by that finding. Stage 3 has not started.

## Scope and provenance

Starting point: Stage 1 acceptance v2 at `c2786e8`. The existing Stage 2 plan
and its parent-plan link were committed first as `9f5778e`. Only Stage 2 is
authorized. No Stage 3 implementation, cloud work, or machine power-mode change.
The GitHub repository was verified private during this task.

Stage 1 artifacts are immutable inputs. The complete original-file SHA256
manifest is `artifacts/stage2/stage1-before.sha256`; verification is retained
in `stage1-integrity.log`. Stage 1 validation and raw evidence are not rewritten.

Qwen3-4B BF16, model snapshot, dependency lock, context 4096, batched tokens 2048,
prefix caching off, chunked prefill on, eager execution, V2 runner off and
FlashInfer sampler off are inherited. Separate `configs/stage2-cap4.env` and
`stage2-cap8.env` differ only in the server sequence cap.

## Instrumentation and readiness

The existing benchmark shell command is factored into `benchmark-point.sh`.
The Stage 1 analyzer accepts explicit request/output counts, retaining its
original 32-request/128-output defaults and acceptance rules. Seven regression
checks passed in WSL, including variable lengths, warmup errors and the pinned
CLI's disabled initial probe.

Stage 2 wraps the installed benchmark in its own client process: 120-second
HTTP timeout and retained request-return objects. Installed vLLM is untouched.
The collector uses a 600-second batch deadline, separate output directories,
one-second GPU and Linux CPU/RAM sampling, and raw `/metrics` snapshots.
Commands, settings, seeds and hashes are retained with each formal batch.

Measurement boundaries:

- Client TTFT includes more than prefill. Exact client E2E is the upstream
  request object's `latency`; per-request TPOT is derived as
  `(latency - ttft) / (output_tokens - 1)`. The unused upstream return-object
  `tpot` field is not treated as a measured zero.
- Server histogram deltas include four warmups and the measured requests.
  Client benchmark aggregates exclude warmups. Two-second settling surrounds
  the final collector's metric snapshots; early preparation used no settling.
- Running/waiting and KV gauges are sampled, not exhaustive peaks. Reserved
  VRAM is not live KV pressure. CPU/RAM telemetry describes the WSL Linux VM;
  it does not establish Windows per-core saturation.
- Windows power mode was Balanced. Windows GPU-process inventory includes
  desktop applications and permission-limited entries; their individual GPU
  cost is unavailable. No competing application was stopped.
- A task-created Windows plotting installer consumed CPU without progress and
  was stopped; plotting dependencies were then installed with uv into an
  isolated analysis directory. Setup overlapped early A screening, adding an
  order/background-load confound. Confirmation and cap contrasts occur after
  setup ended. See `plot-setup-interference.json`; no inference dependency changed.

Initial anchors measured 202.08 and 185.24 output tokens/s (8.33% difference),
failing the <=5% readiness gate. Collection stopped and the server was cleaned
up. Telemetry showed rising temperature; it did not establish thermal causality.
A bounded continuous warmup and one readiness recovery were declared before
continuation. A new validator initially expected one extra probe, but this
pinned CLI defaults `ready_check_timeout_sec=0`: all 20 actual requests had
succeeded. The false-positive checker was fixed and regression-tested, and
the interrupted recovery restarted from warmup after cleanup cooled the GPU.
Both attempts and their evidence remain retained.

The recovered anchors measured 195.03 and 196.72 tokens/s (0.86% difference),
passing readiness. Two continuous 128-request preparation batches were each
capped at 180 seconds; actual warmup work remained below ten minutes. These
preparation results are not formal performance evidence. A separate first
startup failed on CRLF before loading a model; its logs are also retained.

The original 90-minute deadline was retained across recovery; accounting is
conservative wall time including startup and idle time. No budget reset or
extra formal sweep was authorized. See `budget.json`, `recovery-decision.json`,
`collector-correction.json`, `readiness*.json`, and `formal-protocol.json`.

## Formal measurements

One exploratory pass completed all 15 points: **240/240 measured requests**,
zero errors, exact input/output lengths. Output throughput in tokens/s:

| Workload (input/output) | c1 | c2 | c4 | c8 | c16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (512/128) | 54.45 | 89.02 | 159.68 | 165.90 | 168.92 |
| B (2048/128) | 40.22 | 71.36 | 127.24 | 124.16 | 125.30 |
| C (512/512) | 40.32 | 77.40 | 151.33 | 166.90 | 154.45 |

With the declared strict <10% throughput-gain and >25% p95-latency-rise rule,
A and B first flag 4->8. C gains 10.287% at 4->8, so that transition does
**not** meet the rule; its first flag is 8->16. No threshold was rounded to
force a flag. These are screening observations under server cap4.

Before confirmation, `selection.json` chose A at c4/c8/c16: direct Stage 1
continuity, below/at/above coverage, and bounded duration. The choice does not
claim A has the universally most important bottleneck. All screening percentiles
use 16 requests per point and are not production tail-SLO estimates. Confirmation
uses 32 requests, three rounds and orders [4,8,16], [8,16,4], [16,4,8], with at
most one complete repeat if sample CV exceeds 5%.

Confirmation passed on the first complete set: **288/288 measured requests**,
zero errors, exact lengths. No full confirmation repeat was used.

| Client concurrency | Three-round mean output tokens/s | Sample CV |
| --- | ---: | ---: |
| 4 | 170.30 | 3.03% |
| 8 | 178.54 | 4.72% |
| 16 | 170.40 | 1.92% |

The <=5% rule supports this bounded comparison; it does not rule out all drift.
All three c8 values declined across rounds (187.94, 176.03, 171.66), even though
their CV passed. Preserve this order/thermal alternative when interpreting small
differences. See `confirm-stability.json` and every `confirm-r*-c*/` directory.

## Attribution, alternatives and Stage 3 boundary

### Controlled observation

The one allowed contrast used A, client c8, fresh servers for every setting,
four warmups and 32 measured requests, with cap order [4,8], [8,4], [4,8].
All **192/192 measured requests** passed. No confirmation rows were reused,
and no additional contrast or retry was run.

| Metric | Server cap4 | Server cap8 |
| --- | ---: | ---: |
| Mean output tokens/s | 184.22 | 327.46 |
| Throughput sample CV | 1.98% | 3.32% |
| Mean of batch p95 TTFT (ms) | 3092.39 | 529.72 |
| Mean of batch p95 E2E (ms) | 5636.82 | 3193.24 |
| Mean client TPOT (ms/token) | 20.19 | 21.65 |
| Mean of batch p95 TPOT (ms/token) | 21.38 | 23.50 |
| Server queue mean, including warmups (s) | 2.1220 | 0.0052 |
| Server prefill mean, including warmups (s) | 0.2250 | 0.2776 |
| Server decode mean, including warmups (s) | 2.5691 | 2.7356 |
| Maximum sampled live KV fraction | 7.01% | 14.02% |
| Maximum sampled waiting requests | 4 | 3 |

Latency rows are means of three batch summaries, **not pooled percentiles**.
Server histogram deltas include warmups. Cap8 still has transient waiting during
admission/prefill; its waiting gauge is not identically zero. No preemptions were
observed in this contrast. Per-pair throughput gains were 70.23%, 74.95%, and
88.44%; retain the whole range rather than choosing the largest gain. The
77.75% headline is the ratio of group means, not the mean of percentages.

### Execution evidence

Both short PyTorch profiles used cap4 and A, after separate warmup: c1 with two
requests, then c8 with eight requests under a configured 16-iteration bound.
The pinned wrapper stops after its counter exceeds the configured limit; both
actual worker traces contained 16 annotated execution windows. These are initial
request windows, not full or steady-state decode traces. Profiling results are
excluded from throughput comparisons.

| Captured worker trace | Low load, c1 | Near knee, c8 |
| --- | ---: | ---: |
| Prefill or mixed GPU windows | 1 | 2 |
| Decode GPU windows | 15 | 14 |
| First-to-last kernel span (ms) | 757.08 | 910.66 |
| Union of kernel intervals within that span (ms) | 664.09 | 811.14 |
| Decode-window duration not covered by kernels | 25.02% | 27.25% |
| GPU memcpy summed duration (ms) | 0.757 | 0.153 |

Both worker traces contain useful `cpu_op`, CUDA runtime, kernel and GPU-transfer
events. Visible kernels include BF16 GEMM/GEMV and attention; the worker CPU
ranges include linear/matmul and attention dispatch. CPU ranges can nest and
overlap GPU work. Kernel sums, GPU execution-window sums and application wall
time are different quantities. The non-kernel fractions are within these
profiled decode windows; they are **not predicted unprofiled speedups**.

Frontend traces contain no useful operator-level tokenization or scheduler
timing. Those costs remain unattributed. Client setup occurs between profiler
start and the first profiled request, allowing idle/clock-state changes, and
profiling itself adds substantial overhead. No memory-bandwidth, SM-efficiency
or instruction counters were collected. A busy GPU and GEMM/GEMV names do not
establish compute-bound or bandwidth-bound execution.

See [trace analysis](../artifacts/stage2/trace-analysis.json), the compressed
raw traces under `artifacts/stage2/traces/`, and `profile-*-startup.log` plus
`server-recovery-fixed-final.log` for actual profiler events and shutdown.

### Interpretation and remaining alternatives

The causal chain for **A/c8 under cap4** is:

1. Load response: throughput becomes near-flat above c4 while waiting and
   client latency rise; fresh three-round confirmation passes its stability gate.
2. Timing/execution: server queue time is substantial, live KV usage is low,
   no preemptions are observed, and traces show real GPU execution with only
   the admitted requests running.
3. Controlled contrast: changing only the sequence cap produces a large,
   consistent throughput increase and queue-time reduction in all three pairs.
4. Source: pinned `v1/core/sched/scheduler.py:104` sets
   `max_num_running_reqs = max_num_seqs`; lines 566-568 stop admitting waiting
   requests at that cap. The existing `--max-num-seqs` option already controls it.

This supports the admission cap as a dominant **avoidable limitation in this
defined configuration**, not a newly discovered engine defect or a GPU capacity
limit. It does not prove the bottleneck after cap8, at other lengths, in optimized
vLLM, on another GPU, or in a production arrival process. Cap8 trades somewhat
slower per-token decoding for less queueing and more aggregate throughput.
Laptop power/thermal drift and desktop activity remain alternatives for small
differences and variation in effect size. The first contrast restart replaced
the original shell-launched service; later subprocess-owned exits incurred the
stop helper's 30-second polling delay before reaping. All settings used fresh
startup and the same four warmups, but exact idle histories were not identical.

### One candidate for a later Stage 3 decision

**Hypothesis, not an established bottleneck:** some remaining eager decode gaps
may come from avoidable host launch/dispatch work. The source boundary is the
worker model-execution/CUDA-graph dispatch path, including
`v1/worker/gpu_model_runner.py:3817-3822`. Upstream already implements CUDA Graph
dispatch; this baseline deliberately uses eager execution for its fixed WSL
compatibility configuration. Do not build a duplicate graph or scheduler layer.

A future authorized feasibility check could compare the existing supported
graph path against eager execution for the same A workload, after correctness
and compatibility are established. Expected mechanism: fewer host launches
and smaller decode gaps. Refutation: gaps do not shrink, stable throughput does
not improve beyond noise, or compatibility/correctness fails. The current
profiled gaps may instead be profiler overhead, synchronization or dependencies.
No graph-mode experiment, custom design, kernel change or Stage 3 implementation
was performed here. **No-go for claiming a custom contribution from the cap
contrast; any further optimization decision remains open.**

## Reproduction and final checks

Run in WSL with the pinned environment. Output directories are exclusive;
use a fresh output root for any future authorized run rather than overwriting
this evidence. The historical session and recovery scripts explain this run's
exact sequence; they must not be replayed against the retained artifact root.

Reproducible collection entry points are `stage2-session.sh`, the retained
recovery scripts, `stage2-experiment.py`, `stage2-contrast.py` and
`stage2-profile.py`. Batch `command.json` files preserve exact arguments,
settings and collection hashes. `formal-source-sha256.json` freezes collection;
`attribution-source-sha256.json` records later attribution tooling. Analysis-only
changes are separate from the unchanged formal collector hashes.

For a new individually authorized batch, the shared collector accepts a fresh
directory without changing Stage 1 defaults:

```sh
STAGE1_CONFIG="$PWD/configs/stage2-cap4.env" \
STAGE1_STATE_DIR="$PWD/.tmp/stage2" \
STAGE2_BENCH=1 STAGE2_REQUEST_LOG="$PWD/NEW_DIRECTORY/requests.jsonl" \
timeout 600 bash scripts/run-stage1-benchmark.sh \
  --point 512 8 "$PWD/NEW_DIRECTORY" round1-input512-concurrency8 32 128
```

The service must be started and healthy separately. The orchestration layer
adds telemetry, budget checks and exclusive directories; the single-batch shell
command alone does not certify Stage 2 or replace the frozen experiment order.

Final evidence:

- 720/720 formal measured requests; 1,230 successful recorded requests including
  preparation, warmups and profiles. No formal sweep/confirmation/contrast rerun.
- [Combined acceptance](../artifacts/stage2/acceptance.json) passed. Seven
  regression checks passed; collector/config hashes unchanged; every
  original Stage 1 artifact passed SHA256 verification. Runtime dependency check
  passed. See `verification.json`, `regression-final.log`, `runtime-check.log`,
  `formal-collector-integrity.json`, and `stage1-integrity-final.log`.
- Conservative GPU-investigation wall time **45.69/90 minutes**, including
  startup and idle; cloud spend **AUD 0**. See `budget-final.json`.
- Final PID 29920 stopped; port 8000 closed, PID file removed, no vLLM workers
  or GPU compute processes remained. See `cleanup-verification.json` and
  `cleanup-recovery-fixed.log`. Other desktop applications were preserved.
- [Load-response plots](../artifacts/stage2/load-sweep.png) include throughput
  and p50/p95 TTFT, TPOT and E2E, with the cap4 boundary annotated. JSON raw data,
  telemetry, histogram deltas, source copies and traces remain retained.

The evidence and this report are included in the Stage 2 completion commit;
the task's final response identifies the pushed commit. No Stage 3 or cloud work
is included.
