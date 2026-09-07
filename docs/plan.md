# vLLM project plan

Decision date: 2026-09-07. Acceptance updated 2026-09-08: Stage 1 **passed under
protocol v2**, with corrected API checks, regression checks, and a complete
protocol-compliant benchmark rerun. Historical v1 acceptance remains withdrawn.
Stages 2-6 are not started.
This numbering belongs to the vLLM route; archived llama.cpp stages do not count
as vLLM acceptance evidence.

| Stage | Scope | Exit evidence |
| --- | --- | --- |
| 1. Local baseline | Check OS/runtime and GPU compatibility; select and pin vLLM, model, and configuration; run the API | Reproducible environment, stable generation, locks, and baseline measurement |
| 2. Bottleneck investigation | Select a bounded workload; measure latency, throughput, memory, and execution | Evidence of avoidable work or waiting, separating observation from inference |
| 3. Optimization design | Inspect upstream behavior and existing solutions; choose one change | Hypothesis, change boundary, controls, correctness criteria, success threshold, and stop conditions fixed before formal comparison |
| 4. Implementation and comparison | Implement and compare with pinned unmodified baseline | Correctness checks, repeatable results, overheads, limitations, and failure scenarios |
| 5. API delivery | Keep engine API; add only justified controls; containerize and document | Reproducible service invocation, logs/metrics, and experiment commands |
| 6. Cloud reproduction | Deploy after local validation | Same-machine baseline/modified comparison, environment and cost records, exported evidence, and cleanup |

## Boundaries

- Inference optimization is the primary deliverable. API and cloud work support it.
- Cache management, scheduling, and speculative decoding are candidates only.
  Investigate the bottleneck and upstream implementation before choosing.
- Fix hardware, model weights, workload, output requirements, and sampling within
  each comparison. Include strategy overhead and relevant quality, fairness,
  and latency constraints. Compare to a credible existing implementation.
- Repeat measurements and preserve raw evidence. Report regressions and uncertainty.
  If a hypothesis fails, record it and revisit the design; do not promise speedup.
- Develop locally. Cloud budget target is AUD 50 total, including storage, transfer,
  exchange rates, and applicable charges. Do not rent resources now. Multiple GPUs
  are optional and require an experimental reason.
- Keep the repository private. Use a separate task for each authorized stage;
  finish with a validation document, raw evidence, commit, and push. Distinguish
  implementation, automated checks, real-model measurements, and cloud results.

## Immediate next task

Use [stage1-validation.md](stage1-validation.md) and its linked v2 evidence as the
handoff for a separately authorized Stage 2 task. Investigate a bounded workload
before choosing an optimization; current WSL compatibility flags remain explicit
limitations. Do not reuse the invalidated historical v1 selection as a baseline.
