# Stage 1: local vLLM baseline

Status on 2026-09-08: **incomplete; previous pass withdrawn after review**.
Stage 2 and later stages have not started.

## Current evidence and corrections

The model was loaded and served, and raw benchmark data exists. The earlier
17/17 API pass is invalid: requests did not actually disable thinking, and the
responses ended at the token limit while still emitting reasoning. Original API
records are retained as historical evidence, including api-check-latest.json;
that filename does not imply acceptance under the corrected checker.

The checker now sends top-level `chat_template_kwargs.enable_thinking=false` in
the raw HTTP body. Normal and streaming answers must be nonempty, contain no
thinking output, and finish with `stop`. Streams must also have a clean terminal
choice and DONE event, with no recorded transport/parse errors. Generation after
negative requests is checked explicitly. No corrected API run was executed yet.

The benchmark parser now reads the pinned vLLM parallel `errors` array. Missing
or malformed details are unknown rather than zero. Acceptance requires all 32
requests, matching input/output lengths, zero failures/errors, and timing details.
The merger reloads raw JSON, retains all three rounds of each selected group,
checks the complete round set, and recomputes CV rather than trusting old summaries.

## Existing benchmark reanalysis (no new GPU workload)

[Corrected report](../artifacts/stage1/final-benchmark-summary.json) contains 12
rows (384 requests), rather than the previous four last-round rows. All selected
requests passed the corrected data checks. The selected group CVs remain below
5%, but `protocol_compliant` and `benchmark_acceptance_passed` are **false**.

Selection uses base run `20260907T115614Z` and replacements `20260907T135907Z`
and `20260907T140950Z`. The base is itself a rerun of `20260907T113324Z`.
This exceeds the agreed allowance of one three-round rerun. Retaining provenance
is necessary but does not make this post-hoc selection protocol-compliant.
The old merged report is retained as `final-benchmark-summary.superseded.json`;
it must not be used as current acceptance evidence. Original raw results were
not modified. Per-run derived summaries are rebuilt with the corrected parser.

## Locked setup and compatibility limits

- WSL2 Ubuntu-24.04, RTX 4090 Laptop 16,376 MiB, driver 596.49.
- Python 3.12.3, vLLM 0.23.0, PyTorch 2.11.0+cu130, Transformers 4.57.6.
- Qwen/Qwen3-4B BF16, snapshot `1cfa9a7208912126459214e8b04321603b3df60c`;
  [model hashes](../artifacts/stage1/model-lock.json) are retained.
- Localhost:8000, model alias qwen3-4b-baseline, context 4096, memory fraction
  0.80, four sequences, 2048 batched tokens; prefix caching off, chunked prefill on.
- Existing startup logs report FlashAttention 2. V2 runner and FlashInfer sampler
  are disabled, and enforce-eager is enabled. These are compatibility deviations
  from normal optimized defaults, not optimization contributions. Headers are
  referenced from ignored .tmp/python-dev; their provisioning is not currently
  reproduced by the environment setup script.
- Therefore this is a compatibility baseline only, not representative native-Linux
  vLLM performance. Resolve/document setup reproducibility and the compiler path
  before treating it as the final optimization reference.

## Remaining acceptance work

1. When testing is authorized, run the corrected API checker before and after a
   recorded server restart; retain both results and startup evidence. Historical
   API records cannot be relabeled as passes under the new rules.
2. Resolve WSL compatibility setup reproducibility. Before any fresh measurement,
   freeze the final runtime flags and a new protocol explicitly superseding the
   invalidated acceptance attempt. Do not retroactively waive the original rule.
3. Under that protocol, collect a full four-group three-round baseline with at
   most one full three-round rerun; preserve all failed data and stop if still
   unstable. A new run is not authorized by this code-only repair.
4. Only then update the stage to passed. Do not enter Stage 2 now.

## Repair verification boundary

This repair executed no tests, model generation, server startup, or GPU benchmark.
Only existing data was reprocessed and source diffs inspected. Code changes are
not claimed to have passed automated or real-model validation. No cloud spend.

Reanalyze existing evidence without calling a server:

```sh
python scripts/merge-stage1-benchmark.py artifacts/stage1/benchmarks/20260907T115614Z artifacts/stage1/benchmarks/20260907T135907Z artifacts/stage1/benchmarks/20260907T140950Z --output artifacts/stage1/final-benchmark-summary.json
```

This command intentionally exits 1 for the historical protocol deviations, after
writing the diagnostic report. It must not be used to certify overall Stage 1.
