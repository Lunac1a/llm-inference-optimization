# LLM Inference Optimization with vLLM

Identify, implement, and validate one bounded inference optimization on top of
vLLM, deliver it through an API, and reproduce the results on a cloud GPU.

**Status: Stage 1 passed under protocol v2; Stage 2 completed with workload-specific bottleneck attribution.**
For 512-input/128-output requests at client concurrency 8, the existing server
sequence cap 4 limited throughput through queueing. Three paired rounds with
cap 8 raised mean throughput from 184.22 to 327.46 output tokens/s (+77.75%),
with passing stability gates and CPU/CUDA traces. This is a configuration
effect, not a custom optimization. See [Stage 2 validation](docs/stage2-validation.md).
Stage 3 and cloud work have not started.

A [supplemental bandwidth check](docs/stage2-bandwidth-check.md) found strong
kernel-level support for weight-read bandwidth limitation at concurrency 1:
six inferred decode gate/up GEMV samples reached 90.47–95.28% sustained DRAM
peak. This is cold-cache profiling evidence; the end-to-end bottleneck share
and achievable optimization speedup remain unresolved.

API checks passed 18/18 before and after restart. The accepted full benchmark
rerun passed 384/384 requests with four group CVs of 2.70-4.29%. This is a pinned
WSL compatibility baseline, not a custom optimization or cloud result. See
[the Stage 1 validation report](docs/stage1-validation.md) and the raw evidence
under `artifacts/stage1/`.

## Local environment

Inside Ubuntu-24.04 WSL2:

```sh
source ~/.venvs/inference-vllm/bin/activate
vllm --help
```

See [environment setup](docs/environment.md) for the pinned dependencies and
reproduction commands. Keep the Linux environment separate from the old Windows
virtual environment.

## Approach

Develop on the local GPU first. Establish a reproducible baseline, measure a
concrete bottleneck, inspect the existing implementation, and make a targeted
change. Compare against unmodified vLLM under the same constraints. Configuration
switches alone are not a custom optimization contribution.

Use the engine API initially. Add a separate API layer only for a concrete
responsibility. Cloud work follows local validation, with a total budget target
of **AUD 50**, including storage and incidental charges.

See [the staged plan](docs/plan.md). A later Stage 3 decision may investigate
remaining eager decode gaps using existing upstream mechanisms; no optimization
design, implementation, Docker, or cloud work has started.

## Previous exploration

The llama.cpp KV-quantization route is archived at Git tag
`archive/llama-cpp-stage1`, commit `99b39106539123847f860a9ac47415415e92a884`.
The upstream q8_0 control saved KV memory but did not establish a single-request
decode speedup in that setup; the custom q8_kv_128 format was never implemented.
These results are not vLLM results. Full scripts, locks, and evidence remain in
the tagged history.

Inspect the archive without changing this checkout:

```sh
git fetch origin tag archive/llama-cpp-stage1
git show archive/llama-cpp-stage1:docs/stage1-validation.md
```

For the old source layout, clone the repository into a separate directory at that
tag with --recurse-submodules. Model weights and builds were never tracked and
must be supplied separately using the archived locks.

Local legacy files are retained under ignored `.tmp/legacy-llama-cpp-stage1/`.
Existing weights under ignored `models/` are preserved; their suitability for
vLLM has not been verified. The repository remains private.
