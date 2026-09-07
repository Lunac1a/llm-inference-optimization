# LLM Inference Optimization with vLLM

Identify, implement, and validate one bounded inference optimization on top of
vLLM, deliver it through an API, and reproduce the results on a cloud GPU.

**Status: Stage 1 preflight checked; waiting for a WSL2 Linux environment.**
vLLM installation, API generation, baseline measurements, and optimization have
not started. See [the Stage 1 checkpoint](docs/stage1-validation.md).
The model, vLLM version, workload, and optimization target remain undecided.
No vLLM performance or compatibility results are claimed.

## Approach

Develop on the local GPU first. Establish a reproducible baseline, measure a
concrete bottleneck, inspect the existing implementation, and make a targeted
change. Compare against unmodified vLLM under the same constraints. Configuration
switches alone are not a custom optimization contribution.

Use the engine API initially. Add a separate API layer only for a concrete
responsibility. Cloud work follows local validation, with a total budget target
of **AUD 50**, including storage and incidental charges.

See [the staged plan](docs/plan.md). The next task is local environment and model
feasibility, not cloud provisioning or optimization implementation.

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
