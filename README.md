# LLM Inference Optimization with vLLM

Identify and validate one bounded inference-system improvement on top of pinned
vLLM, then deliver the selected local configuration through the existing API.

**Status: a focused local document-QA prefix-cache optimization is validated and
available through the existing API.** Across three paired rounds on one 16K
document with a serial client, mean round p95 follow-up TTFT fell from **3.924 s
to 0.104 s**; follow-up output throughput rose **100.4%**. First requests remain
cold; the measured gain applies to subsequent questions while the prefix is
cached. This is an upstream configuration integration, not a custom algorithm.
See [results and limits](docs/prefix-cache-validation.md) and
[start/query instructions](docs/local-document-qa.md).

Stage 1 and Stage 2 passed. Stage 2 found a workload-specific admission-cap
effect: for 512-input/128-output requests at client concurrency 8, changing the
existing sequence cap from 4 to 8 raised mean throughput from 184.22 to 327.46
output tokens/s (+77.75%). This is a configuration effect, not a custom
optimization. See [Stage 2 validation](docs/stage2-validation.md).

The historical Stage 3 run tested compatibility and A-only screening for six existing
vLLM combinations of FlashAttention/Triton attention, BF16/FP8 KV cache, and
prefix caching. All six passed a short request, but the fixed quality baseline
scored 41/48 against the required 46/48, so formal throughput comparison and
the originally planned Stage 4 delivery were not started at that point. See the [Stage 3 validation report](docs/stage3-validation.md)
and [Stage 3 plan](docs/stage3-plan.md). Cloud work, a new API layer, and custom
kernels are out of scope.

Review found defects in the v1 QA oracle/scorer and formal cache controls.
The 41/48 result remains historical and does not establish a model-quality
failure. The [v2 repair record](docs/stage3-v2-repairs.md) documents corrected
tools and independent materials. The [v2 real-model acceptance](docs/stage3-v2-validation.md)
completed compatibility and screening, but A scored 40/48 against the unchanged
46/48 gate. Review of its failed answers found remaining identifier wording and
process-answer scoring defects; this score does not establish a model-quality
failure. The six-way ranking remains unvalidated. The user subsequently approved
the separate [focused prefix-cache delivery protocol](docs/prefix-cache-plan.md);
its passing result does not reinterpret either historical quality score.

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

Develop and measure on the local GPU first. Keep model weights, the pinned vLLM
runtime, hardware, request materials, sampling, and output requirements fixed
within each comparison. Configuration switches are reported as configuration
effects, not as custom kernel work.

See [the staged plan](docs/plan.md) and the
[Stage 3 execution protocol](docs/stage3-plan.md). The project contribution in
this route is the evidence-backed selection, combination, validation, and local
service delivery of existing vLLM capabilities; this run reached a quality
no-go before selection and does not claim a new CUDA kernel or a weight-level
optimization.

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
