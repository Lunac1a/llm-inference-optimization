# vLLM project plan

Stage 1 and Stage 2 are complete under their own locked protocols. The current
authorized route is the local KV-cache combination study in
[stage3-plan.md](stage3-plan.md), followed by a bounded local API delivery
around the selected configuration. Archived llama.cpp work is not vLLM
acceptance evidence.

| Stage | Scope | Exit evidence |
| --- | --- | --- |
| 1. Local baseline | Pin vLLM, model, WSL compatibility settings, and the existing API baseline | Environment lock, API acceptance, stable benchmark, and preserved raw evidence |
| 2. Bottleneck investigation | Attribute the bounded workload's queueing and execution behavior | Controlled measurements, traces, limits, and no-go boundaries |
| 3. Local KV combination validation | Compare FlashAttention/Triton, BF16/FP8 KV, and prefix caching on shared long-document QA | Compatibility records, fixed request/quality materials, raw responses, telemetry, analysis, hashes, and selection decision |
| 4. Local API delivery | Keep the existing vLLM API and serve only the selected configuration | Normal/streaming/restart/cache/document-switch acceptance and cleaned local service |

## Boundaries

- Keep Qwen3-4B BF16 weights, the pinned model revision, vLLM 0.23.0, and the
  existing WSL compatibility settings.
- Do not add a gateway, frontend, Docker layer, authentication, custom CUDA
  kernel, weight quantization, or lossy KV eviction.
- Use established vLLM switches only. A configuration effect is reported as a
  configuration effect, not as a custom optimization.
- Preserve Stage 1 and Stage 2 evidence. New Stage 3 artifacts are independent
  and must not modify old defaults or raw files.
- Keep the repository private. Each stage ends with a validation document,
  raw evidence, a dedicated commit, and remote verification when credentials
  permit. Separate automated checks, packaging, and real-model measurements.

## Current status

Stage 2 is complete under [stage2-plan.md](stage2-plan.md); see
[stage2-validation.md](stage2-validation.md) for the workload-specific
admission-cap result and its limits. Stage 3 compatibility/screening/quality
collection is complete, but stopped before formal comparison because A scored
41/48 against the 46/48 quality gate. See [stage3-validation.md](stage3-validation.md)
for the no-go evidence. Stage 4 has not started.
