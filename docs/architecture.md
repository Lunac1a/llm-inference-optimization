# Architecture and supported boundaries

The API layer owns HTTP validation, authentication, backend readiness, SSE forwarding and server-side budget selection. The runtime layer owns only processes it starts. Backends expose the standard vLLM chat-completion interface; an API-only deployment can point at an independently managed compatible backend.

No runtime module imports collectors or reads experiment artifacts. Frozen budget mappings are package resources included in the wheel. Model weights are external; model ID/revision and optional local model path are configuration.

## Runtime profiles

All owned profiles retain BF16 weights, eager execution, one GPU, a GPU memory fraction of0.8, and the pinned Qwen3 revision. Profiles are alternatives, not an automatically combined stack.

| Profile | Context | Max sequences | Token batch | KV | Prefix caching | Chunked prefill |
|---|---:|---:|---:|---|---|---|
| document |32768|8|2048|BF16|yes|yes|
| adaptive |4096|4|2048|BF16|no|yes|
| hybrid |16640|8|16640|per-token-head FP8|yes|no|
| cpu-kv |32768|8|2048|BF16 +8 GiB CPU pool|yes|yes|

**Prefix reuse** is an upstream vLLM capability integrated through stable document layout and explicit settings. It benefits warm repeated prefixes, not first requests or arbitrary changing prompts.

**Adaptive budgets** are task-pattern policies calibrated before held-out evaluation. Tests covered a small synthetic integer workload with shared templates. Under batch-invariant execution, quality policy matched uniform1024 total correct count while reducing tokens and batch time. Relative to economy it cost more, and individual answers could regress. Counting remained weak. Neither numerical performance claims nor model-quality guarantees transfer automatically to other tasks, sampling settings, or runtimes. Batch invariance is upstream beta functionality with a throughput cost.

**Hybrid prefill** routes a sequence to original BF16 cold attention only when its query contains the entire context. Partial-prefix and cached sequences retain upstream FP8 attention. Storage and kernels remain upstream; the project supplies scoped routing. This version requires plain Qwen3 causal attention, BF16 Q/K/V and full prefill. Concurrent cold requests can stall other work. The module stays opt-in.

**CPU KV compatibility** validates transfer descriptors against owned contiguous tensors and adapts them to upstream explicit-direction block copies. Stream/event/scheduling behavior remains upstream. The optional8 GiB pool requires at least12 GiB available host memory before startup. Full prior cold-cost/host-memory acceptance failed, so this profile remains explicitly experimental; it is not a default or a promise of larger active GPU context capacity.

Package entry points are inert unless the owned runtime selects the matching profile. They do not patch installed vLLM source files and do not automatically activate in unrelated vLLM processes. Runtime compatibility remains pinned to0.23.0.

## Validation

Product tests cover routing, resource packaging, prompt layout, profile isolation, transport descriptors, backend ownership, HTTP errors, authentication, and streaming cleanup. CPU tests do not establish GPU performance. Release smoke checks exercise real model startup, HTTP calls and owned shutdown separately; raw runs stay local. The repository contains operational documentation and product tests, not generated benchmark traces or experiment history files.
