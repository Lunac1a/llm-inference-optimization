# Architecture and supported boundaries

The API owns HTTP validation, authentication, backend readiness and SSE forwarding. The runtime owns only processes it starts. Backends expose vLLM's standard chat-completion interface; API-only deployments can use an independently managed compatible backend.

The active scope is [mixed attention](research-scope.md). Runtime modules do not import collectors or read research archives. Model weights are external, with a pinned revision or configured local path.

## Runtime profiles

All profiles use BF16 weights, eager execution, one GPU, a GPU memory fraction of 0.8, eight sequence slots and prefix caching. Profiles are alternatives.

| Profile | Context | Token batch | KV | Chunked prefill |
|---|---:|---:|---|---|
| document (comparison) |32768|2048|BF16|yes|
| hybrid |16640|16640|per-token-head FP8|no|
| chunked-hybrid (default) |16640|2048|per-token-head FP8, fixed 4 GiB|yes|

**Serving foundation:** upstream continuous batching, prefix caching and chunked prefill, with stable document layout. Prefix reuse depends on matching tokens and available cache; it does not accelerate arbitrary cold inputs.

**Full hybrid prefill:** uses original BF16 Q/K/V when the query contains the entire context. Cached and partial-prefix routes retain upstream FP8 attention. This earlier comparison requires full prefill and can stall concurrent output.

**Chunked mixed attention:** combines historical per-token-head FP8 KV and original current BF16 K/V with shared softmax normalization. Initial full cold chunks use the existing BF16 path; decode uses upstream attention. Whole-source tiles read only FP8 history or BF16 current inputs; boundary tiles retain masked dual-source reads, followed by a shared compute body. The scoped upstream Triton extension uses `PREFILL_BLOCK_M=64`, contiguous NHD KV layout and plain Qwen3 causal attention. It does not support CPU offload or batch invariance. See [numerical and service evidence](attention-opt.md).

The default mixed profile and explicit BF16 comparison have different resource limits. Controlled comparisons must align resource settings and distinguish precision effects from scheduling changes.

## Isolation and validation

Entry points are inert unless the owned runtime selects their profile. They do not patch installed vLLM source files or activate automatically in unrelated processes. Compatibility remains pinned to vLLM 0.23.0.

Product tests cover attention routing, profile isolation, backend ownership, HTTP errors, authentication and streaming cleanup. CPU tests do not establish GPU performance. Historical numerical, microbenchmark and real-model checks remain separately documented. Earlier budget/CPU-KV service paths are removed; [archive records](archive.md) preserve their sources and evidence.
