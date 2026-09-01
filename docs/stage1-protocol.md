# Stage 1 preregistration: confirm whether KV cache is the target bottleneck

Status: **preregistered before formal measurement** on 2026-09-01
(Australia/Sydney).

This stage diagnoses the unmodified upstream engine only. It does not implement
or evaluate `q8_kv_128`, and it must not be used as evidence that the custom
format exists.

## Question

For single-sequence decode of the locked Qwen3-8B Q4_K_M model on the locked
RTX 4090 Laptop GPU, does increasing resident context make KV-cache capacity
and traffic a material target bottleneck?

F16 is the primary baseline. Upstream `q8_0` is a control intervention: it
changes K/V representation while model weights, engine commit, workload and
hardware remain fixed.

## Locked controls

- Engine: `llama.cpp` `f8dbcd61893702976f9ab03be89c2b9f436d532c`.
- Model: `models/Qwen3-8B-Q4_K_M.gguf`, SHA-256 recorded in
  `locks/model.json`.
- GPU: device 0, all model layers offloaded, Flash Attention forced on.
- Batch/ubatch: 2048/512; CPU threads: 24; polling: 50.
- Decode workload: 128 generated tokens after synthetic resident depths of
  512, 4,096, 16,384 and 32,768 tokens.
- Five measured repetitions per `(depth, KV type)` condition.
- K and V always use the same type: F16/F16 or q8_0/q8_0.
- Conditions are interleaved within each depth to reduce order and thermal
  bias. The official benchmark warm-up remains enabled.

`llama-bench` uses a deterministic synthetic token workload and excludes
tokenization and sampling time. Therefore a second end-to-end check uses one
fixed prompt recipe with the same seed, greedy sampling and generation length
for F16 and q8_0. The microbenchmark is the primary bottleneck measurement; the
CLI check is corroboration.

## Recorded measurements

### Directly measured

- Per-repetition decode tokens/s and elapsed nanoseconds from `llama-bench`.
- GPU temperature, power, clocks, utilization and allocated VRAM sampled with
  `nvidia-smi` during each repetition.
- End-to-end `llama-cli` prompt/decode timings and actual prompt token count.
- Runtime-reported engine commit, GPU, cache types and Flash Attention setting.

### Theoretical estimate

For 36 layers, 8 KV heads and 128 values per head:

```text
F16 bytes/token = 2 (K,V) * 36 * 8 * 128 * 2 = 147,456 bytes = 144 KiB
q8_0 bytes/token = 2 * 36 * 8 * 128 * (34/32) = 78,336 bytes = 76.5 KiB
```

This predicts a 46.875% q8_0 capacity reduction. Agreement with runtime cache
allocation validates the capacity calculation but does not by itself prove a
performance bottleneck.

### Inference

The causal performance signal is the relative q8_0-vs-F16 decode result as
resident depth grows. Because q8_0 also adds dequantization work, a speedup is
not assumed in advance.

Nsight Compute DRAM counters were attempted during preregistration, but the
driver returned `ERR_NVGPUCTRPERM`. No kernel-level DRAM byte count may be
claimed unless that permission changes and a separately recorded rerun occurs.

## Data-quality gate

Formal interpretation requires all of the following:

1. Five successful repetitions for every condition.
2. Every result reports the locked engine/model, CUDA backend, full GPU
   offload, forced Flash Attention, and intended K/V types.
3. Coefficient of variation for decode tokens/s is at most 5% per condition.
4. No persistent competing compute process is observed, and paired conditions
   do not show a greater than 10% median SM-clock difference.

If a condition misses the variability gate, it is rerun once. If it still
misses, Stage 1 is **inconclusive** and not complete.

## Decision and stop conditions

Let `gain(d) = median(q8_0 tokens/s) / median(F16 tokens/s) - 1`.

The target performance bottleneck is **confirmed** only if the quality gate
passes and all of these hold:

1. F16 median decode throughput at depth 32,768 is at least 10% lower than at
   depth 512.
2. At either depth 16,384 or 32,768, q8_0 is at least 5% faster than F16, and
   the gain is larger than the combined measurement noise.
3. `gain(32768) - gain(512)` is at least 5 percentage points.

The target performance bottleneck is **not confirmed** if the quality gate
passes but any of the three performance conditions fails. This is a valid
Stage 1 no-go result: document it and do not begin Stage 2.

Stage 1 is complete only after one of those two determinate outcomes is
documented with raw data, hashes, commands and a clean validation pass. An
inconclusive run does not satisfy the stop condition.
