# Authorized transfer diagnosis and conditional offload continuation

The user's "continue" authorizes resolving the recorded compatibility blocker and
then continuing the original bounded model comparison. Preserve artifacts/kv-offload
and all earlier evidence. New diagnostic directory: artifacts/kv-offload-compat;
conditional model evidence: artifacts/kv-offload-recovery. Commit before GPU.

Diagnosis: at most 10 minutes including cleanup. Nine fixed independent 4 MiB
cases, each in a fresh <=30s subprocess: native batch copy with pageable/pinned
payload, both directions (4); native single swap_blocks with pageable payload,
both directions (2); torch copy_ with pageable payload, both directions (2);
native batch GPU->GPU (1). Explicitly log direction, pin status, submission and
sync results, compare bytes. Pinning is enabled ONLY for these tiny diagnostic
buffers, not globally or for the 8 GiB model cache. No repetitions/driver updates.
These are compatibility checks, never performance-selection measurements.

Decision fixed before probes: if native pageable batch succeeds in both directions,
retain it. Otherwise, if native pageable single swap_blocks succeeds in both,
permit one scoped experiment-only adapter translating native batch descriptors to
grouped calls of existing swap_blocks (explicit-direction cudaMemcpyAsync). Preserve
the connector's scheduling, synchronization/events, LRU and memory sizes; no custom
kernel, installed-package modification or large pinned-memory allocation. Validate
descriptor bounds/alignment, two small multi-block layouts in both directions with
byte equality before model use. No fallback to torch or another connector if this
fails. If neither native path works, stop with preserved diagnostic evidence.

If a transport passes, commit its concrete implementation and a fresh model-run
manifest BEFORE model launch. Resume the exact model/workload, 8GiB CPU pool,
orders, repetition counts, memory gates and performance thresholds in
docs/kv-offload-plan.md. A fresh 30-minute ceiling includes its gate and cleanup.
No GPU retry within model collection; compatibility tests are not model retries.
Report the transport adapter and its CPU overhead as part of the intervention.
Do not attribute measured speedups purely to an upstream default. Record fixed
source hashes, both transfer counters, real eviction/recovery and source-aligned QA.
Finish documentation, preservation checks, owned-process cleanup, commit/push.
