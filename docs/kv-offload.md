# Experimental CPU KV reuse: mechanism verified, acceptance incomplete

The original native batch-copy attempt stopped at its compatibility gate. A
separately authorized [compatibility recovery](kv-offload-recovery-validation.md)
now verifies CPU KV restoration with an experiment-only upstream single-copy
adapter. Its model comparison stopped on a baseline cache-control requirement;
it remains an experimental profile, with no completed repeated acceptance.

The experimental `scripts/kv_offload_entry.py` adapter sets
`--kv-offloading-size 8 --kv-offloading-backend native` only for an explicitly
selected owned offload subprocess; baseline mode adds neither flag. It disables
hybrid plugin selection and selects upstream OffloadingConnector. These flags
offload KV blocks, not model weights. The default `local_document_qa.py` service
was not modified.

`scripts/validate_kv_offload.py` contains preparation, copy-check and collection
phases. Do not rerun its historical `run` or overwrite `artifacts/kv-offload/`.
Future reproduction requires a newly authorized fresh directory and budget, with
all subprocess paths made consistent. A passing small copy test would still need
real external-prefix hits and CPU->GPU byte counters to establish cache recovery.

The continuation uses `scripts/run_kv_offload_recovery.py` and
`scripts/kv_offload_recovery_entry.py`, with the process-local general plugin in
`experiments/kv_offload_compat/`. Only the offload arm sets
`VLLM_PLUGINS=inference_kv_transport` and its plugin directory on `PYTHONPATH`.
It translates transfer descriptors to existing native `swap_blocks` calls, leaving
installed vLLM files and the default service untouched. Do not globally enable this
plugin or force pinned memory. It is version-bound to vLLM 0.23.0.

The recovery wrapper reuses the frozen collector and owns both parent and probe
output paths. Its `prepare` and `run` phases refuse overwriting historical evidence.
Do not rerun them into `artifacts/kv-offload-recovery/`. CPU-only closure is
`python scripts/finalize_kv_offload_recovery.py`; it launches no model or copy probe.
Any future model run needs fresh paths and a preregistered partial-eviction control,
without reinterpreting this stopped run as passing the old gate. Use transfer
counters and separate GPU/external prefix hits, rather than API cached tokens alone.
