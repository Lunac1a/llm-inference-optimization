# Experimental native CPU KV reuse: not accepted for use

The native offload experiment stopped at its small GPU/CPU copy compatibility
gate. It is not an accepted service profile and has no performance result. Read
[validation and limitations](kv-offload-validation.md) before using these modules.

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
