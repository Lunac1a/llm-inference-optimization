# Authorized mixed-load execution after reset API repair

The user explicitly requested execution again after the stopped attempt.
Use fresh evidence at `artifacts/mixed-load-rerun/`. Preserve the complete
`artifacts/mixed-load/` failure record and every older artifact.

The [original frozen plan](mixed-load-plan.md) applies without changes to model,
hardware, requests, arrival trigger (A token 16), output length (128), three paired
repetitions, interference/mitigation gates, conditional single chunking setting,
quality questions, timeout limits or stop rules. A new 30-minute wall budget begins
with this execution's first server start, including cleanup; no retries inside it.
This is an explicitly authorized new execution after an API integration repair,
not a repeat of any valid performance measurement or a changed acceptance threshold.

The committed experiment adapter now sets `VLLM_SERVER_DEV_MODE=1` in its own
loopback subprocess, enabling the fixed 0.23.0 reset endpoint. Verify registration
and HTTP reset handler with a CPU-only mock engine before GPU. Neither a successful
mock nor HTTP 200 proves physical eviction: real A/B cache hits and cold C cache
tokens remain mandatory checks during trials. Keep default API/backend unchanged.

`scripts/run_mixed_load_rerun.py` scopes the existing collector/analyzer to this
new directory without modifying or executing the historical run. Save wrapper,
both plan hashes and collector/adapter hashes at launch. Commit this plan, wrapper
and preflight before starting GPU. If full-prefill interference is below the fixed
gate, stop without chunked trials. Otherwise perform only the declared comparison.
Record real routing, limitations, normal-EOS answer inspection, cleanup and old
evidence verification; update docs and commit/push the current branch.
