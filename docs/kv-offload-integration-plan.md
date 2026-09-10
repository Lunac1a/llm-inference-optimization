# CPU KV API integration: fixed partial-eviction acceptance

User authorized completing CPU integration. Preserve every historical artifact;
earlier <10% baseline gate remains failed. New evidence only in
`artifacts/kv-offload-integration/`. Commit plan, implementation and CPU preflight
before GPU. No FP8/hybrid combination, new scheduler, kernel, driver or package edits.

Reuse the exact model, BF16 weights/KV, GPU, 8 GiB CPU pool, single-copy adapter,
16K documents, 128-token fixed output, seed, serial cold A/B/C -> revisit A/B/C,
three paired orders and six normal-EOS fact questions from kv-offload-plan.md.
Three pairs: offload/baseline, baseline/offload, offload/baseline. Same 30-minute
budget, copy probe <=60s, startup <=180s, requests <=60s, cleanup reserve, memory
and swap guards; no GPU retries. Maximum 36 performance + 12 warmup + 6 quality
requests, plus two short normal-EOS nonstream API checks (first and last offload
server, before warmups). Same model alias, local OpenAI chat endpoint and SSE.

New control fixed before execution: every revisit must have 0 <= GPU local hits
<90% of prompt tokens, proving some eviction; baseline must have zero external
hits/transfers. Offload must have >=90% combined hits, positive external hits and
CPU->GPU bytes. Cold hits <10%; zero preemptions. After each pair, same-document
GPU hit counts must differ by <=16 tokens (one block); otherwise stop. Record
computed prefill tokens separately to explain retained-prefix and restoration
effects. Do not reduce GPU capacity; all six capacities must match and fall
between one full prompt and three. This tests partial eviction, not total eviction.

Unchanged performance acceptance for EACH of three pairs: offload revisit p95
TTFT <=2s and >=30% lower; revisit p95 E2E no regression; cold p95 E2E <=15%
regression. Finish only planned collection on latency failure; stop immediately
on invalid control/runtime/memory/budget failure. No result-selected extra repeats.
Fact answers require normal EOS, source-supported owner/budget, no omission,
contradiction or cross-document leakage; equivalent wording accepted by review.

Integrate an explicit `local_document_qa.py serve --cpu-kv-cache` option using
the existing scoped transport, startup memory guard and owned-process cleanup.
Keep ordinary serve behavior unchanged. Real offload trials must use this same
server class, with nonstream API checks on its first and restarted last instance;
all performance/quality requests exercise streaming. Offline checks cover option
selection, environment restoration and rejection of incompatible prefix-off mode.
Only label the optional profile accepted after all gates and API checks pass.

Record raw request/metrics/telemetry, transfer directions, queue, preemptions,
GPU capacity/active occupancy, CPU budget/RSS/swap, SSE-content gaps (not exact
token intervals), and actual launch flags. Finalize preservation hashes/tests,
cleanup, validation/use docs and commit/push private current branch. If accepted,
report bounded serial-workload support; concurrency, hybrid combination and
single-request contexts larger than GPU capacity remain unverified.
