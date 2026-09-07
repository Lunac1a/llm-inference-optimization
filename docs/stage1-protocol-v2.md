# Stage 1 acceptance protocol v2 (2026-09-08)

Frozen before new benchmark collection. Supersedes the invalidated v1 acceptance,
not its raw evidence. No prior benchmark row may enter this acceptance attempt.

## Runtime and scope

Use configs/stage1-baseline.env and the existing Python dependency lock. Qwen3-4B
BF16 snapshot 1cfa9a7208912126459214e8b04321603b3df60c, one RTX 4090 Laptop,
WSL2, vLLM 0.23.0. Eager mode, V2 runner disabled, FlashInfer sampler disabled:
these are fixed compatibility settings, not a stock optimized vLLM claim.
Python headers are reproducible through setup-wsl-headers.sh. No cloud work.

## Gates

1. Regression checks must reject truncated/thinking API output, missing error
   details, missing rounds, and loss of rounds in merging.
2. Corrected API checker: 18/18 before restart and 18/18 after restart. Save both
   runs and distinct server PID/startup records. Invalid inputs must not prevent
   subsequent generation.
3. Before measured runs, execute three unmeasured preparation batches of 32
   requests (input 512, output 128, concurrency 4) to warm the runtime. Save logs.
   These batches have no acceptance thresholds and cannot be selected as results.
4. Formal matrix: input 512/2048, concurrency 1/4, fixed output 128, zero shared
   prefix, length range ratio 0 (fixed lengths), seed 42, temperature 0, ignore EOS. Each group uses four
   warmups and 32 measured requests. Run all four groups in that order for each
   of three rounds. Client and server run inside WSL; no concurrent GPU work.
5. Record per-request data, p50/p95 TTFT/TPOT/E2E, throughput and one-second GPU
   telemetry. Each of all 12 rows must have 32 successes, zero errors, and exact
   input/output lengths. No OOM. Each group's sample CV must be <=5%.
6. If any group exceeds 5%, preserve the whole run, inspect telemetry, and allow
   exactly one full four-group three-round rerun with identical settings. Select
   that entire rerun, not favorable groups across runs. No additional attempts.
   If still unstable, report a completed investigation but failed acceptance;
   do not mark Stage 1 passed. Non-stability failures stop formal collection.
7. Save source/config hashes before collection. On completion stop only the
   project's process and verify the port and model workers are gone.

API smoke uses non-thinking chat; throughput uses raw completions and measures
fixed work, not answer quality. Three rounds and 32 requests per row are small
samples; no production SLO or cross-hardware speedup claim follows from them.

Pre-collection correction: the first preparation command stopped during dataset
construction, before any requests, because vLLM 0.23.0 interprets range ratio 1
as full variation. Installed source confirms ratio 0 means fixed lengths. Both
collection scripts were corrected and hashes refreshed before formal collection.
