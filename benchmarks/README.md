# Benchmark evidence

The current product entry point is `inference-api serve`; see the [serving guide](../docs/serving.md). This directory contains evaluation tools and evidence, not runtime dependencies.

| Directory | Purpose |
|---|---|
| `results/final-2026-09-19/` | Completed 2340-response evaluation: summary, per-response scores, answer differences and provenance |
| `evaluation/` | Frozen 0.1.3 evaluation and its explicitly authorized continuations |
| `ruler/` | Historical first-round quality evaluation and corpus preparation |
| `latency/` | Historical attention optimization experiments, including rejected candidates |

Read the [final results and limitations](../docs/final-benchmark-results.md) first. [Offline reproduction notes](../docs/final-benchmark-reproduction.md) describe the required local evidence. The current service is 0.1.4; hash verification of the 0.1.3 experiment requires its evaluated source snapshot, as explained in the [serving guide](../docs/serving.md#evidence).

Collectors contain historical paths and fixed run assumptions. They are not a general benchmark CLI and must not be rerun over completed evidence. Raw prompts, model weights, corpora, SSE logs and environment installations remain Git-ignored under local directories. Published JSON files contain compact results only; a fresh clone does not include the full local reproduction assets.
