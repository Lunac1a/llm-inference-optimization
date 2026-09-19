# Final evaluation (0.1.3)

This is the current benchmark entry point. Historical collectors in `../ruler` and `../latency` describe earlier frozen experiments.

Read [the protocol](../../docs/final-benchmark-protocol.md) before running GPU work. The run refuses to overwrite its evidence or silently retry failed jobs. It measures the final mixed service without further tuning.

Use the pinned Linux/WSL GPU Python. Set `INFERENCE_MODEL` to the pinned local Qwen3-4B snapshot, `RULER_SOURCE` to the pinned official checkout with prepared corpora, `RULER_DATA_DEPS` to its data dependencies, `NLTK_DATA` to its tokenizer data, and optionally `INFERENCE_EVAL_ROOT` to a new output directory. Defaults reuse the previously verified local assets. No public data or model files are embedded in the service package.

```bash
python benchmarks/evaluation/prepare.py
python -m unittest discover -s tests -v
python -m unittest discover -s benchmarks/evaluation -p 'test_*.py' -v
python benchmarks/evaluation/run.py
python benchmarks/evaluation/report.py
```

CPU preparation generates and hashes 780 fresh inputs using unchanged upstream tools. The GPU runner performs numerical integration checks, then 2340 quality responses and 54 fixed-output timing measurements. All GPU lifecycle time counts toward the 10800-second budget. Raw inputs, SSE, metrics, launch configuration, source snapshots, environment and ledger stay in the output directory. The report exports compact shareable summaries to `benchmarks/results/final-2026-09-19/` and `docs/final-benchmark-results.md` only after completeness checks succeed.

The current preparation reuses the earlier local corpus/hash manifests as provenance; a fresh clone needs those assets restored or generated through the first-round corpus tooling. This is an explicit local reproduction prerequisite, not a self-contained data download command.
