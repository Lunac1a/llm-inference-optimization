# RULER v1 mixed-KV evaluation

Protocol: [docs/ruler-protocol.md](../../docs/ruler-protocol.md). This is local benchmark tooling, not part of the inference service package.

The official NVIDIA/RULER checkout is pinned in `common.py`'s experiment directory at commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`. Retrieve its Git LFS `english_words.json` payload and verify the pointer SHA-256 before data preparation. Keep all upstream copyright/license files. Task generation, templates and metrics come from upstream; local code wraps Qwen's chat template, owns service startup, collects SSE and computes paired uncertainty.

Use the existing Linux/WSL Python environment with vLLM 0.23.0. Install `requirements-data.txt` with uv into the experiment's `deps/` directory; preserve the GPU environment's NumPy and Torch versions. Model and workspace paths are explicit in `common.py`.

Preparation commands:

```bash
python benchmarks/ruler/prepare.py corpus
python benchmarks/ruler/prepare.py generate
python -m unittest discover -s benchmarks/ruler -p test_ruler.py -v
```

The corpus step obtains all official essay URLs plus SQuAD and HotpotQA, preserves raw downloads, and records hashes. HTTPS/raw-GitHub transport substitutions do not change the selected URL list. The unchanged upstream generators produce 11 samples per task/length; use row ordinal as the stable sample ID because upstream NIAH's `index` field can contain an answer character position. UUID needles are frozen in the saved inputs; seed alone is not sufficient to reconstruct OS-generated UUIDs.

After reviewing and freezing the protocol:

```bash
python benchmarks/ruler/run.py pilot
python benchmarks/ruler/run.py formal
python benchmarks/ruler/report.py
```

GPU commands are explicitly budgeted and reject repeated attempts. Never rerun them into completed evidence. Formal sample count is chosen only by the preregistered pilot-time formula. The scorer imports the official metric functions and extracts the official output-cleaning function without requiring its unrelated NeMo dependency.

Read-only evidence lives under `.local/research/ruler-tradeoff-2026-09-16/`; the generated summary is `docs/ruler-results.md`. A fresh future experiment requires a new evidence directory and protocol. Historical failures and partial results must remain distinguishable from completed evaluations.

The user-paused run has a one-shot continuation in `resume.py`, governed by [the continuation protocol](../../docs/ruler-resume-protocol.md). It preserves 96 completed responses, collects only missing responses, and counts the earlier lifecycle against the original two-hour budget. The original partial arm is snapshotted before its results file is extended; resumed raw SSE and metrics remain in a separate directory. Once attempted, continuation also refuses to rerun.
