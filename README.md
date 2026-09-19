# Mixed-Precision KV Cache Inference

A single-GPU Qwen3/vLLM service using historical FP8 KV and current BF16 K/V mixed attention. The default API runs the mixed implementation; BF16 and earlier attention paths are retained for explicit comparisons.

## Baseline and research paths

Prefix caching, continuous batching, chunked prefill and reusable document prompts form the serving foundation. Project contributions center on mixed-attention routing, dual-source computation, query-tile optimization and single-loop tile-source branching; upstream scheduling, cache management and reused kernels retain their upstream attribution.

| Profile | Purpose | KV | Context | Token batch |
|---|---|---|---:|---:|
| `document` | Explicit BF16 comparison | BF16 | 32768 | 2048 |
| `hybrid` | Earlier full-prefill attention comparison | Per-token-head FP8 | 16640 | 16640 |
| `chunked-hybrid` (default) | Mixed-attention service | Per-token-head FP8, fixed 4 GiB | 16640 | 2048 |

All profiles enable prefix caching and retain BF16 weights, eager execution and eight sequence slots. Full `hybrid` prefill is unchunked; the other two profiles use chunked prefill. Profile resource limits differ, so out-of-box timings are not an attention-only comparison.

See [research scope](docs/research-scope.md), [attention optimization evidence](docs/attention-opt.md) and [architecture](docs/architecture.md).

The public quality evaluation uses [RULER v1](docs/ruler-protocol.md), comparing matched-resource BF16, native FP8 KV and mixed attention at 4K/8K/16K. The first 780-response evaluation is complete: [findings and trade-offs](docs/ruler-findings.md), [full results](docs/ruler-results.md), [benchmark tooling](benchmarks/ruler/README.md).

## Quick start

Version 0.1.4 packages the validated **Qwen3-4B** implementation as a small service. With a complete original Hugging Face snapshot:

```bash
inference-api doctor --model /path/to/Qwen3-4B
inference-api serve --model /path/to/Qwen3-4B --port 8000
```

Client base URL: `http://127.0.0.1:8000/v1`; model name: `qwen3-local`. See the [serving guide](docs/serving.md) for installation, streaming, authentication and deployment. `doctor` checks files and dependency versions without loading the model; GPU capacity and weight identity are not certified by this check.

Use Python 3.12 and the pinned Linux/WSL vLLM 0.23.0 GPU environment. Install into the existing runtime:

```bash
python -m pip install -e .
inference-api serve
```

The API listens on `http://127.0.0.1:8000` and owns its backend on port 8001. The default model is pinned `Qwen/Qwen3-4B`; set `INFERENCE_MODEL` to an existing local model directory to avoid downloading weights. Ctrl+C stops both owned services.

```bash
curl http://127.0.0.1:8000/v1/document/qa \
  -H 'Content-Type: application/json' \
  -d '{"document":"Project Aurora is owned by Mei.","question":"Who owns Project Aurora?"}'
```

The default command starts mixed attention directly. To run the BF16 comparison explicitly, use `inference-api serve --profile document`. Set `stream:true` for SSE. For an existing compatible backend, use `inference-api api --backend-url http://127.0.0.1:8001`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Backend readiness; 503 when unavailable |
| `GET /v1/models` | Backend model list |
| `POST /v1/chat/completions` | OpenAI-compatible chat proxy, including SSE |
| `POST /v1/document/qa` | Stable document prefix followed by the changing question |

See [API usage](docs/api.md) and [installation](docs/installation.md). Interactive API documentation is at `/docs`.

## Repository layout

```text
src/inference_service/
  api.py                 # Chat/document HTTP contracts and SSE
  prompts.py             # Reusable document prefix
  config.py, runtime.py  # Mixed default, comparison profiles and backend lifecycle
  backends/              # Full and chunked mixed-attention implementations
tests/                   # CPU-only product regression tests
benchmarks/evaluation/   # Final frozen evaluation (new inputs, three arms)
benchmarks/ruler/        # Historical first-round RULER tools
benchmarks/latency/      # Historical optimization experiments
docs/                    # Scope, usage, architecture and attention evidence
.local/research/         # Local mixed-attention experiments and raw evidence
.local/archives/         # Historical research and source snapshots (ignored)
```

Other research, training code, old builds and removed service features are archived locally. Version 0.1.2 removes the adaptive/CPU-KV profiles, `/v1/solve` and `--batch-invariant`. See [archive index and migration](docs/archive.md). Archive files and experiment results are not runtime dependencies.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m pip wheel --no-deps --no-build-isolation . -w dist
```

Run the full lifecycle regression suite on Linux/WSL. It needs no GPU or model download; the API-only package needs no Torch/vLLM. GPU performance and numerical evidence are separate from CPU regression checks. Runtime extensions remain pinned to vLLM 0.23.0.

Version 0.1.3 integrates the validated source-branching implementation and makes mixed attention the service default. See the [release and evidence index](docs/release-0.1.3.md) and [final evaluation protocol](docs/final-benchmark-protocol.md). Historical results identify their measured version; they are not silently reassigned to this release.

The expanded evaluation is complete: 2,340 quality responses across 13 tasks, three lengths and 20 new samples per task/length, plus 54 fixed-output measurements. Mixed scored 92.868 versus BF16 Flash's 93.103 (0.252% relative decrease), with 1.94× the token capacity under the same 4 GiB KV budget. At 16K, mixed scored 1.026% lower relative to BF16; quality is not uniformly lossless. Observed 16K request time was 39.4% lower than native FP8 and 19.6% higher than BF16, with cross-session timing limitations. See the [full results and limitations](docs/final-benchmark-results.md), [machine-readable results](benchmarks/results/final-2026-09-19/summary.json), and [preserved interruption history](docs/final-benchmark-status.md).
