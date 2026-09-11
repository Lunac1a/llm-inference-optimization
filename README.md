# Local Inference API

A local Qwen3/vLLM service with server-side reasoning-budget routing and opt-in KV-cache optimizations. Applications use HTTP; running the API does not require experiment scripts, benchmark datasets or result files.

## Quick start

Use Python 3.12 and the pinned Linux/WSL vLLM 0.23.0 environment for GPU execution. The API alone also runs without vLLM.

```bash
# In an environment that already contains the pinned GPU runtime:
python -m pip install -e .
inference-api serve --profile adaptive
```

The API listens on `http://127.0.0.1:8000`, with its owned vLLM backend on port8001. The default model is `Qwen/Qwen3-4B` at a pinned revision; set `INFERENCE_MODEL` to an existing local model directory to avoid downloading weights. Ctrl+C stops both owned services.

```bash
curl http://127.0.0.1:8000/v1/solve \
  -H 'Content-Type: application/json' \
  -d '{"question":"Calculate 23 * 17 - 61.","budget_profile":"economy"}'
```

The **server** selects the native thinking budget. Responses retain the OpenAI chat-completion shape; `X-Budget-Profile`, `X-Task-Type` and `X-Thinking-Budget` headers expose the decision. Set `stream:true` for SSE. Budget routing currently supports the bounded integer-answer contract, not arbitrary chat.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Backend readiness;503 when unavailable |
| `GET /v1/models` | Backend model list |
| `POST /v1/chat/completions` | OpenAI-compatible chat proxy, including SSE |
| `POST /v1/solve` | Automatic integer-task budget selection; adaptive profile |
| `POST /v1/document/qa` | Stable document prefix followed by the changing question |

Interactive API documentation is at `/docs`. See [API examples and configuration](docs/api.md).

## Optimizations

| Runtime profile | Included behavior | Status |
|---|---|---|
| `document` (default) | BF16 weights/KV, upstream prefix caching and reusable document layout | Default local document-QA route |
| `adaptive` | Server-side economy/quality budget policies; native Qwen3 thinking limits | Bounded integer-task optimization |
| `hybrid` | Original BF16 full cold attention with per-token-head FP8 KV storage/reuse | Opt-in prototype;16,640-token context |
| `cpu-kv` | Native8 GiB CPU KV offload plus scoped WSL transport compatibility | Opt-in; failed full performance/memory acceptance |

The two adaptive policies are `economy` (0/512) and `quality` (0/512/1024). Both save work relative to the corresponding uniform budget in the tested synthetic workload. A larger budget is not guaranteed to improve every answer. Optional `--batch-invariant` stabilizes the measured adaptive repeats but costs throughput; it is not enabled by default.

Profile switches use upstream vLLM features. The project code implements question-pattern budget routing, reusable document prompts, the BF16/FP8 attention-routing adapter, scoped CPU-transfer compatibility, and API/lifecycle integration. It does not claim new CUDA kernels or general task-quality guarantees. See [architecture and limits](docs/architecture.md).

## Repository layout

```text
src/inference_service/
  api.py                 # HTTP contracts, forwarding and streaming
  budget.py              # Server-side budget selection
  policies.json          # Frozen deployable mappings, no benchmark data
  prompts.py             # Reusable document prefix
  runtime.py             # Explicit profiles and owned backend lifecycle
  backends/              # Hybrid prefill and CPU KV adapters
tests/                   # CPU-only product regression tests
docs/                    # API and architecture documentation
```

Experiment scripts, generated responses, traces, reports and model weights stay local and are ignored. Already published Git history is retained; new local research commits are not ancestors of this product branch.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m pip wheel --no-deps --no-build-isolation . -w dist
```

No GPU/model download is needed for the regression suite. Runtime extensions remain pinned to vLLM0.23.0; API-only installation does not import or load them. See [installation](docs/installation.md) for the GPU environment.
