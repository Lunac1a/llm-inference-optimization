# Run the mixed KV service

This release serves Qwen3-4B on the validated Linux/WSL2 NVIDIA environment. Mixed KV is the default. It does not manage a model catalog, support arbitrary models or provide a desktop UI.

## Install and start

Use Python 3.12, vLLM 0.23.0 and Transformers 4.57.6. The tested runtime uses PyTorch 2.11.0+cu130 on Ubuntu 24.04/WSL2. Install into that environment:

```bash
python -m pip install -e .
# For an existing uv environment without pip:
# uv pip install --python /path/to/environment/bin/python -e .
```

See [installation](installation.md) to prepare a new GPU environment. A matching NVIDIA driver, CUDA runtime and Python development headers are required; this package does not install system drivers.

Download the original Qwen3-4B snapshot using Hugging Face tools, keeping its configuration, tokenizer and all safetensors shards together. The evaluated revision is `1cfa9a7208912126459214e8b04321603b3df60c`. Then:

```bash
inference-api doctor --model /path/to/Qwen3-4B
inference-api serve --model /path/to/Qwen3-4B --port 8000
```

Alternatively, omit `--model` to let vLLM resolve/download the pinned `Qwen/Qwen3-4B` revision. Local snapshots are checked for architecture and required files, not for cryptographic identity of the weights. Weight-quantized and structurally different models are outside this release's support scope.

The API starts after the backend is ready. Configure clients with **base URL `http://127.0.0.1:8000/v1`** and **model `qwen3-local`**. If a client requires an API key field for an unauthenticated local service, a placeholder value is sufficient. Ctrl+C or SIGTERM stops the API and its owned backend. Logs are in `.local/logs/`; existing processes are never replaced to free a port.

## Call the API

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-local","messages":[{"role":"user","content":"Explain KV caching briefly."}],"stream":true,"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}'
```

Set `stream` to `false` for a JSON response. The document QA endpoint and full request examples are in [API documentation](api.md). API schema is available at `/docs`.

## Server deployment

Run the same command on the server. For direct network access, set an API key before binding beyond localhost:

```bash
export INFERENCE_API_KEY='replace-with-a-long-random-secret'
inference-api serve --model /srv/models/Qwen3-4B --host 0.0.0.0 --port 8000
```

Clients send `Authorization: Bearer <secret>`. For Internet access, place the API behind your HTTPS reverse proxy; keep the backend port 8001 on loopback. `/health` also requires the API key. Do not use multiple API worker processes: this command owns a single model instance. A process supervisor can restart the whole command; health returns 503 if the backend later becomes unavailable.

## Small set of controls

| Control | Default | Meaning |
|---|---|---|
| `--model` / `INFERENCE_MODEL` | pinned Qwen3-4B | local directory or supported repository |
| `--host`, `--port` | 127.0.0.1:8000 | client-facing address |
| `--backend-url` | http://127.0.0.1:8001 | owned loopback backend address; use a different port from the API |
| `INFERENCE_API_KEY` | empty | client authentication; required by CLI for non-loopback binding |
| `INFERENCE_MAX_INFLIGHT` | 8 | active generation requests, including streams; excess requests get 429 |
| `INFERENCE_REQUEST_TIMEOUT` | 120 seconds | backend I/O inactivity timeout, not total generation duration |
| `INFERENCE_LOG_DIR` | .local/logs | backend logs |

The CLI reads environment variables; it does not implicitly read `.env` files. The default mixed configuration keeps the evaluated 4 GiB KV allocation, 2048-token chunks and 16640-token context. This is not automatic VRAM sizing. BF16 weights plus KV and runtime memory must fit on the device.

An unavailable backend returns 502, an I/O timeout returns 504, invalid authentication returns 401, and overload returns 429 with `Retry-After: 1`. If an SSE stream fails after headers were sent, it carries an error event rather than a fabricated completion marker. Disconnecting closes the upstream stream and releases its admission slot.

## Evidence

Version 0.1.4 changes the service wrapper, not the attention kernel. The [2340-response evaluation](final-benchmark-results.md) measured version 0.1.3; it has not been rerun or relabeled as a 0.1.4 benchmark. Frozen benchmark verification must use the evaluated source snapshot/check-out, since the current service wrapper has advanced. Separate 0.1.4 service acceptance is recorded in [delivery notes](release-0.1.4.md).
