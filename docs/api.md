# API usage

`inference-api serve` starts the mixed-attention API and its owned GPU backend (`chunked-hybrid`). Choose `--profile document` for the BF16 comparison or `--profile hybrid` for the earlier full-prefill comparison.

`inference-api api --backend-url http://127.0.0.1:8001` starts only the API. Ensure the external backend is configured for the intended comparison: selecting an API profile does not reconfigure that external process.

## Chat and document QA

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-local","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'

curl http://127.0.0.1:8000/v1/document/qa \
  -H 'Content-Type: application/json' \
  -d '{"document":"Project Aurora. The owner is Ada.","question":"Who owns Project Aurora?","max_tokens":128}'
```

Generic chat forwards backend request fields without server-side budget selection. Document QA places unchanged document bytes before the changing question, disables thinking and uses temperature 0. Send the complete document with each request; there is no persistent document/session store. Prefix reuse depends on matching tokens and available cache.

The document endpoint accepts `document`, `question`, `max_tokens` (1–2048, default 128), and `stream` (default false). Streaming includes upstream usage events. The earlier `hybrid` profile retains its historical document-title prefix; default and `chunked-hybrid` use the standard document prompt.

`GET /health` reports backend readiness, and `GET /v1/models` returns its model list. Responses include `X-Inference-Profile`. The OpenAPI schema is at `/openapi.json`, with interactive documentation at `/docs`.

## Configuration and errors

Export settings from `.env.example` explicitly; a dotenv loader is not included. CLI overrides are `--profile`, `--host`, `--port` and `--backend-url`. API and backend ports must differ for `serve`. Existing listeners are never replaced.

Set `INFERENCE_API_KEY` to require a Bearer token on inference, model and health endpoints. `INFERENCE_BACKEND_API_KEY` is a separate upstream credential; client credentials are not forwarded. The default bind address is loopback.

Backend connection failures before streaming return 502, timeouts return 504, and upstream HTTP error statuses/bodies are preserved. A midstream transport failure emits an error event without manufacturing a successful completion. Client disconnection closes its upstream response.

Version 0.1.2 removes `/v1/solve`, adaptive/CPU-KV profiles and the batch-invariance option. See [archive and migration](archive.md).
