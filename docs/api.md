# API usage

`inference-api serve --profile adaptive` starts the API and its owned GPU backend. `inference-api api --profile adaptive --backend-url http://127.0.0.1:8001` starts only the API against an existing compatible backend. For API-only operation, ensure the external backend enables Qwen3 reasoning parsing and native thinking-budget support; declaring a profile does not reconfigure that external process.

## Automatic budget selection

```bash
curl -i http://127.0.0.1:8000/v1/solve \
  -H 'Content-Type: application/json' \
  -d '{"question":"Record values: A=42, B=17. What is the value for record A?","budget_profile":"economy","stream":false}'
```

| Question pattern | economy | quality |
|---|---:|---:|
| Record lookup |0|0|
| Arithmetic |512|512|
| Sequential transformations |512|1024|
| Subset counting |512|1024|
| Unrecognized pattern |512|512|

These are thinking-token limits, not guaranteed token consumption. The output allowance is budget+192. The endpoint adds an integer-answer instruction and bounded integer grammar, with temperature0 and seed42. Unknown questions still require an integer answer; the fallback is not a general-chat mode. Larger budgets can regress individual answers. Quality/economy are cost choices rather than quality guarantees.

The selected budget is returned in response headers, while the body remains a standard chat completion. Streaming uses the same headers and forwards the upstream SSE, including usage events. No extra model call is used to select the budget. Invalid profiles return422; calling solve outside the adaptive runtime profile returns409.

## Chat and document QA

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-local","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'

curl http://127.0.0.1:8000/v1/document/qa \
  -H 'Content-Type: application/json' \
  -d '{"document":"Project Aurora. The owner is Ada.","question":"Who owns Project Aurora?","max_tokens":128}'
```

Generic chat forwards supported vLLM request fields; it does not silently apply the integer budget policy. Document QA places unchanged document bytes before the question. Send the complete document on every request; there is no persistent document upload or session store. Prefix reuse depends on the backend profile and available cache; adaptive mode disables prefix caching to match its measured configuration.

## Configuration and errors

The settings listed in `.env.example` are read from environment variables. Export them explicitly; a dotenv loader is not included. `--profile`, `--host`, `--port`, `--backend-url` and `--batch-invariant` override their corresponding settings. API and backend ports must differ for the combined `serve` command. Existing listeners are never replaced.

Set `INFERENCE_API_KEY` to require `Authorization: Bearer ...` on inference, model and health endpoints. `INFERENCE_BACKEND_API_KEY` is a separate credential used for the upstream backend; client credentials are not forwarded to it. The default bind address is loopback.

Backend failures before streaming return502, timeouts return504, and upstream HTTP error statuses/bodies are preserved. A midstream transport failure emits an error event; it does not manufacture a successful completion. Disconnecting a client closes its upstream response. [HTTPX streaming lifecycle](https://www.python-httpx.org/async/) and [FastAPI responses](https://fastapi.tiangolo.com/advanced/custom-response/) define the underlying transport behavior.
