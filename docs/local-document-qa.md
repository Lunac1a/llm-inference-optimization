# Use the local document QA API

The service uses the existing OpenAI-compatible vLLM API with Qwen3-4B BF16,
FlashAttention, BF16 KV and prefix caching. It binds only to localhost. The
configuration is `configs/local-document-qa.json`; baseline scripts are unchanged.
For measured results and their limits, see [validation](prefix-cache-validation.md).

The [FP8 capacity comparison](fp8-capacity-validation.md) did not pass its full
acceptance rule: three warm documents benefited, but cold p95 TTFT was 24.54–24.86 s
against a fixed 20 s limit. No capacity-first FP8 option is offered; the BF16
default remains unchanged. Both experimental arms used Triton, so those numbers
are not a benchmark of this FlashAttention default. On this machine the WSL
distribution is named `Ubuntu-24.04` (`wsl -d Ubuntu-24.04` from Windows).

A later [experimental hybrid](hybrid-prefill.md) separates BF16 full cold
attention from FP8 cache storage and passes its own bounded comparison, reducing
cold TTFT by 48.5–55.8% versus a matched FP8 control. It needs full prefill and a
16,640-token context limit; it remains an opt-in prototype, not this default.

In the existing pinned WSL Ubuntu environment, terminal 1:

```bash
cd /mnt/d/Lunacia/Inference
/home/lunacia/.venvs/inference-vllm/bin/python scripts/local_document_qa.py serve
```

Wait for `Ready`. Terminal 2:

```bash
cd /mnt/d/Lunacia/Inference
/home/lunacia/.venvs/inference-vllm/bin/python scripts/local_document_qa.py ask \
  --document artifacts/prefix-cache/document-a.txt --question '项目负责人是谁？'
/home/lunacia/.venvs/inference-vllm/bin/python scripts/local_document_qa.py ask \
  --document artifacts/prefix-cache/document-a.txt --question '正式预算金额是多少？'
```

`ask` reads a UTF-8 text file and returns JSON containing answer `text`, API
`usage`, timing and raw response evidence. It consumes the streaming endpoint
by default; use `--nonstream` for a normal JSON response. A failed or truncated
response exits nonzero. Output is bounded to 128 tokens for this validated slice.
Use **Ctrl+C in terminal 1** to stop the owned service. It refuses to start if
port 8000 is already occupied. Server lifecycle logs live in `.tmp/document-qa/`.

## Calling from an application

Endpoint: `http://127.0.0.1:8000/v1/chat/completions`.
Model name: `stage3-document-qa` (a fixed local API alias).

The reusable `document_prompt(document, question)` function in
`scripts/local_document_qa.py` places constant instructions and the document
before the changing question. Use its exact prompt layout for the measured
behavior. The service is stateless from the caller's perspective: send the
document with every request; vLLM reuses matching cached token blocks internally.
There is no upload/document-ID endpoint or new gateway.

Keep document bytes and initial instructions unchanged between questions.
Do not put timestamps, request IDs, changing conversation history, or the
question before the document. To switch documents, pass another file; no manual
cache reset is required. Cache entries are reusable computation, not stored
conversation state. Restarting the service clears the in-memory cache.

The context limit is 32,768 tokens including the complete prompt and output.
Only sequential questions over the retained 16K document have been benchmarked.
The initial request still processes the document; the measured improvement
applies to subsequent questions while its prefix remains cached. Multi-user
capacity, eviction under pressure and general QA quality are not established.

## Reproducing the comparison

The frozen [plan](prefix-cache-plan.md) and `scripts/validate_prefix_cache.py`
define the comparison. The collector refuses to overwrite the retained
`artifacts/prefix-cache/` evidence. To reproduce, use a fresh checkout of the
pre-measurement implementation commit recorded in the validation report; do
not delete or overwrite the committed results in this checkout.
