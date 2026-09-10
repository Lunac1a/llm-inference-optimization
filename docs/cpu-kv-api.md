# Experimental CPU KV cache API profile

This profile failed full acceptance (cold-request cost and host-swap gates).
Use it only as an explicit experimental option; it is not the recommended default.
It integrates the existing vLLM OffloadingConnector with the scoped WSL
single-copy compatibility adapter. Consult [integration validation](kv-offload-integration-validation.md)
for acceptance status and measured limits. The ordinary BF16 service remains the
default. Both profiles use BF16 weights and BF16 KV; this is not the FP8 hybrid.

In the project's pinned Ubuntu-24.04 WSL environment:

```bash
cd /mnt/d/Lunacia/Inference
/home/lunacia/.venvs/inference-vllm/bin/python scripts/local_document_qa.py serve --cpu-kv-cache
```

The optional profile requires at least 12 GiB WSL MemAvailable before startup and
allocates an 8 GiB CPU KV pool. It requires prefix caching; combining the flag
with `--no-prefix-cache` is rejected. It retains the baseline model revision,
32,768-token context limit, GPU memory budget and localhost port. Port conflicts
are rejected without replacing another service. Ctrl+C stops owned processes.

Applications use the same OpenAI-compatible endpoint and model alias:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"stage3-document-qa","messages":[{"role":"user","content":"你好"}],"max_tokens":128,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}'
```

Set `stream` to `true` for SSE. The existing `local_document_qa.py ask` command
also works, with `--nonstream` selecting JSON responses. Send the complete prompt
on each call; prefix matching and GPU/CPU transfers are internal to vLLM. There
is no file-upload endpoint, persistent session store or new API gateway.

Use this mode for repeated long-prefix queries when GPU eviction would otherwise
cause recomputation. It consumes host memory and transfer bandwidth even though
model weights remain on GPU. CPU cache is volatile and lost on restart. It does
not establish support for a single active context larger than GPU capacity.
Only bounded serial document reuse is the acceptance target; concurrent coding
agents, project-file changes and combination with FP8 require separate validation.

Logs in `.tmp/document-qa/<timestamp>/` include the actual offload flags, plugin
environment, startup and shutdown. The plugin is enabled only in the owned
offload process; installed vLLM is unchanged, pinning is not forced, and there is
no silent fallback if the profile fails. To return to the default, stop the
service and run the same `serve` command without `--cpu-kv-cache`.

Evidence uses independent-title prompts from the hybrid materials to prevent
cross-document prefix sharing. The general `ask` helper has a shared instruction
prefix, so its timings need not match that benchmark exactly. Reproduction needs
fresh evidence paths and a fixed budget; never overwrite archived runs.
