# Experimental BF16 prefill / FP8 cache

This opt-in prototype improves cold response time while retaining three cached
16K document clients on the tested machine. It is separate from the unchanged
default `scripts/local_document_qa.py` service. See
[validation and tradeoffs](hybrid-prefill-validation.md) before using its results.

The reusable pieces are:

- `experiments/hybrid_prefill/hybrid_backend.py`: fixed-version vLLM backend plugin;
  original BF16 full-prefill attention, upstream FP8 writes and cached attention.
- `scripts/validate_hybrid_prefill.py`: `config('hybrid')`, `HybridServer`,
  `launch_environment` and `prompt`; these exact components launched and queried
  both real-model hybrid runs, then cleaned up their owned processes.
- `scripts/hybrid_vllm_entry.py`: the experiment's actual vLLM CLI entry. It records
  final arguments and selects full prefill, preserving the old shared launcher.

Use only the existing WSL Ubuntu-24.04 environment and pinned vLLM 0.23.0. The
plugin is discovered through its local dist-info entry point when the launch
environment explicitly sets `VLLM_PLUGINS=inference_hybrid_prefill` and adds
`experiments/hybrid_prefill` to PYTHONPATH. Native control processes use an empty
plugin selection. Do not install it globally or change the validated default.

The tested API remains `http://127.0.0.1:8000/v1/chat/completions` with model alias
`stage3-document-qa`. Both startup/call/cleanup and mixed cold/cached routing were
exercised by the collector. Use `prompt({'text': document}, question)` for the
tested document layout; send the document every time, keeping its title and text
stable. Distinct document titles at the beginning prevent sharing the initial
instruction block. An existing prefix hit follows native FP8 attention; it does
not get the original-BF16 cold fast path. Prefix hashes do not encode conversation
state and callers must still send the correct document.

The context limit is **16,640 tokens including output**, full prefill is required,
weights remain BF16 and persistent KV is per-token-head FP8. The new full-prefill
settings leave less memory for KV than the old chunked configuration. Only the
three retained synthetic 16K documents and small fact checks are validated.

The collector protects committed evidence: `prepare` refuses an existing output
directory and `run` refuses an existing GPU budget. For a separately authorized
reproduction, use a separate workspace and preregister a new output directory
in the collector source before `prepare` and `run`; its numerical subprocess
must resolve the same directory. Merely changing an imported module variable
does not propagate to that subprocess. Use a fresh 30-minute budget; never delete
or overwrite artifacts/hybrid-prefill/ or invoke the historical environment
recovery against completed results. This is an experimental module interface,
not a newly accepted production CLI profile.
