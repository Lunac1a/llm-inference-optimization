# Native GPU/CPU KV reuse: compatibility stop

Historical first attempt. A separately authorized
[compatibility recovery](kv-offload-recovery-validation.md) later enabled model
CPU restoration; it stopped on a different workload-control gate. The original
failure and evidence below remain unchanged in meaning.

**The current native transfer path did not pass its WSL compatibility check.**
No model server or cold/revisit request was started, so there is no measured
CPU-restoration speedup, slowdown, cache-capacity benefit or answer-quality result.
This is a runtime compatibility result, not rejection of KV tiering as an idea.

## Scope and observed failure

The [fixed plan](kv-offload-plan.md) and collector were committed as `eefddc4`
before GPU use. The plan selected existing vLLM 0.23.0 OffloadingConnector with an
8 GiB CPU KV pool, baseline BF16/FlashAttention configuration and three independent
16K documents. The intended serial A/B/C then A/B/C workload and three paired
rounds were never reached. No alternate connector, model, transport or GPU retry
was added after failure.

CPU source/config/token preflight and three offline collector tests passed. WSL
reported `pin_memory=False`. A single 4 MiB roundtrip probe called the same native
`swap_blocks_batch` operator used by the native offloading DMA path. It raised:

```text
RuntimeError: swap_blocks_batch, /workspace/csrc/libtorch_stable/cache_kernels.cu:145,
cuMemcpyBatchAsync failed at index 0 with error 1
```

The exact traceback is retained in [copy-probe.log](../artifacts/kv-offload/copy-probe.log).
The parent recorded `Native copy probe failed; no model run`. No successful
roundtrip/equality or transfer-time result exists. The probe loops over both
directions without intermediate direction logs; the traceback alone does not
establish which direction failed. Do not infer a directional bandwidth result.

## Established support versus unverified cause

Installed `config/vllm.py` maps `--kv-offloading-size` plus backend `native` to
OffloadingConnector when `VLLM_USE_SIMPLE_KV_OFFLOAD=0`. CPUOffloadingSpec provides
a CPU block pool and default LRU policy; its worker allocates buffers according to
`is_pin_memory_available()`. GPU->CPU uses the native copy operator; CPU->GPU
selects native DMA for sufficiently large pages, with a Triton alternative for
small pages. This is existing upstream capability, not a new algorithm.
The pinned [configuration reference](https://docs.vllm.ai/en/v0.23.0/api/vllm/config/vllm/)
and saved installed sources are the capability evidence; they do not establish
successful execution on this WSL system.

The combination of WSL's disabled pinned-memory path and the native batch-copy
failure is a plausible compatibility explanation. It is **not an isolated root
cause**: no pinned-memory comparison, alternate copy primitive, connector-level
model execution, driver change or native-Linux comparison was performed. A small
copy probe also does not reproduce every connector allocation/layout detail.
Accordingly the precise conclusion is that the selected compatibility gate failed,
not that all CPU KV offloading on Windows/WSL is impossible.

## Resume and next-step boundary

There is no basis to write a CPU-KV restoration speedup on the resume from this
attempt. The bounded plan, source audit, fail-fast check and preserved evidence are
engineering work, but the intended measured optimization remains unvalidated.

A separately authorized follow-up could investigate transfer compatibility or use
a different supported connector/environment. That would need a fresh evidence
directory and a new precommitted scope/budget. It must retain this failure and not
reinterpret it as a failed performance sample. No such follow-up was started here.

## Cleanup and preservation

Elapsed GPU-check budget through cleanup: **4.79 seconds** of 30 minutes. Port 8000
is closed, no GPU compute applications remain, swap usage is zero, and recorded
GPU memory is 823 MiB. No model service or 8 GiB CPU cache was allocated. The
probe subprocess exited with failure and was reaped. See
[cleanup.json](../artifacts/kv-offload/cleanup.json).

The final source/evidence checks and offline test logs are retained in
`artifacts/kv-offload/closure.json`; the new evidence has its own SHA-256 manifest.
All **1,908 old artifact files, 52 frozen source/config files and 9 installed
source files are unchanged**. All **13 offline tests pass** (3 offload controls,
4 mixed-load checks and 6 hybrid checks); these are not real-model acceptance.
The BF16 default, FP8 hybrid prototype and all historical raw artifacts are preserved.
