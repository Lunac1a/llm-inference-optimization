# Stage 1: KV-cache bottleneck validation

Status: **complete — determinate no-go** on 2026-09-01
(Australia/Sydney).

The preregistered target performance bottleneck was **not confirmed** for
single-sequence decode on the locked Qwen3-8B Q4_K_M / RTX 4090 Laptop setup.
KV-cache capacity is material and scales exactly as expected, but replacing F16
with the smaller upstream q8_0 control reduced decode throughput rather than
improving it. Stage 2 did not start and `q8_kv_128` remains unimplemented.

## Boundary and locked inputs

- Parent repository: `https://github.com/Lunac1a/llm-inference-optimization`.
- Parent branch at start: `main`, tracking `origin/main`, clean worktree.
- Parent starting commit: `3b0671c20b8669d5e5229445cbeae794f094b22b`.
- Submodule gitlink, actual HEAD and `locks/upstream.json` all matched
  `f8dbcd61893702976f9ab03be89c2b9f436d532c` (build 10720).
- Model SHA-256 was remeasured before the formal run and matched
  `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- F16/F16 was the primary baseline; upstream q8_0/q8_0 was the control.
- Full GPU offload, forced Flash Attention, engine, model, batch/ubatch,
  threads, output length and hardware were held fixed.

The decision thresholds were frozen in `docs/stage1-protocol.md` before formal
measurement.

## What is theory, measurement and inference

### Theoretical estimate

Qwen3-8B has 36 layers, 8 KV heads, and 128 K/V values per head. Therefore:

```text
F16  = 2 * 36 * 8 * 128 * 2 bytes     = 144 KiB/token
q8_0 = 2 * 36 * 8 * 128 * 34/32 bytes = 76.5 KiB/token
```

The predicted q8_0 reduction is 46.875%. This is an upstream-format estimate,
not a custom-project result.

### Actual measurement

The verbose 16,384-cell end-to-end runs reported 2,304 MiB for F16 and
1,224 MiB for q8_0, exactly matching the estimate. All six runs also reported
build `f8dbcd618`, 37/37 GPU-offloaded layers, enabled Flash Attention, the
intended K/V types, the same 9,622-token prompt and the same sampling controls.

The official `llama-bench` microbenchmark measured 128-token decode at four
resident depths. It excludes tokenization and sampling, which isolates model
execution. Five repetitions were used per condition.

| Resident depth | F16 median t/s | q8_0 median t/s | q8_0 vs F16 | Max CV | Canonical source |
| ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 86.404 | 84.074 | -2.70% | 1.09% | initial run |
| 4,096 | 81.227 | 78.772 | -3.02% | 0.33% | protocol rerun |
| 16,384 | 67.218 | 65.092 | -3.16% | 2.80% | initial run |
| 32,768 | 46.190 | 44.405 | -3.86% | 1.66% | protocol rerun |

The fixed-Prompt end-to-end corroboration used SHA-256
`7f13b774f1854fea4be27ab181b93b0fa0b97c33092b7cc1fd88c02889727b3d`,
seed 42, temperature 0, top-k 1, top-p 1, min-p 0, repeat penalty 1,
16,384 context cells and 128 generated tokens:

| KV type | KV allocation | Median decode t/s | CV |
| --- | ---: | ---: | ---: |
| F16 | 2,304 MiB | 70.48 | 0.52% |
| q8_0 | 1,224 MiB | 69.53 | 0.51% |

q8_0 was 1.35% slower in this end-to-end check. Greedy output text was not
identical across KV types because KV quantization changes model numerics; output
quality was observed but was not evaluated in Stage 1.

### Inference

Long resident context is expensive: canonical F16 decode fell 46.54% from
depth 512 to 32,768. However, the controlled compression intervention did not
recover performance. The likely interpretation is that q8_0 dequantization and
other decode work outweigh any saved KV traffic on this workload. That mechanism
is an inference, not a directly measured kernel attribution.

Nsight Compute connected to the process but returned `ERR_NVGPUCTRPERM`, so no
kernel-level DRAM bytes or throughput are claimed. The raw permission result is
`artifacts/stage1/ncu-permission-probe.log`.

## Quality-gate handling

The initial 4,096-token conditions and 32,768-token F16 condition exceeded the
preregistered 5% coefficient-of-variation limit while GPU temperature reached
89 C. No sample was deleted. The complete initial data remains under
`artifacts/stage1/bench/`.

Following the preregistered rule, only the affected depths were rerun after the
GPU cooled to 60-63 C. All rerun CVs were 0.27-1.66%, so the determinate-outcome
quality gate passed. Canonical source selection is machine-readable in
`locks/stage1.json` and `artifacts/stage1/canonical-bench-summary.csv`.

## Stop-condition evaluation

| Preregistered condition | Result | Status |
| --- | ---: | --- |
| F16 depth-32,768 throughput is at least 10% below depth 512 | -46.54% | pass |
| q8_0 is at least 5% faster at depth 16,384 or 32,768 | -3.16% / -3.86% | fail |
| `gain(32768) - gain(512)` is at least +5 percentage points | -1.17 pp | fail |

The data-quality gate passed but the performance conditions did not. Under the
preregistered decision rule, the result is **target performance bottleneck not
confirmed**. This satisfies the Stage 1 determinate no-go stop condition.

## Reproduction and raw evidence

```powershell
# Build the official benchmark from the pinned, unmodified submodule.
cmd /d /c "call <VS-BuildTools>\VsDevCmd.bat -arch=x64 -host_arch=x64 && cmake --build engine\llama.cpp\build-baseline --target llama-bench --parallel 8"

# Formal microbenchmark and the protocol-required rerun.
.\scripts\run-stage1-bench.ps1
.\scripts\run-stage1-bench.ps1 -Depths 4096,32768 -Repetitions 5 -RunSet bench-rerun

# Fixed-Prompt, fixed-sampling corroboration with full runtime evidence.
.\scripts\run-stage1-e2e.ps1 -RunSet e2e-verbose

# Recompute summaries and artifact checksums.
.\scripts\analyze-stage1-bench.ps1
.\scripts\analyze-stage1-bench.ps1 -RunSet bench-rerun
.\scripts\write-stage1-checksums.ps1
```

Evidence map:

- `artifacts/stage1/bench/`: all 40 initial JSON results, stderr logs and GPU telemetry.
- `artifacts/stage1/bench-rerun/`: all 20 protocol-rerun results and telemetry.
- `artifacts/stage1/e2e-verbose/`: fixed prompt, manifests, output, verbose runtime logs and telemetry.
- `artifacts/stage1/e2e/`: retained compact-output pre-validation run; not used as canonical evidence.
- `artifacts/stage1/canonical-bench-summary.csv`: canonical bottleneck table.
- `artifacts/stage1/e2e-summary.csv`: end-to-end summary.
- `artifacts/stage1/SHA256SUMS.txt`: integrity manifest for Stage 1 artifacts.

## Final boundary

Stage 1 is complete with a no-go result. No custom cache code was added, no
`q8_kv_128` functionality exists, and Stage 2 must remain not started.
