# Research archive and scope cleanup

2026-09-16: active code now contains only the default BF16 baseline and full/chunked mixed-attention paths. Historical research is preserved locally, outside source packaging and CI.

## Local archive

Root: `.local/archives/2026-09-16-research-scope/` (Git-ignored).

| Directory | Contents |
|---|---|
| `product-before-cleanup/` | Pre-cleanup source, tests, documentation, packaging and environment example, including uncommitted changes |
| `training/` | Complete budget-training exploration, data, model files, results and dependencies |
| `legacy-research/` | Earlier shared research tree: stages, cache/CPU-KV, budgets, Graph experiments, scripts, configurations, locks and original manifests |
| `build/`, `dist/`, `generated-egg-info/` | Old generated packages and build metadata, isolated from new builds |

`manifest.json` maps original to archived file paths and records sizes. SHA-256 is recorded for files up to 16 MiB outside dependency/bytecode directories; larger files and dependency bundles have size inventory only. `verification.json` records post-move verification. Original evidence was moved without rewriting its contents. This inventory does not imply a full SHA-256 audit of every large artifact.

The older shared tree includes baseline and initial hybrid/mixed-load evidence because its collectors and manifests span multiple research directions. It is archived intact to preserve provenance. Current attention attribution and optimization directories remain at their original locations under `.local/research/`; an index there links earlier attention and baseline reports in the archive.

Archives are local, not included in a fresh clone. Historical absolute paths and commands describe their original workspace. To reproduce archived tooling, restore its recorded layout in a separate checkout; do not rerun frozen collectors over completed evidence. Existing older archives and model/runtime installations remain intact.

## Removed interfaces in 0.1.2

- `adaptive` and `cpu-kv` profiles; CLI rejects these names.
- `/v1/solve`; requests now receive 404.
- `--batch-invariant` and `INFERENCE_BATCH_INVARIANT`; remove the former from commands and the latter from environment configuration.
- Budget policy code/resources and the CPU-KV plugin entry point.

Use `inference-api serve` for the unchanged default baseline, or `--profile chunked-hybrid` for research. Update clients to standard chat/document requests. Archived behavior requires its archived package/source; it is not available through a compatibility alias.

## Cleanup validation

- 7,269 moved files passed size verification; 4,062 eligible files additionally passed SHA-256 verification.
- All three retained profiles have exactly matching launch arguments, environment settings and document prompts before/after cleanup.
- Seven attention/prompt/license files match the pre-cleanup source byte-for-byte.
- 17 source regression tests and the same 17 tests against an installed wheel passed on Linux/WSL.
- The 0.1.2 wheel's 16 source/data files match the working tree. Removed modules/resources/plugin entries are absent.
- Clean API-only installation passed without Torch or vLLM. No GPU experiment or real-model acceptance was rerun for this cleanup.

Machine-readable evidence: `cleanup-validation.json` and `wheel-validation.json` in the archive root. This validates the scope cleanup; numerical/performance claims still refer to the separate historical attention experiments.
