#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/stage1-common.sh
evidence=artifacts/stage1/acceptance-v2
sha256sum configs/stage1-baseline.env locks/requirements-wsl-cu130.txt docs/stage1-protocol-v2.md scripts/*.py scripts/*.sh > "$evidence/source-sha256.txt"
python3 -m unittest discover -s tests -v > "$evidence/regression.log" 2>&1
model_path="$(stage1_resolve_model_path)"
for batch in 1 2 3; do
    echo "Preparation-$batch"
    "$stage1_vllm" bench serve --backend openai --base-url "$stage1_base_url" \
        --endpoint /v1/completions --model "$model_path" --served-model-name "$SERVED_MODEL_NAME" \
        --dataset-name random --random-input-len 512 --random-output-len 128 \
        --random-range-ratio 0 --random-prefix-len 0 --num-warmups 4 --num-prompts 32 \
        --request-rate inf --max-concurrency 4 --ignore-eos --temperature 0 --seed 42 \
        > "$evidence/preparation-$batch.log" 2>&1
done
bash scripts/run-stage1-benchmark.sh
