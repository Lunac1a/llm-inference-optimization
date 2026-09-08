#!/usr/bin/env bash
# Shared fixed-length collector; Stage 1 defaults and acceptance stay unchanged.
run_benchmark_point() {
    local input_len="$1" concurrency="$2" run_dir="$3" result_name="$4"
    local -a bench_command=("$stage1_vllm" bench serve)
    if [[ "${STAGE2_BENCH:-0}" == 1 ]]; then
        bench_command=("$stage1_python" "$stage1_script_dir/stage2-bench-entry.py")
    fi
    "${bench_command[@]}" \
                --backend openai \
                --base-url "$stage1_base_url" \
                --endpoint /v1/completions \
                --model "$model_path" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --dataset-name random \
                --random-input-len "$input_len" \
                --random-output-len "$BENCH_OUTPUT_LEN" \
                --random-range-ratio 0 \
                --random-prefix-len 0 \
                --num-warmups "$BENCH_NUM_WARMUPS" \
                --num-prompts "$BENCH_NUM_PROMPTS" \
                --request-rate inf \
                --max-concurrency "$concurrency" \
                --ignore-eos \
                --temperature 0 \
                --seed "$BENCH_SEED" \
                --percentile-metrics ttft,tpot,e2el \
                --metric-percentiles 50,95 \
                --save-result \
                --save-detailed \
                --result-dir "$run_dir" \
                --result-filename "$result_name.json" \
                > "$run_dir/$result_name.log" 2>&1
}
