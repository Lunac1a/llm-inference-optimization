#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/stage1-common.sh"
source "$stage1_script_dir/benchmark-point.sh"
model_path="$(stage1_resolve_model_path)"
BENCH_NUM_WARMUPS=0
BENCH_NUM_PROMPTS="$5"
BENCH_OUTPUT_LEN="$2"
run_benchmark_point "$1" "$3" "$4" "round1-input$1-concurrency$3"
