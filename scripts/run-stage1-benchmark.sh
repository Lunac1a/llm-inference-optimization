#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/stage1-common.sh"

source "$stage1_script_dir/benchmark-point.sh"
if [[ "${1:-}" == "--point" ]]; then
    [[ "$#" == 7 ]] || exit 2
    model_path="$(stage1_resolve_model_path)"
    BENCH_NUM_PROMPTS="$6"
    BENCH_OUTPUT_LEN="$7"
    mkdir -p "$4"
    run_benchmark_point "$2" "$3" "$4" "$5"
    exit
fi

resume_dir=""
retry_of=""
only_groups=""
if [[ "${1:-}" == "--resume" ]]; then
    if [[ "$#" -ne 2 ]]; then
        printf 'Usage: %s --resume RUN_DIRECTORY\n' "$0" >&2
        exit 2
    fi
    resume_dir="$2"
elif [[ "${1:-}" == "--only" ]]; then
    if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
        printf 'Usage: %s --only INPUT:CONCURRENCY[,INPUT:CONCURRENCY...] [previous-run-directory]\n' "$0" >&2
        exit 2
    fi
    only_groups="$2"
    retry_of="${3:-}"
elif [[ "$#" -le 1 ]]; then
    retry_of="${1:-}"
else
    printf 'Usage: %s [previous-run-directory] | --resume RUN_DIRECTORY | --only GROUPS [previous-run-directory]\n' "$0" >&2
    exit 2
fi

if ! curl --silent --fail --max-time 5 "$stage1_base_url/health" >/dev/null; then
    printf 'vLLM health check failed at %s; start the service first.\n' "$stage1_base_url" >&2
    exit 3
fi
model_path="$(stage1_resolve_model_path)"

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$stage1_repo_root/artifacts/stage1/benchmarks/$run_id"
if [[ -n "$resume_dir" ]]; then
    if [[ "$resume_dir" = /* ]]; then
        run_dir="$resume_dir"
    else
        run_dir="$stage1_repo_root/$resume_dir"
    fi
    run_id="$(basename -- "$run_dir")"
    [[ -f "$run_dir/protocol.json" ]] || { printf 'Missing protocol.json in resume directory: %s\n' "$run_dir" >&2; exit 3; }
else
    mkdir -p "$run_dir"
    retry_metadata=""
    if [[ -n "$retry_of" ]]; then
        retry_metadata=",\"rerun_of\":\"$retry_of\",\"rerun_reason\":\"previous stability CV exceeded 5 percent\""
    fi
    printf '%s\n' "{\"run_id\":\"$run_id\",\"model_id\":\"$MODEL_ID\",\"model_revision\":\"$MODEL_REVISION\",\"model_snapshot\":\"$model_path\",\"served_model_name\":\"$SERVED_MODEL_NAME\",\"host\":\"$HOST\",\"port\":$PORT,\"dataset\":\"random\",\"request_rate\":\"inf\",\"warmups\":$BENCH_NUM_WARMUPS,\"measured_requests\":$BENCH_NUM_PROMPTS,\"rounds\":$BENCH_ROUNDS,\"seed\":$BENCH_SEED,\"input_lengths\":[512,2048],\"concurrency\":[1,4],\"output_length\":$BENCH_OUTPUT_LEN,\"only_groups\":\"$only_groups\"$retry_metadata}" > "$run_dir/protocol.json"
    cp "$stage1_log_file" "$run_dir/vllm-server-startup.log" || true
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$run_dir/gpu-info.csv"
fi

telemetry_file="$run_dir/gpu-telemetry.csv"
if [[ ! -f "$telemetry_file" ]]; then
    printf 'timestamp_utc,name,memory_used_mib,utilization_gpu_percent,temperature_c,power_draw_w\n' > "$telemetry_file"
fi
(
    while true; do
        timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw \
            --format=csv,noheader,nounits | sed "s/^/$timestamp,/" >> "$telemetry_file"
        sleep 1
    done
) &
telemetry_pid=$!
cleanup() {
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT

for round in $(seq 1 "$BENCH_ROUNDS"); do
    for input_len in 512 2048; do
        for concurrency in 1 4; do
            if [[ -n "$only_groups" && ",$only_groups," != *",$input_len:$concurrency,"* ]]; then
                continue
            fi
            result_name="round${round}-input${input_len}-concurrency${concurrency}"
            if [[ -s "$run_dir/$result_name.json" ]]; then
                printf 'Skipping completed %s\n' "$result_name"
                continue
            fi
            printf 'Running %s (warmups=%s measured=%s)\n' "$result_name" "$BENCH_NUM_WARMUPS" "$BENCH_NUM_PROMPTS"
            run_benchmark_point "$input_len" "$concurrency" "$run_dir" "$result_name"
        done
    done
done

cp "$run_dir/protocol.json" "$stage1_repo_root/artifacts/stage1/benchmarks/latest-protocol.json"
printf 'Benchmark complete: %s\n' "$run_dir"
