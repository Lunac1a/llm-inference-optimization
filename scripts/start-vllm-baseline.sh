#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/stage1-common.sh"

mkdir -p "$stage1_state_dir"
if [[ -f "$stage1_pid_file" ]]; then
    existing_pid="$(<"$stage1_pid_file")"
    if stage1_pid_is_ours "$existing_pid"; then
        printf 'Stage 1 server already running with PID %s\n' "$existing_pid"
        exit 0
    fi
    rm -f -- "$stage1_pid_file"
fi

if stage1_port_is_listening; then
    printf 'Port %s is already in use; refusing to stop or replace an unrelated process.\n' "$PORT" >&2
    exit 3
fi

model_path="$(stage1_resolve_model_path)"
printf 'Starting %s from %s\n' "$SERVED_MODEL_NAME" "$model_path"
printf 'Server log: %s\n' "$stage1_log_file"

export VLLM_USE_V2_MODEL_RUNNER
export VLLM_USE_FLASHINFER_SAMPLER
[[ -f "$PYTHON_DEV_HEADERS/Python.h" ]] || { echo 'Run scripts/setup-wsl-headers.sh first.' >&2; exit 2; }
export C_INCLUDE_PATH="$PYTHON_DEV_HEADERS:$PYTHON_DEV_INCLUDE_ROOT${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"

nohup "$stage1_vllm" serve "$model_path" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --generation-config vllm \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    --enforce-eager \
    > "$stage1_log_file" 2>&1 < /dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$stage1_pid_file"
printf 'Started PID %s. Use scripts/status-vllm-baseline.sh to wait for readiness.\n' "$server_pid"
