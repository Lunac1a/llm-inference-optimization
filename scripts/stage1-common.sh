#!/usr/bin/env bash
set -euo pipefail

stage1_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
stage1_repo_root="$(cd -- "$stage1_script_dir/.." && pwd)"
stage1_config_file="${STAGE1_CONFIG:-$stage1_repo_root/configs/stage1-baseline.env}"
if [[ ! -f "$stage1_config_file" ]]; then
    printf 'Missing Stage 1 config: %s\n' "$stage1_config_file" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$stage1_config_file"

stage1_venv="${INFERENCE_VENV:-$HOME/.venvs/inference-vllm}"
stage1_python="$stage1_venv/bin/python"
stage1_vllm="$stage1_venv/bin/vllm"
stage1_state_dir="${STAGE1_STATE_DIR:-$stage1_repo_root/.tmp/stage1}"
stage1_pid_file="$stage1_state_dir/vllm.pid"
stage1_log_file="$stage1_state_dir/vllm-server.log"
stage1_base_url="http://$HOST:$PORT"

if [[ ! -x "$stage1_python" || ! -x "$stage1_vllm" ]]; then
    printf 'Missing vLLM environment at %s\n' "$stage1_venv" >&2
    exit 2
fi

stage1_resolve_model_path() {
    "$stage1_python" - "$MODEL_ID" "$MODEL_REVISION" "$MODEL_CACHE_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, revision, cache_dir = sys.argv[1:]
print(snapshot_download(repo_id=model_id, revision=revision, cache_dir=cache_dir))
PY
}

stage1_pid_is_ours() {
    local pid="$1"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq -- "$stage1_vllm serve"
}

stage1_port_is_listening() {
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$PORT$"
}
