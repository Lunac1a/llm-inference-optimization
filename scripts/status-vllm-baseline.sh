#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/stage1-common.sh"

if [[ -f "$stage1_pid_file" ]]; then
    server_pid="$(<"$stage1_pid_file")"
    if stage1_pid_is_ours "$server_pid"; then
        printf 'process=running pid=%s\n' "$server_pid"
    else
        printf 'process=not-running pid-file=%s\n' "$stage1_pid_file"
    fi
else
    printf 'process=not-running\n'
fi

health_status="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
    --max-time 5 "$stage1_base_url/health" || true)"
printf 'health_http_status=%s\n' "$health_status"
models_status="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
    --max-time 5 "$stage1_base_url/v1/models" || true)"
printf 'models_http_status=%s expected_model=%s\n' "$models_status" "$SERVED_MODEL_NAME"
