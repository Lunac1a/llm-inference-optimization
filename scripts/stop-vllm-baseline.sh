#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/stage1-common.sh"

if [[ ! -f "$stage1_pid_file" ]]; then
    printf 'No Stage 1 PID file; no project process was stopped.\n'
    exit 0
fi

server_pid="$(<"$stage1_pid_file")"
if ! stage1_pid_is_ours "$server_pid"; then
    printf 'PID file does not identify a live project vLLM process; leaving it untouched.\n' >&2
    exit 2
fi

kill -TERM "$server_pid"
for _ in $(seq 1 30); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        rm -f -- "$stage1_pid_file"
        printf 'Stopped project vLLM process %s.\n' "$server_pid"
        exit 0
    fi
    sleep 1
done

if stage1_pid_is_ours "$server_pid"; then
    kill -KILL "$server_pid"
fi
rm -f -- "$stage1_pid_file"
printf 'Stopped project vLLM process %s after graceful shutdown timeout.\n' "$server_pid"
