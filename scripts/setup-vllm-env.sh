#!/usr/bin/env bash
# Run inside WSL/Linux. Requires Python 3.12 and uv on PATH.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_path="${INFERENCE_VENV:-$HOME/.venvs/inference-vllm}"
bash "$repo_root/scripts/setup-wsl-headers.sh"
if [[ ! -f "$env_path/pyvenv.cfg" ]]; then
    uv venv --python python3.12 "$env_path"
fi
uv pip sync --python "$env_path/bin/python" --torch-backend=cu130 \
    "$repo_root/locks/requirements-wsl-cu130.txt"
uv pip check --python "$env_path/bin/python"
"$env_path/bin/python" "$repo_root/scripts/check-vllm-environment.py"
