#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/stage1-common.sh"

mkdir -p "$stage1_repo_root/artifacts/stage1" "$stage1_state_dir"
printf 'Downloading or resolving %s at revision %s\n' "$MODEL_ID" "$MODEL_REVISION"
model_path="$(stage1_resolve_model_path)"
printf 'model_snapshot=%s\n' "$model_path"

"$stage1_python" - "$model_path" "$MODEL_ID" "$MODEL_REVISION" \
    > "$stage1_repo_root/artifacts/stage1/model-lock.json" <<'PY'
import hashlib
import json
import pathlib
import sys

model_path = pathlib.Path(sys.argv[1])
model_id = sys.argv[2]
revision = sys.argv[3]
files = []
for path in sorted(p for p in model_path.rglob("*") if p.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    files.append({
        "path": path.relative_to(model_path).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })
print(json.dumps({
    "model_id": model_id,
    "revision": revision,
    "snapshot_path": str(model_path),
    "files": files,
}, indent=2))
PY
cp "$stage1_repo_root/artifacts/stage1/model-lock.json" \
    "$stage1_repo_root/artifacts/stage1/model-lock.latest.json"
printf 'Wrote %s\n' "$stage1_repo_root/artifacts/stage1/model-lock.json"
