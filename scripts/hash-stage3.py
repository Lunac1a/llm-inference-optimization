"""Hash every retained Stage 3 artifact except the manifest itself."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stage3_lib import OUT, ROOT, write_json


def main() -> None:
    manifest = OUT / "evidence-sha256.json"
    rows = {}
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows[str(path.relative_to(ROOT)).replace("\\", "/")] = digest
    write_json(manifest, {"schema": "stage3-sha256-v1", "files": rows})
    print(json.dumps({"files": len(rows), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
