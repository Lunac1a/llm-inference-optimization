"""Static and artifact integrity checks for the Stage 3 collector."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

from stage3_lib import OUT, ROOT, load_config


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def check_sources() -> None:
    for name in ("stage3_lib.py", "stage3_prepare.py", "stage3_collect.py", "stage3_analyze.py"):
        path = ROOT / "scripts" / name
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            fail(f"syntax error in {name}: {exc}")


def check_config() -> None:
    config = load_config()
    expected = {"A", "B", "C", "D", "E", "F"}
    if set(config["candidates"]) != expected:
        fail("candidate set is not A-F")
    for label, candidate in config["candidates"].items():
        if candidate["attention_backend"] not in {"FLASH_ATTN", "TRITON_ATTN"}:
            fail(f"unexpected backend for {label}")
        if candidate["kv_cache_dtype"] not in {"bfloat16", "fp8_per_token_head"}:
            fail(f"unexpected KV dtype for {label}")


def check_materials() -> None:
    path = OUT / "materials.json"
    if not path.exists():
        print("materials: not prepared yet")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data.get("quality", [])) != 48:
        fail("quality material count is not 48")
    if set(data.get("documents", {})) != {"8192", "16384"}:
        fail("document lengths are not 8192/16384 targets")
    for target, docs in data["documents"].items():
        if len(docs) != 4:
            fail(f"target {target} does not have four documents")


def check_artifacts() -> None:
    if (OUT / "acceptance.json").exists():
        acceptance = json.loads((OUT / "acceptance.json").read_text(encoding="utf-8"))
        if not acceptance.get("passed"):
            fail("acceptance.json is present but not passed")
    print("artifact checks: ok")


def main() -> None:
    check_sources()
    check_config()
    check_materials()
    check_artifacts()
    print("Stage 3 verifier: passed")


if __name__ == "__main__":
    main()
