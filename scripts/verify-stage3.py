"""Static and artifact integrity checks for the Stage 3 collector."""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
from pathlib import Path
import sys

from stage3_lib import OUT, ROOT, load_config


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def check_sources() -> None:
    for name in ("stage3_lib.py", "stage3_prepare.py", "stage3_collect.py", "stage3_analyze.py", "stage3_request.py"):
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
    if data.get("schema") != "stage3-materials-v2":
        fail("wrong materials version")
    if len(data.get("quality", [])) != 48:
        fail("quality material count is not 48")
    if set(data.get("documents", {})) != {"8192", "16384"}:
        fail("document lengths are not 8192/16384 targets")
    for target, docs in data["documents"].items():
        if len(docs) != 4:
            fail(f"target {target} does not have four documents")
    from stage3_collect import score_quality_answer
    performance_hashes = {doc['sha256'] for docs in data['documents'].values() for doc in docs.values()}
    for question in data['quality']:
        if question['document_sha256'] in performance_hashes:
            fail("quality material overlaps performance material")
        answer = ';'.join(group[0] for group in question['answer_groups'])
        if not score_quality_answer(question, {'status': 200, 'stream_complete': True,
                                               'finish_reason': 'stop', 'text': answer})['correct']:
            fail("invalid answer oracle")


def check_artifacts() -> None:
    manifest_path = ROOT/'artifacts/stage3/evidence-sha256.json'
    if manifest_path.exists():
        files = json.loads(manifest_path.read_text(encoding='utf-8'))['files']
        for name, digest in files.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                fail(f"archived evidence changed: {name}")
        print(f"archived Stage 3 hashes: {len(files)} verified")
    if (OUT / "acceptance.json").exists():
        acceptance = json.loads((OUT / "acceptance.json").read_text(encoding="utf-8"))
        if not acceptance.get("passed"):
            fail("acceptance.json is present but not passed")
    print("artifact checks: ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-path')
    args = parser.parse_args()
    check_sources()
    check_config()
    check_materials()
    check_artifacts()
    if args.tokenizer_path:
        from stage3_prepare import import_tokenizer, ids
        from stage3_collect import formal_prompts
        tokenizer = import_tokenizer(args.tokenizer_path)
        data = json.loads((OUT/'materials.json').read_text(encoding='utf-8'))
        for target, docs in data['documents'].items():
            first, followups = formal_prompts(data, int(target), True)
            def encode(prompt):
                return tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=True,
                                                     add_generation_prompt=True, enable_thinking=False)
            for index, (_, prompt) in enumerate(followups):
                doc = docs[sorted(docs)[index % 4]]
                if len(ids(tokenizer, doc['text'])) != int(target):
                    fail('document token length mismatch')
                a, b = encode(first[index % 4][1]), encode(prompt)
                common = next((i for i, (x,y) in enumerate(zip(a,b)) if x != y), min(len(a),len(b)))
                if common // 16 < int(target) // 16:
                    fail('first visit does not prime the complete shared document prefix')
        for question in data['quality']:
            if len(ids(tokenizer, question['document'])) != 16384:
                fail('quality document token length mismatch')
        print('Pinned tokenizer: exact document lengths and all 32 shared-prefix pairs verified')
    print("Stage 3 offline verifier: passed; this does not certify GPU acceptance")


if __name__ == "__main__":
    main()
