"""Local paths and immutable protocol helpers; never imported by the service."""
import hashlib
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / '.local/research/ruler-tradeoff-2026-09-16'
UPSTREAM = ROOT / 'upstream'
MODEL = '/home/lunacia/.cache/huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c'
ARMS = ['bf16_flash', 'bf16_triton', 'fp8', 'mixed']
LENGTHS = [4096, 8192, 16384]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]

def append(path, obj):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(obj, ensure_ascii=False) + '\n')
