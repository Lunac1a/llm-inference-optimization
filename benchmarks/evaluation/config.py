"""Paths and fixed parameters for the final, non-tuning evaluation."""
import hashlib
import json
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('INFERENCE_EVAL_ROOT', WORKSPACE / '.local/research/ruler-final-2026-09-19'))
PREVIOUS = WORKSPACE / '.local/research/ruler-tradeoff-2026-09-16'
UPSTREAM = Path(os.environ.get('RULER_SOURCE', PREVIOUS / 'upstream'))
DEPS = Path(os.environ.get('RULER_DATA_DEPS', PREVIOUS / 'deps'))
NLTK = Path(os.environ.get('NLTK_DATA', PREVIOUS / 'nltk_data'))
MODEL = os.environ.get('INFERENCE_MODEL', '/home/lunacia/.cache/huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c')
ARMS = ['bf16_flash', 'fp8', 'mixed']
LENGTHS = [4096, 8192, 16384]
N = 20
BUDGET_SECONDS = 10800
PORT = 8035
UPSTREAM_COMMIT = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x]
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def prompt_hash(row):
    return hashlib.sha256(json.dumps(row['prompt_ids'], separators=(',', ':')).encode()).hexdigest()


def verify_frozen():
    frozen = read(ROOT / 'frozen.json')
    for name, sha in frozen.items():
        assert digest(name) == sha, f'Frozen file changed: {name}'

