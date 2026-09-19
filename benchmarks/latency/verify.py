"""Post-run integrity checks; never launches a model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.local/research/mixed-latency-2026-09-18'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads((OUT / 'frozen.json').read_text())
    assert all(digest(ROOT / name) == sha for name, sha in frozen.items())
    summary = json.loads((OUT / 'analysis.json').read_text())
    assert set(summary) == {'bf16_triton', 'mixed', 'fp8', 'bf16_flash'}
    for arm, value in summary.items():
        rows = [json.loads(s) for s in (OUT / arm / 'results.jsonl').read_text().splitlines()]
        assert len(rows) == 13
        assert sum(r['phase'] == 'measure' for r in rows) == 9
        assert sum(r['phase'] == 'warmup' for r in rows) == 3
        assert sum(r['phase'] == 'profile' for r in rows) == 1
        assert all(r['done'] and r['usage']['completion_tokens'] == 1 and r['hits'] == 0
                   and r['preemptions'] == 0 and r['ttft'] > 0 for r in rows)
        assert len(list((OUT / arm / 'responses').glob('*.jsonl'))) == 13
        for path in (OUT / arm / 'responses').glob('*.jsonl'):
            assert json.loads(path.read_text().splitlines()[-1])['data'] == '[DONE]'
        profile = value['profile']
        assert abs(sum(v['ms'] for v in profile['categories'].values()) - profile['kernel_sum_ms']) < 1e-6
        assert profile['categories']['attention']['calls'] == 288
        assert len(profile['chunks']) == 8
        assert profile['gpu_busy_union_ms'] <= profile['gpu_envelope_ms']
    assert summary['mixed']['kv_capacity_tokens'] == summary['fp8']['kv_capacity_tokens'] == 56480
    assert summary['bf16_flash']['kv_capacity_tokens'] == summary['bf16_triton']['kv_capacity_tokens'] == 29120
    ledger = [json.loads(s) for s in (OUT / 'ledger.jsonl').read_text().splitlines()]
    assert len(ledger) == 4 and all(r['reason'] == 'completed' and r['code'] == 0 for r in ledger)
    assert sum(r['seconds'] for r in ledger) < 1800
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 8032)) != 0, 'Experiment port still occupied'
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv'], text=True)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv'], text=True)
    spec = importlib.util.find_spec('vllm')
    package = Path(spec.origin).parent
    paths = [package / 'v1/attention/ops/triton_unified_attention.py',
             package / 'model_executor/models/qwen3.py']
    evidence = {str(p): {'sha256': digest(p)} for p in paths}
    for p in paths:
        (OUT / ('installed-' + p.name)).write_bytes(p.read_bytes())
    validation = {'frozen_files_verified': len(frozen), 'requests_verified': 52,
        'traces_verified': 4, 'gpu_lifecycle_seconds': sum(r['seconds'] for r in ledger),
        'port_8032_released': True, 'compute_processes_after': processes, 'gpu_after': gpu,
        'installed_sources_postrun': evidence,
        'postprocessing_hashes': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}}
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == '__main__': main()
