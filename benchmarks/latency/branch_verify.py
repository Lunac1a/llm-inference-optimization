"""Read-only experiment audit, apart from derived validation/trace summaries."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import analyze

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / '.local/research/mixed-branch-2026-09-18'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def rows(path): return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    frozen = read(ROOT / 'frozen.json')
    assert all(digest(WORKSPACE / path) == sha for path, sha in frozen.items())
    micro = read(ROOT / 'micro.json')
    assert len(micro['numerics']) == 20 and all(r['passed'] for r in micro['numerics'])
    assert len(micro['benchmarks']) == 3
    assert all(len(v) == 20 for r in micro['benchmarks'] for v in r['samples_ms'].values())
    decision = read(ROOT / 'decision.json')
    selection = read(ROOT / 'selection.json')
    historical = ROOT.parent / 'ruler-tradeoff-2026-09-16/runs'
    previously_evaluated = {r['id'] for path in historical.glob('*/results.jsonl') for r in rows(path)}
    assert not set(selection['quality']) & previously_evaluated, 'Quality input was already evaluated'
    ledger = rows(ROOT / 'ledger.jsonl')
    assert all(r['reason'] == 'completed' and r['code'] == 0 for r in ledger)
    total = sum(r['seconds'] for r in ledger)
    prior = sum(r['seconds'] for folder in ['mixed-latency-2026-09-18', 'mixed-split-2026-09-18']
                for r in rows(ROOT.parent / folder / 'ledger.jsonl'))
    assert total <= 2400 and total + prior <= 7200
    latency_count = 0; quality_count = 0; traces = {}
    analyze.ROOT = ROOT
    for job in ['A1', 'B1', 'B2', 'A2']:
        out = ROOT / job
        if not out.exists(): continue
        data = rows(out / 'results.jsonl')
        assert len(data) == 13
        assert all(r['done'] and r['usage']['completion_tokens'] == 1 and r['hits'] == 0
                   and r['preemptions'] == 0 and r['ttft'] > 0 for r in data)
        assert read(out / 'summary.json')['capacity_tokens'] == 56480
        paths = list((out / 'traces').glob('*.gz')); assert len(paths) == 1
        traces[job] = analyze.trace_summary(paths[0])
        assert traces[job]['categories']['attention']['calls'] == 288
        latency_count += len(data)
    for job in ['A-quality', 'B-quality']:
        out = ROOT / job
        if not out.exists(): continue
        data = rows(out / 'results.jsonl'); assert len(data) == 30
        assert len(rows(out / 'warmup.jsonl')) == 1
        assert all(r['cache_hit_delta'] == 0 and r['preemptions'] == 0 for r in data)
        quality_count += len(data)
    for out in (ROOT / job for job in ['A1', 'B1', 'B2', 'A2', 'A-quality', 'B-quality']):
        if not out.exists(): continue
        for path in (out / 'responses').glob('*.jsonl'):
            assert rows(path)[-1]['data'] == '[DONE]'
    if decision['accepted']:
        assert micro['gate']['passed'] and read(ROOT / 'latency-gate.json')['passed']
        assert read(ROOT / 'quality-gate.json')['passed']
        assert latency_count == 52 and quality_count == 60
    with socket.socket() as sock: assert sock.connect_ex(('127.0.0.1', 8034)) != 0
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv'], text=True)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,temperature.gpu,utilization.gpu', '--format=csv'], text=True)
    result = {'frozen_files_verified': len(frozen), 'numerical_cases': 20,
        'quality_inputs_disjoint_from_prior_ruler_runs': True,
        'latency_requests': latency_count, 'quality_requests': quality_count,
        'this_gpu_seconds': total, 'stage_gpu_seconds': prior + total,
        'decision': decision, 'port_8034_released': True, 'compute_processes_after': processes,
        'gpu_after': gpu, 'verifier_sha256': digest(Path(__file__))}
    (ROOT / 'trace-analysis.json').write_text(json.dumps(traces, indent=2))
    (ROOT / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
