"""Post-run verification and source snapshot; does not launch GPU workloads."""
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / '.local/research/mixed-split-2026-09-18'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads((ROOT / 'frozen.json').read_text())
    assert all(digest(WORKSPACE / p) == sha for p, sha in frozen.items()), 'Frozen file changed'
    micro = json.loads((ROOT / 'micro.json').read_text())
    assert len(micro['numerics']) == 20 and all(r['passed'] for r in micro['numerics'])
    assert all(r['inputs_unchanged'] and r['default_equals_original'] for r in micro['numerics'])
    assert len(micro['benchmarks']) == 3
    for row in micro['benchmarks']:
        assert all(len(samples) == 20 for samples in row['samples_ms'].values())
    decision = json.loads((ROOT / 'decision.json').read_text())
    if decision['stopped_at'] == 'micro_gate':
        assert not micro['gate']['passed'] and not decision['accepted']
        assert not any((ROOT / p).exists() for p in ['A1', 'B1', 'B2', 'A2', 'A-quality', 'B-quality'])
    ledger = [json.loads(line) for line in (ROOT / 'ledger.jsonl').read_text().splitlines()]
    assert all(r['reason'] == 'completed' and r['code'] == 0 for r in ledger)
    seconds = sum(r['seconds'] for r in ledger)
    assert seconds <= 2400
    prior = WORKSPACE / '.local/research/mixed-latency-2026-09-18'
    previous = sum(json.loads(line)['seconds'] for line in (prior / 'ledger.jsonl').read_text().splitlines())
    assert previous + seconds <= 7200
    original_hashes = json.loads((prior / 'frozen.json').read_text())
    originals = {p: sha for p, sha in original_hashes.items() if p.startswith('src/inference_service/')}
    assert all(digest(ROOT / 'baseline-source' / p.removeprefix('src/')) == sha for p, sha in originals.items())
    snapshot = ROOT / 'candidate-source'
    if not snapshot.exists():
        shutil.copytree(WORKSPACE / 'src/inference_service', snapshot / 'inference_service',
                        ignore=shutil.ignore_patterns('__pycache__'))
    current = {p: sha for p, sha in frozen.items() if p.startswith('src/inference_service/')}
    assert all(digest(snapshot / p.removeprefix('src/')) == sha for p, sha in current.items())
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 8033)) != 0
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv'], text=True)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,temperature.gpu,utilization.gpu', '--format=csv'], text=True)
    record = {'frozen_files_verified': len(frozen), 'numerical_cases': 20, 'paired_samples_per_shape': 20,
        'old_snapshot_matches_diagnosis': True, 'candidate_snapshot_verified': True,
        'decision': decision, 'this_gpu_seconds': seconds, 'stage_gpu_seconds': previous + seconds,
        'port_8033_released': True, 'compute_processes_after': processes, 'gpu_after': gpu,
        'verifier_sha256': digest(Path(__file__))}
    (ROOT / 'validation.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
