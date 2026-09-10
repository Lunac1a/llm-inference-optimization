"""Bounded independently isolated transfer compatibility matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from stage3_lib import ROOT, write_json, gpu_sample
from validate_fp8_capacity import hashes

OUT = ROOT / 'artifacts/kv-offload-compat'
CASES = [(op, pin, direction) for op, pins in [('batch', (False, True)), ('single', (False,)), ('torch', (False,))]
         for pin in pins for direction in ('d2h', 'h2d')] + [('batch', False, 'd2d')]


def probe(op, pin, direction):
    import torch
    from vllm import _custom_ops as ops
    size = 4 * 2**20
    row = {'op': op, 'pin': pin, 'direction': direction, 'bytes': size, 'stage': 'allocation'}
    print(json.dumps(row), flush=True)
    try:
        expected = (torch.arange(size, dtype=torch.int32) % 251).to(torch.uint8)
        cpu = torch.empty(size, dtype=torch.uint8, pin_memory=pin)
        cpu.copy_(expected)
        src = expected.cuda() if direction in ('d2h', 'd2d') else cpu
        dst = torch.zeros_like(src) if direction == 'd2d' else (
            torch.zeros(size, device='cuda', dtype=torch.uint8) if direction == 'h2d' else
            torch.empty(size, dtype=torch.uint8, pin_memory=pin))
        torch.cuda.synchronize()
        host = dst if direction == 'd2h' else cpu
        row.update({'stage': 'submission', 'payload_host_is_pinned': host.is_pinned()})
        print(json.dumps(row), flush=True)
        start = time.perf_counter()
        if op == 'batch':
            args = [torch.tensor([v], dtype=torch.int64) for v in (src.data_ptr(), dst.data_ptr(), size)]
            ops.swap_blocks_batch(*args, is_src_access_order_any=direction == 'h2d')
        elif op == 'single':
            ops.swap_blocks(src, dst, size, torch.tensor([[0, 0]], dtype=torch.int64))
        else:
            dst.copy_(src, non_blocking=True)
        row['stage'] = 'synchronize'
        torch.cuda.synchronize()
        row.update({'wall_seconds': time.perf_counter() - start, 'equal': torch.equal(expected, dst.cpu()), 'stage': 'finished'})
        assert row['equal']
        row['pass'] = True
    except Exception as exc:
        row.update({'pass': False, 'error': f'{type(exc).__name__}: {exc}'})
    print('RESULT ' + json.dumps(row), flush=True)


def run():
    if OUT.exists(): raise RuntimeError('Refuse overwrite')
    old = hashes(p for p in (ROOT / 'artifacts').rglob('*') if p.is_file())
    OUT.mkdir()
    write_json(OUT / 'old-evidence-hashes.json', old)
    start = time.time()
    write_json(OUT / 'protocol.json', {'start': start, 'deadline': start + 600,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sources': hashes([Path(__file__), ROOT / 'docs/kv-offload-compat-plan.md'])})
    results = []
    try:
        for op, pin, direction in CASES:
            assert time.time() + 60 < start + 600
            label = f'{op}-{int(pin)}-{direction}'
            p = subprocess.run([sys.executable, __file__, 'probe', op, str(int(pin)), direction],
                               capture_output=True, text=True, timeout=30)
            (OUT / (label + '.log')).write_text(p.stdout + p.stderr)
            lines = [l[7:] for l in p.stdout.splitlines() if l.startswith('RESULT ')]
            row = json.loads(lines[-1]) if lines else {'op': op, 'pin': pin, 'direction': direction,
                                                     'pass': False, 'error': 'missing_result', 'returncode': p.returncode}
            results.append(row)
            write_json(OUT / 'results.json', results)
            print(json.dumps(row), flush=True)
    finally:
        write_json(OUT / 'cleanup.json', {'elapsed': time.time() - start, 'gpu': gpu_sample(),
            'compute_processes': subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv'], capture_output=True, text=True).stdout})


if __name__ == '__main__':
    if sys.argv[1] == 'run': run()
    else: probe(sys.argv[2], bool(int(sys.argv[3])), sys.argv[4])
