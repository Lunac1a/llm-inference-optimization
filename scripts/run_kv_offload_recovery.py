"""Fresh offload comparison using a validated scoped native-copy adapter."""
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from stage3_lib import ROOT, write_json
from validate_fp8_capacity import hashes
import validate_kv_offload as original

OUT = ROOT / 'artifacts/kv-offload-recovery'
PLUGIN = ROOT / 'experiments/kv_offload_compat'
original.OUT = OUT
original.__file__ = __file__  # Its guarded probe subprocess must use this fresh wrapper.
base_launch = original.launch


class Server(original.Server):
    def _paths(self):
        python, _, headers = super()._paths()
        return python, str(ROOT / 'scripts/kv_offload_recovery_entry.py'), headers


@contextmanager
def launch(mode, folder):
    old = os.environ.get('PYTHONPATH')
    with base_launch(mode, folder):
        os.environ['PYTHONPATH'] = str(PLUGIN)
        os.environ['VLLM_PLUGINS'] = 'inference_kv_transport' if mode == 'offload' else ''
        try:
            yield
        finally:
            if old is None: os.environ.pop('PYTHONPATH', None)
            else: os.environ['PYTHONPATH'] = old


original.Server = Server
original.launch = launch


def probe():
    import torch
    sys.path.insert(0, str(PLUGIN))
    from transport import make_copy
    results = []
    # Two layouts: multiple small tensors and one cross-layer-sized block tensor.
    for count, page in ((2, 65536), (1, 2359296)):
        cpu = [torch.arange(4 * page, dtype=torch.int32).remainder(251).to(torch.uint8).view(4, page) for _ in range(count)]
        gpu = [torch.zeros_like(t, device='cuda') for t in cpu]
        returned = [torch.zeros_like(t) for t in cpu]
        order = [2, 0, 3, 1]
        for direction, srcs, dsts in [('h2d', cpu, gpu), ('d2h', gpu, returned)]:
            src, dst, sizes = [], [], []
            for a, b in zip(srcs, dsts):
                for i, j in enumerate(order):
                    src.append(a.data_ptr() + j * page)
                    dst.append(b.data_ptr() + i * page)
                    sizes.append(page)
            copy = make_copy(srcs, dsts)
            copy(*[torch.tensor(v, dtype=torch.int64) for v in (src, dst, sizes)], is_src_access_order_any=direction == 'h2d')
            torch.cuda.synchronize()
            for a, b in zip(srcs, dsts):
                assert torch.equal(a.cpu()[order], b.cpu())
            results.append({'count': count, 'page_bytes': page, 'direction': direction, 'equal': True, 'pinned': False})
    write_json(OUT / 'copy-probe.json', {'layouts': results, 'pass': True, 'transport': 'native_single_adapter'})


def run():
    if (OUT / 'execution.json').exists(): raise RuntimeError('Refuse overwrite')
    diagnosis = json.loads((ROOT / 'artifacts/kv-offload-compat/results.json').read_text())
    assert all(any(r['op'] == 'single' and r['direction'] == d and r['pass'] for r in diagnosis) for d in ('h2d', 'd2h'))
    write_json(OUT / 'execution.json', {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sources': hashes([Path(__file__), ROOT / 'scripts/validate_kv_offload.py', ROOT / 'scripts/kv_offload_recovery_entry.py',
                          ROOT / 'docs/kv-offload-plan.md', ROOT / 'docs/kv-offload-compat-plan.md'] +
                         [p for p in PLUGIN.rglob('*') if p.is_file() and '__pycache__' not in str(p)]),
        'transport': 'scoped upstream single-copy adapter', 'performance_gates': 'unchanged'})
    asyncio.run(original.run())


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    {'prepare': original.prepare, 'copy-probe': probe, 'run': run}[sys.argv[1]]()
