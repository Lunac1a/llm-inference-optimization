"""Frozen native CPU KV offload comparison; no source/kernel modifications."""
import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from local_document_qa import configuration, port_open
from stage3_lib import ROOT, OwnedVllm, write_json, gpu_sample, host_sample, metrics_snapshot, parse_prometheus, append_jsonl
from validate_hybrid_prefill import prompt
from validate_mixed_load import request, labelled_delta, PERF, QUALITY
from validate_fp8_capacity import hashes

OUT = ROOT / 'artifacts/kv-offload'
URL = 'http://127.0.0.1:8000'


class Server(OwnedVllm):
    def _paths(self):
        python, _, headers = super()._paths()
        return python, str(ROOT / 'scripts/kv_offload_entry.py'), headers


def memory():
    d = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'MemTotal', 'SwapTotal', 'SwapFree'):
            d[key] = int(value.split()[0]) * 1024
    d['swap_used'] = d['SwapTotal'] - d['SwapFree']
    return d


@contextmanager
def launch(mode, folder):
    values = {'KV_OFFLOAD_MODE': mode, 'KV_OFFLOAD_LAUNCH': str(folder / 'actual-launch.json'),
              'VLLM_PLUGINS': '', 'VLLM_USE_SIMPLE_KV_OFFLOAD': '0'}
    before = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def prepare():
    if OUT.exists():
        raise RuntimeError('Refuse existing evidence')
    import vllm
    from transformers import AutoTokenizer
    from vllm.utils.platform_utils import is_pin_memory_available
    assert vllm.__version__ == '0.23.0'
    mem = memory()
    assert mem['MemAvailable'] >= 12 * 2**30, mem
    data = json.loads((ROOT / 'artifacts/hybrid-prefill/materials.json').read_text())
    model = Server(configuration(), OUT / 'unused').resolve_model()
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokens = {k: [tok.apply_chat_template([{'role': 'user', 'content': prompt(d, q)}], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False) for q in (PERF, QUALITY)]
              for k, d in data['documents'].items()}
    def common(a, b):
        return sum(1 for _ in itertools.takewhile(lambda x: x[0] == x[1], zip(a, b)))
    cross = {a + ':' + b: common(tokens[a][0], tokens[b][0]) for a, b in itertools.combinations(tokens, 2)}
    same = {k: common(t[0], t[1]) for k, t in tokens.items()}
    assert max(cross.values()) < 16 and min(same.values()) > .99 * 16384
    assert all(len(t) + 128 <= 32768 for ts in tokens.values() for t in ts)
    old = hashes(p for p in (ROOT / 'artifacts').rglob('*') if p.is_file())
    frozen = hashes([p for folder in ('configs', 'experiments') for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in str(p)] +
                    [p for p in (ROOT / 'scripts').glob('*.py') if 'kv_offload' not in p.name])
    package = Path(vllm.__file__).parent
    sources = ['config/cache.py', 'config/vllm.py', 'engine/arg_utils.py', 'v1/core/sched/scheduler.py',
               'v1/kv_offload/cpu/spec.py', 'v1/kv_offload/cpu/manager.py', 'v1/kv_offload/cpu/gpu_worker.py',
               'distributed/kv_transfer/kv_connector/v1/offloading_connector.py',
               'distributed/kv_transfer/kv_connector/v1/offloading/metrics.py']
    OUT.mkdir()
    write_json(OUT / 'old-evidence-hashes.json', old)
    write_json(OUT / 'frozen-source-hashes.json', frozen)
    write_json(OUT / 'materials.json', data)
    write_json(OUT / 'preflight.json', {'vllm': vllm.__version__, 'model': model, 'memory': mem,
        'pin_memory_available': is_pin_memory_available(), 'config': configuration(),
        'tokens': {k: list(map(len, ts)) for k, ts in tokens.items()}, 'cross_prefix': cross, 'same_prefix': same,
        'package_hashes': {str(package / s): hashlib.sha256((package / s).read_bytes()).hexdigest() for s in sources},
        'sources': {s: (package / s).read_text() for s in sources}})
    print('CPU preflight passed', flush=True)


def copy_probe():
    import torch
    from vllm import _custom_ops as ops
    from vllm.utils.platform_utils import is_pin_memory_available
    pin = is_pin_memory_available()
    size = 4 * 2**20
    src = (torch.arange(size, device='cuda', dtype=torch.int32) % 251).to(torch.uint8)
    cpu = torch.zeros(size, dtype=torch.uint8, pin_memory=pin)
    dst = torch.zeros_like(src)
    torch.cuda.synchronize()
    times = []
    for a, b, any_order in ((src, cpu, False), (cpu, dst, True)):
        pointers = [torch.tensor([n], dtype=torch.int64, pin_memory=pin) for n in (a.data_ptr(), b.data_ptr(), size)]
        start = time.perf_counter()
        ops.swap_blocks_batch(*pointers, is_src_access_order_any=any_order)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    equal = torch.equal(src, dst)
    write_json(OUT / 'copy-probe.json', {'size_bytes': size, 'pin_memory': pin,
        'wall_seconds_by_direction': times, 'equal': equal, 'purpose': 'compatibility only, not model performance'})
    assert equal
    print('Roundtrip passed', flush=True)


def snapshot():
    s = metrics_snapshot(URL)
    assert s['status'] == 200
    s['selected'] = [r for r in parse_prometheus(s['raw']) if any(t in r['name'] for t in
        ('cache', 'offload', 'request_queue', 'preempt', 'num_requests', 'request_prefill'))]
    return s


def owned_rss(group):
    values = {}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(stat[2]) != group:
                continue
            match = re.search(r'^VmRSS:\s+(\d+) kB', (p / 'status').read_text(), re.M)
            if match:
                values[p.name] = int(match[1]) * 1024
        except (OSError, ValueError):
            pass
    return {'per_pid_bytes': values, 'sum_rss_bytes': sum(values.values())}


class Monitor:
    def __init__(self, folder, group):
        self.folder, self.group = folder, group
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
    def run(self):
        while not self.stop_event.is_set():
            row = {'time': time.time(), 'memory': memory(), 'rss': owned_rss(self.group), 'gpu': gpu_sample()}
            try:
                row['metrics'] = snapshot()
            except Exception as exc:
                row['error'] = str(exc)
            append_jsonl(self.folder / 'telemetry.jsonl', row)
            self.stop_event.wait(1)
    def start(self): self.thread.start()
    def stop(self):
        self.stop_event.set()
        self.thread.join(15)


def controls(delta, mode, phase, prompt_tokens):
    def count(name, direction=None):
        return sum(r['delta'] for r in delta if r['name'] == name and
                   (direction is None or r['labels'].get('transfer_type', '').lower() == direction))
    local = count('vllm:prefix_cache_hits_total')
    external = count('vllm:external_prefix_cache_hits_total')
    restored = count('vllm:kv_offload_total_bytes_total', 'cpu_to_gpu')
    # prometheus_client preserves an existing _total suffix in the middle.
    preempt = count('vllm:num_preemptions_total')
    ok = preempt == 0
    if phase == 'cold':
        ok = ok and local + external < prompt_tokens * .1
    elif mode == 'baseline':
        ok = ok and local < prompt_tokens * .1 and external == 0 and restored == 0
    else:
        ok = ok and local + external >= prompt_tokens * .9 and external > 0 and restored > 0
    return {'pass': ok, 'local_hit_tokens': local, 'external_hit_tokens': external,
            'restored_bytes': restored, 'preemptions': preempt}


async def run():
    import aiohttp
    if (OUT / 'budget.json').exists():
        raise RuntimeError('Refuse rerun')
    assert not port_open(configuration())
    pre = json.loads((OUT / 'preflight.json').read_text())
    docs = json.loads((OUT / 'materials.json').read_text())['documents']
    started = time.time()
    write_json(OUT / 'budget.json', {'start': started, 'deadline': started + 1800, 'minutes': 30})
    write_json(OUT / 'protocol.json', {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sources': hashes([Path(__file__), ROOT / 'scripts/kv_offload_entry.py', ROOT / 'docs/kv-offload-plan.md'])})
    def ensure():
        assert time.time() + 300 < started + 1800, 'Budget exhausted'
        mem = memory()
        assert mem['MemAvailable'] >= 2**30, mem
        assert mem['swap_used'] - pre['memory']['swap_used'] <= 256 * 2**20, mem
    rows = []
    capacities = []
    try:
        ensure()
        p = subprocess.run([sys.executable, __file__, 'copy-probe'], capture_output=True, text=True, timeout=60)
        (OUT / 'copy-probe.log').write_text(p.stdout + p.stderr)
        assert p.returncode == 0, 'Native copy probe failed; no model run'
        async with aiohttp.ClientSession() as session:
            for rep, order in enumerate((('offload', 'baseline'), ('baseline', 'offload'), ('offload', 'baseline')), 1):
                for mode in order:
                    ensure()
                    folder = OUT / f'r{rep}-{mode}'
                    folder.mkdir()
                    server = Server(configuration(), folder / 'server')
                    monitor = None
                    try:
                        print('START ' + folder.name, flush=True)
                        with launch(mode, folder):
                            server.start(timeout=180)
                        log = (folder / 'server/server.log').read_text(errors='replace')
                        capacity = int(re.search(r'GPU KV cache size: ([\d,]+) tokens', log)[1].replace(',', ''))
                        assert 16600 < capacity < 3 * 16446, capacity
                        capacities.append(capacity)
                        assert len(set(capacities)) == 1, capacities
                        if mode == 'offload':
                            assert 'OffloadingConnector' in log
                        write_json(folder / 'capacity.json', {'gpu_tokens': capacity, 'cpu_bytes': 8 * 2**30 if mode == 'offload' else 0})
                        monitor = Monitor(folder, server.process.pid)
                        monitor.start()
                        for i in range(2):
                            await request(session, {'text': '短文。天气晴朗。'}, '写一句话。', 'warm-' + str(i), folder, 16)
                            await asyncio.sleep(1)
                        run_row = {'round': rep, 'mode': mode, 'requests': []}
                        for phase in ('cold', 'revisit'):
                            for key, doc in docs.items():
                                ensure()
                                before = snapshot()
                                r = await request(session, doc, PERF, phase + '-' + key, folder)
                                await asyncio.sleep(1)
                                after = snapshot()
                                delta = labelled_delta(before, after)
                                checked = controls(delta, mode, phase, r['usage']['prompt_tokens'])
                                item = {'phase': phase, 'document': key,
                                    'request': {k: r.get(k) for k in ('ttft', 'e2e', 'output_tok_s', 'start', 'end', 'token_count', 'usage')},
                                    'metrics_before': before, 'metrics_after': after, 'metric_deltas': delta, 'controls': checked}
                                run_row['requests'].append(item)
                                write_json(folder / 'measurement.json', run_row)
                                print(json.dumps({'run': folder.name, 'phase': phase, 'doc': key,
                                                  'ttft': r['ttft'], **checked}), flush=True)
                                assert checked['pass'], checked
                        if rep == 1:
                            for key, doc in docs.items():
                                ensure()
                                await request(session, doc, QUALITY, 'quality-' + key, folder, fixed=False)
                                await asyncio.sleep(1)
                        rows.append(run_row)
                        write_json(OUT / 'measurements.json', rows)
                    finally:
                        if monitor:
                            monitor.stop()
                        server.stop()
                    assert not port_open(configuration())
    except BaseException as exc:
        write_json(OUT / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        write_json(OUT / 'cleanup.json', {'elapsed_seconds': time.time() - started,
            'port_closed': not port_open(configuration()), 'gpu': gpu_sample(), 'memory': memory(),
            'compute_processes': subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv'],
                                               capture_output=True, text=True).stdout})


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'copy-probe', 'run'])
    phase = parser.parse_args().phase
    {'prepare': prepare, 'copy-probe': copy_probe, 'run': lambda: asyncio.run(run())}[phase]()
