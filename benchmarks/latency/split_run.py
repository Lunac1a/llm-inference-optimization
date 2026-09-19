"""Single-candidate gated experiment; no automatic retries or threshold changes."""
import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import psutil
import diagnose
from common import WORKSPACE, ROOT as RULER, MODEL, digest, read_jsonl, write_json, append
from collect import collect, make_spec
from inference_service.config import Settings
from inference_service.runtime import owned_backend

ROOT = WORKSPACE / '.local/research/mixed-split-2026-09-18'


def quality_rows():
    paths = sorted((RULER / 'data/16384').glob('*.jsonl'))
    paths += [RULER / f'data/{length}/cwe.jsonl' for length in [4096, 8192]]
    rows = [r for p in paths for r in read_jsonl(p) if r['sample'] in [5, 6]]
    assert len(rows) == 30
    return rows


def launch(settings, job, out):
    command, env = make_spec(settings, 'mixed', out)
    candidate = job.startswith('B')
    env['PYTHONPATH'] = str(WORKSPACE / 'src' if candidate else ROOT / 'baseline-source')
    env['INFERENCE_MIXED_SPLIT_SOURCES'] = '1' if candidate else '0'
    if not job.endswith('quality'):
        command += ['--profiler-config', json.dumps({'profiler': 'torch',
            'torch_profiler_dir': str(out / 'traces'), 'torch_profiler_with_stack': False,
            'torch_profiler_record_shapes': True, 'ignore_frontend': True})]
    write_json(out / 'launch.json', {'argv': command, 'env': {k: v for k, v in env.items()
        if k.startswith(('VLLM_', 'INFERENCE_', 'PYTHONPATH', 'C_INCLUDE_PATH'))}})
    return command, env


def gpu_state():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem', '--format=csv'], text=True)


def worker(job):
    out = ROOT / job; out.mkdir()
    for name in ['responses', 'metrics', 'traces']: (out / name).mkdir()
    settings = Settings(profile='chunked-hybrid', model=MODEL,
                        backend_url='http://127.0.0.1:8033', log_dir=out / 'engine')
    os.environ['PYTHON_DEV_HEADERS'] = str(WORKSPACE / '.tmp/python-dev/extracted/usr/include/python3.12')
    start = time.monotonic(); before = gpu_state()
    with patch('inference_service.runtime.launch_spec', lambda s: launch(s, job, out)):
        with owned_backend(settings):
            write_json(out / 'startup.json', {'seconds': time.monotonic() - start, 'gpu_before': before})
            if job.endswith('quality'):
                asyncio.run(collect(settings, out, quality_rows(), 'quality'))
            else:
                asyncio.run(diagnose.requests(settings, out))
    logs = '\n'.join(p.read_text(errors='replace') for p in (out / 'engine').glob('*.log'))
    capacity = re.findall(r'GPU KV cache size: ([\d,]+) tokens', logs)
    assert capacity and int(capacity[-1].replace(',', '')) == 56480
    assert 'route=dual_source' in logs and 'prefill_block_m=64' in logs
    if job.startswith('B'): assert 'split_sources=True' in logs
    if not job.endswith('quality'): assert len(list((out / 'traces').glob('*.gz'))) == 1
    write_json(out / 'summary.json', {'capacity_tokens': 56480, 'gpu_after': gpu_state(),
                                      'lifecycle_seconds': time.monotonic() - start})


def execute(job, limit):
    start = time.monotonic(); initial_swap = psutil.swap_memory().used
    assert psutil.virtual_memory().available >= 2 * 2**30
    tracked = {}; reason = 'completed'; code = None
    command = [sys.executable, str(Path(__file__).with_name('split_micro.py'))] if job == 'micro' else [sys.executable, __file__, '--job', job]
    env = dict(os.environ)
    headers = WORKSPACE / '.tmp/python-dev/extracted/usr/include'
    env['C_INCLUDE_PATH'] = os.pathsep.join([str(headers / 'python3.12'), str(headers), env.get('C_INCLUDE_PATH', '')])
    with (ROOT / f'{job}.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        try:
            while child.poll() is None:
                try:
                    for p in psutil.Process(child.pid).children(recursive=True): tracked[p.pid] = p.create_time()
                except psutil.NoSuchProcess: pass
                elapsed = time.monotonic() - start
                available = psutil.virtual_memory().available; swap = psutil.swap_memory().used
                append(ROOT / 'telemetry.jsonl', {'job': job, 'elapsed': elapsed, 'available': available, 'swap': swap})
                if available < 2 * 2**30 or swap - initial_swap > 256 * 2**20:
                    reason = 'memory_guard'; break
                if elapsed > limit - 40:
                    reason = 'time_guard'; break
                time.sleep(1)
            code = child.poll()
            if code not in (None, 0): reason = 'child_failed'
        finally:
            if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
            for pid, created in tracked.items():
                try:
                    p = psutil.Process(pid)
                    if p.create_time() == created: p.terminate()
                except psutil.NoSuchProcess: pass
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
            for pid, created in tracked.items():
                try:
                    p = psutil.Process(pid)
                    if p.create_time() == created: p.kill()
                except psutil.NoSuchProcess: pass
            append(ROOT / 'ledger.jsonl', {'job': job, 'seconds': time.monotonic() - start, 'reason': reason, 'code': code})
    print(json.dumps({'job': job, 'reason': reason, 'code': code}), flush=True)
    assert reason == 'completed' and code == 0, 'Stopped; no retries'


def latency_gate():
    records = {job: read_jsonl(ROOT / job / 'results.jsonl') for job in ['A1', 'B1', 'B2', 'A2']}
    medians = {job: {length: statistics.median(r['ttft'] for r in rows
        if r['phase'] == 'measure' and r['length'] == length) for length in [4096, 8192, 16384]}
        for job, rows in records.items()}
    ratios = [medians[b][16384] / medians[a][16384] for a, b in [('A1', 'B1'), ('A2', 'B2')]]
    scaling = {}
    for length in [4096, 8192]:
        combined = {arm: statistics.median(r['ttft'] for job, rows in records.items() if job.startswith(arm)
                    for r in rows if r['phase'] == 'measure' and r['length'] == length) for arm in ['A', 'B']}
        scaling[length] = combined['B'] / combined['A']
    result = {'medians': medians, 'ratios_16k': ratios, 'ratios_scaling': scaling,
              'passed': max(ratios) <= .97 and max(scaling.values()) <= 1.03}
    write_json(ROOT / 'latency-gate.json', result)
    return result


def quality_gate():
    from report import score
    inputs = {r['id']: r for r in quality_rows()}
    a = {r['id']: r for r in read_jsonl(ROOT / 'A-quality/results.jsonl')}
    b = {r['id']: r for r in read_jsonl(ROOT / 'B-quality/results.jsonl')}
    assert a.keys() == b.keys() == inputs.keys()
    rows = []
    for key, row in inputs.items():
        old, new = a[key], b[key]
        sa, sb = [score(r['prediction'], row['outputs'], row['category']) for r in [old, new]]
        rows.append({'id': key, 'original_score': sa, 'split_score': sb,
            'same_text': old['prediction'] == new['prediction'],
            'passed': sb >= sa and not (old['prediction'].strip() and not new['prediction'].strip())
            and not (old['finish_reason'] != 'length' and new['finish_reason'] == 'length')})
    result = {'rows': rows, 'passed': all(r['passed'] for r in rows)}
    write_json(ROOT / 'quality-gate.json', result)
    return result


def main():
    (ROOT / 'started.json').open('x').write(json.dumps({'utc': time.time()}))
    paths = list((WORKSPACE / 'src').rglob('*.py')) + list((ROOT / 'baseline-source').rglob('*.py'))
    paths += list(Path(__file__).parent.glob('*.py'))
    paths += [WORKSPACE / 'docs/mixed-split-protocol.md']
    paths += [WORKSPACE / f'benchmarks/ruler/{n}.py' for n in ['collect', 'common', 'report']]
    paths += list((RULER / 'data').glob('*/*.jsonl'))
    write_json(ROOT / 'frozen.json', {str(p.relative_to(WORKSPACE)): digest(p) for p in paths})
    write_json(ROOT / 'selection.json', {'latency': [r['id'] for r in diagnose.inputs().values()],
                                        'quality': [r['id'] for r in quality_rows()]})
    write_json(ROOT / 'environment.json', {'gpu': gpu_state(), 'versions': {name: importlib.metadata.version(name)
        for name in ['vllm', 'torch', 'transformers']}})
    assert importlib.metadata.version('vllm') == '0.23.0'
    started = time.monotonic()
    def run(job, cap):
        remaining = 2400 - sum(r['seconds'] for r in read_jsonl(ROOT / 'ledger.jsonl')) if (ROOT / 'ledger.jsonl').exists() else 2400
        assert remaining > 40
        execute(job, min(cap, remaining))
    run('micro', 600)
    if not json.loads((ROOT / 'micro.json').read_text())['gate']['passed']:
        write_json(ROOT / 'decision.json', {'accepted': False, 'stopped_at': 'micro_gate'}); return
    for job in ['A1', 'B1', 'B2', 'A2']: run(job, 450)
    if not latency_gate()['passed']:
        write_json(ROOT / 'decision.json', {'accepted': False, 'stopped_at': 'latency_gate'}); return
    for job in ['A-quality', 'B-quality']: run(job, 450)
    result = quality_gate()
    write_json(ROOT / 'decision.json', {'accepted': result['passed'], 'stopped_at': 'quality_gate',
                                       'total_wall_seconds': time.monotonic() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--job', choices=['A1', 'B1', 'B2', 'A2', 'A-quality', 'B-quality'])
    args = parser.parse_args()
    worker(args.job) if args.job else main()
