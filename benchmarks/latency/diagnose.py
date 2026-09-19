"""Frozen, bounded prefill diagnosis. Does not modify production implementation."""
import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import httpx
import psutil

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / 'benchmarks/ruler'))
from common import MODEL, ROOT as RULER, digest, write_json, read_jsonl, append
from collect import make_spec, counter
from inference_service.config import Settings
from inference_service.runtime import owned_backend

ROOT = WORKSPACE / '.local/research/mixed-latency-2026-09-18'
ARMS = ['bf16_triton', 'mixed', 'fp8', 'bf16_flash']
ORDERS = [[16384, 4096, 8192], [8192, 16384, 4096], [4096, 8192, 16384]]


def inputs():
    return {length: next(r for r in read_jsonl(RULER / 'data' / str(length) / 'niah_single_1.jsonl')
                         if r['sample'] == 0) for length in [4096, 8192, 16384]}


def make_launch(settings, arm, out):
    command, env = make_spec(settings, arm, out)
    command += ['--profiler-config', json.dumps({'profiler': 'torch',
        'torch_profiler_dir': str(out / 'traces'), 'torch_profiler_with_stack': False,
        'torch_profiler_record_shapes': True, 'ignore_frontend': True})]
    write_json(out / 'launch.json', {'argv': command, 'env': {k: v for k, v in env.items()
        if k.startswith(('VLLM_', 'INFERENCE_RUNTIME_PROFILE', 'PYTHONPATH', 'C_INCLUDE_PATH'))}})
    return command, env


async def requests(settings, out):
    rows = inputs()
    async with httpx.AsyncClient(base_url=settings.backend_url, timeout=120, trust_env=False) as client:
        async def request(length, phase, rep):
            row = rows[length]
            name = f'{phase}-{length}-{rep}'
            (await client.post('/reset_prefix_cache')).raise_for_status()
            before_response = await client.get('/metrics'); before_response.raise_for_status()
            before = before_response.text
            if phase == 'profile':
                (await client.post('/start_profile')).raise_for_status()
            start = time.perf_counter(); times = []; text = []; usage = None; done = False; finish = None
            body = {'model': settings.served_model, 'prompt': row['prompt_ids'], 'max_tokens': 1,
                    'ignore_eos': True, 'temperature': 0, 'seed': 42, 'stream': True,
                    'stream_options': {'include_usage': True}}
            async with asyncio.timeout(120):
                async with client.stream('POST', '/v1/completions', json=body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data:'): continue
                        elapsed = time.perf_counter() - start; data = line[5:].strip()
                        append(out / 'responses' / f'{name}.jsonl', {'elapsed': elapsed, 'data': data})
                        if data == '[DONE]': done = True; continue
                        item = json.loads(data)
                        assert not item.get('error'), item
                        if item.get('usage'): usage = item['usage']
                        for choice in item.get('choices', []):
                            if choice.get('text'): times.append(elapsed); text.append(choice['text'])
                            if choice.get('finish_reason'): finish = choice['finish_reason']
            e2e = time.perf_counter() - start
            if phase == 'profile':
                (await client.post('/stop_profile')).raise_for_status()
            after_response = await client.get('/metrics'); after_response.raise_for_status()
            after = after_response.text
            (out / 'metrics' / f'{name}-before.txt').write_text(before)
            (out / 'metrics' / f'{name}-after.txt').write_text(after)
            hits = counter(after, 'vllm:prefix_cache_hits_total') - counter(before, 'vllm:prefix_cache_hits_total')
            preemptions = counter(after, 'vllm:num_preemptions_total') - counter(before, 'vllm:num_preemptions_total')
            record = {'phase': phase, 'length': length, 'rep': rep, 'prompt_tokens': len(row['prompt_ids']),
                      'ttft': times[0] if times else None, 'e2e': e2e, 'usage': usage, 'done': done,
                      'finish': finish, 'text': ''.join(text), 'hits': hits, 'preemptions': preemptions}
            append(out / 'results.jsonl', record)
            assert done and times and usage and finish == 'length', record
            assert usage['prompt_tokens'] == len(row['prompt_ids']) and usage['completion_tokens'] == 1, record
            assert hits == 0 and preemptions == 0 and not (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0), record
            print(json.dumps(record), flush=True)
        for length in [4096, 8192, 16384]: await request(length, 'warmup', 0)
        for rep, order in enumerate(ORDERS):
            for length in order: await request(length, 'measure', rep)
        await request(16384, 'profile', 0)


def worker(arm):
    out = ROOT / arm; out.mkdir()
    for name in ['responses', 'metrics', 'traces']: (out / name).mkdir()
    settings = Settings(profile='chunked-hybrid', model=MODEL,
                        backend_url='http://127.0.0.1:8032', log_dir=out / 'engine')
    os.environ['PYTHON_DEV_HEADERS'] = str(WORKSPACE / '.tmp/python-dev/extracted/usr/include/python3.12')
    start = time.monotonic()
    with patch('inference_service.runtime.launch_spec', lambda s: make_launch(s, arm, out)):
        with owned_backend(settings):
            write_json(out / 'startup.json', {'seconds': time.monotonic() - start})
            asyncio.run(requests(settings, out))
    logs = '\n'.join(p.read_text(errors='replace') for p in (out / 'engine').glob('*.log'))
    capacity = re.findall(r'GPU KV cache size: ([\d,]+) tokens', logs)
    assert capacity
    if arm == 'mixed': assert 'route=dual_source' in logs and 'prefill_block_m=64' in logs
    assert list((out / 'traces').glob('*.pt.trace.json.gz')), 'Missing profiler trace'
    write_json(out / 'summary.json', {'kv_capacity_tokens': int(capacity[-1].replace(',', '')),
                                     'lifecycle_seconds': time.monotonic() - start})


def supervise():
    ROOT.mkdir(parents=True, exist_ok=False)
    paths = list((WORKSPACE / 'src').rglob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    paths += [WORKSPACE / 'docs/mixed-latency-protocol.md', WORKSPACE / 'benchmarks/ruler/collect.py',
              WORKSPACE / 'benchmarks/ruler/common.py']
    paths += [RULER / 'data' / str(n) / 'niah_single_1.jsonl' for n in inputs()]
    write_json(ROOT / 'frozen.json', {str(p.relative_to(WORKSPACE)): digest(p) for p in paths})
    write_json(ROOT / 'inputs.json', inputs())
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv'], text=True)
    write_json(ROOT / 'environment.json', {'gpu': gpu, 'python': sys.version,
        'versions': {name: importlib.metadata.version(name) for name in ['vllm', 'torch', 'transformers']}})
    assert importlib.metadata.version('vllm') == '0.23.0'
    started = time.monotonic()
    for arm in ARMS:
        start = time.monotonic(); initial_swap = psutil.swap_memory().used
        assert psutil.virtual_memory().available >= 2 * 2**30
        tracked = {}; reason = 'completed'; code = None
        with (ROOT / f'{arm}.log').open('x') as log:
            child = subprocess.Popen([sys.executable, __file__, '--arm', arm], stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
            try:
                while child.poll() is None:
                    try:
                        for p in psutil.Process(child.pid).children(recursive=True): tracked[p.pid] = p.create_time()
                    except psutil.NoSuchProcess: pass
                    elapsed = time.monotonic() - start
                    available = psutil.virtual_memory().available; swap = psutil.swap_memory().used
                    append(ROOT / 'telemetry.jsonl', {'arm': arm, 'elapsed': elapsed, 'available': available, 'swap': swap})
                    if available < 2 * 2**30 or swap - initial_swap > 256 * 2**20:
                        reason = 'memory_guard'; break
                    if elapsed > 410 or time.monotonic() - started > 1760:
                        reason = 'time_guard'; break
                    time.sleep(1)
                code = child.poll()
                if code not in (None, 0): reason = 'child_failed'
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
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
                append(ROOT / 'ledger.jsonl', {'arm': arm, 'seconds': time.monotonic() - start,
                    'reason': reason, 'code': code})
            print(json.dumps({'arm': arm, 'reason': reason, 'code': code}), flush=True)
            assert reason == 'completed' and code == 0, 'Stopped; no automatic retries'
    write_json(ROOT / 'completion.json', {'seconds': time.monotonic() - started, 'arms': ARMS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    worker(args.arm) if args.arm else supervise()
