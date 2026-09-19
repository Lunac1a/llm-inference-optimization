"""Bounded, one-attempt final evaluation; no tuning and no automatic retries."""
import argparse
import asyncio
import importlib.metadata
import importlib.util
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import patch

import httpx
import psutil
from config import *

sys.path.insert(0, str(WORKSPACE / 'src'))
sys.path.insert(0, str(WORKSPACE / 'benchmarks/ruler'))
from collect import make_spec, counter
from inference_service.config import Settings
from inference_service.runtime import owned_backend


def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem', '--format=csv'], text=True)


def jobs():
    orders = [['bf16_flash', 'fp8', 'mixed'], ['fp8', 'mixed', 'bf16_flash'], ['mixed', 'bf16_flash', 'fp8']]
    return [(f'{length}-{block}-{arm}', length, block, arm)
            for length, order in zip(LENGTHS, orders) for block in [0, 1]
            for arm in (order if block == 0 else list(reversed(order)))]


async def request(client, out, row, phase, name, fixed=False):
    reset = await client.post('/reset_prefix_cache'); reset.raise_for_status()
    before = await client.get('/metrics'); before.raise_for_status()
    body = {'model': 'qwen3-local', 'prompt': row['prompt_ids'], 'max_tokens': row['max_tokens'],
            'temperature': 0, 'seed': 42, 'stream': True, 'stream_options': {'include_usage': True}}
    if fixed: body['ignore_eos'] = True
    start = time.perf_counter(); times = []; parts = []; usage = None; reason = None; done = False
    raw = out / 'responses' / f'{name}.jsonl'
    with raw.open('x', encoding='utf-8') as stream:
        async with asyncio.timeout(120):
            async with client.stream('POST', '/v1/completions', json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith('data:'): continue
                    elapsed = time.perf_counter() - start; data = line[5:].strip()
                    stream.write(json.dumps({'elapsed': elapsed, 'data': data}) + '\n')
                    if data == '[DONE]': done = True; continue
                    item = json.loads(data)
                    if item.get('error'): raise RuntimeError(item['error'])
                    if item.get('usage'): usage = item['usage']
                    for choice in item.get('choices', []):
                        if choice.get('text'):
                            parts.append(choice['text']); times.append(elapsed)
                        if choice.get('finish_reason'): reason = choice['finish_reason']
    e2e = time.perf_counter() - start
    assert done and usage and reason in ('stop', 'length')
    assert usage['prompt_tokens'] == len(row['prompt_ids'])
    assert 0 < usage['completion_tokens'] <= row['max_tokens']
    if fixed: assert usage['completion_tokens'] == row['max_tokens']
    prediction = ''.join(parts)
    if phase != 'warmup': assert prediction.strip() and times, 'Empty output; no retry'
    after = await client.get('/metrics'); after.raise_for_status()
    (out / 'metrics' / f'{name}-before.txt').write_text(before.text)
    (out / 'metrics' / f'{name}-after.txt').write_text(after.text)
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    hits = counter(after.text, 'vllm:prefix_cache_hits_total') - counter(before.text, 'vllm:prefix_cache_hits_total')
    preemptions = counter(after.text, 'vllm:num_preemptions_total') - counter(before.text, 'vllm:num_preemptions_total')
    record = {'id': row['id'], 'task': row['task'], 'sample': row['sample'], 'target_length': row['target_length'],
        'phase': phase, 'raw_name': name, 'prediction': prediction, 'usage': usage, 'finish_reason': reason,
        'done': done, 'e2e': e2e, 'ttft': times[0] if times else None, 'content_times': times,
        'max_content_gap': max((b - a for a, b in zip(times, times[1:])), default=0),
        'cached_tokens': cached, 'cache_hit_delta': hits, 'preemptions': preemptions}
    append(out / f'{phase}.jsonl', record)
    assert cached == hits == preemptions == 0, 'Non-cold or preempted request'
    return record


async def collect(out, length, block):
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{PORT}', timeout=120, trust_env=False) as client:
        warm = {'id': 'warmup', 'task': 'warmup', 'sample': -1, 'target_length': 0,
                'prompt_ids': [1, 2, 3, 4], 'max_tokens': 1}
        await request(client, out, warm, 'warmup', 'short-warmup', fixed=True)
        timing = {**read(ROOT / 'timing-inputs.json')[str(length)], 'max_tokens': 32}
        await request(client, out, timing, 'warmup', 'timing-warmup', fixed=True)
        for repeat in range(3):
            await request(client, out, timing, 'latency', f'latency-{repeat}', fixed=True)
        data = [r for p in (ROOT / 'data' / str(length)).glob('*.jsonl') for r in rows(p)
                if block * 10 <= r['sample'] < (block + 1) * 10]
        assert len(data) == 130
        for number, row in enumerate(sorted(data, key=lambda r: (r['sample'], r['task'])), 1):
            result = await request(client, out, row, 'quality', row['id'].replace(':', '-'))
            print(json.dumps({'completed': number, 'id': row['id'], 'seconds': round(result['e2e'], 3)}), flush=True)


def worker(job):
    verify_frozen()
    _, length, block, arm = next(r for r in jobs() if r[0] == job)
    out = ROOT / 'runs' / job; out.mkdir(parents=True, exist_ok=False)
    for folder in ['responses', 'metrics']: (out / folder).mkdir()
    # Settings() now selects the final mixed service by default. Comparison overrides are local.
    settings = Settings(model=MODEL, backend_url=f'http://127.0.0.1:{PORT}', log_dir=out / 'engine')
    os.environ['PYTHON_DEV_HEADERS'] = str(WORKSPACE / '.tmp/python-dev/extracted/usr/include/python3.12')
    def launch(s):
        command, env = make_spec(s, arm, out)
        for name in list(env):
            if name.startswith('INFERENCE_MIXED_'): env.pop(name)
        write(out / 'launch.json', {'argv': command, 'env': {k: v for k, v in env.items()
            if k.startswith(('VLLM_', 'INFERENCE_RUNTIME_PROFILE', 'PYTHONPATH', 'C_INCLUDE_PATH'))}})
        return command, env
    started = time.monotonic(); before = gpu()
    with patch('inference_service.runtime.launch_spec', launch):
        with owned_backend(settings):
            write(out / 'startup.json', {'seconds': time.monotonic() - started, 'gpu_before': before})
            asyncio.run(collect(out, length, block))
    logs = '\n'.join(p.read_text(errors='replace') for p in (out / 'engine').glob('*.log'))
    capacity = re.findall(r'GPU KV cache size: ([\d,]+) tokens', logs); assert capacity
    if arm == 'mixed':
        assert 'block_source_branch=True' in logs and 'route=dual_source' in logs and 'prefill_block_m=64' in logs
    else: assert 'DUAL_SOURCE_REGISTER' not in logs
    write(out / 'summary.json', {'arm': arm, 'length': length, 'block': block, 'quality_requests': 130,
        'latency_requests': 3, 'capacity_tokens': int(capacity[-1].replace(',', '')),
        'lifecycle_seconds': time.monotonic() - started, 'gpu_after': gpu()})


def numerical():
    verify_frozen()
    import torch
    from inference_service.backends import dual_source_kernel as product
    from inference_service.backends.chunked_hybrid import BLOCK_SOURCE_BRANCH
    assert BLOCK_SOURCE_BRANCH
    # Reuse the frozen numerical fixtures/reference; never invoke their performance loop.
    spec = importlib.util.spec_from_file_location('fixtures', WORKSPACE / 'benchmarks/latency/branch_micro.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.dual is product
    results = []
    for seed in [42, 43]:
        for case in module.CASES:
            kwargs, state = module.setup(case, seed)
            before = [kwargs[k].view(torch.uint8).clone() for k in ['k', 'v', 'current_key', 'current_value']]
            ref = module.reference(kwargs, state)
            module.original.unified_attention(**kwargs); original = kwargs['out'].clone()
            product.unified_attention(**kwargs); disabled = kwargs['out'].clone()
            product.unified_attention(**kwargs, block_source_branch=True); candidate = kwargs['out'].clone()
            error = module.errors(candidate, ref); delta = module.errors(candidate, original)
            unchanged = all(torch.equal(a, kwargs[k].view(torch.uint8)) for a, k in zip(before, ['k', 'v', 'current_key', 'current_value']))
            passed = (error['finite'] and error['max_abs'] <= .02 and error['rms'] <= .002
                and delta['finite'] and delta['max_abs'] <= .002 and delta['rms'] <= .0002
                and unchanged and torch.equal(disabled, original))
            results.append({'seed': seed, 'case': case, 'reference': error, 'difference': delta,
                            'inputs_unchanged': unchanged, 'passed': passed})
            write(ROOT / 'numerical.json', results)
            assert passed, 'Numerical integration failed'
    print('20 numerical cases passed', flush=True)


def memory():
    values = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    return values['MemAvailable'], values['SwapTotal'] - values['SwapFree']


def used():
    return sum(r['seconds'] for r in rows(ROOT / 'ledger.jsonl')) if (ROOT / 'ledger.jsonl').exists() else 0


def execute(job, cap, swap_start):
    assert cap > 45, 'Budget exhausted before job'
    verify_frozen()
    available, swap = memory(); assert available >= 2 * 2**30 and swap - swap_start <= 256 * 2**20
    start = time.monotonic(); child = None; tracked = {}; code = None; reason = 'interrupted'
    try:
        with (ROOT / f'{job}.log').open('x') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', job],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
            while child.poll() is None:
                try:
                    for p in psutil.Process(child.pid).children(recursive=True): tracked[p.pid] = p.create_time()
                except psutil.NoSuchProcess: pass
                available, swap = memory()
                append(ROOT / 'telemetry.jsonl', {'job': job, 'elapsed': time.monotonic() - start,
                    'available_bytes': available, 'swap_bytes': swap})
                if available < 2 * 2**30 or swap - swap_start > 256 * 2**20:
                    reason = 'memory_guard'; break
                if time.monotonic() - start >= cap - 35:
                    reason = 'time_guard'; break
                time.sleep(1)
            code = child.poll()
            if code is not None: reason = 'completed' if code == 0 else 'child_failed'
    finally:
        if child is not None:
            if child.poll() is None:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
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
    print(json.dumps({'job': job, 'reason': reason, 'gpu_lifecycle_seconds': used()}), flush=True)
    assert reason == 'completed' and code == 0, 'Stopped; failure retained, no retries'


def main():
    assert not (ROOT / 'started.json').exists(), 'Run already attempted'
    data = read(ROOT / 'data-manifest.json'); assert data['inputs'] == 780
    assert Settings().profile == 'chunked-hybrid'
    versions = {name: importlib.metadata.version(name) for name in ['vllm', 'torch', 'transformers', 'numpy', 'httpx', 'psutil']}
    assert versions['vllm'] == '0.23.0'
    write(ROOT / 'environment.json', {'python': sys.version, 'versions': versions, 'gpu': gpu(), 'model': MODEL})
    paths = list((WORKSPACE / 'src/inference_service').rglob('*.py'))
    paths += list(Path(__file__).parent.glob('*.py'))
    paths += [WORKSPACE / 'benchmarks/ruler' / f'{name}.py' for name in ['common', 'collect', 'report']]
    paths += [WORKSPACE / 'benchmarks/latency/branch_micro.py', WORKSPACE / 'docs/final-benchmark-protocol.md',
              WORKSPACE / 'pyproject.toml', ROOT / 'data-manifest.json', ROOT / 'timing-inputs.json', ROOT / 'environment.json']
    paths += [WORKSPACE / '.local/research/mixed-branch-2026-09-18/baseline-source/inference_service/backends/dual_source_kernel.py']
    frozen = {str(p): digest(p) for p in paths}
    for name in ['files', 'upstream_files', 'corpus_files']:
        for path, sha in data[name].items():
            assert digest(path) == sha; frozen[path] = sha
    for path, sha in frozen.items():
        source = Path(path)
        # Store a complete manifest plus source copies; data is already under this run's root.
        if source.suffix in ('.py', '.md', '.toml'):
            relative = source.relative_to(WORKSPACE) if source.is_relative_to(WORKSPACE) else Path('external') / sha / source.name
            target = ROOT / 'frozen-source' / relative
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    write(ROOT / 'frozen.json', frozen)
    verify_frozen()
    write(ROOT / 'started.json', {'utc': time.time(), 'budget_seconds': BUDGET_SECONDS, 'jobs': jobs()})
    _, swap_start = memory()
    status = {'complete': False}
    try:
        execute('numerical', min(300, BUDGET_SECONDS - used()), swap_start)
        for job, *_ in jobs(): execute(job, BUDGET_SECONDS - used(), swap_start)
        status['complete'] = True
    finally:
        status['gpu_lifecycle_seconds'] = used()
        status['completed_quality_responses'] = sum(len(rows(p)) for p in (ROOT / 'runs').glob('*/quality.jsonl'))
        with socket.socket() as sock: status['port_released'] = sock.connect_ex(('127.0.0.1', PORT)) != 0
        status['compute_processes_after'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv'], text=True)
        write(ROOT / 'completion.json', status)


if __name__ == '__main__':
    headers = WORKSPACE / '.tmp/python-dev/extracted/usr/include/python3.12'
    if headers.exists():
        os.environ['C_INCLUDE_PATH'] = os.pathsep.join(filter(None, [str(headers), str(headers.parent), os.environ.get('C_INCLUDE_PATH')]))
    parser = argparse.ArgumentParser(); parser.add_argument('--worker'); args = parser.parse_args()
    def interrupted(*_): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    if args.worker == 'numerical': numerical()
    elif args.worker: worker(args.worker)
    else: main()
