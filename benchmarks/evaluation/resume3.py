"""User-authorized one-shot continuation, preserving the failed original attempt."""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import psutil
from config import *
import run as original

RESUME = ROOT / 'resume-3'
PREV = ROOT / 'resume-2'


def verify_original():
    verify_frozen()
    for name, sha in read(ROOT / 'pause-snapshot.json').items():
        assert digest(ROOT / name) == sha, f'Original evidence changed: {name}'


def verify_previous():
    previous = ROOT / 'resume-1'
    for name, sha in read(previous / 'snapshot.json').items():
        assert digest(previous / name) == sha, f'Previous continuation changed: {name}'
    for name, sha in read(previous / 'frozen.json').items():
        assert digest(name) == sha


def verify_paused():
    for name, sha in read(PREV / 'snapshot.json').items():
        assert digest(PREV / name) == sha, f'Paused evidence changed: {name}'
    for name, sha in read(PREV / 'frozen.json').items():
        assert digest(name) == sha


def verify_resume():
    verify_frozen()
    for name, sha in read(RESUME / 'frozen.json').items():
        assert digest(name) == sha, f'Continuation source changed: {name}'


def missing_jobs():
    return [job for job, *_ in original.jobs() if not any((base / 'runs' / job / 'summary.json').exists() for base in [ROOT, PREV])]


def used():
    return sum(r['seconds'] for r in rows(PREV / 'ledger.jsonl')) + original.used() + sum(r['seconds'] for r in rows(ROOT / 'resume-1/ledger.jsonl')) + (sum(r['seconds'] for r in rows(RESUME / 'ledger.jsonl'))
                              if (RESUME / 'ledger.jsonl').exists() else 0)


def execute(job, swap_start):
    verify_resume()
    gpu_mib = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    assert gpu_mib <= 2048, f'GPU environment memory guard: {gpu_mib} MiB already used'
    cap = BUDGET_SECONDS - used(); assert cap > 45, 'Budget exhausted'
    start = time.monotonic(); child = None; tracked = {}; reason = 'interrupted'; code = None
    try:
        with (RESUME / f'{job}.log').open('x') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', job],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
            while child.poll() is None:
                try:
                    for p in psutil.Process(child.pid).children(recursive=True): tracked[p.pid] = p.create_time()
                except psutil.NoSuchProcess: pass
                available, swap = original.memory()
                append(RESUME / 'telemetry.jsonl', {'job': job, 'elapsed': time.monotonic() - start,
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
        append(RESUME / 'ledger.jsonl', {'job': job, 'seconds': time.monotonic() - start, 'reason': reason, 'code': code})
    print(json.dumps({'job': job, 'reason': reason, 'total_gpu_seconds': used()}), flush=True)
    assert reason == 'completed' and code == 0, 'Continuation stopped; no further automatic retry'



def prior_ids():
    records = rows(PREV / 'runs/16384-1-fp8/quality.jsonl')
    ids = {r['id'] for r in records}
    assert len(ids) == len(records) == 24
    return ids


async def collect_partial(out, length, block):
    async with original.httpx.AsyncClient(base_url=f'http://127.0.0.1:{PORT}', timeout=120, trust_env=False) as client:
        warm = {'id': 'warmup', 'task': 'warmup', 'sample': -1, 'target_length': 0,
                'prompt_ids': [1, 2, 3, 4], 'max_tokens': 1}
        await original.request(client, out, warm, 'warmup', 'short-warmup', fixed=True)
        timing = {**read(ROOT / 'timing-inputs.json')[str(length)], 'max_tokens': 32}
        await original.request(client, out, timing, 'warmup', 'timing-warmup', fixed=True)
        # Three formal timing samples are already complete in the paused segment.
        data = [r for p in (ROOT / 'data' / str(length)).glob('*.jsonl') for r in rows(p)
                if block * 10 <= r['sample'] < (block + 1) * 10]
        data = sorted(data, key=lambda r: (r['sample'], r['task']))
        done = prior_ids()
        assert {r['id'] for r in data[:24]} == done and len(data) == 130
        pending = [r for r in data if r['id'] not in done]
        assert len(pending) == 106
        write(out / 'continued-from.json', {'prior_ids': sorted(done), 'pending_ids': [r['id'] for r in pending],
             'prior_directory': str(PREV / 'runs/16384-1-fp8'), 'timing_reused': 3})
        for number, row in enumerate(pending, 1):
            result = await original.request(client, out, row, 'quality', row['id'].replace(':', '-'))
            print(json.dumps({'completed': number, 'id': row['id'], 'seconds': round(result['e2e'], 3)}), flush=True)


def worker(job):
    original.ROOT = RESUME
    if job == '16384-1-fp8':
        original.collect = collect_partial
    original.worker(job)
    if job == '16384-1-fp8':
        p = RESUME / 'runs' / job / 'summary.json'
        summary = read(p)
        summary.update(quality_requests=106, latency_requests=0, prior_quality_requests=24, prior_latency_requests=3)
        write(p, summary)


def main():
    verify_original(); verify_previous(); verify_paused()
    assert not (RESUME / 'started.json').exists(), 'Continuation already attempted'
    assert read(RESUME / 'power-guard.json')['active'] is True
    assert missing_jobs() == ['16384-1-fp8', '16384-1-bf16_flash', '16384-1-mixed']
    assert len(prior_ids()) == 24
    RESUME.mkdir(exist_ok=True)
    # Aliases allow reuse of the unchanged collector; no original evidence is modified.
    for name in ['data', 'timing-inputs.json']:
        (RESUME / name).symlink_to(ROOT / name, target_is_directory=(ROOT / name).is_dir())
    paths = [Path(__file__), Path(__file__).with_name('report_resumed3.py'),
             WORKSPACE / 'benchmarks/evaluation/keep_awake3.py', WORKSPACE / 'docs/final-benchmark-resume3-protocol.md']
    write(RESUME / 'frozen.json', {str(p): digest(p) for p in paths})
    write(RESUME / 'started.json', {'utc': time.time(), 'prior_gpu_seconds': used(),
        'budget_seconds': BUDGET_SECONDS, 'jobs': missing_jobs(), 'gpu': original.gpu()})
    _, swap_start = original.memory(); status = {'complete': False}
    try:
        for job in missing_jobs(): execute(job, swap_start)
        verify_original(); verify_previous(); verify_paused(); verify_resume()
        status['complete'] = True
    finally:
        status['gpu_lifecycle_seconds'] = used()
        status['completed_quality_responses'] = sum(len(rows(p)) for base in [ROOT, PREV, RESUME]
            for p in (base / 'runs').glob('*/quality.jsonl'))
        with socket.socket() as sock: status['port_released'] = sock.connect_ex(('127.0.0.1', PORT)) != 0
        status['compute_processes_after'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv'], text=True)
        write(RESUME / 'completion.json', status)


if __name__ == '__main__':
    headers = WORKSPACE / '.tmp/python-dev/extracted/usr/include/python3.12'
    if headers.exists(): os.environ['C_INCLUDE_PATH'] = os.pathsep.join(filter(None, [str(headers), str(headers.parent), os.environ.get('C_INCLUDE_PATH')]))
    def interrupted(*_): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(); parser.add_argument('--worker'); args = parser.parse_args()
    if args.worker:
        verify_resume()
        worker(args.worker)
    else: main()
