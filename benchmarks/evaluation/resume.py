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

RESUME = ROOT / 'resume-1'


def verify_original():
    verify_frozen()
    for name, sha in read(ROOT / 'pause-snapshot.json').items():
        assert digest(ROOT / name) == sha, f'Original evidence changed: {name}'


def verify_resume():
    verify_frozen()
    for name, sha in read(RESUME / 'frozen.json').items():
        assert digest(name) == sha, f'Continuation source changed: {name}'


def missing_jobs():
    return [job for job, *_ in original.jobs() if not (ROOT / 'runs' / job / 'summary.json').exists()]


def used():
    return original.used() + (sum(r['seconds'] for r in rows(RESUME / 'ledger.jsonl'))
                              if (RESUME / 'ledger.jsonl').exists() else 0)


def execute(job, swap_start):
    verify_resume()
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


def main():
    verify_original()
    assert not (RESUME / 'started.json').exists(), 'Continuation already attempted'
    assert read(RESUME / 'power-guard.json')['active'] is True
    assert missing_jobs()[0] == '8192-0-bf16_flash' and len(missing_jobs()) == 10
    RESUME.mkdir(exist_ok=True)
    # Aliases allow reuse of the unchanged collector; no original evidence is modified.
    for name in ['data', 'timing-inputs.json']:
        (RESUME / name).symlink_to(ROOT / name, target_is_directory=(ROOT / name).is_dir())
    paths = [Path(__file__), Path(__file__).with_name('report_resumed.py'),
             WORKSPACE / 'benchmarks/evaluation/keep_awake.py', WORKSPACE / 'docs/final-benchmark-resume-protocol.md']
    write(RESUME / 'frozen.json', {str(p): digest(p) for p in paths})
    write(RESUME / 'started.json', {'utc': time.time(), 'prior_gpu_seconds': original.used(),
        'budget_seconds': BUDGET_SECONDS, 'jobs': missing_jobs(), 'gpu': original.gpu()})
    _, swap_start = original.memory(); status = {'complete': False}
    try:
        for job in missing_jobs(): execute(job, swap_start)
        verify_original(); verify_resume()
        status['complete'] = True
    finally:
        status['gpu_lifecycle_seconds'] = used()
        status['completed_quality_responses'] = sum(len(rows(p)) for base in [ROOT, RESUME]
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
        original.ROOT = RESUME
        original.worker(args.worker)
    else: main()
