"""Bounded Stage 2 batches. Run in WSL using the pinned environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage2'
ANALYZE = runpy.run_path(str(ROOT / 'scripts/analyze-stage1-benchmark.py'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def metrics():
    return urllib.request.urlopen('http://127.0.0.1:8000/metrics', timeout=4).read().decode()


def requests_valid(requests, count, input_len, output_len):
    return len(requests) == count + 4 and all(r['success'] and not r['error'] and
        r['prompt_len'] == input_len and r['output_tokens'] == output_len for r in requests)


def monitor(folder, done):
    with (folder / 'telemetry.jsonl').open('x') as handle:
        while not done.is_set():
            row = {'time': time.time()}
            for name, command in {
                'gpu': ['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,pstate', '--format=csv,noheader,nounits'],
                'gpu_processes': ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
            }.items():
                try:
                    row[name] = subprocess.check_output(command, text=True, timeout=4).strip()
                except Exception as exc:
                    row[name] = {'unavailable': str(exc)}
            row['proc_stat'] = Path('/proc/stat').read_text().splitlines()[0]
            row['meminfo'] = Path('/proc/meminfo').read_text()
            row['loadavg'] = Path('/proc/loadavg').read_text()
            try:
                row['metrics'] = metrics()
            except Exception as exc:
                row['metrics_error'] = str(exc)
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            done.wait(1)


def batch(name, input_len, output_len, concurrency, count=16, cap=4):
    budget = json.loads((OUT / 'budget.json').read_text())
    if time.time() + 600 > budget['deadline_epoch']:
        raise RuntimeError('Budget cannot accommodate another bounded batch')
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=False)
    result_name = f'round1-input{input_len}-concurrency{concurrency}'
    env = dict(os.environ, STAGE1_CONFIG=str(ROOT / f'configs/stage2-cap{cap}.env'),
               STAGE1_STATE_DIR=str(ROOT / '.tmp/stage2'), STAGE2_BENCH='1',
               STAGE2_REQUEST_LOG=str(folder / 'requests.jsonl'))
    command = ['bash', 'scripts/run-stage1-benchmark.sh', '--point', str(input_len),
               str(concurrency), str(folder), result_name, str(count), str(output_len)]
    save(folder / 'command.json', {'argv': command, 'environment': {k:v for k,v in env.items() if k.startswith('STAGE')},
                                 'cap': cap, 'count': count, 'start': time.time(),
                                 'hashes': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in [Path(__file__), ROOT/'scripts/stage2-bench-entry.py',
                                                      ROOT/'scripts/benchmark-point.sh', Path(env['STAGE1_CONFIG'])]}})
    time.sleep(2)  # Allow the engine's asynchronous metric publication to settle.
    (folder / 'metrics-before.txt').write_text(metrics())
    done = threading.Event()
    thread = threading.Thread(target=monitor, args=(folder, done))
    thread.start()
    try:
        process = subprocess.Popen(command, env=env, cwd=ROOT, start_new_session=True)
        try:
            code = process.wait(timeout=600)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
            raise
        if code:
            raise RuntimeError(f'Benchmark exited {code}; stop escalation')
        time.sleep(2)
        (folder / 'metrics-after.txt').write_text(metrics())
        row = ANALYZE['summarize_file'](folder / (result_name + '.json'), count, output_len)
        save(folder / 'summary.json', row)
        if not row['request_evidence_passed']:
            raise RuntimeError('Invalid request results; stop escalation')
        requests = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
        if not requests_valid(requests, count, input_len, output_len):
            raise RuntimeError('Probe/warmup/request evidence invalid; stop escalation')
        print(json.dumps({'batch': name, **row}), flush=True)
        return row
    finally:
        done.set()
        thread.join()
        save(folder / 'end.json', {'end': time.time()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('input', type=int)
    p.add_argument('output', type=int)
    p.add_argument('concurrency', type=int)
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--cap', type=int, default=4)
    a = p.parse_args()
    batch(a.name, a.input, a.output, a.concurrency, a.count, a.cap)
