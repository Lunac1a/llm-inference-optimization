"""Budgeted orchestration; no retries, own-process-only cleanup, frozen sources."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil
from common import ROOT, WORKSPACE, ARMS, LENGTHS, digest, write_json, read_jsonl, append

def memory():
    values={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    return values['MemAvailable'], values['SwapTotal']-values['SwapFree']

def used():
    return sum(r['seconds'] for r in read_jsonl(ROOT/'ledger.jsonl') if r['event']=='end') if (ROOT/'ledger.jsonl').exists() else 0

def execute(phase,length,arm,limit):
    available,swap=memory(); assert available>=2*2**30
    started=time.monotonic();child=None;tracked={};reason='completed';code=None
    append(ROOT/'ledger.jsonl',{'event':'start','phase':phase,'length':length,'arm':arm,'utc':time.time(),'limit':limit})
    try:
        with (ROOT/f'{phase}-{length}-{arm}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('collect.py')),'--phase',phase,'--length',str(length),'--arm',arm],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
            while child.poll() is None:
                try:
                    for process in psutil.Process(child.pid).children(recursive=True):tracked[process.pid]=process.create_time()
                except psutil.NoSuchProcess:pass
                available,current_swap=memory()
                append(ROOT/'telemetry.jsonl',{'utc':time.time(),'phase':phase,'length':length,'arm':arm,
                    'available':available,'swap':current_swap,'elapsed':time.monotonic()-started})
                if available<2*2**30 or current_swap-swap>256*2**20:reason='memory_guard';break
                if time.monotonic()-started>=max(1,limit-35):reason='time_guard';break
                time.sleep(1)
            code=child.poll()
            if code not in (None,0):reason='child_failed'
    finally:
        if child is not None:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            for pid,created in tracked.items():
                try:
                    process=psutil.Process(pid)
                    if process.create_time()==created:process.terminate()
                except psutil.NoSuchProcess:pass
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            for pid,created in tracked.items():
                try:
                    process=psutil.Process(pid)
                    if process.create_time()==created and process.is_running():process.kill()
                except psutil.NoSuchProcess:pass
        elapsed=time.monotonic()-started
        append(ROOT/'ledger.jsonl',{'event':'end','phase':phase,'length':length,'arm':arm,'utc':time.time(),
            'seconds':elapsed,'reason':reason,'code':code})
    print(json.dumps({'phase':phase,'arm':arm,'length':length,'seconds':elapsed,'reason':reason}),flush=True)
    assert reason=='completed' and code==0, 'Stopped; failure retained, no retries'

def freeze():
    paths=list((WORKSPACE/'src/inference_service').rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))
    paths += [WORKSPACE/'docs/ruler-protocol.md', ROOT/'data-manifest.json',ROOT/'corpus-manifest.json',ROOT/'environment.json',Path(__file__).with_name('requirements-data.txt')]
    mapping={str(p.relative_to(WORKSPACE)):digest(p) for p in paths}
    data=json.loads((ROOT/'data-manifest.json').read_text())
    for name,sha in data['files'].items():assert digest(ROOT/name)==sha
    for name,sha in data['sources'].items():assert digest(ROOT/'upstream'/name)==sha
    corpus=json.loads((ROOT/'corpus-manifest.json').read_text())
    for name,sha in corpus['files'].items():assert digest(ROOT/'upstream/scripts/data/synthetic/json'/name)==sha
    file=ROOT/'frozen-sources.json'
    if file.exists():assert json.loads(file.read_text())==mapping,'Frozen sources changed'
    else:write_json(file,mapping)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pilot','formal']);args=parser.parse_args()
    freeze()
    if args.phase=='pilot':
        assert not (ROOT/'ledger.jsonl').exists(),'Pilot already attempted'
        for arm in ARMS:
            elapsed=used(); assert elapsed<1200
            execute('pilot',8192,arm,1200-elapsed)
    else:
        assert not (ROOT/'formal-selection.json').exists(),'Formal collection already selected/attempted'
        pilots=[ROOT/'runs'/f'pilot-8192-{arm}' for arm in ARMS]
        assert all((path/'summary.json').exists() for path in pilots)
        pilot_rows=[read_jsonl(path/'results.jsonl') for path in pilots]
        # Sum per-task slowest-arm timings; avoid extrapolating the longest output to every task.
        tasks=sorted({r['task'] for rows in pilot_rows for r in rows})
        t=sum(max(r['e2e'] for rows in pilot_rows for r in rows if r['task']==task) for task in tasks)
        s=max(json.loads((path/'summary.json').read_text())['startup_seconds'] for path in pilots)
        remaining=7200-used()
        feasible=[n for n in (10,5) if 12*s+4*n*3.7*1.25*t <= .8*remaining]
        assert feasible,'Timing gate: neither preregistered sample count fits'
        n=max(feasible)
        write_json(ROOT/'formal-selection.json',{'n':n,'pilot_sum_task_max_seconds':t,'max_startup_seconds':s,
            'predicted_seconds':12*s+4*n*3.7*1.25*t,'remaining_seconds':remaining,'utc':time.time(),'selection_uses_quality':False})
        orders=[ARMS,list(reversed(ARMS)),['fp8','mixed','bf16_flash','bf16_triton']]
        for length,order in zip(LENGTHS,orders):
            for arm in order:
                remaining=7200-used();assert remaining>0
                execute('formal',length,arm,remaining)

if __name__=='__main__':main()
