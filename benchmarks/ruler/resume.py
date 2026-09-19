"""One-shot continuation authorized after user pause; no quality-based retries."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import psutil
from common import ROOT, WORKSPACE, ARMS, digest, write_json, append
from run import used, memory, execute

def resume_partial(limit):
    available,swap=memory(); assert available>=2*2**30
    start=time.monotonic();tracked={};reason='completed';code=None
    append(ROOT/'ledger.jsonl',{'event':'start','phase':'resume','length':4096,'arm':'bf16_triton','utc':time.time(),'limit':limit})
    with (ROOT/'resume-4096-bf16_triton.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('resume_collect.py'))],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        try:
            while child.poll() is None:
                try:
                    for p in psutil.Process(child.pid).children(recursive=True):tracked[p.pid]=p.create_time()
                except psutil.NoSuchProcess:pass
                available,current_swap=memory()
                append(ROOT/'telemetry.jsonl',{'utc':time.time(),'phase':'resume','length':4096,'arm':'bf16_triton',
                    'available':available,'swap':current_swap,'elapsed':time.monotonic()-start})
                if available<2*2**30 or current_swap-swap>256*2**20:reason='memory_guard';break
                if time.monotonic()-start>=limit-35:reason='time_guard';break
                time.sleep(1)
            code=child.poll()
            if code not in (None,0):reason='child_failed'
        except KeyboardInterrupt:
            reason='user_interrupted';raise
        finally:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            for pid,created in tracked.items():
                try:
                    p=psutil.Process(pid)
                    if p.create_time()==created:p.terminate()
                except psutil.NoSuchProcess:pass
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            for pid,created in tracked.items():
                try:
                    p=psutil.Process(pid)
                    if p.create_time()==created and p.is_running():p.kill()
                except psutil.NoSuchProcess:pass
            append(ROOT/'ledger.jsonl',{'event':'end','phase':'resume','length':4096,'arm':'bf16_triton',
                'utc':time.time(),'seconds':time.monotonic()-start,'reason':reason,'code':code})
    assert reason=='completed' and code==0, 'Stopped; no retries'

def main():
    for manifest,base in [('frozen-sources.json',WORKSPACE),('resume-frozen.json',WORKSPACE)]:
        for name,sha in json.loads((ROOT/manifest).read_text()).items():assert digest(base/name)==sha,name
    data=json.loads((ROOT/'data-manifest.json').read_text())
    for name,sha in data['files'].items():assert digest(ROOT/name)==sha,name
    for name,sha in data['sources'].items():assert digest(ROOT/'upstream'/name)==sha,name
    old=json.loads((ROOT/'environment.json').read_text())
    from importlib.metadata import version
    assert all(version(name)==v for name,v in old['runtime'].items())
    assert not (ROOT/'resume-started.json').exists(), 'Continuation already attempted'
    write_json(ROOT/'resume-started.json',{'utc':time.time(),'used_seconds':used(),'budget_seconds':7200})
    resume_partial(7200-used())
    for length,arms in [(4096,['fp8','mixed']),(8192,list(reversed(ARMS))),(16384,['fp8','mixed','bf16_flash','bf16_triton'])]:
        for arm in arms:
            assert 7200-used()>35
            execute('formal',length,arm,7200-used())
    subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],check=True)

if __name__=='__main__':main()
