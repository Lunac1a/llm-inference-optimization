"""Three bounded real-vLLM counter captures, after Windows counter access opens."""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import runpy
import time
import urllib.request

import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2-bandwidth'
STATE=ROOT/'.tmp/stage2-bandwidth'
NCU=ROOT/'.tmp/bandwidth-ncu/ncu'
READER=Path('/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/target/linux-desktop-glibc_2_11_3-x64/ncu')
PY=Path.home()/'.venvs/inference-vllm/bin/python'
METRICS='gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed'
monitor=runpy.run_path(str(ROOT/'scripts/stage2-run.py'))['monitor']


def post(endpoint):
    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/'+endpoint,method='POST'),timeout=120) as response:
        return response.status


def main():
    # Each invocation is one bounded, complete set. Never overwrite captures.
    existing=sorted(p for p in OUT.glob('capture-*') if p.is_dir())
    parse=runpy.run_path(str(ROOT/'scripts/analyze-bandwidth-check.py'))['parse']
    used=0
    for number,folder in enumerate(existing,1):
        if folder.name!=f'capture-{number}' or len(parse(folder/'counters-installed-export.csv'))!=8:
            raise RuntimeError('Cannot resume an incomplete capture')
        used+=json.loads((folder/'end.json').read_text())['end_epoch']-json.loads((folder/'command.json').read_text())['start_epoch']
    if len(existing)>=3:raise RuntimeError('All declared captures exist; no rerun')
    env=dict(os.environ,STAGE1_CONFIG=str(ROOT/'configs/stage2-cap8.env'),STAGE1_STATE_DIR=str(STATE),
             STAGE2_BENCH='1',VLLM_USE_V2_MODEL_RUNNER='0',VLLM_USE_FLASHINFER_SAMPLER='0',
             C_INCLUDE_PATH=f'{ROOT}/.tmp/python-dev/extracted/usr/include/python3.12:{ROOT}/.tmp/python-dev/extracted/usr/include')
    deadline=time.monotonic()+1080-used # include earlier captures; reserve two minutes for probes/cleanup
    for number in range(len(existing)+1,4):
        if time.monotonic()+300>deadline:raise RuntimeError('Insufficient remaining bounded capture budget')
        folder=OUT/f'capture-{number}';folder.mkdir()
        argv=json.loads((ROOT/'artifacts/stage2/server-command.json').read_text())
        argv[argv.index('--max-num-seqs')+1]='8'
        argv+=['--profiler-config','{"profiler":"cuda"}']
        cmd=[str(NCU),'--target-processes','all','--profile-from-start','off','--clock-control','none',
             '--cache-control','all','--kernel-name-base','demangled','--kernel-name','regex:.*cublasGemv.*',
             '--launch-count','8','--metrics',METRICS,'--csv','--log-file',str(folder/'counters.csv'),
             '--export',str(folder/'kernels.ncu-rep'),*argv]
        (folder/'command.json').write_text(json.dumps({'argv':cmd,'start_epoch':time.time()},indent=2))
        with (folder/'server.log').open('w') as log:
            proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        done=threading.Event();thread=threading.Thread(target=monitor,args=(folder,done));thread.start()
        try:
            for _ in range(120):
                if proc.poll() is not None:raise RuntimeError('Profiled server exited during startup')
                try:
                    urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).close();break
                except OSError:time.sleep(1)
            else:raise RuntimeError('Server readiness timeout')
            # Reuse existing fixed-work collector for warmup only.
            warm=folder/'warmup';warm.mkdir()
            env['STAGE2_REQUEST_LOG']=str(warm/'requests.jsonl')
            subprocess.run(['bash','scripts/run-stage1-benchmark.sh','--point','512','1',str(warm),
                            'round1-input512-concurrency1','4','128'],env=env,cwd=ROOT,check=True,timeout=180)
            raw=[json.loads(s) for s in (warm/'requests.jsonl').read_text().splitlines()]
            if len(raw)!=8 or any(not r['success'] or r['error'] for r in raw):raise RuntimeError('Warmup request failure')
            post('start_profile')
            env['STAGE2_REQUEST_LOG']=str(folder/'requests.jsonl')
            subprocess.run(['bash','scripts/stage2-profile-point.sh','512','128','1',str(folder),'2'],
                           env=env,cwd=ROOT,check=True,timeout=180)
            post('stop_profile')
            raw=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
            if len(raw)!=2 or any(not r['success'] or r['error'] or r['prompt_len']!=512 or r['output_tokens']!=128 for r in raw):
                raise RuntimeError('Profile request failure')
        finally:
            try:children=psutil.Process(proc.pid).children(recursive=True)
            except psutil.NoSuchProcess:children=[]
            for child in reversed(children):
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            done.set();thread.join()
            (folder/'end.json').write_text(json.dumps({'end_epoch':time.time()}))
        with (folder/'counters-installed-export.csv').open('x') as output:
            subprocess.run([str(READER),'--import',str(folder/'kernels.ncu-rep'),'--csv',
                            '--page','details','--print-metric-name','name','--print-details','all',
                            '--print-units','base'],stdout=output,stderr=subprocess.STDOUT,check=True,timeout=30)
        if len(parse(folder/'counters-installed-export.csv'))!=8:
            raise RuntimeError('Counter collection failed; no further capture')


if __name__=='__main__':main()
