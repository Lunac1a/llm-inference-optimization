"""Restart the owned Stage 2 server, optionally with bounded torch profiling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2'
STATE=ROOT/'.tmp/stage2'


def restart(cap, label, profile=False):
    saved=OUT/'server-command.json'
    if not saved.exists():
        pid=int((STATE/'vllm.pid').read_text())
        argv=[s.decode() for s in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if s]
        if 'serve' not in argv or not any('inference-vllm/bin/vllm' in s for s in argv):
            raise RuntimeError('Not the owned vLLM server')
        saved.write_text(json.dumps(argv,indent=2))
    argv=json.loads(saved.read_text())
    argv[argv.index('--max-num-seqs')+1]=str(cap)
    env=dict(os.environ,STAGE1_CONFIG=str(ROOT/f'configs/stage2-cap{cap}.env'),STAGE1_STATE_DIR=str(STATE),
             VLLM_USE_V2_MODEL_RUNNER='0',VLLM_USE_FLASHINFER_SAMPLER='0',
             C_INCLUDE_PATH=f'{ROOT}/.tmp/python-dev/extracted/usr/include/python3.12:{ROOT}/.tmp/python-dev/extracted/usr/include')
    (OUT/f'{label}-previous-server.log').write_bytes((STATE/'vllm-server.log').read_bytes())
    subprocess.run(['bash','scripts/stop-vllm-baseline.sh'],cwd=ROOT,env=env,check=True,timeout=45)
    if profile:
        trace=OUT/'traces'/label
        trace.mkdir(parents=True,exist_ok=False)
        argv+=['--profiler-config',json.dumps({'profiler':'torch','torch_profiler_dir':str(trace),
                'torch_profiler_with_stack':False,'torch_profiler_record_shapes':True,
                'ignore_frontend':False,'max_iterations':16})]
    (OUT/f'{label}-server-command.json').write_text(json.dumps({'argv':argv,'cap':cap,'profile':profile},indent=2))
    with (STATE/'vllm-server.log').open('w') as log:
        process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (STATE/'vllm.pid').write_text(str(process.pid)+'\n')
    for _ in range(120):
        if process.poll() is not None: raise RuntimeError('Server crashed on restart')
        try:
            urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).close()
            (OUT/f'{label}-startup.log').write_bytes((STATE/'vllm-server.log').read_bytes())
            return
        except OSError: time.sleep(1)
    raise RuntimeError('Server readiness timed out')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cap',type=int,choices=[4,8]);p.add_argument('label');p.add_argument('--profile',action='store_true');a=p.parse_args()
    restart(a.cap,a.label,a.profile)
