"""Short profiles, separate from performance evidence; never install CUDA tools."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import threading
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2'
runner=runpy.run_path(str(ROOT/'scripts/stage2-run.py'))
restart=runpy.run_path(str(ROOT/'scripts/stage2-service.py'))['restart']


def post(endpoint):
    return urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/'+endpoint,method='POST'),timeout=120).status


if __name__=='__main__':
    selection=json.loads((OUT/'selection.json').read_text())
    for label,c,n in [('profile-low',1,2),('profile-knee',selection['points'][1],selection['points'][1])]:
        restart(4,label,True)
        runner['batch'](label+'-warmup',selection['input'],selection['output'],c,4)
        folder=OUT/label;folder.mkdir(exist_ok=False)
        env=dict(os.environ,STAGE1_CONFIG=str(ROOT/'configs/stage2-cap4.env'),STAGE1_STATE_DIR=str(ROOT/'.tmp/stage2'),
                 STAGE2_BENCH='1',STAGE2_REQUEST_LOG=str(folder/'requests.jsonl'))
        cmd=['bash','scripts/stage2-profile-point.sh',str(selection['input']),str(selection['output']),str(c),str(folder),str(n)]
        (folder/'command.json').write_text(json.dumps({'argv':cmd,'max_engine_iterations':16,'requests':n,'cap':4},indent=2))
        done=threading.Event();thread=threading.Thread(target=runner['monitor'],args=(folder,done));thread.start()
        try:
            (folder/'start-status.txt').write_text(str(post('start_profile')))
            subprocess.run(cmd,env=env,cwd=ROOT,timeout=600,check=True)
        finally:
            try: (folder/'stop-status.txt').write_text(str(post('stop_profile')))
            finally: done.set();thread.join()
        requests=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
        if len(requests)!=n or any(not r['success'] or r['error'] or r['output_tokens']!=selection['output'] for r in requests):
            raise RuntimeError('Profile requests failed; stop')
