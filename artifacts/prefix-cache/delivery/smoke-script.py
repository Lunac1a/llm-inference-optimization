import json,subprocess,sys,time,socket
from pathlib import Path
sys.path.insert(0,'scripts')
from local_document_qa import configuration,port_open
from stage3_lib import http_json,write_json,gpu_sample
root=Path('artifacts/prefix-cache')
assert json.loads((root/'analysis.json').read_text())['performance_and_cache_gates_passed']
assert json.loads((root/'answer-review.json').read_text())['all_passed']
budget=json.loads((root/'budget.json').read_text())
def ensure(seconds):
    assert time.time()+seconds+120<budget['deadline_epoch'],'Insufficient remaining budget'
ensure(180)
assert not port_open(configuration())
out=root/'delivery';out.mkdir(exist_ok=False)
command=[sys.executable,'scripts/local_document_qa.py','serve']
write_json(out/'command.json',{'serve':command})
log=(out/'server.log').open('w')
process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'checks':[]}
try:
    deadline=time.time()+180
    while time.time()<deadline:
        assert process.poll() is None,'Launcher exited before readiness'
        try:
            if http_json('http://127.0.0.1:8000/health',timeout=2)[0]==200:break
        except OSError:pass
        time.sleep(1)
    else:raise TimeoutError('CLI readiness deadline')
    for name,fact,nonstream in [('a','云桥甲-17',False),('b','澄海乙-29',True)]:
        ensure(65)
        args=[sys.executable,'scripts/local_document_qa.py','ask','--document',str(root/f'document-{name}.txt'),'--question','文档编号是什么？']
        if nonstream:args.append('--nonstream')
        r=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',timeout=65)
        write_json(out/f'ask-{name}.json',{'command':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        response=json.loads(r.stdout)
        passed=r.returncode==0 and response.get('stream_complete') and response.get('finish_reason')=='stop' and fact in response.get('text','')
        result['checks'].append({'document':name,'stream':not nonstream,'text':response.get('text'),'passed':bool(passed)})
        assert passed,'CLI answer failure'
finally:
    process.terminate()
    process.wait(timeout=75)
    log.close()
    result['launcher_returncode']=process.returncode
    result['port_closed']=not port_open(configuration())
    result['gpu']=gpu_sample()
    result['compute_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv'],text=True)
    result['budget_elapsed_minutes']=(time.time()-budget['started_epoch'])/60
    result['passed']=len(result['checks'])==2 and all(c['passed'] for c in result['checks']) and result['port_closed'] and process.returncode==0
    write_json(out/'acceptance.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
