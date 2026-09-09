"""Frozen two-point capacity comparison; existing evidence is never overwritten."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import signal
import subprocess
import time
from pathlib import Path

from local_document_qa import configuration, document_prompt, port_open
from stage3_collect import run_batch
from stage3_lib import ROOT, OwnedVllm, write_json, gpu_sample
from validate_prefix_cache import counter, cache_delta

OUT = ROOT / 'artifacts/fp8-capacity'
QUESTIONS = ['文档编号是什么？', '项目负责人是谁，正式预算金额是多少？',
             '项目所在区域是哪里，文档记载的项目风险是什么？']

def config(fp8):
    return {**configuration(), 'attention_backend':'TRITON_ATTN',
            'kv_cache_dtype':'fp8_per_token_head' if fp8 else 'bfloat16'}

def hashes(paths):
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def prepare():
    if OUT.exists():
        raise RuntimeError('Refuse existing evidence directory')
    from transformers import AutoTokenizer
    import vllm, torch
    model = OwnedVllm(config(False), OUT/'unused').resolve_model()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    source = ROOT/'artifacts/stage3-v2/materials.json'
    docs = json.loads(source.read_text())['documents']['16384']
    selected = {k:docs[k] for k in ('doc-a','doc-b','doc-c')}
    tokens = {}
    for key, doc in selected.items():
        tokens[key] = [tokenizer.apply_chat_template([{'role':'user','content':document_prompt(doc['text'],q)}],
                       tokenize=True, add_generation_prompt=True, enable_thinking=False)
                       for q in ['概括本文档的项目情况。',*QUESTIONS]]
    def common(a,b):
        return sum(1 for _ in itertools.takewhile(lambda p:p[0]==p[1],zip(a,b)))
    cross = {a+':'+b:common(tokens[a][0],tokens[b][0]) for a,b in itertools.combinations(tokens,2)}
    same = {k:[common(t[0],x) for x in t[1:]] for k,t in tokens.items()}
    assert all(v<16384*.01 for v in cross.values()), cross
    assert all(v>=16384*.99 for values in same.values() for v in values), same
    assert all(len(x)+128<=32768 for t in tokens.values() for x in t)
    old = hashes(p for p in (ROOT/'artifacts').rglob('*') if p.is_file())
    OUT.mkdir()
    write_json(OUT/'old-evidence-hashes.json',old)
    write_json(OUT/'materials.json',{'documents':selected,'questions':QUESTIONS})
    for key,doc in selected.items():
        (OUT/(key+'.txt')).write_text(doc['text'],encoding='utf-8')
    backend_dir=Path(vllm.__file__).parent/'v1/attention/backends'
    backend_sources={name:(backend_dir/name).read_text() for name in ('flash_attn.py','triton_attn.py')}
    write_json(OUT/'preflight.json',{'vllm':vllm.__version__,'torch':torch.__version__,
        'capability':torch.cuda.get_device_capability(),'gpu':gpu_sample(),'model_path':model,
        'backend_sources':backend_sources,'cross_prefix_tokens':cross,'same_prefix_tokens':same,
        'full_prompt_tokens':{k:list(map(len,v)) for k,v in tokens.items()},
        'configs':[config(False),config(True)],'source_sha256':hashes([source]),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    assert vllm.__version__=='0.23.0'
    print('Preflight passed',flush=True)

class Budget:
    def __init__(self):
        self.start=time.time()
        self.deadline=self.start+1800
        self.record('start')
    def record(self,event,**details):
        write_json(OUT/'budget.json',{'started_epoch':self.start,'deadline_epoch':self.deadline,
            'elapsed_seconds':time.time()-self.start,'budget_minutes':30,'event':event,**details})
    def ensure(self,batch_timeout=180,cleanup_reserve=120):
        if time.time()+batch_timeout+cleanup_reserve>self.deadline:
            raise RuntimeError('Budget exhausted')

def measured(summary, directory):
    before,after=summary['metrics_before'],summary['metrics_after']
    names={r['name'] for r in after['selected']}
    required=['vllm:num_preemptions_total','vllm:request_queue_time_seconds_sum',
              'vllm:request_queue_time_seconds_count']
    if not all(n in names for n in required):
        raise RuntimeError('Required counters missing: '+str(names))
    delta={n:counter(after,n)-counter(before,n) for n in required}
    telemetry=[json.loads(line) for line in (directory/'telemetry.jsonl').read_text().splitlines()]
    gauges={}
    for name in ['vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc']:
        values=[r['value'] for s in telemetry for r in s.get('metrics',{}).get('selected',[]) if r['name']==name]
        if not values:
            raise RuntimeError('Missing gauge '+name)
        gauges[name]={'min':min(values),'max':max(values)}
    return {'ttft':summary['ttft_seconds'],'e2e':summary['e2e_seconds'],
            'throughput':summary['output_throughput_tok_s'],'success':summary['success_count'],
            'requests':summary['request_count'],'cache':cache_delta(summary),'deltas':delta,'gauges':gauges}

def run():
    if (OUT/'budget.json').exists():
        raise RuntimeError('Refuse rerun')
    if port_open(config(False)):
        raise RuntimeError('Port occupied')
    data=json.loads((OUT/'materials.json').read_text())
    sources=[ROOT/p for p in ['docs/fp8-capacity-plan.md','scripts/validate_fp8_capacity.py',
        'scripts/stage3_lib.py','scripts/stage3_collect.py','scripts/stage3_request.py',
        'scripts/local_document_qa.py','configs/local-document-qa.json']]
    write_json(OUT/'protocol.json',{'hashes':hashes(sources),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    budget=Budget()
    records=[]
    try:
        for round_id,order in [(1,[False,True]),(2,[True,False])]:
            for fp8 in order:
                for concurrency in (2,3):
                    cfg=config(fp8)
                    label=f'r{round_id}-'+('fp8' if fp8 else 'bf16')+f'-c{concurrency}'
                    folder=OUT/label
                    server=OwnedVllm(cfg,folder/'server')
                    docs=list(data['documents'].items())[:concurrency]
                    budget.ensure()
                    try:
                        print('START '+label,flush=True)
                        server.start(timeout=180)
                        run_batch(cfg,folder,'warmup',[(str(i),'请写一句关于晴天的短句。') for i in range(2)],1,16,True,budget=budget)
                        result={'round':round_id,'fp8':fp8,'concurrency':concurrency,'label':label,'waves':[]}
                        for wave,q in enumerate(['概括本文档的项目情况。',*QUESTIONS[:2]]):
                            name=f'perf-{wave}'
                            summary=run_batch(cfg,folder,name,[(k,document_prompt(d['text'],q)) for k,d in docs],concurrency,128,True,budget=budget)
                            item=measured(summary,folder/name)
                            result['waves'].append(item)
                            write_json(folder/'measurement.json',result)
                            print(json.dumps({'run':label,'wave':wave,**item}),flush=True)
                        if round_id==1 and concurrency==3:
                            for i,q in enumerate(QUESTIONS):
                                name=f'quality-{i}'
                                run_batch(cfg,folder,name,[(k,document_prompt(d['text'],q)) for k,d in docs],concurrency,128,False,budget=budget)
                                answers=json.loads((folder/name/'requests.json').read_text())
                                if any(r['finish_reason']!='stop' for r in answers):
                                    raise RuntimeError('Quality truncation')
                        records.append(result)
                        write_json(OUT/'measurements.json',records)
                    finally:
                        stopped=server.stop()
                        budget.record('server_stopped',run=label,stop=stopped)
                    if port_open(cfg):
                        raise RuntimeError('Owned port not released')
        print('COLLECTION COMPLETE',flush=True)
    except BaseException as exc:
        write_json(OUT/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        budget.record('collection_finished')
        write_json(OUT/'cleanup.json',{'port_closed':not port_open(config(False)),'gpu':gpu_sample(),
            'compute_processes':subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv'],capture_output=True,text=True).stdout})

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','run'])
    {'prepare':prepare,'run':run}[parser.parse_args().phase]()
