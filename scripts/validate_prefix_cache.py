"""One frozen, three-pair local document prefix-cache comparison; no reruns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
import statistics
import time

from local_document_qa import configuration, document_prompt, ask, port_open
from stage3_collect import run_batch
from stage3_lib import ROOT, OwnedVllm, write_json, gpu_sample, metrics_snapshot

OUT = ROOT / 'artifacts/prefix-cache'
QUESTIONS = [
    ('文档编号是什么？', '云桥甲-17'),
    ('发布日期是什么？', '2026年5月12日'),
    ('项目负责人是谁？', '林若安'),
    ('正式预算金额是多少？', '480万元'),
    ('项目所在区域是哪里？', '墨尔本北区'),
    ('文档记载的项目风险是什么？', '供应商接口延迟'),
]
FOLLOWUPS = [q for q, _ in QUESTIONS] + ['概括该项目的正式流程。', '概括项目的负责人、预算和风险。']


class Budget:
    def __init__(self):
        self.started = time.time()
        self.deadline = self.started + 30 * 60
        self.events = []
        self.record('start')

    def record(self, event, **details):
        self.events.append({'time': time.time(), 'event': event, **details})
        write_json(OUT / 'budget.json', {'started_epoch': self.started,
            'deadline_epoch': self.deadline, 'elapsed_seconds': time.time()-self.started,
            'budget_minutes': 30, 'events': self.events})

    def ensure(self, batch_timeout=180, cleanup_reserve=120):
        if time.time()+batch_timeout+cleanup_reserve > self.deadline:
            raise RuntimeError('30-minute budget: insufficient time for next operation and cleanup')


def counter(snapshot: dict, name: str) -> float:
    if snapshot['status'] != 200:
        raise ValueError('Missing metrics snapshot')
    return sum(row['value'] for row in snapshot['selected'] if row['name'] == name)


def cache_delta(summary):
    before, after = summary['metrics_before'], summary['metrics_after']
    delta = {name: counter(after, 'vllm:prefix_cache_'+name+'_total') -
                   counter(before, 'vllm:prefix_cache_'+name+'_total')
             for name in ['queries', 'hits']}
    delta['hit_ratio'] = delta['hits']/delta['queries'] if delta['queries'] else None
    return delta


def analyze(rows):
    pairs = []
    for index in range(1, 4):
        off, on = [next(r for r in rows if r['round']==index and r['prefix']==p) for p in (False, True)]
        reduction = 100*(1-on['followup']['ttft_seconds']['p95']/off['followup']['ttft_seconds']['p95'])
        regression = 100*(on['followup']['e2e_seconds']['p95']/off['followup']['e2e_seconds']['p95']-1)
        reuse = on['followup_cache']['hit_ratio']
        pairs.append({'round':index, 'followup_p95_ttft_reduction_percent':reduction,
                      'followup_p95_e2e_change_percent':regression,
                      'passes':reduction>=20 and regression<=15 and reuse is not None and reuse>=.90
                          and off['followup_cache']['hits']==0,
                      'off':off,'on':on})
    cv = {}
    for prefix in (False, True):
        values = [r['followup']['output_throughput_tok_s'] for r in rows if r['prefix']==prefix]
        cv[str(prefix)] = 100*statistics.stdev(values)/statistics.mean(values)
    return {'performance_and_cache_gates_passed':all(p['passes'] for p in pairs),
            'answer_review_required':True, 'throughput_cv_percent':cv, 'pairs':pairs}


def run():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite or rerun existing prefix-cache evidence')
    if port_open(configuration()):
        raise SystemExit('Port 8000 occupied; no experiment started')
    OUT.mkdir(parents=True)
    source = ROOT / 'artifacts/stage3-v2/materials.json'
    docs = json.loads(source.read_text(encoding='utf-8'))['documents']['16384']
    document, other = docs['doc-a']['text'], docs['doc-b']['text']
    write_json(OUT/'materials.json', {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'document':document, 'other_document':other, 'questions':QUESTIONS,
        'first_question':'概括本文档的项目情况。', 'followups':FOLLOWUPS})
    for name, text in [('document-a.txt',document),('document-b.txt',other)]:
        (OUT/name).write_text(text,encoding='utf-8')
    sources = ['scripts/local_document_qa.py','scripts/validate_prefix_cache.py',
               'scripts/stage3_lib.py','scripts/stage3_collect.py','scripts/stage3_request.py',
               'configs/local-document-qa.json','docs/prefix-cache-plan.md']
    write_json(OUT/'protocol.json', {'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'order':[[False,True],[True,False],[False,True]], 'rounds':3,'reruns':0,
        'concurrency':1,'performance_output_tokens':128,'budget_minutes':30})
    budget = Budget()
    rows = []
    try:
        for index, order in enumerate([(False,True),(True,False),(False,True)],1):
            for prefix in order:
                label = f'round-{index}-'+('on' if prefix else 'off')
                config = configuration(prefix)
                budget.ensure()
                server = OwnedVllm(config,OUT/'runs'/label)
                try:
                    budget.record('server_start',run=label)
                    server.start(timeout=180)
                    # Short unrelated warmups compile execution without priming the document.
                    run_batch(config,OUT/label,'warmups',[(str(i),'请写一句关于晴天的短句。') for i in range(2)],1,16,True,budget=budget)
                    first = run_batch(config,OUT/label,'first',[('first',document_prompt(document,'概括本文档的项目情况。'))],1,128,True,budget=budget)
                    follow = run_batch(config,OUT/label,'followup',[(f'follow-{i}',document_prompt(document,q)) for i,q in enumerate(FOLLOWUPS)],1,128,True,budget=budget)
                    item={'round':index,'prefix':prefix,'first':first,'followup':follow,
                          'first_cache':cache_delta(first),'followup_cache':cache_delta(follow),
                          'session_throughput_tok_s':(first['output_tokens_sum']+follow['output_tokens_sum'])/(first['wall_seconds']+follow['wall_seconds'])}
                    rows.append(item)
                    write_json(OUT/'rounds.json',rows)
                    print(json.dumps({'run':label,'first_ttft':first['ttft_seconds']['p95'],
                        'followup_p95_ttft':follow['ttft_seconds']['p95'],'cache':item['followup_cache']}),flush=True)
                    if index==1:
                        checks=[]
                        for question, fact in QUESTIONS:
                            budget.ensure(60,120)
                            response=ask(config,document,question)
                            checks.append({'question':question,'required_fact':fact,'response':response})
                            write_json(OUT/label/'answer-checks.json',checks)
                            if not response.get('stream_complete') or response.get('finish_reason')!='stop':
                                raise RuntimeError('Answer check transport/truncation failure')
                        switched=[]
                        for question,fact,stream in [('文档编号是什么？','澄海乙-29',False),('项目负责人是谁？','周明川',True)]:
                            budget.ensure(60,120)
                            response=ask(config,other,question,stream)
                            switched.append({'question':question,'required_fact':fact,'response':response})
                            write_json(OUT/label/'document-switch.json',switched)
                            if not response.get('stream_complete') or response.get('finish_reason')!='stop':
                                raise RuntimeError('Document-switch transport/truncation failure')
                finally:
                    stop=server.stop()
                    budget.record('server_stop',run=label,stop=stop)
                if port_open(config):
                    raise RuntimeError('Owned server did not release port')
        write_json(OUT/'analysis.json',analyze(rows))
    except BaseException as exc:
        write_json(OUT/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        budget.record('collection_finished',port_closed=not port_open(configuration()))
        write_json(OUT/'cleanup.json',{'port_closed':not port_open(configuration()),'gpu':gpu_sample()})


if __name__=='__main__':
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    run()
