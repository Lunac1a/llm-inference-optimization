"""Trace event evidence; duration sums are explicitly not wall-clock shares."""
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2'


def union_duration(intervals):
    if not intervals: return 0
    start,end=sorted(intervals)[0];total=0
    for a,b in sorted(intervals)[1:]:
        if a>end: total+=end-start;start,end=a,b
        else: end=max(end,b)
    return total+end-start


reports=[]
for p in sorted((OUT/'traces').rglob('*.json.gz')):
    with gzip.open(p,'rt') as f: trace=json.load(f)
    events=[e for e in trace.get('traceEvents',[]) if e.get('ph')=='X' and e.get('dur',0)>0]
    cats=Counter(e.get('cat','') for e in events)
    kernels=[e for e in events if e.get('cat')=='kernel']
    cpu=[e for e in events if e.get('cat')=='cpu_op']
    top=[]
    for label,subset in [('kernel',kernels),('cpu_op',cpu)]:
        totals=Counter();counts=Counter()
        for e in subset:totals[e['name']]+=e['dur'];counts[e['name']]+=1
        top.append({'category':label,'events':[{'name':name,'summed_ms':duration/1000,'count':counts[name]} for name,duration in totals.most_common(25)]})
    intervals=[(e['ts'],e['ts']+e['dur']) for e in kernels]
    span=max(b for a,b in intervals)-min(a for a,b in intervals) if intervals else None
    transfers=[e for e in events if 'memcpy' in e.get('cat','').lower() or 'memcpy' in e.get('name','').lower()]
    windows=[]
    for e in events:
        if e.get('cat')!='gpu_user_annotation' or not e['name'].startswith('execute_context_'):continue
        a,b=e['ts'],e['ts']+e['dur']
        clipped=[(max(a,x),min(b,y)) for x,y in intervals if x<b and y>a]
        windows.append({'name':e['name'],'phase':'decode' if e['name'].startswith('execute_context_0(0)') else 'prefill_or_mixed',
                        'window_ms':e['dur']/1000,'kernel_union_ms':union_duration(clipped)/1000})
    phase_summary={}
    for phase in ['prefill_or_mixed','decode']:
        subset=[w for w in windows if w['phase']==phase]
        duration=sum(w['window_ms'] for w in subset);busy=sum(w['kernel_union_ms'] for w in subset)
        phase_summary[phase]={'windows':len(subset),'window_sum_ms':duration,'kernel_union_sum_ms':busy,
                              'non_kernel_fraction':1-busy/duration if duration else None}
    report={'file':str(p.relative_to(OUT)),'categories':dict(cats),'cuda_events_verified':bool(kernels),
            'kernel_span_ms':span/1000 if span else None,'kernel_union_ms':union_duration(intervals)/1000,
            'kernel_duration_sum_ms':sum(e['dur'] for e in kernels)/1000,'top_events':top,
            'transfer_events':len(transfers),'transfer_summed_ms':sum(e['dur'] for e in transfers)/1000,
            'transfer_by_category':{cat:{'count':sum(e.get('cat')==cat for e in transfers),'summed_ms':sum(e['dur'] for e in transfers if e.get('cat')==cat)/1000} for cat in {e.get('cat') for e in transfers}},
            'gpu_execution_windows':windows,'phase_summary':phase_summary,
            'limits':'Overlapping CPU ranges and kernels cannot be summed as wall-clock shares. No memory-bandwidth or SM efficiency counters. Kernel gaps may include dependency waits and host work.'}
    reports.append(report)
(OUT/'trace-analysis.json').write_text(json.dumps(reports,indent=2))
print(json.dumps([{'file':r['file'],'cuda':r['cuda_events_verified'],'kernel_span_ms':r['kernel_span_ms'],'kernel_union_ms':r['kernel_union_ms']} for r in reports],indent=2))
