"""Apply predeclared hybrid gates without modifying raw measurements."""
import json
from validate_hybrid_prefill import OUT
from stage3_lib import write_json

def point_pass(row):
    if len(row['waves'])!=3:
        return False
    for i,w in enumerate(row['waves']):
        if w['success']!=w['requests'] or w['deltas']['vllm:num_preemptions_total']!=0:
            return False
        if w['ttft']['p95']>(20 if i==0 else 2) or w['e2e']['p95']>(30 if i==0 else 8):
            return False
        if i and (w['cache']['hit_ratio'] is None or w['cache']['hit_ratio']<.9):
            return False
    return True

def analyze(rows):
    if len(rows)!=6:
        raise ValueError('Six complete runs required')
    comparisons=[]
    for round_id in (1,2):
        fp8=next(r for r in rows if r['round']==round_id and r['arm']=='fp8')
        hybrid=next(r for r in rows if r['round']==round_id and r['arm']=='hybrid')
        reduction=100*(1-hybrid['waves'][0]['ttft']['p95']/fp8['waves'][0]['ttft']['p95'])
        regression=[100*(hybrid['waves'][i]['e2e']['p95']/fp8['waves'][i]['e2e']['p95']-1) for i in (1,2)]
        comparisons.append({'round':round_id,'cold_ttft_reduction_percent':reduction,
            'followup_e2e_changes_percent':regression,
            'pass':point_pass(hybrid) and reduction>=20 and all(v<=15 for v in regression)})
    return {'points':[{'run':r['label'],'pass':point_pass(r)} for r in rows],
        'comparisons':comparisons,'performance_gates_pass':all(c['pass'] for c in comparisons),
        'source_aligned_answer_review_required':True,'default_profile_changed':False}

if __name__=='__main__':
    result=analyze(json.loads((OUT/'measurements.json').read_text(encoding='utf-8')))
    write_json(OUT/'analysis.json',result)
    print(json.dumps(result,indent=2))
