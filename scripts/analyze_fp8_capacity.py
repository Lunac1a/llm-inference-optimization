"""Apply preregistered performance gates; human fact review remains separate."""
import json
from validate_fp8_capacity import OUT
from stage3_lib import write_json

def point_pass(row):
    for i,w in enumerate(row['waves']):
        if w['success']!=w['requests'] or w['deltas']['vllm:num_preemptions_total']!=0:
            return False
        if w['ttft']['p95']>(20 if i==0 else 2) or w['e2e']['p95']>(30 if i==0 else 8):
            return False
        if i and (w['cache']['hit_ratio'] is None or w['cache']['hit_ratio']<.90):
            return False
    return len(row['waves'])==3

def analyze(rows):
    points=[{'label':r['label'],'pass':point_pass(r)} for r in rows]
    def get(round_id,fp8,c):
        return next(r for r in rows if (r['round'],r['fp8'],r['concurrency'])==(round_id,fp8,c))
    regressions=[]
    for round_id in (1,2):
        a,b=get(round_id,False,2),get(round_id,True,2)
        regressions.extend({'round':round_id,'wave':i,'e2e_regression_percent':100*(b['waves'][i]['e2e']['p95']/a['waves'][i]['e2e']['p95']-1)} for i in (1,2))
    pattern=all(point_pass(get(r,False,2)) and not point_pass(get(r,False,3))
        and point_pass(get(r,True,2)) and point_pass(get(r,True,3)) for r in (1,2))
    return {'points':points,'c2_regressions':regressions,'capacity_pattern_pass':pattern,
        'performance_gates_pass':pattern and all(r['e2e_regression_percent']<=15 for r in regressions),
        'human_fact_review_and_capacity_attribution_required':True}

if __name__=='__main__':
    result=analyze(json.loads((OUT/'measurements.json').read_text()))
    write_json(OUT/'analysis.json',result)
    print(json.dumps(result,indent=2))
