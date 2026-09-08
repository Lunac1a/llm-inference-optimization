"""One max_num_seqs contrast; three paired rounds, alternating order."""
import json
from pathlib import Path
import runpy
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2'
batch=runpy.run_path(str(ROOT/'scripts/stage2-run.py'))['batch']
restart=runpy.run_path(str(ROOT/'scripts/stage2-service.py'))['restart']


if __name__=='__main__':
    selection=json.loads((OUT/'selection.json').read_text())
    values={4:[],8:[]}
    pairs=[]
    for r,order in enumerate([[4,8],[8,4],[4,8]],1):
        pair={}
        for cap in order:
            name=f'contrast-r{r}-cap{cap}'
            restart(cap,name)
            row=batch(name,selection['input'],selection['output'],8,32,cap)
            values[cap].append(row['output_throughput_tok_s'])
            pair[cap]=row['output_throughput_tok_s']
        pairs.append({'round':r,'cap4':pair[4],'cap8':pair[8],'gain_percent':(pair[8]/pair[4]-1)*100})
    groups={cap:{'mean':statistics.mean(v),'sample_cv_percent':statistics.stdev(v)/statistics.mean(v)*100,'values':v} for cap,v in values.items()}
    report={'groups':groups,'pairs':pairs,'stable':all(g['sample_cv_percent']<=5 for g in groups.values()),
            'note':'Attribution only. Both settings restarted each paired round; no confirmation rows reused; no contrast repeats authorized.'}
    (OUT/'contrast-summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
