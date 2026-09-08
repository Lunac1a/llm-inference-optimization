"""Execute only predeclared phases; stop on any batch failure."""
import argparse
import json
from pathlib import Path
import runpy
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/stage2'
batch = runpy.run_path(str(ROOT/'scripts/stage2-run.py'))['batch']


def confirmation(selection, prefix):
    points=selection['points']
    values={c:[] for c in points}
    for r in range(3):
        for c in points[r:]+points[:r]:
            row=batch(f'{prefix}-r{r+1}-c{c}',selection['input'],selection['output'],c,32)
            values[c].append(row['output_throughput_tok_s'])
    summary={c:{'values':v,'mean':statistics.mean(v),'sample_cv_percent':statistics.stdev(v)/statistics.mean(v)*100} for c,v in values.items()}
    passed=all(v['sample_cv_percent']<=5 for v in summary.values())
    (OUT/f'{prefix}-stability.json').write_text(json.dumps({'passed':passed,'groups':summary},indent=2))
    print(json.dumps({'passed':passed,'groups':summary}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['sweep','confirm','confirm-repeat']);a=p.parse_args()
    if not json.loads((OUT/'readiness-recovery.json').read_text())['passed']:
        raise SystemExit('Readiness failed')
    if a.phase=='sweep':
        for w,i,o in [('A',512,128),('B',2048,128),('C',512,512)]:
            for c in [1,2,4,8,16]: batch(f'sweep-{w}-c{c}',i,o,c)
    else:
        confirmation(json.loads((OUT/'selection.json').read_text()),a.phase)
