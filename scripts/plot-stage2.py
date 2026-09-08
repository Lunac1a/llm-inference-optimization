"""Standalone scientific figures from observed data only (separate plot deps)."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.tmp/stage2-plot'))
if sys.platform=='linux': sys.path.insert(0,str(ROOT/'.tmp/stage2-plot-linux'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'artifacts/stage2'
rows=json.loads((OUT/'analysis.json').read_text())['rows']
fig,axes=plt.subplots(1,4,figsize=(16,4))
for workload in ['A','B','C']:
    points=sorted([r for r in rows if r['batch'].startswith(f'sweep-{workload}-')],key=lambda r:r['concurrency'])
    if not points: continue
    x=[r['concurrency'] for r in points]
    axes[0].plot(x,[r['output_throughput_tok_s'] for r in points],'-o',label=workload)
    for ax,metric in zip(axes[1:],['ttft','tpot','e2el']):
        line=ax.plot(x,[r[f'p95_{metric}_ms'] for r in points],'-o',label=f'{workload} p95')[0]
        ax.plot(x,[r[f'p50_{metric}_ms'] for r in points],'--',color=line.get_color(),label=f'{workload} p50')
for ax,title in zip(axes,['Output tokens/s','Client TTFT (ms)','Client TPOT (ms)','Client E2E (ms)']):
    ax.set(xlabel='Client concurrency',ylabel=title,xscale='log',xticks=[1,2,4,8,16])
    ax.set_xticklabels([1,2,4,8,16]);ax.grid(alpha=.25);ax.axvline(4,color='grey',linestyle=':');ax.legend(fontsize=8)
fig.suptitle('Qwen3-4B BF16 | Stage 2 cap4 | 16 measured requests/point | screening only')
fig.tight_layout();fig.savefig(OUT/'load-sweep.png',dpi=180);fig.savefig(OUT/'load-sweep.svg')
