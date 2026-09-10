"""CPU-only summary and closure of a bounded CUDA Graph trial."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from stage3_lib import ROOT, write_json, percentile
from validate_mixed_load import content_times
from finalize_mixed_load import verify
from local_document_qa import configuration, port_open

OUT = ROOT / 'artifacts/cuda-graph'


def read(p): return json.loads(p.read_text())


def main():
    runs, quality = [], []
    materials = read(OUT / 'materials.json')
    for folder in sorted(OUT.glob('r*-*')):
        log = (folder / 'server/server.log').read_text(errors='replace')
        match = re.search(r'GPU KV cache size: ([\d,]+) tokens', log)
        run = {'name': folder.name, 'gpu_kv_tokens': int(match[1].replace(',', '')) if match else None,
               'groups': {}, 'startup': read(folder / 'server/startup.json') if (folder / 'server/startup.json').exists() else None}
        for c in (1, 4):
            rows = []
            for p in sorted(folder.glob(f'c{c}-*.json')):
                raw = read(p)
                if not raw.get('done') or raw.get('error'): continue
                times = content_times(raw)
                gaps = [b-a for a,b in zip(times,times[1:])]
                rows.append({'name': raw['name'], 'tokens': raw['token_count'], 'ttft': raw['ttft'], 'e2e': raw['e2e'],
                    'start': raw['start'], 'end': raw['end'], 'output_tok_s': raw['output_tok_s'],
                    'client_post_first_rate': (raw['token_count']-1)/(times[-1]-times[0]),
                    'max_content_gap': max(gaps), 'p95_content_gap': percentile(gaps,95),
                    'usage': raw['usage'], 'finish_reason': raw['finish_reason']})
            if rows:
                run['groups'][str(c)] = {'requests': rows, 'count': len(rows),
                    'post_first_rate_median': percentile([r['client_post_first_rate'] for r in rows],50),
                    'ttft_p95': percentile([r['ttft'] for r in rows],95),
                    'e2e_p95': percentile([r['e2e'] for r in rows],95),
                    'wave_output_tok_s': sum(r['tokens'] for r in rows)/(max(r['end'] for r in rows)-min(r['start'] for r in rows))}
        run['runtime'] = read(folder / 'runtime-evidence.json') if (folder / 'runtime-evidence.json').exists() else None
        tele = folder / 'telemetry.jsonl'
        if tele.exists():
            ts = [json.loads(line) for line in tele.read_text().splitlines()]
            run['telemetry'] = {'max_sum_rss_bytes': max(t['rss']['sum_rss_bytes'] for t in ts),
                'min_mem_available': min(t['memory']['MemAvailable'] for t in ts),
                'max_swap_used': max(t['memory']['swap_used'] for t in ts)}
        runs.append(run)
        for p in sorted(folder.glob('quality-*.json')):
            raw = read(p)
            key = p.stem.removeprefix('quality-')
            doc = materials['quality'][key]
            correct = all(doc['facts'][k] in raw['text'] and doc['facts'][k] in doc['text'] for k in ('owner','budget'))
            foreign = any(d['facts'][k] in raw['text'] for other,d in materials['quality'].items() if other!=key for k in ('owner','budget'))
            quality.append({'run': folder.name, 'doc': key, 'text': raw['text'],
                'pass': correct and not foreign and raw['finish_reason']=='stop'})
    pairs=[]
    by={r['name']:r for r in runs}
    for n in range(1,4):
        if any(f'r{n}-{m}' not in by for m in ('graph','eager')): continue
        g,e=[by[f'r{n}-{m}'] for m in ('graph','eager')]
        checks={}
        for c,count in (('1',3),('4',4)):
            if c not in g['groups'] or c not in e['groups']: continue
            a,b=g['groups'][c],e['groups'][c]
            rate=a['post_first_rate_median']/b['post_first_rate_median']-1
            e2e=1-a['e2e_p95']/b['e2e_p95']
            checks[c]={'rate_gain':rate,'e2e_reduction':e2e, 'pass':rate>=.1 and e2e>=.05
                and a['ttft_p95']<=max(b['ttft_p95']*1.2,b['ttft_p95']+.05) and a['count']==b['count']==count}
        kv=bool(g['gpu_kv_tokens'] and e['gpu_kv_tokens'] and g['gpu_kv_tokens']>=.9*e['gpu_kv_tokens'])
        pairs.append({'round':n,'groups':checks,'kv_capacity_pass':kv,
            'pass':len(checks)==2 and all(c['pass'] for c in checks.values()) and kv and bool(g['runtime'] and e['runtime'])})
    accepted=len(pairs)==3 and all(p['pass'] for p in pairs) and len(quality)==6 and all(q['pass'] for q in quality) and not (OUT/'failure.json').exists()
    write_json(OUT/'analysis.json',{'accepted':accepted,'runs':runs,'pairs':pairs,'quality':quality,
        'performance_requests':sum(g['count'] for r in runs for g in r['groups'].values()),
        'timing_scope':'client SSE content arrival, not precise token generation; fixed 512 tokens'})
    tests=[]
    for pattern in ('test_cuda_graph.py','test_kv_offload_integration.py','test_mixed_load.py','test_hybrid_prefill.py'):
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p',pattern],capture_output=True,text=True,timeout=60)
        (OUT/(pattern+'.log')).write_text(p.stdout+p.stderr)
        tests.append({'pattern':pattern,'returncode':p.returncode})
    closure={'old_evidence':verify(read(OUT/'old-evidence-hashes.json')),
        'frozen_sources':verify(read(OUT/'frozen-source-hashes.json')),
        'installed_sources':verify(read(OUT/'preflight.json')['package_hashes'],True),'tests':tests,
        'port_closed':not port_open(configuration()), 'compute_processes':subprocess.run(['nvidia-smi',
        '--query-compute-apps=pid,process_name','--format=csv'],capture_output=True,text=True).stdout,'accepted':accepted,'gpu_retry':False}
    write_json(OUT/'closure.json',closure)
    assert all(closure[k]['pass'] for k in ('old_evidence','frozen_sources','installed_sources'))
    assert all(t['returncode']==0 for t in tests)
    assert closure['port_closed'] and closure['compute_processes'].strip()=='pid, process_name'
    write_json(OUT/'sha256-manifest.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.rglob('*') if p.is_file() and p.name!='sha256-manifest.json'})
    print(json.dumps({'closure':closure,'pairs':pairs},indent=2))


if __name__=='__main__':main()
