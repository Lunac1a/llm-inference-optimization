"""Finish the user-paused arm in a separate evidence segment."""
import asyncio
import json
import os
import re
import time
from unittest.mock import patch
import collect as collector
from common import ROOT, WORKSPACE, MODEL, read_jsonl, write_json

ORIGINAL = ROOT/'runs/formal-4096-bf16_triton'
SEGMENT = ROOT/'runs/resume-4096-bf16_triton'

def remaining(rows, saved):
    ids = [r['id'] for r in saved]
    assert len(ids) == len(set(ids)), 'Duplicate saved responses'
    assert ids == [r['id'] for r in rows[:len(ids)]], 'Saved responses are not the fixed prefix'
    for r in saved:
        assert r['cached_tokens'] == r['cache_hit_delta'] == r['preemptions'] == 0
        assert r['finish_reason'] in ('stop', 'length')
    return rows[len(ids):]

def main():
    n=json.loads((ROOT/'formal-selection.json').read_text())['n']
    rows=[r for p in (ROOT/'data/4096').glob('*.jsonl') for r in read_jsonl(p) if r['sample']<n]
    rows.sort(key=lambda r:(r['sample'],r['task']))
    saved=read_jsonl(ORIGINAL/'results.jsonl')
    pending=remaining(rows,saved)
    assert len(rows)==65 and len(saved)==31 and len(pending)==34
    SEGMENT.mkdir(exist_ok=False)
    (SEGMENT/'responses').mkdir();(SEGMENT/'metrics').mkdir()
    settings=collector.Settings(profile='chunked-hybrid',model=MODEL,backend_url='http://127.0.0.1:8031',log_dir=SEGMENT/'engine')
    os.environ['PYTHON_DEV_HEADERS']=str(WORKSPACE/'.tmp/python-dev/extracted/usr/include/python3.12')
    start=time.perf_counter()
    with patch('inference_service.runtime.launch_spec',lambda s:collector.make_spec(s,'bf16_triton',SEGMENT)):
        with collector.owned_backend(settings):
            startup=time.perf_counter()-start
            write_json(SEGMENT/'startup.json',{'seconds':startup})
            asyncio.run(collector.collect(settings,SEGMENT,pending,'resume'))
    logs='\n'.join(p.read_text(errors='replace') for p in (SEGMENT/'engine').glob('*.log'))
    capacity=re.findall(r'GPU KV cache size: ([\d,]+) tokens',logs)
    assert capacity
    fresh=read_jsonl(SEGMENT/'results.jsonl')
    assert not remaining(rows,saved+fresh)
    summary={'requests':len(fresh),'startup_seconds':startup,'lifecycle_seconds':time.perf_counter()-start,
             'kv_capacity_tokens':int(capacity[-1].replace(',','')),'arm':'bf16_triton','phase':'resume','length':4096}
    write_json(SEGMENT/'summary.json',summary)
    # The original prefix and interrupted SSE remain immutable in the snapshot.
    with (ORIGINAL/'results.jsonl').open('a') as stream:
        for row in fresh: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    old_start=json.loads((ORIGINAL/'startup.json').read_text())['seconds']
    write_json(ORIGINAL/'summary.json',{**summary,'requests':len(rows),'phase':'formal',
        'startup_seconds':old_start+startup,'segments':['original','resume-4096-bf16_triton'],
        'lifecycle_seconds':59.692165563+summary['lifecycle_seconds'],
        'note':'31 original and 34 resumed responses; resumed raw SSE/metrics are in the sibling resume directory.'})

if __name__=='__main__': main()
