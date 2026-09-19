"""Read-only audit of completed collection and preserved pause evidence."""
import json
from common import ROOT, WORKSPACE, ARMS, LENGTHS, digest, read_jsonl, write_json

def main():
    for name in ['frozen-sources.json','resume-frozen.json']:
        for path,sha in json.loads((ROOT/name).read_text()).items():
            assert digest(WORKSPACE/path)==sha,path
    snapshot=json.loads((ROOT/'pause-snapshot-manifest.json').read_text())
    for path,sha in snapshot.items():assert digest(ROOT/path)==sha,path
    before=ROOT/'pause-snapshot/formal-4096-bf16_triton'
    current=ROOT/'runs/formal-4096-bf16_triton'
    original=read_jsonl(before/'results.jsonl')
    assert read_jsonl(current/'results.jsonl')[:len(original)]==original
    for path in before.rglob('*'):
        if path.is_file() and path.name!='results.jsonl':
            assert digest(path)==digest(current/path.relative_to(before)),path
    n=json.loads((ROOT/'formal-selection.json').read_text())['n']
    data={r['id']:r for p in (ROOT/'data').glob('*/*.jsonl') for r in read_jsonl(p) if r['sample']<n}
    count=0
    for length in LENGTHS:
        expected={key for key,r in data.items() if r['target_length']==length}
        for arm in ARMS:
            folder=ROOT/'runs'/f'formal-{length}-{arm}'
            summary=json.loads((folder/'summary.json').read_text())
            rows=read_jsonl(folder/'results.jsonl')
            assert len(rows)==summary['requests']==len(expected)==65
            assert {r['id'] for r in rows}==expected
            for r in rows:
                assert r['cached_tokens']==r['cache_hit_delta']==r['preemptions']==0
                assert r['usage']['prompt_tokens']==len(data[r['id']]['prompt_ids'])
                assert r['usage']['completion_tokens']<=data[r['id']]['max_tokens']
                assert r['finish_reason'] in ('stop','length')
                rawfolder=folder
                if length==4096 and arm=='bf16_triton' and r['id'] not in {x['id'] for x in original}:
                    rawfolder=ROOT/'runs/resume-4096-bf16_triton'
                raw=read_jsonl(rawfolder/'responses'/(r['id'].replace(':','-')+'.jsonl'))
                assert raw[-1]['data']=='[DONE]',r['id']
            count+=len(rows)
    ledger=read_jsonl(ROOT/'ledger.jsonl')
    ends=[r for r in ledger if r['event']=='end']
    partial=[r for r in ends if r['code'] is None]
    assert len(partial)==1 and partial[0]['length']==4096 and partial[0]['arm']=='bf16_triton'
    assert all(r['reason']=='completed' and r['code']==0 for r in ends if r not in partial)
    seconds=sum(r['seconds'] for r in ends)
    assert seconds<=7200
    write_json(ROOT/'final-validation.json',{'formal_responses':count,'frozen_hashes_verified':True,
        'original_31_responses_preserved':True,'snapshot_and_original_raw_files_verified':True,
        'sse_done_usage_cache_preemption_checks':True,'lifecycle_seconds':seconds,
        'user_interrupted_ledger_entries':len(partial),'completed_collection_segments':len(ends)-len(partial)})
    print('Verified 780 formal responses, frozen sources, preserved pause evidence and budget.')

if __name__=='__main__':main()
