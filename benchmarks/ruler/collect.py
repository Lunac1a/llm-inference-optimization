"""One owned server, serial cold RULER requests, raw SSE and engine evidence."""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from unittest.mock import patch
import httpx
from common import ROOT, WORKSPACE, MODEL, ARMS, read_jsonl, write_json, append

sys.path.insert(0, str(WORKSPACE/'src'))
from inference_service.config import Settings
from inference_service.runtime import launch_spec, owned_backend

def make_spec(settings, arm, out):
    command, env = launch_spec(settings)
    command[command.index('--attention-backend')+1] = 'FLASH_ATTN' if arm=='bf16_flash' else 'TRITON_ATTN'
    command[command.index('--kv-cache-dtype')+1] = 'bfloat16' if arm.startswith('bf16') else 'fp8_per_token_head'
    env.update(PYTHONPATH=str(WORKSPACE/'src'), VLLM_KV_CACHE_LAYOUT='NHD', VLLM_SERVER_DEV_MODE='1',
               VLLM_PLUGINS='inference_chunked_prefill' if arm=='mixed' else '',
               INFERENCE_RUNTIME_PROFILE='chunked-hybrid' if arm=='mixed' else 'document')
    write_json(out/'launch.json', {'argv':command, 'env':{k:v for k,v in env.items() if k.startswith(('VLLM_','INFERENCE_RUNTIME_PROFILE','PYTHONPATH','C_INCLUDE_PATH'))}})
    return command,env

def counter(metrics, name):
    return sum(float(line.rsplit(' ',1)[1]) for line in metrics.splitlines()
               if line.startswith(name+'{') or line.startswith(name+' '))

async def collect(settings, out, rows, phase):
    async with httpx.AsyncClient(base_url=settings.backend_url, timeout=120, trust_env=False) as client:
        async def request(row, warmup=False):
            if not warmup:
                reset = await client.post('/reset_prefix_cache'); reset.raise_for_status()
            before=(await client.get('/metrics')).text
            start=time.perf_counter(); times=[]; parts=[]; usage=None; reason=None; done=False
            name='warmup' if warmup else row['id'].replace(':','-')
            raw=out/'responses'/f'{name}.jsonl'
            body={'model':settings.served_model,'prompt':row['prompt_ids'],'max_tokens':row['max_tokens'],
                  'temperature':0,'seed':42,'stream':True,'stream_options':{'include_usage':True}}
            async with asyncio.timeout(120):
                async with client.stream('POST','/v1/completions',json=body) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data:'):continue
                        elapsed=time.perf_counter()-start
                        data=line[5:].strip()
                        append(raw,{'elapsed':elapsed,'data':data})
                        if data=='[DONE]':done=True;continue
                        item=json.loads(data)
                        if item.get('error'):raise RuntimeError(item['error'])
                        if item.get('usage'):usage=item['usage']
                        for choice in item.get('choices',[]):
                            if choice.get('text'):
                                parts.append(choice['text']); times.append(elapsed)
                            if choice.get('finish_reason'):reason=choice['finish_reason']
            e2e=time.perf_counter()-start
            assert done and usage and reason in ('stop','length'), (done,usage,reason)
            assert usage['prompt_tokens']==len(row['prompt_ids']), (usage,len(row['prompt_ids']))
            assert 0<=usage['completion_tokens']<=row['max_tokens']
            after=(await client.get('/metrics')).text
            (out/'metrics'/f'{name}-before.txt').write_text(before)
            (out/'metrics'/f'{name}-after.txt').write_text(after)
            cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
            hits=counter(after,'vllm:prefix_cache_hits_total')-counter(before,'vllm:prefix_cache_hits_total')
            preemptions=counter(after,'vllm:num_preemptions_total')-counter(before,'vllm:num_preemptions_total')
            record={'id':row['id'],'task':row['task'],'target_length':row['target_length'],'sample':row['sample'],
                    'prediction':''.join(parts),'usage':usage,'finish_reason':reason,'e2e':e2e,
                    'ttft':times[0] if times else None,'content_times':times,
                    'max_content_gap':max((b-a for a,b in zip(times,times[1:])),default=0),
                    'post_first_rate':(usage['completion_tokens']-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else None,
                    'cached_tokens':cached,'cache_hit_delta':hits,'preemptions':preemptions}
            append(out/('warmup.jsonl' if warmup else 'results.jsonl'),record)
            assert warmup or (cached==0 and hits==0 and preemptions==0), record['id']
            return record
        warm={'id':'warmup','task':'warmup','target_length':0,'sample':-1,'prompt_ids':[1,2,3,4], 'max_tokens':1}
        await request(warm,True)
        for number,row in enumerate(rows,1):
            result=await request(row)
            print(json.dumps({'event':'request','phase':phase,'count':number,'id':row['id'],'seconds':round(result['e2e'],3),
                              'output_tokens':result['usage']['completion_tokens']}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--phase',choices=['pilot','formal'],required=True);parser.add_argument('--length',type=int,required=True)
    args=parser.parse_args()
    out=ROOT/'runs'/f'{args.phase}-{args.length}-{args.arm}';out.mkdir(parents=True,exist_ok=False)
    (out/'responses').mkdir();(out/'metrics').mkdir()
    n=json.loads((ROOT/'formal-selection.json').read_text())['n'] if args.phase=='formal' else None
    rows=[]
    for path in sorted((ROOT/'data'/str(args.length)).glob('*.jsonl')):
        rows += [r for r in read_jsonl(path) if (r['sample']==10 if args.phase=='pilot' else r['sample']<n)]
    assert len(rows)==13*(1 if args.phase=='pilot' else n)
    # Interleave tasks; fixed sample order across all arms, without reference to answers.
    rows.sort(key=lambda r:(r['sample'],r['task']))
    settings=Settings(profile='chunked-hybrid',model=MODEL,backend_url='http://127.0.0.1:8031',log_dir=out/'engine')
    os.environ['PYTHON_DEV_HEADERS']=str(WORKSPACE/'.tmp/python-dev/extracted/usr/include/python3.12')
    start=time.perf_counter()
    with patch('inference_service.runtime.launch_spec',lambda s:make_spec(s,args.arm,out)):
        with owned_backend(settings):
            startup=time.perf_counter()-start
            write_json(out/'startup.json',{'seconds':startup})
            asyncio.run(collect(settings,out,rows,args.phase))
    logs='\n'.join(path.read_text(errors='replace') for path in (out/'engine').glob('*.log'))
    capacity=re.findall(r'GPU KV cache size: ([\d,]+) tokens',logs)
    assert capacity,'KV capacity not found in actual server logs'
    if args.arm=='mixed':
        assert 'DUAL_SOURCE_REGISTER' in logs and 'route=dual_source' in logs and 'prefill_block_m=64' in logs
    write_json(out/'summary.json',{'requests':len(rows),'startup_seconds':startup,'lifecycle_seconds':time.perf_counter()-start,
        'kv_capacity_tokens':int(capacity[-1].replace(',','')),'arm':args.arm,'phase':args.phase,'length':args.length})

if __name__=='__main__':main()
