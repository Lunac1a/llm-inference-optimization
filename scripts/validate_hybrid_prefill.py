"""Predeclared, isolated BF16 prefill / FP8 persistent-cache experiment."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from local_document_qa import configuration, document_prompt, port_open
from stage3_collect import run_batch
from stage3_lib import ROOT, OwnedVllm, write_json, gpu_sample
from validate_fp8_capacity import measured, hashes, QUESTIONS

OUT=ROOT/'artifacts/hybrid-prefill'
PLUGIN=ROOT/'experiments/hybrid_prefill'

def config(arm):
    if arm not in ('bf16','fp8','hybrid'):
        raise ValueError(arm)
    return {**configuration(),'attention_backend':'TRITON_ATTN',
            'kv_cache_dtype':'bfloat16' if arm=='bf16' else 'fp8_per_token_head',
            'max_model_len':16640,'max_num_batched_tokens':16640,'chunked_prefill':False}

def prompt(doc,question):
    # A different first content token prevents one shared instruction cache block.
    return doc['text'].split('。',1)[0]+'\n'+document_prompt(doc['text'],question)

class HybridServer(OwnedVllm):
    def _paths(self):
        python,_,headers=super()._paths()
        return python,str(ROOT/'scripts/hybrid_vllm_entry.py'),headers

@contextmanager
def launch_environment(arm,folder):
    env={'HYBRID_LAUNCH_EVIDENCE':str(folder/'actual-launch.json'),
         'PYTHONPATH':str(PLUGIN),
         'VLLM_PLUGINS':'inference_hybrid_prefill' if arm=='hybrid' else ''}
    previous={k:os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k,v in previous.items():
            if v is None:
                os.environ.pop(k,None)
            else:
                os.environ[k]=v

def source_paths():
    paths=[ROOT/p for p in ['docs/hybrid-prefill-plan.md','scripts/validate_hybrid_prefill.py',
        'scripts/hybrid_vllm_entry.py','scripts/stage3_lib.py','scripts/stage3_collect.py',
        'scripts/stage3_request.py','scripts/local_document_qa.py','scripts/validate_fp8_capacity.py',
        'scripts/validate_prefix_cache.py','configs/local-document-qa.json']]
    return paths+[p for p in PLUGIN.rglob('*') if p.is_file() and '__pycache__' not in str(p)]

def prepare():
    if OUT.exists():
        raise RuntimeError('Refuse overwrite')
    import vllm, torch
    from transformers import AutoTokenizer
    assert vllm.__version__=='0.23.0'
    source=ROOT/'artifacts/fp8-capacity/materials.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    model=OwnedVllm(config('bf16'),OUT/'unused').resolve_model()
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    toks={k:[tokenizer.apply_chat_template([{'role':'user','content':prompt(d,q)}],
              tokenize=True,add_generation_prompt=True,enable_thinking=False)
              for q in ['概括本文档的项目情况。',*QUESTIONS]] for k,d in data['documents'].items()}
    def common(a,b):
        return sum(1 for _ in itertools.takewhile(lambda x:x[0]==x[1],zip(a,b)))
    cross={a+':'+b:common(toks[a][0],toks[b][0]) for a,b in itertools.combinations(toks,2)}
    same={k:[common(v[0],x) for x in v[1:]] for k,v in toks.items()}
    assert all(n<16 for n in cross.values()),cross
    assert all(n>=16384*.99 for values in same.values() for n in values),same
    assert all(len(t)+128<=16640 for values in toks.values() for t in values)
    old=hashes(p for p in (ROOT/'artifacts').rglob('*') if p.is_file())
    OUT.mkdir()
    write_json(OUT/'old-evidence-hashes.json',old)
    write_json(OUT/'materials.json',data)
    package=Path(vllm.__file__).parent
    package_files=['v1/attention/backends/triton_attn.py','v1/attention/ops/triton_prefill_attention.py',
        'v1/attention/ops/triton_reshape_and_cache_flash.py','v1/attention/backends/registry.py',
        'plugins/__init__.py']
    write_json(OUT/'preflight.json',{'vllm':vllm.__version__,'torch':torch.__version__,
        'source_material_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'package_files':{str(package/p):hashlib.sha256((package/p).read_bytes()).hexdigest() for p in package_files},
        'package_sources':{p:(package/p).read_text() for p in package_files},
        'cross_prefix_tokens':cross,'same_prefix_tokens':same,
        'full_prompt_tokens':{k:list(map(len,v)) for k,v in toks.items()},
        'configs':{a:config(a) for a in ('bf16','fp8','hybrid')}})
    print('Material and version preflight passed',flush=True)

class Budget:
    def __init__(self):
        self.start=time.time()
        self.deadline=self.start+1800
        self.events=[]
        self.record('start')
    def record(self,event,**details):
        self.events.append({'time':time.time(),'event':event,**details})
        write_json(OUT/'budget.json',{'started_epoch':self.start,'deadline_epoch':self.deadline,
            'elapsed_seconds':time.time()-self.start,'budget_minutes':30,'events':self.events})
    def ensure(self,batch_timeout=180,cleanup_reserve=120):
        if time.time()+batch_timeout+cleanup_reserve>self.deadline:
            raise RuntimeError('Budget exhausted')

def numerical_checks():
    import torch
    from torch.nn.functional import scaled_dot_product_attention
    sys.path.insert(0,str(PLUGIN))
    from hybrid_backend import HybridImpl, context_attention_fwd
    from hybrid_routes import sequence_routes
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl,TritonAttentionMetadata
    from types import SimpleNamespace
    torch.manual_seed(42)
    results=[]
    def compare(name,actual,expected):
        diff=(actual.float()-expected.float()).abs()
        item={'name':name,'max_abs':diff.max().item(),'rms':diff.square().mean().sqrt().item()}
        item['pass']=item['max_abs']<=.04 and item['rms']<=.005
        results.append(item)
        write_json(OUT/'numerical-checks.json',results)
        assert item['pass'],item
    def sdpa(q,k,v):
        return scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0),
            k.transpose(0,1).unsqueeze(0),v.transpose(0,1).unsqueeze(0),
            is_causal=True,enable_gqa=True).squeeze(0).transpose(0,1)
    for n in (64,256):
        q=torch.randn(n,32,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(n,8,128,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        out=torch.empty_like(q)
        context_attention_fwd(q,k,v,out,torch.tensor([0],device='cuda',dtype=torch.int32),
                              torch.tensor([n],device='cuda',dtype=torch.int32),n)
        compare('cold-'+str(n),out,sdpa(q,k,v))
    # One FP8 cache containing a full cold sequence and another sequence's prefix.
    impl=HybridImpl(32,128,128**-.5,8,None,None,'fp8_per_token_head')
    native=TritonAttentionImpl(32,128,128**-.5,8,None,None,'fp8_per_token_head')
    layer=SimpleNamespace(_k_scale=torch.ones((),device='cuda'),_v_scale=torch.ones((),device='cuda'),layer_name='numerical')
    cache=torch.zeros((7,2,16,8,132),dtype=torch.uint8,device='cuda')
    q=torch.randn(68,32,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(100,8,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    slots=torch.arange(100,device='cuda',dtype=torch.int64)
    # Each instance's scale-view initialization fills scales: initialize both BEFORE writing.
    impl._ensure_scale_caches(cache)
    native._ensure_scale_caches(cache)
    impl.do_kv_cache_update(layer,k,v,cache,slots)
    actual_k=torch.cat((k[:64],k[96:100]))
    actual_v=torch.cat((v[:64],v[96:100]))
    meta=TritonAttentionMetadata(num_actual_tokens=68,max_query_len=64,
        query_start_loc=torch.tensor([0,64,68],device='cuda',dtype=torch.int32),max_seq_len=64,
        seq_lens=torch.tensor([64,36],device='cuda',dtype=torch.int32),
        block_table=torch.tensor([[0,1,2,3],[4,5,6,0]],device='cuda',dtype=torch.int32),
        slot_mapping=torch.cat((slots[:64],slots[96:100])),seq_threshold_3D=16,num_par_softmax_segments=16,
        softmax_segm_output=torch.empty((16,32,16,128),device='cuda',dtype=torch.float32),
        softmax_segm_max=torch.empty((16,32,16),device='cuda',dtype=torch.float32),
        softmax_segm_expsum=torch.empty((16,32,16),device='cuda',dtype=torch.float32),
        use_cascade=False,common_prefix_len=0,cu_prefix_query_lens=None,prefix_kv_lens=None,suffix_kv_lens=None)
    meta.hybrid_routes=sequence_routes([0,64,68],[64,36])
    before=cache.clone()
    out=torch.empty_like(q)
    impl.forward(layer,q,actual_k,actual_v,cache,meta,out)
    ref=torch.empty_like(q)
    native.forward(layer,q,actual_k,actual_v,cache,meta,ref)
    compare('mixed-cold',out[:64],sdpa(q[:64],k[:64],v[:64]))
    compare('mixed-cached',out[64:],ref[64:])
    assert torch.equal(before,cache),'Attention read mutated cache'
    write_json(OUT/'numerical-checks.json',{'comparisons':results,'cache_bytes_unchanged':True})
    print('Numerical gate passed',flush=True)

def run():
    if (OUT/'budget.json').exists():
        raise RuntimeError('Refuse existing GPU evidence')
    if port_open(config('bf16')):
        raise RuntimeError('Port occupied')
    data=json.loads((OUT/'materials.json').read_text(encoding='utf-8'))
    write_json(OUT/'protocol.json',{'sources':hashes(source_paths()),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    budget=Budget()
    rows=[]
    try:
        # A subprocess frees its CUDA context/allocations before server profiling.
        result=subprocess.run([sys.executable,__file__,'numerical'],capture_output=True,text=True,timeout=120)
        (OUT/'numerical.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:
            raise RuntimeError('Numerical checks failed; see numerical.log')
        for round_id,order in [(1,['bf16','fp8','hybrid']),(2,['hybrid','fp8','bf16'])]:
            for arm in order:
                label=f'r{round_id}-{arm}'
                folder=OUT/label
                folder.mkdir()
                cfg=config(arm)
                server=HybridServer(cfg,folder/'server')
                budget.ensure()
                try:
                    print('START '+label,flush=True)
                    with launch_environment(arm,folder):
                        server.start(timeout=180)
                    run_batch(cfg,folder,'warmup',[(str(i),'请写一句关于晴天的短句。') for i in range(2)],1,16,True,budget=budget)
                    row={'round':round_id,'arm':arm,'label':label,'waves':[]}
                    for wave,q in enumerate(['概括本文档的项目情况。',*QUESTIONS[:2]]):
                        name=f'perf-{wave}'
                        summary=run_batch(cfg,folder,name,[(k,prompt(d,q)) for k,d in data['documents'].items()],3,128,True,budget=budget)
                        item=measured(summary,folder/name)
                        row['waves'].append(item)
                        write_json(folder/'measurement.json',row)
                        print(json.dumps({'run':label,'wave':wave,**item}),flush=True)
                    if arm=='hybrid':
                        log=(folder/'server/server.log').read_text(errors='replace')
                        if 'HYBRID_FORWARD cold_long_bf16' not in log or 'HYBRID_FORWARD cached_fp8' not in log:
                            raise RuntimeError('Required hybrid route not executed')
                    if round_id==1:
                        for i,q in enumerate(QUESTIONS):
                            name=f'quality-{i}'
                            run_batch(cfg,folder,name,[(k,prompt(d,q)) for k,d in data['documents'].items()],3,128,False,budget=budget)
                            answers=json.loads((folder/name/'requests.json').read_text())
                            if any(r['finish_reason']!='stop' for r in answers):
                                raise RuntimeError('Quality truncation')
                    rows.append(row)
                    write_json(OUT/'measurements.json',rows)
                finally:
                    stopped=server.stop()
                    budget.record('server_stop',run=label,stop=stopped)
                if port_open(cfg):
                    raise RuntimeError('Owned port not released')
        print('COLLECTION COMPLETE',flush=True)
    except BaseException as exc:
        write_json(OUT/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        budget.record('collection_finished')
        write_json(OUT/'cleanup.json',{'port_closed':not port_open(config('bf16')),'gpu':gpu_sample(),
            'compute_processes':subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv'],capture_output=True,text=True).stdout})

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','run','numerical'])
    {'prepare':prepare,'run':run,'numerical':numerical_checks}[parser.parse_args().phase]()
