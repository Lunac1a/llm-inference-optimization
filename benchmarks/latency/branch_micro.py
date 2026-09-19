"""Compare frozen original, default-off refactor and split-source candidate."""
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / '.local/research/mixed-branch-2026-09-18'
sys.path.insert(0, str(ROOT / 'candidate-source'))
from inference_service.backends import dual_source_kernel as dual
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.kv_cache_interface import KVQuantMode
spec = importlib.util.spec_from_file_location('frozen_dual', ROOT / 'baseline-source/inference_service/backends/dual_source_kernel.py')
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
CASES = [(0,17,8,8,64), (1,2,32,8,128), (7,1,32,8,128), (31,19,32,8,128),
         (32,33,32,8,128), (33,32,16,4,64), (2049,37,32,8,128),
         (4093,129,32,8,128), (4096,2048,32,8,128), (12289,129,32,8,128)]
SHAPES = [(h,2048,32,8,128) for h in [2048,8192,14336]]

class Capture:
    def __init__(self, kernel): self.kernel = kernel; self.last = None
    def __getitem__(self, grid):
        call = self.kernel[grid]
        def invoke(**kwargs):
            self.last = call(**kwargs)
            return self.last
        return invoke
original.kernel_unified_attention = Capture(original.kernel_unified_attention)
dual.kernel_unified_attention = Capture(dual.kernel_unified_attention)

def setup(case, seed=42):
    history,current,heads,kvheads,dim=case;length=history+current
    torch.manual_seed(seed);pages=(length+15)//16
    cache=torch.zeros((pages+1,2,16,kvheads,dim+4),dtype=torch.uint8,device='cuda')
    impl=TritonAttentionImpl(heads,dim,dim**-.5,kvheads,None,None,'fp8_per_token_head')
    k=torch.randn(length,kvheads,dim,dtype=torch.bfloat16,device='cuda')
    v=torch.randn(length,kvheads*3,dim,dtype=torch.bfloat16,device='cuda')[:,kvheads:2*kvheads]
    q=torch.randn(current,heads,dim,dtype=torch.bfloat16,device='cuda')
    table=torch.arange(pages,0,-1,device='cuda',dtype=torch.int32)[None,:]
    idx=torch.arange(length,device='cuda');slots=table[0,idx//16].long()*16+idx%16
    impl.do_kv_cache_update(None,k,v,cache,slots)
    kc,vc=[x.view(torch.float8_e4m3fn) for x in cache.unbind(1)]
    out=torch.empty_like(q)
    kwargs=dict(q=q,k=kc,v=vc,out=out,cu_seqlens_q=torch.tensor([0,current],device='cuda',dtype=torch.int32),
        max_seqlen_q=current,seqused_k=torch.tensor([length],device='cuda',dtype=torch.int32),
        max_seqlen_k=length,softmax_scale=dim**-.5,causal=True,window_size=(-1,-1),block_table=table,
        softcap=0,q_descale=None,k_descale=None,v_descale=None,kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
        k_scale_cache=impl._k_scale_cache,v_scale_cache=impl._v_scale_cache,current_key=k[history:],current_value=v[history:])
    return kwargs,(k,v,slots,impl,case)

def reference(kwargs,state):
    k,v,slots,impl,case=state;history,current,heads,kvheads,dim=case
    hk=kwargs['k'].float()[slots[:history]//16,slots[:history]%16,:,:dim]*impl._k_scale_cache[slots[:history]//16,slots[:history]%16,:,None]
    hv=kwargs['v'].float()[slots[:history]//16,slots[:history]%16,:,:dim]*impl._v_scale_cache[slots[:history]//16,slots[:history]%16,:,None]
    rk=torch.cat([hk,k[history:].float()]).repeat_interleave(heads//kvheads,dim=1).transpose(0,1)
    rv=torch.cat([hv,v[history:].float()]).repeat_interleave(heads//kvheads,dim=1).transpose(0,1)
    refs=[]
    for start in range(0,current,32):
        end=min(start+32,current)
        scores=kwargs['q'][start:end].float().transpose(0,1)@rk.transpose(1,2)*dim**-.5
        mask=torch.arange(history+current,device='cuda')[None,:]>(history+torch.arange(start,end,device='cuda'))[:,None]
        refs.append((scores.masked_fill(mask[None],-torch.inf).softmax(-1)@rv).transpose(0,1))
    return torch.cat(refs)


def errors(out, ref):
    error = out.float() - ref.float()
    return {'finite': bool(torch.isfinite(out).all()), 'max_abs': error.abs().max().item(),
            'rms': error.square().mean().sqrt().item()}


def main():
    result = {'numerics': [], 'benchmarks': []}
    def save(): (ROOT / 'micro.json').write_text(json.dumps(result, indent=2))
    for seed in [42, 43]:
        for case in CASES:
            kwargs, state = setup(case, seed)
            k, v, slots, impl, _ = state
            before = [kwargs['k'].view(torch.uint8).clone(), kwargs['v'].view(torch.uint8).clone(), k.clone(), v.clone()]
            ref = reference(kwargs, state)
            original.unified_attention(**kwargs); baseline = kwargs['out'].clone()
            dual.unified_attention(**kwargs); default = kwargs['out'].clone()
            dual.unified_attention(**kwargs, block_source_branch=True); candidate = kwargs['out'].clone()
            torch.cuda.synchronize()
            old_error, new_error = errors(baseline, ref), errors(candidate, ref)
            difference = errors(candidate, baseline)
            unchanged = all(torch.equal(a, b) for a, b in zip(before, [kwargs['k'].view(torch.uint8), kwargs['v'].view(torch.uint8), k, v]))
            row = {'seed': seed, 'case': case, 'original_reference': old_error, 'candidate_reference': new_error,
                   'candidate_original': difference, 'default_equals_original': torch.equal(default, baseline),
                   'inputs_unchanged': unchanged}
            row['passed'] = (row['default_equals_original'] and unchanged
                and all(e['finite'] and e['max_abs'] <= .02 and e['rms'] <= .002 for e in [old_error, new_error])
                and difference['finite'] and difference['max_abs'] <= .002 and difference['rms'] <= .0002)
            result['numerics'].append(row); save(); print(json.dumps(row), flush=True)
            assert row['passed'], 'Numerical gate failed; no retry'
            del kwargs, state, k, v, ref, baseline, default, candidate, before
    for case in SHAPES:
        kwargs, state = setup(case)
        def call(arm):
            if arm == 'original': original.unified_attention(**kwargs)
            else: dual.unified_attention(**kwargs, block_source_branch=True)
        for _ in range(5):
            call('original'); call('split')
        torch.cuda.synchronize()
        times = {'original': [], 'split': []}
        for rep in range(20):
            for arm in (['original', 'split'] if rep % 2 == 0 else ['split', 'original']):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5): call(arm)
                end.record(); end.synchronize()
                times[arm].append(start.elapsed_time(end) / 5)
        resources = {}
        for arm, module in [('original', original), ('split', dual)]:
            compiled = module.kernel_unified_attention.last
            resources[arm] = {'registers': compiled.n_regs, 'spills': compiled.n_spills,
                'shared_bytes': compiled.metadata.shared, 'num_warps': compiled.metadata.num_warps}
            (ROOT / f'{arm}-{case[0]}.ptx').write_text(compiled.asm['ptx'])
        medians = {a: statistics.median(t) for a, t in times.items()}
        row = {'case': case, 'samples_ms': times, 'medians_ms': medians,
               'ratio': medians['split'] / medians['original'], 'resources': resources}
        result['benchmarks'].append(row); save(); print(json.dumps(row), flush=True)
        del kwargs, state
    ratios = [r['ratio'] for r in result['benchmarks']]
    gm = math.exp(statistics.mean(math.log(x) for x in ratios))
    result['gate'] = {'geomean_ratio': gm, 'worst_ratio': max(ratios),
                      'passed': gm <= .97 and max(ratios) <= 1.03}
    save(); print(json.dumps(result['gate']), flush=True)


if __name__ == '__main__': main()
