"""Fixed vLLM 0.23.0 prototype: BF16 full cold attention, upstream FP8 storage/reuse.

No persistent BF16 duplicate, dequantize/requantize migration or kernel copy.
General plugin registration affects only the explicitly opted-in subprocess.
"""
from dataclasses import replace
import logging
import torch
import vllm
from .routes import sequence_routes
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend, TritonAttentionImpl, TritonAttentionMetadataBuilder,
    context_attention_fwd,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

log=logging.getLogger(__name__)

def register():
    if vllm.__version__!='0.23.0':
        raise RuntimeError('Hybrid prototype only audited for vLLM 0.23.0')
    register_backend(AttentionBackendEnum.TRITON_ATTN,'inference_service.backends.hybrid.HybridBackend')
    log.warning('HYBRID_REGISTER BF16 cold attention; FP8 persistent KV; vLLM 0.23.0')

class HybridBuilder(TritonAttentionMetadataBuilder):
    def build(self,common_prefix_len,common_attn_metadata,fast_build=False):
        metadata=super().build(common_prefix_len,common_attn_metadata,fast_build)
        starts=common_attn_metadata.query_start_loc_cpu.tolist()
        lengths=common_attn_metadata.seq_lens_cpu.tolist()
        metadata.hybrid_routes=sequence_routes(starts,lengths)
        if metadata.max_query_len>1:
            log.warning('HYBRID_SCHEDULE %s',metadata.hybrid_routes)
        return metadata

class HybridBackend(TritonAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return HybridImpl
    @staticmethod
    def get_builder_cls():
        return HybridBuilder

class HybridImpl(TritonAttentionImpl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.kv_cache_dtype!='fp8_per_token_head':
            raise ValueError('Hybrid requires upstream per-token-head FP8 storage')
        if self.alibi_slopes is not None or self.sinks is not None or self.logits_soft_cap!=0:
            raise ValueError('Hybrid prototype only supports plain Qwen3 causal attention')
        if self.sliding_window!=(-1,-1) or self.chunk_lookback!=-1:
            raise ValueError('Hybrid prototype requires full attention')
        self._reported_routes=set()

    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output,
                output_scale=None,output_block_scale=None):
        if attn_metadata is None:
            return super().forward(layer,query,key,value,kv_cache,attn_metadata,output,
                                   output_scale,output_block_scale)
        routes=attn_metadata.hybrid_routes
        if not any(route[3] for route in routes):
            if attn_metadata.max_query_len>1 and 'cached' not in self._reported_routes:
                log.warning('HYBRID_FORWARD cached_fp8 layer=%s',getattr(layer,'layer_name','unknown'))
                self._reported_routes.add('cached')
            return super().forward(layer,query,key,value,kv_cache,attn_metadata,output,
                                   output_scale,output_block_scale)
        if output_scale is not None or output_block_scale is not None:
            raise ValueError('Hybrid does not support output quantization')
        if query.dtype!=torch.bfloat16 or key.dtype!=torch.bfloat16 or value.dtype!=torch.bfloat16:
            raise ValueError('Original cold Q/K/V must be BF16')
        route_key='cold_long' if attn_metadata.max_query_len>=8192 else 'cold_short'
        if route_key not in self._reported_routes:
            log.warning('HYBRID_FORWARD %s_bf16 layer=%s',route_key,getattr(layer,'layer_name','unknown'))
            self._reported_routes.add(route_key)
        for i,(start,end,length,cold) in enumerate(routes):
            q=query[start:end]
            out=output[start:end]
            # One short device subtraction per sliced sequence; no per-layer CPU sync.
            starts=attn_metadata.query_start_loc[i:i+2]-start
            lens=attn_metadata.seq_lens[i:i+1]
            if cold:
                context_attention_fwd(q=q,k=key[start:end],v=value[start:end],o=out,
                    b_start_loc=starts[:1],b_seq_len=lens,max_input_len=end-start,
                    is_causal=True,softmax_scale=self.scale)
            else:
                single=replace(attn_metadata,num_actual_tokens=end-start,max_query_len=end-start,
                    query_start_loc=starts,max_seq_len=length,seq_lens=lens,
                    block_table=attn_metadata.block_table[i:i+1],
                    slot_mapping=attn_metadata.slot_mapping[start:end])
                super().forward(layer,q,key[start:end],value[start:end],kv_cache,single,out)
        return output
