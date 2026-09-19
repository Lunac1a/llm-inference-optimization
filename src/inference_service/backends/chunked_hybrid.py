"""Default mixed prefill; original cold path and native decode retained."""
from dataclasses import replace
import logging
import torch
import vllm
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl, context_attention_fwd
from .hybrid import HybridBackend, HybridImpl
from .dual_source_kernel import PREFILL_BLOCK_M, unified_attention

log = logging.getLogger(__name__)
# Validated single-loop source branching for the mixed serving profile.
BLOCK_SOURCE_BRANCH = True


def register():
    if vllm.__version__ != '0.23.0':
        raise RuntimeError('Dual-source prefill is pinned to vLLM 0.23.0')
    register_backend(AttentionBackendEnum.TRITON_ATTN,
                     'inference_service.backends.chunked_hybrid.ChunkedHybridBackend')
    log.warning('DUAL_SOURCE_REGISTER vllm=0.23.0 history=FP8 current=BF16 decode=upstream prefill_block_m=%d block_source_branch=%s', PREFILL_BLOCK_M, BLOCK_SOURCE_BRANCH)


class ChunkedHybridBackend(HybridBackend):
    @staticmethod
    def get_impl_cls():
        return ChunkedHybridImpl


class ChunkedHybridImpl(HybridImpl):
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if not kv_cache.is_contiguous():
            raise ValueError('Dual-source FP8 requires contiguous NHD before cache writes')
        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if attn_metadata is None or attn_metadata.max_query_len <= 1:
            return TritonAttentionImpl.forward(self, layer, query, key, value, kv_cache,
                                                attn_metadata, output, output_scale, output_block_scale)
        if output_scale is not None or output_block_scale is not None:
            raise ValueError('Dual-source prefill does not support output quantization')
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise ValueError('Dual-source current Q/K/V must be BF16')
        # Upstream inline-scale views assume contiguous NHD pages.
        if not kv_cache.is_contiguous():
            raise ValueError('Dual-source FP8 requires contiguous NHD; cross-layer CPU offload is not validated')
        self._ensure_scale_caches(kv_cache)
        kc, vc = kv_cache.unbind(1)
        if kc.dtype == torch.uint8:
            kc, vc = kc.view(self.fp8_dtype), vc.view(self.fp8_dtype)
        for i, (start, end, length, cold) in enumerate(attn_metadata.hybrid_routes):
            count = end-start
            if count == 0:
                continue
            starts = attn_metadata.query_start_loc[i:i+2]-start
            lens = attn_metadata.seq_lens[i:i+1]
            q, out = query[start:end], output[start:end]
            route = 'cold_bf16' if cold else 'decode_fp8' if count == 1 else 'dual_source'
            route_key = (route, length-count, count)
            if route_key not in self._reported_routes and len(self._reported_routes) < 64:
                log.warning('DUAL_SOURCE_FORWARD layer=%s route=%s history=%d current=%d',
                            getattr(layer, 'layer_name', 'unknown'), route, length-count, count)
                self._reported_routes.add(route_key)
            if cold:
                context_attention_fwd(q=q, k=key[start:end], v=value[start:end], o=out,
                                      b_start_loc=starts[:1], b_seq_len=lens, max_input_len=count,
                                      is_causal=True, softmax_scale=self.scale)
            elif count == 1:
                single = replace(attn_metadata, num_actual_tokens=count, max_query_len=count,
                                 query_start_loc=starts, max_seq_len=length, seq_lens=lens,
                                 block_table=attn_metadata.block_table[i:i+1],
                                 slot_mapping=attn_metadata.slot_mapping[start:end])
                TritonAttentionImpl.forward(self, layer, q, key[start:end], value[start:end], kv_cache, single, out)
            else:
                unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=starts,
                    max_seqlen_q=count, seqused_k=lens, max_seqlen_k=length,
                    softmax_scale=self.scale, causal=True, window_size=(-1,-1),
                    block_table=attn_metadata.block_table[i:i+1], softcap=0,
                    q_descale=None, k_descale=None, v_descale=None,
                    kv_quant_mode=self._kv_quant_mode, k_scale_cache=self._k_scale_cache,
                    v_scale_cache=self._v_scale_cache, current_key=key[start:end], current_value=value[start:end],
                    block_source_branch=BLOCK_SOURCE_BRANCH)
        return output
