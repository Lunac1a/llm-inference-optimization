"""Scoped descriptor adapter to upstream explicit-direction swap_blocks.

No raw foreign pointer dereference: each address must resolve inside the handler's
own contiguous tensors. All stream/event/scheduler logic remains upstream.
"""
import logging


def grouped_mappings(src_ptrs, dst_ptrs, sizes, src_tensors, dst_tensors):
    if not (len(src_ptrs) == len(dst_ptrs) == len(sizes)):
        raise ValueError('Descriptor length mismatch')
    def locate(pointer, size, tensors):
        for i, tensor in enumerate(tensors):
            if not tensor.is_contiguous():
                raise ValueError('Noncontiguous transfer tensor')
            offset = pointer - tensor.data_ptr()
            if 0 <= offset and offset + size <= tensor.numel() * tensor.element_size():
                if offset % size:
                    raise ValueError('Unaligned descriptor')
                return i, offset // size
        raise ValueError('Descriptor outside owned tensor')
    grouped = {}
    for src, dst, size in zip(src_ptrs, dst_ptrs, sizes):
        if size <= 0:
            raise ValueError('Invalid copy size')
        si, so = locate(src, size, src_tensors)
        di, do = locate(dst, size, dst_tensors)
        grouped.setdefault((si, di, size), []).append((so, do))
    return grouped


def make_copy(src_tensors, dst_tensors):
    import torch
    from vllm import _custom_ops as ops
    def copy(src_ptrs, dst_ptrs, sizes, is_src_access_order_any=False):
        grouped = grouped_mappings(src_ptrs.tolist(), dst_ptrs.tolist(), sizes.tolist(), src_tensors, dst_tensors)
        for (si, di, size), mapping in grouped.items():
            ops.swap_blocks(src_tensors[si], dst_tensors[di], size, torch.tensor(mapping, dtype=torch.int64))
    return copy


def install():
    import vllm
    from vllm.v1.kv_offload.cpu import gpu_worker
    if vllm.__version__ != '0.23.0':
        raise RuntimeError('Compatibility adapter audited for 0.23.0 only')
    cls = gpu_worker.SingleDirectionOffloadingHandler
    if getattr(cls, '_local_native_single', False):
        return
    original = cls.__init__
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._swap_blocks_batch = make_copy(self.src_tensors, self.dst_tensors)
        logging.getLogger(__name__).warning('KV_OFFLOAD_COMPAT native_single direction=%s', self.transfer_type)
    cls.__init__ = init
    cls._local_native_single = True
