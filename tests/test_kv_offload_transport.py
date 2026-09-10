import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/kv_offload_compat'))
from transport import grouped_mappings

class Tensor:
    def __init__(self, base, size): self.base, self.size = base, size
    def data_ptr(self): return self.base
    def numel(self): return self.size
    def element_size(self): return 1
    def is_contiguous(self): return True

class MappingTests(unittest.TestCase):
    def test_local_plugin_discovery(self):
        from importlib.metadata import distributions
        plugin = Path(__file__).resolve().parents[1] / 'experiments/kv_offload_compat'
        entries = [e for d in distributions(path=[str(plugin)]) for e in d.entry_points]
        self.assertTrue(any(e.group == 'vllm.general_plugins' and e.name == 'inference_kv_transport'
                            and e.value == 'transport:install' for e in entries))
    def test_permuted_multiple_tensors(self):
        groups = grouped_mappings([1048, 2000, 1016], [3048, 4000, 3000], [16, 32, 16],
                                  [Tensor(1000, 64), Tensor(2000, 64)], [Tensor(3000, 64), Tensor(4000, 64)])
        self.assertEqual(groups, {(0, 0, 16): [(3, 3), (1, 0)], (1, 1, 32): [(0, 0)]})
    def test_invalid_descriptors_rejected_before_copy(self):
        for src, dst, size in [(999, 3000, 16), (1060, 3000, 16), (1001, 3000, 16), (1000, 3060, 16), (1000, 3000, 0)]:
            with self.assertRaises(ValueError):
                grouped_mappings([src], [dst], [size], [Tensor(1000, 64)], [Tensor(3000, 64)])

if __name__ == '__main__': unittest.main()
