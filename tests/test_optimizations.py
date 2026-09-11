import os
import unittest
from unittest.mock import patch

from inference_service.budget import select_budget
from inference_service.backends.routes import sequence_routes
from inference_service.backends.cpu_kv import grouped_mappings
from inference_service import plugins
from inference_service.config import Settings
from inference_service.runtime import launch_spec


class Tensor:
    def __init__(self, base, size=64): self.base, self.size = base, size
    def data_ptr(self): return self.base
    def numel(self): return self.size
    def element_size(self): return 1
    def is_contiguous(self): return True


class OptimizationTests(unittest.TestCase):
    def test_frozen_policies(self):
        self.assertEqual(select_budget("remaining items", "economy"), ("sequential", 512))
        self.assertEqual(select_budget("remaining items", "quality"), ("sequential", 1024))
        self.assertEqual(select_budget("Calculate 23 * 17", "quality"), ("arithmetic", 512))
        self.assertEqual(select_budget("unknown question", "quality"), ("unknown", 512))

    def test_cold_full_context_and_cached_routes_remain_distinct(self):
        self.assertEqual(sequence_routes([0, 100, 101, 109], [100, 500, 1000]),
                         [(0, 100, 100, True), (100, 101, 500, False), (101, 109, 1000, False)])
        self.assertFalse(sequence_routes([0, 16384], [16416])[0][3])
        with self.assertRaises(ValueError): sequence_routes([0, 100], [99])

    def test_cpu_descriptor_mapping_and_boundaries(self):
        self.assertEqual(grouped_mappings([1048, 2000, 1016], [3048, 4000, 3000], [16, 32, 16],
            [Tensor(1000), Tensor(2000)], [Tensor(3000), Tensor(4000)]),
            {(0, 0, 16): [(3, 3), (1, 0)], (1, 1, 32): [(0, 0)]})
        for src, dst, size in ((999, 3000, 16), (1060, 3000, 16), (1001, 3000, 16), (1000, 3060, 16), (1000, 3000, 0)):
            with self.assertRaises(ValueError):
                grouped_mappings([src], [dst], [size], [Tensor(1000)], [Tensor(3000)])

    def test_runtime_profiles_and_environment_are_explicit(self):
        before = dict(os.environ)
        default_args, default_env = launch_spec(Settings())
        hybrid_args, hybrid_env = launch_spec(Settings(profile="hybrid"))
        cpu_args, cpu_env = launch_spec(Settings(profile="cpu-kv"))
        adaptive_args, adaptive_env = launch_spec(Settings(profile="adaptive", batch_invariant=True))
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(default_env["VLLM_PLUGINS"], "")
        self.assertIn("bfloat16", default_args)
        self.assertIn("--no-enable-chunked-prefill", hybrid_args)
        self.assertIn("fp8_per_token_head", hybrid_args)
        self.assertEqual(hybrid_env["VLLM_PLUGINS"], "inference_hybrid_prefill")
        self.assertIn("--kv-offloading-size", cpu_args)
        self.assertEqual(cpu_env["VLLM_PLUGINS"], "inference_kv_transport")
        self.assertIn("--no-enable-prefix-caching", adaptive_args)
        self.assertEqual(adaptive_env["VLLM_BATCH_INVARIANT"], "1")

    def test_unvalidated_profile_combinations_are_rejected(self):
        with self.assertRaises(ValueError): Settings(profile="hybrid", batch_invariant=True)
        with self.assertRaises(ValueError): Settings(backend_url="http://host/path")
        with self.assertRaises(ValueError): Settings(backend_url="http://secret@host")

    def test_entrypoints_are_inert_outside_selected_runtime(self):
        with patch.dict(os.environ, {"INFERENCE_RUNTIME_PROFILE": "document"}):
            plugins.register_hybrid()
            plugins.register_cpu_kv()


if __name__ == "__main__": unittest.main()
