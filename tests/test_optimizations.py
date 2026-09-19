import os
import unittest
from unittest.mock import patch

from inference_service.backends.routes import sequence_routes
from inference_service import plugins
from inference_service.config import Settings
from inference_service.runtime import launch_spec


class OptimizationTests(unittest.TestCase):
    def test_cold_full_context_and_cached_routes_remain_distinct(self):
        self.assertEqual(sequence_routes([0, 100, 101, 109], [100, 500, 1000]),
                         [(0, 100, 100, True), (100, 101, 500, False), (101, 109, 1000, False)])
        self.assertFalse(sequence_routes([0, 16384], [16416])[0][3])
        with self.assertRaises(ValueError): sequence_routes([0, 100], [99])

    def test_runtime_profiles_and_environment_are_explicit(self):
        before = dict(os.environ)
        default_args, default_env = launch_spec(Settings(profile="document"))
        hybrid_args, hybrid_env = launch_spec(Settings(profile="hybrid"))
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(default_env["VLLM_PLUGINS"], "")
        self.assertIn("bfloat16", default_args)
        self.assertIn("--no-enable-chunked-prefill", hybrid_args)
        self.assertIn("fp8_per_token_head", hybrid_args)
        self.assertEqual(hybrid_env["VLLM_PLUGINS"], "inference_hybrid_prefill")
        self.assertIn("--enable-prefix-caching", default_args)
        self.assertIn("--enable-chunked-prefill", default_args)
        self.assertEqual(default_args[default_args.index("--max-model-len")+1], "32768")
        self.assertEqual(default_args[default_args.index("--max-num-seqs")+1], "8")
        self.assertEqual(default_args[default_args.index("--max-num-batched-tokens")+1], "2048")
        self.assertEqual(default_env["VLLM_BATCH_INVARIANT"], "0")
        self.assertNotIn("--kv-offloading-size", default_args)

    def test_unvalidated_profile_combinations_are_rejected(self):
        with self.assertRaises(ValueError): Settings(profile="adaptive")
        with self.assertRaises(ValueError): Settings(profile="cpu-kv")
        with self.assertRaises(ValueError): Settings(backend_url="http://host/path")
        with self.assertRaises(ValueError): Settings(backend_url="http://secret@host")

    def test_entrypoints_are_inert_outside_selected_runtime(self):
        with patch.dict(os.environ, {"INFERENCE_RUNTIME_PROFILE": "document"}):
            plugins.register_hybrid()
            plugins.register_chunked()

    def test_chunked_attention_has_isolated_validated_capacity_and_layout(self):
        with patch.dict(os.environ, {"VLLM_KV_CACHE_LAYOUT": "HND"}):
            args, env = launch_spec(Settings())
        self.assertEqual(Settings().profile, "chunked-hybrid")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings.from_env().profile, "chunked-hybrid")
        self.assertEqual(env["VLLM_PLUGINS"], "inference_chunked_prefill")
        self.assertEqual(env["VLLM_KV_CACHE_LAYOUT"], "NHD")
        self.assertEqual(args[args.index("--kv-cache-memory-bytes")+1], str(4*2**30))
        self.assertEqual(args[args.index("--max-num-batched-tokens")+1], "2048")
        self.assertEqual(args[args.index("--max-model-len")+1], "16640")
        self.assertIn("--enable-chunked-prefill", args)
        self.assertIn("--enforce-eager", args)
        self.assertNotIn("--kv-offloading-size", args)


if __name__ == "__main__": unittest.main()
