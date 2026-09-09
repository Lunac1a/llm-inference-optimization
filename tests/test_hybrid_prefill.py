import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/hybrid_prefill'))
from hybrid_routes import sequence_routes
sys.path.insert(0,str(ROOT/'scripts'))
from validate_hybrid_prefill import config,launch_environment
from local_document_qa import configuration
import os

class HybridRoutes(unittest.TestCase):
    def test_full_cold_is_only_original_bf16_path(self):
        self.assertEqual(sequence_routes([0,100,101,109],[100,500,1000]),
                         [(0,100,100,True),(100,101,500,False),(101,109,1000,False)])
    def test_partial_prefix_must_not_use_incomplete_original_tensors(self):
        self.assertFalse(sequence_routes([0,16384],[16416])[0][3])
    def test_invalid_lengths_fail_closed(self):
        with self.assertRaises(ValueError):
            sequence_routes([0,100],[99])
    def test_control_settings_match_and_default_unchanged(self):
        a,b,c=[config(x) for x in ('bf16','fp8','hybrid')]
        self.assertEqual(b,c)
        self.assertEqual([k for k in a if a[k]!=b[k]],['kv_cache_dtype'])
        self.assertFalse(c['chunked_prefill'])
        self.assertEqual(c['max_num_batched_tokens'],c['max_model_len'])
        self.assertTrue(configuration()['chunked_prefill'])
        self.assertEqual(configuration()['attention_backend'],'FLASH_ATTN')
    def test_plugin_environment_is_scoped_even_on_error(self):
        before=dict(os.environ)
        with self.assertRaises(RuntimeError):
            with launch_environment('hybrid',ROOT/'.tmp/unused-test-path'):
                self.assertEqual(os.environ['VLLM_PLUGINS'],'inference_hybrid_prefill')
                raise RuntimeError('test')
        self.assertEqual(dict(os.environ),before)

if __name__=='__main__':
    unittest.main()
