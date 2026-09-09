import sys
from pathlib import Path
import unittest
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_fp8_capacity import config
from local_document_qa import configuration
from analyze_fp8_capacity import point_pass

class CapacityControls(unittest.TestCase):
    def test_precision_is_only_difference_and_default_is_preserved(self):
        a,b=config(False),config(True)
        self.assertEqual([k for k in a if a[k]!=b[k]],['kv_cache_dtype'])
        self.assertEqual(a['attention_backend'],'TRITON_ATTN')
        self.assertTrue(a['prefix_caching'])
        self.assertEqual(a['dtype'],'bfloat16')
        self.assertEqual(configuration()['attention_backend'],'FLASH_ATTN')
        self.assertEqual(configuration()['kv_cache_dtype'],'bfloat16')

    def test_successful_requests_alone_do_not_prove_capacity(self):
        wave={'success':3,'requests':3,'deltas':{'vllm:num_preemptions_total':0},
              'ttft':{'p95':1},'e2e':{'p95':7},'cache':{'hit_ratio':.99}}
        row={'waves':[deepcopy(wave) for _ in range(3)]}
        self.assertTrue(point_pass(row))
        row['waves'][1]['cache']['hit_ratio']=.5
        self.assertFalse(point_pass(row))
        row['waves'][1]['cache']['hit_ratio']=.99
        row['waves'][2]['ttft']['p95']=2.01
        self.assertFalse(point_pass(row))

if __name__=='__main__':
    unittest.main()
