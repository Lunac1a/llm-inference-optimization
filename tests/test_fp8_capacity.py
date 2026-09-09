import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_fp8_capacity import config
from local_document_qa import configuration

class CapacityControls(unittest.TestCase):
    def test_precision_is_only_difference_and_default_is_preserved(self):
        a,b=config(False),config(True)
        self.assertEqual([k for k in a if a[k]!=b[k]],['kv_cache_dtype'])
        self.assertEqual(a['attention_backend'],'TRITON_ATTN')
        self.assertTrue(a['prefix_caching'])
        self.assertEqual(a['dtype'],'bfloat16')
        self.assertEqual(configuration()['attention_backend'],'FLASH_ATTN')
        self.assertEqual(configuration()['kv_cache_dtype'],'bfloat16')

if __name__=='__main__':
    unittest.main()
