import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from local_document_qa import configuration, document_prompt
from validate_prefix_cache import cache_delta


class PrefixControls(unittest.TestCase):
    def test_only_prefix_flag_differs(self):
        a,b=configuration(False),configuration(True)
        self.assertEqual([k for k in a if a[k]!=b[k]],['prefix_caching'])

    def test_question_does_not_break_document_prefix(self):
        doc='Document facts\n'*1000
        a,b=document_prompt(doc,'First?'),document_prompt(doc,'Second?')
        self.assertEqual(a[:a.index(doc)+len(doc)],b[:b.index(doc)+len(doc)])

    def test_labelled_counters_are_counted_without_created_gauges(self):
        def snap(q,h):
            return {'status':200,'selected':[
                {'name':'vllm:prefix_cache_queries_total','labels':{'engine':'0'},'value':q},
                {'name':'vllm:prefix_cache_hits_total','labels':{'engine':'0'},'value':h},
                {'name':'vllm:prefix_cache_hits_created','labels':{'engine':'0'},'value':999999}]}
        delta=cache_delta({'metrics_before':snap(100,20),'metrics_after':snap(200,115)})
        self.assertEqual(delta,{'queries':100,'hits':95,'hit_ratio':.95})
        self.assertIsNone(cache_delta({'metrics_before':snap(0,0),'metrics_after':snap(0,0)})['hit_ratio'])


if __name__=='__main__':
    unittest.main()
