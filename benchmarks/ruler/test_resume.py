import unittest
from resume_collect import remaining

class ResumeTests(unittest.TestCase):
    def test_only_fixed_unfinished_suffix_is_selected(self):
        rows=[{'id':str(i)} for i in range(3)]
        saved=[{'id':'0','cached_tokens':0,'cache_hit_delta':0,'preemptions':0,'finish_reason':'stop'}]
        self.assertEqual(remaining(rows,saved),rows[1:])
        with self.assertRaises(AssertionError):remaining(rows,saved*2)
        with self.assertRaises(AssertionError):remaining(rows,[{**saved[0],'id':'1'}])
        with self.assertRaises(AssertionError):remaining(rows,[{**saved[0],'preemptions':1}])

if __name__=='__main__':unittest.main()
