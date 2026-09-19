import unittest
from report import paired_interval, validate_records


class EvaluationTests(unittest.TestCase):
    def test_pairing_retains_constant_difference(self):
        lo, hi = paired_interval([[.1] * 20, [.1] * 20])
        self.assertAlmostEqual(lo, 10)
        self.assertAlmostEqual(hi, 10)

    def test_audit_rejects_duplicate_and_cached_results(self):
        source = {'a': {'prompt_ids': [1, 2], 'max_tokens': 3}}
        row = {'id': 'a', 'done': True, 'prediction': 'x', 'finish_reason': 'stop',
               'cached_tokens': 0, 'cache_hit_delta': 0, 'preemptions': 0,
               'usage': {'prompt_tokens': 2, 'completion_tokens': 1}}
        validate_records([row], source)
        with self.assertRaises(AssertionError): validate_records([row, row], source)
        with self.assertRaises(AssertionError): validate_records([{**row, 'cache_hit_delta': 1}], source)
        with self.assertRaises(AssertionError): validate_records([row], source, fixed=True)


if __name__ == '__main__': unittest.main()
