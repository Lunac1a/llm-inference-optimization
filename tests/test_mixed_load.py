"""Offline checks of decision boundaries and SSE accounting, no GPU."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from validate_mixed_load import interference, labelled_delta, request, content_times


class MixedTests(unittest.TestCase):
    def test_gate_needs_two_pairs_and_absolute_increase(self):
        c = {'A_gap': .05, 'B_ttft': .1}
        small = {'A_gap': .11, 'B_ttft': .3}
        bad = {'A_gap': .31, 'B_ttft': .1}
        self.assertFalse(interference([(c, bad), (c, small), (c, small)])['triggered'])
        self.assertTrue(interference([(c, bad), (c, bad), (c, small)])['triggered'])

    def test_labelled_counters(self):
        def sample(value):
            return {'selected': [{'name': 'queue_sum', 'labels': {'model': 'a'}, 'value': value}]}
        self.assertEqual(labelled_delta(sample(2), sample(5))[0]['delta'], 3)

    def test_sse_coalescing_is_not_token_latency(self):
        events = [{'choices': [{'delta': {'content': 'first'}, 'token_ids': list(range(16))}]},
                  {'choices': [{'delta': {'content': 'last'}, 'token_ids': list(range(16, 128)), 'finish_reason': 'length'}]},
                  {'choices': [], 'usage': {'completion_tokens': 128}}]
        class Response:
            status = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            @property
            def content(self):
                async def lines():
                    for e in events:
                        yield ('data: ' + json.dumps(e) + '\n').encode()
                    yield b'data: [DONE]\n'
                return lines()
        class Session:
            def post(self, *args, **kwargs): return Response()
        with tempfile.TemporaryDirectory() as directory:
            counts = []
            row = asyncio.run(request(Session(), {'text': 'doc'}, 'q', 'test', Path(directory),
                                      callback=lambda row, stamp: counts.append(row['token_count'])))
            self.assertEqual(row['token_count'], 128)
            self.assertEqual(len(content_times(row)), 2)
            self.assertEqual(counts[0], 16)


if __name__ == '__main__': unittest.main()
