import json
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from test_stage3_collection import load


class Stage3RepairTests(unittest.TestCase):
    def test_scorer_requires_every_field_and_own_document(self):
        collect, prepare = load('stage3_collect'), load('stage3_prepare')
        spec = prepare.DOC_SPECS[0]
        questions = prepare.quality_questions({'id': 'a', 'sha256': 'hash', 'text': ''}, spec)
        response = {'status': 200, 'stream_complete': True, 'finish_reason': 'stop', 'error': None}
        multi = next(q for q in questions if q['id'] == 'a-cross-2')
        score = lambda q, text: collect.score_quality_answer(q, {**response, 'text': text})['correct']
        self.assertFalse(score(multi, spec['code']))
        self.assertTrue(score(multi, spec['owner'] + '；' + spec['code'] + '。'))
        self.assertFalse(score(multi, '不是' + spec['code'] + '；' + spec['owner']))
        last = next(q for q in questions if q['id'] == 'a-cross-3')
        self.assertFalse(score(last, '最后进行人工抽检'))
        self.assertTrue(score(last, spec['relation'].split('，')[-1]))
        for change in ({'status': 500}, {'stream_complete': False}, {'finish_reason': None}, {'finish_reason': 'length'}):
            self.assertFalse(collect.score_quality_answer(last, {**response, 'text': spec['relation'].split('，')[-1], **change})['correct'])

    def test_question_oracles_match_each_document(self):
        prepare = load('stage3_prepare')
        for spec in prepare.DOC_SPECS:
            qs = prepare.quality_questions({'id': spec['id'], 'sha256': '', 'text': ''}, spec)
            self.assertEqual(len(qs), 12)
            budget = next(q for q in qs if q['id'].endswith('distractor-1'))
            self.assertIn('金额', budget['question'])
            self.assertEqual(budget['answer_groups'], [[spec['budget']]])
            last = next(q for q in qs if q['id'].endswith('cross-3'))
            self.assertEqual(last['answer_groups'], [[spec['relation'].split('，')[-1]]])

    def test_shared_prefix_includes_whole_document(self):
        collect = load('stage3_collect')
        docs = {str(i): {'text': '独立文档' + str(i) + '正文' * 100} for i in range(4)}
        data = {'documents': {'16384': docs}}
        first, follow = collect.formal_prompts(data, 16384, True)
        for i, (_, prompt) in enumerate(follow):
            prefix = first[i % 4][1].split('\n\n问题：')[0]
            self.assertTrue(prompt.startswith(prefix + '\n\n问题：'))
            self.assertIn(docs[str(i % 4)]['text'], prefix)
        _, independent = collect.formal_prompts(data, 16384, False)
        self.assertEqual(len({p.split('。')[0] for _, p in independent}), 16)

    def test_reset_failure_restarts_and_failure_stops(self):
        collect = load('stage3_collect')
        server = Mock(base_url='http://fixture', candidate={'label': 'A'}, run_dir=Path('runs/old'))
        server.start.return_value = {'pid': 2}
        budget = Mock()
        with patch.object(collect, 'reset_prefix_cache', return_value={'ok': False, 'status': 404}), patch.object(collect, 'stop_candidate') as stop:
            result = collect.cold_cache(server, budget, 'test')
            self.assertEqual(result['method'], 'restart')
            stop.assert_called_once()
            server.start.assert_called_once()
            self.assertNotEqual(server.run_dir, Path('runs/old'))
            server.start.side_effect = RuntimeError('startup failed')
            with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                collect.cold_cache(server, budget, 'again')
            self.assertEqual(stop.call_count, 3)

    def test_successful_reset_does_not_restart(self):
        collect = load('stage3_collect')
        server, budget = Mock(), Mock()
        with patch.object(collect, 'reset_prefix_cache', return_value={'ok': True, 'status': 200}):
            self.assertEqual(collect.cold_cache(server, budget, 'test')['method'], 'reset')
        server.start.assert_not_called()

    def test_reset_http_200_with_failure_body_is_not_success(self):
        lib = load('stage3_lib')
        for body in (False, {'ok': False}, {'success': False}, {'error': 'busy'}):
            with patch.object(lib, 'http_json', return_value=(200, body, 0)):
                self.assertFalse(lib.reset_prefix_cache('http://fixture')['ok'])

    def test_missing_done_finish_and_usage_fail_closed(self):
        lib = load('stage3_lib')
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self): return iter(self.lines)
        token = b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n'
        usage = b'data: {"choices":[],"usage":{"completion_tokens":1}}\n'
        done = b'data: [DONE]\n'
        for lines in ([token, usage], [token, done], [token.replace(b'"stop"', b'null'), usage, done]):
            response = Response(); response.lines = lines
            with patch.object(lib.urllib.request, 'urlopen', return_value=response):
                result = lib._stream_chat('http://fixture', {}, 1)
            self.assertFalse(result['stream_complete'])
            self.assertEqual(result['error'], 'incomplete_stream')

    def test_trickling_request_and_batch_have_absolute_deadlines(self):
        lib, collect = load('stage3_lib'), load('stage3_collect')
        active = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                active.set()
                self.send_response(200); self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(b': keepalive\n\n'); self.wfile.flush(); time.sleep(.03)
                except (OSError, ValueError): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            start = time.perf_counter()
            result = lib.stream_chat(f'http://127.0.0.1:{server.server_port}', {}, .6)
            self.assertTrue(active.is_set())
            self.assertEqual(result['error'], 'request_deadline_exceeded')
            self.assertLess(time.perf_counter() - start, 1.6)
            candidate = {'label': 'A', 'host': '127.0.0.1', 'port': server.server_port,
                         'request_timeout_seconds': 10, 'batch_timeout_seconds': .6}
            temp_root = (lib.ROOT / '.tmp' / 'stage3-tests').resolve()
            temp_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=temp_root) as temp, ExitStack() as stack:
                stack.enter_context(patch.object(collect, 'metrics_snapshot', return_value={}))
                stack.enter_context(patch.object(collect, 'Telemetry'))
                start = time.perf_counter()
                with self.assertRaisesRegex(RuntimeError, 'Batch failed'):
                    collect.run_batch(candidate, Path(temp), 'timeout', [('a','a'),('b','b'),('c','c')], 2, 128, True, budget=Mock())
                self.assertLess(time.perf_counter() - start, 1.6)
                rows = json.loads((Path(temp)/'timeout/requests.json').read_text(encoding='utf-8'))
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(r.get('error') for r in rows))
        finally:
            server.shutdown(); server.server_close()

    def test_old_evidence_is_read_only(self):
        lib = load('stage3_lib')
        for name in ('stage1', 'stage2', 'stage2-bandwidth', 'stage3'):
            with self.assertRaises(ValueError):
                lib.write_json(lib.ROOT/'artifacts'/name/'must-not-exist.json', {})
        self.assertEqual(lib.OUT.name, 'stage3-v2')


if __name__ == '__main__':
    unittest.main()
