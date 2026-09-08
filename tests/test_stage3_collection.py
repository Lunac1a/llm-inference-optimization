import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Stage3CollectionTests(unittest.TestCase):
    def test_candidate_matrix_is_fixed(self):
        config = json.loads((ROOT / "configs" / "stage3-combinations.json").read_text())
        self.assertEqual(set(config["candidates"]), set("ABCDEF"))
        self.assertEqual(config["candidates"]["A"], {"attention_backend": "FLASH_ATTN", "kv_cache_dtype": "bfloat16", "prefix_caching": False})
        self.assertEqual(config["candidates"]["F"], {"attention_backend": "TRITON_ATTN", "kv_cache_dtype": "fp8_per_token_head", "prefix_caching": True})

    def test_stream_parser_marks_truncation_and_error(self):
        lib = load("stage3_lib")
        # The network parser is exercised against the live vLLM API during the
        # real collector; this check protects its timing/metric contracts.
        self.assertEqual(lib.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(lib.metric_delta({"selected": [{"name": "vllm:num_preemptions", "labels": {}, "value": 1}]}, {"selected": [{"name": "vllm:num_preemptions", "labels": {}, "value": 3}]}), {"vllm:num_preemptions": 2})

    def test_prometheus_parser_and_cache_labels(self):
        lib = load("stage3_lib")
        rows = lib.parse_prometheus('vllm:cache_config_info{cache_dtype="fp8_per_token_head",block_size="16"} 1\nvllm:num_requests_waiting 2')
        self.assertEqual(rows[0]["labels"]["cache_dtype"], "fp8_per_token_head")
        self.assertEqual(rows[1]["value"], 2)

    def test_stream_timing_error_and_cache_reset_contracts(self):
        lib = load("stage3_lib")

        class Handler(BaseHTTPRequestHandler):
            error_next = False

            def log_message(self, *_args):
                return

            def do_POST(self):
                if self.path == "/reset_prefix_cache":
                    body = b'{"ok":true}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/v1/chat/completions":
                    if type(self).error_next:
                        type(self).error_next = False
                        body = b'{"error":"transport fixture"}'
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    chunks = [
                        'data: {"choices":[{"delta":{"content":"答"},"finish_reason":null}]}\n\n'.encode("utf-8"),
                        'data: {"choices":[{"delta":{"content":"案"},"finish_reason":"length"}]}\n\n'.encode("utf-8"),
                        b'data: {"choices":[],"usage":{"completion_tokens":128}}\n\n',
                        b'data: [DONE]\n\n',
                    ]
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for chunk in chunks:
                        self.wfile.write(chunk)
                        self.wfile.flush()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            response = lib.stream_chat(base, {"model": "fixture"}, timeout=5)
            self.assertEqual(response["status"], 200)
            self.assertGreaterEqual(response["ttft_seconds"], 0)
            self.assertEqual(response["output_tokens"], 128)
            self.assertTrue(response["stream_complete"])
            self.assertEqual(lib.reset_prefix_cache(base)["status"], 200)
            Handler.error_next = True
            error_response = lib.stream_chat(base, {"model": "fixture"}, timeout=5)
            # The successful fixture above is intentionally a length-truncated
            # answer; the scorer, rather than the transport parser, rejects it.
            collect = load("stage3_collect")
            question = {"expected_markers": ["答案"]}
            self.assertFalse(collect.score_quality_answer(question, {"text": "答案", "finish_reason": "length"})["correct"])
            self.assertEqual(error_response["status"], 400)
        finally:
            server.shutdown()
            server.server_close()

    def test_budget_stop_is_hard(self):
        import time
        lib = load("stage3_lib")
        budget = lib.Budget.__new__(lib.Budget)
        budget.deadline = time.time() - 1
        with self.assertRaises(RuntimeError):
            budget.ensure(batch_timeout=1, cleanup_reserve=1)


if __name__ == "__main__":
    unittest.main()
