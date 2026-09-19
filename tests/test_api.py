import asyncio
import json
import unittest

import httpx
from fastapi.testclient import TestClient

from inference_service.api import ChatRequest, create_app
from inference_service.config import Settings


class EventStream(httpx.AsyncByteStream):
    def __init__(self, fail=False):
        self.closed = False
        self.fail = fail

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"42"}}]}\n\n'
        if self.fail:
            raise httpx.ReadError("test transport failure")
        yield b'data: [DONE]\n\n'

    async def aclose(self):
        self.closed = True


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.requests = []

    def handler(self, request):
        self.requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}],
                                       "usage": {"completion_tokens": 2}})

    def client(self, **settings):
        return TestClient(create_app(Settings(**settings), transport=httpx.MockTransport(self.handler)))

    def test_archived_budget_endpoint_is_absent(self):
        with self.client() as client:
            self.assertNotIn("/v1/solve", client.get("/openapi.json").json()["paths"])
            self.assertEqual(client.post("/v1/solve", json={"question": "x"}).status_code, 404)
        self.assertEqual(self.requests, [])

    def test_generic_chat_preserves_fields_without_integer_policy(self):
        with self.client() as client:
            client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}],
                                                     "temperature": .7, "max_tokens": 16})
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["model"], "qwen3-local")
        self.assertEqual(body["temperature"], .7)
        self.assertNotIn("thinking_token_budget", body)

    def test_document_precedes_changing_question(self):
        with self.client() as client:
            for question in ("Who?", "When?"):
                self.assertEqual(client.post("/v1/document/qa", json={"document": "Project A. Ada owns it.", "question": question}).status_code, 200)
        prompts = [json.loads(r.content)["messages"][0]["content"] for r in self.requests]
        self.assertEqual(prompts[0].split("\n\n问题：")[0], prompts[1].split("\n\n问题：")[0])
        self.assertNotEqual(prompts[0], prompts[1])

    def test_authentication_separates_client_and_backend_keys(self):
        with self.client(api_key="front-key", backend_api_key="back-key") as client:
            self.assertEqual(client.get("/v1/models").status_code, 401)
            self.assertEqual(client.get("/v1/models", headers={"Authorization": "Bearer front-key"}).status_code, 200)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].headers["authorization"], "Bearer back-key")

    def test_backend_status_preserved(self):
        app = create_app(transport=httpx.MockTransport(lambda _: httpx.Response(429, json={"error": "busy"})))
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}]})
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.json(), {"error": "busy"})
            self.assertEqual(client.get("/health").status_code, 503)

    def test_connection_and_timeout_failures_are_not_success(self):
        for error, status in ((httpx.ConnectError("offline"), 502), (httpx.ReadTimeout("slow"), 504)):
            def fail(_):
                raise error
            with TestClient(create_app(transport=httpx.MockTransport(fail))) as client:
                self.assertEqual(client.get("/v1/models").status_code, status)
                self.assertEqual(client.get("/health").status_code, 503)

    def test_stream_content_and_close(self):
        stream = EventStream()
        app = create_app(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)))
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}], "stream": True})
        self.assertIn('data: [DONE]', response.text)
        self.assertEqual(response.headers['X-Inference-Profile'], 'chunked-hybrid')
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertTrue(stream.closed)

    def test_interrupted_stream_reports_error_without_fabricating_done(self):
        stream = EventStream(fail=True)
        app = create_app(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)))
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}], "stream": True})
        self.assertIn("upstream_error", response.text)
        self.assertNotIn("[DONE]", response.text)
        self.assertTrue(stream.closed)


class StreamCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_stream_holds_slot_until_closed(self):
        app = create_app(Settings(max_inflight=1), transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=EventStream())))
        async with app.router.lifespan_context(app):
            endpoint = next(r.endpoint for r in app.routes if r.path == '/v1/chat/completions')
            request = ChatRequest(messages=[{'role': 'user', 'content': 'Hi'}], stream=True)
            first = await endpoint(request)
            await anext(first.body_iterator)
            busy = await endpoint(request)
            self.assertEqual(busy.status_code, 429)
            await first.body_iterator.aclose()
            resumed = await endpoint(request)
            self.assertEqual(resumed.status_code, 200)
            await anext(resumed.body_iterator)
            await resumed.body_iterator.aclose()

    async def test_closing_consumer_closes_upstream(self):
        stream = EventStream()
        app = create_app(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)))
        async with app.router.lifespan_context(app):
            endpoint = next(r.endpoint for r in app.routes if r.path == "/v1/chat/completions")
            response = await endpoint(ChatRequest(messages=[{"role": "user", "content": "Hi"}], stream=True))
            await anext(response.body_iterator)
            await response.body_iterator.aclose()
            self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
