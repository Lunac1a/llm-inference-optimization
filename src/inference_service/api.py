"""OpenAI-compatible chat and document QA for attention comparisons."""
from contextlib import asynccontextmanager
import asyncio
import hmac
import json
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from . import __version__
from .config import Settings
from .prompts import document_prompt


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False


class DocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: str = Field(min_length=1, max_length=200000)
    question: str = Field(min_length=1, max_length=16000)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    stream: bool = False


def create_app(settings: Settings | None = None, *, transport=None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        headers = {"Authorization": f"Bearer {settings.backend_api_key}"} if settings.backend_api_key else {}
        async with httpx.AsyncClient(base_url=settings.backend_url.rstrip("/"), headers=headers,
                                     timeout=httpx.Timeout(settings.request_timeout, connect=5),
                                     transport=transport, trust_env=False) as client:
            app.state.backend = client
            app.state.slots = asyncio.Semaphore(settings.max_inflight)
            yield

    app = FastAPI(title="Mixed-Attention Inference API", version=__version__, lifespan=lifespan)

    async def authorize(authorization: str | None = Header(default=None)):
        if settings.api_key and not hmac.compare_digest((authorization or "").encode(), f"Bearer {settings.api_key}".encode()):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    async def forward(path: str, body: dict | None = None, *, headers: dict | None = None):
        client = app.state.backend
        stream = bool(body and body.get("stream"))
        handed_off = False
        response = None
        acquired = False

        def release():
            nonlocal acquired
            if acquired:
                acquired = False
                app.state.slots.release()

        try:
            if body is not None:
                if app.state.slots.locked():
                    return JSONResponse({"error": {"type": "overloaded", "message": "Too many active requests; retry later"}},
                                        429, headers={"Retry-After": "1"})
                await app.state.slots.acquire()
                acquired = True
            request = client.build_request("POST" if body is not None else "GET", path, json=body)
            response = await client.send(request, stream=True)
            response_headers = {"X-Inference-Profile": settings.profile, **(headers or {})}
            if response.status_code >= 400 or not stream:
                content = await response.aread()
                return Response(content, status_code=response.status_code,
                                media_type=response.headers.get("content-type", "application/json"),
                                headers=response_headers)

            async def chunks():
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                except httpx.HTTPError:
                    yield ("data: " + json.dumps({"error": {"type": "upstream_error",
                           "message": "Backend stream interrupted"}}) + "\n\n").encode()
                finally:
                    await response.aclose()
                    release()

            async def cleanup():
                await response.aclose()
                release()

            response_headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            result = StreamingResponse(chunks(), media_type="text/event-stream", headers=response_headers,
                                       background=BackgroundTask(cleanup))
            # Ownership transfers to the streaming iterator only after construction succeeds.
            handed_off = True
            return result
        except httpx.TimeoutException:
            return JSONResponse({"error": {"type": "upstream_timeout", "message": "Backend timed out"}}, 504)
        except httpx.HTTPError:
            return JSONResponse({"error": {"type": "upstream_unavailable", "message": "Backend unavailable"}}, 502)
        finally:
            if response is not None and not handed_off:
                await response.aclose()
            if not handed_off:
                release()

    @app.get("/health", dependencies=[Depends(authorize)])
    async def health():
        try:
            response = await app.state.backend.get("/health", timeout=2)
            ready = response.status_code == 200
        except httpx.HTTPError:
            ready = False
        return JSONResponse({"status": "ready" if ready else "unavailable", "profile": settings.profile},
                            status_code=200 if ready else 503)

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return await forward("/v1/models")

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat(request: ChatRequest):
        body = request.model_dump(exclude_none=True)
        body.setdefault("model", settings.served_model)
        return await forward("/v1/chat/completions", body)

    @app.post("/v1/document/qa", dependencies=[Depends(authorize)])
    async def document_qa(request: DocumentRequest):
        return await forward("/v1/chat/completions", {
            "model": settings.served_model,
            "messages": [{"role": "user", "content": document_prompt(request.document, request.question,
                          independent_title=settings.profile == "hybrid")}],
            "temperature": 0, "max_tokens": request.max_tokens, "stream": request.stream,
            "chat_template_kwargs": {"enable_thinking": False},
            **({"stream_options": {"include_usage": True}} if request.stream else {}),
        })

    return app
