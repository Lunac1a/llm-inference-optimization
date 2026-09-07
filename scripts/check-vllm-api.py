"""Run the fixed Stage 1 API acceptance checks and preserve all responses."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone


def request_json(url: str, payload: dict | None = None, timeout: float = 60.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body, dict(response.headers)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return error.code, body, dict(error.headers)
    except Exception as error:  # preserve transport failures in the artifact
        return None, json.dumps({"transport_error": f"{type(error).__name__}: {error}"}), {}


def stream_chat(url: str, payload: dict):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[str] = []
    events: list[dict] = []
    status = None
    done = False
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    continue
                try:
                    event = json.loads(data)
                    events.append(event)
                    for choice in event.get("choices", []):
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            chunks.append(content)
                except json.JSONDecodeError:
                    events.append({"raw_event": data})
    except urllib.error.HTTPError as error:
        status = error.code
        events.append({"error": error.read().decode("utf-8", errors="replace")})
    except Exception as error:
        events.append({"transport_error": f"{type(error).__name__}: {error}"})
    return status, "".join(chunks), events, done


def decode(body: str):
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw_body": body}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-4b-baseline")
    parser.add_argument("--output-dir", default="artifacts/stage1/api")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result: dict = {
        "run_id": run_id,
        "base_url": base_url,
        "model": args.model,
        "checks": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    def add_check(name: str, passed: bool, **details):
        result["checks"].append({"name": name, "passed": passed, **details})

    status, body, _ = request_json(f"{base_url}/health")
    add_check("health", status == 200, status=status, body=body)
    status, body, _ = request_json(f"{base_url}/v1/models")
    models = decode(body)
    returned_ids = [item.get("id") for item in models.get("data", [])] if isinstance(models, dict) else []
    add_check("models", status == 200 and returned_ids == [args.model], status=status, returned_ids=returned_ids, body=models)

    prompts = [
        {"id": "chinese", "text": "请用一句话说明缓存的作用。"},
        {"id": "english", "text": "In one sentence, explain what a cache does."},
        {"id": "calculation", "text": "Calculate 17 * 19 and give only the number."},
    ]
    common = {
        "model": args.model,
        "max_tokens": 128,
        "temperature": 0,
        "seed": 42,
        "extra_body": {"enable_thinking": False},
    }
    for prompt in prompts:
        for repeat in (1, 2):
            payload = {**common, "messages": [{"role": "user", "content": prompt["text"]}], "stream": False}
            status, body, _ = request_json(f"{base_url}/v1/chat/completions", payload)
            decoded = decode(body)
            content = ""
            if isinstance(decoded, dict) and decoded.get("choices"):
                content = (decoded["choices"][0].get("message") or {}).get("content") or ""
            add_check(
                f"chat_non_stream_{prompt['id']}_{repeat}",
                status == 200 and bool(content.strip()),
                status=status,
                request=payload,
                response=decoded,
            )

            stream_payload = {
                **common,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": prompt["text"]}],
                "stream": True,
            }
            status, content, events, done = stream_chat(f"{base_url}/v1/chat/completions", stream_payload)
            add_check(
                f"chat_stream_{prompt['id']}_{repeat}",
                status == 200 and bool(content.strip()) and done,
                status=status,
                request=stream_payload,
                content=content,
                event_count=len(events),
                events=events,
                done=done,
            )

    invalid_payload = {**common, "model": "missing-stage1-model", "messages": [{"role": "user", "content": "test"}]}
    status, body, _ = request_json(f"{base_url}/v1/chat/completions", invalid_payload)
    invalid_decoded = decode(body)
    add_check(
        "invalid_model_rejected",
        status in {400, 404} and isinstance(invalid_decoded, dict) and bool(invalid_decoded.get("error")),
        status=status,
        request=invalid_payload,
        response=invalid_decoded,
    )

    oversized_text = "token " * 5000
    oversized_payload = {**common, "messages": [{"role": "user", "content": oversized_text}]}
    status, body, _ = request_json(f"{base_url}/v1/chat/completions", oversized_payload)
    oversized_decoded = decode(body)
    add_check(
        "oversized_context_rejected",
        status in {400, 413} and isinstance(oversized_decoded, dict) and bool(oversized_decoded.get("error")),
        status=status,
        request_summary={"model": args.model, "characters": len(oversized_text)},
        response=oversized_decoded,
    )

    status, body, _ = request_json(f"{base_url}/health")
    add_check("health_after_negative_checks", status == 200, status=status, body=body)
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["passed"] = all(check["passed"] for check in result["checks"])
    result["check_count"] = len(result["checks"])
    result["passed_count"] = sum(check["passed"] for check in result["checks"])

    output_path = output_dir / f"api-check-{run_id}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "api-check-latest.json").write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": result["check_count"], "passed_count": result["passed_count"], "output": str(output_path)}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
