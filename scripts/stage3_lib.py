"""Shared, dependency-light primitives for the bounded Stage 3 study."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Any, Iterable
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage3-combinations.json"
OUT = ROOT / "artifacts" / "stage3"
STATE = ROOT / ".tmp" / "stage3"


def now_utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def candidate_config(label: str) -> dict[str, Any]:
    config = load_config()
    if label not in config["candidates"]:
        raise ValueError(f"Unknown Stage 3 candidate: {label}")
    return {**config["common"], **config["candidates"][label], "label": label}


def percentile(values: Iterable[float], pct: float) -> float | None:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Budget:
    def __init__(self, path: Path = OUT / "budget.json") -> None:
        self.path = path
        self.started_at = time.time()
        self.deadline = self.started_at + load_config()["common"]["budget_minutes"] * 60
        self.events: list[dict[str, Any]] = []
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            self.started_at = float(old["started_epoch"])
            self.deadline = float(old["deadline_epoch"])
            self.events = old.get("events", [])
        else:
            self.record("budget_started")

    def record(self, event: str, **details: Any) -> None:
        self.events.append({"time": time.time(), "event": event, **details})
        write_json(self.path, {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started_at)),
            "started_epoch": self.started_at,
            "deadline_epoch": self.deadline,
            "budget_minutes": load_config()["common"]["budget_minutes"],
            "elapsed_seconds": round(time.time() - self.started_at, 3),
            "remaining_seconds": round(max(0.0, self.deadline - time.time()), 3),
            "events": self.events,
        })

    def ensure(self, batch_timeout: float = 600, cleanup_reserve: float = 300) -> None:
        remaining = self.deadline - time.time()
        if remaining < batch_timeout + cleanup_reserve:
            raise RuntimeError(f"Stage 3 budget stop: {remaining:.1f}s remaining")


def http_json(url: str, body: dict[str, Any] | None = None, timeout: float = 10, method: str | None = None) -> tuple[int, Any, float]:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method=method or ("POST" if body is not None else "GET"))
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read()
        status = error.code
    elapsed = time.perf_counter() - started
    try:
        decoded: Any = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        decoded = raw.decode("utf-8", errors="replace")
    return status, decoded, elapsed


def stream_chat(base_url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request_body = dict(body)
    request_body["stream"] = True
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    first_token: float | None = None
    chunks: list[dict[str, Any]] = []
    text_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    status: int | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    item = json.loads(data)
                except json.JSONDecodeError:
                    error = f"invalid_sse_json:{data[:200]}"
                    continue
                chunks.append(item)
                choices = item.get("choices") or []
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if first_token is None:
                            first_token = time.perf_counter()
                        text_parts.append(content)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                if item.get("usage"):
                    usage = item["usage"]
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")[:4000]
    except Exception as exc:  # the raw error is part of the evidence
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    text = "".join(text_parts)
    output_tokens = (usage or {}).get("completion_tokens")
    return {
        "status": status,
        "error": error,
        "text": text,
        "chunks": chunks,
        "finish_reason": finish_reason,
        "usage": usage,
        "ttft_seconds": None if first_token is None else first_token - started,
        "e2e_seconds": ended - started,
        "output_tokens": output_tokens,
        "stream_complete": error is None and bool(chunks) and (finish_reason is not None or text != ""),
    }


def chat_request(base_url: str, prompt: str, model: str, max_tokens: int, stream: bool = True, ignore_eos: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
        body["stream"] = True
        if ignore_eos:
            body["ignore_eos"] = True
        return stream_chat(base_url, body, timeout=120)
    status, decoded, elapsed = http_json(f"{base_url}/v1/chat/completions", body, timeout=120)
    if not isinstance(decoded, dict):
        return {"status": status, "error": str(decoded), "text": "", "e2e_seconds": elapsed}
    choices = decoded.get("choices") or []
    text = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
    return {
        "status": status,
        "error": None if status == 200 else decoded.get("error", decoded),
        "text": text,
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "usage": decoded.get("usage"),
        "e2e_seconds": elapsed,
        "stream_complete": status == 200 and bool(text),
    }


PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)\s*$")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')


def parse_prometheus(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = PROM_LINE.match(line)
        if not match:
            continue
        labels = {k: v.replace('\\"', '"').replace('\\\\', '\\') for k, v in LABEL.findall(match.group(2) or "")}
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        rows.append({"name": match.group(1), "labels": labels, "value": value})
    return rows


def metrics_snapshot(base_url: str) -> dict[str, Any]:
    status, body, elapsed = http_json(f"{base_url}/metrics", timeout=10)
    raw = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    rows = parse_prometheus(raw)
    selected = [
        row for row in rows
        if any(term in row["name"].lower() for term in (
            "cache", "preempt", "request_queue", "request_prefill", "request_decode",
            "num_requests", "running", "waiting",
        ))
    ]
    cache_info = next((row for row in rows if row["name"] == "vllm:cache_config_info"), None)
    return {
        "status": status,
        "elapsed_seconds": elapsed,
        "raw": raw,
        "selected": selected,
        "cache_config": None if cache_info is None else cache_info["labels"],
        "captured_at": time.time(),
    }


def gpu_sample() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--query-gpu=timestamp,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {"captured_at": time.time(), "returncode": result.returncode, "csv": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"captured_at": time.time(), "error": f"{type(exc).__name__}: {exc}"}


def host_sample() -> dict[str, Any]:
    sample: dict[str, Any] = {"captured_at": time.time()}
    try:
        sample["loadavg"] = os.getloadavg()
    except (AttributeError, OSError):
        pass
    try:
        meminfo = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            meminfo[key] = value.strip()
        sample["meminfo"] = {key: meminfo.get(key) for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}
    except OSError:
        pass
    return sample


class Telemetry:
    def __init__(self, base_url: str, path: Path, interval: float = 1.0) -> None:
        self.base_url = base_url
        self.path = path
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, name="stage3-telemetry", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=15)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            sample = {"captured_at": time.time(), "gpu": gpu_sample(), "host": host_sample()}
            try:
                sample["metrics"] = metrics_snapshot(self.base_url)
            except Exception as exc:
                sample["metrics_error"] = f"{type(exc).__name__}: {exc}"
            append_jsonl(self.path, sample)
            self.stop_event.wait(self.interval)


class OwnedVllm:
    def __init__(self, candidate: dict[str, Any], run_dir: Path) -> None:
        self.candidate = candidate
        self.run_dir = run_dir
        self.base_url = f"http://{candidate['host']}:{candidate['port']}"
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.model_path: str | None = None

    def _paths(self) -> tuple[str, str, str]:
        venv = os.environ.get("INFERENCE_VENV", "/home/lunacia/.venvs/inference-vllm")
        return f"{venv}/bin/python", f"{venv}/bin/vllm", os.environ.get(
            "PYTHON_DEV_HEADERS", str(ROOT / ".tmp" / "python-dev" / "extracted" / "usr" / "include" / "python3.12")
        )

    def resolve_model(self) -> str:
        if self.model_path:
            return self.model_path
        python, _, _ = self._paths()
        code = (
            "from huggingface_hub import snapshot_download; "
            f"print(snapshot_download(repo_id={self.candidate['model_id']!r}, revision={self.candidate['model_revision']!r}, "
            f"cache_dir={self.candidate['model_cache_dir']!r}))"
        )
        self.model_path = subprocess.check_output([python, "-c", code], text=True).strip().splitlines()[-1]
        return self.model_path

    def start(self, timeout: float = 180) -> dict[str, Any]:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("owned Stage 3 server already running")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        python, vllm, headers = self._paths()
        model_path = self.resolve_model()
        args = [vllm, "serve", model_path,
                "--host", str(self.candidate["host"]), "--port", str(self.candidate["port"]),
                "--served-model-name", f"stage3-{self.candidate['label'].lower()}",
                "--generation-config", "vllm", "--dtype", self.candidate["dtype"],
                "--tensor-parallel-size", str(self.candidate["tensor_parallel_size"]),
                "--max-model-len", str(self.candidate["max_model_len"]),
                "--gpu-memory-utilization", str(self.candidate["gpu_memory_utilization"]),
                "--max-num-seqs", str(self.candidate["max_num_seqs"]),
                "--max-num-batched-tokens", str(self.candidate["max_num_batched_tokens"]),
                "--attention-backend", self.candidate["attention_backend"],
                "--kv-cache-dtype", self.candidate["kv_cache_dtype"],
                "--calculate-kv-scales", "--enable-chunked-prefill", "--enforce-eager"]
        args.append("--enable-prefix-caching" if self.candidate["prefix_caching"] else "--no-enable-prefix-caching")
        env = dict(os.environ)
        env.update({"VLLM_USE_V2_MODEL_RUNNER": self.candidate["vllm_use_v2_model_runner"],
                    "VLLM_USE_FLASHINFER_SAMPLER": self.candidate["vllm_use_flashinfer_sampler"]})
        if Path(headers, "Python.h").exists():
            root = str(Path(headers).parent)
            env["C_INCLUDE_PATH"] = f"{headers}:{root}" + (f":{env['C_INCLUDE_PATH']}" if env.get("C_INCLUDE_PATH") else "")
        self.log_handle = (self.run_dir / "server.log").open("w", encoding="utf-8")
        started = time.time()
        self.process = subprocess.Popen(args, stdout=self.log_handle, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, text=True, start_new_session=True, env=env)
        write_json(self.run_dir / "server-command.json", {"args": args, "env": {key: env.get(key) for key in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_FLASHINFER_SAMPLER", "C_INCLUDE_PATH")}, "started_at": now_utc()})
        deadline = time.time() + timeout
        last_error = ""
        while time.time() < deadline:
            if self.process.poll() is not None:
                self._close_log()
                raise RuntimeError(f"server exited with {self.process.returncode}: {(self.run_dir / 'server.log').read_text(errors='replace')[-4000:]}")
            try:
                status, body, _ = http_json(f"{self.base_url}/health", timeout=3)
                if status == 200:
                    startup = (self.run_dir / "server.log").read_text(errors="replace")
                    write_json(self.run_dir / "startup.json", {"ready_at": now_utc(), "startup_seconds": time.time() - started, "health": body, "log_tail": startup[-16000:]})
                    return {"status": "ready", "startup_seconds": time.time() - started, "pid": self.process.pid, "args": args}
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
        raise TimeoutError(f"server readiness timeout: {last_error}")

    def stop(self) -> dict[str, Any]:
        if self.process is None:
            return {"status": "not_started"}
        pid = self.process.pid
        started = time.time()
        if self.process.poll() is None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (AttributeError, ProcessLookupError, PermissionError):
                self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (AttributeError, ProcessLookupError, PermissionError):
                    self.process.kill()
                self.process.wait(timeout=15)
        self._close_log()
        result = {"pid": pid, "returncode": self.process.returncode, "stop_seconds": time.time() - started}
        write_json(self.run_dir / "stop.json", result)
        return result

    def _close_log(self) -> None:
        if self.log_handle is not None:
            self.log_handle.flush()
            self.log_handle.close()
            self.log_handle = None


def reset_prefix_cache(base_url: str) -> dict[str, Any]:
    status, body, elapsed = http_json(f"{base_url}/reset_prefix_cache", body={}, timeout=20)
    return {"status": status, "body": body, "elapsed_seconds": elapsed, "ok": status == 200}


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    def totals(snapshot: dict[str, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        for row in snapshot.get("selected", []):
            if not row["labels"]:
                result[row["name"]] = result.get(row["name"], 0.0) + row["value"]
        return result
    left, right = totals(before), totals(after)
    return {key: right.get(key, 0.0) - left.get(key, 0.0) for key in sorted(set(left) | set(right))}
