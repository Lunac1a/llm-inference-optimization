"""Owned vLLM lifecycle and explicit runtime profiles."""
from contextlib import contextmanager
import importlib.metadata
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx
from .config import Settings


def backend_address(settings: Settings):
    address = urlparse(settings.backend_url)
    if address.scheme != "http" or address.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("Owned backends use loopback HTTP; use the api command for an external backend")
    return address.hostname, address.port or 80


def occupied(host: str, port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def launch_spec(settings: Settings) -> tuple[list[str], dict[str, str]]:
    host, port = backend_address(settings)
    full_hybrid = settings.profile == "hybrid"
    chunked_hybrid = settings.profile == "chunked-hybrid"
    hybrid = full_hybrid or chunked_hybrid
    context = 16640 if hybrid else 32768
    args = [sys.executable, "-m", "inference_service.vllm_entry", "serve", settings.model,
            "--revision", settings.revision, "--served-model-name", settings.served_model,
            "--host", host, "--port", str(port), "--generation-config", "vllm",
            "--dtype", "bfloat16", "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", "0.8", "--max-model-len", str(context),
            "--max-num-seqs", "8",
            "--max-num-batched-tokens", str(context if full_hybrid else 2048),
            "--attention-backend", "TRITON_ATTN" if hybrid else "FLASH_ATTN",
            "--kv-cache-dtype", "fp8_per_token_head" if hybrid else "bfloat16",
            "--calculate-kv-scales", "--enforce-eager",
            "--no-enable-chunked-prefill" if full_hybrid else "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--reasoning-parser", "qwen3", "--reasoning-config",
            '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}',
            "--enable-prompt-tokens-details"]
    if chunked_hybrid:
        args += ["--kv-cache-memory-bytes", str(4 * 2**30)]
    plugin = {"hybrid": "inference_hybrid_prefill",
              "chunked-hybrid": "inference_chunked_prefill"}.get(settings.profile, "")
    env = dict(os.environ)
    env.update(VLLM_PLUGINS=plugin, INFERENCE_RUNTIME_PROFILE=settings.profile,
               VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_FLASHINFER_SAMPLER="0",
               VLLM_BATCH_INVARIANT="0", VLLM_USE_SIMPLE_KV_OFFLOAD="0")
    if chunked_hybrid:
        env["VLLM_KV_CACHE_LAYOUT"] = "NHD"
    # Clear inherited backend authentication when no backend key is configured.
    if settings.backend_api_key:
        env["VLLM_API_KEY"] = settings.backend_api_key
    else:
        env.pop("VLLM_API_KEY", None)
    headers = os.getenv("PYTHON_DEV_HEADERS")
    if headers:
        env["C_INCLUDE_PATH"] = os.pathsep.join(filter(None, (headers, str(Path(headers).parent), env.get("C_INCLUDE_PATH"))))
    return args, env


class OwnedBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process = None
        self.log = None

    def start(self, timeout: float = 180):
        if sys.platform != "linux":
            raise RuntimeError("Run the GPU backend on Linux/WSL; the API-only command is portable")
        if self.process is not None:
            raise RuntimeError("Backend already started")
        if importlib.metadata.version("vllm") != "0.23.0":
            raise RuntimeError("This runtime is pinned to vLLM 0.23.0")
        host, port = backend_address(self.settings)
        if occupied(host, port):
            raise RuntimeError(f"Backend port {port} is occupied; no process was replaced")
        args, env = launch_spec(self.settings)
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.settings.log_dir / f"vllm-{time.time_ns()}.log"
        self.log = self.log_path.open("x", encoding="utf-8")
        try:
            self.process = subprocess.Popen(args, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
            headers = {"Authorization": f"Bearer {self.settings.backend_api_key}"} if self.settings.backend_api_key else {}
            with httpx.Client(timeout=2, headers=headers, trust_env=False) as client:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(f"Backend exited; inspect {self.log_path}")
                    try:
                        if client.get(self.settings.backend_url.rstrip("/") + "/health").status_code == 200:
                            return
                    except httpx.HTTPError:
                        pass
                    time.sleep(.5)
            raise TimeoutError(f"Backend startup timed out; inspect {self.log_path}")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process is not None:
            # Signal the entire owned group even if its leader has already exited.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None


@contextmanager
def owned_backend(settings: Settings):
    backend = OwnedBackend(settings)
    try:
        backend.start()
        yield backend
    finally:
        backend.stop()
