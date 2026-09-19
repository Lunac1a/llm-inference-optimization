"""Runtime settings, independent of experiment folders and personal paths."""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

Profile = Literal["document", "hybrid", "chunked-hybrid"]
PROFILES = ("document", "hybrid", "chunked-hybrid")


@dataclass(frozen=True)
class Settings:
    model: str = "Qwen/Qwen3-4B"
    revision: str = "1cfa9a7208912126459214e8b04321603b3df60c"
    served_model: str = "qwen3-local"
    host: str = "127.0.0.1"
    port: int = 8000
    backend_url: str = "http://127.0.0.1:8001"
    profile: Profile = "chunked-hybrid"
    request_timeout: float = 120
    max_inflight: int = 8
    api_key: str = ""
    backend_api_key: str = ""
    log_dir: Path = Path(".local/logs")

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError("Unknown runtime profile")
        parsed = urlparse(self.backend_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("backend_url must be an HTTP(S) origin without credentials")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("backend_url must be an origin, without a path or query")
        if not 1 <= self.port <= 65535 or self.request_timeout <= 0 or self.max_inflight < 1:
            raise ValueError("Invalid port or request timeout")

    @classmethod
    def from_env(cls):
        return cls(model=os.getenv("INFERENCE_MODEL", cls.model),
                   revision=os.getenv("INFERENCE_MODEL_REVISION", cls.revision),
                   served_model=os.getenv("INFERENCE_SERVED_MODEL", cls.served_model),
                   host=os.getenv("INFERENCE_HOST", cls.host),
                   port=int(os.getenv("INFERENCE_PORT", cls.port)),
                   backend_url=os.getenv("INFERENCE_BACKEND_URL", cls.backend_url),
                   profile=os.getenv("INFERENCE_PROFILE", cls.profile),
                   request_timeout=float(os.getenv("INFERENCE_REQUEST_TIMEOUT", cls.request_timeout)),
                   max_inflight=int(os.getenv("INFERENCE_MAX_INFLIGHT", cls.max_inflight)),
                   api_key=os.getenv("INFERENCE_API_KEY", ""),
                   backend_api_key=os.getenv("INFERENCE_BACKEND_API_KEY", ""),
                   log_dir=Path(os.getenv("INFERENCE_LOG_DIR", ".local/logs")))
