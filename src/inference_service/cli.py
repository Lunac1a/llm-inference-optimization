import argparse
from dataclasses import replace
import signal
import time

from .config import PROFILES, Settings


def main():
    parser = argparse.ArgumentParser(description="Local inference API and opt-in vLLM runtime")
    parser.add_argument("command", choices=("serve", "api", "backend"),
                        help="serve: owned backend + API; api: existing backend; backend: vLLM only")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--batch-invariant", action="store_true", default=None)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--backend-url")
    args = parser.parse_args()
    settings = Settings.from_env()
    overrides = {name: getattr(args, name) for name in ("profile", "batch_invariant", "host", "port", "backend_url")
                 if getattr(args, name) is not None}
    settings = replace(settings, **overrides)
    from .api import create_app
    from .runtime import backend_address, occupied, owned_backend
    import uvicorn

    if args.command != "backend" and occupied(settings.host, settings.port):
        parser.error("API port is occupied; no existing process was replaced")
    if args.command == "serve" and backend_address(settings)[1] == settings.port:
        parser.error("API and backend must use different ports")
    if args.command == "api":
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
        return

    def interrupt(*_):
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with owned_backend(settings) as backend:
            if args.command == "serve":
                uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
            else:
                print(f"Backend ready at {settings.backend_url}", flush=True)
                while backend.process.poll() is None:
                    time.sleep(.5)
                raise RuntimeError("Backend exited unexpectedly")
    except KeyboardInterrupt:
        pass
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
