import argparse
from dataclasses import replace
import signal
import time

from .config import PROFILES, Settings


def main():
    parser = argparse.ArgumentParser(description="Qwen3-4B mixed KV inference service")
    parser.add_argument("command", choices=("serve", "api", "backend", "doctor"),
                        help="serve: owned backend + API; api: existing backend; backend: vLLM only")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--backend-url")
    parser.add_argument("--model", help="Local Qwen3-4B snapshot directory or Qwen/Qwen3-4B")
    args = parser.parse_args()
    overrides = {name: getattr(args, name) for name in ("profile", "host", "port", "backend_url", "model")
                 if getattr(args, name) is not None}
    try:
        settings = replace(Settings.from_env(), **overrides)
        if args.command != "api":
            from .preflight import check_runtime
            print(check_runtime(settings), flush=True)
        if args.command == "doctor":
            print("Runtime checks passed. No model loaded; GPU capacity and generation are checked on serve.")
            return
        if args.command != "backend" and settings.host not in ("127.0.0.1", "localhost", "::1") and not settings.api_key:
            raise ValueError("Set INFERENCE_API_KEY before binding the API to a non-loopback address")
    except ValueError as exc:
        parser.error(str(exc))
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
                print(f"API: http://{settings.host}:{settings.port}/v1 | model: {settings.served_model} | profile: {settings.profile}", flush=True)
                uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
            else:
                print(f"Backend ready at {settings.backend_url}", flush=True)
                while backend.process.poll() is None:
                    time.sleep(.5)
                raise RuntimeError("Backend exited unexpectedly")
    except KeyboardInterrupt:
        pass
    except (RuntimeError, TimeoutError, OSError) as exc:
        parser.exit(1, f"Service could not run: {exc}\n")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
