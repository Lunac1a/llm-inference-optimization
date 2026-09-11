"""Entry points are inert unless the owned runtime explicitly selects a profile."""
import os


def register_hybrid():
    if os.environ.get("INFERENCE_RUNTIME_PROFILE") == "hybrid":
        from .backends.hybrid import register
        register()


def register_cpu_kv():
    if os.environ.get("INFERENCE_RUNTIME_PROFILE") == "cpu-kv":
        from .backends.cpu_kv import install
        install()
