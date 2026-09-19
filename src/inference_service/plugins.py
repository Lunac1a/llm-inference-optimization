"""Entry points are inert unless the owned runtime explicitly selects a profile."""
import os


def register_hybrid():
    if os.environ.get("INFERENCE_RUNTIME_PROFILE") == "hybrid":
        from .backends.hybrid import register
        register()


def register_chunked():
    if os.environ.get("INFERENCE_RUNTIME_PROFILE") == "chunked-hybrid":
        from .backends.chunked_hybrid import register
        register()
