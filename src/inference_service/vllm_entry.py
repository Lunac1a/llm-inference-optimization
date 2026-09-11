"""Process-local entry point for the pinned runtime; no global patch installation."""
from vllm.entrypoints.cli.main import main

if __name__ == "__main__":
    main()
