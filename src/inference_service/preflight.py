"""Small, CPU-only checks for the supported serving configuration."""
import importlib.metadata
import json
from pathlib import Path
import sys

from .config import Settings


def check_model(model: str):
    if model == Settings.model:
        return "Pinned Qwen3-4B repository (weights may download on startup)"
    folder = Path(model).expanduser()
    if not folder.is_dir():
        raise ValueError("Use Qwen/Qwen3-4B or an existing local Qwen3-4B directory (--model PATH)")
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        expected = {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36,
                    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Only the Qwen3-4B architecture is supported in this release")
        if config.get("quantization_config") or config.get("use_sliding_window") or config.get("rope_scaling"):
            raise ValueError("Use the original unquantized Qwen3-4B configuration")
        index = folder / "model.safetensors.index.json"
        weights = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()) if index.exists() else {"model.safetensors"}
        if not weights or any(not (folder / name).is_file() for name in weights):
            raise ValueError("Model weights are incomplete; finish downloading all safetensors shards")
        if not (folder / "tokenizer.json").is_file() or not (folder / "tokenizer_config.json").is_file():
            raise ValueError("Model directory must include tokenizer.json and tokenizer_config.json")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("Cannot read model configuration/index; use a complete Hugging Face snapshot") from exc
    return f"Local Qwen3-4B structure and files checked: {folder} (weight identity is not verified)"


def check_runtime(settings: Settings):
    if sys.platform != "linux":
        raise ValueError("Run serve/doctor on Linux or inside WSL2; api-only is portable")
    for package, expected in [("vllm", "0.23.0"), ("transformers", "4.57.6")]:
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"Missing {package}; install the pinned runtime (see docs/installation.md)") from exc
        if actual != expected:
            raise ValueError(f"Expected {package}=={expected}, found {actual}")
    return check_model(settings.model)
