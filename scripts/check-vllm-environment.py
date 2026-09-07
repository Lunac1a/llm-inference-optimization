"""Validate installed runtime and a real vLLM GPU kernel, without model weights."""

import importlib.metadata
import json
import platform
import sys

import torch
import vllm
from vllm import _custom_ops as ops


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable on this GPU")

    # A deterministic matrix product checks CUDA execution, not only discovery.
    x = torch.ones((256, 256), device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(x @ x, torch.full_like(x, 256))

    # Exercise vLLM's compiled extension against an independent PyTorch reference.
    torch.manual_seed(42)
    values = torch.randn((8, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(values)
    eps = 1e-6
    ops.rms_norm(output, values, weight, eps)
    reference = values.float() * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + eps)
    torch.testing.assert_close(output.float(), reference, rtol=0.01, atol=0.01)
    torch.cuda.synchronize()

    packages = ["vllm", "torch", "transformers", "triton", "ninja", "fastapi", "openai", "huggingface-hub"]
    print(json.dumps({
        "status": "passed",
        "python": platform.python_version(),
        "executable": sys.executable,
        "kernel": platform.release(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "checks": ["cuda_available", "bf16_supported", "cuda_matmul", "vllm_rms_norm"],
        "model_loaded": False,
        "api_generation_verified": False,
        "performance_baseline_measured": False,
    }, indent=2))


if __name__ == "__main__":
    main()
