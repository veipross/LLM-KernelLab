from __future__ import annotations

import platform
import sys

import torch
import triton


def main() -> None:
    print("=" * 60)
    print("LLM-KernelLab Environment")
    print("=" * 60)

    print(f"Python:             {sys.version.split()[0]}")
    print(f"Platform:           {platform.platform()}")
    print(f"PyTorch:            {torch.__version__}")
    print(f"PyTorch CUDA:       {torch.version.cuda}")
    print(f"Triton:             {triton.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current environment.")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    print(f"GPU:                {properties.name}")
    print(
        "Compute capability:"
        f" {properties.major}.{properties.minor}"
    )
    print(
        "GPU memory:        "
        f"{properties.total_memory / 1024**3:.2f} GiB"
    )
    print(f"Multiprocessors:     {properties.multi_processor_count}")

    x = torch.randn(1024, device=device, dtype=torch.float16)
    y = x * 2.0
    torch.cuda.synchronize()

    expected = x * 2.0
    if not torch.allclose(y, expected):
        raise RuntimeError("Basic CUDA calculation failed.")

    print("CUDA smoke test:    PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
