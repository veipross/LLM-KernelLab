"""Run a simple RMSNorm correctness smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow this script to be executed directly from the scripts directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from llm_kernels.torch_ops import rms_norm_reference
from llm_kernels.triton_ops import rms_norm_triton


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    torch.manual_seed(2026)

    device = torch.device("cuda")
    dtype = torch.float16
    shape = (2, 128, 4096)
    eps = 1e-6

    x = torch.randn(
        shape,
        device=device,
        dtype=dtype,
    )

    weight = torch.randn(
        shape[-1],
        device=device,
        dtype=dtype,
    )

    expected = rms_norm_reference(
        x,
        weight,
        eps,
    )

    actual = rms_norm_triton(
        x,
        weight,
        eps,
    )

    torch.cuda.synchronize()

    difference = (
        actual.float() - expected.float()
    ).abs()

    max_abs_error = difference.max().item()
    mean_abs_error = difference.mean().item()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-3,
        atol=2e-3,
    )

    print("=" * 60)
    print("RMSNorm Smoke Test")
    print("=" * 60)
    print(f"shape:              {shape}")
    print(f"dtype:              {dtype}")
    print(f"device:             {device}")
    print(f"max_abs_error:      {max_abs_error:.8f}")
    print(f"mean_abs_error:     {mean_abs_error:.8f}")
    print("correctness:        PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
