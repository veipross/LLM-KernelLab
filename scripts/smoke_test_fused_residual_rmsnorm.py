"""Run a correctness smoke test for Fused Residual + RMSNorm."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from llm_kernels.torch_ops import (
    fused_residual_rms_norm_reference,
)
from llm_kernels.triton_ops import fused_residual_rms_norm_triton


def error_statistics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    """Return maximum and mean absolute errors in FP32."""

    difference = (
        actual.float() - expected.float()
    ).abs()
    return difference.max().item(), difference.mean().item()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    torch.manual_seed(2026)

    device = torch.device("cuda")
    dtype = torch.float16
    shape = (2, 128, 4096)
    eps = 1e-6

    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn(shape[-1], device=device, dtype=dtype)

    expected_normalized, expected_residual = (
        fused_residual_rms_norm_reference(
            x,
            residual,
            weight,
            eps,
        )
    )
    actual_normalized, actual_residual = (
        fused_residual_rms_norm_triton(
            x,
            residual,
            weight,
            eps,
        )
    )

    torch.cuda.synchronize()

    normalized_max_error, normalized_mean_error = error_statistics(
        actual_normalized,
        expected_normalized,
    )
    residual_max_error, residual_mean_error = error_statistics(
        actual_residual,
        expected_residual,
    )

    torch.testing.assert_close(
        actual_normalized,
        expected_normalized,
        rtol=2e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        actual_residual,
        expected_residual,
        rtol=0.0,
        atol=0.0,
    )

    print("=" * 72)
    print("Fused Residual + RMSNorm Smoke Test")
    print("=" * 72)
    print(f"shape:                       {shape}")
    print(f"dtype:                       {dtype}")
    print(f"device:                      {device}")
    print(
        "normalized max abs error:    "
        f"{normalized_max_error:.8f}"
    )
    print(
        "normalized mean abs error:   "
        f"{normalized_mean_error:.8f}"
    )
    print(
        "residual_out max abs error:  "
        f"{residual_max_error:.8f}"
    )
    print(
        "residual_out mean abs error: "
        f"{residual_mean_error:.8f}"
    )
    print("correctness:                 PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
