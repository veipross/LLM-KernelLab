"""Triton implementation of fused residual addition and RMSNorm inference."""

from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl

from llm_kernels.torch_ops.fused_residual_rms_norm import (
    _validate_fused_residual_rms_norm_inputs,
)


# This is a conservative cross-dtype limit. A 16384-element FP32 row is
# 64 KiB, matching the working-set bound used by common one-program-per-row
# Triton normalization kernels. The boundary is covered by GPU tests.
_MAX_FUSED_HIDDEN_SIZE = 16384


@triton.jit
def _fused_residual_rms_norm_forward_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    normalized_ptr,
    residual_output_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    Compute one fused residual-add and RMSNorm row per Triton program.

    The sum is explicitly cast back to the input element type before it is
    widened to FP32 for the RMSNorm reduction.
    """

    row_index = tl.program_id(axis=0)

    column_offsets = tl.arange(0, block_size)
    valid_mask = column_offsets < n_cols
    row_start = row_index * n_cols

    x = tl.load(
        x_ptr + row_start + column_offsets,
        mask=valid_mask,
        other=0.0,
    )
    residual = tl.load(
        residual_ptr + row_start + column_offsets,
        mask=valid_mask,
        other=0.0,
    )

    residual_output = (x + residual).to(
        residual_output_ptr.dtype.element_ty
    )

    tl.store(
        residual_output_ptr + row_start + column_offsets,
        residual_output,
        mask=valid_mask,
    )

    residual_fp32 = residual_output.to(tl.float32)
    weight_fp32 = tl.load(
        weight_ptr + column_offsets,
        mask=valid_mask,
        other=0.0,
    ).to(tl.float32)

    square_sum = tl.sum(
        residual_fp32 * residual_fp32,
        axis=0,
    )
    mean_square = square_sum / n_cols
    inverse_rms = tl.rsqrt(mean_square + eps)

    normalized = residual_fp32 * inverse_rms * weight_fp32

    tl.store(
        normalized_ptr + row_start + column_offsets,
        normalized,
        mask=valid_mask,
    )


def _validate_triton_inputs(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> None:
    """Validate constraints specific to the Triton implementation."""

    _validate_fused_residual_rms_norm_inputs(
        x,
        residual,
        weight,
        eps,
    )

    if not x.is_cuda or not residual.is_cuda or not weight.is_cuda:
        raise ValueError(
            "x, residual and weight must all be CUDA tensors."
        )

    for name, tensor in (
        ("x", x),
        ("residual", residual),
        ("weight", weight),
    ):
        if not tensor.is_contiguous():
            raise ValueError(
                "The first Triton Fused Residual + RMSNorm version "
                f"requires contiguous {name}."
            )

    hidden_size = x.shape[-1]
    if hidden_size > _MAX_FUSED_HIDDEN_SIZE:
        raise ValueError(
            f"hidden_size={hidden_size} exceeds the supported limit "
            f"of {_MAX_FUSED_HIDDEN_SIZE} for the one-program-per-row "
            "Triton kernel."
        )


def fused_residual_rms_norm_triton(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Apply fused residual addition and RMSNorm using one Triton kernel launch.

    The operation is inference-only and non-in-place. It first rounds
    ``x + residual`` to ``x.dtype`` and stores that value as
    ``residual_out``. The same rounded value is widened to FP32 for all
    RMSNorm arithmetic. The normalized result is cast back to ``x.dtype`` by
    the output store.

    Inputs must be contiguous CUDA tensors with matching shape, dtype and
    device. Supported dtypes are FP16, BF16 and FP32, and the hidden size must
    not exceed 16384.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(normalized, residual_out)`` with the same shape, dtype and device
        as ``x``. Both outputs use newly allocated, non-aliased storage.
    """

    _validate_triton_inputs(x, residual, weight, eps)

    normalized = torch.empty_like(x)
    residual_output = torch.empty_like(x)

    hidden_size = x.shape[-1]
    row_count = x.numel() // hidden_size

    if row_count == 0:
        return normalized, residual_output

    block_size = triton.next_power_of_2(hidden_size)

    if block_size <= 2048:
        num_warps = 4
    elif block_size <= 8192:
        num_warps = 8
    else:
        num_warps = 16

    grid = (row_count,)

    _fused_residual_rms_norm_forward_kernel[grid](
        x,
        residual,
        weight,
        normalized,
        residual_output,
        n_cols=hidden_size,
        eps=eps,
        block_size=block_size,
        num_warps=num_warps,
    )

    return normalized, residual_output


__all__ = [
    "fused_residual_rms_norm_triton",
]
