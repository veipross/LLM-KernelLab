"""Triton implementation of RMSNorm forward inference."""

from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl


_MAX_FUSED_HIDDEN_SIZE = 65536


@triton.jit
def _rms_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    Compute one RMSNorm row per Triton program.

    All reduction arithmetic is performed in FP32.
    """

    row_index = tl.program_id(axis=0)

    column_offsets = tl.arange(0, block_size)
    valid_mask = column_offsets < n_cols

    row_start = row_index * n_cols

    x = tl.load(
        x_ptr + row_start + column_offsets,
        mask=valid_mask,
        other=0.0,
    ).to(tl.float32)

    weight = tl.load(
        weight_ptr + column_offsets,
        mask=valid_mask,
        other=0.0,
    ).to(tl.float32)

    square_sum = tl.sum(x * x, axis=0)
    mean_square = square_sum / n_cols
    inverse_rms = tl.rsqrt(mean_square + eps)

    output = x * inverse_rms * weight

    tl.store(
        output_ptr + row_start + column_offsets,
        output,
        mask=valid_mask,
    )


def _validate_inputs(x: Tensor, weight: Tensor, eps: float) -> None:
    """Validate inputs accepted by the first Triton RMSNorm version."""

    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must both be CUDA tensors.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if x.device != weight.device:
        raise ValueError(
            f"x and weight must be on the same device, got {x.device} "
            f"and {weight.device}."
        )

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension.")

    if weight.ndim != 1:
        raise ValueError(
            f"weight must be one-dimensional, got {tuple(weight.shape)}."
        )

    if weight.shape[0] != x.shape[-1]:
        raise ValueError(
            "The weight size must equal the final dimension of x: "
            f"weight={weight.shape[0]}, hidden_size={x.shape[-1]}."
        )

    if x.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }:
        raise TypeError(
            "Supported x dtypes are float16, bfloat16 and float32, "
            f"but got {x.dtype}."
        )

    if weight.dtype != x.dtype:
        raise TypeError(
            f"x and weight must have the same dtype, got "
            f"{x.dtype} and {weight.dtype}."
        )

    if not x.is_contiguous():
        raise ValueError(
            "The first Triton RMSNorm version requires contiguous x."
        )

    if not weight.is_contiguous():
        raise ValueError(
            "The first Triton RMSNorm version requires contiguous weight."
        )

    if eps <= 0:
        raise ValueError(f"eps must be positive, but got {eps}.")

    hidden_size = x.shape[-1]
    if hidden_size > _MAX_FUSED_HIDDEN_SIZE:
        raise ValueError(
            f"hidden_size={hidden_size} exceeds the first-version limit "
            f"of {_MAX_FUSED_HIDDEN_SIZE}."
        )


def rms_norm_triton(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Apply RMSNorm over the final tensor dimension using Triton.

    This first implementation is a forward-inference kernel. All leading
    dimensions are flattened into rows while the final dimension is treated
    as the hidden dimension.

    Parameters
    ----------
    x:
        Contiguous CUDA tensor with shape ``(..., hidden_size)``.
    weight:
        Contiguous CUDA tensor with shape ``(hidden_size,)``.
    eps:
        Numerical-stability constant.

    Returns
    -------
    Tensor
        Output tensor with the same shape, dtype and device as ``x``.
    """

    _validate_inputs(x, weight, eps)

    hidden_size = x.shape[-1]
    row_count = x.numel() // hidden_size
    block_size = triton.next_power_of_2(hidden_size)

    output = torch.empty_like(x)

    grid = (row_count,)

    # The selected warp count is a conservative starting point.
    # It will later be tuned through benchmark experiments.
    if block_size <= 2048:
        num_warps = 4
    elif block_size <= 8192:
        num_warps = 8
    else:
        num_warps = 16

    _rms_norm_forward_kernel[grid](
        x,
        weight,
        output,
        n_cols=hidden_size,
        eps=eps,
        block_size=block_size,
        num_warps=num_warps,
    )

    return output


__all__ = ["rms_norm_triton"]
