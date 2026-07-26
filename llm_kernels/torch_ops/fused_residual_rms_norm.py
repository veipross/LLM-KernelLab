"""PyTorch reference implementation of fused residual addition and RMSNorm."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


def _validate_fused_residual_rms_norm_inputs(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> None:
    """Validate inputs shared by the reference and Triton implementations."""

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension.")

    if residual.shape != x.shape:
        raise ValueError(
            "x and residual must have the same shape, got "
            f"{tuple(x.shape)} and {tuple(residual.shape)}."
        )

    if weight.ndim != 1:
        raise ValueError(
            f"weight must be one-dimensional, got {tuple(weight.shape)}."
        )

    hidden_size = x.shape[-1]
    if hidden_size == 0:
        raise ValueError("The hidden dimension must be non-empty.")

    if weight.shape[0] != hidden_size:
        raise ValueError(
            "The weight size must equal the final dimension of x: "
            f"weight={weight.shape[0]}, hidden_size={hidden_size}."
        )

    if x.device != residual.device or x.device != weight.device:
        raise ValueError(
            "x, residual and weight must be on the same device, got "
            f"{x.device}, {residual.device} and {weight.device}."
        )

    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "Supported dtypes are float16, bfloat16 and float32, "
            f"but got x.dtype={x.dtype}."
        )

    if residual.dtype != x.dtype or weight.dtype != x.dtype:
        raise TypeError(
            "x, residual and weight must have the same dtype, got "
            f"{x.dtype}, {residual.dtype} and {weight.dtype}."
        )

    for name, tensor in (
        ("x", x),
        ("residual", residual),
        ("weight", weight),
    ):
        if tensor.requires_grad:
            raise RuntimeError(
                "Fused Residual + RMSNorm is inference-only; "
                f"{name}.requires_grad must be False because backward "
                "is not implemented."
            )

    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(
            f"eps must be finite and positive, but got {eps}."
        )


def fused_residual_rms_norm_reference(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Add a residual tensor and apply RMSNorm using explicit PyTorch operations.

    The operation has the following exact semantic order::

        residual_out = (x + residual).to(x.dtype)
        residual_fp32 = residual_out.float()
        inverse_rms = rsqrt(mean(residual_fp32 ** 2) + eps)
        normalized = (
            residual_fp32 * inverse_rms * weight.float()
        ).to(x.dtype)

    The dtype conversion after the addition is semantically significant:
    RMSNorm consumes the rounded ``residual_out``, not a higher-precision
    intermediate sum.

    This first version is inference-only and non-in-place. Inputs that require
    gradients are rejected explicitly. Neither output aliases an input.

    Parameters
    ----------
    x:
        Input tensor with shape ``(..., hidden_size)``.
    residual:
        Residual tensor with the same shape, dtype and device as ``x``.
    weight:
        One-dimensional RMSNorm weight with length ``hidden_size`` and the
        same dtype and device as ``x``.
    eps:
        Finite positive numerical-stability constant.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(normalized, residual_out)``. Both tensors have the same shape,
        dtype and device as ``x`` and own storage separate from the inputs.
    """

    _validate_fused_residual_rms_norm_inputs(
        x,
        residual,
        weight,
        eps,
    )

    residual_out = torch.add(x, residual).to(dtype=x.dtype)

    residual_fp32 = residual_out.float()
    weight_fp32 = weight.float()

    mean_square = residual_fp32.square().mean(
        dim=-1,
        keepdim=True,
    )
    inverse_rms = torch.rsqrt(mean_square + eps)

    normalized_fp32 = residual_fp32 * inverse_rms * weight_fp32
    normalized = normalized_fp32.to(dtype=x.dtype)

    return normalized, residual_out


class TorchFusedResidualRMSNorm(nn.Module):
    """Inference-only module backed by the fused PyTorch reference function."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, but got {hidden_size}."
            )

        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(
                f"eps must be finite and positive, but got {eps}."
            )

        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype),
            requires_grad=False,
        )

    def forward(
        self,
        x: Tensor,
        residual: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return fused_residual_rms_norm_reference(
            x,
            residual,
            self.weight,
            self.eps,
        )


__all__ = [
    "TorchFusedResidualRMSNorm",
    "fused_residual_rms_norm_reference",
]
