"""PyTorch reference implementations of RMSNorm."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate_inputs(x: Tensor, weight: Tensor) -> None:
    """Validate RMSNorm input tensors."""

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension.")

    if weight.ndim != 1:
        raise ValueError(
            f"weight must be one-dimensional, but got shape {tuple(weight.shape)}."
        )

    if weight.shape[0] != x.shape[-1]:
        raise ValueError(
            "The weight size must equal the final dimension of x: "
            f"weight={weight.shape[0]}, hidden_size={x.shape[-1]}."
        )

    if x.device != weight.device:
        raise ValueError(
            f"x and weight must be on the same device, got {x.device} "
            f"and {weight.device}."
        )

    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, but got dtype={x.dtype}.")

    if not weight.is_floating_point():
        raise TypeError(
            f"weight must be floating point, but got dtype={weight.dtype}."
        )


def rms_norm_reference(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute RMSNorm using explicit PyTorch operations.

    The reduction is performed in FP32 to improve numerical stability:

        rms = sqrt(mean(x ** 2) + eps)
        output = x / rms * weight

    Parameters
    ----------
    x:
        Input tensor. RMSNorm is applied over the final dimension.
    weight:
        One-dimensional affine weight with length ``x.shape[-1]``.
    eps:
        Numerical-stability constant.

    Returns
    -------
    Tensor
        Tensor with the same shape and dtype as ``x``.
    """

    _validate_inputs(x, weight)

    if eps <= 0:
        raise ValueError(f"eps must be positive, but got {eps}.")

    input_dtype = x.dtype

    x_fp32 = x.float()
    weight_fp32 = weight.float()

    mean_square = x_fp32.square().mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(mean_square + eps)

    output_fp32 = x_fp32 * inverse_rms * weight_fp32
    return output_fp32.to(dtype=input_dtype)


class TorchRMSNorm(nn.Module):
    """Small RMSNorm module backed by :func:`rms_norm_reference`."""

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

        if eps <= 0:
            raise ValueError(f"eps must be positive, but got {eps}.")

        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm_reference(x, self.weight, self.eps)


__all__ = [
    "TorchRMSNorm",
    "rms_norm_reference",
]
