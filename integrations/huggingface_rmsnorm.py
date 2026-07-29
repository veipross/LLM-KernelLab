"""Hugging Face RMSNorm integration backed by the LLM-KernelLab Triton kernel."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from llm_kernels.triton_ops import rms_norm_triton


_SUPPORTED_HF_RMSNORM_TYPES = (
    LlamaRMSNorm,
    Qwen2RMSNorm,
)

_SUPPORTED_TRITON_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


def _huggingface_rms_norm_reference(
    hidden_states: Tensor,
    weight: Tensor,
    eps: float,
) -> Tensor:
    """Match the RMSNorm calculation order used by Llama and Qwen2."""

    input_dtype = hidden_states.dtype
    hidden_states_fp32 = hidden_states.to(torch.float32)
    variance = hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized_fp32 = hidden_states_fp32 * torch.rsqrt(variance + eps)

    # Keep this multiplication order consistent with Hugging Face.
    return weight * normalized_fp32.to(input_dtype)


def _get_module_eps(module: nn.Module) -> float:
    """Read the epsilon attribute used by a Hugging Face RMSNorm module."""

    if hasattr(module, "variance_epsilon"):
        return float(module.variance_epsilon)

    if hasattr(module, "eps"):
        return float(module.eps)

    raise TypeError(
        f"Cannot determine RMSNorm epsilon from {type(module).__name__}."
    )


class HuggingFaceTritonRMSNorm(nn.Module):
    """
    Drop-in RMSNorm module for Hugging Face Llama and Qwen2 models.

    In CUDA inference mode, supported contiguous inputs are executed by the
    LLM-KernelLab Triton RMSNorm kernel. Other cases use a PyTorch fallback
    matching the Hugging Face computation order, so CPU execution and
    gradient-based workflows remain safe.
    """

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
        self.variance_epsilon = float(eps)
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )

    @classmethod
    def from_huggingface_module(
        cls,
        module: nn.Module,
    ) -> "HuggingFaceTritonRMSNorm":
        """Create an adapter while preserving the original weight parameter."""

        if not isinstance(module, _SUPPORTED_HF_RMSNORM_TYPES):
            supported_names = ", ".join(
                module_type.__name__
                for module_type in _SUPPORTED_HF_RMSNORM_TYPES
            )
            raise TypeError(
                f"Expected one of ({supported_names}), but got "
                f"{type(module).__name__}."
            )

        if module.weight.ndim != 1:
            raise ValueError(
                "Hugging Face RMSNorm weight must be one-dimensional, got "
                f"{tuple(module.weight.shape)}."
            )

        adapter = cls(
            hidden_size=module.weight.numel(),
            eps=_get_module_eps(module),
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        # Preserve the exact Parameter object and therefore the state_dict key,
        # optimizer references, device, dtype and loaded model values.
        adapter.weight = module.weight
        adapter.train(module.training)
        return adapter

    def _can_use_triton(self, hidden_states: Tensor) -> bool:
        """Return whether the current call satisfies the Triton fast path."""

        return (
            not torch.is_grad_enabled()
            and hidden_states.is_cuda
            and self.weight.is_cuda
            and hidden_states.device == self.weight.device
            and hidden_states.dtype == self.weight.dtype
            and hidden_states.dtype in _SUPPORTED_TRITON_DTYPES
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim < 1:
            raise ValueError(
                "hidden_states must have at least one dimension."
            )

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "The final hidden dimension does not match the RMSNorm "
                f"weight: hidden={hidden_states.shape[-1]}, "
                f"weight={self.hidden_size}."
            )

        if self._can_use_triton(hidden_states):
            return rms_norm_triton(
                hidden_states.contiguous(),
                self.weight.contiguous(),
                self.variance_epsilon,
            )

        return _huggingface_rms_norm_reference(
            hidden_states,
            self.weight,
            self.variance_epsilon,
        )

    def extra_repr(self) -> str:
        return (
            f"{tuple(self.weight.shape)}, "
            f"eps={self.variance_epsilon}"
        )


def replace_huggingface_rmsnorm_modules(
    model: nn.Module,
) -> Sequence[str]:
    """
    Recursively replace Llama/Qwen2 RMSNorm modules in ``model``.

    Returns the fully-qualified module names that were replaced.
    """

    replaced_names: list[str] = []

    def _replace(parent: nn.Module, prefix: str) -> None:
        for child_name, child_module in list(parent.named_children()):
            qualified_name = (
                f"{prefix}.{child_name}" if prefix else child_name
            )

            if isinstance(child_module, _SUPPORTED_HF_RMSNORM_TYPES):
                replacement = (
                    HuggingFaceTritonRMSNorm.from_huggingface_module(
                        child_module
                    )
                )
                setattr(parent, child_name, replacement)
                replaced_names.append(qualified_name)
                continue

            _replace(child_module, qualified_name)

    _replace(model, prefix="")
    return tuple(replaced_names)


__all__ = [
    "HuggingFaceTritonRMSNorm",
    "replace_huggingface_rmsnorm_modules",
]
