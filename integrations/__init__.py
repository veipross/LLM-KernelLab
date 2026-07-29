"""Integration utilities for external LLM frameworks."""

from integrations.huggingface_rmsnorm import (
    HuggingFaceTritonRMSNorm,
    replace_huggingface_rmsnorm_modules,
)

__all__ = [
    "HuggingFaceTritonRMSNorm",
    "replace_huggingface_rmsnorm_modules",
]
