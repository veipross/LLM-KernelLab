"""Tests for Hugging Face RMSNorm integration."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from integrations.huggingface_rmsnorm import (
    HuggingFaceTritonRMSNorm,
    replace_huggingface_rmsnorm_modules,
)


HF_RMSNORM_CLASSES = (
    LlamaRMSNorm,
    Qwen2RMSNorm,
)


@pytest.mark.parametrize("rmsnorm_class", HF_RMSNORM_CLASSES)
def test_from_huggingface_module_preserves_parameter_and_eps(
    rmsnorm_class: type[nn.Module],
) -> None:
    """The adapter must preserve the original parameter object and epsilon."""

    hidden_size = 128
    eps = 1e-5

    original = rmsnorm_class(hidden_size, eps=eps)
    original.train()

    adapter = HuggingFaceTritonRMSNorm.from_huggingface_module(original)

    assert adapter.weight is original.weight
    assert adapter.hidden_size == hidden_size
    assert adapter.variance_epsilon == pytest.approx(eps)
    assert adapter.training is original.training


@pytest.mark.parametrize("rmsnorm_class", HF_RMSNORM_CLASSES)
@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ),
)
def test_cpu_fallback_matches_huggingface(
    rmsnorm_class: type[nn.Module],
    dtype: torch.dtype,
) -> None:
    """CPU execution must safely use the PyTorch fallback path."""

    torch.manual_seed(2026)

    hidden_size = 256
    eps = 1e-6

    original = rmsnorm_class(hidden_size, eps=eps).to(dtype=dtype).eval()

    with torch.no_grad():
        original.weight.uniform_(0.5, 1.5)

    adapter = (
        HuggingFaceTritonRMSNorm
        .from_huggingface_module(original)
        .eval()
    )

    hidden_states = torch.randn(
        2,
        4,
        hidden_size,
        dtype=dtype,
    )

    with torch.inference_mode():
        expected = original(hidden_states)
        actual = adapter(hidden_states)

    assert adapter._can_use_triton(hidden_states) is False
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton RMSNorm integration tests.",
)
@pytest.mark.parametrize("rmsnorm_class", HF_RMSNORM_CLASSES)
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    (
        (torch.float16, 3e-3, 3e-3),
        (torch.bfloat16, 2e-2, 2e-2),
        (torch.float32, 1e-4, 1e-4),
    ),
)
def test_cuda_fast_path_matches_huggingface(
    rmsnorm_class: type[nn.Module],
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    """CUDA inference must enter the Triton path and match Hugging Face."""

    torch.manual_seed(2026)

    hidden_size = 4096
    eps = 1e-6

    original = (
        rmsnorm_class(hidden_size, eps=eps)
        .cuda()
        .to(dtype=dtype)
        .eval()
    )

    with torch.no_grad():
        original.weight.uniform_(0.5, 1.5)

    adapter = (
        HuggingFaceTritonRMSNorm
        .from_huggingface_module(original)
        .eval()
    )

    hidden_states = torch.randn(
        2,
        8,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )

    with torch.inference_mode():
        assert adapter._can_use_triton(hidden_states) is True

        expected = original(hidden_states)
        actual = adapter(hidden_states)

    assert actual.shape == hidden_states.shape
    assert actual.dtype == hidden_states.dtype
    assert actual.device == hidden_states.device
    assert adapter.weight is original.weight

    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
    )


class _NestedRMSNormModel(nn.Module):
    """Small nested module used to test recursive replacement."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        self.input_norm = LlamaRMSNorm(hidden_size)
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            Qwen2RMSNorm(hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_norm(hidden_states)
        return self.block(hidden_states)


def test_recursive_replacement_preserves_state_dict_keys() -> None:
    """Recursive replacement must keep parameter names and loaded values."""

    torch.manual_seed(2026)

    hidden_size = 64
    model = _NestedRMSNormModel(hidden_size).eval()

    state_dict_before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

    replaced_names = replace_huggingface_rmsnorm_modules(model)

    assert replaced_names == (
        "input_norm",
        "block.1",
    )
    assert isinstance(model.input_norm, HuggingFaceTritonRMSNorm)
    assert isinstance(model.block[1], HuggingFaceTritonRMSNorm)

    state_dict_after = model.state_dict()

    assert state_dict_after.keys() == state_dict_before.keys()

    for name, expected_value in state_dict_before.items():
        torch.testing.assert_close(
            state_dict_after[name],
            expected_value,
            atol=0,
            rtol=0,
        )


def test_recursive_replacement_keeps_cpu_forward_output() -> None:
    """Replacing nested modules must not change CPU inference output."""

    torch.manual_seed(2026)

    hidden_size = 64
    model = _NestedRMSNormModel(hidden_size).eval()
    hidden_states = torch.randn(2, 3, hidden_size)

    with torch.inference_mode():
        expected = model(hidden_states)

    replaced_names = replace_huggingface_rmsnorm_modules(model)

    with torch.inference_mode():
        actual = model(hidden_states)

    assert len(replaced_names) == 2
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
