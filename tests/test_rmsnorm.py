"""Correctness tests for PyTorch and Triton RMSNorm implementations."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from llm_kernels.torch_ops import TorchRMSNorm, rms_norm_reference
from llm_kernels.triton_ops import rms_norm_triton


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RMSNorm Triton tests require an NVIDIA CUDA GPU.",
)


TEST_SHAPES = [
    (1, 128),
    (4, 512),
    (2, 7, 1024),
    (128, 4096),
    (3, 17, 5120),
    (8, 8192),
]

TEST_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return relative and absolute tolerances for a dtype."""

    if dtype == torch.float16:
        return 2e-3, 2e-3

    if dtype == torch.bfloat16:
        return 1e-2, 1e-2

    return 1e-5, 1e-5


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_triton_matches_reference(
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(2026)

    device = torch.device("cuda")

    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn(shape[-1], device=device, dtype=dtype)

    expected = rms_norm_reference(x, weight, eps=1e-6)
    actual = rms_norm_triton(x, weight, eps=1e-6)

    torch.cuda.synchronize()

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_reference_matches_pytorch_functional(
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(2026)

    device = torch.device("cuda")

    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn(shape[-1], device=device, dtype=dtype)

    expected = F.rms_norm(
        x,
        normalized_shape=(shape[-1],),
        weight=weight,
        eps=1e-6,
    )
    actual = rms_norm_reference(x, weight, eps=1e-6)

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_output_metadata(dtype: torch.dtype) -> None:
    device = torch.device("cuda")

    x = torch.randn(
        2,
        3,
        4096,
        device=device,
        dtype=dtype,
    )
    weight = torch.ones(
        4096,
        device=device,
        dtype=dtype,
    )

    output = rms_norm_triton(x, weight)

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device
    assert output.is_contiguous()


def test_torch_module_forward() -> None:
    device = torch.device("cuda")

    module = TorchRMSNorm(
        hidden_size=4096,
        eps=1e-6,
        device=device,
        dtype=torch.float16,
    )

    x = torch.randn(
        2,
        8,
        4096,
        device=device,
        dtype=torch.float16,
    )

    expected = rms_norm_reference(x, module.weight, module.eps)
    actual = module(x)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-3,
        atol=2e-3,
    )


def test_invalid_weight_size() -> None:
    device = torch.device("cuda")

    x = torch.randn(
        4,
        4096,
        device=device,
        dtype=torch.float16,
    )
    weight = torch.ones(
        2048,
        device=device,
        dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="weight size"):
        rms_norm_triton(x, weight)


def test_dtype_mismatch() -> None:
    device = torch.device("cuda")

    x = torch.randn(
        4,
        4096,
        device=device,
        dtype=torch.float16,
    )
    weight = torch.ones(
        4096,
        device=device,
        dtype=torch.float32,
    )

    with pytest.raises(TypeError, match="same dtype"):
        rms_norm_triton(x, weight)


def test_non_contiguous_input_rejected() -> None:
    device = torch.device("cuda")

    x = torch.randn(
        4096,
        8,
        device=device,
        dtype=torch.float16,
    ).transpose(0, 1)

    assert not x.is_contiguous()

    weight = torch.ones(
        x.shape[-1],
        device=device,
        dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="contiguous x"):
        rms_norm_triton(x, weight)


def test_cpu_input_rejected() -> None:
    x = torch.randn(4, 128, dtype=torch.float32)
    weight = torch.ones(128, dtype=torch.float32)

    with pytest.raises(ValueError, match="CUDA tensors"):
        rms_norm_triton(x, weight)
