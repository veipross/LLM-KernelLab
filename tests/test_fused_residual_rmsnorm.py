"""Tests for PyTorch and Triton Fused Residual + RMSNorm implementations."""

from __future__ import annotations

import pytest
import torch

from llm_kernels.torch_ops import (
    TorchFusedResidualRMSNorm,
    fused_residual_rms_norm_reference,
)
from llm_kernels.triton_ops import fused_residual_rms_norm_triton


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Fused Residual + RMSNorm Triton tests require CUDA.",
)

TEST_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

NUMERIC_CASES = [
    (1, 128),
    (8, 513),
    (128, 4096),
    (8, 5120),
    (1, 8192),
]


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return relative and absolute tolerances for a dtype."""

    if dtype == torch.float16:
        return 2e-3, 2e-3

    if dtype == torch.bfloat16:
        return 1e-2, 1e-2

    return 1e-5, 1e-5


def explicit_formula(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent explicit formula used as the semantic oracle."""

    residual_out = torch.add(x, residual).to(dtype=x.dtype)
    residual_fp32 = residual_out.to(dtype=torch.float32)
    weight_fp32 = weight.to(dtype=torch.float32)

    mean_square = torch.mean(
        residual_fp32 * residual_fp32,
        dim=-1,
        keepdim=True,
    )
    inverse_rms = torch.rsqrt(mean_square + eps)
    normalized = (
        residual_fp32 * inverse_rms * weight_fp32
    ).to(dtype=x.dtype)

    return normalized, residual_out


def assert_outputs_match(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype,
) -> None:
    """Compare both normalized and rounded-residual outputs."""

    actual_normalized, actual_residual = actual
    expected_normalized, expected_residual = expected

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(
        actual_normalized,
        expected_normalized,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        actual_residual,
        expected_residual,
        rtol=0.0,
        atol=0.0,
    )


def assert_no_input_alias(
    outputs: tuple[torch.Tensor, torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
) -> None:
    """Assert that outputs own storage distinct from every input."""

    output_pointers = {
        output.untyped_storage().data_ptr()
        for output in outputs
    }
    input_pointers = {
        input_tensor.untyped_storage().data_ptr()
        for input_tensor in inputs
    }

    assert len(output_pointers) == len(outputs)
    assert output_pointers.isdisjoint(input_pointers)


@pytest.mark.parametrize("rows,hidden_size", NUMERIC_CASES)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_reference_matches_explicit_formula_cpu(
    rows: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(2026 + rows + hidden_size)

    x = torch.randn(rows, hidden_size, dtype=dtype)
    residual = torch.randn(rows, hidden_size, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype)

    expected = explicit_formula(x, residual, weight, eps=1e-6)
    actual = fused_residual_rms_norm_reference(
        x,
        residual,
        weight,
        eps=1e-6,
    )

    assert_outputs_match(actual, expected, dtype)


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("rows,hidden_size", NUMERIC_CASES)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_triton_matches_reference(
    rows: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(2026 + rows + hidden_size)

    device = torch.device("cuda")
    x = torch.randn(rows, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(
        rows,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    expected = fused_residual_rms_norm_reference(
        x,
        residual,
        weight,
        eps=1e-6,
    )
    actual = fused_residual_rms_norm_triton(
        x,
        residual,
        weight,
        eps=1e-6,
    )

    torch.cuda.synchronize()
    assert_outputs_match(actual, expected, dtype)


def make_special_inputs(
    kind: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic special-value input pairs."""

    shape = (8, 513)
    element_count = shape[0] * shape[1]
    base = torch.linspace(
        -1.0,
        1.0,
        steps=element_count,
        device=device,
        dtype=dtype,
    ).reshape(shape)

    if kind == "equal":
        return base, base.clone()

    if kind == "cancel":
        return base, -base

    if kind == "zero":
        return torch.zeros_like(base), torch.zeros_like(base)

    if kind == "large":
        return (
            torch.full_like(base, 1000.0),
            torch.full_like(base, -250.0),
        )

    if kind == "small":
        return (
            torch.full_like(base, 1e-4),
            torch.full_like(base, -5e-5),
        )

    raise ValueError(f"Unknown special input kind: {kind}")


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize(
    "kind",
    ["equal", "cancel", "zero", "large", "small"],
)
def test_triton_special_inputs(kind: str) -> None:
    device = torch.device("cuda")
    dtype = torch.float16

    x, residual = make_special_inputs(
        kind,
        device=device,
        dtype=dtype,
    )
    weight = torch.linspace(
        0.5,
        1.5,
        steps=x.shape[-1],
        device=device,
        dtype=dtype,
    )

    expected = fused_residual_rms_norm_reference(
        x,
        residual,
        weight,
    )
    actual = fused_residual_rms_norm_triton(
        x,
        residual,
        weight,
    )

    torch.cuda.synchronize()
    assert_outputs_match(actual, expected, dtype)


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_multidimensional_output_metadata(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    shape = (2, 3, 513)

    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn(shape[-1], device=device, dtype=dtype)

    normalized, residual_out = fused_residual_rms_norm_triton(
        x,
        residual,
        weight,
    )

    for output in (normalized, residual_out):
        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert output.device == x.device
        assert output.is_contiguous()

    expected = fused_residual_rms_norm_reference(
        x,
        residual,
        weight,
    )
    assert_outputs_match((normalized, residual_out), expected, dtype)


def test_reference_does_not_modify_or_alias_inputs_cpu() -> None:
    torch.manual_seed(2026)

    x = torch.randn(2, 3, 513)
    residual = torch.randn(2, 3, 513)
    weight = torch.randn(513)

    original_inputs = tuple(
        tensor.clone()
        for tensor in (x, residual, weight)
    )

    outputs = fused_residual_rms_norm_reference(
        x,
        residual,
        weight,
    )

    for actual, original in zip(
        (x, residual, weight),
        original_inputs,
    ):
        torch.testing.assert_close(
            actual,
            original,
            rtol=0.0,
            atol=0.0,
        )

    assert_no_input_alias(outputs, (x, residual, weight))


@pytest.mark.cuda
@requires_cuda
def test_triton_does_not_modify_or_alias_inputs() -> None:
    torch.manual_seed(2026)

    device = torch.device("cuda")
    x = torch.randn(2, 3, 513, device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(513, device=device, dtype=torch.float16)

    original_inputs = tuple(
        tensor.clone()
        for tensor in (x, residual, weight)
    )

    outputs = fused_residual_rms_norm_triton(
        x,
        residual,
        weight,
    )
    torch.cuda.synchronize()

    for actual, original in zip(
        (x, residual, weight),
        original_inputs,
    ):
        torch.testing.assert_close(
            actual,
            original,
            rtol=0.0,
            atol=0.0,
        )

    assert_no_input_alias(outputs, (x, residual, weight))


def test_torch_module_forward_cpu() -> None:
    module = TorchFusedResidualRMSNorm(
        hidden_size=513,
        eps=1e-6,
        dtype=torch.float32,
    )

    assert not module.weight.requires_grad

    x = torch.randn(2, 3, 513)
    residual = torch.randn_like(x)

    expected = fused_residual_rms_norm_reference(
        x,
        residual,
        module.weight,
        module.eps,
    )
    actual = module(x, residual)

    assert_outputs_match(actual, expected, torch.float32)


def test_shape_mismatch_rejected() -> None:
    x = torch.randn(4, 128)
    residual = torch.randn(2, 128)
    weight = torch.ones(128)

    with pytest.raises(ValueError, match="same shape"):
        fused_residual_rms_norm_reference(x, residual, weight)


def test_dtype_mismatch_rejected() -> None:
    x = torch.randn(4, 128, dtype=torch.float32)
    residual = torch.randn(4, 128, dtype=torch.float16)
    weight = torch.ones(128, dtype=torch.float32)

    with pytest.raises(TypeError, match="same dtype"):
        fused_residual_rms_norm_reference(x, residual, weight)


def test_device_mismatch_rejected() -> None:
    x = torch.randn(4, 128)
    residual = torch.empty(4, 128, device="meta")
    weight = torch.ones(128)

    with pytest.raises(ValueError, match="same device"):
        fused_residual_rms_norm_reference(x, residual, weight)


def test_weight_length_rejected() -> None:
    x = torch.randn(4, 128)
    residual = torch.randn_like(x)
    weight = torch.ones(64)

    with pytest.raises(ValueError, match="weight size"):
        fused_residual_rms_norm_reference(x, residual, weight)


@pytest.mark.parametrize("eps", [0.0, -1e-6, float("nan")])
def test_invalid_eps_rejected(eps: float) -> None:
    x = torch.randn(4, 128)
    residual = torch.randn_like(x)
    weight = torch.ones(128)

    with pytest.raises(ValueError, match="finite and positive"):
        fused_residual_rms_norm_reference(
            x,
            residual,
            weight,
            eps=eps,
        )


@pytest.mark.parametrize("input_name", ["x", "residual", "weight"])
def test_requires_grad_rejected(input_name: str) -> None:
    x = torch.randn(4, 128)
    residual = torch.randn_like(x)
    weight = torch.ones(128)

    tensors = {
        "x": x,
        "residual": residual,
        "weight": weight,
    }
    tensors[input_name].requires_grad_(True)

    with pytest.raises(
        RuntimeError,
        match=rf"{input_name}\.requires_grad",
    ):
        fused_residual_rms_norm_reference(
            tensors["x"],
            tensors["residual"],
            tensors["weight"],
        )


def test_empty_hidden_dimension_rejected() -> None:
    x = torch.empty(4, 0)
    residual = torch.empty_like(x)
    weight = torch.empty(0)

    with pytest.raises(ValueError, match="non-empty"):
        fused_residual_rms_norm_reference(x, residual, weight)


def test_cpu_input_rejected_by_triton() -> None:
    x = torch.randn(4, 128)
    residual = torch.randn_like(x)
    weight = torch.ones(128)

    with pytest.raises(ValueError, match="CUDA tensors"):
        fused_residual_rms_norm_triton(x, residual, weight)


def make_non_contiguous(
    target: str,
    *,
    rows: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create one non-contiguous CUDA input selected by name."""

    device = torch.device("cuda")
    dtype = torch.float16

    x = torch.randn(rows, hidden_size, device=device, dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.ones(hidden_size, device=device, dtype=dtype)

    if target == "x":
        x = torch.randn(
            hidden_size,
            rows,
            device=device,
            dtype=dtype,
        ).transpose(0, 1)
    elif target == "residual":
        residual = torch.randn(
            hidden_size,
            rows,
            device=device,
            dtype=dtype,
        ).transpose(0, 1)
    elif target == "weight":
        weight = torch.ones(
            hidden_size * 2,
            device=device,
            dtype=dtype,
        )[::2]
    else:
        raise ValueError(f"Unknown target: {target}")

    return x, residual, weight


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("target", ["x", "residual", "weight"])
def test_non_contiguous_input_rejected(target: str) -> None:
    x, residual, weight = make_non_contiguous(
        target,
        rows=4,
        hidden_size=513,
    )

    with pytest.raises(ValueError, match=rf"contiguous {target}"):
        fused_residual_rms_norm_triton(x, residual, weight)


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize(
    "hidden_size,dtype",
    [
        (16383, torch.float16),
        (16384, torch.float16),
        (16384, torch.bfloat16),
        (16384, torch.float32),
    ],
)
def test_supported_hidden_size_boundary(
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(2026 + hidden_size)

    device = torch.device("cuda")
    x = torch.randn(1, hidden_size, device=device, dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    expected = fused_residual_rms_norm_reference(
        x,
        residual,
        weight,
    )
    actual = fused_residual_rms_norm_triton(
        x,
        residual,
        weight,
    )

    torch.cuda.synchronize()
    assert_outputs_match(actual, expected, dtype)


@pytest.mark.cuda
@requires_cuda
def test_hidden_size_above_limit_rejected() -> None:
    hidden_size = 16385
    device = torch.device("cuda")

    x = torch.zeros(
        1,
        hidden_size,
        device=device,
        dtype=torch.float16,
    )
    residual = torch.zeros_like(x)
    weight = torch.ones(
        hidden_size,
        device=device,
        dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="supported limit of 16384"):
        fused_residual_rms_norm_triton(x, residual, weight)
