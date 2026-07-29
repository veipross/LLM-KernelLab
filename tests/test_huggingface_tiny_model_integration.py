"""整模型级 Hugging Face Llama/Qwen2 RMSNorm 集成测试。

这些测试使用随机初始化的微型模型，不下载任何预训练权重。
目标是验证：
1. 完整模型中的所有 Llama/Qwen2 RMSNorm 都能被自动替换；
2. 替换前后 state_dict 键名和参数数值保持不变；
3. CUDA 推理时确实调用 LLM-KernelLab 的 Triton RMSNorm；
4. 替换前后的完整模型 logits 在合理误差范围内一致。
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch
from torch import nn

import integrations.huggingface_rmsnorm as integration_module
from integrations import (
    HuggingFaceTritonRMSNorm,
    replace_huggingface_rmsnorm_modules,
)
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


_HF_RMSNORM_TYPES = (
    LlamaRMSNorm,
    Qwen2RMSNorm,
)


def _build_tiny_llama() -> nn.Module:
    """构造不依赖外部权重的微型 Llama CausalLM。"""

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return LlamaForCausalLM(config)


def _build_tiny_qwen2() -> nn.Module:
    """构造不依赖外部权重的微型 Qwen2 CausalLM。"""

    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return Qwen2ForCausalLM(config)


_MODEL_BUILDERS: tuple[tuple[str, Callable[[], nn.Module]], ...] = (
    ("llama", _build_tiny_llama),
    ("qwen2", _build_tiny_qwen2),
)


def _count_modules(
    model: nn.Module,
    module_types: type[nn.Module] | tuple[type[nn.Module], ...],
) -> int:
    """统计模型中指定类型的模块数量。"""

    return sum(
        1
        for module in model.modules()
        if isinstance(module, module_types)
    )


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """复制 state_dict，避免后续参数引用影响比较。"""

    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


@pytest.mark.parametrize(("model_name", "builder"), _MODEL_BUILDERS)
def test_tiny_model_replacement_preserves_state_dict(
    model_name: str,
    builder: Callable[[], nn.Module],
) -> None:
    """整模型替换必须保持 state_dict 键名和参数数值不变。"""

    del model_name
    torch.manual_seed(2026)

    model = builder().eval()
    state_dict_before = _clone_state_dict(model)

    original_rmsnorm_count = _count_modules(
        model,
        _HF_RMSNORM_TYPES,
    )
    assert original_rmsnorm_count > 0

    replaced_names = replace_huggingface_rmsnorm_modules(model)

    assert len(replaced_names) == original_rmsnorm_count
    assert len(set(replaced_names)) == original_rmsnorm_count
    assert _count_modules(model, _HF_RMSNORM_TYPES) == 0
    assert (
        _count_modules(model, HuggingFaceTritonRMSNorm)
        == original_rmsnorm_count
    )

    state_dict_after = model.state_dict()

    assert tuple(state_dict_after.keys()) == tuple(state_dict_before.keys())

    for name, expected_value in state_dict_before.items():
        torch.testing.assert_close(
            state_dict_after[name],
            expected_value,
            atol=0,
            rtol=0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for complete-model Triton integration tests.",
)
@pytest.mark.parametrize(("model_name", "builder"), _MODEL_BUILDERS)
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    (
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 5e-2, 5e-2),
        (torch.float32, 2e-4, 2e-4),
    ),
)
def test_tiny_model_cuda_logits_match_and_use_triton(
    model_name: str,
    builder: Callable[[], nn.Module],
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """替换后的整模型必须调用 Triton，且 logits 与基线保持一致。"""

    del model_name
    torch.manual_seed(2026)

    baseline_model = (
        builder()
        .cuda()
        .to(dtype=dtype)
        .eval()
    )
    triton_model = copy.deepcopy(baseline_model).eval()

    expected_state_dict = _clone_state_dict(triton_model)

    original_rmsnorm_count = _count_modules(
        triton_model,
        _HF_RMSNORM_TYPES,
    )
    assert original_rmsnorm_count > 0

    replaced_names = replace_huggingface_rmsnorm_modules(triton_model)

    assert len(replaced_names) == original_rmsnorm_count
    assert _count_modules(triton_model, _HF_RMSNORM_TYPES) == 0
    assert (
        _count_modules(triton_model, HuggingFaceTritonRMSNorm)
        == original_rmsnorm_count
    )

    # 确认替换动作没有改变模型参数。
    actual_state_dict = triton_model.state_dict()
    assert tuple(actual_state_dict.keys()) == tuple(
        expected_state_dict.keys()
    )
    for name, expected_value in expected_state_dict.items():
        torch.testing.assert_close(
            actual_state_dict[name],
            expected_value,
            atol=0,
            rtol=0,
        )

    triton_call_count = 0
    original_triton_function = integration_module.rms_norm_triton

    def _counted_rms_norm_triton(*args, **kwargs):
        nonlocal triton_call_count
        triton_call_count += 1
        return original_triton_function(*args, **kwargs)

    monkeypatch.setattr(
        integration_module,
        "rms_norm_triton",
        _counted_rms_norm_triton,
    )

    input_ids = torch.randint(
        low=0,
        high=128,
        size=(2, 8),
        device="cuda",
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        expected_output = baseline_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        actual_output = triton_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    # 两层微型模型中，每个 RMSNorm 在单次前向中应调用一次。
    assert triton_call_count == original_rmsnorm_count

    assert actual_output.logits.shape == expected_output.logits.shape
    assert actual_output.logits.dtype == expected_output.logits.dtype
    assert actual_output.logits.device == expected_output.logits.device

    torch.testing.assert_close(
        actual_output.logits,
        expected_output.logits,
        atol=atol,
        rtol=rtol,
    )
