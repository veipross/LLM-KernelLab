"""Benchmark Hugging Face RMSNorm integration for Llama and Qwen2.

This script performs two benchmark levels without downloading model weights:

1. Module-level benchmark:
   Hugging Face Llama/Qwen2 RMSNorm vs LLM-KernelLab Triton adapter.

2. Complete-model benchmark:
   Randomly initialized compact Llama/Qwen2 CausalLM models before and after
   recursive RMSNorm replacement.

The complete-model "single_token_no_cache" case is a decode-like single-token
forward without KV cache. It must not be reported as true autoregressive decode
latency. The prefill cases run full-sequence forward with ``use_cache=False``.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch import nn
import transformers
import triton

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


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

FAMILY_ORDER = [
    "llama",
    "qwen2",
]

MODULE_PROVIDER_ORDER = [
    "huggingface",
    "triton_adapter",
]

MODEL_PROVIDER_ORDER = [
    "huggingface_model",
    "triton_rmsnorm_model",
]

HF_RMSNORM_CLASSES: dict[str, type[nn.Module]] = {
    "llama": LlamaRMSNorm,
    "qwen2": Qwen2RMSNorm,
}

HF_RMSNORM_TYPES = (
    LlamaRMSNorm,
    Qwen2RMSNorm,
)

QUICK_MODULE_CASES = [
    ("single_token_b1", 1, 1, 4096),
    ("single_token_b8", 8, 1, 4096),
    ("prefill_b1_s128", 1, 128, 4096),
]

FULL_MODULE_CASES = [
    *QUICK_MODULE_CASES,
    ("prefill_b1_s512", 1, 512, 4096),
    ("prefill_b4_s128", 4, 128, 4096),
    ("prefill_b1_s128_h8192", 1, 128, 8192),
]

QUICK_MODEL_CASES = [
    ("single_token_no_cache", 1, 1),
    ("prefill_s32", 1, 32),
]

FULL_MODEL_CASES = [
    *QUICK_MODEL_CASES,
    ("prefill_s128", 1, 128),
    ("prefill_b4_s32", 4, 32),
]


def synchronize() -> None:
    torch.cuda.synchronize()


def run_command(command: list[str]) -> str:
    try:
        output = subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()

        return "\n".join(
            line.rstrip() for line in output.splitlines()
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"Unavailable: {exc}"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def save_environment_report(
    path: Path,
    args: argparse.Namespace,
) -> None:
    """Save benchmark environment and command-line configuration."""

    path.parent.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(0)

    lines = [
        "LLM-KernelLab Hugging Face RMSNorm Integration Benchmark",
        "=" * 76,
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"git_commit: {get_git_commit()}",
        f"platform: {platform.platform()}",
        f"python: {sys.version}",
        f"pytorch: {torch.__version__}",
        f"pytorch_cuda: {torch.version.cuda}",
        f"triton: {triton.__version__}",
        f"transformers: {transformers.__version__}",
        f"gpu: {properties.name}",
        (
            "compute_capability: "
            f"{properties.major}.{properties.minor}"
        ),
        (
            "gpu_memory_gib: "
            f"{properties.total_memory / 1024**3:.2f}"
        ),
        f"multiprocessors: {properties.multi_processor_count}",
        f"tf32_matmul_allowed: {torch.backends.cuda.matmul.allow_tf32}",
        f"mode: {'quick' if args.quick else 'full'}",
        f"run_index: {args.run_index}",
        f"families: {','.join(args.families)}",
        f"dtypes: {','.join(args.dtypes)}",
        f"scopes: {','.join(args.scopes)}",
        f"warmup_ms: {args.warmup_ms}",
        f"repetition_ms: {args.rep_ms}",
        f"model_hidden_size: {args.model_hidden_size}",
        f"model_intermediate_size: {args.model_intermediate_size}",
        f"model_layers: {args.model_layers}",
        f"model_attention_heads: {args.model_attention_heads}",
        f"model_key_value_heads: {args.model_key_value_heads}",
        f"model_vocab_size: {args.model_vocab_size}",
        "",
        "===== nvidia-smi =====",
        run_command(["nvidia-smi"]),
        "",
        "===== nvcc =====",
        run_command(["nvcc", "--version"]),
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def dtype_tolerances(
    dtype: torch.dtype,
    *,
    complete_model: bool,
) -> tuple[float, float]:
    """Return ``(rtol, atol)`` for module or complete-model validation."""

    if complete_model:
        if dtype == torch.float16:
            return 1e-2, 1e-2

        if dtype == torch.bfloat16:
            return 5e-2, 5e-2

        return 2e-4, 2e-4

    if dtype == torch.float16:
        return 3e-3, 3e-3

    if dtype == torch.bfloat16:
        return 2e-2, 2e-2

    return 1e-4, 1e-4


def output_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    difference = (
        actual.float() - expected.float()
    ).abs()
    return difference.max().item(), difference.mean().item()


def provider_order_for_case(
    providers: Sequence[str],
    run_index: int,
    case_index: int,
) -> list[str]:
    """Rotate provider order across cases and repeated runs."""

    provider_list = list(providers)
    offset = (run_index - 1 + case_index) % len(provider_list)
    return provider_list[offset:] + provider_list[:offset]


def measure_first_call(
    function: Callable[[], torch.Tensor],
) -> tuple[torch.Tensor, float]:
    """Measure synchronized wall-clock latency for the first inference call."""

    synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        output = function()

    synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return output, elapsed_ms


def benchmark_function(
    function: Callable[[], torch.Tensor],
    warmup_ms: int,
    repetition_ms: int,
) -> tuple[float, float]:
    """Measure steady-state P50 and P95 GPU latency."""

    with torch.inference_mode():
        timings = triton.testing.do_bench(
            function,
            warmup=warmup_ms,
            rep=repetition_ms,
            quantiles=[0.50, 0.95],
        )

    p50_ms, p95_ms = (float(value) for value in timings)
    return p50_ms, p95_ms


def count_triton_calls(
    function: Callable[[], torch.Tensor],
) -> tuple[torch.Tensor, int]:
    """Run one validation call while counting adapter Triton invocations."""

    call_count = 0
    original_function = integration_module.rms_norm_triton

    def counted_function(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return original_function(*args, **kwargs)

    integration_module.rms_norm_triton = counted_function

    try:
        with torch.inference_mode():
            output = function()
        synchronize()
    finally:
        integration_module.rms_norm_triton = original_function

    return output, call_count


def count_modules(
    model: nn.Module,
    module_types: type[nn.Module] | tuple[type[nn.Module], ...],
) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(module, module_types)
    )


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_rmsnorm_module(
    family: str,
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
) -> tuple[nn.Module, HuggingFaceTritonRMSNorm]:
    """Create matching Hugging Face and Triton-adapter RMSNorm modules."""

    module_class = HF_RMSNORM_CLASSES[family]
    original = (
        module_class(hidden_size, eps=eps)
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

    return original, adapter


def benchmark_module_case(
    *,
    family: str,
    case_name: str,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    dtype_name: str,
    run_index: int,
    case_index: int,
    warmup_ms: int,
    repetition_ms: int,
    eps: float,
) -> list[dict[str, object]]:
    """Validate and benchmark one Hugging Face RMSNorm module case."""

    dtype = DTYPE_MAP[dtype_name]
    seed = 2026 + batch_size + sequence_length + hidden_size
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    original, adapter = make_rmsnorm_module(
        family=family,
        hidden_size=hidden_size,
        dtype=dtype,
        eps=eps,
    )

    hidden_states = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )

    functions: dict[str, Callable[[], torch.Tensor]] = {
        "huggingface": lambda: original(hidden_states),
        "triton_adapter": lambda: adapter(hidden_states),
    }

    with torch.inference_mode():
        reference = functions["huggingface"]()
    synchronize()

    triton_validation, triton_call_count = count_triton_calls(
        functions["triton_adapter"]
    )

    rtol, atol = dtype_tolerances(
        dtype,
        complete_model=False,
    )
    torch.testing.assert_close(
        triton_validation,
        reference,
        rtol=rtol,
        atol=atol,
    )

    if triton_call_count != 1:
        raise RuntimeError(
            "The module-level adapter did not execute exactly one Triton "
            f"RMSNorm call; observed {triton_call_count}."
        )

    ordered_providers = provider_order_for_case(
        MODULE_PROVIDER_ORDER,
        run_index,
        case_index,
    )
    order_text = "|".join(ordered_providers)

    print()
    print(
        "Module case: "
        f"family={family}, case={case_name}, "
        f"batch={batch_size}, seq={sequence_length}, "
        f"hidden={hidden_size}, dtype={dtype_name}"
    )
    print(f"  Provider order: {order_text}")

    records: list[dict[str, object]] = []

    for provider_position, provider in enumerate(
        ordered_providers,
        start=1,
    ):
        print(f"  Benchmarking {provider:<22}", end="", flush=True)

        record: dict[str, object] = {
            "scope": "module",
            "run_index": run_index,
            "case_index": case_index,
            "provider_position": provider_position,
            "case_provider_order": order_text,
            "family": family,
            "case_name": case_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "rows": batch_size * sequence_length,
            "hidden_size": hidden_size,
            "numel": hidden_states.numel(),
            "dtype": dtype_name,
            "provider": provider,
            "expected_triton_calls": 1 if provider == "triton_adapter" else 0,
            "validated_triton_calls": (
                triton_call_count if provider == "triton_adapter" else 0
            ),
            "status": "ok",
            "error": "",
        }

        try:
            function = functions[provider]
            output, first_call_ms = measure_first_call(function)

            torch.testing.assert_close(
                output,
                reference,
                rtol=rtol,
                atol=atol,
            )

            max_error, mean_error = output_errors(output, reference)
            p50_ms, p95_ms = benchmark_function(
                function,
                warmup_ms,
                repetition_ms,
            )

            record.update(
                {
                    "first_call_ms": first_call_ms,
                    "p50_ms": p50_ms,
                    "p95_ms": p95_ms,
                    "max_abs_error": max_error,
                    "mean_abs_error": mean_error,
                }
            )

            print(
                f" PASSED | P50={p50_ms:.6f} ms "
                f"| P95={p95_ms:.6f} ms"
            )

        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "error": repr(exc),
                    "first_call_ms": float("nan"),
                    "p50_ms": float("nan"),
                    "p95_ms": float("nan"),
                    "max_abs_error": float("nan"),
                    "mean_abs_error": float("nan"),
                }
            )
            print(f" FAILED | {exc!r}")

        records.append(record)

    del triton_validation
    del reference
    del hidden_states
    del adapter
    del original

    torch.cuda.empty_cache()
    return records


def build_model_config(
    family: str,
    args: argparse.Namespace,
    max_position_embeddings: int,
) -> LlamaConfig | Qwen2Config:
    common_arguments = {
        "vocab_size": args.model_vocab_size,
        "hidden_size": args.model_hidden_size,
        "intermediate_size": args.model_intermediate_size,
        "num_hidden_layers": args.model_layers,
        "num_attention_heads": args.model_attention_heads,
        "num_key_value_heads": args.model_key_value_heads,
        "max_position_embeddings": max_position_embeddings,
        "rms_norm_eps": args.eps,
        "attention_dropout": 0.0,
        "use_cache": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
    }

    if family == "llama":
        return LlamaConfig(**common_arguments)

    if family == "qwen2":
        return Qwen2Config(**common_arguments)

    raise ValueError(f"Unknown model family: {family}")


def build_model(
    family: str,
    config: LlamaConfig | Qwen2Config,
) -> nn.Module:
    if family == "llama":
        return LlamaForCausalLM(config)

    if family == "qwen2":
        return Qwen2ForCausalLM(config)

    raise ValueError(f"Unknown model family: {family}")


def make_complete_models(
    family: str,
    dtype: torch.dtype,
    args: argparse.Namespace,
    max_position_embeddings: int,
) -> tuple[nn.Module, nn.Module, tuple[str, ...], int]:
    """Create equal-weight baseline and Triton-replaced compact models."""

    config = build_model_config(
        family=family,
        args=args,
        max_position_embeddings=max_position_embeddings,
    )

    seed = 2026 + args.model_hidden_size + args.model_layers
    torch.manual_seed(seed)

    baseline_model = build_model(family, config).eval()
    triton_model = build_model(family, config).eval()
    triton_model.load_state_dict(baseline_model.state_dict())

    original_rmsnorm_count = count_modules(
        triton_model,
        HF_RMSNORM_TYPES,
    )
    if original_rmsnorm_count <= 0:
        raise RuntimeError(
            f"No Hugging Face RMSNorm modules found in {family} model."
        )

    replaced_names = tuple(
        replace_huggingface_rmsnorm_modules(triton_model)
    )

    if len(replaced_names) != original_rmsnorm_count:
        raise RuntimeError(
            "RMSNorm replacement count mismatch: "
            f"expected={original_rmsnorm_count}, "
            f"actual={len(replaced_names)}."
        )

    if count_modules(triton_model, HF_RMSNORM_TYPES) != 0:
        raise RuntimeError(
            "Original Hugging Face RMSNorm modules remain after replacement."
        )

    baseline_model = (
        baseline_model
        .cuda()
        .to(dtype=dtype)
        .eval()
    )
    triton_model = (
        triton_model
        .cuda()
        .to(dtype=dtype)
        .eval()
    )

    return (
        baseline_model,
        triton_model,
        replaced_names,
        original_rmsnorm_count,
    )


def benchmark_complete_model_case(
    *,
    family: str,
    case_name: str,
    batch_size: int,
    sequence_length: int,
    dtype_name: str,
    baseline_model: nn.Module,
    triton_model: nn.Module,
    replaced_names: tuple[str, ...],
    original_rmsnorm_count: int,
    run_index: int,
    case_index: int,
    warmup_ms: int,
    repetition_ms: int,
    vocab_size: int,
) -> list[dict[str, object]]:
    """Validate and benchmark one compact complete-model case."""

    dtype = DTYPE_MAP[dtype_name]
    seed = 3026 + batch_size + sequence_length
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
        device="cuda",
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)

    def baseline_function() -> torch.Tensor:
        return baseline_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=0,
        ).logits

    def triton_function() -> torch.Tensor:
        return triton_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=0,
        ).logits

    functions: dict[str, Callable[[], torch.Tensor]] = {
        "huggingface_model": baseline_function,
        "triton_rmsnorm_model": triton_function,
    }

    with torch.inference_mode():
        reference = baseline_function()
    synchronize()

    triton_validation, triton_call_count = count_triton_calls(
        triton_function
    )

    rtol, atol = dtype_tolerances(
        dtype,
        complete_model=True,
    )
    torch.testing.assert_close(
        triton_validation,
        reference,
        rtol=rtol,
        atol=atol,
    )

    if triton_call_count != original_rmsnorm_count:
        raise RuntimeError(
            "Complete-model Triton call count mismatch: "
            f"expected={original_rmsnorm_count}, "
            f"actual={triton_call_count}."
        )

    ordered_providers = provider_order_for_case(
        MODEL_PROVIDER_ORDER,
        run_index,
        case_index,
    )
    order_text = "|".join(ordered_providers)
    parameter_count = model_parameter_count(baseline_model)

    print()
    print(
        "Model case: "
        f"family={family}, case={case_name}, "
        f"batch={batch_size}, seq={sequence_length}, "
        f"dtype={dtype_name}"
    )
    print(
        f"  Parameters: {parameter_count:,} | "
        f"Replaced RMSNorm modules: {original_rmsnorm_count}"
    )
    print(f"  Provider order: {order_text}")

    records: list[dict[str, object]] = []

    for provider_position, provider in enumerate(
        ordered_providers,
        start=1,
    ):
        print(f"  Benchmarking {provider:<22}", end="", flush=True)

        record: dict[str, object] = {
            "scope": "complete_model",
            "run_index": run_index,
            "case_index": case_index,
            "provider_position": provider_position,
            "case_provider_order": order_text,
            "family": family,
            "case_name": case_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "tokens": batch_size * sequence_length,
            "hidden_size": baseline_model.config.hidden_size,
            "intermediate_size": baseline_model.config.intermediate_size,
            "num_hidden_layers": baseline_model.config.num_hidden_layers,
            "num_attention_heads": baseline_model.config.num_attention_heads,
            "num_key_value_heads": baseline_model.config.num_key_value_heads,
            "vocab_size": vocab_size,
            "parameter_count": parameter_count,
            "dtype": dtype_name,
            "provider": provider,
            "replaced_rmsnorm_count": len(replaced_names),
            "expected_triton_calls": (
                original_rmsnorm_count
                if provider == "triton_rmsnorm_model"
                else 0
            ),
            "validated_triton_calls": (
                triton_call_count
                if provider == "triton_rmsnorm_model"
                else 0
            ),
            "benchmark_semantics": (
                "single-token forward without KV cache"
                if case_name == "single_token_no_cache"
                else "prefill-style full-sequence forward without KV cache"
            ),
            "status": "ok",
            "error": "",
        }

        try:
            function = functions[provider]
            output, first_call_ms = measure_first_call(function)

            torch.testing.assert_close(
                output,
                reference,
                rtol=rtol,
                atol=atol,
            )

            max_error, mean_error = output_errors(output, reference)
            p50_ms, p95_ms = benchmark_function(
                function,
                warmup_ms,
                repetition_ms,
            )

            tokens_per_second = (
                batch_size * sequence_length
            ) / (p50_ms / 1000.0)

            record.update(
                {
                    "first_call_ms": first_call_ms,
                    "p50_ms": p50_ms,
                    "p95_ms": p95_ms,
                    "tokens_per_second": tokens_per_second,
                    "max_abs_error": max_error,
                    "mean_abs_error": mean_error,
                }
            )

            print(
                f" PASSED | P50={p50_ms:.6f} ms "
                f"| P95={p95_ms:.6f} ms "
                f"| tokens/s={tokens_per_second:.2f}"
            )

        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "error": repr(exc),
                    "first_call_ms": float("nan"),
                    "p50_ms": float("nan"),
                    "p95_ms": float("nan"),
                    "tokens_per_second": float("nan"),
                    "max_abs_error": float("nan"),
                    "mean_abs_error": float("nan"),
                }
            )
            print(f" FAILED | {exc!r}")

        records.append(record)

    del triton_validation
    del reference
    del attention_mask
    del input_ids

    torch.cuda.empty_cache()
    return records


def add_speedup_column(
    dataframe: pd.DataFrame,
    *,
    group_columns: list[str],
    baseline_provider: str,
    output_column: str,
) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe[output_column] = float("nan")

    if dataframe.empty:
        return dataframe

    for _, group in dataframe.groupby(group_columns):
        valid = group[group["status"] == "ok"]
        baseline_rows = valid[
            valid["provider"] == baseline_provider
        ]

        if baseline_rows.empty:
            continue

        baseline_latency = float(
            baseline_rows.iloc[0]["p50_ms"]
        )

        for index in group.index:
            latency = dataframe.at[index, "p50_ms"]

            if pd.isna(latency) or float(latency) <= 0:
                continue

            dataframe.at[index, output_column] = (
                baseline_latency / float(latency)
            )

    return dataframe


def insert_metadata(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    metadata = [
        ("timestamp", datetime.now().isoformat(timespec="seconds")),
        ("git_commit", get_git_commit()),
        ("gpu", torch.cuda.get_device_name(0)),
        ("torch_version", torch.__version__),
        ("torch_cuda_version", str(torch.version.cuda)),
        ("triton_version", triton.__version__),
        ("transformers_version", transformers.__version__),
    ]

    for position, (column, value) in enumerate(metadata):
        dataframe.insert(position, column, value)

    return dataframe


def print_module_summary(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        return

    columns = [
        "family",
        "case_name",
        "batch_size",
        "sequence_length",
        "hidden_size",
        "dtype",
        "provider",
        "p50_ms",
        "p95_ms",
        "speedup_vs_huggingface",
        "max_abs_error",
        "status",
    ]

    print()
    print("=" * 150)
    print("Hugging Face RMSNorm Module Benchmark Summary")
    print("=" * 150)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        print(dataframe[columns].to_string(index=False))

    print("=" * 150)


def print_model_summary(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        return

    columns = [
        "family",
        "case_name",
        "batch_size",
        "sequence_length",
        "dtype",
        "provider",
        "p50_ms",
        "p95_ms",
        "tokens_per_second",
        "speedup_vs_huggingface_model",
        "replaced_rmsnorm_count",
        "max_abs_error",
        "status",
    ]

    print()
    print("=" * 170)
    print("Compact Llama/Qwen2 Complete-Model Benchmark Summary")
    print("=" * 170)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        260,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        print(dataframe[columns].to_string(index=False))

    print("=" * 170)


def validate_arguments(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if args.run_index <= 0:
        raise ValueError("--run-index must be positive.")

    if args.warmup_ms <= 0:
        raise ValueError("--warmup-ms must be positive.")

    if args.rep_ms <= 0:
        raise ValueError("--rep-ms must be positive.")

    if args.eps <= 0:
        raise ValueError("--eps must be positive.")

    positive_model_arguments = {
        "--model-hidden-size": args.model_hidden_size,
        "--model-intermediate-size": args.model_intermediate_size,
        "--model-layers": args.model_layers,
        "--model-attention-heads": args.model_attention_heads,
        "--model-key-value-heads": args.model_key_value_heads,
        "--model-vocab-size": args.model_vocab_size,
    }

    for name, value in positive_model_arguments.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    if (
        args.model_hidden_size
        % args.model_attention_heads
        != 0
    ):
        raise ValueError(
            "--model-hidden-size must be divisible by "
            "--model-attention-heads."
        )

    if (
        args.model_attention_heads
        % args.model_key_value_heads
        != 0
    ):
        raise ValueError(
            "--model-attention-heads must be divisible by "
            "--model-key-value-heads."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Hugging Face Llama/Qwen2 RMSNorm modules and compact "
            "complete models before and after Triton RMSNorm replacement."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a smaller FP16 development benchmark with fewer cases."
        ),
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=["module", "model"],
        default=["module", "model"],
        help="Benchmark scopes to execute.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=FAMILY_ORDER,
        default=FAMILY_ORDER,
        help="Hugging Face model families to benchmark.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=list(DTYPE_MAP),
        default=list(DTYPE_MAP),
        help="Data types used by the full benchmark.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=1,
        help="Positive repeated-run index used to rotate provider order.",
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=100,
        help="Warmup duration supplied to triton.testing.do_bench.",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=300,
        help="Measurement duration supplied to triton.testing.do_bench.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="RMSNorm epsilon.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Root directory for CSV and environment outputs.",
    )
    parser.add_argument(
        "--model-hidden-size",
        type=int,
        default=512,
        help="Hidden size of the randomly initialized compact models.",
    )
    parser.add_argument(
        "--model-intermediate-size",
        type=int,
        default=1376,
        help="MLP intermediate size of the compact models.",
    )
    parser.add_argument(
        "--model-layers",
        type=int,
        default=4,
        help="Number of Transformer layers in each compact model.",
    )
    parser.add_argument(
        "--model-attention-heads",
        type=int,
        default=8,
        help="Number of attention heads in each compact model.",
    )
    parser.add_argument(
        "--model-key-value-heads",
        type=int,
        default=4,
        help="Number of key/value heads in each compact model.",
    )
    parser.add_argument(
        "--model-vocab-size",
        type=int,
        default=2048,
        help="Vocabulary size of the compact models.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    if args.quick:
        dtype_names = ["fp16"]
        module_cases = QUICK_MODULE_CASES
        model_cases = QUICK_MODEL_CASES
    else:
        dtype_names = args.dtypes
        module_cases = FULL_MODULE_CASES
        model_cases = FULL_MODEL_CASES

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_directory = args.output_dir / "csv"
    environment_directory = args.output_dir / "environment"

    module_output_path = (
        csv_directory
        / f"huggingface_rmsnorm_module_benchmark_{timestamp}.csv"
    )
    model_output_path = (
        csv_directory
        / f"huggingface_rmsnorm_model_benchmark_{timestamp}.csv"
    )
    environment_path = (
        environment_directory
        / f"huggingface_rmsnorm_environment_{timestamp}.txt"
    )

    csv_directory.mkdir(parents=True, exist_ok=True)
    save_environment_report(environment_path, args)

    print("=" * 96)
    print("LLM-KernelLab Hugging Face RMSNorm Integration Benchmark")
    print("=" * 96)
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"Triton:          {triton.__version__}")
    print(f"Transformers:    {transformers.__version__}")
    print(f"Git commit:      {get_git_commit()}")
    print(f"Mode:            {'quick' if args.quick else 'full'}")
    print(f"Run index:       {args.run_index}")
    print(f"Scopes:          {', '.join(args.scopes)}")
    print(f"Families:        {', '.join(args.families)}")
    print(f"Dtypes:          {', '.join(dtype_names)}")
    print(f"Warmup:          {args.warmup_ms} ms")
    print(f"Repetition:      {args.rep_ms} ms")
    print(
        "Compact model:   "
        f"hidden={args.model_hidden_size}, "
        f"intermediate={args.model_intermediate_size}, "
        f"layers={args.model_layers}, "
        f"heads={args.model_attention_heads}, "
        f"kv_heads={args.model_key_value_heads}, "
        f"vocab={args.model_vocab_size}"
    )
    print("=" * 96)
    print(
        "Important: single_token_no_cache is a decode-like single-token "
        "forward without KV cache, not true autoregressive decode latency."
    )

    module_records: list[dict[str, object]] = []
    model_records: list[dict[str, object]] = []
    global_case_index = 0

    if "module" in args.scopes:
        for dtype_name in dtype_names:
            for family in args.families:
                for (
                    case_name,
                    batch_size,
                    sequence_length,
                    hidden_size,
                ) in module_cases:
                    module_records.extend(
                        benchmark_module_case(
                            family=family,
                            case_name=case_name,
                            batch_size=batch_size,
                            sequence_length=sequence_length,
                            hidden_size=hidden_size,
                            dtype_name=dtype_name,
                            run_index=args.run_index,
                            case_index=global_case_index,
                            warmup_ms=args.warmup_ms,
                            repetition_ms=args.rep_ms,
                            eps=args.eps,
                        )
                    )
                    global_case_index += 1

    if "model" in args.scopes:
        max_sequence_length = max(
            sequence_length
            for _, _, sequence_length in model_cases
        )
        max_position_embeddings = max(
            64,
            max_sequence_length,
        )

        for dtype_name in dtype_names:
            dtype = DTYPE_MAP[dtype_name]

            for family in args.families:
                print()
                print(
                    f"Building compact {family} models for dtype={dtype_name}"
                )

                (
                    baseline_model,
                    triton_model,
                    replaced_names,
                    original_rmsnorm_count,
                ) = make_complete_models(
                    family=family,
                    dtype=dtype,
                    args=args,
                    max_position_embeddings=max_position_embeddings,
                )

                try:
                    for (
                        case_name,
                        batch_size,
                        sequence_length,
                    ) in model_cases:
                        model_records.extend(
                            benchmark_complete_model_case(
                                family=family,
                                case_name=case_name,
                                batch_size=batch_size,
                                sequence_length=sequence_length,
                                dtype_name=dtype_name,
                                baseline_model=baseline_model,
                                triton_model=triton_model,
                                replaced_names=replaced_names,
                                original_rmsnorm_count=(
                                    original_rmsnorm_count
                                ),
                                run_index=args.run_index,
                                case_index=global_case_index,
                                warmup_ms=args.warmup_ms,
                                repetition_ms=args.rep_ms,
                                vocab_size=args.model_vocab_size,
                            )
                        )
                        global_case_index += 1
                finally:
                    del triton_model
                    del baseline_model
                    torch.cuda.empty_cache()

    module_dataframe = pd.DataFrame(module_records)
    model_dataframe = pd.DataFrame(model_records)

    if not module_dataframe.empty:
        module_dataframe = add_speedup_column(
            module_dataframe,
            group_columns=[
                "family",
                "case_name",
                "batch_size",
                "sequence_length",
                "hidden_size",
                "dtype",
            ],
            baseline_provider="huggingface",
            output_column="speedup_vs_huggingface",
        )
        module_dataframe = insert_metadata(module_dataframe)
        module_dataframe.to_csv(
            module_output_path,
            index=False,
            float_format="%.9f",
        )
        print_module_summary(module_dataframe)

    if not model_dataframe.empty:
        model_dataframe = add_speedup_column(
            model_dataframe,
            group_columns=[
                "family",
                "case_name",
                "batch_size",
                "sequence_length",
                "dtype",
            ],
            baseline_provider="huggingface_model",
            output_column="speedup_vs_huggingface_model",
        )
        model_dataframe = insert_metadata(model_dataframe)
        model_dataframe.to_csv(
            model_output_path,
            index=False,
            float_format="%.9f",
        )
        print_model_summary(model_dataframe)

    print()
    if not module_dataframe.empty:
        print(f"Module CSV saved to:      {module_output_path}")
    if not model_dataframe.empty:
        print(f"Model CSV saved to:       {model_output_path}")
    print(f"Environment saved to:     {environment_path}")
    print(
        "The complete-model benchmark uses randomly initialized compact "
        "models and does not download pretrained weights."
    )
    print(
        "single_token_no_cache is not a true KV-cache decode benchmark."
    )

    failed_frames = []

    if not module_dataframe.empty:
        failed_frames.append(
            module_dataframe[module_dataframe["status"] != "ok"]
        )

    if not model_dataframe.empty:
        failed_frames.append(
            model_dataframe[model_dataframe["status"] != "ok"]
        )

    if any(not frame.empty for frame in failed_frames):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
