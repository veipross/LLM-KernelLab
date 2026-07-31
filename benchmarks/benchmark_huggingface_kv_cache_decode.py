"""Benchmark true KV-cache decode for compact Hugging Face Llama and Qwen2.

The benchmark does not download pretrained weights. It constructs randomly
initialized compact models, performs a real prefill with ``use_cache=True``,
and then measures one-token decode while passing and updating
``past_key_values``.

For steady-state decode timing, each provider owns an independent
``DynamicCache``. After every timed one-token decode, the cache is cropped back
to the original prefill length outside the CUDA-event timing interval. This
keeps the context length fixed without including cache-reset bookkeeping in the
reported GPU latency.
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
    DynamicCache,
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
}

FAMILY_ORDER = [
    "llama",
    "qwen2",
]

PROVIDER_ORDER = [
    "huggingface_model",
    "triton_rmsnorm_model",
]

HF_RMSNORM_TYPES = (
    LlamaRMSNorm,
    Qwen2RMSNorm,
)


class BenchmarkError(RuntimeError):
    """Raised when benchmark validation detects an invalid result."""


def synchronize() -> None:
    torch.cuda.synchronize()


def run_command(command: list[str]) -> str:
    try:
        output = subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return "\n".join(line.rstrip() for line in output.splitlines())
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


def provider_order_for_case(
    providers: Sequence[str],
    run_index: int,
    case_index: int,
) -> list[str]:
    provider_list = list(providers)
    offset = (run_index - 1 + case_index) % len(provider_list)
    return provider_list[offset:] + provider_list[:offset]


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


def output_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    difference = (actual.detach().float() - expected.detach().float()).abs()
    return difference.max().item(), difference.mean().item()


def dtype_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 5e-2, 5e-2
    raise ValueError(f"Unsupported dtype: {dtype}")


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list.")

    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, quantile).item())


def summarize_latencies(values: list[float]) -> tuple[float, float, float]:
    return (
        percentile(values, 0.50),
        percentile(values, 0.95),
        float(sum(values) / len(values)),
    )


def count_triton_calls(
    function: Callable[[], Any],
) -> tuple[Any, int]:
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
        "use_cache": True,
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


def make_equal_models(
    family: str,
    dtype: torch.dtype,
    args: argparse.Namespace,
    max_position_embeddings: int,
) -> tuple[nn.Module, nn.Module, tuple[str, ...], int]:
    config = build_model_config(
        family=family,
        args=args,
        max_position_embeddings=max_position_embeddings,
    )

    seed = 2026 + args.model_hidden_size + args.model_layers
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    baseline_model = build_model(family, config).eval()
    triton_model = build_model(family, config).eval()
    triton_model.load_state_dict(baseline_model.state_dict(), strict=True)

    original_rmsnorm_count = count_modules(triton_model, HF_RMSNORM_TYPES)
    if original_rmsnorm_count <= 0:
        raise BenchmarkError(
            f"No Hugging Face RMSNorm modules found in {family} model."
        )

    replaced_names = tuple(
        replace_huggingface_rmsnorm_modules(triton_model)
    )

    if len(replaced_names) != original_rmsnorm_count:
        raise BenchmarkError(
            "RMSNorm replacement count mismatch: "
            f"expected={original_rmsnorm_count}, "
            f"actual={len(replaced_names)}."
        )

    if count_modules(triton_model, HF_RMSNORM_TYPES) != 0:
        raise BenchmarkError(
            "Original Hugging Face RMSNorm modules remain after replacement."
        )

    adapter_count = count_modules(triton_model, HuggingFaceTritonRMSNorm)
    if adapter_count != original_rmsnorm_count:
        raise BenchmarkError(
            "Triton adapter count mismatch: "
            f"expected={original_rmsnorm_count}, actual={adapter_count}."
        )

    baseline_model = baseline_model.cuda().to(dtype=dtype).eval()
    triton_model = triton_model.cuda().to(dtype=dtype).eval()

    return (
        baseline_model,
        triton_model,
        replaced_names,
        original_rmsnorm_count,
    )


def run_prefill(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )


def run_decode(
    model: nn.Module,
    next_token: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: DynamicCache,
):
    return model(
        input_ids=next_token,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )


def validate_case(
    *,
    baseline_model: nn.Module,
    triton_model: nn.Module,
    prefill_input_ids: torch.Tensor,
    prefill_attention_mask: torch.Tensor,
    decode_tokens: torch.Tensor,
    original_rmsnorm_count: int,
    dtype: torch.dtype,
) -> dict[str, float | int]:
    with torch.inference_mode():
        baseline_prefill = run_prefill(
            baseline_model,
            prefill_input_ids,
            prefill_attention_mask,
        )
    synchronize()

    triton_prefill, prefill_triton_calls = count_triton_calls(
        lambda: run_prefill(
            triton_model,
            prefill_input_ids,
            prefill_attention_mask,
        )
    )

    baseline_cache = baseline_prefill.past_key_values
    triton_cache = triton_prefill.past_key_values

    if not isinstance(baseline_cache, DynamicCache):
        raise BenchmarkError(
            "Baseline prefill did not return DynamicCache: "
            f"{type(baseline_cache)}"
        )
    if not isinstance(triton_cache, DynamicCache):
        raise BenchmarkError(
            "Triton prefill did not return DynamicCache: "
            f"{type(triton_cache)}"
        )
    if baseline_cache is triton_cache:
        raise BenchmarkError("Baseline and Triton unexpectedly share one cache.")

    prefill_length = prefill_input_ids.shape[1]
    if baseline_cache.get_seq_length() != prefill_length:
        raise BenchmarkError("Baseline prefill cache length is incorrect.")
    if triton_cache.get_seq_length() != prefill_length:
        raise BenchmarkError("Triton prefill cache length is incorrect.")
    if prefill_triton_calls != original_rmsnorm_count:
        raise BenchmarkError(
            "Prefill Triton call count mismatch: "
            f"expected={original_rmsnorm_count}, "
            f"actual={prefill_triton_calls}."
        )

    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(
        triton_prefill.logits,
        baseline_prefill.logits,
        rtol=rtol,
        atol=atol,
    )

    prefill_max_error, prefill_mean_error = output_errors(
        triton_prefill.logits,
        baseline_prefill.logits,
    )

    decode_max_errors: list[float] = []
    decode_mean_errors: list[float] = []
    decode_call_counts: list[int] = []

    for step in range(decode_tokens.shape[1]):
        next_token = decode_tokens[:, step : step + 1]
        total_length = prefill_length + step + 1
        decode_attention_mask = torch.ones(
            (prefill_input_ids.shape[0], total_length),
            device=prefill_input_ids.device,
            dtype=torch.long,
        )

        with torch.inference_mode():
            baseline_decode = run_decode(
                baseline_model,
                next_token,
                decode_attention_mask,
                baseline_cache,
            )
        synchronize()

        triton_decode, current_triton_calls = count_triton_calls(
            lambda: run_decode(
                triton_model,
                next_token,
                decode_attention_mask,
                triton_cache,
            )
        )

        baseline_cache = baseline_decode.past_key_values
        triton_cache = triton_decode.past_key_values

        if baseline_cache.get_seq_length() != total_length:
            raise BenchmarkError(
                "Baseline decode cache length mismatch at step "
                f"{step + 1}."
            )
        if triton_cache.get_seq_length() != total_length:
            raise BenchmarkError(
                "Triton decode cache length mismatch at step "
                f"{step + 1}."
            )
        if current_triton_calls != original_rmsnorm_count:
            raise BenchmarkError(
                "Decode Triton call count mismatch at step "
                f"{step + 1}: expected={original_rmsnorm_count}, "
                f"actual={current_triton_calls}."
            )

        torch.testing.assert_close(
            triton_decode.logits,
            baseline_decode.logits,
            rtol=rtol,
            atol=atol,
        )

        max_error, mean_error = output_errors(
            triton_decode.logits,
            baseline_decode.logits,
        )
        decode_max_errors.append(max_error)
        decode_mean_errors.append(mean_error)
        decode_call_counts.append(current_triton_calls)

    return {
        "validated_prefill_triton_calls": prefill_triton_calls,
        "validated_decode_triton_calls_per_token": min(decode_call_counts),
        "prefill_max_abs_error": prefill_max_error,
        "prefill_mean_abs_error": prefill_mean_error,
        "decode_max_abs_error": max(decode_max_errors),
        "decode_mean_abs_error": (
            sum(decode_mean_errors) / len(decode_mean_errors)
        ),
        "validated_decode_steps": len(decode_call_counts),
    }


def warmup_prefill(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    iterations: int,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            output = run_prefill(model, input_ids, attention_mask)
            del output
    synchronize()


def measure_prefill(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    iterations: int,
) -> list[float]:
    latencies_ms: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for _ in range(iterations):
            start_event.record()
            output = run_prefill(model, input_ids, attention_mask)
            end_event.record()
            end_event.synchronize()
            latencies_ms.append(float(start_event.elapsed_time(end_event)))
            del output

    return latencies_ms


def build_prefill_cache(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> DynamicCache:
    with torch.inference_mode():
        output = run_prefill(model, input_ids, attention_mask)
    synchronize()

    cache = output.past_key_values
    if not isinstance(cache, DynamicCache):
        raise BenchmarkError(
            f"Expected DynamicCache, received {type(cache)}."
        )
    return cache


def warmup_decode(
    *,
    model: nn.Module,
    next_token: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: DynamicCache,
    prefill_length: int,
    iterations: int,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            output = run_decode(
                model,
                next_token,
                attention_mask,
                cache,
            )
            synchronize()

            if cache.get_seq_length() != prefill_length + 1:
                raise BenchmarkError(
                    "Decode warmup cache length did not increase by one."
                )

            cache.crop(prefill_length)
            if cache.get_seq_length() != prefill_length:
                raise BenchmarkError(
                    "Decode warmup cache crop did not restore prefill length."
                )
            del output


def measure_decode(
    *,
    model: nn.Module,
    next_token: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: DynamicCache,
    prefill_length: int,
    iterations: int,
) -> list[float]:
    latencies_ms: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for _ in range(iterations):
            if cache.get_seq_length() != prefill_length:
                raise BenchmarkError(
                    "Decode measurement did not start at fixed context length."
                )

            start_event.record()
            output = run_decode(
                model,
                next_token,
                attention_mask,
                cache,
            )
            end_event.record()
            end_event.synchronize()

            latencies_ms.append(float(start_event.elapsed_time(end_event)))

            if cache.get_seq_length() != prefill_length + 1:
                raise BenchmarkError(
                    "Timed decode cache length did not increase by one."
                )

            # Keep crop outside the CUDA-event timing interval.
            cache.crop(prefill_length)
            if cache.get_seq_length() != prefill_length:
                raise BenchmarkError(
                    "Timed decode cache crop did not restore prefill length."
                )
            del output

    return latencies_ms


def benchmark_provider(
    *,
    provider: str,
    model: nn.Module,
    prefill_input_ids: torch.Tensor,
    prefill_attention_mask: torch.Tensor,
    next_token: torch.Tensor,
    decode_attention_mask: torch.Tensor,
    prefill_length: int,
    warmup_iterations: int,
    measure_iterations: int,
) -> dict[str, float]:
    print(f"  Warming up {provider:<22}", end="", flush=True)

    warmup_prefill(
        model=model,
        input_ids=prefill_input_ids,
        attention_mask=prefill_attention_mask,
        iterations=warmup_iterations,
    )

    cache = build_prefill_cache(
        model=model,
        input_ids=prefill_input_ids,
        attention_mask=prefill_attention_mask,
    )

    warmup_decode(
        model=model,
        next_token=next_token,
        attention_mask=decode_attention_mask,
        cache=cache,
        prefill_length=prefill_length,
        iterations=warmup_iterations,
    )
    print(" done")

    print(f"  Measuring {provider:<23}", end="", flush=True)

    prefill_latencies = measure_prefill(
        model=model,
        input_ids=prefill_input_ids,
        attention_mask=prefill_attention_mask,
        iterations=measure_iterations,
    )

    decode_latencies = measure_decode(
        model=model,
        next_token=next_token,
        attention_mask=decode_attention_mask,
        cache=cache,
        prefill_length=prefill_length,
        iterations=measure_iterations,
    )

    prefill_p50, prefill_p95, prefill_mean = summarize_latencies(
        prefill_latencies
    )
    decode_p50, decode_p95, decode_mean = summarize_latencies(
        decode_latencies
    )

    print(
        " done | "
        f"prefill P50={prefill_p50:.6f} ms | "
        f"decode P50={decode_p50:.6f} ms | "
        f"decode P95={decode_p95:.6f} ms"
    )

    return {
        "prefill_p50_ms": prefill_p50,
        "prefill_p95_ms": prefill_p95,
        "prefill_mean_ms": prefill_mean,
        "decode_p50_ms": decode_p50,
        "decode_p95_ms": decode_p95,
        "decode_mean_ms": decode_mean,
        "decode_tokens_per_second": 1000.0 / decode_p50,
    }


def benchmark_case(
    *,
    family: str,
    dtype_name: str,
    prefill_length: int,
    run_index: int,
    case_index: int,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    dtype = DTYPE_MAP[dtype_name]
    max_position_embeddings = max(
        args.model_max_position_embeddings,
        prefill_length + args.validation_decode_steps + 1,
    )

    (
        baseline_model,
        triton_model,
        replaced_names,
        original_rmsnorm_count,
    ) = make_equal_models(
        family=family,
        dtype=dtype,
        args=args,
        max_position_embeddings=max_position_embeddings,
    )

    seed = 3026 + prefill_length + case_index
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    prefill_input_ids = torch.randint(
        low=3,
        high=args.model_vocab_size,
        size=(args.batch_size, prefill_length),
        generator=generator,
        device="cuda",
        dtype=torch.long,
    )
    prefill_attention_mask = torch.ones_like(
        prefill_input_ids,
        dtype=torch.long,
    )
    decode_tokens = torch.randint(
        low=3,
        high=args.model_vocab_size,
        size=(args.batch_size, args.validation_decode_steps),
        generator=generator,
        device="cuda",
        dtype=torch.long,
    )
    benchmark_next_token = decode_tokens[:, 0:1]
    decode_attention_mask = torch.ones(
        (args.batch_size, prefill_length + 1),
        device="cuda",
        dtype=torch.long,
    )

    print()
    print("=" * 104)
    print(
        "KV-cache case: "
        f"family={family}, dtype={dtype_name}, "
        f"batch={args.batch_size}, prefill_length={prefill_length}"
    )
    print(
        f"  Parameters: {model_parameter_count(baseline_model):,} | "
        f"Replaced RMSNorm modules: {original_rmsnorm_count}"
    )

    validation = validate_case(
        baseline_model=baseline_model,
        triton_model=triton_model,
        prefill_input_ids=prefill_input_ids,
        prefill_attention_mask=prefill_attention_mask,
        decode_tokens=decode_tokens,
        original_rmsnorm_count=original_rmsnorm_count,
        dtype=dtype,
    )

    print(
        "  Validation passed | "
        f"prefill max error={validation['prefill_max_abs_error']:.9f} | "
        f"decode max error={validation['decode_max_abs_error']:.9f} | "
        f"Triton calls/forward="
        f"{validation['validated_decode_triton_calls_per_token']}"
    )

    models = {
        "huggingface_model": baseline_model,
        "triton_rmsnorm_model": triton_model,
    }

    ordered_providers = provider_order_for_case(
        PROVIDER_ORDER,
        run_index,
        case_index,
    )
    order_text = "|".join(ordered_providers)
    print(f"  Provider order: {order_text}")

    records: list[dict[str, object]] = []

    try:
        for provider_position, provider in enumerate(
            ordered_providers,
            start=1,
        ):
            metrics = benchmark_provider(
                provider=provider,
                model=models[provider],
                prefill_input_ids=prefill_input_ids,
                prefill_attention_mask=prefill_attention_mask,
                next_token=benchmark_next_token,
                decode_attention_mask=decode_attention_mask,
                prefill_length=prefill_length,
                warmup_iterations=args.warmup_iterations,
                measure_iterations=args.measure_iterations,
            )

            is_triton = provider == "triton_rmsnorm_model"
            record: dict[str, object] = {
                "scope": "kv_cache_decode",
                "run_index": run_index,
                "case_index": case_index,
                "provider_position": provider_position,
                "case_provider_order": order_text,
                "family": family,
                "dtype": dtype_name,
                "provider": provider,
                "batch_size": args.batch_size,
                "prefill_length": prefill_length,
                "decode_input_tokens": 1,
                "hidden_size": args.model_hidden_size,
                "intermediate_size": args.model_intermediate_size,
                "num_hidden_layers": args.model_layers,
                "num_attention_heads": args.model_attention_heads,
                "num_key_value_heads": args.model_key_value_heads,
                "vocab_size": args.model_vocab_size,
                "parameter_count": model_parameter_count(baseline_model),
                "replaced_rmsnorm_count": len(replaced_names),
                "cache_type": "DynamicCache",
                "use_cache": True,
                "logits_to_keep": 1,
                "benchmark_semantics": (
                    "real one-token KV-cache decode at fixed context length; "
                    "cache crop excluded from CUDA-event timing"
                ),
                "warmup_iterations": args.warmup_iterations,
                "measure_iterations": args.measure_iterations,
                "validation_decode_steps": args.validation_decode_steps,
                "expected_prefill_triton_calls": (
                    original_rmsnorm_count if is_triton else 0
                ),
                "validated_prefill_triton_calls": (
                    validation["validated_prefill_triton_calls"]
                    if is_triton
                    else 0
                ),
                "expected_decode_triton_calls_per_token": (
                    original_rmsnorm_count if is_triton else 0
                ),
                "validated_decode_triton_calls_per_token": (
                    validation[
                        "validated_decode_triton_calls_per_token"
                    ]
                    if is_triton
                    else 0
                ),
                "prefill_max_abs_error": (
                    validation["prefill_max_abs_error"] if is_triton else 0.0
                ),
                "prefill_mean_abs_error": (
                    validation["prefill_mean_abs_error"] if is_triton else 0.0
                ),
                "decode_max_abs_error": (
                    validation["decode_max_abs_error"] if is_triton else 0.0
                ),
                "decode_mean_abs_error": (
                    validation["decode_mean_abs_error"] if is_triton else 0.0
                ),
                "status": "ok",
                "error": "",
                **metrics,
            }
            records.append(record)
    finally:
        del triton_model
        del baseline_model
        del prefill_input_ids
        del prefill_attention_mask
        del decode_tokens
        del benchmark_next_token
        del decode_attention_mask
        torch.cuda.empty_cache()

    return records


def add_speedup_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["prefill_speedup_vs_huggingface"] = float("nan")
    dataframe["decode_speedup_vs_huggingface"] = float("nan")

    group_columns = [
        "run_index",
        "family",
        "dtype",
        "batch_size",
        "prefill_length",
    ]

    for _, group in dataframe.groupby(group_columns):
        baseline_rows = group[
            group["provider"] == "huggingface_model"
        ]
        if baseline_rows.empty:
            continue

        baseline_prefill = float(
            baseline_rows.iloc[0]["prefill_p50_ms"]
        )
        baseline_decode = float(
            baseline_rows.iloc[0]["decode_p50_ms"]
        )

        for index in group.index:
            prefill_latency = float(dataframe.at[index, "prefill_p50_ms"])
            decode_latency = float(dataframe.at[index, "decode_p50_ms"])
            dataframe.at[index, "prefill_speedup_vs_huggingface"] = (
                baseline_prefill / prefill_latency
            )
            dataframe.at[index, "decode_speedup_vs_huggingface"] = (
                baseline_decode / decode_latency
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


def print_summary(dataframe: pd.DataFrame) -> None:
    columns = [
        "family",
        "dtype",
        "prefill_length",
        "provider",
        "prefill_p50_ms",
        "prefill_p95_ms",
        "prefill_speedup_vs_huggingface",
        "decode_p50_ms",
        "decode_p95_ms",
        "decode_tokens_per_second",
        "decode_speedup_vs_huggingface",
        "validated_decode_triton_calls_per_token",
        "decode_max_abs_error",
        "status",
    ]

    print()
    print("=" * 190)
    print("Compact Llama/Qwen2 True KV-Cache Decode Benchmark Summary")
    print("=" * 190)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        300,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        print(dataframe[columns].to_string(index=False))

    print("=" * 190)


def save_environment_report(
    path: Path,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(0)

    lines = [
        "LLM-KernelLab Hugging Face True KV-Cache Decode Benchmark",
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
        f"compute_capability: {properties.major}.{properties.minor}",
        f"gpu_memory_gib: {properties.total_memory / 1024**3:.2f}",
        f"multiprocessors: {properties.multi_processor_count}",
        f"run_index: {args.run_index}",
        f"families: {','.join(args.families)}",
        f"dtypes: {','.join(args.dtypes)}",
        f"prefill_lengths: {','.join(str(v) for v in args.prefill_lengths)}",
        f"batch_size: {args.batch_size}",
        f"warmup_iterations: {args.warmup_iterations}",
        f"measure_iterations: {args.measure_iterations}",
        f"validation_decode_steps: {args.validation_decode_steps}",
        f"model_hidden_size: {args.model_hidden_size}",
        f"model_intermediate_size: {args.model_intermediate_size}",
        f"model_layers: {args.model_layers}",
        f"model_attention_heads: {args.model_attention_heads}",
        f"model_key_value_heads: {args.model_key_value_heads}",
        f"model_vocab_size: {args.model_vocab_size}",
        f"model_max_position_embeddings: {args.model_max_position_embeddings}",
        "timing: CUDA events",
        "decode_cache_policy: DynamicCache.crop(prefill_length) outside timing",
        "external_weights_downloaded: false",
        "",
        "===== nvidia-smi =====",
        run_command(["nvidia-smi"]),
        "",
        "===== nvcc =====",
        run_command(["nvcc", "--version"]),
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def validate_arguments(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    if args.run_index <= 0:
        raise ValueError("--run-index must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.warmup_iterations <= 0:
        raise ValueError("--warmup-iterations must be positive.")
    if args.measure_iterations <= 0:
        raise ValueError("--measure-iterations must be positive.")
    if args.validation_decode_steps <= 0:
        raise ValueError("--validation-decode-steps must be positive.")
    if args.eps <= 0:
        raise ValueError("--eps must be positive.")

    for value in args.prefill_lengths:
        if value <= 0:
            raise ValueError("All --prefill-lengths must be positive.")

    positive_model_arguments = {
        "--model-hidden-size": args.model_hidden_size,
        "--model-intermediate-size": args.model_intermediate_size,
        "--model-layers": args.model_layers,
        "--model-attention-heads": args.model_attention_heads,
        "--model-key-value-heads": args.model_key_value_heads,
        "--model-vocab-size": args.model_vocab_size,
        "--model-max-position-embeddings": (
            args.model_max_position_embeddings
        ),
    }

    for name, value in positive_model_arguments.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    if args.model_hidden_size % args.model_attention_heads != 0:
        raise ValueError(
            "--model-hidden-size must be divisible by "
            "--model-attention-heads."
        )
    if args.model_attention_heads % args.model_key_value_heads != 0:
        raise ValueError(
            "--model-attention-heads must be divisible by "
            "--model-key-value-heads."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark real one-token KV-cache decode for randomly initialized "
            "compact Hugging Face Llama/Qwen2 models before and after Triton "
            "RMSNorm replacement."
        )
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
        default=["fp16"],
        help="Data types to benchmark. Start with fp16 for quick validation.",
    )
    parser.add_argument(
        "--prefill-lengths",
        nargs="+",
        type=int,
        default=[32],
        help="Context lengths produced by prefill before one-token decode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for prefill and decode.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=1,
        help="Positive repeated-run index used to rotate provider order.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=5,
        help="Untimed warmup calls per prefill and decode provider.",
    )
    parser.add_argument(
        "--measure-iterations",
        type=int,
        default=30,
        help="Timed calls used to calculate P50 and P95.",
    )
    parser.add_argument(
        "--validation-decode-steps",
        type=int,
        default=4,
        help="Sequential decode steps used for correctness validation.",
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
        help="Hidden size of each compact model.",
    )
    parser.add_argument(
        "--model-intermediate-size",
        type=int,
        default=1376,
        help="MLP intermediate size of each compact model.",
    )
    parser.add_argument(
        "--model-layers",
        type=int,
        default=4,
        help="Number of Transformer layers.",
    )
    parser.add_argument(
        "--model-attention-heads",
        type=int,
        default=8,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--model-key-value-heads",
        type=int,
        default=4,
        help="Number of key/value heads.",
    )
    parser.add_argument(
        "--model-vocab-size",
        type=int,
        default=2048,
        help="Vocabulary size.",
    )
    parser.add_argument(
        "--model-max-position-embeddings",
        type=int,
        default=2048,
        help="Minimum maximum position embedding length.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_directory = args.output_dir / "csv"
    environment_directory = args.output_dir / "environment"
    csv_directory.mkdir(parents=True, exist_ok=True)

    csv_path = (
        csv_directory
        / f"huggingface_kv_cache_decode_benchmark_{timestamp}.csv"
    )
    environment_path = (
        environment_directory
        / f"huggingface_kv_cache_decode_environment_{timestamp}.txt"
    )
    save_environment_report(environment_path, args)

    print("=" * 104)
    print("LLM-KernelLab Hugging Face True KV-Cache Decode Benchmark")
    print("=" * 104)
    print(f"GPU:                   {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:               {torch.__version__}")
    print(f"Triton:                {triton.__version__}")
    print(f"Transformers:          {transformers.__version__}")
    print(f"Git commit:            {get_git_commit()}")
    print(f"Run index:             {args.run_index}")
    print(f"Families:              {', '.join(args.families)}")
    print(f"Dtypes:                {', '.join(args.dtypes)}")
    print(
        "Prefill lengths:        "
        + ", ".join(str(value) for value in args.prefill_lengths)
    )
    print(f"Warmup iterations:     {args.warmup_iterations}")
    print(f"Measure iterations:    {args.measure_iterations}")
    print(f"Validation steps:      {args.validation_decode_steps}")
    print(
        "Compact model:         "
        f"hidden={args.model_hidden_size}, "
        f"intermediate={args.model_intermediate_size}, "
        f"layers={args.model_layers}, "
        f"heads={args.model_attention_heads}, "
        f"kv_heads={args.model_key_value_heads}, "
        f"vocab={args.model_vocab_size}"
    )
    print("Timing:                CUDA events")
    print(
        "Decode cache policy:   crop back to fixed prefill length "
        "outside timing"
    )
    print("External downloads:    none")
    print("=" * 104)

    records: list[dict[str, object]] = []
    case_index = 0

    for dtype_name in args.dtypes:
        for family in args.families:
            for prefill_length in args.prefill_lengths:
                records.extend(
                    benchmark_case(
                        family=family,
                        dtype_name=dtype_name,
                        prefill_length=prefill_length,
                        run_index=args.run_index,
                        case_index=case_index,
                        args=args,
                    )
                )
                case_index += 1

    dataframe = pd.DataFrame(records)
    if dataframe.empty:
        raise BenchmarkError("No benchmark records were generated.")

    dataframe = add_speedup_columns(dataframe)
    dataframe = insert_metadata(dataframe)
    dataframe.to_csv(csv_path, index=False, float_format="%.9f")

    print_summary(dataframe)
    print()
    print(f"CSV saved to:           {csv_path}")
    print(f"Environment saved to:   {environment_path}")
    print(
        "This benchmark performed prefill first, passed independent "
        "DynamicCache objects into one-token decode, and used use_cache=True."
    )


if __name__ == "__main__":
    main()
