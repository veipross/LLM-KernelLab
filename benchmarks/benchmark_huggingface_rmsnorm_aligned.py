"""对齐 hidden_size=512 的 Hugging Face RMSNorm 模块级 Benchmark。

目的：
- 使用与紧凑 Llama/Qwen2 整模型完全一致的 hidden_size=512；
- 使用与整模型场景一致的 batch/sequence shape；
- 比较 Hugging Face 原始 RMSNorm 与 LLM-KernelLab Triton 适配器；
- 判断整模型端到端收益消失是否来自小 hidden size。

该脚本不会下载任何模型权重。
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch import nn
import transformers
import triton

from integrations import HuggingFaceTritonRMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

FAMILIES = {
    "llama": LlamaRMSNorm,
    "qwen2": Qwen2RMSNorm,
}

CASES = (
    ("single_token_no_cache", 1, 1),
    ("prefill_s32", 1, 32),
    ("prefill_s128", 1, 128),
    ("prefill_b4_s32", 4, 32),
)

PROVIDERS = (
    "huggingface",
    "triton_adapter",
)


def synchronize() -> None:
    torch.cuda.synchronize()


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


def dtype_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 3e-3, 3e-3

    if dtype == torch.bfloat16:
        return 2e-2, 2e-2

    return 1e-4, 1e-4


def provider_order_for_case(
    run_index: int,
    case_index: int,
) -> list[str]:
    providers = list(PROVIDERS)
    offset = (run_index - 1 + case_index) % len(providers)
    return providers[offset:] + providers[:offset]


def measure_first_call(
    function: Callable[[], torch.Tensor],
) -> tuple[torch.Tensor, float]:
    synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = function()

    synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output, elapsed_ms


def benchmark_function(
    function: Callable[[], torch.Tensor],
    *,
    warmup_ms: int,
    repetition_ms: int,
) -> tuple[float, float]:
    with torch.inference_mode():
        timings = triton.testing.do_bench(
            function,
            warmup=warmup_ms,
            rep=repetition_ms,
            quantiles=[0.50, 0.95],
        )

    p50_ms, p95_ms = (float(value) for value in timings)
    return p50_ms, p95_ms


def output_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    difference = (
        actual.float() - expected.float()
    ).abs()

    return difference.max().item(), difference.mean().item()


def make_modules(
    family: str,
    *,
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
) -> tuple[nn.Module, HuggingFaceTritonRMSNorm]:
    module_class = FAMILIES[family]

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


def benchmark_case(
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
    dtype = DTYPE_MAP[dtype_name]

    seed = (
        2026
        + run_index
        + batch_size
        + sequence_length
        + hidden_size
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    original, adapter = make_modules(
        family,
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

        if not adapter._can_use_triton(hidden_states):
            raise RuntimeError(
                "当前输入没有进入 Triton 快速路径。"
            )

        triton_output = functions["triton_adapter"]()

    synchronize()

    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(
        triton_output,
        reference,
        rtol=rtol,
        atol=atol,
    )

    ordered_providers = provider_order_for_case(
        run_index,
        case_index,
    )
    provider_order_text = "|".join(ordered_providers)

    print()
    print(
        f"Case: family={family}, case={case_name}, "
        f"batch={batch_size}, seq={sequence_length}, "
        f"rows={batch_size * sequence_length}, "
        f"hidden={hidden_size}, dtype={dtype_name}"
    )
    print(f"  Provider order: {provider_order_text}")

    records: list[dict[str, object]] = []

    for provider_position, provider in enumerate(
        ordered_providers,
        start=1,
    ):
        print(
            f"  Benchmarking {provider:<16}",
            end="",
            flush=True,
        )

        record: dict[str, object] = {
            "run_index": run_index,
            "case_index": case_index,
            "provider_position": provider_position,
            "case_provider_order": provider_order_text,
            "family": family,
            "case_name": case_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "rows": batch_size * sequence_length,
            "hidden_size": hidden_size,
            "numel": hidden_states.numel(),
            "dtype": dtype_name,
            "provider": provider,
            "input_is_contiguous": bool(
                hidden_states.is_contiguous()
            ),
            "weight_is_contiguous": bool(
                adapter.weight.is_contiguous()
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

            max_error, mean_error = output_errors(
                output,
                reference,
            )

            p50_ms, p95_ms = benchmark_function(
                function,
                warmup_ms=warmup_ms,
                repetition_ms=repetition_ms,
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

    del triton_output
    del reference
    del hidden_states
    del adapter
    del original
    torch.cuda.empty_cache()

    return records


def add_speedup_column(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["speedup_vs_huggingface"] = float("nan")

    group_columns = [
        "family",
        "case_name",
        "batch_size",
        "sequence_length",
        "hidden_size",
        "dtype",
    ]

    for _, group in dataframe.groupby(group_columns):
        valid = group[group["status"] == "ok"]
        baseline_rows = valid[
            valid["provider"] == "huggingface"
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

            dataframe.at[
                index,
                "speedup_vs_huggingface",
            ] = baseline_latency / float(latency)

    return dataframe


def save_environment(
    path: Path,
    args: argparse.Namespace,
) -> None:
    properties = torch.cuda.get_device_properties(0)

    lines = [
        "LLM-KernelLab hidden_size=512 aligned RMSNorm benchmark",
        "=" * 72,
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
        f"hidden_size: {args.hidden_size}",
        f"run_index: {args.run_index}",
        f"families: {','.join(args.families)}",
        f"dtypes: {','.join(args.dtypes)}",
        f"warmup_ms: {args.warmup_ms}",
        f"repetition_ms: {args.rep_ms}",
        "",
        "===== nvidia-smi =====",
        run_command(["nvidia-smi"]),
        "",
        "===== nvcc =====",
        run_command(["nvcc", "--version"]),
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Hugging Face RMSNorm and Triton adapter "
            "with hidden_size=512 aligned to compact models."
        )
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILIES),
        default=tuple(FAMILIES),
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=tuple(DTYPE_MAP),
        default=tuple(DTYPE_MAP),
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")

    if args.run_index <= 0:
        raise ValueError("--run-index 必须为正数。")

    if args.hidden_size <= 0:
        raise ValueError("--hidden-size 必须为正数。")

    if args.warmup_ms <= 0:
        raise ValueError("--warmup-ms 必须为正数。")

    if args.rep_ms <= 0:
        raise ValueError("--rep-ms 必须为正数。")

    if args.eps <= 0:
        raise ValueError("--eps 必须为正数。")


def main() -> None:
    args = parse_args()
    validate_args(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_directory = args.output_dir / "csv"
    environment_directory = args.output_dir / "environment"

    csv_directory.mkdir(parents=True, exist_ok=True)

    csv_path = (
        csv_directory
        / f"huggingface_rmsnorm_aligned_h{args.hidden_size}_"
        f"run{args.run_index}_{timestamp}.csv"
    )
    environment_path = (
        environment_directory
        / f"huggingface_rmsnorm_aligned_h{args.hidden_size}_"
        f"run{args.run_index}_{timestamp}.txt"
    )

    save_environment(environment_path, args)

    print("=" * 96)
    print("LLM-KernelLab hidden_size 对齐 RMSNorm Benchmark")
    print("=" * 96)
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Triton:       {triton.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Git commit:   {get_git_commit()}")
    print(f"Run index:    {args.run_index}")
    print(f"Hidden size:  {args.hidden_size}")
    print(f"Families:     {', '.join(args.families)}")
    print(f"Dtypes:       {', '.join(args.dtypes)}")
    print(f"Warmup:       {args.warmup_ms} ms")
    print(f"Repetition:   {args.rep_ms} ms")
    print("=" * 96)

    records: list[dict[str, object]] = []
    case_index = 0

    for dtype_name in args.dtypes:
        for family in args.families:
            for (
                case_name,
                batch_size,
                sequence_length,
            ) in CASES:
                records.extend(
                    benchmark_case(
                        family=family,
                        case_name=case_name,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        hidden_size=args.hidden_size,
                        dtype_name=dtype_name,
                        run_index=args.run_index,
                        case_index=case_index,
                        warmup_ms=args.warmup_ms,
                        repetition_ms=args.rep_ms,
                        eps=args.eps,
                    )
                )
                case_index += 1

    dataframe = pd.DataFrame(records)
    dataframe = add_speedup_column(dataframe)

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

    dataframe.to_csv(
        csv_path,
        index=False,
        float_format="%.9f",
    )

    summary_columns = [
        "family",
        "case_name",
        "batch_size",
        "sequence_length",
        "rows",
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
    print("=" * 160)
    print("hidden_size 对齐 Benchmark 汇总")
    print("=" * 160)

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
        print(
            dataframe[summary_columns].to_string(
                index=False
            )
        )

    print("=" * 160)
    print(f"CSV 已保存：{csv_path}")
    print(f"环境文件已保存：{environment_path}")

    failed = dataframe[dataframe["status"] != "ok"]
    if not failed.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
