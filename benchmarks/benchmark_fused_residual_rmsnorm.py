"""Benchmark Fused Residual + RMSNorm forward inference implementations."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn.functional as F
import triton

from llm_kernels.torch_ops import (
    fused_residual_rms_norm_reference,
)
from llm_kernels.triton_ops import (
    fused_residual_rms_norm_triton,
    rms_norm_triton,
)


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

PROVIDER_ORDER = [
    "torch_eager",
    "torch_native",
    "torch_compile",
    "triton_unfused",
    "triton_fused",
]

QUICK_CASES = [
    (1, 4096),
    (8, 4096),
    (32, 4096),
    (128, 4096),
    (512, 5120),
    (256, 8192),
]

FULL_CASES = [
    *QUICK_CASES,
    (2048, 4096),
]


def torch_fused_body(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit add and FP32-accumulated RMSNorm used by eager/compile."""

    residual_out = torch.add(x, residual).to(dtype=x.dtype)
    residual_fp32 = residual_out.float()
    weight_fp32 = weight.float()

    mean_square = residual_fp32.square().mean(
        dim=-1,
        keepdim=True,
    )
    inverse_rms = torch.rsqrt(mean_square + eps)
    normalized = (
        residual_fp32 * inverse_rms * weight_fp32
    ).to(dtype=x.dtype)

    return normalized, residual_out


def torch_native_body(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual add followed by PyTorch native RMSNorm."""

    residual_out = torch.add(x, residual).to(dtype=x.dtype)
    normalized = F.rms_norm(
        residual_out,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        eps=eps,
    )
    return normalized, residual_out


def triton_unfused_body(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One PyTorch add kernel followed by the v0.1 Triton RMSNorm kernel."""

    residual_out = torch.add(x, residual).to(dtype=x.dtype)
    normalized = rms_norm_triton(
        residual_out,
        weight,
        eps,
    )
    return normalized, residual_out


def dtype_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-3, 2e-3

    if dtype == torch.bfloat16:
        return 1e-2, 1e-2

    return 1e-5, 1e-5


def synchronize() -> None:
    torch.cuda.synchronize()


def run_command(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
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


def save_environment_report(path: Path, run_index: int) -> None:
    """Save environment metadata without touching v0.1 result files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(0)

    lines = [
        "LLM-KernelLab Fused Residual + RMSNorm Benchmark Environment",
        "=" * 72,
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"git_commit: {get_git_commit()}",
        f"run_index: {run_index}",
        f"platform: {platform.platform()}",
        f"python: {sys.version}",
        f"pytorch: {torch.__version__}",
        f"pytorch_cuda: {torch.version.cuda}",
        f"triton: {triton.__version__}",
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
        "",
        "===== nvidia-smi =====",
        run_command(["nvidia-smi"]),
        "",
        "===== nvcc =====",
        run_command(["nvcc", "--version"]),
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def make_provider(
    provider: str,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    """Build one provider callable for a benchmark case."""

    if provider == "torch_eager":
        return lambda: torch_fused_body(
            x,
            residual,
            weight,
            eps,
        )

    if provider == "torch_native":
        return lambda: torch_native_body(
            x,
            residual,
            weight,
            eps,
        )

    if provider == "torch_compile":
        torch._dynamo.reset()
        compiled_function = torch.compile(
            torch_fused_body,
            fullgraph=True,
            dynamic=False,
        )
        return lambda: compiled_function(
            x,
            residual,
            weight,
            eps,
        )

    if provider == "triton_unfused":
        return lambda: triton_unfused_body(
            x,
            residual,
            weight,
            eps,
        )

    if provider == "triton_fused":
        return lambda: fused_residual_rms_norm_triton(
            x,
            residual,
            weight,
            eps,
        )

    raise ValueError(f"Unknown provider: {provider}")


def measure_first_call(
    function: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[tuple[torch.Tensor, torch.Tensor], float]:
    """Measure synchronized wall-clock latency for the first call."""

    synchronize()
    start_time = time.perf_counter()

    outputs = function()

    synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return outputs, elapsed_ms


def benchmark_function(
    function: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    warmup_ms: int,
    repetition_ms: int,
) -> tuple[float, float]:
    """Measure steady-state P50 and P95 GPU latency."""

    timings = triton.testing.do_bench(
        function,
        warmup=warmup_ms,
        rep=repetition_ms,
        quantiles=[0.50, 0.95],
    )
    p50_ms, p95_ms = (float(value) for value in timings)
    return p50_ms, p95_ms


def calculate_logical_bandwidth_gbps(
    x: torch.Tensor,
    weight: torch.Tensor,
    latency_ms: float,
) -> float:
    """
    Calculate minimum logical bandwidth for the common two-output contract.

    The numerator counts one read each of x, residual and weight, plus one
    write each of normalized and residual_out. It deliberately does not model
    extra intermediate traffic in an unfused provider and is not an Nsight
    measurement of physical DRAM traffic.
    """

    logical_elements = 4 * x.numel() + weight.numel()
    logical_bytes = logical_elements * x.element_size()
    seconds = latency_ms / 1000.0
    return logical_bytes / seconds / 1e9


def provider_order_for_case(
    providers: list[str],
    run_index: int,
    case_index: int,
) -> list[str]:
    """Rotate provider order across cases and repeated benchmark runs."""

    offset = (run_index - 1 + case_index) % len(providers)
    return providers[offset:] + providers[:offset]


def output_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    difference = (
        actual.float() - expected.float()
    ).abs()
    return difference.max().item(), difference.mean().item()


def benchmark_case(
    rows: int,
    hidden_size: int,
    dtype_name: str,
    providers: list[str],
    run_index: int,
    case_index: int,
    warmup_ms: int,
    repetition_ms: int,
    eps: float,
) -> list[dict[str, object]]:
    """Validate and benchmark every provider for one shape and dtype."""

    dtype = DTYPE_MAP[dtype_name]
    device = torch.device("cuda")

    torch.manual_seed(2026 + rows + hidden_size)

    x = torch.randn(
        rows,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    residual = torch.randn_like(x)
    weight = torch.randn(
        hidden_size,
        device=device,
        dtype=dtype,
    )

    reference_normalized, reference_residual = (
        fused_residual_rms_norm_reference(
            x,
            residual,
            weight,
            eps,
        )
    )
    synchronize()

    rtol, atol = dtype_tolerances(dtype)
    ordered_providers = provider_order_for_case(
        providers,
        run_index,
        case_index,
    )
    order_text = "|".join(ordered_providers)

    print()
    print(
        f"Case: rows={rows}, hidden={hidden_size}, "
        f"dtype={dtype_name}"
    )
    print(f"  Provider order: {order_text}")

    records: list[dict[str, object]] = []

    for provider_position, provider in enumerate(
        ordered_providers,
        start=1,
    ):
        print(f"  Benchmarking {provider:<16}", end="", flush=True)

        record: dict[str, object] = {
            "run_index": run_index,
            "case_index": case_index,
            "provider_position": provider_position,
            "case_provider_order": order_text,
            "rows": rows,
            "hidden_size": hidden_size,
            "numel": x.numel(),
            "dtype": dtype_name,
            "provider": provider,
            "status": "ok",
            "error": "",
        }

        try:
            function = make_provider(
                provider,
                x,
                residual,
                weight,
                eps,
            )

            outputs, first_call_ms = measure_first_call(function)
            normalized, residual_out = outputs

            torch.testing.assert_close(
                normalized,
                reference_normalized,
                rtol=rtol,
                atol=atol,
            )
            torch.testing.assert_close(
                residual_out,
                reference_residual,
                rtol=0.0,
                atol=0.0,
            )

            normalized_max_error, normalized_mean_error = output_errors(
                normalized,
                reference_normalized,
            )
            residual_max_error, residual_mean_error = output_errors(
                residual_out,
                reference_residual,
            )

            p50_ms, p95_ms = benchmark_function(
                function,
                warmup_ms,
                repetition_ms,
            )
            logical_gbps = calculate_logical_bandwidth_gbps(
                x,
                weight,
                p50_ms,
            )

            record.update(
                {
                    "first_call_ms": first_call_ms,
                    "p50_ms": p50_ms,
                    "p95_ms": p95_ms,
                    "logical_effective_gbps": logical_gbps,
                    "normalized_max_abs_error": normalized_max_error,
                    "normalized_mean_abs_error": normalized_mean_error,
                    "residual_max_abs_error": residual_max_error,
                    "residual_mean_abs_error": residual_mean_error,
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
                    "logical_effective_gbps": float("nan"),
                    "normalized_max_abs_error": float("nan"),
                    "normalized_mean_abs_error": float("nan"),
                    "residual_max_abs_error": float("nan"),
                    "residual_mean_abs_error": float("nan"),
                }
            )
            print(f" FAILED | {exc!r}")

        records.append(record)

    del reference_normalized
    del reference_residual
    del residual
    del weight
    del x

    torch.cuda.empty_cache()
    return records


def add_speedup_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add speedup ratios against the three primary baselines."""

    dataframe = dataframe.copy()

    baselines = {
        "torch_native": "speedup_vs_torch_native",
        "torch_compile": "speedup_vs_torch_compile",
        "triton_unfused": "speedup_vs_triton_unfused",
    }

    for column in baselines.values():
        dataframe[column] = float("nan")

    group_columns = [
        "rows",
        "hidden_size",
        "dtype",
    ]

    for _, group in dataframe.groupby(group_columns):
        valid = group[group["status"] == "ok"]

        baseline_latencies: dict[str, float] = {}
        for provider in baselines:
            matches = valid[valid["provider"] == provider]
            if not matches.empty:
                baseline_latencies[provider] = float(
                    matches.iloc[0]["p50_ms"]
                )

        for index in group.index:
            latency = dataframe.at[index, "p50_ms"]

            if pd.isna(latency) or float(latency) <= 0:
                continue

            for provider, output_column in baselines.items():
                baseline_latency = baseline_latencies.get(provider)
                if baseline_latency is not None:
                    dataframe.at[index, output_column] = (
                        baseline_latency / float(latency)
                    )

    return dataframe


def print_summary(dataframe: pd.DataFrame) -> None:
    columns = [
        "rows",
        "hidden_size",
        "dtype",
        "provider",
        "p50_ms",
        "p95_ms",
        "speedup_vs_torch_native",
        "speedup_vs_torch_compile",
        "speedup_vs_triton_unfused",
        "logical_effective_gbps",
        "status",
    ]

    print()
    print("=" * 150)
    print("Fused Residual + RMSNorm Benchmark Summary")
    print("=" * 150)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        print(dataframe[columns].to_string(index=False))

    print("=" * 150)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark eager, native, compiled, unfused Triton and fused "
            "Triton Residual + RMSNorm implementations."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the required FP16 development-validation cases.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=list(DTYPE_MAP),
        default=list(DTYPE_MAP),
        help="Data types used by the full benchmark.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDER_ORDER,
        default=PROVIDER_ORDER,
        help="Providers to benchmark.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=1,
        help=(
            "Positive repeated-run index used to rotate provider order."
        ),
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
        "--output",
        type=Path,
        default=None,
        help="Optional independent fused-result CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if args.run_index <= 0:
        raise ValueError("--run-index must be positive.")

    if args.warmup_ms <= 0:
        raise ValueError("--warmup-ms must be positive.")

    if args.rep_ms <= 0:
        raise ValueError("--rep-ms must be positive.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        args.output
        if args.output is not None
        else PROJECT_ROOT
        / "results"
        / "csv"
        / (
            "fused_residual_rmsnorm_benchmark_"
            f"{timestamp}.csv"
        )
    )
    environment_path = (
        PROJECT_ROOT
        / "results"
        / "environment"
        / (
            "fused_residual_rmsnorm_environment_"
            f"{timestamp}.txt"
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_environment_report(environment_path, args.run_index)

    cases = QUICK_CASES if args.quick else FULL_CASES
    dtype_names = ["fp16"] if args.quick else args.dtypes

    print("=" * 88)
    print("LLM-KernelLab Fused Residual + RMSNorm Benchmark")
    print("=" * 88)
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"Triton:          {triton.__version__}")
    print(f"Git commit:      {get_git_commit()}")
    print(f"Mode:            {'quick' if args.quick else 'full'}")
    print(f"Run index:       {args.run_index}")
    print(f"Providers:       {', '.join(args.providers)}")
    print(f"Dtypes:          {', '.join(dtype_names)}")
    print(f"Warmup:          {args.warmup_ms} ms")
    print(f"Repetition:      {args.rep_ms} ms")
    print("=" * 88)

    all_records: list[dict[str, object]] = []
    case_index = 0

    for dtype_name in dtype_names:
        for rows, hidden_size in cases:
            records = benchmark_case(
                rows=rows,
                hidden_size=hidden_size,
                dtype_name=dtype_name,
                providers=args.providers,
                run_index=args.run_index,
                case_index=case_index,
                warmup_ms=args.warmup_ms,
                repetition_ms=args.rep_ms,
                eps=args.eps,
            )
            all_records.extend(records)
            case_index += 1

    dataframe = pd.DataFrame(all_records)
    dataframe.insert(
        0,
        "timestamp",
        datetime.now().isoformat(timespec="seconds"),
    )
    dataframe.insert(1, "git_commit", get_git_commit())
    dataframe.insert(
        2,
        "gpu",
        torch.cuda.get_device_name(0),
    )
    dataframe.insert(
        3,
        "torch_version",
        torch.__version__,
    )
    dataframe.insert(
        4,
        "torch_cuda_version",
        str(torch.version.cuda),
    )
    dataframe.insert(
        5,
        "triton_version",
        triton.__version__,
    )

    dataframe = add_speedup_columns(dataframe)
    dataframe.to_csv(
        output_path,
        index=False,
        float_format="%.9f",
    )

    print_summary(dataframe)

    print()
    print(f"CSV saved to:         {output_path}")
    print(f"Environment saved to: {environment_path}")
    print(
        "Logical effective bandwidth is a minimum traffic metric, "
        "not an Nsight DRAM measurement."
    )

    failed_rows = dataframe[dataframe["status"] != "ok"]
    if not failed_rows.empty:
        print()
        print("One or more benchmark providers failed:")
        print(
            failed_rows[
                [
                    "rows",
                    "hidden_size",
                    "dtype",
                    "provider",
                    "error",
                ]
            ].to_string(index=False)
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
