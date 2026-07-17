"""Benchmark PyTorch and Triton RMSNorm forward implementations."""

from __future__ import annotations

import argparse
import platform
import shutil
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

from llm_kernels.torch_ops import rms_norm_reference
from llm_kernels.triton_ops import rms_norm_triton


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

PROVIDER_ORDER = [
    "torch_eager",
    "torch_native",
    "torch_compile",
    "triton",
]

FULL_CASES = [
    (1, 4096),
    (128, 4096),
    (2048, 4096),
    (512, 5120),
    (256, 8192),
]

QUICK_CASES = [
    (1, 4096),
    (128, 4096),
    (2048, 4096),
]


def torch_rms_norm_body(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    RMSNorm tensor operations used by both Eager and torch.compile.

    Input checks are intentionally excluded so that both providers execute
    the same mathematical workload.
    """

    input_dtype = x.dtype

    x_fp32 = x.float()
    weight_fp32 = weight.float()

    mean_square = x_fp32.square().mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(mean_square + eps)

    output = x_fp32 * inverse_rms * weight_fp32
    return output.to(dtype=input_dtype)


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


def save_environment_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    properties = torch.cuda.get_device_properties(0)

    lines = [
        "LLM-KernelLab RMSNorm Benchmark Environment",
        "=" * 60,
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"git_commit: {get_git_commit()}",
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
    weight: torch.Tensor,
    eps: float,
) -> Callable[[], torch.Tensor]:
    hidden_size = x.shape[-1]

    if provider == "torch_eager":
        return lambda: torch_rms_norm_body(x, weight, eps)

    if provider == "torch_native":
        return lambda: F.rms_norm(
            x,
            normalized_shape=(hidden_size,),
            weight=weight,
            eps=eps,
        )

    if provider == "torch_compile":
        # Each benchmark case gets a fresh compiled graph. This avoids
        # shape-guard cache limits when testing many hidden sizes.
        torch._dynamo.reset()

        compiled_function = torch.compile(
            torch_rms_norm_body,
            fullgraph=True,
            dynamic=False,
        )

        return lambda: compiled_function(x, weight, eps)

    if provider == "triton":
        return lambda: rms_norm_triton(x, weight, eps)

    raise ValueError(f"Unknown provider: {provider}")


def measure_first_call(
    function: Callable[[], torch.Tensor],
) -> tuple[torch.Tensor, float]:
    """
    Measure wall-clock first-call latency.

    For torch.compile and Triton this includes initial compilation.
    Steady-state latency is measured separately.
    """

    synchronize()
    start_time = time.perf_counter()

    output = function()

    synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return output, elapsed_ms


def benchmark_function(
    function: Callable[[], torch.Tensor],
    warmup_ms: int,
    repetition_ms: int,
) -> tuple[float, float, float, float]:
    timings = triton.testing.do_bench(
        function,
        warmup=warmup_ms,
        rep=repetition_ms,
        quantiles=[0.20, 0.50, 0.80, 0.95],
    )

    p20_ms, p50_ms, p80_ms, p95_ms = (
        float(value) for value in timings
    )

    return p20_ms, p50_ms, p80_ms, p95_ms


def calculate_effective_bandwidth_gbps(
    x: torch.Tensor,
    weight: torch.Tensor,
    latency_ms: float,
) -> float:
    """
    Calculate minimum logical memory bandwidth.

    The numerator includes:
      - one read of x;
      - one write of output;
      - one logical read of the shared weight vector.

    This is an effective comparison metric rather than exact DRAM traffic.
    """

    logical_elements = (
        2 * x.numel()
        + weight.numel()
    )

    logical_bytes = logical_elements * x.element_size()
    seconds = latency_ms / 1000.0

    return logical_bytes / seconds / 1e9


def benchmark_case(
    rows: int,
    hidden_size: int,
    dtype_name: str,
    providers: list[str],
    warmup_ms: int,
    repetition_ms: int,
    eps: float,
) -> list[dict[str, object]]:
    dtype = DTYPE_MAP[dtype_name]
    device = torch.device("cuda")

    seed = 2026 + rows + hidden_size
    torch.manual_seed(seed)

    x = torch.randn(
        rows,
        hidden_size,
        device=device,
        dtype=dtype,
    )

    weight = torch.randn(
        hidden_size,
        device=device,
        dtype=dtype,
    )

    reference = rms_norm_reference(
        x,
        weight,
        eps,
    )
    synchronize()

    rtol, atol = dtype_tolerances(dtype)
    records: list[dict[str, object]] = []

    print()
    print(
        f"Case: rows={rows}, hidden={hidden_size}, "
        f"dtype={dtype_name}"
    )

    for provider in providers:
        print(f"  Benchmarking {provider:<14}", end="", flush=True)

        record: dict[str, object] = {
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
                weight,
                eps,
            )

            output, first_call_ms = measure_first_call(function)

            difference = (
                output.float() - reference.float()
            ).abs()

            max_abs_error = difference.max().item()
            mean_abs_error = difference.mean().item()

            denominator = reference.float().abs().clamp_min(1e-6)
            max_relative_error = (
                difference / denominator
            ).max().item()

            torch.testing.assert_close(
                output,
                reference,
                rtol=rtol,
                atol=atol,
            )

            p20_ms, p50_ms, p80_ms, p95_ms = benchmark_function(
                function,
                warmup_ms,
                repetition_ms,
            )

            effective_gbps = calculate_effective_bandwidth_gbps(
                x,
                weight,
                p50_ms,
            )

            record.update(
                {
                    "first_call_ms": first_call_ms,
                    "p20_ms": p20_ms,
                    "p50_ms": p50_ms,
                    "p80_ms": p80_ms,
                    "p95_ms": p95_ms,
                    "effective_gbps_min": effective_gbps,
                    "max_abs_error": max_abs_error,
                    "mean_abs_error": mean_abs_error,
                    "max_relative_error": max_relative_error,
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
                    "p20_ms": float("nan"),
                    "p50_ms": float("nan"),
                    "p80_ms": float("nan"),
                    "p95_ms": float("nan"),
                    "effective_gbps_min": float("nan"),
                    "max_abs_error": float("nan"),
                    "mean_abs_error": float("nan"),
                    "max_relative_error": float("nan"),
                }
            )

            print(f" FAILED | {exc!r}")

        records.append(record)

    del reference
    del weight
    del x

    torch.cuda.empty_cache()
    return records


def add_speedup_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["speedup_vs_eager"] = float("nan")
    dataframe["speedup_vs_native"] = float("nan")

    group_columns = [
        "rows",
        "hidden_size",
        "dtype",
    ]

    for _, group in dataframe.groupby(group_columns):
        valid = group[group["status"] == "ok"]

        eager_rows = valid[
            valid["provider"] == "torch_eager"
        ]
        native_rows = valid[
            valid["provider"] == "torch_native"
        ]

        eager_latency = (
            float(eager_rows.iloc[0]["p50_ms"])
            if not eager_rows.empty
            else None
        )

        native_latency = (
            float(native_rows.iloc[0]["p50_ms"])
            if not native_rows.empty
            else None
        )

        for index in group.index:
            latency = dataframe.at[index, "p50_ms"]

            if pd.isna(latency) or float(latency) <= 0:
                continue

            if eager_latency is not None:
                dataframe.at[index, "speedup_vs_eager"] = (
                    eager_latency / float(latency)
                )

            if native_latency is not None:
                dataframe.at[index, "speedup_vs_native"] = (
                    native_latency / float(latency)
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
        "speedup_vs_eager",
        "speedup_vs_native",
        "effective_gbps_min",
        "max_abs_error",
        "status",
    ]

    print()
    print("=" * 120)
    print("RMSNorm Benchmark Summary")
    print("=" * 120)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        print(dataframe[columns].to_string(index=False))

    print("=" * 120)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PyTorch Eager, PyTorch native, "
            "torch.compile and Triton RMSNorm."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run three FP16 cases for an initial validation."
        ),
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
        help="Implementations to benchmark.",
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
        help="Optional output CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

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
        / f"rmsnorm_benchmark_{timestamp}.csv"
    )

    environment_path = (
        PROJECT_ROOT
        / "results"
        / "environment"
        / f"rmsnorm_environment_{timestamp}.txt"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.parent.mkdir(parents=True, exist_ok=True)

    save_environment_report(environment_path)

    cases = QUICK_CASES if args.quick else FULL_CASES
    dtype_names = ["fp16"] if args.quick else args.dtypes

    print("=" * 80)
    print("LLM-KernelLab RMSNorm Benchmark")
    print("=" * 80)
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"Triton:          {triton.__version__}")
    print(f"Git commit:      {get_git_commit()}")
    print(f"Mode:            {'quick' if args.quick else 'full'}")
    print(f"Providers:       {', '.join(args.providers)}")
    print(f"Dtypes:          {', '.join(dtype_names)}")
    print(f"Warmup:          {args.warmup_ms} ms")
    print(f"Repetition:      {args.rep_ms} ms")
    print("=" * 80)

    all_records: list[dict[str, object]] = []

    for dtype_name in dtype_names:
        for rows, hidden_size in cases:
            records = benchmark_case(
                rows=rows,
                hidden_size=hidden_size,
                dtype_name=dtype_name,
                providers=args.providers,
                warmup_ms=args.warmup_ms,
                repetition_ms=args.rep_ms,
                eps=args.eps,
            )

            all_records.extend(records)

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

    latest_csv = output_path.parent / "rmsnorm_benchmark_latest.csv"
    shutil.copyfile(output_path, latest_csv)

    latest_environment = (
        environment_path.parent
        / "rmsnorm_environment_latest.txt"
    )
    shutil.copyfile(environment_path, latest_environment)

    print_summary(dataframe)

    print()
    print(f"CSV saved to:         {output_path}")
    print(f"Latest CSV:           {latest_csv}")
    print(f"Environment saved to: {environment_path}")

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
