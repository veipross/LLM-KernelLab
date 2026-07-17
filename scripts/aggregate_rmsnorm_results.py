"""Aggregate repeated RMSNorm benchmarks and generate figures/report."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DTYPE_ORDER = ["fp16", "bf16", "fp32"]

PROVIDER_ORDER = [
    "torch_eager",
    "torch_native",
    "torch_compile",
    "triton",
]

PROVIDER_LABELS = {
    "torch_eager": "PyTorch Eager",
    "torch_native": "PyTorch Native",
    "torch_compile": "torch.compile",
    "triton": "Triton",
}

CASE_ORDER = [
    (1, 4096),
    (128, 4096),
    (2048, 4096),
    (512, 5120),
    (256, 8192),
]

FILE_PATTERN = re.compile(
    r"^rmsnorm_(fp16|bf16|fp32)_run(\d+)\.csv$"
)

REQUIRED_COLUMNS = {
    "git_commit",
    "gpu",
    "torch_version",
    "torch_cuda_version",
    "triton_version",
    "rows",
    "hidden_size",
    "dtype",
    "provider",
    "status",
    "p50_ms",
    "p95_ms",
    "effective_gbps_min",
    "max_abs_error",
    "mean_abs_error",
    "max_relative_error",
}


def discover_result_files(input_dir: Path) -> list[tuple[Path, str, int]]:
    """Find and validate repeated benchmark result files."""

    discovered: list[tuple[Path, str, int]] = []

    for path in sorted(input_dir.glob("rmsnorm_*_run*.csv")):
        match = FILE_PATTERN.match(path.name)

        if match is None:
            continue

        dtype_name = match.group(1)
        run_number = int(match.group(2))

        discovered.append((path, dtype_name, run_number))

    expected = {
        (dtype_name, run_number)
        for dtype_name in DTYPE_ORDER
        for run_number in (1, 2, 3)
    }

    actual = {
        (dtype_name, run_number)
        for _, dtype_name, run_number in discovered
    }

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    if missing:
        raise FileNotFoundError(
            f"Missing benchmark runs: {missing}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected benchmark runs: {unexpected}"
        )

    return discovered


def load_all_runs(
    files: list[tuple[Path, str, int]],
) -> pd.DataFrame:
    """Load all benchmark CSV files into one DataFrame."""

    frames: list[pd.DataFrame] = []

    for path, expected_dtype, run_number in files:
        dataframe = pd.read_csv(path)

        missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if len(dataframe) != 20:
            raise ValueError(
                f"{path.name} should contain 20 rows, "
                f"but contains {len(dataframe)}."
            )

        actual_dtypes = set(dataframe["dtype"].astype(str))
        if actual_dtypes != {expected_dtype}:
            raise ValueError(
                f"{path.name} contains unexpected dtypes: "
                f"{sorted(actual_dtypes)}"
            )

        failed = dataframe[
            dataframe["status"].astype(str) != "ok"
        ]

        if not failed.empty:
            raise RuntimeError(
                f"{path.name} contains failed benchmark rows."
            )

        dataframe = dataframe.copy()
        dataframe["run"] = run_number
        dataframe["source_file"] = path.name

        frames.append(dataframe)

    all_runs = pd.concat(frames, ignore_index=True)

    expected_rows = 9 * 20
    if len(all_runs) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} total records, "
            f"but got {len(all_runs)}."
        )

    return all_runs


def aggregate_runs(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate three runs using median and stability statistics."""

    group_columns = [
        "dtype",
        "rows",
        "hidden_size",
        "provider",
    ]

    aggregate = (
        all_runs.groupby(group_columns, as_index=False)
        .agg(
            run_count=("run", "nunique"),
            p50_median_ms=("p50_ms", "median"),
            p50_mean_ms=("p50_ms", "mean"),
            p50_min_ms=("p50_ms", "min"),
            p50_max_ms=("p50_ms", "max"),
            p50_std_ms=("p50_ms", "std"),
            p95_median_ms=("p95_ms", "median"),
            effective_gbps_median=(
                "effective_gbps_min",
                "median",
            ),
            max_abs_error_max=(
                "max_abs_error",
                "max",
            ),
            mean_abs_error_median=(
                "mean_abs_error",
                "median",
            ),
            max_relative_error_max=(
                "max_relative_error",
                "max",
            ),
        )
    )

    aggregate["p50_cv_percent"] = (
        aggregate["p50_std_ms"]
        / aggregate["p50_mean_ms"]
        * 100.0
    )

    aggregate["shape"] = (
        aggregate["rows"].astype(str)
        + "×"
        + aggregate["hidden_size"].astype(str)
    )

    key_columns = [
        "dtype",
        "rows",
        "hidden_size",
    ]

    latency_table = aggregate.pivot_table(
        index=key_columns,
        columns="provider",
        values="p50_median_ms",
        aggfunc="first",
    )

    for baseline, output_column in [
        ("torch_eager", "speedup_vs_eager"),
        ("torch_native", "speedup_vs_native"),
        ("torch_compile", "speedup_vs_compile"),
    ]:
        baseline_map = latency_table[baseline].to_dict()

        aggregate[output_column] = [
            baseline_map[
                (
                    row.dtype,
                    row.rows,
                    row.hidden_size,
                )
            ]
            / row.p50_median_ms
            for row in aggregate.itertuples()
        ]

    dtype_rank = {
        dtype_name: index
        for index, dtype_name in enumerate(DTYPE_ORDER)
    }

    case_rank = {
        case: index
        for index, case in enumerate(CASE_ORDER)
    }

    provider_rank = {
        provider: index
        for index, provider in enumerate(PROVIDER_ORDER)
    }

    aggregate["_dtype_rank"] = aggregate["dtype"].map(dtype_rank)
    aggregate["_case_rank"] = [
        case_rank[(row.rows, row.hidden_size)]
        for row in aggregate.itertuples()
    ]
    aggregate["_provider_rank"] = aggregate["provider"].map(
        provider_rank
    )

    aggregate = aggregate.sort_values(
        [
            "_dtype_rank",
            "_case_rank",
            "_provider_rank",
        ]
    ).drop(
        columns=[
            "_dtype_rank",
            "_case_rank",
            "_provider_rank",
        ]
    )

    return aggregate.reset_index(drop=True)


def create_latency_figures(
    aggregate: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Create one latency comparison chart per dtype."""

    figure_dir.mkdir(parents=True, exist_ok=True)

    shape_order = [
        f"{rows}×{hidden_size}"
        for rows, hidden_size in CASE_ORDER
    ]

    for dtype_name in DTYPE_ORDER:
        subset = aggregate[
            aggregate["dtype"] == dtype_name
        ]

        pivot = subset.pivot(
            index="shape",
            columns="provider",
            values="p50_median_ms",
        ).reindex(shape_order)

        pivot = pivot[PROVIDER_ORDER].rename(
            columns=PROVIDER_LABELS
        )

        axis = pivot.plot(
            kind="bar",
            figsize=(11, 6),
            width=0.82,
            logy=True,
        )

        axis.set_title(
            f"RMSNorm Median P50 Latency — {dtype_name.upper()}"
        )
        axis.set_xlabel("Rows × Hidden Size")
        axis.set_ylabel("P50 latency (ms, log scale)")
        axis.grid(axis="y", linestyle="--", alpha=0.4)
        axis.legend(title="Provider")
        axis.tick_params(axis="x", rotation=0)

        figure = axis.get_figure()
        figure.tight_layout()
        figure.savefig(
            figure_dir
            / f"rmsnorm_latency_{dtype_name}.png",
            dpi=180,
        )
        plt.close(figure)


def create_speedup_figure(
    aggregate: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Create Triton speedup versus PyTorch Native chart."""

    triton_rows = aggregate[
        (aggregate["provider"] == "triton")
        & (aggregate["rows"] > 1)
    ]

    shape_order = [
        f"{rows}×{hidden_size}"
        for rows, hidden_size in CASE_ORDER
        if rows > 1
    ]

    pivot = triton_rows.pivot(
        index="shape",
        columns="dtype",
        values="speedup_vs_native",
    ).reindex(shape_order)

    pivot = pivot[DTYPE_ORDER].rename(
        columns={
            "fp16": "FP16",
            "bf16": "BF16",
            "fp32": "FP32",
        }
    )

    axis = pivot.plot(
        kind="bar",
        figsize=(11, 6),
        width=0.82,
    )

    axis.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1.0,
    )
    axis.set_title(
        "Triton RMSNorm Speedup vs PyTorch Native\n"
        "(Median of 3 Runs on RTX 4090)"
    )
    axis.set_xlabel("Rows × Hidden Size")
    axis.set_ylabel("Speedup (×)")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend(title="Data type")
    axis.tick_params(axis="x", rotation=0)

    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(
        figure_dir / "rmsnorm_triton_speedup.png",
        dpi=180,
    )
    plt.close(figure)


def create_stability_figure(
    aggregate: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Create Triton P50 coefficient-of-variation chart."""

    triton_rows = aggregate[
        aggregate["provider"] == "triton"
    ]

    shape_order = [
        f"{rows}×{hidden_size}"
        for rows, hidden_size in CASE_ORDER
    ]

    pivot = triton_rows.pivot(
        index="shape",
        columns="dtype",
        values="p50_cv_percent",
    ).reindex(shape_order)

    pivot = pivot[DTYPE_ORDER].rename(
        columns={
            "fp16": "FP16",
            "bf16": "BF16",
            "fp32": "FP32",
        }
    )

    axis = pivot.plot(
        kind="bar",
        figsize=(11, 6),
        width=0.82,
    )

    axis.set_title(
        "Triton RMSNorm P50 Stability Across Three Runs"
    )
    axis.set_xlabel("Rows × Hidden Size")
    axis.set_ylabel("Coefficient of variation (%)")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend(title="Data type")
    axis.tick_params(axis="x", rotation=0)

    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(
        figure_dir / "rmsnorm_triton_stability.png",
        dpi=180,
    )
    plt.close(figure)


def write_markdown_report(
    aggregate: pd.DataFrame,
    all_runs: pd.DataFrame,
    report_path: Path,
) -> None:
    """Write a GitHub-ready RMSNorm benchmark report."""

    report_path.parent.mkdir(parents=True, exist_ok=True)

    triton_rows = aggregate[
        aggregate["provider"] == "triton"
    ].copy()

    main_rows = triton_rows[
        triton_rows["rows"] > 1
    ]

    summary_records: list[dict[str, object]] = []

    for dtype_name in DTYPE_ORDER:
        subset = main_rows[
            main_rows["dtype"] == dtype_name
        ]

        best_row = subset.loc[
            subset["speedup_vs_native"].idxmax()
        ]

        summary_records.append(
            {
                "Data type": dtype_name.upper(),
                "Median speedup vs Native": (
                    subset["speedup_vs_native"].median()
                ),
                "Best speedup vs Native": (
                    best_row["speedup_vs_native"]
                ),
                "Best shape": best_row["shape"],
                "Max Triton P50 CV (%)": (
                    subset["p50_cv_percent"].max()
                ),
            }
        )

    summary_table = pd.DataFrame(summary_records)

    detail_table = triton_rows[
        [
            "dtype",
            "shape",
            "p50_median_ms",
            "p95_median_ms",
            "speedup_vs_eager",
            "speedup_vs_native",
            "speedup_vs_compile",
            "effective_gbps_median",
            "p50_cv_percent",
            "max_abs_error_max",
        ]
    ].copy()

    detail_table.columns = [
        "Dtype",
        "Shape",
        "P50 median (ms)",
        "P95 median (ms)",
        "Speedup vs Eager",
        "Speedup vs Native",
        "Speedup vs compile",
        "Effective GB/s",
        "P50 CV (%)",
        "Max abs error",
    ]

    numeric_columns = [
        "P50 median (ms)",
        "P95 median (ms)",
        "Speedup vs Eager",
        "Speedup vs Native",
        "Speedup vs compile",
        "Effective GB/s",
        "P50 CV (%)",
        "Max abs error",
    ]

    detail_table[numeric_columns] = detail_table[
        numeric_columns
    ].round(6)

    summary_table[
        [
            "Median speedup vs Native",
            "Best speedup vs Native",
            "Max Triton P50 CV (%)",
        ]
    ] = summary_table[
        [
            "Median speedup vs Native",
            "Best speedup vs Native",
            "Max Triton P50 CV (%)",
        ]
    ].round(4)

    first = all_runs.iloc[0]

    markdown = f"""# RMSNorm Benchmark Report

## Environment

- GPU: {first["gpu"]}
- PyTorch: {first["torch_version"]}
- PyTorch CUDA: {first["torch_cuda_version"]}
- Triton: {first["triton_version"]}
- Git commit: {first["git_commit"]}
- Independent runs: 3 per data type
- Measurement: 100 ms warmup, 300 ms repeated benchmark

## Method

The benchmark compares PyTorch Eager, PyTorch Native,
`torch.compile`, and a custom Triton RMSNorm forward kernel.

P50 and P95 values below are the medians of three independent
benchmark runs. Correctness is checked against the FP32-accumulated
PyTorch reference implementation before timing.

## Summary

{summary_table.to_markdown(index=False)}

The one-row `1×4096` case is retained for launch-overhead analysis,
but excluded from the main speedup summary and primary speedup figure.

## Triton Detailed Results

{detail_table.to_markdown(index=False)}

## Figures

### FP16 latency

![FP16 latency](../results/figures/rmsnorm_latency_fp16.png)

### BF16 latency

![BF16 latency](../results/figures/rmsnorm_latency_bf16.png)

### FP32 latency

![FP32 latency](../results/figures/rmsnorm_latency_fp32.png)

### Triton speedup

![Triton speedup](../results/figures/rmsnorm_triton_speedup.png)

### Stability

![Triton stability](../results/figures/rmsnorm_triton_stability.png)

## Notes

- `effective_gbps_median` is a logical effective-bandwidth estimate
  based on minimum input, output, and weight traffic.
- It is not a direct Nsight Compute measurement of physical DRAM traffic.
- FP16 and BF16 reductions use FP32 accumulation.
- Small kernels around several microseconds are influenced strongly
  by launch overhead and timer resolution.
"""

    report_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    input_dir = PROJECT_ROOT / "results" / "csv"
    figure_dir = PROJECT_ROOT / "results" / "figures"

    combined_path = (
        input_dir / "rmsnorm_benchmark_all_runs.csv"
    )
    aggregate_path = (
        input_dir / "rmsnorm_benchmark_aggregate.csv"
    )
    report_path = (
        PROJECT_ROOT
        / "docs"
        / "rmsnorm_benchmark_report.md"
    )

    files = discover_result_files(input_dir)
    all_runs = load_all_runs(files)
    aggregate = aggregate_runs(all_runs)

    all_runs.to_csv(
        combined_path,
        index=False,
        float_format="%.9f",
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
        float_format="%.9f",
    )

    create_latency_figures(aggregate, figure_dir)
    create_speedup_figure(aggregate, figure_dir)
    create_stability_figure(aggregate, figure_dir)

    write_markdown_report(
        aggregate,
        all_runs,
        report_path,
    )

    triton_summary = aggregate[
        aggregate["provider"] == "triton"
    ][
        [
            "dtype",
            "shape",
            "p50_median_ms",
            "p95_median_ms",
            "speedup_vs_native",
            "speedup_vs_compile",
            "p50_cv_percent",
            "max_abs_error_max",
        ]
    ]

    print("=" * 110)
    print("Aggregated Triton RMSNorm Results")
    print("=" * 110)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.width",
        160,
        "display.float_format",
        lambda value: f"{value:.6f}",
    ):
        print(triton_summary.to_string(index=False))

    print("=" * 110)
    print(f"Loaded files:       {len(files)}")
    print(f"Raw records:        {len(all_runs)}")
    print(f"Aggregate records:  {len(aggregate)}")
    print(f"Combined CSV:       {combined_path}")
    print(f"Aggregate CSV:      {aggregate_path}")
    print(f"Figures directory: {figure_dir}")
    print(f"Markdown report:    {report_path}")


if __name__ == "__main__":
    main()
