"""PyTorch Profiler：分析 Hugging Face RMSNorm 整模型收益不明显的原因。

默认场景：随机初始化紧凑 Llama、FP16、batch=1、seq=32、hidden=512。
不会下载预训练权重。
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
import transformers

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

MODEL_CLASS = {
    "llama": LlamaForCausalLM,
    "qwen2": Qwen2ForCausalLM,
}

CONFIG_CLASS = {
    "llama": LlamaConfig,
    "qwen2": Qwen2Config,
}

HF_RMSNORM_TYPES = (LlamaRMSNorm, Qwen2RMSNorm)


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


def safe_float(obj: object, names: tuple[str, ...]) -> float:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def is_cuda_event(event: object) -> bool:
    return "cuda" in str(getattr(event, "device_type", "")).lower()


def build_config(args: argparse.Namespace):
    cls = CONFIG_CLASS[args.family]
    return cls(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.key_value_heads,
        max_position_embeddings=max(128, args.sequence_length),
        rms_norm_eps=args.eps,
        attention_dropout=0.0,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )


def build_models(args: argparse.Namespace):
    dtype = DTYPE_MAP[args.dtype]
    config = build_config(args)
    cls = MODEL_CLASS[args.family]

    torch.manual_seed(args.seed)
    baseline = cls(config).eval()
    triton_model = cls(config).eval()
    triton_model.load_state_dict(baseline.state_dict())

    expected = sum(
        isinstance(module, HF_RMSNORM_TYPES)
        for module in triton_model.modules()
    )
    replaced = tuple(
        replace_huggingface_rmsnorm_modules(triton_model)
    )

    if expected <= 0 or len(replaced) != expected:
        raise RuntimeError(
            f"RMSNorm 替换失败：expected={expected}, actual={len(replaced)}"
        )

    baseline = baseline.cuda().to(dtype=dtype).eval()
    triton_model = triton_model.cuda().to(dtype=dtype).eval()
    return baseline, triton_model, replaced


def build_inputs(args: argparse.Namespace):
    torch.manual_seed(args.seed + 1)
    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.sequence_length),
        device="cuda",
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def make_forward(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    def forward() -> torch.Tensor:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=0,
        ).logits

    return forward


def validate_outputs(
    baseline_fn: Callable[[], torch.Tensor],
    triton_fn: Callable[[], torch.Tensor],
    dtype: torch.dtype,
) -> tuple[float, float]:
    with torch.inference_mode():
        baseline = baseline_fn()
        triton_output = triton_fn()
    synchronize()

    if dtype == torch.float16:
        rtol, atol = 1e-2, 1e-2
    elif dtype == torch.bfloat16:
        rtol, atol = 5e-2, 5e-2
    else:
        rtol, atol = 2e-4, 2e-4

    torch.testing.assert_close(
        triton_output,
        baseline,
        rtol=rtol,
        atol=atol,
    )
    diff = (triton_output.float() - baseline.float()).abs()
    return diff.max().item(), diff.mean().item()


def warmup(function: Callable[[], torch.Tensor], iterations: int) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            function()
    synchronize()


def measure_latency(
    function: Callable[[], torch.Tensor],
    warmup_iterations: int,
    measured_iterations: int,
) -> tuple[float, float]:
    warmup(function, warmup_iterations)

    synchronize()
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(measured_iterations):
            function()
    synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / measured_iterations

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    synchronize()
    start.record()
    with torch.inference_mode():
        for _ in range(measured_iterations):
            function()
    end.record()
    synchronize()
    event_ms = start.elapsed_time(end) / measured_iterations
    return wall_ms, event_ms


@contextmanager
def annotated_rmsnorm() -> Iterator[None]:
    llama_forward = LlamaRMSNorm.forward
    qwen_forward = Qwen2RMSNorm.forward
    adapter_forward = HuggingFaceTritonRMSNorm.forward

    def llama_wrapped(self, hidden_states):
        with record_function("rmsnorm::huggingface_llama"):
            return llama_forward(self, hidden_states)

    def qwen_wrapped(self, hidden_states):
        with record_function("rmsnorm::huggingface_qwen2"):
            return qwen_forward(self, hidden_states)

    def adapter_wrapped(self, hidden_states):
        with record_function("rmsnorm::llm_kernellab_triton"):
            return adapter_forward(self, hidden_states)

    LlamaRMSNorm.forward = llama_wrapped
    Qwen2RMSNorm.forward = qwen_wrapped
    HuggingFaceTritonRMSNorm.forward = adapter_wrapped
    try:
        yield
    finally:
        LlamaRMSNorm.forward = llama_forward
        Qwen2RMSNorm.forward = qwen_forward
        HuggingFaceTritonRMSNorm.forward = adapter_forward


def operator_rows(profiler: Any) -> list[dict[str, object]]:
    rows = []
    for event in profiler.key_averages():
        rows.append(
            {
                "operator": str(getattr(event, "key", "")),
                "count": int(getattr(event, "count", 0)),
                "self_cpu_time_total_us": safe_float(
                    event, ("self_cpu_time_total",)
                ),
                "cpu_time_total_us": safe_float(
                    event, ("cpu_time_total",)
                ),
                "self_device_time_total_us": safe_float(
                    event,
                    ("self_device_time_total", "self_cuda_time_total"),
                ),
                "device_time_total_us": safe_float(
                    event,
                    ("device_time_total", "cuda_time_total"),
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["self_device_time_total_us"]),
            float(row["self_cpu_time_total_us"]),
        ),
        reverse=True,
    )
    return rows


def is_profiler_annotation(name: str) -> bool:
    """判断一个设备侧事件是否只是 record_function 标记。"""

    return (
        name.startswith("model_iteration::")
        or name.startswith("rmsnorm::")
    )


def raw_cuda_kernel_events(profiler: Any) -> list[Any]:
    """返回真实 CUDA Kernel 事件，排除用户范围标记。"""

    return [
        event
        for event in profiler.events()
        if (
            is_cuda_event(event)
            and not is_profiler_annotation(
                str(getattr(event, "name", ""))
            )
        )
    ]


def kernel_rows(profiler: Any) -> list[dict[str, object]]:
    """聚合真实 CUDA Kernel，包括 PyTorch 与 Triton Kernel。

    使用原始 CUDA 事件可以捕获 Triton Kernel；同时显式排除
    ``record_function`` 生成的设备侧范围标记，避免把范围误当 Kernel。
    """

    aggregate = defaultdict(lambda: {"count": 0, "total_us": 0.0})

    for event in raw_cuda_kernel_events(profiler):
        name = str(getattr(event, "name", ""))
        duration = safe_float(
            event,
            (
                "self_device_time_total",
                "self_cuda_time_total",
                "device_time_total",
                "cuda_time_total",
            ),
        )
        aggregate[name]["count"] += 1
        aggregate[name]["total_us"] += duration

    rows = []
    for name, values in aggregate.items():
        count = int(values["count"])
        total_us = float(values["total_us"])
        rows.append(
            {
                "kernel": name,
                "count": count,
                "total_device_time_us": total_us,
                "average_device_time_us": total_us / count if count else 0.0,
            }
        )

    rows.sort(
        key=lambda row: float(row["total_device_time_us"]),
        reverse=True,
    )
    return rows


def iter_cpu_subtree(event: Any):
    """遍历一个 CPU FunctionEvent 及其全部 CPU 子事件。"""

    yield event

    for child in getattr(event, "cpu_children", ()):
        yield from iter_cpu_subtree(child)


def marker_metrics(
    profiler: Any,
    marker_names: set[str],
    provider: str,
) -> dict[str, float]:
    """统计 RMSNorm CPU 标记和真实 CUDA Kernel。

    PyTorch 原生算子的 Kernel 可从 CPU 标记子树关联得到；Triton
    Kernel 在当前 Profiler 版本中未关联到 Python 标记，因此按
    ``_rms_norm_forward_kernel`` 的原始 CUDA 事件单独统计。
    """

    marker_events = [
        event
        for event in profiler.events()
        if (
            str(getattr(event, "name", "")) in marker_names
            and "cpu" in str(
                getattr(event, "device_type", "")
            ).lower()
        )
    ]

    marker_cpu_us = sum(
        safe_float(event, ("cpu_time_total",))
        for event in marker_events
    )

    marker_kernel_count = 0
    marker_kernel_total_us = 0.0

    if provider == "triton_rmsnorm_model":
        triton_events = [
            event
            for event in raw_cuda_kernel_events(profiler)
            if "_rms_norm_forward_kernel" in str(
                getattr(event, "name", "")
            )
        ]
        marker_kernel_count = len(triton_events)
        marker_kernel_total_us = sum(
            safe_float(
                event,
                (
                    "self_device_time_total",
                    "self_cuda_time_total",
                    "device_time_total",
                    "cuda_time_total",
                ),
            )
            for event in triton_events
        )
    else:
        for marker_event in marker_events:
            for event in iter_cpu_subtree(marker_event):
                for kernel in getattr(event, "kernels", ()):
                    marker_kernel_count += 1
                    marker_kernel_total_us += float(
                        getattr(kernel, "duration", 0.0)
                    )

    return {
        "marker_count": float(len(marker_events)),
        "marker_cpu_us": marker_cpu_us,
        "marker_kernel_count": float(marker_kernel_count),
        "marker_kernel_total_us": marker_kernel_total_us,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def profile_provider(
    provider: str,
    function: Callable[[], torch.Tensor],
    args: argparse.Namespace,
    output_dir: Path,
    prefix: str,
) -> dict[str, object]:
    warmup(function, args.profiler_warmup_iterations)

    with annotated_rmsnorm():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            with torch.inference_mode():
                for _ in range(args.profile_iterations):
                    with record_function(f"model_iteration::{provider}"):
                        function()
                    profiler.step()

    synchronize()

    trace_path = output_dir / f"{prefix}_{provider}_trace.json"
    operator_path = output_dir / f"{prefix}_{provider}_operators.csv"
    kernel_path = output_dir / f"{prefix}_{provider}_cuda_kernels.csv"

    profiler.export_chrome_trace(str(trace_path))
    operators = operator_rows(profiler)
    kernels = kernel_rows(profiler)
    write_csv(operator_path, operators)
    write_csv(kernel_path, kernels)

    marker_names = {
        "huggingface_model": {
            "rmsnorm::huggingface_llama",
            "rmsnorm::huggingface_qwen2",
        },
        "triton_rmsnorm_model": {"rmsnorm::llm_kernellab_triton"},
    }
    kernel_count = sum(int(row["count"]) for row in kernels)
    kernel_total_us = sum(float(row["total_device_time_us"]) for row in kernels)

    rmsnorm_metrics = marker_metrics(
        profiler,
        marker_names[provider],
        provider,
    )
    marker_count = int(
        rmsnorm_metrics["marker_count"]
    )
    marker_cpu_us = float(
        rmsnorm_metrics["marker_cpu_us"]
    )
    marker_kernel_count = int(
        rmsnorm_metrics["marker_kernel_count"]
    )
    marker_device_us = float(
        rmsnorm_metrics["marker_kernel_total_us"]
    )

    print(f"\n{provider} Top CUDA Kernels")
    for row in kernels[:12]:
        print(
            f"  {row['total_device_time_us']:10.3f} us | "
            f"count={row['count']:4d} | {row['kernel']}"
        )

    return {
        "provider": provider,
        "profile_iterations": args.profile_iterations,
        "cuda_kernel_count": kernel_count,
        "cuda_kernels_per_iteration": kernel_count / args.profile_iterations,
        "summed_cuda_kernel_time_us": kernel_total_us,
        "summed_cuda_kernel_time_per_iteration_us": (
            kernel_total_us / args.profile_iterations
        ),
        "rmsnorm_marker_count": marker_count,
        "rmsnorm_markers_per_iteration": marker_count / args.profile_iterations,
        "rmsnorm_marker_cpu_per_iteration_us": (
            marker_cpu_us / args.profile_iterations
        ),
        "rmsnorm_marker_kernel_count": marker_kernel_count,
        "rmsnorm_marker_kernels_per_iteration": (
            marker_kernel_count / args.profile_iterations
        ),
        "rmsnorm_marker_device_per_iteration_us": (
            marker_device_us / args.profile_iterations
        ),
        "trace_path": str(trace_path),
        "operator_csv_path": str(operator_path),
        "cuda_kernel_csv_path": str(kernel_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("llama", "qwen2"), default="llama")
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP), default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1376)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--key-value-heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--latency-warmup-iterations", type=int, default=50)
    parser.add_argument("--latency-iterations", type=int, default=300)
    parser.add_argument("--profiler-warmup-iterations", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "profiler",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")
    positive = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "layers": args.layers,
        "attention_heads": args.attention_heads,
        "key_value_heads": args.key_value_heads,
        "vocab_size": args.vocab_size,
        "latency_iterations": args.latency_iterations,
        "profile_iterations": args.profile_iterations,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} 必须为正数。")
    if args.hidden_size % args.attention_heads != 0:
        raise ValueError("hidden_size 必须能被 attention_heads 整除。")
    if args.attention_heads % args.key_value_heads != 0:
        raise ValueError("attention_heads 必须能被 key_value_heads 整除。")


def main() -> None:
    args = parse_args()
    validate_args(args)
    dtype = DTYPE_MAP[args.dtype]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = (
        f"{args.family}_{args.dtype}_b{args.batch_size}_"
        f"s{args.sequence_length}_h{args.hidden_size}_{timestamp}"
    )
    output_dir = args.output_dir / prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("LLM-KernelLab Hugging Face RMSNorm 整模型性能归因")
    print("=" * 100)
    print(f"Git commit:       {get_git_commit()}")
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:          {torch.__version__}")
    print(f"CUDA:             {torch.version.cuda}")
    print(f"Transformers:     {transformers.__version__}")
    print(f"Family/Dtype:     {args.family}/{args.dtype}")
    print(f"Batch/Sequence:   {args.batch_size}/{args.sequence_length}")
    print(f"Hidden/Layers:    {args.hidden_size}/{args.layers}")
    print(f"Output directory: {output_dir}")
    print("=" * 100)

    baseline_model, triton_model, replaced = build_models(args)
    input_ids, attention_mask = build_inputs(args)
    baseline_fn = make_forward(baseline_model, input_ids, attention_mask)
    triton_fn = make_forward(triton_model, input_ids, attention_mask)

    max_error, mean_error = validate_outputs(
        baseline_fn, triton_fn, dtype
    )
    print(
        f"正确性通过：max_abs_error={max_error:.9f}, "
        f"mean_abs_error={mean_error:.9f}, replaced={len(replaced)}"
    )

    latency = {}
    for provider, function in (
        ("huggingface_model", baseline_fn),
        ("triton_rmsnorm_model", triton_fn),
    ):
        wall_ms, event_ms = measure_latency(
            function,
            args.latency_warmup_iterations,
            args.latency_iterations,
        )
        latency[provider] = {
            "wall_average_ms": wall_ms,
            "cuda_event_average_ms": event_ms,
        }
        print(
            f"{provider}: wall={wall_ms:.6f} ms, "
            f"cuda_event={event_ms:.6f} ms"
        )

    rows = []
    for provider, function in (
        ("huggingface_model", baseline_fn),
        ("triton_rmsnorm_model", triton_fn),
    ):
        row = profile_provider(
            provider,
            function,
            args,
            output_dir,
            prefix,
        )
        row.update(latency[provider])
        row.update(
            {
                "git_commit": get_git_commit(),
                "gpu": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "torch_cuda_version": str(torch.version.cuda),
                "transformers_version": transformers.__version__,
                "family": args.family,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "hidden_size": args.hidden_size,
                "layers": args.layers,
                "replaced_rmsnorm_count": len(replaced),
                "max_abs_error": max_error,
                "mean_abs_error": mean_error,
            }
        )
        rows.append(row)

    comparison_path = output_dir / f"{prefix}_comparison.csv"
    write_csv(comparison_path, rows)

    baseline_latency = latency["huggingface_model"]
    triton_latency = latency["triton_rmsnorm_model"]
    wall_speedup = (
        baseline_latency["wall_average_ms"]
        / triton_latency["wall_average_ms"]
    )
    event_speedup = (
        baseline_latency["cuda_event_average_ms"]
        / triton_latency["cuda_event_average_ms"]
    )

    print("\n" + "=" * 100)
    print("总体对比")
    print("=" * 100)
    print(f"墙钟平均延迟加速比：{wall_speedup:.6f}×")
    print(f"CUDA Event 平均延迟加速比：{event_speedup:.6f}×")
    for row in rows:
        print(f"\n{row['provider']}")
        print(
            "  CUDA Kernel 数/迭代："
            f"{row['cuda_kernels_per_iteration']:.2f}"
        )
        print(
            "  CUDA Kernel 累计时间/迭代："
            f"{row['summed_cuda_kernel_time_per_iteration_us']:.3f} us"
        )
        print(
            "  RMSNorm 标记数/迭代："
            f"{row['rmsnorm_markers_per_iteration']:.2f}"
        )
        print(
            "  RMSNorm 标记 CPU 时间/迭代："
            f"{row['rmsnorm_marker_cpu_per_iteration_us']:.3f} us"
        )
        print(
            "  RMSNorm 真实 CUDA Kernel 数/迭代："
            f"{row['rmsnorm_marker_kernels_per_iteration']:.2f}"
        )
        print(
            "  RMSNorm 真实 CUDA Kernel 时间/迭代："
            f"{row['rmsnorm_marker_device_per_iteration_us']:.3f} us"
        )

    print(f"\n对比 CSV：{comparison_path}")
    print("Chrome Trace 可用 chrome://tracing 或 Perfetto 打开。")
    print("Profiler 用于归因，正式性能结论仍以三轮 Benchmark 为准。")


if __name__ == "__main__":
    main()
