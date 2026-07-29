"""诊断 Hugging Face Llama/Qwen2 中 RMSNorm 输入的内存布局。

用途：
1. 构造随机初始化的紧凑 Llama/Qwen2 模型，不下载预训练权重；
2. 将全部 Hugging Face RMSNorm 替换为 LLM-KernelLab Triton 适配器；
3. 在完整前向过程中记录每个 RMSNorm 输入的 shape、stride 和连续性；
4. 判断适配器中的 ``hidden_states.contiguous()`` 是否可能产生额外复制。

运行：
    python scripts/diagnose_huggingface_rmsnorm_layout.py

输出：
    results/csv/huggingface_rmsnorm_input_layout_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from integrations import (
    HuggingFaceTritonRMSNorm,
    replace_huggingface_rmsnorm_modules,
)


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

CASES = (
    ("single_token_no_cache", 1, 1),
    ("prefill_s32", 1, 32),
    ("prefill_s128", 1, 128),
    ("prefill_b4_s32", 4, 32),
)


def build_config(
    family: str,
    *,
    hidden_size: int,
    intermediate_size: int,
    layers: int,
    attention_heads: int,
    key_value_heads: int,
    vocab_size: int,
    max_position_embeddings: int,
) -> LlamaConfig | Qwen2Config:
    common = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": layers,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": key_value_heads,
        "max_position_embeddings": max_position_embeddings,
        "rms_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "use_cache": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
    }

    if family == "llama":
        return LlamaConfig(**common)
    if family == "qwen2":
        return Qwen2Config(**common)
    raise ValueError(f"未知模型族：{family}")


def build_model(
    family: str,
    config: LlamaConfig | Qwen2Config,
) -> nn.Module:
    if family == "llama":
        return LlamaForCausalLM(config)
    if family == "qwen2":
        return Qwen2ForCausalLM(config)
    raise ValueError(f"未知模型族：{family}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="诊断整模型 RMSNorm 输入的连续性和 stride。"
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("llama", "qwen2"),
        default=("llama", "qwen2"),
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=tuple(DTYPE_MAP),
        default=("fp16", "bf16", "fp32"),
    )
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1376)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--key-value-heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "csv",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")

    positive_values = {
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "layers": args.layers,
        "attention_heads": args.attention_heads,
        "key_value_heads": args.key_value_heads,
        "vocab_size": args.vocab_size,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} 必须为正数。")

    if args.hidden_size % args.attention_heads != 0:
        raise ValueError("hidden_size 必须能被 attention_heads 整除。")

    if args.attention_heads % args.key_value_heads != 0:
        raise ValueError(
            "attention_heads 必须能被 key_value_heads 整除。"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir
        / f"huggingface_rmsnorm_input_layout_{timestamp}.csv"
    )

    records: list[dict[str, object]] = []

    print("=" * 96)
    print("LLM-KernelLab Hugging Face RMSNorm 输入布局诊断")
    print("=" * 96)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "模型配置："
        f"hidden={args.hidden_size}, "
        f"intermediate={args.intermediate_size}, "
        f"layers={args.layers}, "
        f"heads={args.attention_heads}, "
        f"kv_heads={args.key_value_heads}, "
        f"vocab={args.vocab_size}"
    )
    print("=" * 96)

    max_position_embeddings = max(
        128,
        max(sequence_length for _, _, sequence_length in CASES),
    )

    for dtype_name in args.dtypes:
        dtype = DTYPE_MAP[dtype_name]

        for family in args.families:
            torch.manual_seed(2026)
            torch.cuda.manual_seed_all(2026)

            config = build_config(
                family,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                layers=args.layers,
                attention_heads=args.attention_heads,
                key_value_heads=args.key_value_heads,
                vocab_size=args.vocab_size,
                max_position_embeddings=max_position_embeddings,
            )
            model = (
                build_model(family, config)
                .cuda()
                .to(dtype=dtype)
                .eval()
            )

            replaced_names = tuple(
                replace_huggingface_rmsnorm_modules(model)
            )
            if not replaced_names:
                raise RuntimeError(
                    f"{family} 模型中没有找到可替换的 RMSNorm。"
                )

            module_names = {
                module: name
                for name, module in model.named_modules()
                if isinstance(module, HuggingFaceTritonRMSNorm)
            }

            active_case = {"name": "", "batch": 0, "seq": 0}
            hook_handles = []

            def make_hook(module_name: str):
                def hook(
                    module: nn.Module,
                    inputs: tuple[torch.Tensor, ...],
                ) -> None:
                    del module
                    hidden_states = inputs[0]

                    records.append(
                        {
                            "family": family,
                            "dtype": dtype_name,
                            "case_name": active_case["name"],
                            "batch_size": active_case["batch"],
                            "sequence_length": active_case["seq"],
                            "module_name": module_name,
                            "shape": str(tuple(hidden_states.shape)),
                            "stride": str(tuple(hidden_states.stride())),
                            "is_contiguous": bool(
                                hidden_states.is_contiguous()
                            ),
                            "storage_offset": int(
                                hidden_states.storage_offset()
                            ),
                            "device": str(hidden_states.device),
                            "input_dtype": str(hidden_states.dtype),
                            "weight_is_contiguous": bool(
                                module_names_inverse[
                                    module_name
                                ].weight.is_contiguous()
                            ),
                        }
                    )

                return hook

            module_names_inverse = {
                name: module for module, name in module_names.items()
            }

            for module, module_name in module_names.items():
                hook_handles.append(
                    module.register_forward_pre_hook(
                        make_hook(module_name)
                    )
                )

            try:
                for case_name, batch_size, sequence_length in CASES:
                    active_case.update(
                        {
                            "name": case_name,
                            "batch": batch_size,
                            "seq": sequence_length,
                        }
                    )

                    input_ids = torch.randint(
                        low=0,
                        high=args.vocab_size,
                        size=(batch_size, sequence_length),
                        device="cuda",
                        dtype=torch.long,
                    )
                    attention_mask = torch.ones_like(input_ids)

                    before = len(records)

                    with torch.inference_mode():
                        model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )

                    torch.cuda.synchronize()
                    case_records = records[before:]

                    expected_calls = len(replaced_names)
                    actual_calls = len(case_records)

                    if actual_calls != expected_calls:
                        raise RuntimeError(
                            "RMSNorm 调用次数不一致："
                            f"family={family}, dtype={dtype_name}, "
                            f"case={case_name}, "
                            f"expected={expected_calls}, "
                            f"actual={actual_calls}"
                        )

                    contiguous_calls = sum(
                        bool(record["is_contiguous"])
                        for record in case_records
                    )

                    print(
                        f"{family:5s} | {dtype_name:4s} | "
                        f"{case_name:24s} | "
                        f"连续={contiguous_calls}/{actual_calls} | "
                        f"非连续={actual_calls-contiguous_calls}"
                    )

                    del attention_mask
                    del input_ids

            finally:
                for handle in hook_handles:
                    handle.remove()

                del model
                torch.cuda.empty_cache()

    fieldnames = [
        "family",
        "dtype",
        "case_name",
        "batch_size",
        "sequence_length",
        "module_name",
        "shape",
        "stride",
        "is_contiguous",
        "storage_offset",
        "device",
        "input_dtype",
        "weight_is_contiguous",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)

    total_calls = len(records)
    contiguous_calls = sum(
        bool(record["is_contiguous"]) for record in records
    )
    non_contiguous_calls = total_calls - contiguous_calls

    print()
    print("=" * 96)
    print("诊断汇总")
    print("=" * 96)
    print(f"总 RMSNorm 调用：{total_calls}")
    print(f"连续输入：{contiguous_calls}")
    print(f"非连续输入：{non_contiguous_calls}")
    print(
        "潜在 contiguous() 复制比例："
        f"{non_contiguous_calls / total_calls * 100.0:.2f}%"
        if total_calls
        else "潜在 contiguous() 复制比例：N/A"
    )
    print(f"CSV 已保存：{output_path}")

    if non_contiguous_calls == 0:
        print(
            "结论：本次紧凑模型测试中，RMSNorm 输入全部连续，"
            "hidden_states.contiguous() 不会产生实际数据复制。"
        )
    else:
        print(
            "结论：存在非连续 RMSNorm 输入，适配器中的 "
            "hidden_states.contiguous() 可能产生额外复制，"
            "需要进一步按模块和场景定位。"
        )


if __name__ == "__main__":
    main()
