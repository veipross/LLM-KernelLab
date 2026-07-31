# LLM-KernelLab

面向大语言模型（Large Language Model，LLM）推理优化的 GPU Kernel 实验项目。

本项目基于 **PyTorch + Triton** 实现并优化 Transformer 推理中的关键算子，探索以下 GPU Kernel 优化技术：

- Triton GPU Kernel 开发
- 算子融合（Operator Fusion）
- GPU Memory Access 优化
- Kernel Launch 开销降低
- Hugging Face 模型集成
- 正确性验证、Benchmark 与 Profiler 性能归因

## 项目背景

Transformer 类大模型推理过程中，大量计算时间消耗在少量高频算子上。

本项目围绕以下典型算子和模型集成场景展开：

- RMSNorm
- Residual + RMSNorm
- Hugging Face Llama/Qwen2 RMSNorm 替换
- 真实 KV Cache 单 Token Decode

项目对比的执行方式包括：

- PyTorch Eager
- PyTorch Native Operator
- `torch.compile`
- Triton Custom Kernel
- Hugging Face 原始模型
- 替换 Triton RMSNorm 后的 Hugging Face 模型

并通过正确性测试、多轮 Benchmark、输入布局诊断和 PyTorch Profiler 分析优化效果。

# 当前进展

## v0.1.0 —— RMSNorm Triton Kernel

已完成并发布。

实现内容：

- PyTorch RMSNorm Reference 实现
- Triton RMSNorm Forward Kernel
- 数值正确性验证
- 多 Shape Benchmark
- 多数据类型测试
- 三轮正式实验与性能报告

支持精度：

- FP16
- BF16
- FP32

实验环境：

- GPU：NVIDIA GeForce RTX 4090
- PyTorch：2.7.1+cu126
- Triton：3.3.1
- CUDA：12.6

完成闭环：

```text
Kernel 实现
    ↓
正确性验证
    ↓
Benchmark 测试
    ↓
性能分析
```

## v0.2.0 —— Fused Residual + RMSNorm

已完成并发布。

在 Transformer Block 中，经常存在：

```python
hidden_states = hidden_states + residual
hidden_states = RMSNorm(hidden_states)
```

传统方式需要分别执行：

```text
Residual Add Kernel
        +
RMSNorm Kernel
```

v0.2.0 实现了：

```text
Fused Residual + RMSNorm Triton Kernel
```

将以下操作融合为一次 GPU Kernel Launch：

- Residual Addition
- RMSNorm

已完成：

- PyTorch Reference 实现
- 未融合 Triton 实现
- 融合 Triton Forward Kernel
- 推理场景优化算子
- 完整正确性测试
- 多轮 Benchmark
- 实验环境自动记录
- 中文技术报告

RTX 4090 代表性结果：

- 相比 PyTorch Native，最高约 `3×` 加速
- 相比未融合 Triton，在典型中小规模场景获得约 `1.1～1.5×` 加速

## v0.3.0 —— Hugging Face RMSNorm Integration

当前开发版本，开发分支：

```text
feature/v0.3.0-llm-integration
```

本版本将 Triton RMSNorm 从独立 Kernel 实验扩展到 Hugging Face Llama 和 Qwen2 模型集成场景。

### Hugging Face 集成

实现文件：

```text
integrations/huggingface_rmsnorm.py
integrations/__init__.py
```

已实现：

- `HuggingFaceTritonRMSNorm`
- `from_huggingface_module`
- `replace_huggingface_rmsnorm_modules`
- LlamaRMSNorm 自动递归替换
- Qwen2RMSNorm 自动递归替换
- 保留原始 `Parameter` 对象
- 保持 `state_dict` 键和值一致
- CUDA `inference_mode` 下使用 Triton RMSNorm
- CPU、训练模式或不支持条件下回退 Hugging Face 风格 PyTorch 实现

### 正确性测试

测试文件：

```text
tests/test_huggingface_rmsnorm_integration.py
tests/test_huggingface_tiny_model_integration.py
```

覆盖内容：

- Llama、Qwen2
- FP16、BF16、FP32
- CPU fallback
- CUDA Triton 快速路径
- 参数对象保留
- `state_dict` 保持
- 递归替换
- 随机初始化紧凑 Llama/Qwen2 整模型
- logits 正确性
- Triton 实际调用次数
- 真实 KV Cache Prefill 与 Decode

当前完整测试结果：

```text
136 passed
```

### 模块级 Benchmark

Hugging Face 原始 RMSNorm 与 Triton 适配器三轮正式结果，hidden size 为 4096/8192：

- FP16 平均约 `4.67×`
- BF16 平均约 `4.66×`
- FP32 平均约 `2.75×`

与紧凑整模型一致的 hidden size 512 对齐实验：

- FP16 平均约 `4.86×`
- BF16 平均约 `4.59×`
- FP32 平均约 `3.56×`
- 总体成对平均约 `4.34×`

因此，紧凑整模型未获得明显端到端加速，并不是因为 hidden size 512 时 Triton RMSNorm 本身变慢。

### 紧凑整模型配置

Llama 和 Qwen2 均采用随机初始化配置，不下载预训练权重：

| 参数 | 数值 |
|---|---:|
| hidden size | 512 |
| intermediate size | 1376 |
| Transformer layers | 4 |
| attention heads | 8 |
| key/value heads | 4 |
| vocabulary size | 2048 |
| 每个模型 RMSNorm 数量 | 9 |

无 KV Cache 的单 Token 整模型结果：

- FP16 基本持平，约 `+0.10%`
- BF16 基本持平，约 `-0.62%`
- FP32 约慢 `6.52%`

该场景只是无 KV Cache 的单 Token 前向，不能称为真实自回归 Decode。

### 输入布局诊断

诊断文件：

```text
scripts/diagnose_huggingface_rmsnorm_layout.py
```

结果：

- RMSNorm 总调用次数：216
- 连续输入：216
- 非连续输入：0
- 权重全部连续
- 潜在 `contiguous()` 复制比例：0%

因此，整模型没有明显加速不是由 `hidden_states.contiguous()` 数据复制导致。

### PyTorch Profiler 归因

代表性场景：

- 模型：Llama
- 精度：FP16
- batch size：1
- sequence length：32
- hidden size：512
- layers：4
- RMSNorm 数量：9

Hugging Face 原始模型：

| 指标 | 结果 |
|---|---:|
| 墙钟平均延迟 | 1.009114 ms |
| CUDA Event 平均延迟 | 1.010705 ms |
| 整模型 CUDA Kernel 数 | 183 个/前向 |
| 整模型 Kernel 累计时间 | 341.901 μs/前向 |
| RMSNorm Kernel 数 | 72 个/前向 |
| RMSNorm Kernel 时间 | 104.275 μs/前向 |

Triton RMSNorm 模型：

| 指标 | 结果 |
|---|---:|
| 墙钟平均延迟 | 1.035308 ms |
| CUDA Event 平均延迟 | 1.028106 ms |
| 整模型 CUDA Kernel 数 | 120 个/前向 |
| 整模型 Kernel 累计时间 | 247.220 μs/前向 |
| RMSNorm Kernel 数 | 9 个/前向 |
| RMSNorm Kernel 时间 | 9.767 μs/前向 |

归因结论：

- RMSNorm Kernel 数从 72 降到 9，减少 `87.5%`
- RMSNorm Kernel 时间减少约 `90.6%`
- 整模型 Kernel 数减少约 `34.4%`
- 整模型 Kernel 累计时间减少约 `27.7%`
- 微型模型端到端延迟仍基本持平
- 收益受到主机调度、Kernel Launch 间隙、框架开销和其他模型算子影响
- Profiler 用于性能归因，正式性能结论以三轮 Benchmark 为准

### 真实 KV Cache Decode Benchmark

Benchmark 文件：

```text
benchmarks/benchmark_huggingface_kv_cache_decode.py
```

实验流程：

1. 随机初始化紧凑 Llama 或 Qwen2，不下载外部权重；
2. Hugging Face 原始模型和 Triton 模型使用完全相同的参数；
3. 先执行 Prefill，并使用 `use_cache=True` 生成独立的 `DynamicCache`；
4. 后续每次只输入一个新 Token；
5. 正确传递并原地更新 `past_key_values`；
6. 每次测量后使用 `DynamicCache.crop(prefill_length)` 恢复固定上下文长度；
7. 缓存裁剪放在 CUDA Event 计时区间之外；
8. 模型构造、权重复制、RMSNorm 替换和首次 Triton 编译不计入稳态延迟；
9. Provider 顺序按模型、精度、上下文长度和轮次进行轮换。

正式实验覆盖：

- Llama、Qwen2
- FP16、BF16
- batch size 1
- Prefill 长度 32、128、512
- 三轮正式实验
- 每个场景预热 20 次
- 每个场景测量 100 次

三轮共 72 条 Provider 记录，全部为 `status=ok`。

正确性结果：

- 每次 Prefill 实际调用 9 次 Triton RMSNorm
- 每个 Decode Token 实际调用 9 次 Triton RMSNorm
- FP16 三轮最大绝对误差：`0.001953125`
- BF16 三轮最大绝对误差：`0.015625`

三轮总体结果：

| 阶段 | 成对平均加速比 |
|---|---:|
| Prefill | `0.9959×` |
| 单 Token Decode | `0.9962×` |

最终结论：

> 在随机初始化的紧凑 Llama/Qwen2 模型、batch size 1、Prefill 长度 32/128/512、FP16/BF16 和真实 DynamicCache 单 Token Decode 条件下，替换 Triton RMSNorm 后的端到端 Decode 性能与 Hugging Face 原始模型基本持平，三轮成对平均加速比约为 `0.9962×`。

这说明模块级 RMSNorm Kernel 加速不能直接等同于整模型端到端 Decode 加速。

相关中文报告：

- [`docs/v0.3.0_huggingface_rmsnorm_h512_benchmark_report_zh.md`](docs/v0.3.0_huggingface_rmsnorm_h512_benchmark_report_zh.md)
- [`docs/v0.3.0_huggingface_rmsnorm_profiler_report_zh.md`](docs/v0.3.0_huggingface_rmsnorm_profiler_report_zh.md)
- [`docs/v0.3.0_huggingface_kv_cache_decode_report_zh.md`](docs/v0.3.0_huggingface_kv_cache_decode_report_zh.md)

# 已支持算子与能力

已完成：

- [x] RMSNorm
- [x] Fused Residual + RMSNorm
- [x] Hugging Face Llama/Qwen2 RMSNorm Integration
- [x] Hugging Face 整模型正确性验证
- [x] 输入布局诊断
- [x] PyTorch Profiler 性能归因
- [x] True KV Cache Decode Evaluation

计划：

- [ ] SwiGLU
- [ ] Rotary Position Embedding
- [ ] Causal Softmax
- [ ] 更多 Transformer 推理算子

# 项目结构

```text
LLM-KernelLab/
├── llm_kernels/
│   ├── torch_ops/
│   │   ├── rms_norm.py
│   │   └── fused_residual_rms_norm.py
│   └── triton_ops/
│       ├── rms_norm.py
│       └── fused_residual_rms_norm.py
├── integrations/
│   ├── __init__.py
│   └── huggingface_rmsnorm.py
├── benchmarks/
│   ├── benchmark_rmsnorm.py
│   ├── benchmark_fused_residual_rmsnorm.py
│   ├── benchmark_huggingface_rmsnorm_integration.py
│   ├── benchmark_huggingface_rmsnorm_aligned.py
│   └── benchmark_huggingface_kv_cache_decode.py
├── tests/
├── scripts/
├── docs/
└── results/
```

# 安装

创建环境：

```bash
conda create -n kernel-lab python=3.10
conda activate kernel-lab
```

安装依赖：

```bash
pip install -r requirements.txt
```

# 测试

运行完整测试：

```bash
pytest -q
```

当前结果：

```text
136 passed
```

# Benchmark

RMSNorm：

```bash
python benchmarks/benchmark_rmsnorm.py
```

Fused Residual + RMSNorm：

```bash
python benchmarks/benchmark_fused_residual_rmsnorm.py
```

Hugging Face RMSNorm 集成：

```bash
python benchmarks/benchmark_huggingface_rmsnorm_integration.py
```

hidden size 512 对齐实验：

```bash
python benchmarks/benchmark_huggingface_rmsnorm_aligned.py
```

真实 KV Cache Decode：

```bash
python benchmarks/benchmark_huggingface_kv_cache_decode.py \
  --families llama qwen2 \
  --dtypes fp16 bf16 \
  --prefill-lengths 32 128 512
```

# 技术说明

## 为什么使用 Triton？

Triton 可以：

- 自定义 GPU Kernel
- 优化 Memory Access Pattern
- 减少 Kernel Launch
- 实现算子融合
- 使用 Python 风格代码快速迭代 GPU Kernel

## 为什么进行 Kernel Fusion？

传统执行：

```text
Operation A
    ↓
写入 Intermediate Tensor
    ↓
Operation B
```

融合执行：

```text
Operation A + Operation B
    ↓
Single Kernel Launch
```

可以减少：

- 中间 Tensor 创建
- Global Memory 访问
- Kernel Launch 开销

## 为什么模块级加速没有完全转化为整模型加速？

整模型端到端延迟不仅由 RMSNorm 决定，还包括：

- Attention
- Linear / GEMM
- MLP
- KV Cache 更新
- Python 和框架调度
- Kernel Launch 间隙
- CUDA 同步和运行时开销

因此，即使 RMSNorm Kernel 数和累计时间显著下降，整模型端到端延迟仍可能基本持平。

# License

MIT License
