# LLM-KernelLab

面向大语言模型（Large Language Model, LLM）推理优化的 GPU Kernel
实验项目。

本项目基于 **PyTorch + Triton** 实现并优化 Transformer
推理中的关键算子，探索 GPU Kernel 优化技术，包括：

-   Triton GPU Kernel 开发
-   算子融合（Operator Fusion）
-   GPU Memory Access 优化
-   Kernel Launch 开销降低
-   性能 Benchmark 与分析

# 项目背景

Transformer 类大模型推理过程中，大量计算时间消耗在少量高频算子上。

本项目针对 LLM 推理中的典型算子：

-   RMSNorm
-   Residual + RMSNorm

实现不同执行方式：

-   PyTorch Eager
-   PyTorch Native Operator
-   torch.compile
-   Triton Custom Kernel

并通过 Benchmark 分析优化效果。

# 当前进展

## v0.1.0 ------ RMSNorm Triton Kernel

已完成版本。

实现：

-   PyTorch RMSNorm Reference 实现
-   Triton RMSNorm Forward Kernel
-   数值正确性验证
-   多 Shape Benchmark
-   多 Data Type 测试
-   性能报告整理

支持：

-   FP16
-   BF16
-   FP32

实验环境：

-   GPU: NVIDIA GeForce RTX 4090
-   PyTorch: 2.7.1+cu126
-   Triton: 3.3.1
-   CUDA: 12.6

完成闭环：

    Kernel 实现

          ↓

    正确性验证

          ↓

    Benchmark 测试

          ↓

    性能分析

# v0.2.0 ------ Fused Residual + RMSNorm

当前版本。

在 Transformer Block 中，经常存在：

``` python
hidden_states = hidden_states + residual
hidden_states = RMSNorm(hidden_states)
```

传统方式：

    Residual Add Kernel

            +

    RMSNorm Kernel

需要多个 GPU Kernel 调用。

本版本实现：

    Fused Residual + RMSNorm Triton Kernel

将：

-   Residual Addition
-   RMSNorm

融合为一次 GPU Kernel Launch。

# v0.2.0 已实现功能

完成：

-   PyTorch Reference 实现
-   Triton Fused Forward Kernel
-   推理场景优化算子
-   完整 Correctness Test
-   Benchmark 测试框架
-   实验环境自动记录

测试结果：

    pytest

    112 passed

# Benchmark 结果

Benchmark 对比：

  实现方式         说明
  ---------------- ---------------
  PyTorch Eager    基础实现
  PyTorch Native   原生算子
  torch.compile    编译优化
  Triton Unfused   未融合 Triton
  Triton Fused     融合 Kernel

测试：

-   FP16
-   BF16
-   FP32

不同：

-   batch size
-   hidden dimension

RTX 4090 实验结果：

-   相比 PyTorch Native，最高约 3× 加速
-   相比 Unfused Triton Kernel，在典型中小规模场景获得约 1.1～1.5× 加速

实验结果：

    results/

    ├── csv/

    └── environment/

# 已支持算子

已完成：

-   [x] RMSNorm
-   [x] Fused Residual + RMSNorm

计划：

-   [ ] SwiGLU
-   [ ] Rotary Position Embedding
-   [ ] Causal Softmax
-   [ ] 更多 Transformer 推理算子

# 项目结构

    LLM-KernelLab/

    ├── llm_kernels/
    │
    │   ├── torch_ops/
    │   │   ├── rms_norm.py
    │   │   └── fused_residual_rms_norm.py
    │   │
    │   ├── triton_ops/
    │   │   ├── rms_norm.py
    │   │   └── fused_residual_rms_norm.py
    │
    ├── benchmarks/
    │
    ├── tests/
    │
    ├── scripts/
    │
    ├── docs/
    │
    └── results/

# 安装

创建环境：

``` bash
conda create -n kernel-lab python=3.10

conda activate kernel-lab
```

安装依赖：

``` bash
pip install -r requirements.txt
```

# 测试

运行：

``` bash
pytest -q
```

当前：

    112 passed

# Benchmark

RMSNorm：

``` bash
python benchmarks/benchmark_rmsnorm.py
```

Fused Residual RMSNorm：

``` bash
python benchmarks/benchmark_fused_residual_rmsnorm.py
```

# 技术说明

## 为什么使用 Triton？

Triton 可以：

-   自定义 GPU Kernel
-   优化 Memory Access Pattern
-   减少 Kernel Launch
-   实现算子融合

## 为什么进行 Kernel Fusion？

传统执行：

    Operation A

    ↓

    写入 Intermediate Tensor

    ↓

    Operation B

融合执行：

    Operation A + Operation B

    ↓

    Single Kernel Launch

减少：

-   中间 Tensor 创建
-   Global Memory 访问
-   Kernel Launch 开销

# License

MIT License
