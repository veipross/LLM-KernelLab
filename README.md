# LLM-KernelLab

面向大模型推理的 Triton/CUDA 融合算子优化与性能评测项目。

## Project Goals

本项目针对 Transformer 推理中的常用算子，分别实现：

- PyTorch Eager
- torch.compile
- Triton Kernel
- CUDA C++ Extension

并从以下维度进行统一评测：

- 数值正确性
- GPU Kernel 延迟
- P50 / P95 延迟
- 有效显存带宽
- 不同数据类型性能
- 不同输入 Shape 性能
- 相对 PyTorch 的加速比
- Transformer Block 端到端性能

## Planned Operators

- [ ] RMSNorm
- [ ] Fused Residual + RMSNorm
- [ ] SwiGLU
- [ ] Rotary Position Embedding
- [ ] Causal Softmax
- [ ] Mini Transformer Block Integration

## Hardware

Initial benchmark platform:

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- CUDA Toolkit: 12.6
- PyTorch: 2.7.1
- Triton: 3.3.1

## Repository Structure

```text
llm_kernels/
├── torch_ops/
├── triton_ops/
└── cuda_ops/

benchmarks/
tests/
integrations/
docs/
results/

## 4. 创建许可证

我们自己的代码使用 MIT License：

```bash
cat > LICENSE <<'EOF'
MIT License

Copyright (c) 2026 Zhang Yixiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## RMSNorm v0.1 Results

The first milestone implements and evaluates RMSNorm with:

- PyTorch Eager
- PyTorch Native
- `torch.compile`
- Custom Triton kernel

Benchmarks were collected on an NVIDIA GeForce RTX 4090 using three
independent runs per data type. Reported P50 and P95 values are medians
across the three runs.

`rows` represents the flattened token dimension:

```text
rows = batch_size × sequence_length
```

The one-row `1×4096` case is used only for kernel-launch-overhead analysis
and is excluded from the main speedup summary.

| Data type | Median speedup vs PyTorch Native | Best speedup | Best shape |
|---|---:|---:|---:|
| FP16 | 4.91× | 5.33× | 128×4096 |
| BF16 | 4.84× | 5.33× | 128×4096 |
| FP32 | 2.12× | 2.93× | 128×4096 |

![Triton RMSNorm speedup](results/figures/rmsnorm_triton_speedup.png)

See the [full RMSNorm benchmark report](docs/rmsnorm_benchmark_report.md)
for latency, P95, stability, numerical error, and effective-bandwidth results.
