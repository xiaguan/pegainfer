# Decode 阶段 Launch Overhead 分析

> 测试日期: 2026-02-15
> 硬件: RTX 5070 Ti, Qwen3-4B (BF16)
> 工具: nsys 2024.6.2, 测试用例 `tests/bench_decode.rs`
> 命令: `nsys profile -o /tmp/decode_trace -f true -t cuda,cublas --cuda-memory-usage=true ./target/release/deps/bench_decode-*`

## 总览

| 指标 | 值 |
|------|-----|
| 总 wall time (GPU 视角) | 662 ms |
| 总 kernel 执行时间 | 328 ms (49.5%) |
| 总 gap 时间 | 335 ms (50.5%) |
| Kernel launch 总数 | 12,314 |
| 平均 gap | 27.2 μs |

**结论: GPU 约一半时间在空转等待下一个 kernel 被提交。**

但注意上面包含了 warmup/prefill 阶段的大间隙（cuBLAS JIT 编译等）。稳态 decode 的数据更有参考价值，见下节。

## 稳态单步 Decode 分解

取第 10 步 decode（跳过 warmup 和首次路径）：

| 指标 | 值 |
|------|-----|
| Wall time | 15,200 μs (15.2 ms) |
| Kernel 执行时间 | 11,971 μs (79%) |
| Gap 时间 | 3,229 μs (21%) |
| Kernel launch 次数 | 400 |
| 平均 gap | ~8 μs |
| TPOT (含 overhead) | ~15.2 ms |
| TPOT (理论无 overhead) | ~12.0 ms |

**稳态 GPU 利用率 ≈ 79%, launch overhead ≈ 21%。**

## Gap 分布

| 间隙大小 | 数量 | 占比 | 累计时间 |
|----------|------|------|---------|
| <1μs | 541 | 4.4% | 0.4 ms |
| 1-5μs | 2,185 | 17.7% | 3.6 ms |
| 5-10μs | 6,251 | 50.8% | 45.7 ms |
| 10-50μs | 3,092 | 25.1% | 75.3 ms |
| 50-100μs | 219 | 1.8% | 13.1 ms |
| 0.1-1ms | 17 | 0.1% | 3.7 ms |
| >1ms | 8 | 0.1% | 192.9 ms |

主体是 5-50μs 的间隙，对应 `cudaLaunchKernel` 的 CPU 端开销（平均 5.2μs/次）。

8 个 >1ms 大间隙来源：

| 间隙 | 位置 | 原因 |
|------|------|------|
| 156 ms | `rms_norm_batched` → `cutlass GEMM` | Prefill 首次 cuBLAS 调用, JIT 编译 |
| 25 ms | `rms_norm` → `gemvx` | Decode 首步 cuBLAS 路径选择 |
| 1-3 ms | `embedding_batched` → `rms_norm_batched` | Warmup 期间分配 |
| 1 ms | `fused_mlp_output` → `add` | 疑似 `cuMemAllocAsync` |

## Kernel 耗时排名

| Kernel | 占比 | 总时间 | 次数 | 均值 |
|--------|------|--------|------|------|
| fused_mlp_intermediate | 34.2% | 112 ms | 900 | 125 μs |
| cuBLAS gemvx (GEMV) | 29.1% | 95 ms | 3,627 | 26 μs |
| fused_mlp_output | 23.2% | 76 ms | 900 | 85 μs |
| cutlass GEMM (prefill) | 6.2% | 20 ms | 504 | 40 μs |
| fused_gqa_attention | 4.2% | 14 ms | 1,188 | 12 μs |
| rms_norm | 1.9% | 6 ms | 1,827 | 3.4 μs |
| add | 0.6% | 1.8 ms | 1,944 | 0.9 μs |
| copy | 0.3% | 1.0 ms | 1,154 | 0.8 μs |

MLP (fused_mlp_intermediate + fused_mlp_output) 占 57.4%，是绝对大头。

## 每步 400 次 Kernel Launch 的来源

28 层 × ~14 次/层 + 3 次 final = ~395 次，与实测 400 吻合。

每层 ~14 次的构成（推测）：
- rms_norm: 2 次（input_layernorm + post_attn_layernorm）
- cuBLAS gemv: ~5 次（q_proj, k_proj, v_proj 各 1 + o_proj 1 + lm_head 分摊... 实际 cuBLAS 可能内部多次 launch）
- fused_attention: 1 次
- copy: ~2 次（中间 buffer 拷贝）
- fused_mlp: 2 次（intermediate + output）
- add: 2 次（attention residual + mlp residual）

## 优化方向

### 方案 A: CUDA Graph（消除 launch gap）

- 预期收益: TPOT 15.2ms → ~12.0ms, 提速 ~27%
- 代价: 需要预分配所有中间 buffer, ops 改为 in-place 写入, 处理动态参数 (pos, seq_len) 更新
- 复杂度: 高

### 方案 B: 更多 Kernel Fusion（减少 launch 次数）

- 把 `rms_norm` + 后续 GEMV 合并
- 把 `add`(residual) fuse 进前一个 kernel
- 预期: 400 次 → ~200 次, gap 减半, ~10% 提速
- 复杂度: 中

### 方案 C: 预分配 Buffer Pool（消除 cuMemAllocAsync）

- 当前每步有 `cuMemAllocAsync` / `cuMemFreeAsync` 调用（各 ~20K 次）
- 预分配固定 buffer 可消除分配开销, 也为 CUDA Graph 铺路
- 复杂度: 中

### 建议顺序

1. **Buffer 预分配**（方案 C）— 低风险, 为后续铺路
2. **Kernel Fusion**（方案 B）— 减少 launch 次数, 独立收益
3. **CUDA Graph**（方案 A）— 在 C+B 基础上做, 吃掉剩余 gap

## 复现命令

```bash
# 编译
cargo test -r --test bench_decode --no-run

# Profile
nsys profile -o /tmp/decode_trace -f true -t cuda,cublas --cuda-memory-usage=true \
  ./target/release/deps/bench_decode-* --nocapture

# 分析
nsys stats /tmp/decode_trace.nsys-rep --report cuda_gpu_kern_sum
nsys stats /tmp/decode_trace.nsys-rep --report cuda_api_sum

# 直接查 SQLite
sqlite3 /tmp/decode_trace.sqlite "
WITH ordered AS (
  SELECT start, end, ROW_NUMBER() OVER (ORDER BY start) as rn
  FROM CUPTI_ACTIVITY_KIND_KERNEL
),
gaps AS (
  SELECT (b.start - a.end) as gap_ns
  FROM ordered a JOIN ordered b ON b.rn = a.rn + 1
)
SELECT
  ROUND(SUM(gap_ns)/1e6, 2) as total_gap_ms,
  ROUND((SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL)/1e6, 2) as total_kernel_ms
FROM gaps;"
```
