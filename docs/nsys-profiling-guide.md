# nsys 性能分析操作手册

> 适用于 rust-llm 项目, RTX 5070 Ti + CUDA 12.8 环境

## 工具路径

```bash
# nsys (Nsight Systems) — timeline 分析, 看 kernel 间隙/调度
/usr/local/cuda/bin/nsys   # 或直接 nsys (已在 PATH)

# ncu (Nsight Compute) — 单 kernel 深度分析, 看 roofline/带宽/计算利用率
/usr/local/cuda/bin/ncu    # 不在 PATH, 需要用绝对路径

nsys --version   # 2024.6.2
/usr/local/cuda/bin/ncu --version  # 2025.1.1
```

## 两个工具的区别

| | nsys (Nsight Systems) | ncu (Nsight Compute) |
|--|---|---|
| 用途 | 全局 timeline, kernel 间隙, CPU/GPU 交互 | 单个 kernel 内部指标 |
| 开销 | 低 (~10% 减速) | 高 (每个 kernel 重放多次, 100x+ 减速) |
| 输出 | `.nsys-rep` → timeline + SQLite | `.ncu-rep` → 详细 metrics |
| 适合回答 | "时间花在哪了？gap 多大？" | "这个 kernel 带宽利用率多少？compute-bound 还是 memory-bound？" |

## Step 1: 准备 Benchmark

用专门的 test 而不是 e2e test, 原因:
- 控制 warmup (让 cuBLAS JIT 编译在测量区间之外)
- 控制 decode 步数 (20 步足够看稳态, 不会太长)
- 关闭 log 噪声

Benchmark test 在 `tests/bench_decode.rs`, 结构:

```
warmup: prefill + 5 decode steps  → 触发 cuBLAS JIT, CUDA context 初始化
bench:  prefill + 20 decode steps → 这段被 profile
```

编译 (只编译不运行):

```bash
cargo test -r --test bench_decode --no-run
```

编译产物在 `target/release/deps/bench_decode-<hash>`, 直接用这个二进制给 nsys。

## Step 2: 采集 Trace

```bash
nsys profile \
  -o /tmp/decode_trace \
  -f true \
  -t cuda,cublas \
  --cuda-memory-usage=true \
  ./target/release/deps/bench_decode-* --nocapture
```

参数说明:

| 参数 | 含义 |
|------|------|
| `-o /tmp/decode_trace` | 输出文件路径 (自动加 `.nsys-rep` 后缀) |
| `-f true` | 覆盖同名文件 |
| `-t cuda,cublas` | 追踪 CUDA runtime + cuBLAS 调用 |
| `--cuda-memory-usage=true` | 记录 GPU 内存分配/释放 |
| `--nocapture` | 传给 test binary, 让 eprintln 输出可见 |

注意:
- nsys 会给进程加约 10% 的额外开销, TPOT 数据会偏高
- 不要加 `-t osrt` (OS runtime tracing), 在容器/无权限环境会报 warning 且不必要
- 产物: `/tmp/decode_trace.nsys-rep` + 自动生成 `/tmp/decode_trace.sqlite`

## Step 3: 快速查看汇总统计

### Kernel 耗时排名

```bash
nsys stats /tmp/decode_trace.nsys-rep --report cuda_gpu_kern_sum
```

输出每种 kernel 的总时间、次数、均值、最大最小值。用来判断:
- 哪个 kernel 占比最大 (优化它收益最高)
- 每种 kernel 被调用了多少次 (判断 fusion 机会)

### CUDA API 调用统计

```bash
nsys stats /tmp/decode_trace.nsys-rep --report cuda_api_sum
```

关注:
- `cudaLaunchKernel` 的次数和平均耗时 → launch overhead
- `cuMemAllocAsync` / `cuMemFreeAsync` 的次数 → 内存分配开销
- `cuStreamSynchronize` 的次数和总时间 → CPU 等 GPU 的阻塞时间
- `cuMemcpyDtoHAsync` → device→host 拷贝 (argmax 结果等)

## Step 4: 用 SQLite 做深度分析

nsys 自动导出 SQLite, 可以直接查询。核心表: `CUPTI_ACTIVITY_KIND_KERNEL`, 每行是一次 kernel 执行, 有 `start`/`end` 时间戳 (纳秒)。

### 计算 kernel 间 gap 总量

```sql
sqlite3 /tmp/decode_trace.sqlite "
WITH ordered AS (
  SELECT start, end, (end - start) as dur,
    ROW_NUMBER() OVER (ORDER BY start) as rn
  FROM CUPTI_ACTIVITY_KIND_KERNEL
),
gaps AS (
  SELECT (b.start - a.end) as gap_ns
  FROM ordered a
  JOIN ordered b ON b.rn = a.rn + 1
)
SELECT
  ROUND(SUM(gap_ns)/1e6, 2) as total_gap_ms,
  ROUND((SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL)/1e6, 2) as total_kernel_ms,
  ROUND((SELECT MAX(end)-MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL)/1e6, 2) as wall_ms
FROM gaps;"
```

### Gap 按大小分桶

```sql
sqlite3 /tmp/decode_trace.sqlite "
WITH ordered AS (
  SELECT start, end,
    ROW_NUMBER() OVER (ORDER BY start) as rn
  FROM CUPTI_ACTIVITY_KIND_KERNEL
),
gaps AS (
  SELECT (b.start - a.end) as gap_ns
  FROM ordered a
  JOIN ordered b ON b.rn = a.rn + 1
),
buckets AS (
  SELECT
    CASE
      WHEN gap_ns < 1000 THEN '  <1us'
      WHEN gap_ns < 5000 THEN ' 1-5us'
      WHEN gap_ns < 10000 THEN ' 5-10us'
      WHEN gap_ns < 50000 THEN '10-50us'
      WHEN gap_ns < 100000 THEN '50-100us'
      WHEN gap_ns < 1000000 THEN '0.1-1ms'
      ELSE '>1ms'
    END as bucket,
    gap_ns
  FROM gaps
)
SELECT bucket, COUNT(*) as count,
  ROUND(SUM(gap_ns)/1e6, 2) as total_ms,
  ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM gaps), 1) as pct
FROM buckets GROUP BY bucket ORDER BY bucket;"
```

### 找出大间隙 (>1ms) 的上下文

```sql
sqlite3 /tmp/decode_trace.sqlite "
WITH ordered AS (
  SELECT start, end, demangledName,
    ROW_NUMBER() OVER (ORDER BY start) as rn
  FROM CUPTI_ACTIVITY_KIND_KERNEL
),
big_gaps AS (
  SELECT (b.start - a.end) as gap_ns,
    sa.value as prev_kernel, sb.value as next_kernel
  FROM ordered a
  JOIN ordered b ON b.rn = a.rn + 1
  JOIN StringIds sa ON a.demangledName = sa.id
  JOIN StringIds sb ON b.demangledName = sb.id
  WHERE (b.start - a.end) > 1000000
)
SELECT ROUND(gap_ns/1e6, 2) as gap_ms,
  SUBSTR(prev_kernel, 1, 40) as after,
  SUBSTR(next_kernel, 1, 40) as before
FROM big_gaps ORDER BY gap_ns DESC;"
```

### 单步 Decode 分解

用 `argmax_kernel` 作为 decode step 的分界点, 取稳态步 (跳过前几步 warmup):

```sql
sqlite3 /tmp/decode_trace.sqlite "
WITH argmax AS (
  SELECT k.start, k.end,
    ROW_NUMBER() OVER (ORDER BY k.start) as step
  FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON k.demangledName = s.id
  WHERE s.value LIKE 'argmax%'
),
boundary AS (
  SELECT
    (SELECT end FROM argmax WHERE step = 10) as t0,
    (SELECT end FROM argmax WHERE step = 11) as t1
),
step_kernels AS (
  SELECT (end - start) as dur
  FROM CUPTI_ACTIVITY_KIND_KERNEL
  WHERE start >= (SELECT t0 FROM boundary)
    AND end <= (SELECT t1 FROM boundary)
)
SELECT
  COUNT(*) as kernel_count,
  ROUND(SUM(dur)/1e3, 1) as kernel_us,
  ROUND(((SELECT t1 FROM boundary)-(SELECT t0 FROM boundary))/1e3, 1) as wall_us,
  ROUND(100.0*SUM(dur)/((SELECT t1 FROM boundary)-(SELECT t0 FROM boundary)), 1) as gpu_util_pct
FROM step_kernels;"
```

## Step 5: 可视化 (可选)

```bash
# 方法 1: Perfetto (浏览器)
# 把 .nsys-rep 文件下载到本地, 打开 https://ui.perfetto.dev 拖入

# 方法 2: nsys-ui (如果有桌面环境)
nsys-ui /tmp/decode_trace.nsys-rep
```

Perfetto 里可以:
- 看 GPU kernel timeline (缩放到单步 decode 级别)
- 看 CUDA API 调用 timeline (CPU 侧)
- 对比 CPU 和 GPU 的时间线, 看哪里 CPU 在等 GPU

## 常见场景速查

| 想知道什么 | 用什么 |
|-----------|--------|
| 时间都花在哪些 kernel 上 | `--report cuda_gpu_kern_sum` |
| launch overhead 占比多少 | SQLite gap 分析 |
| cuBLAS 内部选了什么 kernel | `--report cuda_gpu_kern_sum` 看 kernel 名字 |
| 有没有不必要的 sync | `--report cuda_api_sum` 看 `cuStreamSynchronize` 次数 |
| 内存分配是否频繁 | `--report cuda_api_sum` 看 `cuMemAllocAsync` 次数 |
| 单个 kernel 的带宽/计算利用率 | 改用 `ncu` (见下节) |

## 附: ncu 单 Kernel 分析 (简要)

```bash
# 对特定 kernel 跑 roofline 分析
/usr/local/cuda/bin/ncu --set roofline \
  -k rms_norm_kernel \
  ./target/release/deps/bench_decode-* --nocapture

# 看 SM 和 HBM 利用率
/usr/local/cuda/bin/ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained,\
  dram__throughput.avg.pct_of_peak_sustained \
  -k fused_mlp_intermediate_kernel \
  ./target/release/deps/bench_decode-* --nocapture
```

注意 ncu 会让每个 kernel 重放多次收集 metrics, 整体运行时间会**慢 100 倍以上**, 建议用 `-k` 只分析目标 kernel。
