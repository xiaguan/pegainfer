# DeepSeek-V3.2 推理复现

> **Status**: Phase 2d — logits 对齐调试中，已修 3 个确认 bug，仍 0/5
> **TL;DR**: 在 pegainfer 上复现 DSV3.2 (671B MoE) 8x H20-3e 推理。FP8 权重，MLA + MoE + TP1/EP8。
> **Next action**: 排查 MoE routed expert 路径（疑似 DeepEP combine 与 pre-weighting 双重计权）。

---

## 硬件

8x NVIDIA H20-3e, 每卡 141GB HBM3, 总计 ~1.1TB。NVLink 互联。

- DSV3.2 FP8 权重 ~671GB → 放完剩 ~460GB 给 KV cache + activation，宽裕
- BF16 (~1.34TB) 放不下，**FP8 是必需项**
- H20 bandwidth-bound (4TB/s HBM)，对 MLA decode 友好

## 本地环境速查

- **Python (transformers / 验证脚本)**: `/root/develop/xingming/vllm_test/.venv/bin/python`
- **模型权重目录**: `/data/models/DeepSeek-V3.2`（通过环境变量 `PEGAINFER_DSV32_MODEL_PATH` 传入）

## 并行策略

- **Attention**: TP8（8 卡全部参与）
- **MoE**: EP8（每卡 256/8 = 32 routed experts，shared expert 复制到每卡）
- **Dense FFN（前 3 层）**: TP8
- **通信**: NCCL AllReduce (TP), All-to-All (EP)

## 阶段规划

### Phase 0 — 骨架：权重加载 + 多卡基础设施

目标：把 DSV3.2 FP8 权重正确加载到 8 卡上，按 TP8/EP8 切分。

- [x] DSV3.2 模型配置解析（61 layers, MLA dims, MoE params, FP8 quant config, YaRN RoPE）
- [x] safetensors FP8 权重加载（`Fp8Matrix` 类型：raw e4m3 + block-wise scale_inv）
- [x] 单卡 partial load 验证（embedding + 4 层，含 1 MoE w/ 256 experts，~15GB VRAM）
- [ ] cudarc 多 GPU context 管理 — Qwen3 TP 已有蓝本：`DeviceContext::new_with_device(ordinal)` per-rank context, `Qwen3Executor::from_runtime` 多卡初始化流程 (`model_executor.rs:493-559`)
- [ ] NCCL 通信初始化（8 卡 NVLink 拓扑）— Qwen3 TP 已用 `Comm::from_devices()` + `attach_tp_comm()` 做 AllReduce；DSV3.2 额外需要 All-to-All (EP)，需确认 cudarc NCCL bindings 是否封装了 `ncclSend`/`ncclRecv`
- [ ] 权重分发：attention TP8 切分（仿 Qwen3 `load_tensor_2d_row_shard`）, MoE EP8 分配（每卡 32 experts）, shared expert 复制

验收：权重全部上卡，显存占用符合预期。

### Phase 0.5 — FP8 GEMM 集成（DeepGEMM）

目标：把 DeepGEMM 的 SM90 FP8 block-scale GEMM kernel 集成进 pegainfer，在单卡 partial load 上验证 forward correctness。

**方案**：直接依赖 DeepGEMM submodule（`third_party/DeepGEMM`，MIT 协议），只用 kernel 头文件层，host 侧 torch 胶水用 CUDA driver API 重写。

DeepGEMM 架构分三层，我们只取底两层：

```
[不用] Python API / Host C++ (csrc/)  ← 依赖 torch::Tensor, ATen
   ↓
[取]   JIT 编译逻辑                     ← nvcc 编译 template instantiation → cubin
   ↓
[取]   CUDA Kernel headers              ← deep_gemm/include/，纯 CUDA + CUTLASS header-only
       (sm90_fp8_gemm_1d2d.cuh)
```

torch 在 DeepGEMM 中仅用于两件事（均可平替）：
- `torch::Tensor` 当 metadata accessor（读 data_ptr / stride / dtype）→ 换成 raw pointer + shape
- `torch::empty` 分配 GPU memory → 换成 cudarc `alloc_zeros`

集成步骤：

- [x] `build.rs` 加 DeepGEMM include path（`third_party/DeepGEMM/deep_gemm/include` + CUTLASS headers），SM90a + C++20 + `--expt-relaxed-constexpr`
- [x] 写 `csrc/fp8_gemm.cu`：thin C wrapper，用 `cuTensorMapEncodeTiled` 构造 TMA descriptor，AOT 编译两组 tile config（block_m=64/128, block_n=128, block_k=128），使用 1D2D kernel（bf16 输出）
- [x] Rust FFI 暴露 `fp8_gemm_cuda(a, scale_a, b, scale_b, d, m, n, k, stream)`（d 为 bf16）
- [x] 在线 activation FP8 量化 kernel（bf16 → fp8 e4m3 + per-token 1×128 block scale）— 从 TRT-LLM `scale_1x128_kernel` 抽取到 `csrc/fp8_quantize.cu`，零外部依赖
- [x] 单卡 partial load 上验证：embedding → RMSNorm → FP8 quantize → q_a_proj GEMM → 对比 HF reference（max err 1.56e-2, mean err 1.22e-3）
- [x] 修复 norm 权重加载：DSV3.2 checkpoint 中 norm 权重为 f32，新增 `load_1d_f32_as_bf16` 做正确转换

验收：FP8 GEMM 输出与 HF fp8 dequant + matmul 对齐（误差 < 1e-2）。✅ 已通过。

### Phase 1 — MLA Forward（前 3 层 Dense 打通）

目标：MLA attention + Dense FFN 前向，前 3 层 hidden states 与 HF 对齐。

- [x] KV cache 改 per-layer paged buffer（page_size=64），适配 FlashMLA 期望的 `[num_blocks, 64, h_kv, head_dim]`
- [x] FlashMLA dense decode kernel 集成（`build.rs` 编译 + thin C wrapper + Rust FFI），同 DeepGEMM 模式
- [x] FP8 linear ops 封装（`ops/fp8.rs`）：`fp8_quantize_into` / `fp8_gemm_into` / `fp8_linear_into`，支持共享量化
- [x] Absorbed weight 预计算（`weights.rs` 加载时 CPU dequant kv_b_proj → W_UK/W_UV bf16）
- [x] cuBLAS strided batched GEMM（`csrc/linear.cu`）用于 Q absorption / V de-absorption
- [x] MLA CUDA kernels（`csrc/mla.cu`）：partial RMSNorm、k_rope RoPE、q_rope extract+RoPE+copy、KV cache write
- [x] MLA forward 全流程（`forward.rs`）：Q/KV path + absorption + FlashMLA 3-phase + de-absorption + o_proj + Dense FFN
- [x] YaRN RoPE cos/sin cache 预计算并挂到 DsV32Model
- [x] 端到端验证：前 3 层 hidden states 与 unabsorbed reference 对齐（Layer 0: mean_err=0.026, Layer 1: 0.093, Layer 2: 0.089）
- [ ] Prefill path（当前 decode only，bs=1 per request）

验收：前 3 层 hidden states 与 reference 对齐。✅ 已通过（`dsv32_forward_3_layers` 测试；历史离线校验脚本与中间产物已清理）。

### Phase 2 — MoE Forward + 多卡（8x H20 prefill）

目标：8 卡 TP8/EP8 prefill forward 跑通，全 61 层 output logits 与 reference 对齐。

#### Phase 2a — 多卡 Executor ✅

- [x] DSV3.2 多卡 executor（per-rank DeviceContext + 8 线程并行加载，56s）
- [x] EP8 权重分配加载：每卡 32 routed experts + shared expert 全卡复制
- [x] Embedding/norm/attention 权重全卡复制（TP1：不切分 attention）

#### Phase 2b — DeepEP Intranode 集成

EP8 跨卡通信，端到端的唯一硬卡点。

- [x] `build.rs` 编译 DeepEP intranode kernel（`-DDISABLE_NVSHMEM -DTOPK_IDX_BITS=64`，SM90a）
- [x] C wrapper（`csrc/deep_ep.cu`）：5 个 extern "C" 函数（layout, barrier, notify_dispatch, dispatch, combine）
- [x] Rust FFI（`ffi.rs`）：DeepEP 5 函数 + CUDA IPC/HostAlloc/Memcpy helpers + i32→i64 cast kernel
- [x] Buffer 管理（`deep_ep.rs`）：NVLink buffer alloc + IPC handle exchange (Arc<Barrier> + Arc<Mutex<>>) + host-mapped recv counters
- [x] Executor 集成（`executor.rs`）：EP>1 时并行初始化 DeepEP buffers，挂到 DsV32Model
- [x] Forward 集成（`forward.rs`）：`forward_moe_ep` 实现完整 dispatch→compute→combine pipeline

#### Phase 2c — MoE Forward ✅

- [x] Sigmoid gating + TopK-8 routing + group-limited normalization（`csrc/moe.cu`）
- [x] Shared expert 本地 FFN
- [x] Routed expert FFN（per-expert sequential FP8 GEMM，decode 路径）
- [x] MoE forward 集成到 `forward_layer`（`forward.rs:forward_moe`）

#### Phase 2d — 端到端 Prefill

DeepEP 就绪后，补齐剩余胶水打通全模型。

- [x] `forward_moe` 接入 DeepEP dispatch/combine（`forward_moe_ep`，替代 EP1 local-only 路径）
- [x] Final RMSNorm + lm_head GEMM（`forward_final`：RMSNorm → bf16 GEMM with `output_projection()`，`tie_word_embeddings` 复用 embedding 矩阵）
- [x] Executor forward loop（`forward_prefill_sparse` + `forward_final`，`DsV32Executor::forward` 8 rank 并行 dispatch）
- [x] **DeepEP combine timeout** — 修复 4 个参数错误后 combine 通过，详见下方调试记录
- [x] **切换到真正 prefill path**
  - FlashMLA SM90 sparse prefill kernel（`d_qk=576, d_v=512`，NSA）
  - batch embedding + batch GEMM（seq_len > 1）
  - FlashMLA sparse prefill attention 接入主执行路径
  - MoE DeepEP dispatch/combine 支持 batch tokens
- [ ] 全模型 output logits 与 Python reference 对齐
- [ ] （可选）Grouped GEMM（DeepGEMM `GroupedContiguous`）替代 per-expert sequential

验收：给定 prompt，8 卡 prefill forward output logits 与 reference 一致。

#### DeepEP 记录（精简）

- combine timeout 相关排查已完成并关闭，当前阶段关注点已转为 logits 对齐。
- 详细逐步调试日志已从项目文档中清理，仅保留状态与结论。

### Phase 3 — 生成循环 + 服务化

目标：完整 decode 生成，接入 `/v1/completions` API。

- [x] `main.rs` 接入 DSV3.2 模型识别与 HTTP server 启动路径
- [x] 最小 `scheduler_dsv32`：串行请求执行，`prefill` 后复用 MLA/KV 做单 token decode
- [x] H20 smoke test：`/v1/completions` 可起服务并返回 `hello world` / `1+1=` / reasoning 风格请求
- [ ] Decode path：单 token MLA + MoE forward
- [ ] KV cache 管理（MLA 压缩格式的 append/管理）
- [ ] CUDA Graph capture（decode path）
- [x] `bench_serving` 接入 DSV3.2
- [ ] E2E 生成验证 / 输出质量回归

当前边界（精简）：

- 当前服务链路已打通，但生成质量仍明显不对；本阶段验证的是“能起服务、能收发请求”，不是“输出已正确”。
- 当前生成主路径已经切到 `prefill + decode reuse`：
  - prompt 先走 sparse prefill；
  - 后续 token 走单 token decode，并复用 paged MLA KV。
- H20 当前 smoke timing：
  - `hello world`, `max_tokens=8`: `~1.78s`
  - `1+1=`, `max_tokens=8`: `~1.81s`
  - reasoning prompt, `max_tokens=16`: `~5.84s`
- 当前并行形态是 `TP1 + EP8`，不是 `DP8`：
  - attention / dense FFN 不做 TP 切分，每张卡各算一份；
  - routed MoE 通过 DeepEP 在 8 卡之间分发与合并。

验收：greedy decode 输出与 reference 一致，API 可用。

### Phase 4 — 性能优化

- [x] Prefill EP MoE：`forward_moe_ep()` 从逐 token local expert 执行改成按 expert 分组、按 `max_bs` 分块批处理
- [x] H20 `64 -> 64` 回归：TTFT 从 `7.08-7.34s` 降到 `0.76-0.97s`，steady TPOT 从 `180-190ms/token` 降到 `64-76ms/token`
- [x] EP local expert gather/scatter kernel 化：把逐 token `DtoD gather` / `moe_weighted_add` 改成 batched gather/scatter，原始版本把 `64 -> 64` 压到 `TTFT 0.28-0.43s`、`steady TPOT 54-58ms/token`，但 greedy correctness 回归
- [x] 找到 regression 根因并修复：bug 不是 FFI/layout，而是 batched scatter 把同一 token 的 expert 累加顺序从 top-k-slot-major 改成 expert-major，在 bf16 read-modify-write 下触发 greedy 漂移；现已改成按 top-k slot 保序 batching，`e2e_dsv32_small` 恢复 `44/44` 通过，H20 `64 -> 64` accepted baseline 为 `TTFT 452.17ms`、`steady TPOT 53.46ms/token`
- [x] 热点画像切换：`fp8_gemm` 已升到 `42.5%` kernel time，超过 `cached_notify_combine 33.5%`；`cuMemcpyDtoDAsync` 基本退出主路径，但 chunk metadata `H2D` 仍然很高
- [x] 失败经验已记录：device-built slot-major EP metadata 与 GPU greedy sampling 都没有把 `64 -> 64` benchmark 拉出 accepted baseline，其中后者还带来 correctness 回归，已撤回
- [ ] 通信与计算 overlap（双 micro-batch 流水线）
- [ ] Decode 侧 SM 分区实验
- [ ] MTP speculative decoding
- [ ] 性能 benchmark vs vLLM/SGLang

## 已有基础设施（可复用/参考）

Qwen3 TP 已跑通多卡推理，DSV3.2 多卡部分可直接参考：

| 组件 | 位置 | DSV3.2 可复用程度 |
|------|------|----------------|
| 多 GPU context 管理 | `DeviceContext::new_with_device()` (`tensor.rs`) | 直接复用 |
| NCCL comm 初始化 | `Comm::from_devices()` (`model_executor.rs:536`) | 直接复用 AllReduce；All-to-All 需新增 |
| per-rank worker 线程 | `RankWorker` + `WorkerCommand` (`model_executor.rs:831-923`) | 架构可复用，内部 Lane 换成 DSV3.2 |
| TP 权重切分加载 | `load_tensor_2d_row_shard` (`weight_loader.rs:131`) | bf16 column/row shard 可复用；FP8 shard 需新增 |
| NCCL AllReduce bench | `benches/nccl_bench.rs` | 已有 TP2 PCIe 数据，需补 8 卡 NVLink 数据 |
| `ModelExecutor` trait | `model_executor.rs:455-464` | DSV3.2 executor 实现同一 trait |
| `attach_tp_comm` 模式 | `qwen3/weights.rs:392` | 同一模式：load → attach comm → run |
| DeepGEMM FP8 kernel | `third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` | kernel 直接用；host 侧 TMA setup 已重写（`csrc/fp8_gemm.cu`），bf16 输出 |
| DeepGEMM grouped GEMM | 同上，`GemmType::GroupedContiguous` / `GroupedMasked` | MoE expert 计算直接可用 |
| FlashMLA dense decode | `third_party/FlashMLA/csrc/sm90/decode/dense/` | SM90 MLA decode kernel；host 侧 torch 胶水已重写（`csrc/flash_mla.cu`），3-phase: metadata → decode → combine |
| MLA paged KV cache | `src/model/dsv32/mla_kv.rs` | per-layer paged buffer (page_size=64)，`MlaKvPool`/`MlaKvState`，天然匹配 FlashMLA kcache 格式 |

## DeepGEMM 1D2D 集成要点

> 后来者看这里，不用重新挖。

### 1D1D vs 1D2D 的区别

| | 1D1D | 1D2D |
|---|---|---|
| Scale A (activation) | 1D per-token `[ceil(K/128), padded(M,4)]` | 同左 |
| Scale B (weight) | 1D per-channel `[ceil(K/128), padded(N,4)]` | **2D per-block `[ceil(N/128), ceil(K/128)]`** |
| SFB 加载方式 | TMA descriptor | **全局内存直接读（math warp）** |
| 输出类型 | FP32 | **BF16** |
| D TMA descriptor | 无 swizzle, FP32 | **128B swizzle, BF16** |

**用哪个？** Hopper block-scale FP8 主路径全走 1D2D。SGLang（`deep_gemm_wrapper/entrypoint.py:84`）和 vLLM（`deep_gemm.py:120`）均如此。DeepGEMM 默认 recipe `(1, 128, 128)` 在 `gran_n != 1` 时分派到 1D2D（`gemm.hpp:87`）。1D1D 仅用于 `gran_n==1` (per-channel) 和 k_grouped GEMM — 不是我们的场景。

DSV3.2 checkpoint 权重 scale 本身就是 2D block-scale `[ceil(N/128), ceil(K/128)]`，天然匹配 1D2D 的 `kMajorSFB = Major::K`。

### Kernel 模板参数（对比 1D1D）

1D2D 比 1D1D 多三个模板参数：

```
kMajorSFB        — cute::UMMA::Major::K (DSV3.2 权重 scale K-major) 或 Major::MN
kSwizzleDMode    — BF16 输出的 TMA swizzle: 128 (block_n=128 时)
kNumLastStages   — SM90 kernel body 不使用，保留给 SM100，AOT 设 0 即可
epilogue_type_t  — EpilogueIdentity（普通 GEMM）或 EpilogueHeadSplits（MHA 分头）
```

1D1D 最后一个参数是 `cd_dtype_t = float`；1D2D 替换为 `epilogue_type_t`（输出固定 bf16）。

### TMA Descriptor 设置

| Descriptor | 数据类型 | gmem layout | smem block | swizzle |
|---|---|---|---|---|
| A | UINT8 (fp8) | [M, K] row-major | [block_m, block_k] | 128B |
| B | UINT8 (fp8) | [N, K] row-major | [block_n, block_k] | 128B |
| SFA | FLOAT32 | [ceil(K/128), padded(M,4)] K-chunk-major | [block_m, 1] | 无 |
| D | **BFLOAT16** | [M, N] row-major | [block_m, block_n] | **128B** |

**SFB 不走 TMA** — kernel 里 math warp 直接 `__ldg()` 从全局内存读。这是 1D2D 和 1D1D 的核心区别。

D 的 128B swizzle 导致 `TMA_D_BLOCK_N = 128 / sizeof(bf16) = 64`，kernel 发两次 TMA store 覆盖 block_n=128。

### Kernel Launch Args（1D2D 签名）

```
(sfb,              // float* — 直接指针，不是 TMA
 grouped_layout,   // int* — Normal GEMM 传 nullptr
 shape_m, shape_n, shape_k,   // uint32_t
 tensor_map_a, tensor_map_b, tensor_map_d, tensor_map_sfa)
```

对比 1D1D: `(gmem_a, gmem_b, grouped_layout, tensor_map_buffer, shapes, tma_a, tma_b, tma_sfa, tma_sfb, tma_cd)` — 注意顺序和数量都不同。

### Shared Memory 用量

1D2D 比 1D1D 省显存：D 缩小一半 (bf16 vs fp32)，且无 per-stage SFB。

| 组件 | Config 1 (64×128, 8 stages) | Config 2 (128×128, 5 stages) |
|---|---|---|
| smem_d (bf16) | 16384 | 32768 |
| stages × (A+B+SFA) | 8 × 24832 = 198656 | 5 × 33280 = 166400 |
| smem_sfb (K=7168) | 224 | 224 |
| barriers | 128 | 80 |
| **Total** | **≈ 215 KB** | **≈ 199 KB** |

SM90 smem capacity = 232448 bytes (227 KB)，两组 config 均在容量内。1D1D 相同 block 尺寸只能跑 7/4 stages，1D2D 多出 1 stage 因为 D buffer 更小且无 per-stage SFB。

## MLA Forward 实现要点

> 后来者看这里，不用重新挖。

### 维度速查

| 名称 | 值 | 含义 |
|------|------|------|
| hidden_size | 7168 | 隐藏维度 |
| num_heads | 128 | 注意力头数 |
| q_lora_rank | 1536 | Q 低秩压缩维度 |
| kv_lora_rank | 512 | KV 低秩压缩维度 |
| qk_nope_head_dim | 128 | 每头 Q/K 非 RoPE 维度 |
| qk_rope_head_dim | 64 | 每头 Q/K RoPE 维度 |
| v_head_dim | 128 | 每头 V 维度 |
| q_head_dim | 192 | = nope 128 + rope 64 |
| kv_a_proj_dim | 576 | = kv_lora_rank 512 + rope 64 |
| kv_b_head_dim | 256 | = nope 128 + v 128 |

### 权重矩阵形状

所有 FP8 权重存储为 `[output_dim, input_dim]`。GEMM 用法: `Y[M, N] = W[M, K] @ X[K, N]`。

| 权重 | 形状 | 类型 | 输入 → 输出 |
|------|------|------|-------------|
| q_a_proj | [1536, 7168] | FP8 | hidden → q_compressed |
| q_a_layernorm | [1536] | bf16 | RMSNorm weight |
| q_b_proj | [24576, 1536] | FP8 | q_compressed → q_full (128×192) |
| kv_a_proj | [576, 7168] | FP8 | hidden → [c_kv(512), k_rope(64)] |
| kv_a_layernorm | [512] | bf16 | RMSNorm weight (只 norm c_kv 部分) |
| kv_b_proj | [32768, 512] | FP8 | c_kv → [k_nope(128), v(128)] × 128 heads |
| o_proj | [7168, 16384] | FP8 | attn_output (128×128) → hidden |

### MLA 数据流（Decode，含 Absorption）

MLA 的核心优化：不在 decode 时展开 kv_b_proj（会产生 32768d），而是把 kv_b_proj 的信息"吸收"进 Q 侧和 O 侧。

```
hidden [7168]
  │
  ├─── Q Path ────────────────────────────────────────────────────
  │   FP8 quantize → q_a_proj [1536, 7168]      → q_compressed [1536]
  │   RMSNorm(q_a_layernorm)                     → q_compressed [1536]
  │   FP8 quantize → q_b_proj [24576, 1536]      → q_full [24576] = 128 heads × 192
  │   ↓ per head h:
  │   split → q_nope_h [128] + q_rope_h [64]
  │   RoPE(q_rope_h)
  │   ★ Q absorption: q_absorbed_h = q_nope_h @ W_UK_h → [512]
  │   assemble → q_h = [q_absorbed_h(512), q_rope_h(64)] = [576]
  │   ↓ all heads:
  │   Q_absorbed [128, 576]     ← FlashMLA 的 Q 输入
  │
  ├─── KV Path ───────────────────────────────────────────────────
  │   FP8 quantize → kv_a_proj [576, 7168]       → kv_a [576]
  │   split → c_kv [512] + k_rope [64]
  │   RMSNorm(kv_a_layernorm, only c_kv part)    → c_kv [512]
  │   RoPE(k_rope)
  │   concat → [c_kv(512), k_rope(64)] = [576]   → 写入 KV cache
  │
  ├─── FlashMLA Decode ──────────────────────────────────────────
  │   Q: [batch, 128, 1, 576]  (q_seq_per_hk=128, h_k=1)
  │   K cache: [num_blocks, 64, 1, 576]  (per-layer paged)
  │   → attn_out [batch, 1, 128, 512]  (d_v = kv_lora_rank)
  │   ≈ [128, 512] per token    ← 每头 512d，是 c_kv 的 attention 加权
  │
  ├─── V De-absorption + O Projection ──────────────────────────
  │   ★ V de-absorption per head h:
  │   v_out_h = attn_out_h [512] @ W_UV_h^T [512, 128] → [128]
  │   concat all heads → [128 × 128] = [16384]
  │   FP8 quantize → o_proj [7168, 16384]        → attn_output [7168]
  │
  └─── Residual: hidden += attn_output
```

### Absorption 原理

标准 MLA attention per head h:
```
score_h = (q_nope_h @ k_nope_h^T) + (q_rope_h @ k_rope^T)
        = (q_nope_h @ (W_UK_h @ c_kv)^T) + (q_rope_h @ k_rope^T)
        = (q_nope_h @ W_UK_h) @ c_kv^T + q_rope_h @ k_rope^T
```

定义 `q_absorbed_h = [q_nope_h @ W_UK_h, q_rope_h]` (576d)，`kv_cache = [c_kv, k_rope]` (576d)，则：
```
score_h = q_absorbed_h @ kv_cache^T   ← 一次 dot product，FlashMLA 直接算
```

V 侧类似：标准路径 `v_h = W_UV_h @ c_kv`，absorption 后 FlashMLA 直接用 c_kv 做 V（d_v=512），出来再做 `v_out_h = attn_out_h @ W_UV_h^T` 还原到 v_head_dim=128。

### W_UK / W_UV 提取

kv_b_proj 是 FP8 [32768, 512] = [128 heads × 256, 512]。每头 256 = k_nope 128 + v 128。

**加载时一次性 dequant**，提取两组 bf16 权重：

```
kv_b_proj_bf16 [32768, 512] → reshape [128, 256, 512]
W_UK_h = kv_b_proj_bf16[h, 0:128,   :] → [128, 512]   # K nope 权重
W_UV_h = kv_b_proj_bf16[h, 128:256, :] → [128, 512]   # V 权重
```

存储为两个 bf16 buffer：
- `w_uk`: [128, 128, 512] → 内存 128 × 128 × 512 × 2 = 16 MB/layer
- `w_uv`: [128, 128, 512] → 16 MB/layer
- 合计 32 MB/layer × 61 = ~2 GB，可接受

### Q Absorption 的 GEMM 策略

q_full 输出格式：`[h0_nope(128), h0_rope(64), h1_nope(128), h1_rope(64), ...]` — nope 和 rope 交替排列。

**cuBLAS Strided Batched GEMM**（`cublasGemmStridedBatchedEx`，支持 bs > 1）:

```
Q absorption: C_h = W_UK_h^T @ q_nope_h    (per head, batch = 128 heads)

cuBLAS column-major 视角 (W_UK_h 存储 [nope, kv_lora] row-major = [kv_lora, nope] col-major):
  opA = N:  A_h = W_UK_h col-major [kv_lora=512, nope=128],  lda=512
  opB = N:  B_h = q_nope_h col-major [nope=128, bs],          ldb=q_b_proj_dim(24576)
  C_h = q_absorbed col-major [kv_lora=512, bs],               ldc=num_heads*kv_a_proj_dim(73728)
  m=512, n=bs, k=128

  strideA = nope * kv_lora = 128 * 512 = 65536    (between heads in W_UK)
  strideB = q_head_dim = 192                        (between heads in q_full interleaved layout)
  strideC = kv_a_proj_dim = 576                     (between heads in FlashMLA Q buffer)
```

stride_B=192 利用 q_full 的 nope/rope 交替布局，自然跳过 rope gap。stride_C=576 直接写入 FlashMLA Q 的 absorbed 部分，rope 部分由 `mla_rope_q_copy_cuda` 填充。

V de-absorption 类似:
```
V de-absorption: C_h = W_UV_h @ attn_out_h

  opA = T:  A_h col-major [kv_lora=512, v_dim=128], lda=512, → transposed [v_dim, kv_lora]
  opB = N:  B_h = attn_out_h col-major [kv_lora=512, bs],    ldb=num_heads*kv_lora(65536)
  C_h = v_out_h col-major [v_dim=128, bs],                   ldc=o_proj_input_dim(16384)
  m=128, n=bs, k=512

  strideA = v_dim * kv_lora = 128 * 512 = 65536
  strideB = kv_lora = 512
  strideC = v_dim = 128
```

**为什么用 cuBLAS 不用 DeepGEMM einsum？** Absorption 是标准 bf16 batched GEMM，不涉及 FP8 block-scale 或 TMA descriptor。cuBLAS 已经 init 且对这些尺寸性能好。DeepGEMM einsum 的 JIT 基础设施搬过来工程量大、收益不大。

### FlashMLA 调用参数（DSV3.2 具体值）

```
flash_mla_decode(
    q:          [batch, 128, 1, 576]    // q_seq_per_hk = seqlen_q * (h_q/h_k) = 1*128
    kcache:     [num_blocks, 64, 1, 576] // per-layer paged buffer 的 layer slice
    o:          [batch, 1, 128, 512]
    ...
    h_q: 128,  h_k: 1,  d_k: 576,  d_v: 512,
    softmax_scale: (192)^{-0.5} * yarn_mscale²,
    is_causal: 0  (decode 不需要 causal mask)
)
```

**关键 stride 计算**（flash_mla.cu 内部）:
```
q_batch_stride = 128 × 1 × 576 = 73728
q_row_stride   = 1 × 576 = 576
q_head_stride  = 576
o_batch_stride = 1 × 128 × 512 = 65536
o_row_stride   = 512
o_head_stride  = 128 × 512 = 65536
```

### FP8 Linear Pipeline

每次 FP8 GEMM 前需要做 activation FP8 量化。封装为 `fp8_linear_into`:

```
input [hidden_dim, bs] bf16
  → fp8_quantize_1x128:  fp8_act [M, K] + scale_a [ceil(K/128), padded(M,4)]
  → fp8_gemm_cuda:       output [N, bs] bf16
```

**共享量化**：同一 input 同时供 q_a_proj 和 kv_a_proj 时，FP8 量化只做一次。

**Scratch buffer 策略**：FP8 activation buf 和 scale buf 大小取决于 input 维度，预分配最大尺寸（hidden_size=7168 对应 7168 bytes fp8 + 56×padded(M,4)×4 bytes scales）。

### RoPE 处理

MLA 的 RoPE 只作用于 rope 维度（64d），不影响 nope 维度：
- q_rope: q_full 中每头的 [h*192+128 : h*192+192]（交替排列）
- k_rope: kv_a 输出的 [512:576]（连续 64d）

DSV3.2 使用 YaRN RoPE，参数在 config 中。softmax_scale 需要乘 `mscale²`。

### Dense FFN Forward（前 3 层）

```
normed [7168, bs]  (post_attention_layernorm 输出)
  FP8 quantize → gate_proj [18432, 7168]  → gate [18432, bs]
  FP8 quantize → up_proj   [18432, 7168]  → up   [18432, bs]
  (共享同一份 normed FP8 量化)
  silu(gate) * up                          → act  [18432, bs]
  FP8 quantize → down_proj [7168, 18432]  → ffn_out [7168, bs]
  residual: hidden += ffn_out
```

### 现有基础设施映射

| 需要的操作 | 函数 | 位置 |
|-----------|------|------|
| Embedding lookup | `ops::embedding_batch` | ✅ 直接用 |
| RMSNorm batched | `ops::rms_norm_batch_into` | ✅ 直接用 |
| Fused add + RMSNorm | `ops::fused_add_rms_norm_batch_into` | ✅ 直接用 |
| SiLU × up (分离 gate/up) | `ops::silu_mul_batch_into` | ✅ 直接用 |
| BF16 GEMM | `ops::gemm_into` | ✅ 直接用 |
| FP8 quantize + GEMM | `ops::fp8::fp8_linear_into` | ✅ 已封装 |
| FP8 shared quantize | `ops::fp8::fp8_quantize_into` + `fp8_gemm_into` | ✅ 已封装 |
| FlashMLA 3-phase | `ffi::flash_mla_{get_metadata,decode,combine}` | ✅ forward.rs 直接调 FFI |
| Strided batched GEMM | `ffi::gemm_strided_batched_cuda` | ✅ 已新增（`csrc/linear.cu`） |
| MLA RoPE (k_rope) | `ffi::mla_rope_kv_cuda` | ✅ 已新增（`csrc/mla.cu`） |
| MLA RoPE (q_rope extract+copy) | `ffi::mla_rope_q_copy_cuda` | ✅ 已新增 |
| Partial RMSNorm (c_kv only) | `ffi::rms_norm_partial_cuda` | ✅ 已新增 |
| KV cache write (scatter) | `ffi::mla_kv_cache_write_cuda` | ✅ 已新增 |
| YaRN RoPE cos/sin precompute | `precompute_yarn_rope` (`weight_loader.rs`) | ✅ 已实现，挂到 `DsV32Model.cos_cache/sin_cache` |

## 探索方向

- **Decode 侧 SM 分区 overlap**: 双 micro-batch decode，attention 与 MoE dispatch+compute+combine 通过 SM 分区并行。RTX 5070 Ti 实验表明 32 SM (46%) 达峰值 89%。需验证 H20 的 SM-bandwidth 曲线和共享 HBM 时的干扰。

## 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-04-14 | 直接上 DSV3.2-0324，跳过 DSV2-Lite | 8x H20-3e 141GB 够用，不浪费时间在小模型 |
| 2026-04-14 | FP8 为必需项 | BF16 1.34TB 超过 8 卡 1.1TB 总容量 |
| 2026-04-14 | TP8 + EP8 | 8 卡环境的自然切分：attention 全卡 TP，MoE 每卡 32 experts |
| 2026-04-14 | 新增 `Fp8Matrix` 类型 | 现有 `DeviceMatrix` 是 bf16，FP8 需要 raw bytes + block-wise scale_inv 分开存 |
| 2026-04-14 | partial load 支持 | 671GB 单卡放不下，测试用 `from_safetensors_partial` 只加载前 N 层 |
| 2026-04-14 | FP8 GEMM 用 DeepGEMM | DeepSeek 自研，专为 128×128 block-scale FP8 优化，SM90 TMA+WGMMA，自带 grouped GEMM (MoE)。kernel header-only 无 torch 依赖，host 侧 torch 胶水可用 CUDA driver API 平替。TRT-LLM fp8_blockscale 底层也是 DeepGEMM，直接依赖源头更干净 |
| 2026-04-14 | DeepGEMM AOT 编译 | DSV3.2 矩阵尺寸已知，build.rs nvcc 预编译固定 tile config，不需要运行时 JIT |
| 2026-04-14 | DeepGEMM 编译选项 | SM90a (`-gencode=arch=compute_90a,code=sm_90a`) + C++20 + `--expt-relaxed-constexpr --expt-extended-lambda`，需要 DeepGEMM 自带 CUTLASS v2.1.1 (不可与 flashinfer 的 v4.4.2 混用) |
| 2026-04-14 | 两组 tile config | block_m=64/block_n=128/7stages (decode小M) + block_m=128/block_n=128/4stages (prefill大M)，kNumSMs 默认 132 可通过 `PEGAINFER_DG_NUM_SMS` 覆盖 |
| 2026-04-14 | activation 量化从 TRT-LLM 抽取 | flashinfer 的 `mxfp8_quantize` 是 SM100/SF_VEC=32 格式，不匹配 DeepGEMM 的 1×128 block-scale。flashinfer 自己支持 DSV3.2 也是靠 TRT-LLM 内嵌的 `scale_1x128_kernel`（`fp8_blockscale_gemm_kernel.cuh`）。直接抽取该 kernel 到 `csrc/fp8_quantize.cu`，去掉所有 TRT-LLM 依赖 |
| 2026-04-14 | Scale 布局 K-chunk-major | TRT-LLM `scale_1x128_kernel` 输出 dequant scale 为 `[ceil(K/128), padded(M, 4)]`（K-chunk 维在前，M 维在内），与 DeepGEMM TMA scale descriptor 期望一致。注意不是 `[M, ceil(K/128)]` |
| 2026-04-14 | GEMM 切 1D2D | SGLang（`deep_gemm_wrapper/entrypoint.py:84`）和 vLLM（`deep_gemm.py:120`）在 Hopper block-scale 主路径均走 1D2D。DeepGEMM 默认 recipe `(1, 128, 128)` 在 `gran_n != 1` 时分派到 `sm90_fp8_gemm_1d2d`（`gemm.hpp:87`），grouped/masked GEMM 在 SM90 写死 1D2D（`gemm.hpp:194,258`）。1D1D 仅用于 `gran_n==1` per-channel 和 k_grouped，不是我们的 case |
| 2026-04-14 | 1D2D AOT config: 64×128/8stages + 128×128/5stages | 依据 DeepGEMM heuristics (`sm90.hpp`)：block_m<=64 用 1 warpgroup (128 math threads)，否则 2 warpgroups (256)。stages 取 SM90 smem 容量 232448 下的最大值。kMajorSFB=K 匹配 DSV3.2 checkpoint scale layout，kSwizzleDMode=128 匹配 bf16 block_n=128，kNumLastStages=0 因 SM90 kernel 不使用该参数 |
| 2026-04-14 | 1D2D 输出 bf16 (非 fp32) | DeepGEMM 1D2D API 强制 `d.scalar_type() == torch::kBFloat16`。kernel 内部 fp32 accumulator → bf16 cast 后经 TMA store 写回。DSV3.2 forward 后续层本就需要 bf16 输入，省去显式 cast |
| 2026-04-14 | norm 权重 f32→bf16 转换 | DSV3.2 checkpoint 中 layernorm 权重存为 f32，而 Qwen3 系列为 bf16。`load_tensor_1d` 直接按 bf16 读会长度翻倍。新增 `load_1d_f32_as_bf16` 做显式转换 |
| 2026-04-14 | MLA attention 用 FlashMLA | DeepSeek 自研 MLA kernel（`third_party/FlashMLA`），SM90 dense decode 达 3000 GB/s / 660 TFLOPS (H800)。kernel 天然适配 DSV3.2 MLA 维度：`head_size_k=576` (c_KV 512 + k_R 64)，`head_size_v=512`，MQA 模式 (`h_kv=1`)。集成模式同 DeepGEMM：只取 kernel 层，host 侧 torch 胶水用 CUDA driver API 重写 |
| 2026-04-14 | KV cache 切 per-layer paged，page_size=64 | FlashMLA dense decode 硬性要求 `page_block_size=64`（`dense_decode.h:67`），且 kcache 为 per-layer 独立 buffer `[num_blocks, page_block_size, num_heads_k, head_size_k]`。现有 `KvPool` all-layers-in-one-page 布局（`kv_pool.rs`）改为 per-layer paged buffer，`PagePool`/`KvState` 分配逻辑不变，`KvLayout` 几何调整即可 |
| 2026-04-14 | Absorption 用 cuBLAS strided batched GEMM，不用 DeepGEMM einsum | Absorption/de-absorption 是标准 bf16 batched GEMM（非 FP8 block-scale），无需 TMA descriptor。cuBLAS `cublasGemmStridedBatchedEx` 已 init 且对 [128,512]×[128,bs] 这类尺寸性能好。DeepGEMM einsum 虽有现成 `bhr,hdr->bhd` kernel，但 JIT 基础设施（`compiler->build`、torch tensor 胶水）搬过来工程量大且 bf16 场景收益不大 |
| 2026-04-14 | W_UK/W_UV CPU dequant | kv_b_proj FP8 [32768, 512] 在加载时一次性 CPU dequant 到 bf16，拆成 W_UK/W_UV 上传 GPU。32 MB/layer × 61 ≈ 2 GB 可接受。CPU dequant 避免写 GPU dequant kernel，加载是一次性开销 |
| 2026-04-14 | Partial RMSNorm kernel | kv_a_layernorm 只 norm c_kv 前 512d，保留 k_rope 后 64d。标准 `rms_norm_batch_into` 会 norm 整个 576d。新增 `rms_norm_partial_cuda`（`csrc/mla.cu`）做 sub-range norm |
| 2026-04-14 | MLA RoPE 用 half-split 格式 | DSV3.2 用 transformers 标准 `rotate_half`：pairs 是 (x[i], x[i+half_dim])，与 Qwen3 `precompute_rope` 的 cos/sin cache 布局一致。不是 interleaved pairs |
| 2026-04-14 | Dense FFN 分离 gate/up | Qwen3 gate_up fused 为一个投影 → `silu_mul_fused_batch_into`。DSV3.2 gate_proj 和 up_proj 是独立 FP8 权重 → 用 `silu_mul_batch_into`（分离 gate/up 输入）。共享 FP8 量化（normed 只量化一次供两个投影） |
| 2026-04-15 | Phase 1 验证用 unabsorbed reference | transformers 不支持 DSV3.2，无法用 HF model 做 reference。历史上使用离线脚本手动加载 safetensors + FP8 dequant，走 **unabsorbed** MLA 路径（标准 QKV + attention）做 f32 reference。数学上等价于 absorbed 路径（证明见 doc「Absorption 原理」节）。单 token decode (pos=0) 下 attention 是 trivial 的（softmax=1.0），验证覆盖了投影、norm、RoPE、absorption/de-absorption、FFN 残差。 |
| 2026-04-15 | Phase 1 精度阈值 | 3 层 forward mean_err < 0.1，max_err < 5×layer_count。FP8 量化误差在 mean 维度控制良好（0.026 → 0.093），max 维度个别 outlier 达 2.8–3.7（7168 维中 < 0.1% 维度），属于 FP8 block-scale 正常范围 |
| 2026-04-15 | DeepEP 做 EP8 All-to-All | DeepSeek 自研 MoE 通信库（`third_party/DeepEP`，MIT 协议）。intranode kernel 通过 NVLink IPC buffer 实现 All-to-All，达 153-158 GB/s（H800 NVLink）。集成模式同 DeepGEMM：只取 kernel 层，host 侧 torch 胶水用 CUDA driver API 重写 |
| 2026-04-15 | DeepEP 只用 intranode | 8x H20 单机场景只需 intranode normal kernels（NVLink），编译时 `-DDISABLE_NVSHMEM` 去掉 RDMA 依赖。Low-latency kernels 留给后续 decode 优化 |
| 2026-04-15 | Prefill 优先 | Phase 2 先跑通 8 卡 prefill，decode 留 Phase 3。prefill 用 intranode normal kernels（支持大 batch），decode 后续视需要引入 low-latency kernels |
| 2026-04-15 | MoE 不用 FlashInfer，用 DeepGEMM grouped GEMM + 自写 routing | FlashInfer MoE 深度绑定 TRT-LLM 基础设施（JIT 编译、CUTLASS extensions、TVM-FFI），剥离代价大于自写。routing kernel（sigmoid + group-limited TopK）逻辑确定，自写几十行 CUDA。grouped GEMM 直接用已集成的 DeepGEMM `GroupedContiguous`。FlashInfer MoE 核心竞争力在 Blackwell SM100+ CuTe DSL 融合 pipeline，SM90 (H20) 收益不大 |
| 2026-04-15 | 路线 B：跳过单卡 MoE 验证，直推 8 卡端到端 | DeepEP 是 EP8 唯一跨卡通信，其余（final norm、lm_head、forward loop）都是一天内的活。先啃 DeepEP，然后一步到位全模型 logits 对齐。历史上曾临时用 token-by-token decode 验证 logits；该路径现已退场，prefill 只保留 sparse 实现 |

## DeepEP Intranode 集成要点

> 后来者看这里，不用重新挖。

### 架构分层

```
[不用] Python API (deep_ep/buffer.py)        ← 依赖 torch.distributed, pybind11
   ↓
[不用] Host C++ (csrc/deep_ep.cpp)           ← 依赖 torch::Tensor, ATen, pybind11
   ↓
[取]   Kernel API (csrc/kernels/api.cuh)     ← 纯 C 函数签名，参数全是 raw pointer + int
   ↓
[取]   CUDA Kernels (csrc/kernels/*.cu)      ← 纯 CUDA，无外部依赖
```

### 三种模式（只用第一种）

| 模式 | 场景 | 通信方式 | 文件 | 我们用？ |
|------|------|---------|------|---------|
| Intranode normal | prefill / training | NVLink IPC | `intranode.cu` | ✅ |
| Internode normal | 多机 prefill | NVLink + RDMA (NVSHMEM) | `internode.cu` | ✗ |
| Low-latency | decode | pure RDMA (NVSHMEM) | `internode_ll.cu` | ✗（留 Phase 3） |

### Intranode 数据流

```
Step 1: get_dispatch_layout(topk_idx, num_experts)
  → num_tokens_per_rank [8]
  → is_token_in_rank [num_tokens, 8]
  → num_tokens_per_expert [256]

Step 2: notify_dispatch(...)  ← NVLink barrier + 交换 token counts
  → GPU 间通过 IPC buffer 交换 per-rank size prefix matrix
  → CPU busy-wait 读 mapped host memory 获取 recv count（不兼容 CUDA Graph）

Step 3: intranode::dispatch(x, topk_idx, topk_weights, is_token_in_rank, ...)
  → recv_x [num_recv_tokens, hidden]          ← 本卡收到的 tokens
  → recv_topk_idx [num_recv_tokens, num_topk]  ← 对应 expert indices
  → recv_topk_weights [num_recv_tokens, num_topk]
  → recv_src_idx [num_recv_tokens]              ← combine 时用
  → handle (prefix matrices + send_head)

Step 4: [本地 expert FFN 计算]

Step 5: intranode::combine(x_out, topk_weights, src_idx, handle, ...)
  → combined_x [num_orig_tokens, hidden]       ← 加权 reduce 回原卡
```

### Buffer 管理（需重写的 host 侧逻辑）

DeepEP Buffer 构造过程（`deep_ep.cpp` 构造函数）：

1. 每卡 `cudaMalloc` 分配 NVLink buffer（大小由 `Config::get_nvl_buffer_size_hint()` 计算）
2. `cudaIpcGetMemHandle` 获取本卡 buffer 的 IPC handle
3. 通过 all_gather 交换 IPC handles（原实现用 `torch.distributed`，我们改为 Rust 线程间直传）
4. `cudaIpcOpenMemHandle` 打开其他 7 卡的 buffer → `buffer_ptrs[8]`
5. 上传 `buffer_ptrs` 到 GPU（`buffer_ptrs_gpu`）
6. 同样方式交换 barrier signal buffer → `barrier_signal_ptrs[8]` / `barrier_signal_ptrs_gpu`
7. 分配 mapped host memory（`cudaHostAlloc` + `cudaHostGetDevicePointer`）用于 CPU-GPU 同步：
   - `moe_recv_counter_mapped`：总 recv token count
   - `moe_recv_expert_counter_mapped`：per-expert recv token count
8. 分配 workspace（32 MB）

### Kernel 编译选项

```
-DDISABLE_NVSHMEM          # 去掉 RDMA 依赖
-DTOPK_IDX_BITS=64         # DSV3.2 用 int64 topk indices（与 Python 端一致）
SM90a                       # H20 = SM90
```

intranode kernel 模板参数 `kNumRanks` 需 AOT 实例化为 8（在 `launch.cuh` 的 `LAUNCH_DISPATCH` 宏展开）。

### Config 参数（8 卡 intranode）

```
num_sms = 20           # 通信占用的 SM 数（偶数，每 2 个 SM 一个 channel）
num_max_nvl_chunked_send_tokens = 6    # dispatch config for EP8
num_max_nvl_chunked_recv_tokens = 256  # dispatch config for EP8
```

NVLink buffer 大小估算（`get_nvl_buffer_size_hint`）：
```
channels = num_sms / 2 = 10
per_channel = 8 ranks × (2*1+3) ints + 8 × 256 × hidden_bytes + ...
```
DSV3.2 hidden=7168 bf16 → hidden_bytes=14336 → 约 300 MB/卡 NVLink buffer。

### 与 DeepGEMM/FlashMLA 集成对比

| | DeepGEMM | FlashMLA | DeepEP |
|---|---|---|---|
| kernel 依赖 | header-only CUTLASS | SM90 MLA kernel | 纯 CUDA + configs.cuh |
| torch 在 host 侧的角色 | metadata accessor | metadata accessor | **重度使用**（tensor alloc, stream mgmt, IPC exchange） |
| 我们需要重写的 host 逻辑 | TMA descriptor setup | 3-phase launch 参数计算 | **Buffer 管理 + IPC exchange + CPU-GPU sync** |
| 编译复杂度 | 高（CUTLASS headers, TMA） | 中 | 低（纯 CUDA runtime） |
| NVSHMEM 依赖 | 无 | 无 | intranode 无需（`-DDISABLE_NVSHMEM`） |
