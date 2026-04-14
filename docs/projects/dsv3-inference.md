# DeepSeek-V3 推理复现

> **Status**: Phase 0.5 完成 — FP8 forward 已验证对齐，进入 Phase 1 (MLA forward)
> **TL;DR**: 在 pegainfer 上复现 DSV3.2 (671B MoE) 8x H20-3e 推理。FP8 权重，MLA + MoE + TP8/EP8。
> **Next action**: KV cache 改 per-layer paged (page_size=64) + FlashMLA dense decode kernel 集成。

---

## 硬件

8x NVIDIA H20-3e, 每卡 141GB HBM3, 总计 ~1.1TB。NVLink 互联。

- DSV3 FP8 权重 ~671GB → 放完剩 ~460GB 给 KV cache + activation，宽裕
- BF16 (~1.34TB) 放不下，**FP8 是必需项**
- H20 bandwidth-bound (4TB/s HBM)，对 MLA decode 友好

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
- [ ] NCCL 通信初始化（8 卡 NVLink 拓扑）— Qwen3 TP 已用 `Comm::from_devices()` + `attach_tp_comm()` 做 AllReduce；DSV3 额外需要 All-to-All (EP)，需确认 cudarc NCCL bindings 是否封装了 `ncclSend`/`ncclRecv`
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
- [x] 修复 norm 权重加载：DSV3 checkpoint 中 norm 权重为 f32，新增 `load_1d_f32_as_bf16` 做正确转换

验收：FP8 GEMM 输出与 HF fp8 dequant + matmul 对齐（误差 < 1e-2）。✅ 已通过。

### Phase 1 — MLA Forward（前 3 层 Dense 打通）

目标：MLA attention + Dense FFN 前向，前 3 层 hidden states 与 HF 对齐。

- [x] KV cache 改 per-layer paged buffer（page_size=64），适配 FlashMLA 期望的 `[num_blocks, 64, h_kv, head_dim]`
- [x] FlashMLA dense decode kernel 集成（`build.rs` 编译 + thin C wrapper + Rust FFI），同 DeepGEMM 模式
- [ ] MLA projection：hidden → c_KV (512d) + k_R (64d)，q → q_C + q_R
- [ ] Decode 时 absorb 优化（W_UK @ c_KV 还原 K，或 absorb 进 W_O）
- [ ] RoPE 仅作用于 k_R / q_R 部分
- [ ] Dense FFN forward（TP8 AllReduce）
- [ ] RMSNorm, residual, embedding

验收：前 3 层 logits/hidden states 与 reference 对齐。

### Phase 2 — MoE Forward（全模型 prefill）

目标：MoE routing + expert dispatch/combine，全 61 层 prefill forward 跑通。

- [ ] Sigmoid gating + TopK-8 routing + normalization
- [ ] EP8 All-to-All dispatch（token → expert 所在卡）
- [ ] Expert FFN 计算
- [ ] EP8 All-to-All combine（结果 → 原卡）
- [ ] Shared expert 本地计算 + 加回
- [ ] 全模型 prefill logits 对齐

验收：给定 prompt，output logits 与 reference 一致。

### Phase 3 — 生成循环 + 服务化

目标：完整 decode 生成，接入 `/v1/completions` API。

- [ ] Decode path：单 token MLA + MoE forward
- [ ] KV cache 管理（MLA 压缩格式的 append/管理）
- [ ] CUDA Graph capture（decode path）
- [ ] 接入 `GenericServerEngine<DeepSeekV3Model>`
- [ ] E2E 生成验证

验收：greedy decode 输出与 reference 一致，API 可用。

### Phase 4 — 性能优化

- [ ] 通信与计算 overlap（双 micro-batch 流水线）
- [ ] Decode 侧 SM 分区实验
- [ ] MTP speculative decoding
- [ ] 性能 benchmark vs vLLM/SGLang

## 已有基础设施（可复用/参考）

Qwen3 TP 已跑通多卡推理，DSV3 多卡部分可直接参考：

| 组件 | 位置 | DSV3 可复用程度 |
|------|------|----------------|
| 多 GPU context 管理 | `DeviceContext::new_with_device()` (`tensor.rs`) | 直接复用 |
| NCCL comm 初始化 | `Comm::from_devices()` (`model_executor.rs:536`) | 直接复用 AllReduce；All-to-All 需新增 |
| per-rank worker 线程 | `RankWorker` + `WorkerCommand` (`model_executor.rs:831-923`) | 架构可复用，内部 Lane 换成 DSV3 |
| TP 权重切分加载 | `load_tensor_2d_row_shard` (`weight_loader.rs:131`) | bf16 column/row shard 可复用；FP8 shard 需新增 |
| NCCL AllReduce bench | `benches/nccl_bench.rs` | 已有 TP2 PCIe 数据，需补 8 卡 NVLink 数据 |
| `ModelExecutor` trait | `model_executor.rs:455-464` | DSV3 executor 实现同一 trait |
| `attach_tp_comm` 模式 | `qwen3/weights.rs:392` | 同一模式：load → attach comm → run |
| DeepGEMM FP8 kernel | `third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` | kernel 直接用；host 侧 TMA setup 已重写（`csrc/fp8_gemm.cu`），bf16 输出 |
| DeepGEMM grouped GEMM | 同上，`GemmType::GroupedContiguous` / `GroupedMasked` | MoE expert 计算直接可用 |
| FlashMLA dense decode | `third_party/FlashMLA/csrc/sm90/decode/dense/` | SM90 MLA decode kernel；host 侧 torch 胶水已重写（`csrc/flash_mla.cu`），3-phase: metadata → decode → combine |
| MLA paged KV cache | `src/model/dsv3/mla_kv.rs` | per-layer paged buffer (page_size=64)，`MlaKvPool`/`MlaKvState`，天然匹配 FlashMLA kcache 格式 |

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

DSV3 checkpoint 权重 scale 本身就是 2D block-scale `[ceil(N/128), ceil(K/128)]`，天然匹配 1D2D 的 `kMajorSFB = Major::K`。

### Kernel 模板参数（对比 1D1D）

1D2D 比 1D1D 多三个模板参数：

```
kMajorSFB        — cute::UMMA::Major::K (DSV3 权重 scale K-major) 或 Major::MN
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

## 探索方向

- **Decode 侧 SM 分区 overlap**: 双 micro-batch decode，attention 与 MoE dispatch+compute+combine 通过 SM 分区并行。RTX 5070 Ti 实验表明 32 SM (46%) 达峰值 89%。需验证 H20 的 SM-bandwidth 曲线和共享 HBM 时的干扰。

## 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-04-14 | 直接上 DSV3-0324，跳过 DSV2-Lite | 8x H20-3e 141GB 够用，不浪费时间在小模型 |
| 2026-04-14 | FP8 为必需项 | BF16 1.34TB 超过 8 卡 1.1TB 总容量 |
| 2026-04-14 | TP8 + EP8 | 8 卡环境的自然切分：attention 全卡 TP，MoE 每卡 32 experts |
| 2026-04-14 | 新增 `Fp8Matrix` 类型 | 现有 `DeviceMatrix` 是 bf16，FP8 需要 raw bytes + block-wise scale_inv 分开存 |
| 2026-04-14 | partial load 支持 | 671GB 单卡放不下，测试用 `from_safetensors_partial` 只加载前 N 层 |
| 2026-04-14 | FP8 GEMM 用 DeepGEMM | DeepSeek 自研，专为 128×128 block-scale FP8 优化，SM90 TMA+WGMMA，自带 grouped GEMM (MoE)。kernel header-only 无 torch 依赖，host 侧 torch 胶水可用 CUDA driver API 平替。TRT-LLM fp8_blockscale 底层也是 DeepGEMM，直接依赖源头更干净 |
| 2026-04-14 | DeepGEMM AOT 编译 | DSV3 矩阵尺寸已知，build.rs nvcc 预编译固定 tile config，不需要运行时 JIT |
| 2026-04-14 | DeepGEMM 编译选项 | SM90a (`-gencode=arch=compute_90a,code=sm_90a`) + C++20 + `--expt-relaxed-constexpr --expt-extended-lambda`，需要 DeepGEMM 自带 CUTLASS v2.1.1 (不可与 flashinfer 的 v4.4.2 混用) |
| 2026-04-14 | 两组 tile config | block_m=64/block_n=128/7stages (decode小M) + block_m=128/block_n=128/4stages (prefill大M)，kNumSMs 默认 132 可通过 `PEGAINFER_DG_NUM_SMS` 覆盖 |
| 2026-04-14 | activation 量化从 TRT-LLM 抽取 | flashinfer 的 `mxfp8_quantize` 是 SM100/SF_VEC=32 格式，不匹配 DeepGEMM 的 1×128 block-scale。flashinfer 自己支持 DSV3 也是靠 TRT-LLM 内嵌的 `scale_1x128_kernel`（`fp8_blockscale_gemm_kernel.cuh`）。直接抽取该 kernel 到 `csrc/fp8_quantize.cu`，去掉所有 TRT-LLM 依赖 |
| 2026-04-14 | Scale 布局 K-chunk-major | TRT-LLM `scale_1x128_kernel` 输出 dequant scale 为 `[ceil(K/128), padded(M, 4)]`（K-chunk 维在前，M 维在内），与 DeepGEMM TMA scale descriptor 期望一致。注意不是 `[M, ceil(K/128)]` |
| 2026-04-14 | GEMM 切 1D2D | SGLang（`deep_gemm_wrapper/entrypoint.py:84`）和 vLLM（`deep_gemm.py:120`）在 Hopper block-scale 主路径均走 1D2D。DeepGEMM 默认 recipe `(1, 128, 128)` 在 `gran_n != 1` 时分派到 `sm90_fp8_gemm_1d2d`（`gemm.hpp:87`），grouped/masked GEMM 在 SM90 写死 1D2D（`gemm.hpp:194,258`）。1D1D 仅用于 `gran_n==1` per-channel 和 k_grouped，不是我们的 case |
| 2026-04-14 | 1D2D AOT config: 64×128/8stages + 128×128/5stages | 依据 DeepGEMM heuristics (`sm90.hpp`)：block_m<=64 用 1 warpgroup (128 math threads)，否则 2 warpgroups (256)。stages 取 SM90 smem 容量 232448 下的最大值。kMajorSFB=K 匹配 DSV3 checkpoint scale layout，kSwizzleDMode=128 匹配 bf16 block_n=128，kNumLastStages=0 因 SM90 kernel 不使用该参数 |
| 2026-04-14 | 1D2D 输出 bf16 (非 fp32) | DeepGEMM 1D2D API 强制 `d.scalar_type() == torch::kBFloat16`。kernel 内部 fp32 accumulator → bf16 cast 后经 TMA store 写回。DSV3 forward 后续层本就需要 bf16 输入，省去显式 cast |
| 2026-04-14 | norm 权重 f32→bf16 转换 | DSV3 checkpoint 中 layernorm 权重存为 f32，而 Qwen3 系列为 bf16。`load_tensor_1d` 直接按 bf16 读会长度翻倍。新增 `load_1d_f32_as_bf16` 做显式转换 |
| 2026-04-14 | MLA attention 用 FlashMLA | DeepSeek 自研 MLA kernel（`third_party/FlashMLA`），SM90 dense decode 达 3000 GB/s / 660 TFLOPS (H800)。kernel 天然适配 DSV3 MLA 维度：`head_size_k=576` (c_KV 512 + k_R 64)，`head_size_v=512`，MQA 模式 (`h_kv=1`)。集成模式同 DeepGEMM：只取 kernel 层，host 侧 torch 胶水用 CUDA driver API 重写 |
| 2026-04-14 | KV cache 切 per-layer paged，page_size=64 | FlashMLA dense decode 硬性要求 `page_block_size=64`（`dense_decode.h:67`），且 kcache 为 per-layer 独立 buffer `[num_blocks, page_block_size, num_heads_k, head_size_k]`。现有 `KvPool` all-layers-in-one-page 布局（`kv_pool.rs`）改为 per-layer paged buffer，`PagePool`/`KvState` 分配逻辑不变，`KvLayout` 几何调整即可 |
