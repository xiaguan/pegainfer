# DeepSeek-V3 推理复现

> **Status**: Phase 0 — 骨架搭建中
> **TL;DR**: 在 pegainfer 上复现 DSV3.2 (671B MoE) 8x H20-3e 推理。FP8 权重，MLA + MoE + TP8/EP8。
> **Next action**: Phase 0.5 — activation FP8 量化 kernel + 单卡 q_a_proj forward 验证。

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
       (sm90_fp8_gemm_1d1d.cuh)
```

torch 在 DeepGEMM 中仅用于两件事（均可平替）：
- `torch::Tensor` 当 metadata accessor（读 data_ptr / stride / dtype）→ 换成 raw pointer + shape
- `torch::empty` 分配 GPU memory → 换成 cudarc `alloc_zeros`

集成步骤：

- [x] `build.rs` 加 DeepGEMM include path（`third_party/DeepGEMM/deep_gemm/include` + CUTLASS headers），SM90a + C++20 + `--expt-relaxed-constexpr`
- [x] 写 `csrc/fp8_gemm.cu`：thin C wrapper，用 `cuTensorMapEncodeTiled` 构造 TMA descriptor，AOT 编译两组 tile config（block_m=64/128, block_n=128, block_k=128）
- [x] Rust FFI 暴露 `fp8_gemm_cuda(a, scale_a, b, scale_b, d, m, n, k, stream)`
- [ ] 在线 activation FP8 量化 kernel（bf16 → fp8 + block-wise scale）
- [ ] 单卡 partial load 上验证：embedding → layer0 MLA q_a_proj → 对比 HF reference

验收：FP8 GEMM 输出与 HF fp8 dequant + matmul 对齐（误差 < 1e-2）。

### Phase 1 — MLA Forward（前 3 层 Dense 打通）

目标：MLA attention + Dense FFN 前向，前 3 层 hidden states 与 HF 对齐。

- [ ] MLA attention kernel — 压缩 KV cache (c_KV 512d + k_R 64d)
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
| DeepGEMM FP8 kernel | `third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh` | kernel 直接用；host 侧 TMA setup 需重写（去 torch） |
| DeepGEMM grouped GEMM | 同上，`GemmType::GroupedContiguous` / `GroupedMasked` | MoE expert 计算直接可用 |

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
