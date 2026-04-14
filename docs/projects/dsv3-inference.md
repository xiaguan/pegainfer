# DeepSeek-V3 推理复现

> **Status**: Phase 0 — 骨架搭建中
> **TL;DR**: 在 pegainfer 上复现 DSV3.2 (671B MoE) 8x H20-3e 推理。FP8 权重，MLA + MoE + TP8/EP8。
> **Next action**: Phase 0 — cudarc 多 GPU context + NCCL 通信初始化 + 权重 TP8/EP8 分发。

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
- [ ] cudarc 多 GPU context 管理
- [ ] NCCL 通信初始化（8 卡 NVLink 拓扑）
- [ ] 权重分发：attention TP8 切分, MoE EP8 分配, shared expert 复制

验收：权重全部上卡，显存占用符合预期。

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
