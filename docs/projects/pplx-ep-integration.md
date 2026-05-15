# pplx-garden EP 后端接入 dsv4-flash

**创建时间**：2026-05-15
**状态**：draft（待 review）
**当前 blocker**：本文档尚未达成共识，先 review 后再开工。

## TL;DR

把 `pegainfer-comm`（pplx-garden 派生）的 NVLink + RDMA MoE all-to-all 后端从 skeleton 接成可用实现，给 dsv4-flash **decode MoE** 提供另一条通信路径，运行时通过开关切换到它；走 pplx 时 decode CUDA Graph 全局关闭。范围只覆盖 routed expert 这一段 dispatch/combine，prefill、shared expert、attention、indexer 不动。**不**引入 trait/dyn 抽象——只有一个实现，直接用 concrete 类型。

## 工作场景

- 集群：单节点 NVLink + 多节点 RDMA（H20-3e 8 卡内 NVLink、跨节点 RoCE/IB）。
- 模型：DeepSeek V4-Flash，`n_routed_experts=256`、`num_experts_per_tok=6`，EP8 单节点已跑通。
- 目标：在不破坏 NCCL 路径的前提下，把 pplx EP 路径接入到 `moe_rank_lane_bf16_hidden` / `decode_moe_ag_rs_*` 入口，跑通 1×N decode 一致性，作为后续扩到多节点的基础。

## 现状（读码确认过的事实）

### pegainfer-comm 公共表面（skeleton）

- `EpAllToAll` trait：`dispatch / combine / poll / release` 四个 `&self` 方法，对象安全，`Send + Sync`。
- `EpBackendBuilder::build()`：**两种 feature 模式都返回 Err**——`hw-rdma` off 时 `BackendUnavailable`，`hw-rdma` on 时 `Unimplemented`。
- `EpTopology`：只带 `world_size / rank / num_experts / hidden_dim / max_num_tokens` 五个字段，`#[non_exhaustive]`。
- `DispatchPlan / CombinePlan`：极简，目前只承诺 `num_tokens / num_experts_per_token / accumulate`。
- `SendBuf / RecvBuf`：裸 device pointer + elem_count + elem_size + 可选 scale pointer；调用方持有底层 allocation 的所有权。
- `RdmaBackend`（`src/backend/rdma.rs`）：私有类型，四个 trait 方法全是 `todo!()`，构造函数当前只存了 `EpTopology`，没拿 `AllToAllContext`。

### pplx wrapper（`crates/pegainfer-comm-p2p-all-to-all/`）

- `AllToAllContext::new(...)`：21 个参数，需要外部传入 `TransferEngine`、`rank_handles`、预注册的 send/recv buffer + MR、host pointer arrays（sync/send/recv），构造时启动一个 `"p2p_all_to_all Worker"` 后台线程，固定 CPU 亲和性。
- 调用形态是 **四步**（不是 trait 现在写的两步）：
  - `dispatch_send` —— 提交 token 散发，cuda kernel + worker 协同
  - `dispatch_recv` —— 拉回 token，需要 `out_num_tokens_ptr`、tokens_per_expert 等设备元数据
  - `combine_send` —— expert 输出回程
  - `combine_recv` —— combine 完成，按 indices/weights 在 host 端 dtype 转换写回
- 内部依赖：`a2a_kernels::a2a_dispatch_send` 等 CUDA kernel 接 `fabric-lib` 的 IB Verbs / GDRCopy 路径，**有 host 侧 worker loop 参与每次 op 的进度**。

### dsv4-flash 当前 MoE 通信路径

- decode：`pegainfer-deepseek-v4/src/runtime/moe.rs:1323` 的 `decode_moe_ag_rs_bf16_hidden_with_scratch`
  - NCCL `all_gather_bf16_hidden_into`（拼全局 hidden）
  - 本地路由 + grouped FP4 GEMM（local experts）
  - NCCL `reduce_scatter_f32_hidden_into`（聚合到本地 token）
  - 数据流是 **dense AG/RS**，不是 sparse dispatch/combine。
- prefill：`moe.rs:1289` 的 `moe_rank_lane_bf16_hidden`
  - 路由 → expand → grouped GEMM → reduce → `all_reduce_f32_hidden_in_place`
  - 也是 all-reduce 形态，不是 A2A。
- 通信抽象层：`pegainfer-deepseek-v4/src/runtime/collectives.rs` 包了一组 NCCL `Comm`-based helper，所有 MoE 通信都过它。
- 没有任何 dispatch/combine 形态的接口，**需要新增**而不是替换。

### 不做的事

- **不**改 attention / indexer / compressor 的通信路径。
- **不**拆 shared expert（仍走本地 GEMM）。
- **不**接 prefill。prefill 现在的 `moe_rank_lane_bf16_hidden` 是 NCCL all_reduce，本期保持不动；理由是一次 prefill 几百~几千 token，kernel 时间长，通信占比小，先用 decode 验证一致性更划算。
- **不**追求 CUDA Graph 兼容：pplx 路径有后台 worker + host bookkeeping + 同步 flag，跟 graph capture 不兼容。决策：**走 pplx 时全局关闭 decode CUDA Graph 捕获**；想用 graph 就走 NCCL 那条路。
- **不**引入 trait/dyn 切换。只有 NCCL 和 pplx 两条路径，编译 + 配置就够分发，犯不上做 `dyn MoeBackend`。

## 设计

### 分层

```
dsv4-flash MoE (rank lane)
    │
    ├── 走 NCCL AG/RS（已有）—— CUDA Graph 友好
    │
    └── 走 pegainfer-comm（新增）—— eager only，graph 关闭
            │
            └── EpBackend → AllToAllContext → a2a-kernels + fabric-lib
```

- 切换粒度：**整 process 一致**，由 CLI/Config 决定，启动后不变；同一 layer 不会跨后端。

### pegainfer-comm 表面简化

skeleton 里的 `EpAllToAll` trait + `Box<dyn EpAllToAll>` 删掉。`EpBackend` 改成 concrete 结构，inherent 方法直接暴露四步：

```rust
impl EpBackend {
    pub fn dispatch_send(&self, ...) -> Result<...>;
    pub fn dispatch_recv(&self, ...) -> Result<...>;
    pub fn combine_send(&self, ...) -> Result<...>;
    pub fn combine_recv(&self, ...) -> Result<...>;
}
```

四步分开暴露，而不是合成 `dispatch / combine`——pplx 这个拆分本来就是给 host 一个空隙跑 shared expert / 其它计算用的，合起来就等于浪费。

`SendBuf / RecvBuf / DispatchPlan / CombinePlan / EpTopology` 这些 plain data 容器保留，但去掉 `#[non_exhaustive]` 这种 skeleton 期的过度保险——一个实现，没有"演化兼容"问题。

### dsv4 集成入口

新增 `pegainfer-deepseek-v4/src/runtime/moe_pplx.rs`（flat layout，无 `mod.rs`）：

- `decode_moe_pplx_bf16_hidden_with_scratch(ctx, config, weights, ptr_cache, ep, layer, input, token_ids, scratches)`
  - 顺序大致：`dispatch_send` → 同流跑 shared expert → `dispatch_recv` → grouped FP4 GEMM → `combine_send` → 同流跑后续 layer 准备 → `combine_recv` 写回 hidden。
  - shared expert / grouped GEMM 仍复用现有 helper。

`RankWorker` 的 MoE 调用点直接 `if let Some(ep) = &self.ep { ... } else { decode_moe_ag_rs_bf16_hidden_with_scratch(...) }`。没有枚举，没有 trait。

### scratch / buffer 所有权

pplx 路径的 send/recv buffer 必须**预注册到 fabric-lib 的 MR**，不能复用现有 `MoeAgRsScratch` 的 `CudaSlice`：

- 新增 `MoePplxScratch`，持有 send/recv buffer 的 device pointer + `MemoryRegionHandle`，按 `max_num_tokens × hidden_dim` 上限分配一次，生命周期跟 `RankWorker` 一致。
- AG/RS scratch 在 pplx 路径不需要的字段（`global_hidden / global_token_ids / partial_routed / local_routed`）就别分配，省 VRAM。
- pplx 期望 expert-major packed send buffer，dsv4 现有 `expanded_input` 也是 expert-major，但 `x_stride` 在 pplx 是 **token stride**（不是 row stride），接的时候逐项核对。

### 初始化位置

`pegainfer-deepseek-v4/src/direct.rs` 里 `RankWorker::spawn` 阶段，跟 NCCL `Comm` 同级：

```
RankWorker::spawn
    ├── set_current(device)
    ├── NCCL Comm 初始化（已有）
    ├── 当 cfg!(feature="pplx-ep") && config.moe_backend == Pplx:
    │       TransferEngine 初始化
    │       send/recv buffer 分配 + MR 注册
    │       EpBackend::build()（内部启动 worker thread + pin CPU）
    └── 进入 worker main loop
```

控制平面（rank handles 交换）走现有 NCCL bootstrap 路径同样的 rendezvous 方式即可，**不**新引入 process 级别的初始化阶段。

### 运行时切换

- 编译期：`pegainfer-comm` 的 `hw-rdma` feature 已经存在；dsv4 加一个 `pplx-ep` feature，关掉时 `moe_pplx.rs` 整个 `cfg`-out，不拉 fabric-lib 依赖。
- 运行时：`Config` 加 `moe_backend: MoeBackend { Nccl, Pplx }`，CLI `--moe-backend nccl|pplx`，默认 `nccl`。选 `pplx` 时：
  - `pplx-ep` feature 必须开，否则启动报错。
  - decode CUDA Graph 自动关闭（不需要用户单独传参）。

## 分步实施计划

### Step 0：本文档 review ← **当前位置**

确认 scratch/buffer 形态、初始化位置、CLI 入口。

### Step 1：pegainfer-comm 去 skeleton，砍 trait

- 删 `EpAllToAll` trait 与 `Box<dyn EpAllToAll>`，`EpBackend` 改 concrete + inherent 四步方法。
- 改 `EpBackendBuilder::build()`：`hw-rdma` 分支真正构造 `AllToAllContext`。
- `EpBackendBuilder` 补足 `AllToAllContext::new` 需要的参数（`transfer_engine / rank_handles / send_recv_buffers / imm_base / dp_size / node_size / ...`）。
- 写一个 single-node 2-rank 的集成测试，loopback 验证 dispatch_send→recv 与 combine_send→recv 数据正确。

### Step 2：dsv4 加 pplx 路径

- `pegainfer-deepseek-v4/src/runtime/moe_pplx.rs` 写 `decode_moe_pplx_bf16_hidden_with_scratch`，路由 / grouped GEMM / shared expert 复用现有 helper，只把 AG/RS 替换成 dispatch_send→recv + combine_send→recv。
- 新增 `MoePplxScratch`，跟 `MoeAgRsScratch` 同级，按 `cfg(feature="pplx-ep")` 在 `RankWorker` 里二选一持有。
- `RankWorker` MoE 调用点加 `if let Some(ep) = &self.ep { ... } else { 现有 NCCL 路径 }`。
- `Config` / CLI 加 `--moe-backend`，pplx 时强制关 decode CUDA Graph。

### Step 3：1×N decode 一致性

- 同一 prompt 在 NCCL 与 pplx 两条路径下，logits 差异在已接受的数值阈值内（不要求 bit-level，因为 reduce 顺序不同）。
- dsv4 现有 e2e 20 例 golden 对照。

### Step 4：decode 性能 + overlap profiling

- nsys decode trace，看 `dispatch_send` 后 host 空隙是否被 shared expert 填满，`dispatch_recv` 等待是否把 grouped GEMM 推到了通信之后。
- 决定后续是否值得调 `combine_send/recv` 与下一层 attention 准备的 overlap。

### Step 5（后续，本文档外）

- 跨节点 RDMA 真正跑起来。
- prefill 是否替换。
- KV transfer / PD handoff（见 `docs/projects/deepseek-v4/prefix-paged-kv-pd-handoff.md`）。

## 风险 / Open Questions

1. **fabric-lib 编译依赖**：`hw-rdma` / `pplx-ep` 开启时会拉 `libibverbs-sys` / `gdrapi-sys`，CI 默认 lane 仍要 `cargo check` 通过——skeleton 已经做了 default-off 的保护，开起来后要确认 link OK，以及在没 RDMA 硬件的开发机上能跳过运行时探测。
2. **数据布局对齐**：dsv4 现有 `MoeAgRsScratch::expanded_input` 是 BF16 expert-major packed，跟 pplx 的 send buffer 在 stride 约定上要逐项核对（pplx 的 `x_stride` 是 **token stride**，不是 row stride；`x_scale_stride_*` 也分 elem/token 两个 stride，FP8 路径不开就传 0）。
3. **rank handles 怎么交换**：`AllToAllRankHandle::address` 需要在所有 rank 之间互换才能 setup peer 连接。复用现有 NCCL bootstrap 那套机制即可，但具体怎么挂上去要在 Step 1 落地时确定（最简单是 NCCL all-gather string + rank0 集中分发）。
4. **buffer 容量上限**：`max_num_tokens` 决定 send/recv buffer 一次性预留多少 VRAM，需要按现有 decode 的最大 batch 上界算清楚——单 rank decode 当前 bs=1，buffer 不大；但要为 bs>1 的 continuous batching 预留接口（先按当前上界，留 TODO）。

## 不在范围内

- attention / indexer / compressor 的通信
- shared expert 拆分
- CUDA Graph 兼容（走 pplx 时全局关闭）
- prefill 路径替换（仍 NCCL all_reduce）
- KV transfer / PD handoff（见 `docs/projects/deepseek-v4/prefix-paged-kv-pd-handoff.md`）

## 当前进度（2026-05-16）

**已落地**
- `pegainfer-comm` 去 skeleton + 砍 trait：`EpBackend` 是 concrete struct，四步 `dispatch_send / dispatch_recv / combine_send / combine_recv` 是 inherent method，构造走 `EpBackend::new(EpBackendParams)`，外加 `tokens_per_expert_ptr()` 让下游 grouped GEMM 拿 per-expert 计数。`unsafe impl Send` 让 EpBackend 可以从外部线程移交进 RankWorker。
- **砍掉 LibTorch 依赖**：a2a-kernels 自己定义 cxx `ScalarType` enum（namespace 改 `a2a_kernels::`），a2a-kernels 与 p2p-all-to-all 的 `torch-lib` dep 全部移除。`hw-rdma` feature 现在只需要 CUDA + RDMA Verbs + GDRCopy，不再拉 LibTorch / pyo3。
- dsv4 加 `pplx-ep` feature，optional 依赖 `pegainfer-comm/hw-rdma`。
- `runtime/moe_pplx.rs`：`decode_moe_pplx_bf16_hidden_with_scratch` body 完整——本地 route → dispatch_send → shared expert（overlap） → dispatch_recv → host 端 prefix-sum 出 expert_indptr → grouped FP4 expert → combine_send → combine_recv（`accumulate=true`，把 shared expert 折进 routed 输出）。
- `state.rs` 加 `MoePplxScratch`（MR-recv buffer 还是占位，要在 bootstrap 阶段注册）+ `MoeRunContext` / `MoePplxRunContext` 把两条 MoE 路径统一成一个参数。
- `block_decode_rank_lane_bf16_hidden_with_scratch`（含 batch 变体）签名改成 `moe: &mut MoeRunContext<'_>`，内部 `dispatch_decode_moe_step` 按 `moe.pplx.is_some()` 分发到两条路径。
- `RankWorker` 新增 `RankCommand::EnablePplx { ep_backend }`；`DeepSeekV4DirectGenerator::enable_pplx(Vec<EpBackend>)` 把 per-rank 后端塞进对应 worker。
- `cargo check -p pegainfer-comm` 通过。dsv4 因为 pegainfer-kernels 在本机 CUDA/flashinfer SDK 缺失编译不了（pre-existing），结构性 review 看 diff。

**剩下的全是机器侧的事**
1. **Bootstrap**：用户在自己的 entry binary 里造 `Vec<EpBackend>`——`fabric_lib::TransferEngine` 初始化、跨 rank 交换 `AllToAllRankHandle`、用 `cuda-lib::CudaDeviceMemory` 分配 send/recv buffer、`fabric_lib::MemoryRegionHandle` 注册 MR，然后 `EpBackend::new(EpBackendParams { ... })`。这套 rendezvous 跟 pplx-garden 的 Python 前端做的是一回事；可以复用 NCCL bootstrap 通道交换 rank-handle 字节。
2. 调 `generator.enable_pplx(backends)` 一次（在 prefill 之前），后续 decode 自动走 pplx 分发。
3. 跑 dsv4 e2e 20 例做一致性对照（注意 combine_recv 的 BF16 accumulate 与 AG/RS 路径的 F32 累加在数值上有差，可能需要放宽 atol）。
4. nsys profile，看 dispatch_send → shared expert → dispatch_recv 之间的 host 间隙是否被填满；用此调 expert_indptr 上 host prefix-sum 的 sync 开销是否需要换 GPU kernel。

CUDA Graph：dsv4 decode 当前**没有**开 graph capture（`start_engine` 里 `enable_cuda_graph=true` 只是一行 warn），所以这次接入不需要额外关闭逻辑。
