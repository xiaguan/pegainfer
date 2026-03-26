# Project: ModelForward Trait Extraction

**Status**: Active
**TL;DR**: 把 model 的 forward 语义抽成 `ModelForward` trait，将 generation loop 从两个 model 文件中提取为共享代码。设计面向 bs > 1 未来需求，weights (`&self`) 与 per-request mutable state (`State`) 分离。

---

## Problem

`model.rs` (Qwen3, 1103 行) 和 `qwen35_model.rs` (Qwen3.5, 1323 行) 有约 400 行近乎 copy-paste 的 generation orchestration 代码：

- `generate()` — take state → prefill → decode loop → put state
- `generate_streaming_with_callback()` — 同上 + 回调和提前终止
- CUDA Graph capture/replay 模板
- `select_token()`, `take_decode_bufs()`, `take_kv_cache()`, `take_graph_state()`

每加一个模型就会多复制 200+ 行。

## Design

### 调用者视角

调用者是 generation loop（现在）和 scheduler（将来 bs > 1）。它的心智模型：

```
reset state → forward(prompt) → read logits → select token
            → forward(token)  → read logits → select token
            → ...
```

调用者不知道也不该知道：prefill 和 decode 用不同的 GPU 策略、层是 full attention 还是 linear attention、有没有 CUDA Graph。

### Trait 定义

```rust
trait GenerationState {
    fn logits(&self) -> &DeviceVec;
    fn reset(&mut self) -> Result<()>;
}

trait ModelForward {
    type State: GenerationState;

    fn create_state(&self) -> Result<Self::State>;
    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()>;
    fn is_stop_token(&self, token_id: u32) -> bool;
}
```

- **一个 `forward` 方法**，不区分 prefill 和 decode。`tokens.len() > 1` → model 内部走 batch prefill；`tokens.len() == 1` → 走 decode path。这是 deep module：简单接口隐藏大量实现复杂度。
- **`&self` = weights（不可变）**，`State` = per-request 可变状态。面向 bs > 1：N 个 state 共享一份 weights。
- **Logits 从 `state.logits()` 取**。decode path logits 在 DecodeBuffers 里；prefill path logits 写入 state 的独立字段。

### State 类型

```
Qwen3State:   DecodeBuffers + KVCache + CudaGraphState + Option<DeviceVec> (prefill logits)
Qwen35State:  DecodeBuffers35 + KVCache + RecurrentState + CudaGraphState35 + Option<DeviceVec>
```

`reset()` 重置 KV cache 和 recurrent state；保留 decode buffers 分配和 captured CUDA Graph。

### 模型内部差异（trait 背后隐藏的）

| 维度 | Qwen3 | Qwen3.5 |
|---|---|---|
| 层类型 | 同构 full attention | 24 linear + 8 full attention，per-layer dispatch |
| State 组件 | KVCache | KVCache + RecurrentState |
| Norm 变体 | rms_norm | rms_norm_offset (1+w) |
| Output projection | lm_head 或 tied embeddings | 始终 tied embeddings |
| Prefill 策略 | Batched GEMM + PrefillBuffers | Per-layer + GdrChunkwiseScratch |
| Decode buffers | DecodeBuffers | DecodeBuffers35（不同中间维度） |

### 共享 generation loop

从 model impl 中提取到 `server_engine.rs`，使用 `ModelForward` trait：

```rust
fn generate<M: ModelForward>(
    model: &M, ctx: &DeviceContext, state: &mut M::State,
    prompt_tokens: &[u32], max_new_tokens: usize,
    params: &SamplingParams, rng: &mut StdRng,
) -> Result<Vec<u32>> { ... }
```

约 20 行，替代每个模型 ~120 行的重复实现。

## Model struct 变化

State 字段从 model 移出后，model struct 变成纯 weights + config：

```rust
pub struct Qwen3Model {
    ctx: DeviceContext,
    config: Config,
    embed_tokens: DeviceMatrix,
    lm_head: Option<DeviceMatrix>,
    layers: Vec<TransformerBlock>,
    norm: DeviceVec,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    enable_cuda_graph: bool,
}
// 无 decode_bufs, kv_cache, graph_state — 全部移到 Qwen3State
```

内部方法（`decode_one_token`, `decode_kernels`, `process_all_layers_batch` 等）签名不变 — 它们已经通过参数接收 mutable state。

## Next Action

实现 trait 和 state 类型，提取 generation loop，编译通过 + 测试通过。
