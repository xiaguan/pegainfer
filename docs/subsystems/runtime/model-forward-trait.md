# Project: ModelForward Trait Extraction

**Status**: Active
**TL;DR**: Extract `ModelForward` trait from model forward semantics. Lift the generation loop out of two model files into shared code. Designed for future bs > 1: weights (`&self`) separated from per-request mutable `State`.

---

## Problem

`model.rs` (Qwen3, 1103 lines) and `qwen35_model.rs` (Qwen3.5, 1323 lines) contained ~400 lines of near-identical generation orchestration:

- `generate()` — take state → prefill → decode loop → put state
- `generate_streaming_with_callback()` — same + callback and early termination
- CUDA Graph capture/replay boilerplate
- `select_token()`, `take_decode_bufs()`, `take_kv_cache()`, `take_graph_state()`

Every new model would copy 200+ lines.

## Design

### Caller's perspective

The caller is the generation loop (today) and the scheduler (future bs > 1). Its mental model:

```
reset state → forward(prompt) → read logits → select token
            → forward(token)  → read logits → select token
            → ...
```

The caller does not and should not know: that prefill and decode use different GPU strategies, whether a layer is full attention or linear attention, or whether CUDA Graph is involved.

### Trait definition

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

- **Single `forward` method** — no prefill/decode distinction at the interface. `tokens.len() > 1` → batch prefill internally; `tokens.len() == 1` → decode path. This is a deep module: simple interface hiding substantial implementation complexity.
- **`&self` = weights (immutable)**, `State` = per-request mutable state. Designed for bs > 1: N states share one set of weights.
- **Logits read from `state.logits()`**. Decode-path logits live in DecodeBuffers; prefill-path logits are written to a separate field on the state.

### State types

```
Qwen3State:   DecodeBuffers + KVCache + CudaGraphState + Option<DeviceVec> (prefill logits)
Qwen3.5:      direct `ModelForward` state retired; production and tests go through the scheduler / batch APIs
```

`reset()` semantics now matter only for the Qwen3 direct path.

### Model-internal differences (hidden behind the trait)

| Dimension | Qwen3 | Qwen3.5 |
|---|---|---|
| Layer type | Homogeneous full attention | 24 linear + 8 full attention, per-layer dispatch |
| State components | KVCache | KVCache + RecurrentState |
| Norm variant | rms_norm | rms_norm_offset (1+w) |
| Output projection | lm_head or tied embeddings | Always tied embeddings |
| Prefill strategy | Batched GEMM + PrefillBuffers | Per-layer + GdrChunkwiseScratch |
| Decode buffers | DecodeBuffers | DecodeBuffers35 (different intermediate dimensions) |

### Shared generation loop

Extracted from model impls into `server_engine.rs`, using the `ModelForward` trait:

```rust
fn generate<M: ModelForward>(
    model: &M, ctx: &DeviceContext, state: &mut M::State,
    prompt_tokens: &[u32], max_new_tokens: usize,
    params: &SamplingParams, rng: &mut StdRng,
) -> Result<Vec<u32>> { ... }
```

~20 lines, replacing ~120 lines of duplicated code per model.

## Model struct changes

After moving state fields out, the model struct becomes pure weights + config:

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
// No decode_bufs, kv_cache, graph_state — all moved to Qwen3State
```

Internal methods (`decode_one_token`, `decode_kernels`, `process_all_layers_batch`, etc.) keep their signatures unchanged — they already receive mutable state via parameters.

## Next Action

Trait and state types implemented, generation loop extracted. Compiles and passes all tests. Ready for review.
