# DeepSeek V4 Decode Scratch

**Created**: 2026-05-12
**Status**: complete

## TL;DR

DeepSeek V4 decode hot-path allocation scratch work is complete for the current direct decode path. Token-id, entry embedding/HC-expand, HC pre/post, shared-expert, attention projection/output/index, ratio-4 attention aux, final logits-boundary, MoE AG/RS collective-boundary, MoE route/workspace capacity buffers, and grouped FP4 quant workspace now use reusable rank-owned storage. Forced NUMA-aware rank-worker pinning is enabled and derives the NUMA node from each CUDA device's PCI bus id through the CUDA driver API plus sysfs, with no silent fallback. Latest forced NUMA fixed bench is `34.34-35.36ms/token`, with repeated same-code runs at `32.75-33.90ms/token`; all runs keep hash `6346f03343d75a65`. Allocation proof now covers direct runtime symbols and CUDA driver function-table lookup: baseline `cudaMalloc/cudaFree = 12944/12848` fell to current `136/32`; `cuMemAllocAsync/cuMemFreeAsync/cuMemsetD8Async`, `cudaMallocAsync/cudaFreeAsync/cudaMemsetAsync`, and `cuGetProcAddress` are all `0` in the current fixed bench. The earlier nsys `cuMemAllocAsync` attribution is treated as profiler/runtime-internal noise, not an application-visible hot-path API.

## Preparation

- **Read**:
  - `docs/index.md` - confirmed the completed sub-70 decode work and related DeepSeek V4 optimization docs.
  - `docs/projects/deepseek-v4-sub80-decode.md` - confirmed the last retained changes: rank-worker affinity plus uninitialized fully-overwritten temporaries; post-uninit f32 all-reduce skew did not materially improve, so remaining wins should target launch/allocation structure.
  - `pegainfer-deepseek-v4/src/direct/worker.rs` - decode entry still allocates token-id storage with `clone_htod(&[token_id])` per token.
  - `pegainfer-deepseek-v4/src/runtime/*.rs` - allocation inventory shows many return-new-buffer operators in decode; MoE plan construction still owns dynamic metadata storage and zeroed counters.
- **Relevant history**:
  - `docs/projects/deepseek-v4-sub80-decode.md` - removing unnecessary zero-fill moved fixed long decode to `63.35-64.51ms/token`, proving allocation/API overhead is still a real bucket.
  - NUMA-aware worker placement regressed because the current decode path is host-launch/API heavy; reducing per-token CUDA allocator/API work should come before revisiting PCIe-local NUMA placement.
- **Plan**:
  1. Build a decode allocation inventory from `rg "alloc|alloc_zeros|clone_htod"` and classify each site as fixed-shape scratch, dynamic-shape scratch, semantic-zero state, or non-decode/prefill-only.
  2. Add a small per-rank `RankDecodeScratch` in the direct worker for the decode token-id buffer, replacing per-token `clone_htod(&[token_id])` with `memcpy_htod` into reusable device storage.
  3. Extend scratch into non-MoE fixed-shape buffers next: HC pre/post intermediate f32 buffers, logits/all-gather scratch, and common hidden outputs where writer kernels fully overwrite storage.
  4. Leave MoE EP metadata/workspace as a separate phase: preallocate route weights/indices, expert counts/indptr/cursors, compact maps, expanded input/output, and active tile lists by capacity; keep explicit zeroing only for counters/state that require it.
  5. Gate each slice with local `cargo fmt --check`, local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`, 5090 exact E2E, and fixed long bench `--prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42`.
- **Risks / open questions**:
  - Borrowing reusable scratch through existing return-new-buffer APIs can get awkward; the clean design may require `*_into` variants for hot operators.
  - CUDA graph pointer stability improves only after enough operator outputs stop allocating.
  - MoE needs careful capacity accounting because active experts and tile counts are dynamic even when storage is static.

## Execution Log

### Step 1: initial allocation inventory

- Current high-confidence decode allocation sites:
  - `pegainfer-deepseek-v4/src/direct/worker.rs`: per-token `clone_htod(&[token_id])`.
  - `pegainfer-deepseek-v4/src/runtime/core.rs`: HC mixes/pre/post/comb, HC output, embedding output, norm output, linear outputs, SwiGLU output, residual add output, logits output.
  - `pegainfer-deepseek-v4/src/runtime/collectives.rs`: BF16 all-gather output, u32 all-gather output, f32 reduce-scatter output, f32 all-reduce scratch, HC-post all-reduce output.
  - `pegainfer-deepseek-v4/src/runtime/attention_base.rs`: top-k index output buffers.
  - `pegainfer-deepseek-v4/src/runtime/indexer.rs` and `pegainfer-deepseek-v4/src/runtime/compressor.rs`: decode score/top-k/compressor scratch remains allocated in helper calls.
  - `pegainfer-deepseek-v4/src/runtime/moe.rs`: route outputs, compact maps, expert indptr/cursor/counts, expanded input/output, grouped FP4 outputs.
- Current classification:

| Class | Examples | Direction |
| --- | --- | --- |
| Fixed-shape non-MoE scratch | token id, hidden outputs, HC intermediate f32 buffers, logits shard | move into per-rank scratch first |
| Dynamic-shape but bounded scratch | top-k indices, compressor/indexer scores, all-gather/reduce-scatter outputs | preallocate by max decode capacity |
| Semantic-zero state | MoE expert cursors/counts, mapping counters, cache reset buffers | keep explicit clear, but avoid storage allocation |
| MoE dynamic content | route weights/indices, expert indptr, active experts/tile list | separate EP scratch phase |
| Prefill/cache setup | rope/cache allocation, prompt token upload | not first-priority steady decode work |

### Step 1a: steady decode allocation list

Scope: current direct rank-lane decode path:

`run_decode_on_rank_lane` -> embedding + embedding all-reduce -> HC expand -> `config.n_layers` x `block_decode_rank_lane_bf16_hidden_with_scratch` -> final local logits -> logits all-gather.

This table is the working backlog. "Dynamic" below means dynamic storage allocation, not dynamic content. MoE routing/counts are dynamic content, but their storage should still move to capacity-based scratch.

| Priority | Stage | Allocation site | Frequency | Current storage | Zero required? | Plan |
| --- | --- | --- | ---: | --- | --- | --- |
| Done | decode token id | `direct/worker.rs::RankDecodeScratch::new` | worker startup | `CudaSlice<u32>[1]` | no | Already moved from per-token `clone_htod(&[token_id])` to reusable per-rank buffer. |
| Done | embedding | `core.rs::embedding_rank_local_into` | 1/token/rank | `Bf16HiddenStates[config.dim, capacity]` | no, kernel overwrites | Moved to `DecodeEntryScratch::embedding`; prefill remains on the owned path. |
| Done | HC expand | `core.rs::hc_expand_bf16_hidden_into` | 1/token/rank | `HcHiddenStates[config.dim, capacity, hc]` | no, kernel overwrites | Moved to `DecodeEntryScratch::hc_expand`; layer 0 borrows this scratch as input, later layers still return owned HC outputs. |
| Done | HC pre-norm attention | `core.rs::hc_pre_norm_bf16_hidden_scratch` | 1/layer/token/rank | `mixes: f32[capacity * mix_hc]` | no, kernel overwrites | Moved to `HcPreNormScratch`, reused by attention and FFN HC pre-norm in decode. |
| Done | HC pre-norm attention | `core.rs::hc_pre_norm_bf16_hidden_scratch` | 1/layer/token/rank | `post: f32[capacity * hc]` | no, kernel overwrites | Moved to scratch-backed `HcPreStateView`; lifetime extends until HC post. |
| Done | HC pre-norm attention | `core.rs::hc_pre_norm_bf16_hidden_scratch` | 1/layer/token/rank | `comb: f32[capacity * hc * hc]` | no, kernel overwrites | Moved to scratch-backed `HcPreStateView`; lifetime extends until HC post. |
| Done | HC pre-norm attention | `core.rs::hc_pre_norm_bf16_hidden_scratch` | 1/layer/token/rank | `Bf16HiddenStates[dim, capacity]` | no, kernel overwrites | Moved to `HcPreNormScratch::out`; metadata `seq_len` is set to the logical decode step length. |
| Done | attention projections | `attention_base.rs::attention_project_bf16_hidden_scratch` via `*_into` projection/norm kernels | 1/layer/token/rank | `qr_raw`, `qr_norm`, `q_raw`, `q`, `kv_raw`, `kv` hidden buffers | no, kernels overwrite | Moved active ratio 0/4 decode paths to `AttentionProjectionScratch` and borrowed `AttentionProjectionsView`; owned API remains for prefill/non-scratch callers. |
| Done | raw attention top-k | `attention_base.rs::window_topk_indices_decode_into` | 1/layer/token/rank | `i32[sliding_window]` | no, kernel overwrites | Moved to `AttentionIndexScratch::window_idxs`; callers pass logical `topk` separately. |
| Done | indexed attention output | `attention.rs::indexed_attention_cache_bf16_hidden_into` | 1/layer/token/rank | `Bf16HiddenStates[q_hidden_dim, 1]` | no, kernel overwrites | Moved to `AttentionOutputScratch::attn_out` for ratio 0 and ratio 4 active decode paths. |
| Done | attention output projection low-rank | `attention.rs::attention_output_project_bf16_hidden_scratch` via `core.rs::bf16_linear_bf16_hidden_into` | 1/layer/token/rank | `Bf16HiddenStates[o_lora_rank, 1]` | no, kernel overwrites | Moved to `AttentionOutputScratch::low_rank`. |
| Done | attention output projection final | `attention.rs::attention_output_project_bf16_hidden_scratch` via `core.rs::bf16_linear_bf16_hidden_into` | 1/layer/token/rank | `Bf16HiddenStates[dim, 1]` | no, kernel overwrites | Moved to `AttentionOutputScratch::out`, then borrowed by HC post all-reduce. |
| Done | ratio-4 compressor output | `compressor.rs::compressor_overlap_decode_bf16_hidden_with_dim_scratch` | only when `(start_pos + 1) % 4 == 0` for ratio-4 layers | `weighted: f32[head_dim]`, `Bf16HiddenStates[head_dim, 1]` | no, kernel overwrites when present | Moved to `AttentionAuxScratch`; reused for main attention compressor and indexer compressor with logical `head_dim`. |
| Done | ratio-4 indexer q path | `indexer.rs::indexer_scores_decode_bf16_hidden_scratch` via `fp8_linear_bf16_hidden_into`, compressor scratch, and `bf16_linear_bf16_hidden_into` | ratio-4 layers/token/rank | q/intermediate/weights hidden buffers | no, kernels overwrite | Moved to `AttentionAuxScratch::{indexer_q,indexer_weights}` plus shared compressor scratch. |
| Done | ratio-4 indexer scores | `indexer.rs::indexer_scores_decode_bf16_hidden_scratch` | ratio-4 layers when compressed_len > 0 | `f32[compressed_len]` | no, kernel overwrites | Moved to `AttentionAuxScratch::indexer_scores`; NCCL all-reduce uses a `slice_mut(0..compressed_len)` logical prefix. |
| Done | ratio-4 indexer top-k | `indexer.rs::indexer_topk_indices_decode_into` | ratio-4 layers when compressed_len > 0 | `i32[min(index_topk, compressed_len)]` | no, kernel overwrites | Moved to `AttentionIndexScratch::compress_idxs`; capacity is `index_topk`, logical top-k may be smaller. |
| Done | top-k concat | `compressor.rs::concat_topk_indices_into` | compressed layers with compressed_len > 0 | `i32[window_topk + compress_topk]` | no, kernel overwrites | Moved to `AttentionIndexScratch::topk_idxs`; indexed attention now accepts capacity buffers and reads only logical `topk`. |
| Done | attention all-reduce scratch | `collectives.rs::all_reduce_hidden_fp32_hc_post_view_into` | 1/layer/token/rank | `f32[dim * capacity]` | no, conversion kernel overwrites | Moved from implicit TLS allocation to `HcPostScratch::attention_reduce_temp`. |
| Done | attention HC post output | `collectives.rs::all_reduce_hidden_fp32_hc_post_view_into` | 1/layer/token/rank | `HcHiddenStates[dim, capacity, hc]` | no, kernel overwrites | Moved to `HcPostScratch::attention_out`. |
| Done | HC pre-norm FFN | `core.rs::hc_pre_norm_bf16_hidden_scratch` | 1/layer/token/rank | same four buffers as attention HC pre-norm | no | Reuses the same scratch slots after attention HC post consumes attention pre-state. |
| Done | MoE hidden all-gather | `collectives.rs::all_gather_bf16_hidden_into` | 1/layer/token/rank | `Bf16HiddenStates[dim, capacity * world_size]` | no, NCCL overwrites | Moved to `MoeAgRsScratch::global_hidden`. |
| Done | MoE token all-gather | `collectives.rs::all_gather_u32_into` | hash-routed layers/token/rank | `u32[capacity * world_size]` | no, NCCL overwrites | Moved to `MoeAgRsScratch::global_token_ids`; used only for hash-routed layers. |
| Done / watch | MoE route weights/indices | `moe.rs::hash_route_bf16_hidden_into` / `score_route_bf16_hidden_into` | 1/layer/token/rank | `weights: f32[global_capacity * topk]`, `indices: i32[global_capacity * topk]` | no, route kernels overwrite | Moved to `MoeAgRsScratch`; dynamic values, static capacity. Latest bench band is higher than best final-boundary run, so keep under performance watch. |
| Done / watch | MoE local mapping | `moe.rs::build_moe_fused_route_plan_into` | 1/layer/token/rank | `pos_to_token`, `pos_to_token_topk`, `token_topk_to_pos`, `expert_indptr`, `expert_cursor`, `local_count` | yes, mapping kernel clears pos maps to `-1` and counters/cursors/indptr to `0` | Moved to `MoeAgRsScratch`; no host D2H count path. Semantic zero remains inside `deepseek_moe_local_mapping_cuda`. |
| Done / watch | MoE expanded input | `moe.rs::expand_moe_fused_input_into` | 1/layer/token/rank | `Bf16HiddenStates[dim, global_capacity * topk]` | no, kernel overwrites | Moved to `MoeAgRsScratch::expanded_input`. |
| Done / watch | MoE grouped expert outputs | `moe.rs::local_experts_forward_packed_bf16_hidden_scratch` via grouped FP4 linears and activation | 3 linears + activation per MoE layer/token/rank | intermediate hidden buffers for w1/w3/activation/w2 | no, kernels overwrite | Moved to `MoeAgRsScratch::{expert_gate, expert_up, expert_activated, expert_out}`. |
| Done | MoE grouped FP4 quant workspace | `deepseek_moe_fp4_grouped_linear_with_workspace_cuda` | 3 grouped FP4 linears per MoE layer/token/rank | act quant bytes and act scale bytes | no, TileLang act quant overwrites | Moved from C-side file-static quant scratch/mutex path to `MoeAgRsScratch::{fp4_act_workspace,fp4_act_scale_workspace}` for the scratch decode path. |
| Done / watch | MoE reduce output | `moe.rs::reduce_moe_fused_output_f32_into` | 1/layer/token/rank | `F32HiddenStates[dim, global_capacity]` | no, kernel overwrites | Moved to `MoeAgRsScratch::partial_routed`. |
| Done | MoE reduce-scatter output | `collectives.rs::reduce_scatter_f32_hidden_into` | 1/layer/token/rank | `F32HiddenStates[dim, local_capacity]` | no, NCCL overwrites | Moved to `MoeAgRsScratch::local_routed`. |
| Done | shared expert MLP | `core.rs::shared_expert_forward_bf16_hidden_scratch` via `fp8_linear_bf16_hidden_into` and `swiglu_clamp_bf16_hidden_into` | 3 linears + activation per layer/token/rank | gate/up/activated/out hidden buffers | no, kernels overwrite | Moved to `SharedExpertScratch`, used only by decode MoE AG/RS path; routed MoE dynamic content remains unchanged. |
| Done | routed + shared add | `moe.rs::add_f32_bf16_to_bf16_hidden_into` | 1/layer/token/rank | `Bf16HiddenStates[dim, local_capacity]` | no, kernel overwrites | Moved to `MoeAgRsScratch::out`; block HC post borrows this output. |
| Done | FFN HC post output | `core.rs::hc_post_bf16_hidden_view_into` | 1/layer/token/rank | `HcHiddenStates[dim, capacity, hc]` | no, kernel overwrites | Moved to two-slot `HcPostScratch::layer_outputs` ping-pong storage so adjacent layers never alias input/output. |
| Done | final HC head | `core.rs::hc_head_bf16_hidden_into` | 1/token/rank | `mixes: f32[capacity * hc]`, `pre: f32[capacity * hc]`, `Bf16HiddenStates[dim, capacity]` | no, kernels overwrite | Moved to `FinalLogitsScratch`; retained only when combined with the whole final logits boundary. |
| Done | final head norm | `core.rs::rms_norm_bf16_hidden_into` inside `final_logits_rank_local_bf16_hidden_into` | 1/token/rank | `Bf16HiddenStates[dim, capacity]` | no, kernel overwrites | Moved to `FinalLogitsScratch::normed`. |
| Done | local logits | `core.rs::rank_local_logits_from_hidden_into` | 1/token/rank | `F32Logits[local_vocab]` | no, kernel overwrites | Earlier isolated logits scratch was rejected; the complete final-boundary slice is retained at `33.87-36.06ms/token`. |
| Done | gathered logits | `core.rs::all_gather_logits_into` | 1/token/rank | `F32Logits[vocab]` | no, NCCL overwrites | Earlier isolated logits scratch was rejected; the complete final-boundary slice is retained. Rank0 D2H remains the sampling boundary. |

Decode-capable but not expected on the current DeepSeek V4 ratio set:

| Stage | Allocation site | Why not first |
| --- | --- | --- |
| non-overlap compressed attention | `attention.rs::attention_decode_compressed_nonoverlap_rank_local_bf16_hidden` | Current hot path is ratio `0` and ratio `4`; keep this path correct but do after active paths. |
| non-overlap compressor decode | `compressor.rs::compressor_nonoverlap_decode_bf16_hidden` | Same reason; scratch design should cover it later with the same optional weighted/out buffers. |
| `compress_topk_indices_decode` | `attention_base.rs::compress_topk_indices_decode` | Used by non-overlap compressed decode, not the current ratio-4 indexer path. |

Explicitly excluded from steady decode allocation work:

| Site | Reason |
| --- | --- |
| `attention_base.rs::precompute_rope_cache` | Worker cache setup, not per-token steady decode. |
| `state.rs::{Bf16Cache, CompressorDecodeState, LayerDecodeCache}` constructors | Cache allocation/lifecycle should move under scheduler policy later, but not a steady decode token allocation. |
| `direct/worker.rs::run_prefill_on_rank_lane` prompt `clone_htod(prompt_tokens)` | Prefill path, not decode steady path. |
| `direct/worker.rs::{zero_cuda_slice, fill_f32_cuda_slice}` | Cache reset/setup; keep semantic clear separate from allocation cleanup. |
| `moe.rs::build_moe_expert_ptr_cache` pointer `clone_htod` | Worker startup cache, not per-token decode. |
| prefill attention/indexer/compressor helpers | Prefill path; scratching decode first keeps scope controlled. |

Recommended order:

1. Keep the worker-level token-id scratch.
2. Add explicit per-rank scratch slots for HC pre/post outputs and f32 HC pre-state, because they happen twice per layer and have fixed decode shape.
3. Add attention projection/top-k/indexed-attention scratch for ratio 0 and ratio 4 active paths. Active ratio 0/4 attention projection, indexed-attention output, output projection, window top-k, ratio-4 compressor/indexer aux buffers, ratio-4 indexer top-k, and concat top-k are done.
4. Add shared expert MLP scratch, which is fixed-shape and independent of routed EP metadata.
5. Move MoE AG/RS metadata/workspace into capacity scratch, preserving explicit zeroing for counters/cursors/indptr.
6. After enough outputs are scratch-owned, revisit CUDA graph capture/pointer stability and NUMA placement.

### Step 2: reusable decode token-id buffer

- Added `RankDecodeScratch` in `pegainfer-deepseek-v4/src/direct/worker.rs`.
- Each rank worker now allocates a single device `CudaSlice<u32>` during worker startup.
- Decode updates that buffer with `memcpy_htod(&[token_id], ...)` instead of allocating a fresh device slice through `clone_htod(&[token_id])` every token.
- This is intentionally small:
  - It removes one per-rank per-token CUDA allocation from the decode entry.
  - It does not change request shape, batching semantics, MoE routing, or operator math.
  - It establishes the per-rank scratch ownership pattern before larger `*_into` operator changes.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- 5090 validation:
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
  - Fixed-token bench command:
    - `cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42`
  - Round 1: `steady_tpot_ms.avg = 60.392635`, p50 `59.647041`, p95 `63.891137`, p99 `64.974255`, hash `6346f03343d75a65`.
  - Round 2: `steady_tpot_ms.avg = 63.281963`, p50 `62.509018`, p95 `67.054209`, p99 `67.988933`, hash `6346f03343d75a65`.
- Unexpected:
  - A first remote sync attempt used full-repo `rsync -az --delete --exclude target --exclude .git ...` and stalled in rsync for about 10 minutes before being killed. For this repo, remote test syncs should use a touched-file list with `rsync -azR` unless a full workspace refresh is explicitly needed.

### Step 3: inconclusive local/gathered logits scratch

- Attempt:
  - Added `rank_local_logits_from_hidden_into`, `final_logits_rank_local_bf16_hidden_into`, and `all_gather_logits_into`.
  - Extended `RankDecodeScratch` with preallocated local logits and gathered logits buffers.
  - Decode wrote final local logits into scratch, then NCCL all-gathered into scratch before rank0 D2H sampling.
- Why it looked reasonable:
  - The logits kernels/NCCL fully overwrite their outputs.
  - The slice removes two per-token output allocations at the sampling boundary.
  - It does not alter token ids, request shape, MoE routing, or math.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- 5090 correctness:
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed-token bench:
  - Round 1: `steady_tpot_ms.avg = 69.704037`, p50 `68.856804`, p95 `73.525524`, hash `6346f03343d75a65`.
  - Round 2: `steady_tpot_ms.avg = 61.865297`, p50 `61.129638`, p95 `65.252542`, hash `6346f03343d75a65`.
  - Round 3: `steady_tpot_ms.avg = 65.381488`, p50 `64.499136`, p95 `69.048344`, hash `6346f03343d75a65`.
- Decision:
  - Mark this slice inconclusive and keep it reverted for now.
  - The code change is theoretically neutral-to-positive: it removes two per-token storage allocations, does not add kernels/sync, and does not change math. The observed TPOT movement is therefore weak evidence by itself.
  - Revisit logits scratch after the benchmark variance investigation below, and only keep it with profile evidence showing `cuMemAllocAsync/cuMemFreeAsync` reduction plus no clear median/trimmed-mean regression.

### Step 4: benchmark stability check

- Reason:
  - The logits scratch slice produced correct tokens but noisy TPOT. Because the slice only changes storage ownership at the sampling boundary, a single-process or two-process fixed bench is not enough to conclude that the slice caused a real slowdown.
  - The fixed command remains the acceptance command, but small allocation slices need extra evidence.
- Re-established remote code state:
  - The 5090 workspace was synced back to token-id scratch only.
  - `rg` confirmed there are no `rank_local_logits_from_hidden_into`, `final_logits_rank_local_bf16_hidden_into`, or `all_gather_logits_into` definitions in the current code.
- Four fixed-command processes on token-id-only code:

| Log | avg | p50 | p95 | p99 | max | first decode | hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `/tmp/dsv4_bench_stability_fixed_1.log` | `61.099432` | `60.376053` | `64.802595` | `65.774030` | `67.089848` | `56.738065` | `6346f03343d75a65` |
| `/tmp/dsv4_bench_stability_fixed_2.log` | `71.502624` | `70.639244` | `75.497320` | `76.920276` | `78.263084` | `66.378303` | `6346f03343d75a65` |
| `/tmp/dsv4_bench_stability_fixed_3.log` | `65.107777` | `64.347618` | `68.618779` | `70.021786` | `70.988918` | `60.277474` | `6346f03343d75a65` |
| `/tmp/dsv4_bench_stability_fixed_4.log` | `66.391150` | `65.591206` | `70.168230` | `71.550711` | `72.757244` | `61.498745` | `6346f03343d75a65` |

- Fixed-command spread:
  - Avg of avgs: `66.025246ms`.
  - Min/max avg: `61.099432ms` / `71.502624ms`.
  - Process-level avg spread: `10.403192ms`.
- Longer single-process probe:
  - Command changed only warmup/iters: `--prompt-len 1 --output-len 160 --warmup 4 --iters 8 --seed 42`.
  - `/tmp/dsv4_bench_stability_longiters1.log`: avg `67.791737`, p50 `66.929738`, p95 `71.599330`, p99 `72.734634`, max `74.635324`, samples `1264`, first decode `62.715692`, hash `6346f03343d75a65`.
- Interpretation:
  - Token hash is stable, so token-dependent EPLB/routing is not the source of these run-to-run swings for this prompt/seed.
  - The fixed bench is still valid as the official gate for large changes, but it is too noisy to classify a small allocation slice from one or two process runs.
  - For small scratch slices, use the fixed command plus either more process repetitions, a longer warmup/iters stability probe, or profiler/API evidence. The profiler evidence must separate `cuMemAllocAsync/cuMemFreeAsync/cuMemsetD8Async` and host API count/total from NCCL wall time, because NCCL wall includes waiting for rank arrival.
  - Do not treat one slow run after a build/sync as proof of a regression unless token hash, repeated median, or profile counters agree.

### Step 5: per-iteration bench reporting

- Change:
  - `pegainfer-server/src/bin/bench_serving.rs` now includes request-level `iterations` in JSON output.
  - Each measured iteration records TTFT, first decode step, steady TPOT stats, E2E time, generated token count, and generated token hash.
  - Text output is unchanged; this is meant for performance forensics and A/B parsing.
- Validation:
  - Local `cargo fmt --check` passed.
  - Local `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - Local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - 5090 `cargo check --release -p pegainfer-server --features deepseek-v4` passed after loading Cargo environment in the non-interactive SSH shell.
- Per-iteration fixed bench with `cargo run --release`:

| Log | overall avg | iter steady avg values | per-process iter spread | hash |
| --- | ---: | --- | ---: | --- |
| `/tmp/dsv4_bench_iter_detail_1.log` | `68.743ms` | `68.762`, `68.740`, `68.728` | `0.034ms` | `6346f03343d75a65` |
| `/tmp/dsv4_bench_iter_detail_2.log` | `65.737ms` | `65.806`, `65.756`, `65.649` | `0.157ms` | `6346f03343d75a65` |

- Per-iteration fixed bench with already-built `target/release/bench_serving`:

| Log | overall avg | iter steady avg values | per-process iter spread | hash |
| --- | ---: | --- | ---: | --- |
| `/tmp/dsv4_bench_iter_direct_1.log` | `65.629ms` | `65.642`, `65.688`, `65.559` | `0.128ms` | `6346f03343d75a65` |
| `/tmp/dsv4_bench_iter_direct_2.log` | `57.965ms` | `57.958`, `57.972`, `57.966` | `0.014ms` | `6346f03343d75a65` |

- Interpretation:
  - The measured iterations inside one process are extremely stable, while different processes land in different speed bands.
  - `cargo run` is not the root cause; the direct binary still produced both mid-60ms and sub-60ms bands.
  - Benchmark aggregation is not hiding a single slow request. Each process behaves as a coherent fast/slow engine instance.
  - For scratch slices, keep the official fixed command for acceptance, but judge small deltas with process-level repeated runs or profiler counters. A single 61ms vs 69ms pair can be pure process-band movement.
  - Next benchmark investigation should capture GPU clocks/power/P-state and rank-arrival timing during fast and slow process bands.
  - The direct binary runs ended with a rank-7 NCCL abort panic during shutdown after JSON had already been written. Treat this as a cleanup issue to fix separately; do not mix it into decode TPOT interpretation.

### Step 6: first clock/power probe for process speed bands

- Reason:
  - Since each process is internally stable but different processes land in different speed bands, capture GPU P-state/clock/power alongside the direct binary bench.
- Probe command shape:
  - Build once, then run `target/release/bench_serving` directly with the fixed arguments.
  - In parallel, sample `nvidia-smi --query-gpu=timestamp,index,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 500`.
- Results:

| Bench log | Clock log | overall avg | iter steady avg values | hash |
| --- | --- | ---: | --- | --- |
| `/tmp/dsv4_bench_clocked_1.log` | `/tmp/dsv4_clock_trace_1.csv` | `62.693ms` | `62.763`, `62.674`, `62.643` | `6346f03343d75a65` |
| `/tmp/dsv4_bench_clocked_2.log` | `/tmp/dsv4_clock_trace_2.csv` | `63.799ms` | `63.860`, `63.798`, `63.738` | `6346f03343d75a65` |

- Active-sample clock summary:
  - Both runs: all GPUs stayed in `P1` during active samples.
  - Both runs: memory clock stayed at `13801MHz` for all GPUs during active samples.
  - Run 1 SM average range across GPUs: about `2859-2937MHz`; run 2: about `2857-2938MHz`.
  - Active utilization averages were about `88-91%`.
- Interpretation:
  - This did not catch a fast `~58ms` or slow `~68ms` process band, so it is not the final root-cause evidence.
  - Within the captured mid band, there is no obvious low-P-state or low-memory-clock explanation.
  - The next useful trace is not more aggregate `nvidia-smi`; it should capture fast/slow bands with rank-arrival timestamps or CUDA/NCCL ranges, because the remaining hypothesis is still launch/arrival skew or process-level scheduling state rather than token content.

### Step 7: local/gathered logits scratch retry

- Reason:
  - After adding per-iteration reporting, the earlier logits scratch result needed a fair retry.
  - This slice is mechanically clean: preallocate decode-only local logits and gathered logits in `RankDecodeScratch`, write final local logits into scratch, then NCCL all-gather into scratch.
- Implementation attempted:
  - `core.rs`: added `rank_local_logits_from_hidden_into`, `final_logits_rank_local_bf16_hidden_into`, and `all_gather_logits_into`.
  - `worker.rs`: extended `RankDecodeScratch` with `local_logits` and `gathered_logits`; decode used the scratch-backed final logits path.
  - Prefill stayed on the old allocation path so the slice only measured steady decode.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - `git diff --check` passed.
- 5090 correctness:
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_logits_retry_bench1.log` | `67.119ms` | `66.664ms` | `71.415ms` | `62.177ms` | `66.229`, `67.535`, `67.593` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_logits_retry_bench2.log` | `69.386ms` | `68.612ms` | `73.079ms` | `64.328ms` | `69.273`, `69.440`, `69.445` | `6346f03343d75a65` |

- Additional direct-binary probes:

| Log | overall avg | p50 | p95 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_logits_retry_direct1.log` | `67.187ms` | `66.324ms` | `70.939ms` | `62.188ms` | `67.059`, `67.235`, `67.266` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_logits_retry_direct2.log` | `66.153ms` | `65.297ms` | `69.679ms` | `61.097ms` | `66.202`, `66.144`, `66.112` | `6346f03343d75a65` |

- Decision:
  - Rejected for now and reverted.
  - Correctness is clean, but this slice no longer reproduced the token-id-only fast band (`57.965ms` direct run) and stayed in `66-69ms` over four runs.
  - The most likely engineering lesson is that removing final-stage allocations alone is not a useful isolated optimization. It may perturb CUDA allocator/memory-pool or NCCL buffer behavior without addressing the dominant per-layer allocation/launch/skew structure.
  - Revisit only as part of a larger final-stage scratch/profile slice, with CUDA API counts proving `cuMemAllocAsync/cuMemFreeAsync` reduction and TPOT judged across process bands.
  - After revert, local `cargo fmt --check`, `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`, and `git diff --check` passed. `rg` confirmed no `rank_local_logits_from_hidden_into`, `final_logits_rank_local_bf16_hidden_into`, or `all_gather_logits_into` remains in `pegainfer-deepseek-v4/src`.

### Step 8: CUDA API allocation baseline

- Reason:
  - The optimization goal requires profiler evidence, not TPOT alone. Before moving larger fixed-shape scratch, capture the current token-id-only baseline for allocator/API pressure.
- Profile command shape:
  - Current code state: token-id scratch retained, logits scratch reverted.
  - Workload: `target/release/bench_serving --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42`.
  - Profiler: `nsys profile --force-overwrite=true --trace=cuda,nvtx --stats=true -o /tmp/dsv4_alloc_baseline ...`.
- Artifacts:
  - `/tmp/dsv4_alloc_baseline.nsys-rep`
  - `/tmp/dsv4_alloc_baseline.sqlite`
  - `/tmp/dsv4_alloc_baseline_cuda_api_cuda_api_sum.csv`
  - `/tmp/dsv4_alloc_baseline.log`
- CUDA API summary:

| API | Calls | Total time |
| --- | ---: | ---: |
| `cuMemAllocAsync` | `422,559` | `6.348s` |
| `cuMemFreeAsync` | `338,132` | `4.959s` |
| `cuMemsetD8Async` | `70,230` | `0.299s` |
| `cudaLaunchKernel` | `540,071` | `2.560s` |
| `cuLaunchKernelEx` | `26,899` | `0.283s` |
| `cuMemcpyHtoDAsync_v2` | `83,640` | `26.589s` |
| `cuMemcpyDtoHAsync_v2` | `22` | `0.038s` |
| `cudaMalloc` | `272` | `0.036s` |
| `cudaFree` | `96` | `0.028s` |
| `cudaMemsetAsync` | `112` | `0.001s` |

- GPU memory operation summary from nsys:

| Operation | Count | Total time | Total size |
| --- | ---: | ---: | ---: |
| Host-to-Device memcpy | `84,008` | `19.421s` | `177,161MB` |
| CUDA memset | `70,342` | `0.038s` | `299MB` |
| Device-to-Host memcpy | `22` | `0.0002s` | `11.377MB` |

- Interpretation:
  - Allocation/free API count is still enormous for a fixed long decode. That supports continuing with per-rank scratch, especially per-layer fixed-shape buffers.
  - `cuMemcpyHtoDAsync_v2` total is large because model/cache setup and repeated host-to-device staging are included in full-process profiling; do not read it as steady-token transfer cost without range isolation.
  - Full fixed-bench nsys tracing is expensive and distorted by profiler overhead. Use this as an API-count baseline, not a TPOT source.
  - Next profile should use NVTX/range isolation or a shorter profile-only decode segment to separate setup/prefill from steady decode allocator counts.

### Step 9: retained decode HC pre-norm scratch

- Reason:
  - `hc_pre_norm_bf16_hidden` runs twice per layer in decode and allocated four fully-overwritten buffers per call: `mixes`, `post`, `comb`, and normalized hidden output.
  - This is a higher-frequency fixed-shape target than final logits and does not touch MoE dynamic routing content.
- Implementation:
  - Added `HcPreStateView<'a>` and `HcPreNormScratch` in `runtime/state.rs`.
  - Added `hc_pre_norm_bf16_hidden_scratch` in `runtime/core.rs`; it writes `mixes/post/comb/out` into reusable scratch and returns a borrowed pre-state view.
  - Added `hc_post_bf16_hidden_view` and `all_reduce_hidden_fp32_hc_post_view` so HC post can consume scratch-backed `post/comb`.
  - Added `block_decode_rank_lane_bf16_hidden_with_scratch` and switched the direct decode worker to it.
  - Prefill remains on its existing return-new-buffer path. After later scratch slices made the decode scratch path stable, the unused old decode block wrapper was removed so new code cannot accidentally route around rank scratch.
  - Scratch is capacity-based: `HcPreNormScratch` owns buffers sized by `seq_capacity` and model config; the current direct scheduler passes capacity `1` because each rank decode command carries one token, but kernels are not newly specialized.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - `git diff --check` passed.
- 5090 correctness:
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | p99 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_hcprenorm_bench1.log` | `56.757ms` | `55.879ms` | `60.545ms` | `61.474ms` | `52.194ms` | `56.747`, `56.820`, `56.704` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_hcprenorm_bench2.log` | `59.948ms` | `58.971ms` | `64.035ms` | `64.794ms` | `54.913ms` | `59.950`, `59.910`, `59.984` | `6346f03343d75a65` |

- Decision:
  - Retain this slice.
  - It moves a high-frequency fixed-shape allocation group out of the hot path and lands the fixed long bench in the target `~60ms/token` band.
  - The first run reached `56.76ms`, so the next realistic target is stabilizing this band and moving remaining fixed-shape attention/shared-MLP storage.
- Profile note:
  - A full fixed-bench nsys profile with default CUDA event tracing became unusably slow and had to be interrupted. The nsys warning explicitly noted that device-side CUDA event completion tracing can add overhead and false stream dependencies.
  - Retrying with `--cuda-event-trace=false` completed, but the resulting CUDA API counts were invalid for A/B comparison: `cudaLaunchKernel` jumped from the baseline `540,071` calls to `19,453,472` calls despite the ordinary fixed bench completing in about the expected wall time and producing the same token hash.
  - Do not use `/tmp/dsv4_hcprenorm_alloc_noevt_cuda_api_cuda_api_sum.csv` as allocation-reduction evidence.
  - Next profiling attempt should be shorter and range-isolated, or use an internal counter around the decode steady range, before relying on API counts for this slice.

### Step 10: retained shared expert scratch

- Reason:
  - Shared expert MLP is fixed-shape and runs once per layer in decode, independent of routed MoE dynamic metadata.
  - It allocated four fully-overwritten Bf16 hidden outputs per layer: `gate`, `up`, `activated`, and `out`.
- Implementation:
  - Added `SharedExpertScratch` in `runtime/state.rs`.
  - Added `fp8_linear_bf16_hidden_into`, `swiglu_clamp_bf16_hidden_into`, and `shared_expert_forward_bf16_hidden_scratch` in `runtime/core.rs`.
  - Added `decode_moe_ag_rs_bf16_hidden_with_shared_scratch` in `runtime/moe.rs`; it changes only the shared expert branch, leaving routed MoE content, counts, indptr, compact maps, and reduce-scatter behavior unchanged.
  - Extended `block_decode_rank_lane_bf16_hidden_with_scratch` and `RankDecodeScratch` to pass reusable shared expert storage.
  - Prefill and non-decode return-new-buffer APIs remain intact; the unused decode AG/RS wrapper without shared scratch was later removed.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
- 5090 correctness:
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | p99 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_shared_bench1.log` | `49.267ms` | `48.459ms` | `52.784ms` | `54.227ms` | `44.637ms` | `49.223`, `49.289`, `49.289` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_shared_bench2.log` | `46.897ms` | `46.130ms` | `50.393ms` | `51.613ms` | `42.285ms` | `46.903`, `46.878`, `46.911` | `6346f03343d75a65` |

- Decision:
  - Retain this slice.
  - It moved another high-frequency fixed-shape allocation group into rank scratch and pushed fixed long decode well past the original `55ms/token` stretch target.
  - The remaining large allocation buckets are now more likely attention projection/top-k/indexer/compressor outputs, HC post outputs, and MoE dynamic metadata/workspace.
- Profile note:
  - Full-process nsys API count is currently not reliable after scratch reuse; see Step 9.
  - Do not claim allocator count reduction from nsys yet. The next profile work should add a range-isolated or internal steady-decode counter before using API counts as acceptance evidence.

### Step 11: retained attention output scratch and old decode wrapper cleanup

- Reason:
  - The attention output tail is fixed-shape on the active ratio `0` and ratio `4` decode paths.
  - `indexed_attention_cache_bf16_hidden` allocated the per-layer local attention output, then `attention_output_project_bf16_hidden` allocated the low-rank `wo_a` output and final `wo_b` output.
  - These buffers are fully overwritten and are consumed immediately by the HC post all-reduce, so they fit the per-rank scratch ownership model.
- Implementation:
  - Added `AttentionOutputScratch` in `runtime/state.rs` with `attn_out`, `low_rank`, and `out` buffers sized from model config and decode capacity.
  - Added `indexed_attention_cache_bf16_hidden_into` in `runtime/attention.rs`.
  - Added `bf16_linear_bf16_hidden_into` in `runtime/core.rs`.
  - Added `attention_output_project_bf16_hidden_scratch`, which writes `wo_a` and `wo_b` outputs into `AttentionOutputScratch` and returns a borrowed hidden view.
  - Updated `block_decode_rank_lane_bf16_hidden_with_scratch` so ratio `0` and ratio `4` active decode paths feed scratch-backed attention output directly into `all_reduce_hidden_fp32_hc_post_view`.
  - Removed the now-unused old public `block_decode_rank_lane_bf16_hidden` owned-output decode wrapper, its private owned ratio-4 collective helper, the old `all_reduce_hidden_fp32_hc_post` wrapper, and the old `decode_moe_ag_rs_bf16_hidden` wrapper. This keeps the direct decode path from having an attractive non-scratch bypass.
  - Kept prefill helpers and still-used public primitive operators intact.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - `git diff --check` passed.
  - `rg` found no remaining code references to `block_decode_rank_lane_bf16_hidden(`, `attention_decode_rank_local_collective_bf16_hidden`, `all_reduce_hidden_fp32_hc_post(`, or `decode_moe_ag_rs_bf16_hidden(`.
- 5090 validation:
  - Remote source gate passed: `cargo fmt --check`, `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`, and `cargo check --release -p pegainfer-server --features deepseek-v4`.
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | p99 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_attention_output_bench1.log` | `44.506ms` | `43.808ms` | `47.877ms` | `49.192ms` | `40.418ms` | `44.556`, `44.485`, `44.478` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_attention_output_bench2.log` | `48.283ms` | `47.543ms` | `51.939ms` | `52.922ms` | `43.301ms` | `48.250`, `48.339`, `48.259` | `6346f03343d75a65` |

- Decision:
  - Retain this slice.
  - It removes three fixed-shape attention-tail allocations per layer from the active decode paths without changing token content, routing content, or math.
  - Fixed long decode remains well below the earlier `55ms/token` stretch target, though process-level bands still exist.
  - Next non-MoE allocation targets are attention projection outputs, raw/window top-k buffers, ratio-4 indexer/compressor scratch, HC post output ownership, embedding/HC expand, and final HC/head-norm storage.
- Profile note:
  - This slice still lacks reliable allocator-count proof because the earlier full-process nsys path produced distorted API counts after scratch reuse.
  - Keep the profile acceptance item open until a range-isolated steady decode counter or short profiler run confirms `cuMemAllocAsync/cuMemFreeAsync/cuMemsetD8Async` reductions.

### Step 12: retained attention index scratch

- Reason:
  - Window top-k indices are generated once per active attention layer in decode.
  - Ratio-4 layers additionally allocate indexer top-k output when `compressed_len > 0`, and then allocate a concat top-k buffer.
  - These are bounded by model config (`sliding_window`, `index_topk`) and the kernels overwrite the valid prefix, so they are static storage with dynamic logical length.
- Implementation:
  - Added `AttentionIndexScratch` in `runtime/state.rs` with `window_idxs`, `compress_idxs`, and `topk_idxs`.
  - Added `window_topk_indices_decode_into`, `indexer_topk_indices_decode_into`, and `concat_topk_indices_into`.
  - Updated ratio `0` and ratio `4` decode scratch paths to use these buffers.
  - Relaxed `indexed_attention_cache_bf16_hidden_into` top-k validation from exact length to capacity length: `topk_idxs.len() >= topk`. The kernel still receives and reads only logical `topk`.
  - Kept allocating wrapper functions for prefill and non-scratch callers.
- Bug caught during validation:
  - First 5090 E2E failed at case 0 layer 2 because the scratch window buffer length was `640` while logical `topk` was `134`.
  - Root cause: capacity storage was passed to a wrapper that still enforced exact logical length.
  - Fix: treat top-k buffers as capacity-owned storage and keep logical length as the separate `topk` argument.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - `git diff --check` passed.
- 5090 validation:
  - Remote source gate passed: `cargo fmt --check`, `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`, and `cargo check --release -p pegainfer-server --features deepseek-v4`.
  - Exact E2E passed after the capacity/logical-length fix: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | p99 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_attention_index_bench1.log` | `44.427ms` | `43.626ms` | `47.919ms` | `49.252ms` | `42.388ms` | `44.681`, `44.644`, `43.955` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_attention_index_bench2.log` | `44.145ms` | `43.350ms` | `47.711ms` | `48.616ms` | `41.087ms` | `44.122`, `44.181`, `44.133` | `6346f03343d75a65` |

- Decision:
  - Retain this slice.
  - It removes three bounded `i32` allocation sites from active attention decode paths and keeps fixed long decode in the `44ms/token` band.
  - Bench shutdown still sometimes reports the pre-existing rank-7 NCCL abort panic after JSON output and scheduler exit; do not mix that cleanup issue into TPOT interpretation.
- Profile note:
  - Allocator-count evidence remains open for the combined scratch work. This slice should reduce allocation count, but the objective requires a reliable range-isolated profile or internal steady-decode counter before claiming that as measured proof.

### Step 13: retained ratio-4 attention aux scratch

- Reason:
  - Active ratio-4 decode still allocated several bounded non-MoE buffers:
    - main overlap compressor `weighted: f32[head_dim]` and compressed output `Bf16HiddenStates[head_dim, 1]` at compression positions.
    - indexer overlap compressor `weighted/out` with `index_head_dim`.
    - indexer q projection output.
    - indexer weights projection output.
    - indexer scores `f32[compressed_len]`.
  - These buffers are dynamic in logical length/presence, but their storage is bounded by model config and cache capacity.
- Implementation:
  - Added `AttentionAuxScratch` in `runtime/state.rs`.
  - Added `compressor_overlap_decode_bf16_hidden_with_dim_scratch`.
  - Added `indexer_scores_decode_bf16_hidden_scratch`.
  - Ratio-4 decode now reuses one compressor scratch for the main attention compressor and the indexer compressor because their lifetimes are sequential.
  - Indexer scores are stored in a capacity buffer sized by `max_position_embeddings.div_ceil(4)`, but NCCL all-reduce uses `slice_mut(0..score_len)`, so collectives do not process unused capacity.
  - Kept old allocating wrappers for prefill/non-scratch callers.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - `git diff --check` passed.
- 5090 validation:
  - Remote source gate passed: `cargo fmt --check`, `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`, and `cargo check --release -p pegainfer-server --features deepseek-v4`.
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | p99 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_attention_aux_bench1.log` | `41.588ms` | `41.405ms` | `43.083ms` | `43.877ms` | `41.994ms` | `41.573`, `41.663`, `41.530` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_attention_aux_bench2.log` | `36.910ms` | `36.760ms` | `38.192ms` | `39.727ms` | `36.549ms` | `36.855`, `36.979`, `36.897` | `6346f03343d75a65` |

- Decision:
  - Retain this slice.
  - It removes the remaining bounded ratio-4 compressor/indexer allocations from the active attention decode path and moves fixed long decode into the high-30/low-40ms band.
  - The two process bands remain, but each process is internally stable and token hash is unchanged.
  - Bench shutdown still sometimes reports the pre-existing rank-7 NCCL abort panic after JSON output and scheduler exit; keep it separate from decode TPOT.
- Profile note:
  - This slice is a strong candidate for the required allocator-count proof because it removes many per-ratio-4-layer `cuMemAllocAsync/cuMemFreeAsync` calls and avoids `alloc_zeros` for scores/weighted/output buffers.
  - The next profiling step should be range-isolated steady decode or an internal counter; full-process nsys remains too distorted to use as proof.

### Step 14: retained attention projection scratch

- Reason:
  - `attention_project_bf16_hidden` is the active attention entry for ratio `0` and ratio `4` decode layers.
  - It allocated six hidden buffers per active attention layer: `qr_raw`, normalized `qr`, `q_raw`, normalized `q`, `kv_raw`, and normalized/quantized `kv`.
  - These buffers are fixed by model config and decode capacity; kernels fully overwrite them.
- Implementation:
  - Added `AttentionProjectionScratch` and `AttentionProjectionsView` in `runtime/state.rs`.
  - Added `rms_norm_bf16_hidden_into` and `head_rms_norm_bf16_hidden_into` in `runtime/core.rs`.
  - Added `attention_project_bf16_hidden_scratch` and `apply_rope_attention_projections_view` in `runtime/attention_base.rs`.
  - Added `indexed_attention_cache_bf16_hidden_view_into` in `runtime/attention.rs`.
  - Updated active ratio `0` and ratio `4` decode paths to use scratch-backed projection views.
  - Kept the owned `AttentionProjections` API for prefill and non-scratch callers.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - `git diff --check` passed.
- 5090 validation:
  - Remote source gate passed: `cargo fmt --check`, `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4`, and `cargo check --release -p pegainfer-server --features deepseek-v4`.
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | p99 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_attention_projection_bench1.log` | `37.758ms` | `37.728ms` | `39.496ms` | `40.060ms` | `38.175ms` | `38.125`, `38.160`, `36.988` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_attention_projection_bench2.log` | `39.547ms` | `39.328ms` | `41.091ms` | `42.160ms` | `39.336ms` | `39.514`, `39.583`, `39.543` | `6346f03343d75a65` |

- Decision:
  - Retain this slice.
  - It removes the remaining high-frequency fixed-shape attention projection allocations from active ratio `0` and ratio `4` decode paths.
  - The two fixed long runs remain in the high-30ms band with stable hash and internally stable measured iterations.
- Profile note:
  - Allocator-count evidence is still open. After this slice, the next useful work is a range-isolated steady decode profile or internal allocator/API counter before moving into MoE EP capacity scratch.

### Step 15: cudaProfilerApi range profile was not usable as allocation proof

- Reason:
  - After Step 14, the profiler needed to exclude model load, cache setup, warmup, and prefill from CUDA API counts.
  - A temporary local diagnostic wrapped only measured iterations in `bench_serving.rs::measure_timings` with `cudaProfilerStart/Stop`.
  - That hook was only for this experiment and was removed afterward; normal bench code keeps only the per-iteration JSON reporting.
- First attempt:
  - Command shape:
    - `nsys profile --force-overwrite=true --trace=cuda,nvtx --cuda-event-trace=false --capture-range=cudaProfilerApi --capture-range-end=stop --stats=true -o /tmp/dsv4_scratch_profile_range target/release/bench_serving --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42`
  - Result: no report generated.
  - Cause: the remote binary had not been rebuilt after adding the temporary profiler hook, so the capture range never opened.
- Second attempt:
  - Rebuilt remote `target/release/bench_serving`, reran the same command, and generated:
    - `/tmp/dsv4_scratch_profile_range.nsys-rep`
    - `/tmp/dsv4_scratch_profile_range.sqlite`
  - The measured JSON inside the profiled run had stable token hash `6346f03343d75a65`, but nsys overhead moved the reported TPOT to `44.745ms/token`; do not use this as a normal bench result.
- CUDA API summary from that profiled range:

| API | Calls | Total time |
| --- | ---: | ---: |
| `cudaLaunchKernel` | `11,671,872` | `44.706s` |
| `cuMemAllocAsync` | `3,999,504` | `40.211s` |
| `cuMemFreeAsync` | `3,999,504` | `32.435s` |
| `cuMemsetD8Async` | `1,077,384` | `3.328s` |
| `cuLaunchKernelEx` | `592,584` | `2.430s` |
| `cuMemcpyDtoHAsync_v2` | `480` | `1.391s` |
| `cuMemcpyHtoDAsync_v2` | `5,328` | `0.068s` |

- GPU memory operation summary:

| Operation | Count | Total size | Total time |
| --- | ---: | ---: | ---: |
| CUDA memset | `1,077,384` | `3,080MB` | `559.219ms` |
| Host-to-Device memcpy | `5,328` | `146MB` | not used |
| Device-to-Host memcpy | `480` | `248MB` | not used |

- Decision:
  - Do not use this profile as allocator-count proof.
  - The range itself worked, but the API counts are far larger than expected for `3 * 160` measured decode tokens across `8` rank workers, and they are also far larger than the earlier full-process token-id-only baseline.
  - The most likely issue is profiler/stat distortion around CUDA graph or repeated traced runtime activity, not an actual millions-per-run decode allocation behavior.
  - The acceptance item remains open: the next proof should either use a smaller targeted trace that can be reconciled with token/layer/rank counts, or add internal hot-path allocation counters at the Rust wrapper boundary and use nsys only as a sanity check.
  - NCCL wall time remains a separate topic because NCCL ranges include rank-arrival waiting; do not mix NCCL wait with allocator proof.

### Step 16: retained decode entry embedding / HC-expand scratch

- Reason:
  - Decode entry still allocated two fully-overwritten buffers per token per rank before entering the layer loop:
    - `embedding_rank_local`: `Bf16HiddenStates[dim, 1]`.
    - `hc_expand_bf16_hidden`: `HcHiddenStates[dim, 1, hc]`.
  - This is lower frequency than per-layer attention/MLP scratch, but it is static storage and helps keep the decode path moving toward explicit rank-owned buffers.
- Implementation:
  - Added `DecodeEntryScratch` in `runtime/state.rs`.
  - Added `embedding_rank_local_into` and `hc_expand_bf16_hidden_into` in `runtime/core.rs`.
  - Extended `RankDecodeScratch` with entry scratch.
  - Decode now writes embedding into `scratch.entry.embedding`, all-reduces that buffer in place, then writes HC expand into `scratch.entry.hc_expand`.
  - Layer 0 borrows `scratch.entry.hc_expand` as the initial HC input; later layers still use the existing owned HC return path. The larger per-layer HC post double-buffer refactor is intentionally left as a separate slice.
  - Prefill remains on the existing owned allocation path.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
- 5090 validation:
  - Remote `cargo fmt --check` passed.
  - Remote `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - Remote `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | p99 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_entry_bench1.log` | `37.954ms` | `37.826ms` | `39.391ms` | `39.953ms` | `37.637ms` | `38.021`, `37.932`, `37.908` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_entry_bench2.log` | `39.484ms` | `39.315ms` | `40.997ms` | `41.787ms` | `39.253ms` | `39.513`, `39.492`, `39.447` | `6346f03343d75a65` |

- Decision:
  - Retain this slice.
  - TPOT is effectively neutral relative to Step 14, but correctness is clean and the allocation ownership direction is right.
  - The next non-MoE scratch slice with material upside is per-layer HC post/all-reduce output ownership. That requires double-buffered HC storage or a borrowed-view block return and should not be mixed with entry scratch.

### Step 17: retained per-layer HC post double-buffer scratch

- Reason:
  - Active decode still allocated HC outputs at high frequency:
    - attention branch all-reduce plus HC post output once per layer.
    - FFN HC post output once per layer.
  - The attention all-reduce path also kept its f32 conversion scratch in an implicit thread-local cache. That avoided repeated allocation after warmup, but it hid ownership outside the rank worker and made pointer/capacity reasoning worse.
- Implementation:
  - Added `HcPostScratch` in `runtime/state.rs`:
    - `attention_reduce_temp: f32[dim * capacity]`.
    - `attention_out: HcHiddenStates[dim, capacity, hc]`.
    - `layer_outputs: Vec<HcHiddenStates>` with two ping-pong output slots.
  - Added `hc_post_bf16_hidden_view_into` in `runtime/core.rs`.
  - Added `all_reduce_hidden_fp32_hc_post_view_into` in `runtime/collectives.rs`.
  - Updated `block_decode_rank_lane_bf16_hidden_with_scratch` to write attention HC post and FFN HC post into caller-provided scratch rather than returning newly allocated HC outputs.
  - Updated `run_decode_on_rank_lane` to alternate layer outputs between two HC slots:
    - layer 0 reads `DecodeEntryScratch::hc_expand` and writes slot 0.
    - layer 1 reads slot 0 and writes slot 1.
    - later layers ping-pong, so input and output storage are always distinct.
  - Removed now-unused old decode wrappers:
    - `all_reduce_hidden_fp32_hc_post_view`.
    - `hc_post_bf16_hidden_view`.
  - Prefill remains on the owned `hc_post_bf16_hidden` path.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
- 5090 validation:
  - Remote `cargo fmt --check` passed.
  - Remote `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - Remote `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | p99 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_hcpost_bench1.log` | `37.470ms` | `37.327ms` | `38.925ms` | `39.481ms` | `37.510ms` | `37.484`, `37.491`, `37.434` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_hcpost_bench2.log` | `39.737ms` | `39.547ms` | `41.196ms` | `42.246ms` | `39.346ms` | `39.777`, `39.742`, `39.691` | `6346f03343d75a65` |

- Decision:
  - Retain this slice.
  - TPOT stays in the same high-30ms process bands, but the implementation removes two per-layer HC output allocations and makes the attention f32 all-reduce scratch explicitly rank-owned.
  - This should make the eventual allocation proof easier to reconcile because one more hidden TLS scratch owner is gone from the active decode block path.
  - Remaining non-MoE allocation buckets are final HC/head-norm/logits boundary and any inactive compressed path wrappers. The next material frontier is still MoE EP capacity scratch, but allocator proof should be fixed before claiming completion.

### Step 18: retained complete final logits-boundary scratch

- Reason:
  - The earlier isolated local/gathered logits scratch retry was rejected because it stayed in a bad `66-69ms/token` process band and did not prove benefit.
  - After the per-layer scratch work, the right retry is the whole final boundary, not logits alone:
    - final HC head `mixes/pre/out`.
    - final RMSNorm output.
    - local logits.
    - gathered logits.
  - All of these are fixed by model config and decode capacity; kernels or NCCL fully overwrite the valid storage.
- Implementation:
  - Added `FinalLogitsScratch` in `runtime/state.rs`.
  - Added:
    - `hc_head_bf16_hidden_into`.
    - `rank_local_logits_from_hidden_into`.
    - `final_logits_rank_local_bf16_hidden_into`.
    - `all_gather_logits_into`.
  - Extended `RankDecodeScratch` with final-boundary storage.
  - Decode now writes final HC head, norm, local logits, and gathered logits into reusable rank scratch before rank0 D2H sampling.
  - Prefill remains on the existing owned final logits path.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
- 5090 validation:
  - Remote `cargo fmt --check` passed.
  - Remote `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - Remote `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | p99 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_final_bench1.log` | `33.875ms` | `33.793ms` | `35.196ms` | `35.863ms` | `33.561ms` | `33.883`, `33.866`, `33.875` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_final_bench2.log` | `36.060ms` | `35.934ms` | `37.477ms` | `38.349ms` | `35.656ms` | `36.021`, `35.990`, `36.170` | `6346f03343d75a65` |

- Decision:
  - Retain this slice.
  - This supersedes the earlier isolated logits-scratch rejection: logits alone was a bad slice, but the complete final-boundary scratch is correct and materially faster.
  - Non-MoE fixed-shape decode scratch is now substantially converged. Remaining active allocations are dominated by MoE AG/RS metadata/workspace plus collectives around MoE, where dynamic content must be separated from static capacity storage and semantic-zero counters.

### Step 19: retained MoE AG/RS collective-boundary scratch

- Reason:
  - Before touching dynamic MoE route metadata, the AG/RS decode path still had fixed collective/output allocations:
    - global hidden all-gather.
    - gathered token ids for hash-routed layers.
    - local routed reduce-scatter output.
    - final routed-plus-shared BF16 add output.
  - These buffers are static storage by local decode capacity and world size. Their content is dynamic, but no semantic-zero initialization is required because NCCL or kernels overwrite the valid region.
- Implementation:
  - Added `MoeAgRsScratch` in `runtime/state.rs`.
  - Added:
    - `all_gather_bf16_hidden_into`.
    - `all_gather_u32_into`.
    - `reduce_scatter_f32_hidden_into`.
    - `add_f32_bf16_to_bf16_hidden_into`.
  - Replaced the old decode AG/RS wrapper with `decode_moe_ag_rs_bf16_hidden_with_scratch`, which now writes AG/RS boundary outputs into reusable rank scratch while leaving route weights/indices, mapping, expanded input, grouped expert workspace, and partial routed output on the existing owned paths.
  - Removed now-unused owned collective wrappers:
    - `all_gather_bf16_hidden`.
    - `all_gather_u32`.
    - `reduce_scatter_f32_hidden`.
- Dynamic/static/zero classification for this slice:

| Buffer | Dynamic content? | Static storage? | Semantic zero? |
| --- | --- | --- | --- |
| `global_hidden` | yes, token hidden content | yes, `local_capacity * world_size` | no, NCCL overwrites |
| `global_token_ids` | yes, token ids | yes, `local_capacity * world_size` | no, NCCL overwrites |
| `local_routed` | yes, reduced routed output | yes, `local_capacity` | no, NCCL overwrites |
| `out` | yes, routed+shared output | yes, `local_capacity` | no, add kernel overwrites |

- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
- 5090 validation:
  - Remote `cargo fmt --check` passed.
  - Remote `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - Remote `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_moe_collective_bench1.log` | `37.901ms` | `37.757ms` | `39.519ms` | `38.020ms` | `37.904`, `37.856`, `37.944` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_moe_collective_bench2.log` | `35.369ms` | `35.269ms` | `36.727ms` | `35.370ms` | `35.379`, `35.403`, `35.325` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_moe_collective_bench3.log` | `35.872ms` | `35.714ms` | `37.389ms` | `36.041ms` | `35.859`, `35.888`, `35.870` | `6346f03343d75a65` |

- Decision:
  - Retain this slice.
  - It does not reproduce the `33.875ms` fastest final-stage process band, but two of three runs are in the `35-36ms` band and correctness/hash are stable.
  - Since this slice removes fixed MoE AG/RS storage allocations and does not alter dynamic routing math, keep it as the bridge into the remaining MoE route/workspace scratch phase.
  - Next MoE scratch work should target route weights/indices, local mapping arrays, expanded input, grouped expert intermediates, partial routed output, and distinguish semantic zero from dynamic content.

### Step 20: retained MoE route/workspace capacity scratch, performance watch

- Reason:
  - After Step 19, active MoE still allocated the dynamic-content storage every decode layer:
    - route weights/indices.
    - local mapping arrays.
    - expanded input.
    - grouped expert `w1/w3/activation/w2` intermediates.
    - partial routed f32 output before reduce-scatter.
  - These values are dynamic per token/layer, but their storage is bounded by `local_capacity * world_size * n_activated_experts`, model config, and local expert count.
- Implementation:
  - Extended `MoeAgRsScratch` with route and workspace buffers.
  - Added borrowed route/plan views:
    - `RoutedExpertsView`.
    - `MoeFusedRoutePlanView`.
  - Added scratch-backed helpers:
    - `hash_route_bf16_hidden_into`.
    - `score_route_bf16_hidden_into`.
    - `build_moe_fused_route_plan_into`.
    - `expand_moe_fused_input_into`.
    - `local_experts_forward_packed_bf16_hidden_scratch`.
    - `reduce_moe_fused_output_f32_into`.
  - Decode AG/RS now keeps route/workspace storage in per-rank scratch.
  - Owned route/workspace APIs remain for prefill and non-decode paths.
- Dynamic/static/zero classification:

| Buffer | Dynamic content? | Static storage? | Semantic zero? |
| --- | --- | --- | --- |
| `route_weights` / `route_indices` | yes, route decisions and weights | yes, `global_capacity * topk` | no, route kernels overwrite |
| `pos_to_token` / `pos_to_token_topk` / `token_topk_to_pos` | yes, local expert mapping | yes, route capacity | yes, mapping kernel clears to `-1` before writing valid positions |
| `expert_indptr` / `expert_cursor` / `local_count` | yes, local counts/prefix/cursors | yes, local expert count | yes, mapping kernel clears to `0` before count/prefix/fill |
| `expanded_input` | yes, gathered routed hidden | yes, route capacity | no, expand kernel overwrites |
| `expert_gate` / `expert_up` / `expert_activated` / `expert_out` | yes, expert GEMM intermediates | yes, route capacity | no, kernels overwrite |
| `partial_routed` | yes, f32 routed partial | yes, global capacity | no, reduce kernel overwrites |

- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
- 5090 validation:
  - Remote `cargo fmt --check` passed.
  - Remote `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - Remote `cargo check --release -p pegainfer-server --features deepseek-v4` passed.
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_scratch_moe_workspace_bench1.log` | `39.719ms` | `39.523ms` | `41.254ms` | `39.349ms` | `39.745`, `39.746`, `39.666` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_moe_workspace_bench2.log` | `39.291ms` | `39.072ms` | `40.874ms` | `38.886ms` | `39.288`, `39.250`, `39.336` | `6346f03343d75a65` |
| `/tmp/dsv4_scratch_moe_workspace_bench3.log` | `37.877ms` | `37.718ms` | `39.381ms` | `37.845ms` | `37.876`, `37.894`, `37.862` | `6346f03343d75a65` |

- Decision:
  - Retain for now because it completes the requested MoE dynamic-content/static-storage separation with clean correctness and stable token hash.
  - Keep it under performance watch: latest band `37.88-39.72ms` is higher than the best final-boundary band `33.87-36.06ms` and slightly higher than the MoE collective-only band `35.37-37.90ms`.
  - The next required proof is allocator/API measurement. Without that, do not call the overall objective complete.

### Step 21: direct CUDA runtime API allocation counter

- Reason:
  - nsys CUDA API summaries were not trustworthy for this work:
    - A full-process profile after HC pre-norm scratch inflated `cudaLaunchKernel` to tens of millions of calls.
    - A `cudaProfilerApi` measured-iteration range generated artifacts, but reported millions of `cuMemAllocAsync/cuMemFreeAsync` calls for only `3 * 160` measured decode tokens, which does not reconcile with layer/rank/token counts.
  - The next proof needed a lower-distortion counter for host API calls.
- Implementation:
  - Added `tools/cuda_api_counter.c`, an `LD_PRELOAD` shim that intercepts:
    - `cuMemAllocAsync`, `cuMemFreeAsync`, `cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemsetD8Async`.
    - `cudaMalloc`, `cudaFree`, `cudaMallocAsync`, `cudaFreeAsync`, `cudaMemsetAsync`.
    - kernel launch and async memcpy counters for context.
  - Built on 5090 with:
    - `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl`
  - Baseline was built from a clean detached worktree at commit `da3e2d5`:
    - `feat(deepseek-v4): use gpu moe ag-rs rank workers (#97)`
  - Baseline build required:
    - `PEGAINFER_TILELANG_PYTHON=/root/develop/xingming/.venv/bin/python`
    - symlinked `pegainfer-kernels/third_party/flashinfer` from the active 5090 workspace, because the clean worktree did not contain that untracked third-party checkout.
- Commands:
  - Current:
    - `LD_PRELOAD=/tmp/cuda_api_counter.so target/release/bench_serving --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42`
  - Baseline:
    - same command from `/tmp/pegainfer_alloc_baseline_1778583623`.
- Counter results:

| API | Baseline calls | Baseline total | Current calls | Current total | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `cudaMalloc` | `12944` | `2.219s` | `152` | `9.697ms` | Strong evidence that hot-path owned CUDA storage allocation pressure moved into startup scratch. |
| `cudaFree` | `12848` | `6.389s` | `48` | `7.113ms` | Strong evidence that per-token/per-layer allocation lifetime churn was removed. |
| `cudaMallocAsync` | `0` | `0` | `0` | `0` | Not used through the directly linked runtime API. |
| `cudaFreeAsync` | `0` | `0` | `0` | `0` | Not used through the directly linked runtime API. |
| `cudaMemsetAsync` | `0` | `0` | `0` | `0` | The observed semantic zero paths are not visible as runtime `cudaMemsetAsync` through this shim. |
| `cuMemAllocAsync` | `0` | `0` | `0` | `0` | The benchmark binary does not directly import this symbol; this shim cannot validate nsys' internal-runtime attribution. |
| `cuMemFreeAsync` | `0` | `0` | `0` | `0` | Same limitation as above. |
| `cuMemsetD8Async` | `0` | `0` | `0` | `0` | Same limitation as above. |
| `cudaLaunchKernel` | `18500664` | `50.148s` | `17670392` | `53.708s` | Useful only as rough context under interposition overhead; not used as TPOT evidence. |

- Bench under counter:

| Build | steady avg | p50 | p95 | hash |
| --- | ---: | ---: | ---: | --- |
| Baseline `da3e2d5` | `127.420ms` | `129.361ms` | `139.692ms` | not reported by that older bench JSON |
| Current scratch | `35.368ms` | `35.235ms` | `36.708ms` | `6346f03343d75a65` |

- Dynamic symbol check:
  - `nm -D target/release/bench_serving` shows direct imports for `cudaMalloc`, `cudaFree`, and `cudaMemsetAsync`.
  - It does not show direct imports for `cuMemAllocAsync`, `cuMemFreeAsync`, or `cuMemsetD8Async`.
  - `ldd` confirms both baseline and current link `libcuda.so.1` and `libcudart.so.12`.
- Decision:
  - Keep `tools/cuda_api_counter.c` as a low-distortion diagnostic tool.
  - Treat this as strong proof for directly linked CUDA runtime allocation host API reduction.
  - This step alone did not prove the CUDA driver function-table path. Step 24 extends the counter with `cuGetProcAddress` interception and supersedes this limitation.

### Step 22: source-level CUDA allocation residual audit

- Reason:
  - The direct counter proves current runtime API allocation calls are much lower, but the codebase still has CUDA-side `cudaMalloc` sites.
  - Those sites need to be classified so future cleanup does not confuse startup/cache growth, inactive fallbacks, and actual steady decode allocations.
- Cleanup:
  - Removed the unused `cudaProfilerStart` / `cudaProfilerStop` FFI declarations from `pegainfer-kernels/src/ffi.rs`.
  - No production Rust caller remained; the previous measured-iteration `cudaProfilerApi` hook had already been removed from `bench_serving.rs`.
  - Remote sync note: one follow-up sync accidentally used `rsync -az docs/index.md docs/projects/deepseek-v4-decode-scratch.md pegainfer-kernels/src/ffi.rs 5090:/root/develop/xingming/pegainfer/`, which flattened the three paths into untracked repo-root files on 5090. The files were removed immediately with `rm -f index.md deepseek-v4-decode-scratch.md ffi.rs`, then resynced correctly with `rsync -azR ...`. Future touched-file syncs should always preserve relative paths with `-R` or use explicit destination paths.
- Source audit:

| Site | Classification | Current action |
| --- | --- | --- |
| `pegainfer-kernels/csrc/deepseek_v4/deepseek_hc.cu::deepseek_ensure_hc_f32_scratch` | file-static per-device growth cache used by HC mix/logits-weight conversion helpers | Not a per-token Rust-owned allocation, but it can still allocate on first capacity growth; keep as a future C-side scratch migration candidate only after API proof says it matters. |
| `pegainfer-kernels/csrc/deepseek_v4/deepseek_attention.cu::{deepseek_ensure_bf16_scratch, deepseek_ensure_f32_scratch}` | file-static sparse-attention padding cache | Keep classified as internal growth cache; current scratch work already removed Rust-side active attention output/index/aux allocations. |
| `pegainfer-kernels/csrc/deepseek_v4/deepseek_indexer.cu::{deepseek_ensure_bf16_scratch, deepseek_ensure_byte_scratch}` | file-static FP4 quant scratch for indexer helpers | Keep classified as internal growth cache; ratio-4 indexer Rust buffers are already scratch-backed. |
| `pegainfer-kernels/csrc/deepseek_v4/deepseek_quant.cu::deepseek_ensure_byte_scratch` | file-static quant scratch | Keep classified as internal growth cache; grouped TileLang FP4 MoE path uses caller-owned output buffers. |
| `pegainfer-kernels/csrc/deepseek_v4/deepseek_quant.cu::deepseek_flashinfer_fp8_linear_cuda` local `cudaMalloc`/`cudaFree` group | non-current DeepSeek decode fallback shape | Do not optimize in this slice unless profiling shows the current direct decode invokes it; current decode uses the graph-safe/out-buffer FP8/BF16 wrappers and grouped TileLang FP4 MoE. |
| `pegainfer-kernels/csrc/deepseek_v4/deepseek_moe.cu::deepseek_ensure_f32_scratch` | file-static MoE route score cache | Keep classified as internal growth cache; Rust route weights/indices and mapping/workspace are now capacity scratch. |
| `pegainfer-kernels/csrc/deepseek_v4/deepseek_moe.cu::deepseek_moe_route_cuda` local `cudaMalloc`/`cudaFree` group | old route helper / debug-style path | Not part of the retained AG/RS scratch decode path; keep on cleanup watch so nobody routes production decode through it again. |
| `pegainfer-kernels/csrc/linear.cu` cuBLAS workspace allocation | process/global cuBLAS workspace | Startup/lifecycle allocation, not decode token allocation. |
| `pegainfer-kernels/src/tensor.rs`, tests, benches, `deepseek_kernel_check.rs` | generic tensor constructors, tests, benches, debug bins | Excluded from steady decode backlog unless they are reached from `run_decode_on_rank_lane`. |

- Decision:
  - The current hot-path backlog remains source-clean at the Rust operator boundary: active direct decode calls the scratch-backed `*_into`/borrowed variants.
  - The remaining `cudaMalloc` count of `152` under `LD_PRELOAD` is consistent with startup/cache/internal allocations rather than the old per-token/per-layer owned-buffer churn.
  - Do not mark the profile acceptance item complete from this audit alone; it supports the counter evidence but still does not explain the earlier nsys-only driver async attribution.
- Validation:
  - Local `cargo fmt --check` passed.
  - Local `git diff --check` passed.
  - Local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - 5090 `cargo fmt --check` passed.
  - 5090 `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - 5090 production-code grep found no remaining profiler symbol outside this document's historical notes.

### Step 23: caller-owned grouped FP4 workspace and forced NUMA-aware rank workers

- Reason:
  - MoE route/workspace scratch removed Rust-owned hot allocations, but the grouped FP4 wrapper still entered a C-side per-device quant scratch cache guarded by a mutex.
  - That wrapper could still call `cudaMalloc/cudaFree` on first capacity growth and made the scratch decode path less explicit.
  - NUMA pinning also still had a silent fallback to `allowed_cpus[rank % len]`, which made it too easy to accidentally benchmark the wrong topology.
- Implementation:
  - Added `deepseek_moe_fp4_grouped_linear_with_workspace_cuda` in `pegainfer-kernels/csrc/deepseek_v4/deepseek_quant.cu`.
  - Added FFI for the caller-owned workspace variant in `pegainfer-kernels/src/ffi.rs`.
  - Extended `MoeAgRsScratch` with:
    - `fp4_act_workspace: CudaSlice<u8>`
    - `fp4_act_scale_workspace: CudaSlice<u8>`
  - Workspace capacity is model/capacity based:
    - act bytes: `route_capacity * max(config.dim, config.moe_inter_dim)`
    - scale bytes: `route_capacity * ceil(max(config.dim, config.moe_inter_dim) / 128)`
  - The three grouped FP4 expert linears reuse the same workspace on the same stream, so no aliasing occurs across concurrent kernels.
  - Changed rank-worker affinity to forced NUMA-aware pinning:
    - `cuDeviceGetPCIBusId` resolves the CUDA device to a PCI bus id without initializing CUDA runtime.
    - `/sys/bus/pci/devices/<pci>/numa_node` resolves the PCI device to the NUMA node.
    - `/sys/devices/system/node/node<numa>/cpulist` supplies the target CPU list.
    - Missing PCI/NUMA/cpulist data, empty cpuset intersection, or failed `pthread_setaffinity_np` now panics instead of silently falling back.
- 5090 topology evidence:

| GPU | PCI-derived NUMA CPU target |
| --- | --- |
| `0..=3` | NUMA0, `0-31,64-95` |
| `4..=7` | NUMA1, `32-63,96-127` |

- Forced pin logs:
  - Rank workers pinned as `GPU0..3 -> CPU0..3`, `GPU4..7 -> CPU36..39`.
  - Logs appeared in `/tmp/dsv4_forced_numa_fp4_workspace_e2e.log`, `/tmp/dsv4_forced_numa_fp4_workspace_bench.log`, and `/tmp/dsv4_forced_numa_fp4_workspace_bench2.log`.
- Validation:
  - Local `cargo fmt --check` passed.
  - Local `git diff --check` passed.
  - Local `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - 5090 `cargo fmt --check` passed.
  - 5090 `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - 5090 rebuilt `bench_serving` and `deepseek_v4_e2e` release bins after the affinity change.
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
- 5090 fixed bench:

| Log | overall avg | p50 | p95 | first decode avg | iteration steady avg values | hash |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `/tmp/dsv4_numa_fp4_workspace_bench1.log` | `32.858ms` | `32.375ms` | `34.748ms` | `31.768ms` | `32.837`, `32.869`, `32.870` | `6346f03343d75a65` |
| `/tmp/dsv4_numa_fp4_workspace_bench2.log` | `33.901ms` | `33.417ms` | `35.782ms` | `32.557ms` | `33.953`, `33.875`, `33.874` | `6346f03343d75a65` |
| `/tmp/dsv4_numa_fp4_workspace_bench3_no_taskset.log` | `33.118ms` | `32.675ms` | `35.066ms` | `31.623ms` | `33.105`, `33.117`, `33.133` | `6346f03343d75a65` |
| `/tmp/dsv4_forced_numa_fp4_workspace_bench.log` | `35.365ms` | `34.871ms` | `37.514ms` | `33.896ms` | `35.375`, `35.392`, `35.327` | `6346f03343d75a65` |
| `/tmp/dsv4_forced_numa_fp4_workspace_bench2.log` | `34.343ms` | `33.840ms` | `36.343ms` | `33.229ms` | `34.331`, `34.348`, `34.351` | `6346f03343d75a65` |

- Counter result:

| API | Previous current | New current | Interpretation |
| --- | ---: | ---: | --- |
| `cudaMalloc` | `152` | `136` | Caller-owned grouped FP4 workspace removed another C-side growth-cache allocation group from the scratch path. |
| `cudaFree` | `48` | `32` | Same lifetime reduction as above. |
| `cuMemAllocAsync` | `0` | `0` | Still not directly imported/called through this shim. |
| `cuMemFreeAsync` | `0` | `0` | Same as above. |
| `cuMemsetD8Async` | `0` | `0` | Same as above. |

- Decision:
  - Keep forced NUMA-aware pinning, but derive topology dynamically from CUDA PCI id and sysfs NUMA metadata rather than encoding the 5090 ordinal split in runtime logic.
  - Keep caller-owned grouped FP4 workspace. It improves the allocation proof and removes a mutex/cache lookup from the decode scratch path.
  - Treat the post-forced fixed bench band as `34.34-35.36ms/token`; the earlier rebuilt same-code runs show the fast process band can still land at `32.86-33.90ms/token`.

### Step 23a: review fixes before PR

- Sub-agent review found two code risks worth fixing before opening the PR:
  - Forced NUMA pinning initially encoded the 5090 ordinal split as `GPU0..3 -> NUMA0` and `GPU4..7 -> NUMA1`.
  - MoE scratch capacity checks used mutable `seq_len` metadata as both logical length and allocation capacity.
- Fixes:
  - Affinity now resolves topology at startup from CUDA driver `cuDeviceGetPCIBusId` plus sysfs NUMA metadata, then intersects that NUMA CPU list with the process's allowed cpuset. There is still no fallback.
  - A 5090 E2E retry caught that using CUDA runtime `cudaDeviceGetPCIBusId` can initialize `libcudart` and fail on symbol mismatch (`cudaDevResourceGenerateDesc`). The implementation now uses the already-present CUDA driver binding instead.
  - `Bf16HiddenStates`, `F32HiddenStates`, and `HcHiddenStates` expose buffer-derived `seq_capacity()` helpers. The decode `*_into` paths that mutate logical `seq_len` now check capacity from underlying storage, and NCCL all-gather/reduce-scatter use logical prefix slices rather than whole-capacity buffers.
  - `tools/cuda_api_counter.c` now keeps separate function-table pointers for normal driver symbols and `_ptsz` variants, so the diagnostic shim does not mix per-thread-default-stream entry points with base entry points.
- Local validation after these fixes:
  - `cargo fmt` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl` passed.
- 5090 validation after these fixes:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
  - Release `deepseek_v4_e2e` rebuilt and passed: `All 20 DeepSeek V4 exact cases passed`.
  - Dynamic NUMA logs resolved `GPU0..3` to PCI buses on NUMA0 and `GPU4..7` to PCI buses on NUMA1, pinning ranks to `CPU0..3` and `CPU36..39`.
  - Fixed bench log `/tmp/dsv4_pr_driver_numa_bench.log`: steady TPOT avg `35.253ms`, p50 `34.800ms`, p95 `37.335ms`, first decode avg `33.743ms`, hash `6346f03343d75a65`.
  - `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl` passed; `nm -D` exported base and `_ptsz` wrappers for `cuMemAllocAsync`, `cuMemFreeAsync`, and `cuMemsetD8Async`.
  - Process shutdown still prints the existing NCCL communicator abort panic after the benchmark JSON is emitted. This is recorded as a shutdown cleanup issue, not decode TPOT evidence.

### Step 24: CUDA driver function-table allocation counter

- Reason:
  - The first `LD_PRELOAD` counter proved directly linked CUDA runtime symbols but did not cover the CUDA driver function-table path.
  - CUDA runtimes can obtain driver entry points through `cuGetProcAddress`; if that happened here, direct symbol interposition alone would miss calls to `cuMemAllocAsync`, `cuMemFreeAsync`, or `cuMemsetD8Async`.
- Implementation:
  - Extended `tools/cuda_api_counter.c` to intercept both:
    - `cuGetProcAddress_v2(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*)`
    - `cuGetProcAddress(const char*, void**, int, cuuint64_t)`
  - The shim replaces returned function pointers for:
    - `cuMemAllocAsync`
    - `cuMemFreeAsync`
    - `cuMemsetD8Async`
    - their `_ptsz` variants, using separate stored real function pointers from the base symbols
  - Added counters:
    - `cuGetProcAddress`
    - `cuGetProcAddress_replaced`
- Build proof:
  - Local:
    - `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter.so tools/cuda_api_counter.c -ldl`
  - 5090:
    - `gcc -shared -fPIC -O2 -Wall -Wextra -o /tmp/cuda_api_counter_getproc.so tools/cuda_api_counter.c -ldl`
    - `nm -D /tmp/cuda_api_counter_getproc.so` exported `cuGetProcAddress`, `cuGetProcAddress_v2`, `cuMemAllocAsync`, `cuMemFreeAsync`, and `cuMemsetD8Async`; the PR-prep fix also exports the matching `_ptsz` wrappers.
- 5090 fixed bench under enhanced counter:
  - Log: `/tmp/dsv4_forced_numa_getproc_counter2.log`
  - Command:
    - `LD_PRELOAD=/tmp/cuda_api_counter_getproc.so target/release/bench_serving --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42`

| Metric | Value |
| --- | ---: |
| steady TPOT avg | `32.752ms` |
| steady TPOT p50 | `32.237ms` |
| steady TPOT p95 | `34.792ms` |
| iteration steady avg values | `32.749`, `32.725`, `32.782` |
| hash | `6346f03343d75a65` |

| API | Current calls | Current total | Interpretation |
| --- | ---: | ---: | --- |
| `cuMemAllocAsync` | `0` | `0` | Not used by the current fixed bench through direct symbol or driver function-table path. |
| `cuMemFreeAsync` | `0` | `0` | Same as above. |
| `cuMemsetD8Async` | `0` | `0` | Same as above. |
| `cuGetProcAddress` | `0` | `0` | CUDA runtime did not request driver API pointers through this path in the benchmark process. |
| `cuGetProcAddress_replaced` | `0` | `0` | No hidden function-table calls existed to replace. |
| `cudaMalloc` | `136` | `10.197ms` | Startup/cache/internal allocation count after scratch migration and caller-owned grouped FP4 workspace. |
| `cudaFree` | `32` | `7.459ms` | Startup/cache/internal free count after scratch migration. |
| `cudaMallocAsync` | `0` | `0` | Not used. |
| `cudaFreeAsync` | `0` | `0` | Not used. |
| `cudaMemsetAsync` | `0` | `0` | Not used. |

- Decision:
  - Treat the profile acceptance criterion as satisfied for application-visible CUDA APIs:
    - The directly linked runtime API count/total dropped from baseline `cudaMalloc/cudaFree = 12944/12848` to current `136/32`.
    - The named driver async APIs are `0` both through direct symbol interposition and the driver function-table path.
  - The earlier nsys-only `cuMemAllocAsync/cuMemFreeAsync/cuMemsetD8Async` numbers are not accepted as hot-path evidence because they do not reconcile with token/layer/rank counts and are not visible through either application API path.

## Debrief

- **Outcome**:
  - Decode hot-path CUDA allocation ownership moved to per-rank reusable scratch for the current direct DeepSeek V4 decode path.
  - Non-MoE fixed-shape scratch and MoE EP dynamic-content/static-storage capacity scratch are both implemented.
  - Forced NUMA-aware rank-worker pinning is enabled with no silent fallback and resolves NUMA placement from CUDA driver PCI id plus sysfs.
  - Fixed bench is well beyond the original `~60ms/token` and `55ms/token` targets, landing in the `34.34-35.36ms/token` forced band with repeated same-code fast runs around `32.75-33.90ms/token`.
- **Pitfalls encountered**:
  - Single-process bench bands can move several milliseconds; retained/rejected decisions need repeated fixed runs and token hash, not one run.
  - `rsync` with multiple source files and a directory destination can flatten paths; use `rsync -azR` for touched-file syncs.
  - `cargo check` does not rebuild already-built release bins; benchmark and E2E binaries must be rebuilt after code changes that affect runtime behavior.
  - Plain `LD_PRELOAD` direct symbol interposition does not prove absence of CUDA driver function-table calls; `cuGetProcAddress` interception closes that named-driver-async-API gap.
- **Lessons learned**:
  - For this codebase, the meaningful allocation proof is application-visible CUDA API count/total plus source-level decode-path inventory. NCCL wall time and nsys internal-runtime attribution can include waiting or profiler artifacts.
  - MoE scratch must keep dynamic content, static storage, and semantic zero separate: route/count values are dynamic; storage is capacity-based; only counters/cursors/indptr and mapping sentinels require semantic clears.
  - NUMA-aware pinning needs to fail loudly on topology/cpuset mismatch and must read topology from the device, not from assumed ordinal ranges.
