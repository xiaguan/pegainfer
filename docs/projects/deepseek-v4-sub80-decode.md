# DeepSeek V4 Sub-80 Decode

**Created**: 2026-05-12
**Status**: active

## TL;DR

After the grouped MoE pointer-cache commit, DeepSeek V4 fixed-token long bench is now `83.37-89.65ms/token`. This follow-up targets the remaining gap toward a stable sub-80ms decode by first identifying which post-cache collective windows still carry rank-arrival skew. No new `bs=1` or `seq_len=1` specialization is allowed.

## Preparation

- **Read**:
  - `docs/index.md` - confirmed the just-completed 90ms project and the active MoE AG/RS history.
  - `docs/projects/deepseek-v4-90ms-decode.md` - confirmed pointer caching moved the long bench below 90ms and reduced the nsys reduce-scatter synchronization window.
  - `docs/projects/deepseek-moe-tilelang-review.md` - confirmed the earlier rank-stage proof method: group NCCL kernels into logical collectives and separate arrival skew from post-arrival tail.
- **Relevant history**:
  - The latest effective commit is `ee01f0b feat(deepseek-v4): cache moe grouped expert pointers`.
  - Post-change nsys still shows large f32 all-reduce windows: `AllReduce_Sum_f32 = 5803.26ms`, while `ReduceScatter_Sum_f32` dropped to `2040.33ms`.
- **Plan**:
  1. Inspect `/tmp/dsv4_90ms_ptr_cache_profile.sqlite` on 5090 and reproduce the logical-collective grouping method for post-change all-reduce/reduce-scatter/all-gather kernels.
  2. Identify the largest remaining collective windows by sequence position and infer whether they align with attention all-reduce, MoE reduce-scatter, embedding, indexer score all-reduce, or final logits all-gather.
  3. Choose the next implementation slice from evidence, not from kernel-name intuition.
  4. Keep fixed-token long bench and exact E2E as the gate for any code change.
- **Risks / open questions**:
  - The current trace may lack NVTX labels at collective call sites, so the first pass may need sequence-position inference before adding scoped labels.
  - Nsight Systems kernel duration is still synchronization-window evidence, not pure transfer time.

## Execution Log

### Step 1: group post-cache NCCL kernels
- Parsed `/tmp/dsv4_90ms_ptr_cache_profile.sqlite` on 5090 and grouped consecutive NCCL kernels by 8 ranks to estimate logical collective windows.
- Post-cache profile command source: fixed-token bench with `--prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42`; nsys profiler overhead makes absolute TPOT higher, so the trace is used for relative window decomposition.
- Main result:

| NCCL kernel | Kernels | Rank groups | Kernel duration sum | Logical wall sum | Start-skew sum | Post-arrival tail sum | Avg wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AllReduce_Sum_f32` | 9048 | 1131 | `5803.26ms` | `1940.21ms` | `1370.35ms` | `569.86ms` | `1715.5us` |
| `ReduceScatter_Sum_f32` | 6075 | 759 | `2040.33ms` | `378.97ms` | `367.46ms` | `11.50ms` | `499.3us` |
| `AllGather` | 6656 | 832 | `1013.17ms` | `149.57ms` | `138.16ms` | `11.41ms` | `179.8us` |
| `AllReduce_Sum_bf16` | 144 | 18 | `10.01ms` | `2.06ms` | `1.69ms` | `0.37ms` | `114.4us` |

- Interpretation:
  - The remaining dominant collective window is f32 all-reduce, not grouped MoE FP4 compute.
  - `AllReduce_Sum_f32` still contains a large rank-arrival component: `1370.35ms / 1940.21ms` logical wall over the profiled run.
  - `ReduceScatter_Sum_f32` improved after grouped MoE pointer caching, but its wall is still almost entirely arrival skew, so NCCL duration must not be read as pure transfer time.
  - The next code slice should label or isolate f32 all-reduce call sites before changing algorithms, because the kernel name alone mixes attention hidden all-reduce, indexer score reductions, and any other f32 all-reduce with the same element count.

## Debrief
