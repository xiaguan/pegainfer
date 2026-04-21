# DSV3.2 Optimization Directions — SGLang / vLLM Cross-Reference

> **Status**: Build-level D2 integration landed (NVSHMEM hard-dep in `build.rs`; `e2e_dsv32_small` 44/44 still passing). No runtime NVSHMEM calls yet — next steps in §D2-integration.
> **Owner**: TBD (split by direction).
> **Derived from**: `docs/areas/dsv32.md` #5 rebaseline profile (2026-04-21).
> **Last updated**: 2026-04-21 (build.rs NVSHMEM link landed)

## TL;DR
- D2 (DeepEP low-latency mode) is the only direction that **structurally removes** `cached_notify_combine` — LL polls per-expert `rdma_recv_flag` instead of a global `barrier_block`, so the 391us cross-rank skew stops gating combine. SGLang resolves AUTO → LOW_LATENCY for any non-extend batch; this is the well-trodden path. The gotcha is that LL is built on NVSHMEM/IBGDA and pegainfer's `csrc/deep_ep.cu:1-2` is intentionally NVSHMEM-disabled, so adopting it is a real integration project, not a flag flip.
- D1 (DeepGEMM grouped GEMM) is the obvious replacement for the 42.58% `fp8_gemm` slice. Both SGLang and vLLM use **GroupedMasked** (`fp8_m_grouped_gemm_nt_masked`) for the LL/decode path, **GroupedContiguous** for the prefill path. SGLang has explicitly deprecated the contiguous path for non-W4AFP8 normal-mode decode (`sglang/.../ep_moe/layer.py:232,242`).
- D3 (CPU jitter) is half-explained by SGLang/vLLM being process-per-rank with `psutil.cpu_affinity` / `numactl --cpunodebind --membind` per process; pegainfer's single-process-N-thread topology pays the contention.
- D4 (combine accumulator) — pegainfer rounds bf16 between every topk add (`csrc/moe.cu:226-227`); SGLang's `_moe_sum_reduce_kernel` keeps an fp32 accumulator across all topk and casts once. Cheap fix, also unlocks order-independent batching.
- D5 (overlap) — DeepEP's normal `Buffer.dispatch` accepts `allocate_on_comm_stream`; SGLang explicitly overlaps shared-experts on `self.alt_stream` (`models/deepseek_v2.py:776-781`) and has full TBO/SBO machinery. For bs=1 single-request the win is small until D2 lands.

## Context
See `docs/areas/dsv32.md` section 9 #5. Two facts dominate the prioritization here: (a) `cached_notify_combine` is a CPU-jitter-driven barrier, not a comms kernel; (b) `fp8_gemm` is per-expert, one GEMM per local-expert per layer, which is exactly the workload `fp8_m_grouped_gemm_nt_masked` is designed for.

---

## D1. Grouped GEMM for routed experts

### What SGLang does
- **DeepEP-LL decode path uses GroupedMasked, FP8.** Entry: `sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:263` (gate-up) and `:341` (down) — both call `deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(...)`. Wrapper at `sglang/python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py:26-64`, which dispatches to `deep_gemm.fp8_m_grouped_gemm_nt_masked` (= `m_grouped_fp8_gemm_nt_masked`, see `third_party/DeepGEMM/deep_gemm/__init__.py:77`).
- **Mask is device-built.** The `masked_m` tensor — number of valid recv tokens per local expert — is filled by Triton kernel `compute_masked_m_triton_kernel` at `sglang/.../ep_moe/kernels.py:972-979` from the `seg_indptr` produced by an in-place sort on `recv_topk_idx`. The whole pipeline (sort → seg_indptr → masked_m) lives in `sglang/.../ep_moe/kernels.py:1054-1063`. No host roundtrip on the decode hot path.
- **Decode vs prefill split.** `sglang/.../ep_moe/layer.py:228-242` shows the runtime branch: `format_is_deepep_normal` (= prefill / extend) used to call `forward_deepgemm_contiguous` and now hard-asserts deprecated; `format_is_deepep_ll` (= decode) goes to `forward_cutlass_w4afp8_masked` for W4AFP8 or to the masked DeepGEMM path through the modular runner (`moe_runner/deep_gemm.py:454` `pre_permute_deepep_ll_to_deep_gemm`). So in practice the live path for FP8 decode is **GroupedMasked**; GroupedContiguous is reserved for legacy / extend formats.
- **SiLU×Up is a separate Triton kernel after gate-up GEMM**, fused with per-128 FP8 quantization of the down-input. Kernel: `silu_and_mul_masked_post_quant_fwd` at `sglang/.../ep_moe/kernels.py:287-426`, called between the two grouped GEMMs at `sglang/.../moe_runner/deep_gemm.py:306-313`. There is no SiLU-fused-into-GEMM epilogue; SGLang accepts the extra kernel because it also needs to produce per-token-per-expert fp8 scales for the down GEMM.

### What vLLM does
- **Same split.** `vllm/vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py:30,428,443` calls `fp8_m_grouped_gemm_nt_masked` for both gate-up and down in the LL/batched path; `vllm/.../experts/deep_gemm_moe.py:37,287,300` calls `m_grouped_fp8_gemm_nt_contiguous` for the contiguous (prefill) path.
- The selection happens in `vllm/.../fused_moe/all2all_utils.py:147-172`: when `moe.use_deepep_ll_kernels` is true, `DeepEPLLPrepareAndFinalize` is wired in and pairs with `BatchedDeepGemmExperts`. `use_deepep_ll_kernels` is true iff `all2all_backend == "deepep_low_latency"` (`vllm/.../fused_moe/config.py:955-956`).
- vLLM uses the same masked semantic (per-expert valid-row count tensor) — the recv buffer shape `[num_local_experts, max_dispatch * num_ranks, hidden]` is exactly what DeepGEMM masked expects.

### Implication for pegainfer
- **Cheapest path**: build `masked_m[num_local_experts]` on device from the existing `recv_topk_idx` (or directly from DeepEP's per-expert `packed_recv_count`, which is what LL dispatch returns — see `third_party/DeepEP/deep_ep/buffer.py:613`). With normal dispatch we currently get `recv_x` packed contiguously, not per-expert-batched, so we'd need to **switch to LL dispatch first** (D2) to get the layout the masked kernel wants. With normal dispatch, the path is GroupedContiguous + a `m_indices[recv_token]→expert` array (host-built then H2D, or build via segmented scan). SGLang's `ep_moe/kernels.py:1044-1063` is the reference for that build.
- **Gotcha 1**: DeepGEMM masked requires the recv buffer to be `[E, M_max, K]` with `M_max` rounded up to the kernel's block-M (note in `sglang/.../ep_moe/kernels.py:1063`). Pegainfer's current `ep_recv_x` (`src/model/dsv32/forward.rs:142`) is `[max_recv_tokens * hidden]` flat — needs reshape/realloc to `[num_local_experts, M_max_per_expert, hidden]`.
- **Gotcha 2**: SiLU is not free — budget for an extra `silu_and_mul_masked_post_quant_fwd`-equivalent, ~1-2% of decode time, which we can probably steal from the existing per-expert SiLU we already run.

---

## D2. DeepEP low-latency mode

### What is in DeepEP itself
- LL is a **separate kernel family**: `third_party/DeepEP/csrc/kernels/internode_ll.cu` (combine at `:715`, dispatch earlier in the same file). API exposed via `Buffer.low_latency_dispatch` (`third_party/DeepEP/deep_ep/buffer.py:548`) and `Buffer.low_latency_combine` (`:617`).
- **Yes, LL eliminates `cached_notify_combine`.** Cross-rank synchronization is **per-expert**, via atomic flags `rdma_recv_flag[expert_idx]`. Senders bump the flag with `nvshmemi_ibgda_amo_nonfetch_add` after their data write (see the AMO send around `internode_ll.cu:920-931`); the receiver-side warps spin on `ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0` (`internode_ll.cu:956`). There is no `barrier_block<kNumRanks>` anywhere in the LL combine — contrast `intranode.cu:618,628` for the normal `cached_notify_combine`. So a slow rank only delays the experts it owns, not the global step.
- **Weighted-combine semantic preserved.** `Buffer.low_latency_combine` docstring (`buffer.py:622-636`) explicitly says "reduce **with weights**"; `topk_weights: [num_combined_tokens, num_topk] float`. The kernel applies the weight per-token-per-expert in fp32 then accumulates: `internode_ll.cu:700-711` (`accum[k] += static_cast<float>(...) * weight`). The dst write goes `int4`-vectorized straight into `combined_x` — caller does NOT post-weight.
- **NVLink intranode IS supported.** `third_party/DeepEP/README.md:300` lists `[x] Support NVLink protocol for intranode low-latency kernels`. The `allow_nvlink_for_low_latency_mode` flag (`buffer.py:38,108`) lets it use NVLink for in-node hops while still using IBGDA for atomic signaling.

### What SGLang does
- **AUTO resolution chooses LL for decode automatically.** `sglang/.../layers/moe/utils.py:113-120`:
  ```
  def resolve(self, is_extend_in_batch):
      if is_extend_in_batch:  return DeepEPMode.NORMAL
      else:                   return DeepEPMode.LOW_LATENCY
  ```
  Selected per-batch in `sglang/.../token_dispatcher/deepep.py:830-836`.
- Two `_DeepEPDispatcherImpl` classes (`token_dispatcher/deepep.py:372` Normal, `:534` LowLatency); LL impl wraps `buffer.low_latency_dispatch`/`combine` directly (`:617,:698`). Buffer is allocated with `low_latency_mode=True` (`:225`).
- LL imposes a hard layout constraint: recv shape is `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` (`buffer.py:586,593`). This is the layout DeepGEMM masked wants — the choice is paired by design.

### What vLLM does
- Yes — `DeepEPLLPrepareAndFinalize` in `vllm/.../fused_moe/prepare_finalize/deepep_ll.py:53`, called via `vllm/.../fused_moe/all2all_utils.py:172`. Combine site: `deepep_ll.py:418-427` calls `buffer.low_latency_combine`.

### Implication for pegainfer
- **Single biggest structural change** — and the one that actually deletes the `cached_notify_combine` 35% slice rather than just shrinking it.
- **Hard prerequisite (verified 2026-04-21, see §D2-smoke below)**: NVSHMEM init. `csrc/deep_ep.cu:1-2` says "NVSHMEM disabled, kNumRanks=8 for EP8" — we'd need to wire `nvshmem_init_with_uniqueid` (UID bootstrap, no MPI/PMI needed) into the Rust startup. SGLang's bootstrap (`token_dispatcher/deepep.py:124-132`) is a reference for the unique-id exchange. **Integration plan is hard-dep, not feature-flagged**: going forward DeepEP will be built with NVSHMEM enabled unconditionally, `-DDISABLE_NVSHMEM` goes away, `libnvshmem_host.so` + `libnvshmem_device.a` link into the Rust binary as required deps.
- LL combine path is the bf16 + per-128 LogFMT meta packet format (`internode_ll.cu:766-770`); pegainfer's current `combined_x` is bf16 — compatible. We do NOT need to change the rest of the model.
- **Two-buffer constraint**: "you cannot hold more than 2 LL kernels' results at a single moment" (`buffer.py:559,627`). For a 1-layer-at-a-time decode this is fine.
- LL dispatch's quantization (FP8 with per-128 scales, `buffer.py:586-591`) is built in. We currently dispatch bf16; LL with `use_fp8=False` keeps bf16 (`buffer.py:592-593`), so we don't have to take on FP8 quant just to switch.

### §D2-smoke. NVSHMEM feasibility check on H20 (2026-04-21)

Before committing to D2 integration work, ran a ~130-line C/CUDA smoke test to confirm NVSHMEM can initialize and execute the primitive DeepEP LL depends on (`nvshmem_int_atomic_add`, used for the per-expert `rdma_recv_flag` in `internode_ll.cu:920-931`). Source: `/root/develop/xingming/nvshmem_smoke/` on H20.

Setup:
- NVSHMEM 3.6.5.0 built from GitHub source (`git clone https://github.com/NVIDIA/nvshmem`) with `NVSHMEM_IBGDA_SUPPORT=ON`, `NVSHMEM_MPI_SUPPORT=OFF`, `NVSHMEM_USE_GDRCOPY=ON`. Installed to `/usr/local/nvshmem`.
- Bootstrap via `nvshmemx_get_uniqueid` + `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, ...)`. UID distributed across 8 forked processes via `mmap(MAP_SHARED|MAP_ANONYMOUS)`. **No MPI/PMI/Hydra** — same bootstrap shape the Rust integration will use.
- Payload: 8 PEs, each does `nvshmem_int_atomic_add(target, my_pe+1, /*peer=*/0)`. Expected final value on PE 0: `1+2+...+8 = 36`.

Results:
| Config | `NVSHMEM_IB_ENABLE_IBGDA` | `NVSHMEM_DISABLE_P2P` | Transport actually used | Outcome |
|---|---|---|---|---|
| default | 0 | 0 | NVLink P2P | ✅ target=36 |
| forced IBGDA | 1 | 1 | IBGDA over mlx5 RoCE | ✅ target=36 |

The second row is the decisive one: `NVSHMEM_DISABLE_P2P=1` blocks the intra-node P2P fast path, forcing the atomic through IBGDA on mlx5 RoCE HCAs. The `ibv_dereg_mr` tail in the debug log confirms the IB transport was actively registered and used. This is exactly the path `nvshmemi_ibgda_amo_nonfetch_add` takes inside `internode_ll.cu`.

Conclusions:
- NVSHMEM starts fine on H20-3e with RoCE + NVLink. No hang, no IB probe failures.
- IBGDA atomics work end-to-end on this hardware. D2 does not have a hidden technical blocker.
- UID bootstrap from a plain fork() tree is sufficient — fits the Rust executor shape (thread-per-rank today, process-per-rank if D3 demands it).
- Build only produces `libnvshmem_host.so` + `libnvshmem_device.a` (no combined `libnvshmem.so`), so `build.rs` must link both explicitly: `-lnvshmem_host -lnvshmem_device`.

Remaining unknowns not answered by this smoke test (must verify during integration):
- Whether DeepEP LL's `allow_nvlink_for_low_latency_mode=True` mode cleanly prefers NVLink P2P for the data path while still using IBGDA atomics for signaling on this box.
- Whether DeepEP's LL kernels (when rebuilt without `-DDISABLE_NVSHMEM`) coexist with pegainfer's current intranode dispatch/combine code on the same `Buffer` instance, or whether we need two parallel code paths during transition.
- Whether NVSHMEM's worker threads and pegainfer's tokio runtime interact poorly (e.g. NVSHMEM creating its own pthreads that fight for the same cores D3 will pin).

### §D2-integration. Build-level integration landed (2026-04-21)

**What's in the current build:**
- `build.rs` compiles DeepEP with NVSHMEM as a **hard dependency** (no feature flag, no fallback). Sources extended from 3 files (`intranode.cu` / `layout.cu` / `runtime.cu`) to 5 — added `internode.cu` and `internode_ll.cu`. `-DDISABLE_NVSHMEM` removed.
- NVSHMEM 3.6.5 lives at `/usr/local/nvshmem` on H20 (see `docs/resources/h20-development-guide.md` §9.1 for source path and cmake options). Build picks it up via `NVSHMEM_HOME` env var with that default.
- DeepEP's 5 kernel .cu files + `csrc/deep_ep.cu` wrapper are compiled with `-rdc=true` so device-side NVSHMEM calls (`nvshmemi_ibgda_amo_nonfetch_add` etc.) become cross-TU relocatable references.
- A dedicated `nvcc -dlink` step resolves those device references against `libnvshmem_device.a`, producing `deepep_nvshmem_dlink.o` which goes into `libkernels_cuda.a`.
- Final Rust link line adds `-lnvshmem_host`, `--whole-archive libnvshmem_device.a --no-whole-archive` (whole-archive is **required** so CUDA fatbin registration ctors in NVSHMEM's own device TUs are preserved), plus `-libverbs -lmlx5 -lgdrapi`. `rpath` embeds `/usr/local/nvshmem/lib` so the binary doesn't need `LD_LIBRARY_PATH` at runtime.

**What's NOT in the current build:**
- No runtime code actually calls NVSHMEM. The binary now **links** against `libnvshmem_host.so.3` (verify with `ldd target/release/pegainfer`) but never invokes `nvshmemx_init_*`. All existing DSV3.2 decode and prefill still goes through `deep_ep::intranode::dispatch/combine`, which doesn't touch NVSHMEM.
- `csrc/deep_ep.cu` still only exports the existing intranode wrappers (`deep_ep_intranode_dispatch` / `deep_ep_intranode_combine` / `deep_ep_cached_notify_combine` / `deep_ep_notify_dispatch` / `deep_ep_get_dispatch_layout` / `deep_ep_intranode_barrier`). LL / internode wrappers not added yet.
- Rust executor (`src/model/dsv32/executor.rs`, `src/model/dsv32/deep_ep.rs`) does no NVSHMEM bootstrap. No `nvshmemx_get_uniqueid` / `nvshmemx_init_attr` call graph.

**Verification (post-integration):**
- `cargo build -r` on H20 succeeds, producing `target/release/pegainfer` (~39 MB) and `target/release/bench_serving` (~38 MB).
- `ldd target/release/pegainfer | grep nvshmem` shows `libnvshmem_host.so.3 => /usr/local/nvshmem/lib/libnvshmem_host.so.3`.
- `cargo test --release --test e2e_dsv32_small -- --ignored` — **44/44 passing in 397.27s**, decode throughput `17.5-18.3 tok/s` (no regression vs `#3` accepted baseline `~18.7 tok/s`; tiny drop is within per-run variance).

**Gotchas / "don't remove these":**
- The three-step dance — `-rdc=true` + explicit `nvcc -dlink` + whole-archive `libnvshmem_device.a` — is not optional. Skipping any one of them produces undefined-symbol errors at final link: removing `-rdc=true` breaks cross-TU device calls; removing `-dlink` leaves `nvshmemi_*` device symbols unresolved; removing whole-archive drops the `__fatbinwrap_*_device_cu_*` registration stubs that NVSHMEM's own translation units emit for `init_device.cu` / `transfer_device.cu`.
- `--whole-archive` is scoped via `-Wl,--whole-archive <archive.a> -Wl,--no-whole-archive` around **only** `libnvshmem_device.a`. Don't pull other static libs into whole-archive scope — it will massively bloat the binary.
- NVSHMEM build-time dependency also pulls in `<infiniband/mlx5dv.h>` at the DeepEP kernel compile step (via `configs.cuh` → `nvshmem.h`). Ubuntu 22.04's `libibverbs-dev` + `libmlx5-1` provide this. On a fresh box add `apt install libibverbs-dev libmlx5-dev` before first build.

**Next steps (in order, for post-compact resumption):**

1. **Add LL/internode extern C wrappers in `csrc/deep_ep.cu`:**
   - `deep_ep_get_unique_id(uint8_t* out_id, size_t id_size)` → `deep_ep::internode::get_unique_id()`
   - `deep_ep_internode_init(const uint8_t* root_unique_id, size_t id_size, int rank, int num_ranks, bool low_latency_mode)` → `deep_ep::internode::init(...)`
   - `deep_ep_internode_finalize()`, `deep_ep_internode_barrier()`, `deep_ep_internode_alloc(size, alignment)`, `deep_ep_internode_free(ptr)`
   - `deep_ep_low_latency_dispatch(...)` / `deep_ep_low_latency_combine(...)` / `deep_ep_clean_low_latency_buffer(...)` — match the signatures in `third_party/DeepEP/csrc/kernels/api.cuh:273-347`.
   - `deep_ep.cu` will need to replace its `#include "api.cuh"` with one that doesn't rely on `DISABLE_NVSHMEM` gating — but it already passes `-rdc=true` + NVSHMEM includes per build.rs, so no code change needed for the include to stabilize. Just add the new extern "C" functions.

2. **Rust side: add NVSHMEM bootstrap in executor startup.** Before `DeepEpBuffer` construction: rank 0 calls `deep_ep_get_unique_id`, broadcast the 128B uniqueid to other 7 ranks via the existing thread channels in `src/model/dsv32/deep_ep.rs` (same pattern that distributes `cudaIpcMemHandle_t` today), each rank calls `deep_ep_internode_init(unique_id, rank, 8, /*low_latency_mode=*/true)`. For EP8 single-node with `num_ranks <= NUM_MAX_NVL_PEERS=8`, `runtime.cu:58-68` shows the code path skips `cpu_rdma_team` split entirely — no extra work needed.

3. **Allocate LL buffers.** DeepEP LL needs the recv buffer shape `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` allocated via `nvshmem_align`. `buffer.py:548-611` (Python reference) shows the full allocation set: `rdma_recv_x`, `rdma_recv_count`, `rdma_send_x`, `rdma_recv_flag`, `mask_buffer`, `workspace`.

4. **Wire LL into decode path.** `forward_moe_ep()` in `src/model/dsv32/forward.rs` currently calls `deep_ep_intranode_dispatch` / `deep_ep_cached_notify_combine` / `deep_ep_intranode_combine`. Swap to `deep_ep_low_latency_dispatch` / `deep_ep_low_latency_combine` for `seq_len == 1` (decode) only; keep intranode path for prefill initially.

5. **Env vars at runtime.** May need `NVSHMEM_IB_ENABLE_IBGDA=1` depending on whether LL kernels fall back to CPU-proxy RDMA without it. The smoke test passed at default settings (`IBGDA=0`, `P2P=1`) so P2P handled the atomic — but LL's recv-flag polling may strictly require IBGDA. Verify first call either works or fails loudly, then pin the required env in executor startup.

**Critical docs to re-read after compact (in priority order):**
- This file (§D2-smoke, §D2-integration, §D2 main) — decision context and verification state.
- `docs/areas/dsv32.md` §5-optimization-log entry and §9 #5 profile — bottleneck numbers that justify D2 in the first place.
- `docs/resources/h20-development-guide.md` §9.1 (gitignored, H20-local only) — NVSHMEM install path, link flags, smoke test binary location.
- `third_party/DeepEP/csrc/kernels/api.cuh:273-347` — LL kernel signatures that new wrappers must match.
- `third_party/DeepEP/csrc/kernels/runtime.cu:41-93` — NVSHMEM bootstrap pattern to wrap.
- `third_party/DeepEP/csrc/kernels/internode_ll.cu:715-930` — LL combine kernel semantics (this is what ultimately deletes `cached_notify_combine` from the profile).
- `third_party/DeepEP/deep_ep/buffer.py:548-636` — Python reference for LL buffer allocation and call sequence. Not executable in pegainfer, but the cleanest source of "what shapes/args do these kernels actually need."

---

## D3. CPU-side jitter / rank sync discipline

### What SGLang does
- **Process per rank.** Each `(tp_rank, ep_rank)` is a separate `scheduler` Python process (`sglang/.../managers/scheduler.py:3099` sets per-process `setproctitle("sglang::scheduler...")`). Independent allocators, independent GIL.
- **Optional CPU pinning, off by default.** `set_gpu_proc_affinity` at `sglang/.../utils/common.py:2227-2260` divides physical cores by TP-per-node and calls `psutil.Process().cpu_affinity(bind_cpu_ids)`. Triggered only if `SGLANG_SET_CPU_AFFINITY` env var is set (`scheduler.py:3109`).
- **Optional NUMA bind.** `--numa-node` server arg → `numa_bind_to_node(numa_node[gpu_id])` at `scheduler.py:3113-3116`.
- No explicit pre-MoE handshake / warmup — they rely on per-step DP all-reduce and DeepEP's own `notify_dispatch` to align.

### What vLLM does
- **Process per rank** via `multiproc_executor` (separate processes; not threads).
- **NUMA via numactl wrapper, not Python affinity.** `vllm/.../utils/numa_utils.py:173-207` builds `--cpunodebind=N --membind=N` (or `--physcpubind=...`) and prefixes the worker subprocess command. Auto-detected via NVML CPU affinity bitmap at `vllm/.../platforms/cuda.py:692-705,720-731`.
- Only NIXL KV worker explicitly pins via `os.sched_setaffinity` (`vllm/.../distributed/kv_transfer/.../worker.py:225-229`); MoE workers do not pin individual CPUs themselves — they inherit numactl from launch.

### Implication for pegainfer
- Pegainfer is single-process-multi-thread (8 EP ranks share one Tokio runtime, one Rust allocator, one Python-free CUDA driver). Confirmed by `forward_moe_ep` using one `ep_*` buffer set per rank (`src/model/dsv32/forward.rs:289-388`) inside the same struct.
- This is the most plausible source of the 391us p50 start-skew: any rank that picks up an OS interrupt, allocator lock, or async-runtime stall delays only itself, while the others have already arrived at the barrier.
- **Cheap experiment** before committing to D2: pin each rank's `cudaSetDevice` thread with `pthread_setaffinity_np` to a dedicated core (or core-pair on the same NUMA node as the GPU). If the skew drops, we have confirmation. SGLang's `set_gpu_proc_affinity` formula (`utils/common.py:2243-2256`) is a reasonable starting heuristic.
- **Bigger bet**: switch to process-per-rank. This is a load-bearing architectural change for pegainfer; not cheap, but matches both reference implementations and is the only way to fully eliminate cross-rank allocator contention.

---

## D4. Accumulator dtype in MoE output combine

### What SGLang does
- **fp32 accumulator.** `_moe_sum_reduce_kernel` at `sglang/.../layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:1024,1032`:
  ```
  accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
  for i in tl.range(0, topk_num, ...):
      tile = tl.load(...)
      accumulator += tile.to(tl.float32)
  accumulator *= routed_scaling_factor
  tl.store(..., accumulator.to(input_ptr.dtype.element_ty), ...)
  ```
  Single fp32 register per (token, dim), single bf16 cast at the end. Order-independent.
- This is the post-expert reduce (after `silu(gate) * up`-then-down, weights already applied per-row inside the matmul kernel — see `fused_moe.py:312`: `accumulator = accumulator * moe_weight[:, None]`).

### What vLLM does
- **bf16 accumulator** in the standalone CUDA kernel: `vllm/csrc/moe/moe_align_sum_kernels.cu:350-363` — `scalar_t x = 0.0; for k: x += input[...]; out = x;` — same bf16-rounding-between-adds semantic as pegainfer.
- vLLM relies on the per-expert GEMM Triton kernel having already done the topk-weight multiplication in fp32 (see `vllm/.../fused_moe/fused_moe.py:300`), so the moe_sum input is `weight[k] * row[k]` and the bf16 sum-across-k loses less than pegainfer's combine-add-after-DeepEP.

### Implication for pegainfer
- Pegainfer (`csrc/moe.cu:210-228` `moe_scatter_weighted_add_rows_kernel`) does the worst-case version: read dst as bf16, do `dst + weight * src` in fp32, **round back to bf16 every iteration**, and the iterations are scattered (different recv rows hit the same dst row), so the sum is non-deterministic AND order-dependent in precision.
- **Cheap fix**: stage `[bs, topk, hidden]` of weighted bf16 contributions, then run an SGLang-style fp32-accumulator reduce kernel. This matches SGLang behavior 1:1, removes order-dependence, and unlocks the kind of expert-major batching you'd want for a future fused dispatch+combine path.
- Caveat: this requires `[bs, topk, hidden]` of staging memory (~`2 * bs * topk * 7168` bytes) instead of the in-place scatter. For bs=1 decode, that's 100KB — negligible.

---

## D5. Dispatch/combine overlap with compute

### What DeepEP supports
- **Yes, user-supplied stream control.** `Buffer.dispatch` / `combine` accept `allocate_on_comm_stream: bool` (`buffer.py:295,330,410`). When set, output tensor ownership is transferred to the internal comm stream; the caller can then `record_event()` and synchronize their compute stream against it for fine-grained overlap.
- For LL, `low_latency_dispatch` / `combine` accept `async_finish: bool` and `return_recv_hook: bool` (`buffer.py:553,618`). With `return_recv_hook=True`, the kernel only issues the RDMA writes; the caller invokes the returned `hook()` later to actually wait for arrival. This is the primitive for compute/combine overlap.

### What SGLang does
- **Shared-experts overlap with dispatch on `alt_stream`.** `sglang/.../models/deepseek_v2.py:776-781`:
  ```
  self.alt_stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(self.alt_stream):
      shared_output = self._forward_shared_experts(hidden_states)
  ```
  When `sbo_overlap_dispatch_flag` is on (`:795-803`), shared-experts are kicked into the `_deepep_dispatch_hook` callback which runs them concurrently with the dispatch all-to-all.
- **Two-Batch Overlap (TBO)** is a heavier scheme: `sglang/python/sglang/srt/batch_overlap/two_batch_overlap.py` splits the batch in two and overlaps each half's dispatch with the other half's compute. Enable: `--enable-two-batch-overlap` (`sglang/.../server_args.py:600`).

### What vLLM does
- vLLM's DBO (Dual-Batch Overlap) machinery in `vllm/.../fused_moe/prepare_finalize/deepep_ll.py:254,405-417` (`a2a_idx = dbo_current_ubatch_id()`, `dbo_maybe_run_recv_hook()`) — dual ubatch ping-pong using `low_latency_combine`'s `return_recv_hook=True`.

### Implication for pegainfer
- **Lowest priority for bs=1.** TBO assumes you have ≥2 batches' worth of independent work to interleave. Single-request decode has no shared-expert overlap opportunity at all (DSV3.2 has no shared experts of consequence in the routed MoE — they're all routed). The only win would be overlapping layer N's combine with layer N+1's `RMSNorm + Q proj`, which is bounded by how much of layer N+1 doesn't depend on layer N's output (answer: only the residual, ~1us).
- Real value here unlocks once we move to continuous batching. Not the first lever.

---

## D6. (no high-confidence cross-cutting finding)

Skipped per quality-bar instruction. Everything else read (router kernels, MLA absorption, RMSNorm fusion) is either already done in pegainfer or so small it would not move the #5 profile.

---

## Recommended ranking

1. **D2 + D1 as a coupled work item** — adopting DeepEP LL forces the recv layout that DeepGEMM masked wants. Doing them together lands the structural fix to `cached_notify_combine` (35% slice deleted) AND replaces the per-expert `fp8_gemm` calls with one masked grouped-GEMM (42% slice → ~half). Cost: NVSHMEM init in Rust, recv-buffer reshape, masked-GEMM wiring, and a `silu_and_mul_masked_post_quant`-equivalent. Estimated 2-3 weeks; expected GPU-time impact 40-60% of decode.
2. **D3 (CPU pinning, single-process variant first)** — 1-day experiment: pin each EP rank's CUDA-driving thread via `pthread_setaffinity_np` to a NUMA-local core. If the 391us skew drops, ship it; if it doesn't, that's a strong signal we need process-per-rank, which is a much bigger discussion. Either outcome is worth knowing before #1.
3. **D4 (fp32 accumulator)** — 1-day kernel rewrite, removes a real correctness/stability hazard (the 4B+ token accumulation rounding bug analog from Qwen3.5 work), small but free perf win because we batch bs=1 anyway.
4. **D5 (compute/combine overlap)** — defer until continuous batching lands. For bs=1 the pre-LL barrier wait is so much bigger than any compute we could overlap that it's noise.

---

## Open questions / to verify on H20

- ~~Does NVSHMEM init succeed on our 8×H20 single-node configuration?~~ **Resolved 2026-04-21**: yes on both NVLink P2P and IBGDA/RoCE paths. The box has 5× mlx5 RoCE HCAs (`link_layer: Ethernet`, NUMA-aligned to GPU pairs), not NVLink-only as earlier assumed. See §D2-smoke.
- DeepGEMM masked kernel's `M_max` block-size constraint (`sglang/.../ep_moe/kernels.py:1063` mentions `block_m`). We need to know what `M_max` will actually be for our `bs=1, topk=8, EP=8` and whether the rounding overhead is meaningful.
- What is the actual per-rank wall-clock time spent inside `deep_ep_intranode_combine`'s receive-side spin (vs the cnc barrier)? If the receive-spin itself is significant under jitter, LL won't help as much as expected — only if the bulk of the 35% is the explicit `barrier_block` will the benefit be ~full.
- Whether pegainfer's tokio runtime will tolerate `pthread_setaffinity_np` on driver threads, or whether we need to pull driver work off tokio entirely.
- Whether DeepEP's LL recv-flag spin + NVSHMEM's own worker threads + pegainfer's tokio workers can share the same physical cores without introducing new jitter (adjacent to D3).
