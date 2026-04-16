# DSV3 Offline Accuracy Audit

**Created**: 2026-04-16
**Status**: complete
**TL;DR**: Broad DSV3 offline audit remains split into focused contract slices. The latest local `vllm` comparison keeps the highest-risk post-embedding gaps concentrated in DeepEP combine weighting/index semantics, absorbed MLA `kv_b_proj` materialization semantics, and missing FP8 block-scale validation.

## Preparation

- **Read**:
  - `docs/index.md` — confirmed there is an active broad DSV3 bring-up doc and an existing accuracy-parity workflow reference.
  - `docs/projects/dsv3-inference.md` — established current DSV3 status: 61-layer forward runs on 8x H20, combine timeout is fixed, remaining gap is logits parity with likely suspicion around routed MoE / combine weighting.
  - `docs/resources/accuracy-parity-playbook.md` — reinforced the debugging order: define truth source, find first diff, then only split the segment that actually drifts.
- **Relevant history**:
  - `docs/projects/dsv3-inference.md` — recorded prior confirmed DSV3 bugs in DeepEP parameter wiring, `num_channels` vs `num_sms`, and combine input/output buffer mismatches; these are high-risk areas for the current offline audit.
  - `docs/resources/accuracy-parity-playbook.md` — warns against comparing the whole stack at once and against debugging already-diverged token traces; useful guardrail while 8-GPU repro is unavailable.
- **Plan**:
  1. Build a post-embedding operator map for the DSV3 path in pegainfer, covering shapes, dtypes, scale semantics, and layout assumptions from RMSNorm through final `lm_head`.
  2. Spawn four parallel `gpt-5.4` `high` sub-agents to compare pegainfer against `vllm`, `third_party/FlashMLA`, `third_party/DeepGEMM`, and `third_party/DeepEP`, each focused only on the contract and parameter semantics of its assigned implementation.
  3. Merge the sub-agent findings into a ranked shortlist of the most likely parity risks that can explain the current logits mismatch without needing 8-GPU execution.
  4. Add lightweight local checks where feasible on the current machine: compile-level validation, small-shape contract checks, or single-operator probes that do not require full 8-GPU replay.
- **Risks / open questions**:
  - Without 8 GPUs, we may only be able to narrow the fault domain and validate operator contracts, not fully prove the final root cause.
  - The current DSV3 path mixes decode-kernel reuse, sparse-prefill bring-up, FP8 scaling, and EP8 communication; a code-level mismatch may still depend on multi-rank runtime behavior.
  - Focused DeepEP / MoE semantic review still needs an explicit pass over the exact handoff between pegainfer `deep_ep.rs` / `forward.rs` / `executor.rs` / CUDA wrappers and `third_party/DeepEP`, especially around dispatch-vs-combine buffer spaces and pre-weighted combine semantics.

### Focused Pass: DeepEP / MoE Dispatch-Combine Semantics (2026-04-16)

- **Read**:
  - `docs/index.md` — confirmed this focused audit should reuse the active DSV3 offline accuracy doc instead of creating a duplicate.
  - `docs/projects/dsv3-inference.md` — recorded previously fixed DeepEP integration bugs and highlighted the still-open suspicion around combine weighting.
  - `docs/resources/accuracy-parity-playbook.md` — constrained the task to contract-level first-diff analysis instead of broad stack speculation.
- **Relevant history**:
  - `docs/projects/dsv3-inference.md` — prior fixes already touched `cached_notify_combine`, recv-space `topk_weights`, and dispatch/combine token-count parameter swaps, so this pass should treat all remaining semantics around these interfaces as high risk rather than re-auditing unrelated kernels first.
  - `docs/projects/dsv3-offline-accuracy-audit.md` — the existing offline audit already scoped DeepEP as one of four external contract reviews; this pass is the pegainfer-side consolidation for that item.
- **Plan**:
  1. Compare pegainfer `src/model/dsv3/deep_ep.rs`, `src/model/dsv3/forward.rs`, `src/model/dsv3/executor.rs`, `src/ffi.rs`, `csrc/deep_ep.cu`, and `csrc/moe.cu` against the matching `third_party/DeepEP` intranode APIs and reference MoE call pattern, extracting exact parameter, buffer-space, and ordering contracts.
  2. Trace the routed-MoE dataflow end to end: routing logits/top-k selection and normalization, dispatch metadata generation, recv/send tensor ownership, expert compute input/output spaces, combine invocation, and final output weighting assumptions.
  3. Rank the top five suspicious mismatches or assumptions specifically around token counts, dispatch vs combine spaces, `topk_weights` semantics, pre-weighting vs post-weighting, normalization location, prefix matrices, ownership, and barrier/order guarantees.
  4. For each ranked item, classify what can be checked on a single 16GB GPU by local shape/contract inspection or unit-style probes, versus what inherently requires multi-GPU execution to validate.
- **Risks / open questions**:
  - `third_party/DeepEP` may encode some assumptions in Python wrappers or test code rather than in the CUDA wrapper entrypoints alone, so the audit may need to follow the call chain beyond the raw kernels.
  - Some correctness hazards only manifest when rank-local send/recv counts diverge, so single-GPU inspection can confirm contract mismatches but not fully falsify all multi-rank ordering bugs.

### 2026-04-16 DeepGEMM / FP8 Contract Slice

- **Read**:
  - `docs/index.md` — confirmed this investigation should stay inside the active DSV3 accuracy audit doc instead of creating a parallel project file.
  - `docs/projects/dsv3-offline-accuracy-audit.md` — reused the active DSV3 audit as the canonical place to log this narrower FP8 / DeepGEMM contract pass.
  - `docs/resources/accuracy-parity-playbook.md` — constrained the work to contract-first comparison and a ranked suspicion list rather than speculating past the first credible mismatch.
- **Relevant history**:
  - `docs/projects/dsv3-offline-accuracy-audit.md` — prior audit already isolated FP8 scaling and external-kernel contract review as a likely parity-risk area worth an offline pass on a single 16GB GPU.
  - `docs/projects/dsv3-inference.md` — existing DSV3 bring-up history shows multiple bugs came from parameter/shape contract mismatches, so this pass should aggressively verify layout and semantic assumptions.
- **Plan**:
  1. Read the pegainfer FP8/DeepGEMM integration files named by the task: `csrc/fp8_gemm.cu`, `csrc/fp8_quantize.cu`, `src/ops/fp8.rs`, `src/model/dsv3/weights.rs`, `src/weight_loader.rs`, plus DSV3 `forward.rs` call sites.
  2. Read the matching `third_party/DeepGEMM` code paths for grouped contiguous-layout GEMM, scale layout, TMA descriptor setup, and epilogue expectations; only pull local `vLLM` FP8 usage as a helper reference where pegainfer-vs-DeepGEMM semantics remain ambiguous.
  3. Produce a ranked top-5 mismatch shortlist with exact file references, why each could move logits, and which checks are feasible on a single 16GB GPU without requiring the 8-GPU full DSV3 run.
- **Risks / open questions**:
  - Some DeepGEMM contracts are encoded indirectly through template parameters, layout tags, or scale-shape assertions rather than one central spec file, so the comparison may require stitching multiple call layers together.
  - A few accuracy-impacting mismatches may only become visible when routed experts exercise grouped GEMM shapes that are hard to reproduce end-to-end on one 16GB GPU; those cases may need contract probes instead of full-model validation.

### 2026-04-16 FlashMLA Contract Slice

- **Read**:
  - `docs/index.md` — confirmed this FlashMLA-only pass should stay attached to the active DSV3 offline audit.
  - `docs/projects/dsv3-inference.md` — recovered the existing bring-up notes for paged MLA KV layout, dense decode integration, and the current sparse-prefill bring-up status.
  - `docs/resources/accuracy-parity-playbook.md` — kept the work scoped to contract-level first-diff suspects instead of speculative whole-stack debugging.
- **Relevant history**:
  - `docs/projects/dsv3-inference.md` — already records that sparse prefill is unfinished and that decode currently reuses FlashMLA dense decode with paged BF16 KV.
  - `src/model/dsv3/executor.rs` — contains an existing inline warning that sparse prefill has a `topk_length` issue for short sequences, which is directly relevant to FlashMLA sparse-prefill semantics.
- **Plan**:
  1. Compare pegainfer `csrc/flash_mla.cu`, `csrc/flash_mla_prefill.cu`, `csrc/mla.cu`, `src/model/dsv3/mla_kv.rs`, and `src/model/dsv3/forward.rs` against `third_party/FlashMLA` interface and kernel-side contract points.
  2. Rank the top five contract mismatches or risky assumptions in metadata layout, KV cache layout, head dims, page/block layout, sparse-prefill index semantics, padding, RoPE packing, softmax scaling, and output combine/de-absorb sequencing.
  3. Classify which checks can run on the local single 16GB RTX 5070 Ti and which require SM90 hardware or multi-GPU replay.
- **Risks / open questions**:
  - The local GPU is SM120, so FlashMLA's SM90 kernel bodies cannot be run directly here; some checks must remain static or shape-level only.
  - A few items are true contract mismatches, while others are high-risk caller assumptions that only become wrong if the truth source uses a different scaling or scheduling policy.

### 2026-04-16 Local vLLM Contract Comparison Slice

- **Read**:
  - `docs/index.md` — confirmed the active DSV3 accuracy audit doc should be reused for this narrower comparison pass.
  - `docs/projects/dsv3-offline-accuracy-audit.md` — reused prior DSV3 focused findings so this pass can avoid re-auditing already-ranked DeepEP issues outside the user’s requested scope.
  - `docs/resources/accuracy-parity-playbook.md` — constrained the work to truth-source contract comparison and first-diff-style narrowing rather than broad architectural summary.
- **Relevant history**:
  - `docs/projects/dsv3-offline-accuracy-audit.md` — prior focused passes already identified likely DeepEP and FP8 risk areas, so this comparison should only check whether the local `vllm` checkout confirms or contradicts those specific suspicions.
  - `docs/projects/dsv3-inference.md` — recorded that the remaining DSV3 gap is logits parity, which keeps final norm / lm_head and routed-path weighting semantics in scope.
- **Plan**:
  1. Read pegainfer DSV3 files that define post-embedding semantics for MLA/attention, MoE routing and combine weighting, final norm / `lm_head`, and FP8 call sites.
  2. Read the matching local `vllm` DeepSeek-V3 implementation under `/data/code/workspace-rustllm/vllm`, limiting the pass to the same contract surfaces rather than summarizing the whole model stack.
  3. Rank the top five most suspicious pegainfer-vs-vLLM mismatches with exact file references on both sides, one short reason each, and mark which checks are feasible on a single 16 GB GPU.
- **Risks / open questions**:
  - The local `vllm` checkout may encode some behavior through fused/custom-op wrappers, so a few contracts may need to be inferred from adjacent call sites instead of one implementation file.
  - Some MLA and EP/MoE behaviors may require reading both Python model code and the custom-op interface to separate semantic mismatches from backend-specific execution details.

## Execution Log

### Step 1: Start offline audit and delegate external contract review
- User approved the project doc and explicitly requested heavy use of parallel `gpt-5.4 high` sub-agents instead of doing the comparison work on the main thread.
- Decision: keep the main thread focused on pegainfer-side operator-map consolidation and final integration of findings; delegate external reference analysis to four parallel sub-agents scoped to `vllm`, `FlashMLA`, `DeepGEMM`, and `DeepEP`.
- Delegated agents:
  - `Singer` — `vllm` contract review
  - `Helmholtz` — `FlashMLA` contract review
  - `Parfit` — `DeepGEMM` / FP8 contract review
  - `Beauvoir` — `DeepEP` / MoE dispatch-combine contract review
- Main-thread context gathered:
  - `src/model/dsv3/forward.rs` confirms two execution paths exist today: `forward_prefill` (token-by-token decode reuse) and `forward_prefill_sparse` (batch sparse prefill with NSA indexer).
  - `src/model/dsv3/forward.rs` and `src/model/dsv3/weights.rs` expose the likely accuracy-sensitive handoff points: FP8 quant + GEMM, absorbed MLA weights, NSA indexer buffers, routed MoE weights, and DeepEP recv/combine buffer spaces.
- Sub-agent control notes:
  - Two agents initially stopped at a `project-doc` review gate inherited from repo skills; user approval was forwarded and both were explicitly told to continue execution without stopping again.
  - No external findings have returned yet; current status is active parallel analysis rather than a blocked state.
- Result: in progress

### Step 2: First external findings — DeepEP / routed MoE contract review
- `Beauvoir` returned the first high-signal result set focused on `DeepEP` / dispatch-combine semantics.
- High-confidence finding #1:
  - pegainfer treats `ep_recv_topk_idx` as a global expert id in `src/model/dsv3/forward.rs`, but DeepEP dispatch rewrites recv-side indices into per-rank local expert ids.
  - Impact: routed experts on nonzero ranks can be skipped or filtered incorrectly during local expert execution.
- High-confidence finding #2:
  - pegainfer only zero-initializes the recv token slot when `k == 0`, but the first local expert can appear at any top-k position after dispatch filtering.
  - Impact: expert outputs can accumulate on top of stale input activations instead of starting from zero.
- Additional risk findings:
  - pre-weighting vs combine-weight semantics may be mismatched, especially since `ep_combined_topk_weights` is produced but not consumed.
  - pegainfer reimplements part of DeepEP's NVL buffer / memset contract manually, which raises silent-metadata-corruption risk.
  - several dispatch/combine buffer dimensions are hardcoded to EP8 assumptions in forward buffers.
- Result: partial success; findings recorded, pending corroboration from `vllm`, `FlashMLA`, and `DeepGEMM` agents.

### Step 3: FP8 / DeepGEMM contract review
- `Parfit` returned a focused FP8 / DeepGEMM comparison.
- Most suspicious FP8-side finding:
  - `kv_b_proj` is materialized into bf16 during weight loading for the absorbed MLA path, instead of preserving the live DeepGEMM semantics of on-kernel dequantization with f32 accumulation and a single final bf16 store.
  - Impact: MLA absorption / de-absorption can drift numerically even if the main FP8 linear path is otherwise correct.
- Additional FP8-side risks:
  - pegainfer's custom `fp8_e4m3_to_f32` may not match PyTorch / CUDA `float8_e4m3fn` edge semantics exactly.
  - `weight_scale_inv` shape/orientation is trusted but not strongly validated against DeepGEMM's `Major::K` expectation.
  - `config.weight_block_size` is parsed, but the live kernel path is effectively hardcoded to `128x128`.
  - activation quantization and TMA descriptor setup are hand-written and lack a dedicated helper-level parity test against DeepGEMM reference helpers.
- Result: partial success; strongest new lead is the absorbed `kv_b_proj` path, pending corroboration from `vllm` and `FlashMLA`.

### Step 4: FlashMLA / sparse prefill contract review
- `Helmholtz` returned the MLA-specific comparison.
- Highest-signal finding:
  - pegainfer's NSA indexer writes padded sparse-prefill indices as `0` for short sequences, but FlashMLA sparse prefill treats `-1` or `>= s_kv` as invalid entries, and pegainfer does not currently supply `topk_length`.
  - Impact: early queries can repeatedly attend token `0` instead of masking padded sparse edges, creating stable but misleading numerical drift.
- Additional MLA-side risks:
  - `dense decode` hardcodes `num_sm_parts = 72`, while FlashMLA upstream derives split scheduling from an SM-aware formula.
  - split-KV scratch sizing assumes only `num_sm_parts`, whereas upstream sizes against `batch_size + num_sm_parts`.
  - `softmax_scale / softmax_mscale` are caller-owned semantics; FlashMLA will not correct a wrong scale convention.
  - `[NoPE(512), RoPE(64)]` packing is manually maintained in pegainfer without a strong contract assertion.
- Result: partial success; the sparse-prefill invalid-index semantic mismatch is now a top-tier accuracy suspect alongside DeepEP routed-MoE issues.

### Step 2: Read pegainfer FP8 / DeepGEMM integration end-to-end
- Opened `csrc/fp8_gemm.cu`, `csrc/fp8_quantize.cu`, `src/ops/fp8.rs`, `src/model/dsv3/weights.rs`, `src/weight_loader.rs`, `src/model/dsv3/forward.rs`, and `src/ffi.rs`.
- Confirmed the live FP8 linear path is:
  1. bf16 `HiddenStates` interpreted as row-major `[M, K]`
  2. `fp8_quantize_1x128_cuda` produces FP8 activations plus `scale_a`
  3. `fp8_gemm_cuda` launches DeepGEMM SM90 1D2D with `kMajorSFB = Major::K`
  4. kernel writes bf16 output directly
- Confirmed the absorbed MLA path is special: `kv_b_proj` is not run through DeepGEMM at inference time; it is dequantized on CPU to bf16 once during loading, then used by bf16 batched GEMMs.
- Result: success

### Step 3: Compare pegainfer assumptions against DeepGEMM’s stated contracts
- Opened `third_party/DeepGEMM/README.md`, `third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`, `third_party/DeepGEMM/deep_gemm/include/deep_gemm/common/scheduler.cuh`, `third_party/DeepGEMM/deep_gemm/utils/math.py`, `third_party/DeepGEMM/tests/test_layout.py`, and `third_party/DeepGEMM/tests/generators.py`.
- Verified the core SM90 1D2D contract:
  - A/B are K-major row-major FP8
  - `scale_a` is transposed/TMA-aligned FP32 with M as inner dimension
  - `scale_b` is consumed according to `kMajorSFB`; for `Major::K`, K-chunk is contiguous
  - scale promotion happens inside the kernel before a final bf16 store
- Noted that pegainfer hand-rolls the TMA descriptors and scale layouts instead of using DeepGEMM’s layout helpers, so the integration has no external validation layer.
- Result: success

### Step 4: Check what can be validated on the available single-GPU machine
- Queried environment:
  - `PEGAINFER_DSV3_MODEL_PATH` is unset.
  - Local GPU is `NVIDIA GeForce RTX 5070 Ti, 16303 MiB, compute_cap=12.0`.
- Confirmed from `build.rs` that DeepGEMM FP8 wrapper compilation is gated behind SM90 targets, so this machine is suitable for source-level and CPU-side contract checks but not for directly exercising the current SM90a FP8 wrapper path.
- Identified existing ignored single-model tests in `src/model/dsv3/weights.rs` that become usable once a model path and suitable runtime target are available.
- Result: success

### Step 5: Compare pegainfer FlashMLA integration against local `third_party/FlashMLA`
- Opened pegainfer-side files `csrc/flash_mla.cu`, `csrc/flash_mla_prefill.cu`, `csrc/mla.cu`, `src/model/dsv3/mla_kv.rs`, `src/model/dsv3/forward.rs`, `csrc/nsa_indexer.cu`, and `src/model/dsv3/config.rs`.
- Opened upstream-side files `third_party/FlashMLA/flash_mla/flash_mla_interface.py`, `third_party/FlashMLA/README.md`, `third_party/FlashMLA/csrc/api/dense_decode.h`, `third_party/FlashMLA/csrc/api/sparse_fwd.h`, `third_party/FlashMLA/csrc/params.h`, `third_party/FlashMLA/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`, `third_party/FlashMLA/csrc/smxx/decode/combine/combine.cu`, `third_party/FlashMLA/csrc/sm90/prefill/sparse/phase1.cuh`, and `third_party/FlashMLA/tests/ref.py`.
- High-confidence findings:
  - Sparse prefill invalid-index contract is violated today: pegainfer pads unused indices with `0` in `csrc/nsa_indexer.cu`, but FlashMLA sparse prefill only treats `-1` or `>= s_kv` as invalid when `topk_length` is not provided.
  - Dense decode scheduler partitioning differs from upstream because pegainfer hardcodes `num_sm_parts = 72`, while upstream derives it from device SM count and shape.
  - Split-KV accum buffers are sized to `num_sm_parts` rather than upstream's conservative `batch_size + num_sm_parts` upper bound.
  - FlashMLA is scale-agnostic; pegainfer's DeepSeek-specific `softmax_scale` override is plausible but must match the truth source exactly or every attention logit drifts.
- Additional environment note:
  - The current machine is `NVIDIA GeForce RTX 5070 Ti, 16303 MiB, compute_cap=12.0`, so SM90 FlashMLA kernels cannot be executed locally; only static or helper-level checks are immediately available here.
- Result: success

### Step 6: Compare pegainfer DSV3 post-embedding semantics against the local `vllm` checkout
- Opened pegainfer files:
  - `src/model/dsv3/forward.rs`
  - `src/model/dsv3/weights.rs`
  - `src/ops/fp8.rs`
  - `csrc/fp8_quantize.cu`
  - `csrc/moe.cu`
- Opened local `vllm` files:
  - `/data/code/workspace-rustllm/vllm/vllm/model_executor/models/bailing_moe_linear.py`
  - `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/mla.py`
  - `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/attention/mla_attention.py`
  - `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`
  - `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py`
  - `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py`
  - `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py`
  - `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py`
  - `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/linear.py`
  - `/data/code/workspace-rustllm/vllm/vllm/model_executor/layers/logits_processor.py`
- Ranked the highest-signal mismatches:
  1. Pegainfer pre-weights routed expert outputs before `DeepEP` combine and still passes nontrivial `topk_weights`, while local `vllm` delegates weight application/reduction to `low_latency_combine`.
  2. Pegainfer interprets recv-side expert ids as global contiguous ids, while local `vllm` explicitly maps global ids into DeepEP physical/local expert space.
  3. Pegainfer zeros a recv accumulation slot only when `k == 0`, so a token whose first local expert is not top-k slot 0 can accumulate onto stale data.
  4. Pegainfer’s absorbed MLA path depends on a handwritten CPU FP8 e4m3 decode plus bf16 materialization, while local `vllm` dequantizes through the quant layer and caches transposed `W_UK` / `W_UV`.
  5. Pegainfer loads FP8 `weight_scale_inv` tensors with minimal shape/layout validation, while local `vllm` validates block shapes and normalizes FP8 scale parameter naming before kernel use.
- Single-16GB-GPU feasibility:
  - Items 3, 4, and 5 are directly checkable with local probes or one-layer/operator tests.
  - Items 1 and 2 are strongly supported by source inspection and small mocked checks, but full runtime proof still needs multi-rank `DeepEP`.
- Result: success

## Debrief
- **Outcome**: Completed the requested offline contract slices for both DeepGEMM/FP8 and FlashMLA, each with a ranked shortlist of concrete parity risks tied to exact source locations.
- **Pitfalls encountered**:
  - No local `third_party/vllm` checkout exists, so the comparison stayed anchored on DeepGEMM itself plus the repository’s own DSV3 reference scripts.
  - The current workstation is a 16 GB Blackwell consumer GPU, while the FP8 wrapper under review is explicitly an SM90a path, so runtime validation had to be framed as “possible with model path / target hardware” versus “possible immediately here”.
  - FlashMLA has different contracts for dense decode, sparse decode, and sparse prefill; mixing these paths leads to false positives.
- **Lessons learned**:
  - The biggest parity risk is not the visible `fp8_gemm_cuda` call itself, but the split contract between “live FP8 dequant inside DeepGEMM” and “early bf16 materialization for absorbed MLA weights.”
  - `load_fp8_matrix` currently trusts checkpoint scale tensors too much; even a one-axis swap would silently poison every FP8 projection.
  - For FlashMLA sparse prefill, invalid-index representation and `topk_length` are first-class parts of the contract, not optional hygiene.
  - For FlashMLA dense decode, `num_sm_parts` changes split-KV reduction order and therefore belongs in accuracy review, not only performance tuning.
  - The local `vllm` checkout reinforces that pegainfer’s riskiest remaining DSV3 drift is still routed-MoE EP semantics, especially where weights are applied and how expert ids are interpreted after dispatch.
- **Follow-ups**:
  - Add explicit assertions that every FP8 `weight_scale_inv` shape matches `[ceil(rows/128), ceil(cols/128)]` and that `config.weight_block_size == [128, 128]` before any DeepGEMM path is used.
  - Add a parity test that compares `dequant_fp8_to_bf16_host` against PyTorch `float8_e4m3fn` for sampled tensors, especially `kv_b_proj`.
  - Fix sparse-prefill padding semantics before trusting short-sequence FlashMLA sparse-prefill outputs.
  - Add a contract probe that computes upstream-style `num_sm_parts` and compares dense-decode split metadata against pegainfer's hardcoded `72`.
  - Add a focused `DeepEP` parity probe that verifies combine consumes router weights exactly once and that recv expert ids are decoded in the same space as local expert storage.

### 2026-04-16 DeepEP / MoE Dispatch-Combine Semantic Slice
- **Outcome**: Completed a focused pegainfer-vs-DeepEP audit for routed-MoE dispatch/combine semantics and isolated five concrete accuracy-risk mismatches or assumptions.
- **Pitfalls encountered**:
  - DeepEP’s authoritative semantics are split across wrapper docs, runtime checks, kernel comments, and `tests/test_intranode.py`; reading only the wrapper signatures would have missed the strongest bugs.
  - The largest risks sit in pegainfer’s post-dispatch local expert loop, not in the C wrapper signatures themselves.
- **Lessons learned**:
  - `recv_topk_idx` coming out of DeepEP dispatch is rank-local expert space, not global expert ids.
  - If the framework chooses a pre-weighted combine path, it must still respect DeepEP’s recv-space tensor semantics and cannot key “first local expert” logic off the original top-k slot order.
- **Follow-ups**:
  - Fix `src/model/dsv3/forward.rs` so `ep_recv_topk_idx` is interpreted as local expert ids and the recv slot is zeroed on the first valid local expert, not only when `k == 0`.
  - Add a focused EP8 probe modeled on `third_party/DeepEP/tests/test_intranode.py` that validates dispatched `recv_topk_idx`, `recv_topk_weights`, and combined outputs against the expected invariants.
