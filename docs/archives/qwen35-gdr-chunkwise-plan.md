# Qwen3.5 GDR Chunk-Wise Plan

> **TL;DR:** The chunk-wise GDR prefill plan landed in the real Qwen3.5 runtime, restored `e2e_qwen35`, and brought `(2048,1)` TTFT down to the `~222ms` range. This doc is archived because the plan has been executed and its outcome is now reflected in the broader Qwen3.5 optimization record.
>
> **Status:** Archived. The plan is complete and superseded by the runtime state documented in `projects/qwen35-4b-optimization.md`.

## Goal

Replace Qwen3.5-4B linear-attention prefill's current fused-recurrent GDR path with a chunk-wise implementation that materially reduces TTFT on prefill-heavy workloads, while preserving the existing decode path and recurrent-state semantics.

For the first pass, the target is deliberately narrow:

- batch size `1`
- forward-only
- Qwen3.5 fixed shapes only
- no varlen support yet

Success means:

- prefill-heavy TTFT at `(2048, 1)` improves meaningfully beyond the current `~378ms`
- standalone GDR prefill time drops well below the current `~7.24ms @ seq=2048`
- decode TPOT remains unchanged
- final recurrent state remains compatible with the existing decode kernel

## Current State

The current baseline is already much better than the original per-token implementation:

- TTFT `(2048, 1)` dropped from `3.89s` to `~378ms`
- the standalone `gated_delta_rule_prefill_into` microbench is `~7.24ms` at `seq=2048`
- `nsys` shows the old launch / copy / alloc churn is gone

The problem has changed. The bottleneck is now the kernel itself:

- `gated_delta_rule_prefill_kernel` is `48.2%` of prefill GPU time
- batched GEMM is another `~44%`
- `ncu` shows the kernel is not DRAM-bound:
  - `Registers Per Thread = 254`
  - `Block Limit Registers = 2`
  - `Theoretical Occupancy = 16.67%`
  - `Achieved Occupancy = 14.19%`
  - `Achieved Active Warps / SM = 6.81`
  - `DRAM Throughput = 0.75%`

Conclusion: the current fused-recurrent kernel is not suffering from host overhead anymore. It is limited by its own execution shape, especially register pressure and low occupancy.

There is now also a concrete in-tree chunk-wise feasibility signal:

- chunk-wise scratch / contract code compiles in pegainfer
- multi-stage Triton AOT build scaffolding compiles in pegainfer
- `chunk_state` and `chunk_o` stage kernels compile through Triton AOT
- both stage kernels launch successfully on synthetic tensors at Qwen3.5 shapes
- on the same prepared `q / k / w / u / g / state` tensors, the in-tree stage kernels roughly align with FLA stage references:
  - `v_new` max abs diff `~4.26e-2`
  - `final_state` max abs diff `~8.09e-2`
  - `chunk_state` max abs diff `~7.08e-2`
  - `output` max abs diff `~6.88e-3`

These are not end-to-end guarantees yet, but they are strong evidence that the current stage boundaries and state layout are viable.

There is now also a mixed full-pipeline validation at realistic gate distribution (`g < 0`, cumulative within chunk), using FLA for the prep stages and the in-tree kernels for the recurrent and output stages:

- output is finite at `seq=2048`
- final recurrent state is exact after aligning layouts:
  - FLA full chunk-wise returns `[H, K, V]`
  - pegainfer decode expects `[H, V, K]`
  - after transposing the FLA state to `[H, V, K]`, max / mean diff is `0.0`
- output drift remains small:
  - output max abs diff `~7.37e-2`
  - output mean abs diff `~3.22e-3`

This is the strongest current feasibility signal because it exercises the actual stage boundary we care about: third-party-style chunk prep feeding the in-tree decode-compatible state/output stages.

The follow-up localization work also found the main correctness bug in the in-tree path:

- `chunk_state` snapshots and final state layout were correct
- `chunk_o` itself was mathematically aligned with FLA when fed the same `h / v_new / g`
- the real drift came from `gdr_chunk_state_qwen35_kernel` writing `v_new` after multiplying by `exp(g_last - g_t)`
- after fixing the write order, stage-level alignment tightened to:
  - `v_new` diff vs FLA: `max ~1.95e-3`
  - `chunk_o` output diff vs FLA: `max ~1.22e-4`
  - final recurrent state diff after layout alignment: `0.0`

Stage-level runtime evidence also looks healthy on synthetic prepared tensors at Qwen3.5 shape (`seq=2048`, `H=32`, `K=V=128`):

- in-tree `chunk_state + chunk_o`: `~0.35ms`
- FLA `chunk_state + chunk_o` reference stages: `~0.26ms`
- FLA prep (`g_cumsum + A + solve_tril + w/u`): `~0.25ms`
- FLA full chunk-wise path: `~0.47ms`
- FLA fused recurrent reference: `~2.46ms`

This does not prove the full in-tree path will land exactly at FLA numbers, but it does show the current stage kernels are already in the right performance regime and far below the old fused-recurrent scale.

The mixed full pipeline also benchmarks well at `seq=2048` on synthetic Qwen3.5-shaped tensors:

- FLA prep + in-tree `chunk_state + chunk_o`: `~0.37ms`
- FLA full chunk-wise path: `~0.45ms`

This matters because it shows the in-tree back half is already not the bottleneck. The remaining engineering risk is concentrated in the prep stages, especially `solve_tril` and `w/u` generation.

## Working Conclusions

1. Chunk-wise prefill is not an optional micro-optimization.
It is the first credible path to a large next-step improvement. Continuing to tune the current fused-recurrent kernel may still help, but that path is unlikely to close the remaining gap by itself.

2. This is mainly an algorithm-shape problem, not a single-kernel tuning problem.
The current kernel keeps a full recurrent state tile live through a token loop. Chunk-wise reduces the time-axis serialization from token granularity to chunk granularity and converts more work into matrix-style kernels.

3. We should not force chunk-wise into the current single-symbol AOT model.
The chunk-wise path is naturally a multi-stage pipeline with explicit scratch. Trying to hide that under one monolithic symbol would make the implementation harder to reason about and harder to tune.

4. The first implementation should be narrow and explicit.
Batch=`1`, fixed Qwen3.5 shape, forward-only, no varlen. The point of the first version is to establish the runtime contract and the performance shape.

5. The in-tree back half is already good enough to keep.
`chunk_state` / `chunk_o` are not the main blocker anymore. They already preserve the decode-compatible final-state contract and are in the right performance regime. The remaining question is how aggressively to own the prep stages.

6. FlashInfer is an engineering reference, not a drop-in target for this machine.
Its current GDN prefill path is SM90-only CUDA. It is still useful for workspace, launcher, and state-continuation design.

7. A hand-written fixed-size `solve_tril` is possible, but it is not the obvious fastest route.
The first naïve in-tree `solve_tril` attempt ran into exactly the kind of Triton pain this project should avoid: compile complexity and awkward scalarized recurrence. That is evidence for a pragmatic first delivery strategy:

- keep the in-tree Qwen3.5-specific stage kernels
- minimally vendor or adapt the FLA prep kernels needed for fixed-length chunk size `64`
- only re-own those prep stages once the full path is integrated and benchmarked in Rust

## Qwen3.5 Assumptions

The first-pass implementation should assume the real Qwen3.5 linear-attention shape:

- `num_q_heads = 16`
- `num_k_heads = 16`
- `num_v_heads = 32`
- `key_dim = 128`
- `value_dim = 128`
- `chunk_size = 64`

Other important assumptions:

- `g = -exp(A_log) * softplus(a + dt_bias)`
- `beta = sigmoid(b)`
- q/k still require L2 normalization
- scale remains `1 / sqrt(key_dim)`
- current recurrent state is effectively `[num_value_heads, value_dim, key_dim]` in fp32, i.e. `[H, V, K]`

That final point matters: the prefill path may change, but the decode path currently expects the existing state contract to remain intact.

## Reference Paths

### Historical In-Tree Baseline

The original fused-recurrent path is no longer the live runtime path and its dedicated Triton kernel has been removed from tree after the chunk-wise rewrite landed. The useful surviving integration points for comparison are:

- `src/ops.rs`
- `src/qwen35_model.rs`
- `benches/ops/ops_qwen35_state_bench.rs`

Its role is now historical: it established the first structural win by removing host-side per-token orchestration, but it is no longer a maintained fallback.

### FlashInfer CUDA Reference

These files are useful references for how a chunk-wise runtime surface can look:

- `../flashinfer/flashinfer/jit/gdn.py`
- `../flashinfer/csrc/gdn_prefill_launcher.cu`
- `../flashinfer/csrc/prefill_kernel_delta_rule_sm90.cu`
- `../flashinfer/include/flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh`
- `../flashinfer/tests/gdn/test_prefill_delta_rule.py`

What to learn from this path:

- explicit launcher contract instead of hiding scratch internally
- explicit workspace buffer
- explicit `cu_seqlens` and continuation contract
- raw output-state layout and how continuation is tested
- how chunked prefill is chained across multiple calls

What not to assume:

- this is not a general CUDA reference for all devices
- the shipped implementation is SM90-only
- it should not be treated as a copy-paste drop-in for our Blackwell-focused path

### FLA Triton Reference

These files are the clearest chunk-wise algorithm reference on the Triton side:

- `../flash-linear-attention/fla/ops/gated_delta_rule/chunk.py`
- `../flash-linear-attention/fla/ops/common/chunk_delta_h.py`
- `../flash-linear-attention/fla/ops/common/chunk_o.py`
- `../flash-linear-attention/fla/ops/common/chunk_scaled_dot_kkt.py`
- `../flash-linear-attention/fla/ops/gated_delta_rule/wy_fast.py`
- `../flash-linear-attention/tests/ops/test_gated_delta.py`

What to learn from this path:

- the chunk-wise decomposition itself
- the split between `A / w / u` preparation, chunk-state recurrence, and output kernel
- how to test chunk-wise against recurrent semantics
- which parts of the pipeline naturally want their own kernels

Most likely first-v1 vendoring cut for fixed-length Qwen3.5:

- `solve_tril` fixed-length `BT=64` path
- `recompute_w_u_fwd_kernel`

The rest of the prep surface is already straightforward enough to own in-tree.

## Proposed Runtime Shape

The planned in-tree chunk-wise prefill path should look like this:

```text
QKV / A / B projections
  -> q/k l2 norm + head expansion
  -> g / beta preparation
  -> chunk-local cumulative gate
  -> A preparation
  -> solve_tril
  -> w / u preparation
  -> chunk_state kernel
  -> chunk_o kernel
  -> final state writeback
```

Expected scratch for the first implementation:

- `g_cumsum`
- `A`
- `w`
- `u`
- per-chunk state snapshots
- `v_new`

This is the main architectural shift relative to the current path: GDR prefill becomes a multi-stage operator with explicit scratch, not a single opaque kernel call.

## Task Breakdown

### Phase 0: Lock the Contract

Deliverables:

- define the exact tensor/layout contract for the chunk-wise prefill operator
- lock the recurrent-state layout that decode will continue to consume
- decide whether Q/K normalization lives inside the kernels or in a prep stage
- decide the scratch ownership model in Rust

Exit condition:

- one agreed input/output/scratch contract that both Rust and kernel code can target

### Phase 1: Runtime and Scratch Plumbing

Deliverables:

- add explicit scratch buffers for chunk-wise GDR prefill
- add operator entry points that can launch multiple kernel stages
- retire the temporary fused-recurrent path once chunk-wise is stable enough to own the runtime surface

Notes:

- this is where the current Triton AOT model likely needs to expand beyond one-symbol-per-op thinking
- the important design choice is not “Triton or CUDA”; it is whether the runtime can express a multi-stage prefill op cleanly

### Phase 2: Prep Stages

Deliverables:

- prepare `g`
- prepare `beta`
- prepare chunk-local cumulative gate
- prepare `A`
- solve lower-triangular system
- prepare `w` and `u`

Notes:

- this phase can initially prioritize clarity over optimal performance
- first-pass prep stages do not need to be the fastest part of the pipeline as long as they do not erase the chunk-wise gain

### Phase 3: Chunk-State Kernel

Deliverables:

- per-chunk recurrent state update
- optional intermediate chunk-state snapshot storage
- final state writeback in the existing runtime layout
- `v_new` production for the output stage

Notes:

- this kernel replaces the role currently played by the fused-recurrent token loop
- state layout discipline matters more than clever instruction scheduling in the first pass

### Phase 4: Chunk Output Kernel

Deliverables:

- inter-chunk contribution from chunk state
- intra-chunk causal contribution from prepared values
- output writeback in the current batched hidden-state layout

Notes:

- this is the stage where chunk-wise parallelism should become visible in performance

### Phase 5: Qwen3.5 Integration

Deliverables:

- wire the new path into `Qwen35Model` prefill
- preserve the current decode path
- gate the new path behind a clear runtime or build-time switch until it is stable

Exit condition:

- one end-to-end prefill path in pegainfer can exercise the chunk-wise operator

### Phase 6: Performance and Cleanup

Deliverables:

- microbench comparison against current fused-recurrent baseline
- end-to-end TTFT comparison at `(2048, 1)`
- `nsys` and `ncu` inspection
- remove or reduce redundant intermediate conversions and copies

Desired outcome:

- chunk-wise moves the bottleneck to larger matrix-style kernels instead of a low-occupancy recurrent loop

## Risks

### Numerical Stability

This is the main algorithmic risk. The chunk-wise path introduces more moving pieces than the current fused-recurrent baseline, especially around `A`, triangular solve, and `w / u` preparation.

### Runtime Complexity

The current runtime surface is simple: one call, one kernel symbol. Chunk-wise wants explicit scratch and multiple stages. That complexity should be admitted and designed, not hidden.

### Overfitting to Qwen3.5

The first pass should be Qwen3.5-specific on purpose. The mistake would be pretending the first version is generic when it is not.

### Fast Path Fragmentation

If we later add a vendored SM90 CUDA path and an in-tree Triton path, the fallback and capability matrix must stay understandable.

## Non-Goals For First Pass

- varlen
- backward / training
- general head sizes beyond Qwen3.5
- replacing decode GDR
- fully generic multi-model chunk-wise runtime
- perfect kernel tuning from day one

## Recommended Next Step

Do not jump straight into a full port of FlashInfer or a monolithic CUDA rewrite.

The immediate next step should be:

1. profile the real chunk-wise request path stage by stage to see whether prep or `chunk_state/chunk_o` now dominates
2. decide whether the prep stages should stay in-tree or be further aligned with FLA-style kernels
3. clean up the fallback / benchmarking surface now that the chunk-wise path is the accepted runtime baseline

At this point, the architecture question is closed; the remaining work is performance tuning and code-shape cleanup.
