# Qwen3 Tensor Parallelism Design

> **TL;DR:** Add `TP=2` support for `Qwen3-4B` as the first model-parallel milestone. The goal is correctness and a clean architectural foundation for larger dense models and future MoE work, not a fully generalized distributed runtime on day one.
>
> **Status:** Active. Initial scope definition for the first tensor-parallel milestone.

## Goal

Add tensor parallelism for `Qwen3-4B` with a narrow and explicit first target:

- support `TP=2`
- preserve `TP=1`
- keep the milestone focused on model-parallel correctness
- establish the right abstractions for later large-model and MoE work

This milestone is about making pegainfer capable of serving a single model replica across two GPUs. It is not about multi-replica throughput scaling.

## Why This Matters

Large-model serving and MoE serving both require model-parallel building blocks. For pegainfer, tensor parallelism is the first such building block.

The immediate value is:

- larger dense models become reachable without forcing a single-GPU fit
- the runtime starts carrying rank-local weights and rank-local execution state
- later MoE work can build on model-parallel foundations instead of retrofitting them into a single-GPU design

## Scope

This first pass is intentionally narrow:

- model: `Qwen3-4B`
- parallel degree: `TP=2`
- focus: correctness and architecture
- deployment target: a single machine

The first milestone does not need to solve every parallelism problem. It needs to prove that pegainfer can run one dense model replica across two GPUs without breaking correctness or making the architecture harder to evolve.

## Design Constraints

- Tensor parallelism must be model-parallel, not data-parallel in disguise.
- The design must serve future larger dense models and MoE, not only Qwen3-4B.
- `TP=1` must remain a supported and healthy path.
- The first version may simplify some weight handling, but those simplifications must not block later movement toward a more fully sharded design.
- The external user experience should stay simple. Tensor parallel support should not require the user to reason about a heavyweight distributed system.
- The design should avoid coupling the abstraction too tightly to one specific kernel path or one specific Qwen3 implementation detail.

## Explicit Non-Goals

The first TP milestone does not aim to do the following:

- support `Qwen3.5`
- support MoE expert parallelism
- support data parallelism
- support pipeline parallelism
- preserve current CUDA Graph behavior from day one
- introduce vocab-parallel embedding or vocab-parallel `lm_head` in the first pass
- optimize every path for peak throughput before the basic design is proven correct

## Simplifications Allowed In First Pass

The first pass is allowed to trade some efficiency for speed of validation, as long as the trade does not poison the long-term design.

Allowed simplifications:

- tied embedding / `lm_head` may be replicated instead of sharded
- some paths may be brought up incrementally as long as the final milestone still has a clear acceptance target
- correctness and maintainable abstraction take priority over immediate performance parity

These are first-pass scope controls, not permanent architectural commitments.

## What Success Looks Like

The milestone is successful when all of the following are true:

- `Qwen3-4B` runs correctly with `TP=2`
- existing single-GPU behavior remains intact with `TP=1`
- the runtime can be reasonably extended later toward larger dense models and MoE-related model parallelism
- the resulting design is understandable and does not create a one-off parallel path that the rest of the codebase must work around

## Acceptance Criteria

Primary acceptance criteria:

- `Qwen3-4B` under `TP=2` passes the existing end-to-end test suite, or passes an equivalent minimally adjusted `e2e` path if the test harness needs TP-aware setup
- generated outputs under `TP=2` match the `TP=1` baseline for the covered `e2e` cases
- `TP=1` continues to pass its existing `e2e` coverage
- runtime stability is acceptable: no hangs, no cross-device state corruption, no obvious lifecycle failures during model load or generation

The core bar for this milestone is straightforward:

- `TP=2` for `Qwen3-4B` must be real, correct, and regression-safe

## Out Of Scope Questions

The following questions are intentionally deferred until after the first TP milestone is proven:

- how far to push throughput optimization in the first TP implementation
- when to restore or redesign CUDA Graph support for TP paths
- when to shard vocab-facing weights instead of replicating them
- whether later multi-GPU support should expand first toward larger dense models, MoE, or broader serving topology work

## Summary

This milestone should stay disciplined.

The job is not to build a full distributed inference platform in one step. The job is to make pegainfer capable of correct `TP=2` execution for `Qwen3-4B`, while establishing the architectural boundary that future large dense and MoE work can build on.
