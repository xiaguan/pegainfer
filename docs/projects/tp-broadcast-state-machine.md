# Tensor Parallel Broadcast State Machine

> **Status:** Draft.
>
> This note records the architectural direction we have converged on while bringing up Qwen3 tensor parallelism. It is intentionally higher-level than the Qwen3-specific TP design doc.

## Core Idea

Tensor parallel execution should be modeled as:

- one controller that decides the next step
- multiple long-lived workers that each own their rank-local state
- a broadcast command stream that keeps those workers in sync

This is a better fit than treating tensor parallelism as shared mutable state borrowed across threads.

## What We Are Rejecting

We do **not** want the long-term design to look like:

- the main thread owns all rank-local KV state objects
- worker threads borrow `&mut` access into those objects
- cross-thread correctness relies on raw pointer wrappers plus timing assumptions

That shape was acceptable for bring-up, but it is not the right steady-state architecture.

## Architectural Model

### Controller

The controller is the only place that decides:

- which requests run in the next step
- which high-level state transitions should happen
- the ordered command stream for the workers

The controller is not the owner of rank-local execution objects.

In particular:

- the scheduler thread should not directly run GPU work
- rank 0 execution should also happen on a worker thread
- the controller should only coordinate commands and consume step results

### Workers

Each worker owns its rank-local state, including:

- its local model shard
- its local decode buffers
- its local KV pool
- its local per-request KV state

Workers should not depend on the controller holding rank-local mutable objects on their behalf.

### Broadcast Commands

The central synchronization mechanism is an ordered broadcast command stream.

The controller emits a command. All workers execute the same command in the same order. Correctness comes from deterministic replicated state transitions, not from sharing one mutable object graph.

At this level, the important abstraction is not "shared state", but:

- broadcasted commands
- replicated local state
- barriered step completion

## Why This Matters

This model matches what TP actually is:

- each rank mutates only its own local shard
- the logical request state must stay synchronized across ranks
- collective ops are just part of the step operator stream

This means TP is better understood as a replicated state machine than as a multithreaded shared-memory object model.

## KV Cache Implication

KV cache was the first place where the shared-state model started to break down, but the issue is broader than KV.

The key observation is:

- a logical request page is really a rank-synchronous page group
- each rank owns its local physical shard
- allocation and release must be consistent across ranks

So the long-term design should avoid central ownership of `Vec<KvState>` in the controller.

Instead:

- workers own rank-local KV state
- the controller drives its lifecycle by broadcasting deterministic commands

## Command Shape

We do not need a complicated distributed framework for the first version.

The first useful model is simply:

- one controller
- one channel per worker
- one broadcast helper that fans out the same command to all workers
- one barrier that waits for all worker responses before the next command

The transport can be `crossbeam-channel`.

The important part is the protocol, not the transport.

## Command Levels

There are two related command levels to think about.

### State Mutation Commands

These describe deterministic state transitions, for example:

- allocate request-local state
- ensure KV capacity
- advance KV length
- retire or drop request-local state

These commands keep replicated worker-local state machines aligned.

### Step Execution Commands

These describe actual model work, for example:

- prefill step
- decode step
- unified step

These commands run the operator stream. Their execution may include collective ops such as NCCL all-reduce.

In practice, step commands may subsume some state mutations. The important decision is not whether they are merged or split, but that they are broadcast commands over replicated local state.

## Current Decision

For the next design iteration, we choose **coarse-grained step commands** as the primary protocol shape.

That means the protocol should look more like:

- `RunPrefillStep`
- `RunDecodeStep`
- `RunUnifiedStep`
- `DropRequest`
- `Shutdown`

and less like:

- `EnsureCapacity`
- `Advance`
- `Reset`

as first-class protocol messages.

The reason is simple:

- the protocol should stay short and stable
- internal KV/state mutation details should remain an implementation detail of a step
- worker-local state transitions should be grouped into one step-shaped operation whenever possible

This keeps the command layer focused on controller intent, not on exposing every low-level mutation primitive.

## Design Preference Under This Decision

Under the coarse-grained command decision, the implementation should try to keep step execution as function-like as practical:

- input: worker-local request state plus step input
- output: updated worker-local request state plus step result

Even if the implementation mutates in place for performance, the architectural intent should still be one step = one state transition.

## Request Identity

The broadcast protocol should use an explicit request identity type:

- `RequestId(u64)`

This should be assigned by the controller from a monotonically increasing counter.

Why this choice:

- process-local uniqueness is enough for the current single-node TP design
- it is cheaper and simpler than UUIDs
- it is more stable and more explicit than using scheduler slots as the protocol identity

Internal worker-side slots or maps may still exist, but protocol messages should identify requests by `RequestId`, not by positional indices.

## Synchronization Rule

The system should obey a simple rule:

- no worker starts command `N+1` until all workers have finished command `N`

This gives us:

- deterministic ordering
- simpler failure handling
- no need for the controller to reason about cross-thread mutable borrows

## Near-Term Design Guidance

For the next round of cleanup, we should bias toward:

- worker-owned rank-local state
- broadcast command protocol
- explicit barrier semantics
- fewer raw pointer based cross-thread borrows

We should bias away from:

- central controller ownership of rank-local execution objects
- command payloads that smuggle `&mut` semantics through raw pointers
- designs that rely on multiple parallel vectors and index alignment to identify one logical request

## Open Questions

These are still unresolved and should be decided incrementally:

- how failure rollback should work if one worker cannot apply a broadcast command

## Non-Goals For This Note

This note does not specify:

- the final Rust type layout
- the final command enum
- the final scheduler API
- CUDA Graph integration

It only records the architectural decision that TP should move toward broadcast commands over replicated worker-local state.

## Current Payload Decisions

For the next refactor step, the command payload should be request-oriented rather than vector-oriented.

That means:

- command payloads should carry `Vec<RequestStepItem>`
- each element represents one logical request participating in the step
- protocol semantics should not depend on aligning multiple parallel vectors by index

For the current coarse-grained step commands, the intended shapes are:

- `RunPrefillStep { requests: Vec<PrefillStepItem>, echo: bool }`
- `RunDecodeStep { requests: Vec<DecodeStepItem> }`
- `RunUnifiedStep { prefill: Vec<PrefillStepItem>, decode: Vec<DecodeStepItem> }`

At minimum:

- `PrefillStepItem` should contain `request_id` and prompt tokens
- `DecodeStepItem` should contain `request_id` and the decode token

Additional fields can be added later if needed, but request identity should be part of the item itself.

## Current Execution Shape

The current implementation should keep only one executor shape for Qwen3:

- one `Qwen3Executor`
- one primary local lane
- zero or more additional rank workers

Under this model:

- single-GPU execution is the `tp_size == 1` case
- tensor parallel execution is the `tp_size > 1` case

The controller-side protocol should not branch into separate single-GPU and TP command shapes.

Instead:

- both modes use the same coarse-grained `StepCommand`
- rank 0 executes the command on the primary worker thread
- additional workers, when present, receive the same command through broadcast fanout
- the scheduler thread does not directly execute the step

## Worker-Local Request State

Workers should keep their own request-local execution state keyed by `RequestId`.

For the first design pass, the minimal worker-local state should be:

- rank-local KV state

The controller should continue to own user-facing lifecycle state such as:

- token stream sender
- sampling params
- generation counts
- finish bookkeeping

This preserves the existing scheduler role while moving rank-local execution state to the workers.

## Result And Acknowledgement Shape

For now, the protocol should stay asymmetric:

- non-primary workers return only acknowledgement or step failure
- the primary worker returns the step artifacts needed by the controller

This keeps result flow simple:

- workers stay responsible for local execution
- the controller still resolves artifacts into user-visible effects

## Sampling Ownership

Sampling is now split into two responsibilities:

- the controller owns sampling policy and random input generation
- the primary worker executes sampling and logprob extraction

Concretely:

- `SamplingParams` and per-step random values are carried in the step items
- worker threads run the GPU sampling path
- the controller only consumes CPU step artifacts

This preserves deterministic controller-owned randomness without pulling GPU work back into the scheduler thread.

## Request Destruction

For now, request destruction should be an explicit protocol action:

- `DropRequest { request_id }`

This is preferred over implicit destruction hidden inside unrelated step commands.

It keeps request lifecycle transitions visible at the command layer and matches the broader broadcast state-machine model.
