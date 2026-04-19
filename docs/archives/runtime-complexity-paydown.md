# Project: Runtime Complexity Paydown

**Status**: Archived
**TL;DR**: Archived runtime-paydown record. It captures the phase where runtime branching cleanup, API-surface tightening, and module-boundary cleanup were treated as a dedicated delivery-risk reduction effort.

---

## Goal

Bring runtime complexity back under control before the next wave of model-family support.

The point is not abstraction for its own sake. The point is to make future model integration more predictable, keep the serving path coherent, and avoid growing a separate runtime for every new attention pattern.

## Focus

- reduce runtime branching across model families
- stabilize the core runtime contracts that serving depends on
- lower the marginal cost of integrating the next non-standard attention model

## Success

- new model-family work starts from a shared runtime foundation, not a fresh execution stack
- runtime behavior is easier to reason about across prefill, decode, state, and serving
- framework debt stops compounding faster than model coverage

## Current Focus

- architecture: figure out the right abstraction boundary for model-specific execution details (prefill/decode orchestration, state management, attention patterns)
- start pulling orchestration code out of long model files without changing prefill/decode behavior

## Progress

- first pass completed: internal runtime modules such as decode buffers, KV cache, recurrent state, and FFI bindings were pulled back behind crate-local visibility
- first pass completed: unused public helpers in `model.rs` and `qwen35_model.rs` were removed or restricted to test-only code
- first pass completed: bench-only scaffolding stopped using outward-facing visibility, which makes `unreachable-pub` useful again for the real crate surface
- second pass completed: `ops.rs` stopped exposing or carrying legacy primitive wrappers that real models no longer call
- second pass completed: ops microbenches were pruned so they track kernels on actual model paths instead of historical one-off primitives
- third pass completed: unused logging branches and `DeviceVecView` were deleted instead of preserved as configurable surface
- third pass completed: tokenizer streaming internals, weight-loader helpers, Qwen3/Qwen3.5 weight structs, and most Qwen3.5 config plumbing were pulled back to crate-local visibility

## Next Action

Design the abstraction boundary between shared runtime infrastructure and model-specific execution logic, then incrementally extract from `model.rs` (1103 lines) and `qwen35_model.rs` (1323 lines).
