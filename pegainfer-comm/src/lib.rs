//! PegaInfer comm-backend public surface (skeleton).
//!
//! This crate sketches the public abstraction PegaInfer will use for
//! cross-rank data movement (EP all-to-all first; future data-movement
//! surfaces later). It is a **skeleton**: the type surface is reviewable, but
//! `EpBackendBuilder::build` is fail-closed in both feature modes — no
//! caller can obtain a usable backend in this PR. The wiring lands in a
//! follow-up PR; see the `README.md` for the staging plan.
//!
//! The default-feature surface is intentionally hardware-free so the main
//! PegaInfer CI lane can `cargo check -p pegainfer-comm` on a barebones
//! development machine without CUDA, GDRCopy, or RDMA Verbs headers.
//!
//! # Feature flags
//!
//! - `default = []` — pure Rust surface: trait, plan, handle, buffer, error,
//!   builder. NO link to wrapper-crate types, NO probe of CUDA / Verbs.
//!   `EpBackendBuilder::build` returns [`Error::BackendUnavailable`].
//! - `hw-rdma` — compiles the `crate::backend::rdma` module, which
//!   depends on the `p2p-all-to-all` wrapper crate (which itself
//!   transitively activates the CUDA subsystem). Building with `hw-rdma`
//!   requires the matching CUDA, RDMA Verbs, and GDRCopy development /
//!   runtime components on the host. In this skeleton PR,
//!   `EpBackendBuilder::build` still returns [`Error::Unimplemented`]
//!   under `hw-rdma`; the feature exists so the wrapper-crate build chain
//!   is exercised, not so a usable backend is produced. The fail-closed
//!   branch is replaced with real construction in the wiring PR.
//!
//! # Public-surface invariants for this skeleton
//!
//! 1. The `EpAllToAll` trait, all plan / handle / buffer / error / builder
//!    types in this crate's default-feature surface MUST NOT reference any
//!    type from a wrapper crate (`p2p-all-to-all`, `fabric-lib`, `cuda-lib`,
//!    `torch-lib`, `a2a-kernels`, `cuda-sys`, `cudart-sys`, `gdrapi-sys`,
//!    `libibverbs-sys`). Backend errors are erased through a `source` field
//!    of type `Box<dyn std::error::Error + Send + Sync>`.
//! 2. Backend modules live under `crate::backend::*` and are only compiled
//!    when the matching feature is on. They MUST NOT re-export
//!    wrapper-crate types through `pegainfer-comm`'s public namespace.
//! 3. Diagnostic markers (`HW_RDMA_ENABLED`) are diagnostic-only and not
//!    part of the stable API; they exist for build-system / runtime
//!    introspection.
//! 4. Until the wiring PR lands, `EpBackendBuilder::build` is fail-closed
//!    in all feature modes. No caller can obtain an `EpBackend` whose
//!    trait methods would panic.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

mod buffer;
mod builder;
mod error;
mod handle;
mod plan;
mod r#trait;

pub use buffer::{RecvBuf, SendBuf};
pub use builder::{EpBackend, EpBackendBuilder, EpTopology};
pub use error::{Error, Result};
pub use handle::{AnyHandle, CombineHandle, DispatchHandle, Poll};
pub use plan::{CombinePlan, DispatchPlan};
pub use r#trait::EpAllToAll;

/// Diagnostic marker. `true` when the `hw-rdma` feature is active.
///
/// This is a build-system / runtime introspection signal, not part of the
/// stable public API. Code that needs to react to backend availability
/// should call `EpBackendBuilder::build` and dispatch on the returned
/// `Result`, not on this constant.
pub const HW_RDMA_ENABLED: bool = cfg!(feature = "hw-rdma");

#[cfg(feature = "hw-rdma")]
mod backend;
