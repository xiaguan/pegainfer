//! PegaInfer comm-backend public surface.
//!
//! Default-off this crate exposes only the [`HW_RDMA_ENABLED`] diagnostic
//! marker and the public [`Error`] type — no wrapper-crate dependency, no
//! CUDA / RDMA Verbs / GDRCopy probing.
//!
//! With the `hw-rdma` feature enabled, [`EpBackend`] wraps the upstream
//! `pplx-garden` NVLink + RDMA all-to-all context with a thin Rust surface
//! tailored for PegaInfer's MoE call sites: `dispatch_send / dispatch_recv
//! / combine_send / combine_recv`, kept separate so callers can overlap
//! host-side compute between send and recv.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

mod error;
pub use error::{Error, Result};

/// `true` when the `hw-rdma` feature is active in this build. Diagnostic
/// only; do not use as a stability marker.
pub const HW_RDMA_ENABLED: bool = cfg!(feature = "hw-rdma");

#[cfg(feature = "hw-rdma")]
mod ep_backend;
#[cfg(feature = "hw-rdma")]
pub use ep_backend::{
    EpBackend, EpBackendParams, EpDtypes, EpRankBuffers, EpTopology, ScalarType,
};

/// Re-exports of the underlying `pplx-garden` building blocks. Available
/// under the `hw-rdma` feature so PegaInfer-side bootstrap code can build
/// `EpBackendParams` without taking direct dependencies on the vendored
/// crates.
#[cfg(feature = "hw-rdma")]
pub mod raw {
    pub use cuda_lib;
    pub use fabric_lib;
    pub use p2p_all_to_all;
}
