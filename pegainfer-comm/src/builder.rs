//! Backend construction.
//!
//! The builder is the only entry point that turns a configured topology
//! into a usable [`EpBackend`]. The default-feature build of this crate
//! is hardware-free: [`EpBackendBuilder::build`] always reports
//! [`crate::Error::BackendUnavailable`] until a hardware backend feature
//! is enabled.
//!
//! Hardware backends, when enabled, live in `crate::backend::*` and are
//! NEVER re-exported through `pegainfer-comm`'s public namespace — the
//! only way to obtain one is through this builder.

use crate::buffer::{RecvBuf, SendBuf};
use crate::error::{Error, Result};
use crate::handle::{AnyHandle, CombineHandle, DispatchHandle, Poll};
use crate::plan::{CombinePlan, DispatchPlan};
use crate::r#trait::EpAllToAll;

/// EP topology description. Captures the parts of the rank layout the
/// backend needs to size internal buffers and resolve peers.
///
/// Skeleton: only the fields PegaInfer currently passes are present.
/// Marked `#[non_exhaustive]` so additions are non-breaking.
///
/// Construct with [`EpTopology::new`]; the struct is `#[non_exhaustive]`
/// so callers outside this crate cannot use a struct literal. Fields
/// stay `pub` for read-side ergonomics on the builder.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct EpTopology {
    /// Total number of ranks participating in the all-to-all.
    pub world_size: u32,
    /// Rank of the current process (0-based, < `world_size`).
    pub rank: u32,
    /// Total number of experts across all ranks.
    pub num_experts: u32,
    /// Hidden dimension of the token tensors moved by this backend.
    pub hidden_dim: u32,
    /// Maximum number of tokens any one dispatch on this backend can
    /// carry. Used to size internal staging buffers at construction
    /// time so per-call dispatch never allocates.
    pub max_num_tokens: u32,
}

impl EpTopology {
    /// Construct a topology description from the rank-layout values the
    /// backend needs at build time.
    ///
    /// The skeleton-PR contract is: this constructor is the only stable
    /// way to obtain an `EpTopology` from outside this crate. Adding a
    /// new field in a follow-up PR will add a new constructor variant
    /// (or a builder) rather than break this signature, but any caller
    /// that wants forward-compatibility should treat this signature as
    /// subject to revision while the public surface is in skeleton form.
    pub fn new(
        world_size: u32,
        rank: u32,
        num_experts: u32,
        hidden_dim: u32,
        max_num_tokens: u32,
    ) -> Self {
        Self { world_size, rank, num_experts, hidden_dim, max_num_tokens }
    }
}

/// Builder for [`EpBackend`].
///
/// Construct with [`EpBackendBuilder::new`], configure with chained
/// setters, then call [`EpBackendBuilder::build`].
#[derive(Debug, Default)]
pub struct EpBackendBuilder {
    topology: Option<EpTopology>,
}

impl EpBackendBuilder {
    /// Start a new builder with no topology configured.
    pub fn new() -> Self {
        Self { topology: None }
    }

    /// Set the EP topology. Required before [`Self::build`].
    pub fn topology(mut self, topology: EpTopology) -> Self {
        self.topology = Some(topology);
        self
    }

    /// Finalize the configuration and construct the backend.
    ///
    /// # Skeleton-PR behavior
    ///
    /// While the public surface is in skeleton form, `build` is
    /// **fail-closed in both feature modes**:
    ///
    /// - default-off: returns [`Error::BackendUnavailable`] — no
    ///   hardware backend feature is active.
    /// - `hw-rdma`: returns [`Error::Unimplemented`] — the
    ///   `RdmaBackend` adapter exists as a type but its
    ///   dispatch / combine / poll / release wiring is not yet
    ///   implemented (lands in a follow-up PR).
    ///
    /// This is intentional: it guarantees that callers in either build
    /// mode cannot obtain an `EpBackend` whose trait methods would
    /// panic. The fail-closed branch will be replaced with real
    /// construction logic in the wiring PR.
    ///
    /// # Errors
    ///
    /// - [`Error::BackendUnavailable`] when no hardware backend feature
    ///   is active.
    /// - [`Error::Unimplemented`] while `hw-rdma` is on but its wiring
    ///   has not yet landed.
    /// - [`Error::InvalidPlan`] when the configured topology is missing
    ///   or inconsistent with the active backend.
    /// - [`Error::Backend`] when (after wiring) the underlying backend's
    ///   own construction fails (e.g. RDMA device enumeration, CUDA
    ///   context creation).
    pub fn build(self) -> Result<EpBackend> {
        #[cfg(not(feature = "hw-rdma"))]
        {
            let _ = self.topology;
            Err(Error::BackendUnavailable {
                reason: "no hardware backend feature active",
                required_feature: "hw-rdma",
            })
        }

        #[cfg(feature = "hw-rdma")]
        {
            let _topology =
                self.topology.ok_or(Error::InvalidPlan("topology not configured"))?;
            Err(Error::Unimplemented {
                what: "RdmaBackend dispatch/combine/poll/release wiring (skeleton PR; landed separately)",
            })
        }
    }
}

/// Concrete backend handle returned by [`EpBackendBuilder::build`].
///
/// Opaque wrapper around the active hardware backend; implements
/// [`EpAllToAll`] by delegation. The inner backend type is intentionally
/// not exposed so backends can be swapped without breaking the public
/// surface.
pub struct EpBackend {
    inner: Box<dyn EpAllToAll>,
}

impl std::fmt::Debug for EpBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EpBackend").finish_non_exhaustive()
    }
}

impl EpAllToAll for EpBackend {
    fn dispatch(
        &self,
        plan: &DispatchPlan,
        send_buf: &SendBuf<'_>,
        recv_buf: &mut RecvBuf<'_>,
    ) -> Result<DispatchHandle> {
        self.inner.dispatch(plan, send_buf, recv_buf)
    }

    fn combine(
        &self,
        plan: &CombinePlan,
        send_buf: &SendBuf<'_>,
        recv_buf: &mut RecvBuf<'_>,
    ) -> Result<CombineHandle> {
        self.inner.combine(plan, send_buf, recv_buf)
    }

    fn poll(&self, handle: &AnyHandle) -> Result<Poll> {
        self.inner.poll(handle)
    }

    fn release(&self, handle: AnyHandle) -> Result<()> {
        self.inner.release(handle)
    }
}
