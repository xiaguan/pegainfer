//! RDMA + CUDA EP all-to-all backend.
//!
//! Thin adapter over `p2p_all_to_all::AllToAllContext`. The wrapper-crate
//! type is held privately and never appears in the public surface; all
//! interaction goes through the [`crate::EpAllToAll`] trait.
//!
//! Skeleton: handle bookkeeping, scale-buffer plumbing, and the actual
//! dispatch/combine wiring will be filled in a follow-up PR. In this
//! skeleton PR the type is **not constructible** from
//! [`crate::EpBackendBuilder::build`] — `build` fails closed with
//! [`crate::Error::Unimplemented`], so callers cannot reach the
//! `todo!()` trait bodies below. The struct + impl exist here so the
//! shape of the planned adapter is reviewable; they are
//! `#[allow(dead_code)]`-gated until the wiring lands.

#![allow(dead_code)]

use crate::buffer::{RecvBuf, SendBuf};
use crate::builder::EpTopology;
use crate::error::Result;
use crate::handle::{AnyHandle, CombineHandle, DispatchHandle, Poll};
use crate::plan::{CombinePlan, DispatchPlan};
use crate::r#trait::EpAllToAll;

/// Concrete RDMA-backed implementation of [`EpAllToAll`].
///
/// Holds the upstream `AllToAllContext` plus whatever per-call state the
/// backend needs to translate between the public `Plan` / `Buf` /
/// `Handle` types and the wrapper crate's own API.
pub(crate) struct RdmaBackend {
    /// Configured topology. Held so we can validate per-call plans
    /// against the original construction parameters.
    _topology: EpTopology,
    // Skeleton: the real backend will hold
    //   _ctx: p2p_all_to_all::AllToAllContext,
    //   _handles: HandleTable,
    // here. Wiring lands in §8.5 along with the integration test
    // against the live wrapper crate.
}

impl RdmaBackend {
    /// Construct a backend for the given topology.
    pub(crate) fn new(topology: EpTopology) -> Result<Self> {
        Ok(Self { _topology: topology })
    }
}

impl EpAllToAll for RdmaBackend {
    fn dispatch(
        &self,
        _plan: &DispatchPlan,
        _send_buf: &SendBuf<'_>,
        _recv_buf: &mut RecvBuf<'_>,
    ) -> Result<DispatchHandle> {
        todo!(
            "§8.5: translate DispatchPlan + buffers into AllToAllContext::dispatch_send"
        )
    }

    fn combine(
        &self,
        _plan: &CombinePlan,
        _send_buf: &SendBuf<'_>,
        _recv_buf: &mut RecvBuf<'_>,
    ) -> Result<CombineHandle> {
        todo!(
            "§8.5: translate CombinePlan + buffers into AllToAllContext::combine_send"
        )
    }

    fn poll(&self, _handle: &AnyHandle) -> Result<Poll> {
        todo!("§8.5: poll backend completion queue")
    }

    fn release(&self, _handle: AnyHandle) -> Result<()> {
        todo!("§8.5: free internal handle-table slot")
    }
}
