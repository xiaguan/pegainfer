//! Dispatch and combine plans.
//!
//! A "plan" carries the per-call routing information that varies between
//! invocations of the same backend: the routing indices, the per-rank /
//! per-expert token counts, and the dtypes of the payload. The backend
//! configuration (world size, device list, EP topology) lives on the
//! backend object itself and is established at construction time.

/// Dispatch plan: routing decisions for a single forward call.
///
/// Skeleton. Field shape is intentionally minimal; concrete fields will be
/// filled when wiring into PegaInfer's request scheduler. Adding fields is
/// not a breaking change as long as we keep this `#[non_exhaustive]`.
///
/// Construct with [`DispatchPlan::new`]; the struct is `#[non_exhaustive]`
/// so callers outside this crate cannot use a struct literal.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    /// Number of tokens fed into this dispatch (`bound_m` upstream).
    pub num_tokens: u32,
    /// Number of experts each token is routed to.
    pub num_experts_per_token: u32,
}

impl DispatchPlan {
    /// Construct a dispatch plan with the per-call routing values.
    ///
    /// Constructor signature is subject to revision while the public
    /// surface is in skeleton form; future fields will be added through
    /// a new constructor variant or a builder rather than by breaking
    /// this signature in place.
    pub fn new(num_tokens: u32, num_experts_per_token: u32) -> Self {
        Self { num_tokens, num_experts_per_token }
    }
}

/// Combine plan: paired with a prior dispatch.
///
/// Skeleton. See [`DispatchPlan`].
///
/// Construct with [`CombinePlan::new`]; the struct is `#[non_exhaustive]`
/// so callers outside this crate cannot use a struct literal.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CombinePlan {
    /// Number of tokens that participated in the paired dispatch.
    pub num_tokens: u32,
    /// Whether the combine should accumulate into the output buffer.
    pub accumulate: bool,
}

impl CombinePlan {
    /// Construct a combine plan paired with a prior dispatch.
    ///
    /// Constructor signature is subject to revision while the public
    /// surface is in skeleton form; see [`DispatchPlan::new`].
    pub fn new(num_tokens: u32, accumulate: bool) -> Self {
        Self { num_tokens, accumulate }
    }
}
