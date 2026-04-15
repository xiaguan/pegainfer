//! DeepSeek-V3.2 model: MLA + MoE transformer (671B).

pub(crate) mod config;
pub mod executor;
pub(crate) mod forward;
pub(crate) mod mla_kv;
pub(crate) mod weights;

pub use config::ParallelConfig;
pub use executor::DsV3Executor;
pub use weights::DsV3Model;
pub(crate) use weights::{AbsorbedMlaWeights, Fp8Matrix};
