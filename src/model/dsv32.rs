//! DeepSeek-V3.2 model: MLA + MoE transformer (671B).

pub(crate) mod config;
pub(crate) mod deep_ep;
pub mod executor;
pub(crate) mod forward;
pub(crate) mod mla_kv;
pub(crate) mod weights;

pub use config::ParallelConfig;
pub use executor::DsV32Executor;
pub use weights::DsV32Model;
pub(crate) use weights::Fp8Matrix;
