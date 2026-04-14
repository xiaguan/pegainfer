//! DeepSeek-V3.2 model: MLA + MoE transformer (671B).

pub(crate) mod config;
pub(crate) mod mla_kv;
mod weights;

pub use weights::DsV3Model;
