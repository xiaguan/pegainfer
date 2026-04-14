//! DeepSeek-V3.2 model: MLA + MoE transformer (671B).

pub(crate) mod config;
mod weights;

pub use weights::DsV3Model;
