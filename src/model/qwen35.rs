//! Qwen3.5 model: mixed full attention + linear attention transformer.

pub(crate) mod config;
mod decode;
mod decode_buffers;
mod forward;
mod prefill;
pub(crate) mod prefill_buffers;
mod recurrent_state;
mod weights;

pub use forward::Qwen35State;
pub use weights::Qwen35Model;
