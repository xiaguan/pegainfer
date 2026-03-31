//! Qwen3.5 model: mixed full attention + linear attention transformer.

mod batch_decode;
pub(crate) mod batch_decode_graph;
pub(crate) mod config;
mod decode_buffers;
mod forward;
mod prefill;
pub mod prefill_buffers;
mod recurrent_state;
mod single_token_buffers;
mod weights;

pub use forward::Qwen35State;
pub use weights::Qwen35Model;
