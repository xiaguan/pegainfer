//! Model implementations: Qwen3 and Qwen3.5.

use anyhow::Result;
use rand::rngs::StdRng;

use crate::sampler::SamplingParams;
use crate::tensor::DeviceVec;

pub(crate) mod cuda_graph;
pub(crate) mod kv_cache;

pub mod dsv32;
pub mod qwen3;
pub mod qwen35;

pub use dsv32::{DsV32Executor, DsV32Model, ParallelConfig};
pub use qwen3::{ModelRuntimeConfig, Qwen3Model, Qwen3State, TensorParallelConfig};
pub use qwen35::Qwen35Model;

// ============================================================================
// ModelForward trait — used by the Qwen3 direct-request path
// ============================================================================

/// Per-request mutable state. Separate from model weights for bs > 1 future.
pub trait GenerationState {
    fn logits(&self) -> &DeviceVec;
    fn reset(&mut self) -> Result<()>;
}

/// Deep module interface: one `forward` method hides prefill/decode strategy,
/// layer types, CUDA Graph, buffer management, KV cache, and recurrent state.
pub trait ModelForward: Send {
    type State: GenerationState + Send;

    fn create_state(&self) -> Result<Self::State>;
    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()>;
    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32>;
    fn is_stop_token(&self, token_id: u32) -> bool;
}
