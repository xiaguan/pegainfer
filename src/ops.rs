//! GPU operations on device tensors.

mod attention;
mod elementwise;
mod embedding;
mod linear;
mod norm;
mod recurrent;
mod sampling;

#[cfg(test)]
mod tests;

// pub re-exports (used by benches and bin targets)
pub use attention::{prefill_attention_hd256_batch, prefill_attention_hd256_batch_with_scratch};
pub use elementwise::{add_batch, silu_mul_batch};
pub use embedding::{embedding_batch, embedding_decode_into};
pub use linear::{fused_mlp_into, gemm, gemv};
pub use norm::{
    fused_add_rms_norm_into, rms_norm_batch_offset_into, rms_norm_into, rms_norm_offset_into,
};
pub use recurrent::gated_delta_rule_prefill_chunkwise_into;
pub use sampling::{argmax, argmax_batched, gpu_sample, gpu_sample_into};

// pub(crate) re-exports
#[cfg(test)]
pub(crate) use attention::flash_attention_prefill_hd256_into;
pub(crate) use attention::{
    PrefillPagedPlan, paged_attention_batch_decode_into, paged_attention_decode_into,
    prefill_attention_paged_into, qk_norm_rope_batch_decode_into, qk_norm_rope_into,
};
pub(crate) use elementwise::{
    add_batch_into, extract_vec, silu_mul_batch_into, silu_mul_fused_batch_into,
};
pub(crate) use linear::deinterleave_qkv_into;
pub(crate) use linear::{fused_mlp_gate_up_into, gemm_into, gemm_rows_into, gemv_rows, linear};
pub(crate) use norm::{
    fused_add_rms_norm_batch_into, rms_norm, rms_norm_batch_into, rms_norm_gated_batch_into,
};
pub(crate) use recurrent::{conv1d_prefill_batch_into, gated_delta_rule_decode_into};
