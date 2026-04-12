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
pub use elementwise::{add_batch, silu_mul_batch};
pub use embedding::{embedding_batch, embedding_decode_into};
pub use linear::{gemm, gemv};
pub use norm::{
    fused_add_rms_norm_into, rms_norm_batch_offset_into, rms_norm_into, rms_norm_offset_into,
};
pub use recurrent::gated_delta_rule_prefill_chunkwise_into;
pub use sampling::{argmax, flashinfer_topk_row_states_bytes, gpu_sample, gpu_sample_into};

// pub(crate) re-exports
pub(crate) use attention::{
    PrefillPagedPlan, paged_attention_batch_decode_hd256_into, paged_attention_batch_decode_into,
    prefill_attention_paged_into, qk_norm_partial_rope_batched_decode_hd256_into,
    qk_norm_rope_batch_decode_into,
};
pub(crate) use elementwise::{
    add_batch_into, extract_vec, extract_vec_into, silu_mul_batch_into, silu_mul_fused_batch_into,
    write_vec_into,
};
pub(crate) use linear::{gemm_into, gemm_rows_into, linear};
pub(crate) use norm::{
    fused_add_rms_norm_batch_into, rms_norm, rms_norm_batch_into, rms_norm_gated_batch_into,
};
pub(crate) use recurrent::{
    conv1d_decode_into, conv1d_prefill_batch_into, gated_delta_rule_decode_vec_into,
};
