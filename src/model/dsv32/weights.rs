use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use log::{debug, info};
use std::collections::HashMap;
use std::time::Instant;

use super::config::{DsV32Config, FfnType, ParallelConfig};
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec};
use crate::weight_loader::{
    deserialize_shards, load_shard_info, load_tensor_2d, mmap_shards, precompute_yarn_rope,
};

// ---------------------------------------------------------------------------
// FP8 weight: raw e4m3 bytes + block-wise scale_inv
// ---------------------------------------------------------------------------

/// FP8 weight matrix with block-wise dequantization scales.
///
/// Stored as raw `u8` on GPU (1 byte per element, fp8_e4m3fn) plus a 2D f32
/// scale_inv tensor. During forward, a CUDA kernel dequantizes on-the-fly:
///   bf16_val = fp8_val * scale_inv[row_block][col_block]
/// where block indices are (row / block_size[0], col / block_size[1]).
pub(crate) struct Fp8Matrix {
    /// Raw FP8 e4m3 data on GPU: [rows * cols] bytes
    pub(crate) data: CudaSlice<u8>,
    /// Block-wise inverse scales: [ceil(rows/block_h), ceil(cols/block_w)] f32
    pub(crate) scale_inv: CudaSlice<f32>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) scale_rows: usize,
    pub(crate) scale_cols: usize,
}

// ---------------------------------------------------------------------------
// MLA attention weights
// ---------------------------------------------------------------------------

/// Multi-head Latent Attention (MLA) weights for one layer.
///
/// KV path:  hidden → kv_a_proj [576, 7168] → (split c_KV 512d, k_rope 64d)
///                                            → kv_a_layernorm(c_KV)
///                                            → kv_b_proj [32768, 512] → (k_nope, v)
///
/// Q path:   hidden → q_a_proj [1536, 7168]  → q_a_layernorm
///                  → q_b_proj [24576, 1536]  → (q_nope, q_rope)
pub(super) struct MlaWeights {
    // Q low-rank path
    pub(super) q_a_proj: Fp8Matrix,
    pub(super) q_a_layernorm: DeviceVec,
    pub(super) q_b_proj: Fp8Matrix,

    // KV compressed path
    pub(super) kv_a_proj_with_mqa: Fp8Matrix,
    pub(super) kv_a_layernorm: DeviceVec,
    pub(super) kv_b_proj: Fp8Matrix,

    // Output
    pub(super) o_proj: Fp8Matrix,
}

// ---------------------------------------------------------------------------
// NSA Indexer weights (per layer)
// ---------------------------------------------------------------------------

/// Native Sparse Attention (NSA) indexer weights for one layer.
///
/// The indexer computes per-query relevance scores over KV positions:
///   index_q = wq_b(q_compressed)              → [T, n_heads, head_dim]
///   index_k = k_norm(wk(hidden))              → [T, head_dim]
///   weights = weights_proj(hidden)            → [T, n_heads]
///   score   = sum_h relu(q_h · k) * weights_h → [T, T]
///   indices = causal_topk(score, topk)         → [T, topk]
///
/// Layout: rope(64) + nope(64), NOT the main MLA's nope+rope ordering.
pub(super) struct IndexerWeights {
    /// Q projection from q_compressed: FP8 [n_heads * head_dim, q_lora_rank]
    pub(super) wq_b: Fp8Matrix,
    /// K projection from hidden: FP8 [head_dim, hidden_size]
    pub(super) wk: Fp8Matrix,
    /// LayerNorm weight for k (with bias, NOT RMSNorm): bf16 [head_dim]
    pub(super) k_norm_weight: DeviceVec,
    /// LayerNorm bias for k: bf16 [head_dim]
    pub(super) k_norm_bias: DeviceVec,
    /// Head-mixing weight projection: bf16 [n_heads, hidden_size]
    pub(super) weights_proj: DeviceMatrix,
}

// ---------------------------------------------------------------------------
// Dense FFN weights (layers 0..first_k_dense_replace)
// ---------------------------------------------------------------------------

pub(super) struct DenseFFN {
    pub(super) gate_proj: Fp8Matrix,
    pub(super) up_proj: Fp8Matrix,
    pub(super) down_proj: Fp8Matrix,
}

// ---------------------------------------------------------------------------
// MoE FFN weights (layers first_k_dense_replace..num_hidden_layers)
// ---------------------------------------------------------------------------

/// Single expert FFN weights.
pub(super) struct ExpertFFN {
    pub(super) gate_proj: Fp8Matrix,
    pub(super) up_proj: Fp8Matrix,
    pub(super) down_proj: Fp8Matrix,
}

/// MoE gate (router) weights.
pub(super) struct MoeGate {
    /// Router weight: [n_routed_experts, hidden_size] bf16
    pub(super) weight: DeviceMatrix,
    /// Score correction bias: [n_routed_experts] f32
    pub(super) e_score_correction_bias: CudaSlice<f32>,
}

/// Full MoE layer: gate + routed experts + shared expert.
pub(super) struct MoeFfnWeights {
    pub(super) gate: MoeGate,
    pub(super) experts: Vec<ExpertFFN>,
    pub(super) shared_expert: ExpertFFN,
}

// ---------------------------------------------------------------------------
// Transformer block
// ---------------------------------------------------------------------------

pub(super) enum FfnWeights {
    Dense(DenseFFN),
    MoE(MoeFfnWeights),
}

pub(super) struct TransformerBlock {
    pub(super) input_layernorm: DeviceVec,
    pub(super) mla: MlaWeights,
    /// Pre-computed absorbed MLA weights (W_UK, W_UV from kv_b_proj dequant).
    pub(super) absorbed: AbsorbedMlaWeights,
    /// NSA indexer weights (None if model has no indexer config).
    pub(super) indexer: Option<IndexerWeights>,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) ffn: FfnWeights,
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// DeepSeek-V3.2 model weights.
pub struct DsV32Model {
    pub(super) ctx: DeviceContext,
    pub(super) config: DsV32Config,
    pub(super) parallel: ParallelConfig,
    pub(super) embed_tokens: DeviceMatrix,
    pub(super) lm_head: Option<DeviceMatrix>,
    pub(super) layers: Vec<TransformerBlock>,
    pub(super) norm: DeviceVec,
    /// Precomputed YaRN RoPE cos/sin cache: [max_seq_len, qk_rope_head_dim] bf16
    pub(super) cos_cache: DeviceVec,
    pub(super) sin_cache: DeviceVec,
    /// DeepEP intranode buffer for EP All-to-All (None if EP1).
    pub(super) deep_ep_buffer: Option<super::deep_ep::DeepEpBuffer>,
}

/// Pre-computed absorbed MLA weights for one layer.
///
/// At load time, kv_b_proj [32768, 512] FP8 is dequantized and split:
///   reshape as [128 heads, 256, 512] → per head [k_nope(128), v(128)] × 512
///   W_UK_h = kv_b_proj_bf16[h, 0:128,   :] → [128, 512]   (K nope weights)
///   W_UV_h = kv_b_proj_bf16[h, 128:256, :] → [128, 512]   (V weights)
///
/// Stored contiguously: w_uk [128, 128, 512], w_uv [128, 128, 512] in bf16.
/// Each head's sub-matrix is row-major [head_dim, kv_lora_rank].
pub(crate) struct AbsorbedMlaWeights {
    /// W_UK: [num_heads, qk_nope_head_dim, kv_lora_rank] bf16
    pub(crate) w_uk: CudaSlice<bf16>,
    /// W_UV: [num_heads, v_head_dim, kv_lora_rank] bf16
    pub(crate) w_uv: CudaSlice<bf16>,
}

unsafe impl Send for DsV32Model {}
unsafe impl Sync for DsV32Model {}

// ---------------------------------------------------------------------------
// FP8 loading helpers
// ---------------------------------------------------------------------------

fn find_tensor<'a>(
    shards: &'a [safetensors::SafeTensors<'a>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>> {
    if let Some(&idx) = weight_map.get(name) {
        shards[idx]
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("Failed to load tensor '{}': {}", name, e))
    } else {
        for shard in shards {
            if let Ok(t) = shard.tensor(name) {
                return Ok(t);
            }
        }
        Err(anyhow::anyhow!("Tensor '{}' not found in any shard", name))
    }
}

/// Load an FP8 weight matrix and its block-wise scale_inv tensor.
fn load_fp8_matrix(
    ctx: &DeviceContext,
    shards: &[safetensors::SafeTensors],
    weight_map: &HashMap<String, usize>,
    weight_name: &str,
    scale_name: &str,
) -> Result<Fp8Matrix> {
    // Load raw FP8 data (1 byte per element)
    let weight_tensor = find_tensor(shards, weight_map, weight_name)?;
    let shape = weight_tensor.shape();
    anyhow::ensure!(
        shape.len() == 2,
        "FP8 weight '{}' expected 2D, got {:?}",
        weight_name,
        shape
    );
    let rows = shape[0];
    let cols = shape[1];
    let raw_data = weight_tensor.data();
    anyhow::ensure!(
        raw_data.len() == rows * cols,
        "FP8 weight '{}' size mismatch: expected {} bytes, got {}",
        weight_name,
        rows * cols,
        raw_data.len()
    );
    let data = ctx
        .stream
        .clone_htod(raw_data)
        .map_err(|e| anyhow::anyhow!("H2D copy for '{}' failed: {}", weight_name, e))?;

    // Load scale_inv (f32)
    let scale_tensor = find_tensor(shards, weight_map, scale_name)?;
    let scale_shape = scale_tensor.shape();
    anyhow::ensure!(
        scale_shape.len() == 2,
        "Scale '{}' expected 2D, got {:?}",
        scale_name,
        scale_shape
    );
    let scale_rows = scale_shape[0];
    let scale_cols = scale_shape[1];
    let scale_bytes = scale_tensor.data();
    anyhow::ensure!(
        scale_bytes.len() == scale_rows * scale_cols * 4,
        "Scale '{}' size mismatch",
        scale_name
    );
    let scale_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(scale_bytes.as_ptr().cast::<f32>(), scale_rows * scale_cols)
    };
    let scale_inv = ctx
        .stream
        .clone_htod(scale_f32)
        .map_err(|e| anyhow::anyhow!("H2D copy for '{}' failed: {}", scale_name, e))?;

    Ok(Fp8Matrix {
        data,
        scale_inv,
        rows,
        cols,
        scale_rows,
        scale_cols,
    })
}

/// Load a 1D tensor stored as f32 and convert to bf16 DeviceVec.
/// DSV3.2 norm weights are stored as f32 in the checkpoint.
fn load_1d_f32_as_bf16(
    ctx: &DeviceContext,
    shards: &[safetensors::SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<DeviceVec> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let data = tensor.data();
    anyhow::ensure!(
        data.len() % 4 == 0,
        "f32 tensor '{}' byte length not multiple of 4",
        name
    );
    let f32_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), data.len() / 4) };
    let bf16_data: Vec<half::bf16> = f32_slice.iter().map(|&x| half::bf16::from_f32(x)).collect();
    DeviceVec::from_host(ctx, &bf16_data)
}

/// Load a 1D f32 tensor from safetensors.
fn load_1d_f32(
    ctx: &DeviceContext,
    shards: &[safetensors::SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<CudaSlice<f32>> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let data = tensor.data();
    anyhow::ensure!(
        data.len() % 4 == 0,
        "f32 tensor '{}' byte length not multiple of 4",
        name
    );
    let f32_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), data.len() / 4) };
    ctx.stream
        .clone_htod(f32_slice)
        .map_err(|e| anyhow::anyhow!("H2D copy for '{}' failed: {}", name, e))
}

/// Dequantize an FP8 matrix on CPU and return bf16 host data.
fn dequant_fp8_to_bf16_host(
    ctx: &DeviceContext,
    fp8: &Fp8Matrix,
    block_h: usize,
    block_w: usize,
) -> Vec<bf16> {
    // Download FP8 data and scales to CPU
    let fp8_host: Vec<u8> = ctx.stream.clone_dtoh(&fp8.data).expect("D2H fp8 data");
    let scale_host: Vec<f32> = ctx
        .stream
        .clone_dtoh(&fp8.scale_inv)
        .expect("D2H scale_inv");
    ctx.stream.synchronize().expect("sync");

    let rows = fp8.rows;
    let cols = fp8.cols;
    let mut out = vec![bf16::from_f32(0.0); rows * cols];

    for r in 0..rows {
        for c in 0..cols {
            let raw_byte = fp8_host[r * cols + c];
            // FP8 e4m3: sign(1) | exp(4) | man(3), bias=7
            let fp8_f32 = fp8_e4m3_to_f32(raw_byte);
            let scale_r = r / block_h;
            let scale_c = c / block_w;
            // scale_inv layout: [ceil(rows/block_h), ceil(cols/block_w)]
            // stored row-major: scale_inv[scale_r][scale_c]
            let scale_idx = scale_r * fp8.scale_cols + scale_c;
            let scale = scale_host[scale_idx];
            out[r * cols + c] = bf16::from_f32(fp8_f32 * scale);
        }
    }
    out
}

/// Convert FP8 e4m3 byte to f32.
fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let man = bits & 0x7;

    if exp == 0 && man == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    // e4m3 special: exp=15, man=7 is NaN (no inf in e4m3)
    if exp == 15 && man == 7 {
        return f32::NAN;
    }

    let bias = 7i32;
    let (effective_exp, effective_man) = if exp == 0 {
        // Subnormal: value = (-1)^sign * 2^(1-bias) * (0.man)
        (1 - bias, man as f32 / 8.0)
    } else {
        // Normal: value = (-1)^sign * 2^(exp-bias) * (1.man)
        (exp as i32 - bias, 1.0 + man as f32 / 8.0)
    };

    let val = effective_man * (2.0f32).powi(effective_exp);
    if sign == 1 { -val } else { val }
}

/// Compute absorbed MLA weights from kv_b_proj for one layer.
fn compute_absorbed_weights(
    ctx: &DeviceContext,
    kv_b_proj: &Fp8Matrix,
    num_heads: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    block_size: [usize; 2],
) -> Result<AbsorbedMlaWeights> {
    let kv_b_head_dim = qk_nope_head_dim + v_head_dim; // 256
    assert_eq!(kv_b_proj.rows, num_heads * kv_b_head_dim);
    assert_eq!(kv_b_proj.cols, kv_lora_rank);

    // Dequantize to bf16 on CPU
    let bf16_data = dequant_fp8_to_bf16_host(ctx, kv_b_proj, block_size[0], block_size[1]);

    // Split into W_UK and W_UV
    // kv_b_proj_bf16: [num_heads * kv_b_head_dim, kv_lora_rank] row-major
    // For head h: rows [h*kv_b_head_dim .. (h+1)*kv_b_head_dim]
    //   W_UK_h: first qk_nope_head_dim rows → [qk_nope_head_dim, kv_lora_rank]
    //   W_UV_h: next v_head_dim rows         → [v_head_dim, kv_lora_rank]

    let uk_size = num_heads * qk_nope_head_dim * kv_lora_rank;
    let uv_size = num_heads * v_head_dim * kv_lora_rank;
    let mut w_uk_host = Vec::with_capacity(uk_size);
    let mut w_uv_host = Vec::with_capacity(uv_size);

    for h in 0..num_heads {
        let head_offset = h * kv_b_head_dim * kv_lora_rank;
        // W_UK_h: first qk_nope_head_dim rows
        for r in 0..qk_nope_head_dim {
            let row_start = head_offset + r * kv_lora_rank;
            w_uk_host.extend_from_slice(&bf16_data[row_start..row_start + kv_lora_rank]);
        }
        // W_UV_h: next v_head_dim rows
        for r in 0..v_head_dim {
            let row_start = head_offset + (qk_nope_head_dim + r) * kv_lora_rank;
            w_uv_host.extend_from_slice(&bf16_data[row_start..row_start + kv_lora_rank]);
        }
    }

    let w_uk = ctx
        .stream
        .clone_htod(&w_uk_host)
        .map_err(|e| anyhow::anyhow!("H2D w_uk failed: {e}"))?;
    let w_uv = ctx
        .stream
        .clone_htod(&w_uv_host)
        .map_err(|e| anyhow::anyhow!("H2D w_uv failed: {e}"))?;

    Ok(AbsorbedMlaWeights { w_uk, w_uv })
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

impl DsV32Model {
    /// Load the full model (all layers). Will OOM on a single GPU — intended
    /// for multi-GPU loading where weights are sharded externally.
    pub fn from_safetensors(model_path: &str, device_ordinal: usize) -> Result<Self> {
        Self::load(model_path, device_ordinal, None, ParallelConfig::default())
    }

    /// Load only the first `max_layers` layers. Useful for single-GPU testing.
    pub fn from_safetensors_partial(
        model_path: &str,
        device_ordinal: usize,
        max_layers: usize,
    ) -> Result<Self> {
        Self::load(
            model_path,
            device_ordinal,
            Some(max_layers),
            ParallelConfig::default(),
        )
    }

    /// Load model with expert-parallel sharding.
    /// Each rank loads only its slice of routed experts; everything else is replicated.
    pub fn from_safetensors_parallel(
        model_path: &str,
        device_ordinal: usize,
        parallel: ParallelConfig,
    ) -> Result<Self> {
        Self::load(model_path, device_ordinal, None, parallel)
    }

    /// Load partial model with expert-parallel sharding. For testing.
    pub fn from_safetensors_partial_parallel(
        model_path: &str,
        device_ordinal: usize,
        max_layers: usize,
        parallel: ParallelConfig,
    ) -> Result<Self> {
        Self::load(model_path, device_ordinal, Some(max_layers), parallel)
    }

    fn load(
        model_path: &str,
        device_ordinal: usize,
        max_layers: Option<usize>,
        parallel: ParallelConfig,
    ) -> Result<Self> {
        let is_primary_log_rank = parallel.tp_rank == 0 && parallel.ep_rank == 0;
        if is_primary_log_rank {
            info!(
                "Loading DeepSeek-V3.2 from: {} (device={}, tp={}/{}, ep={}/{})",
                model_path,
                device_ordinal,
                parallel.tp_rank,
                parallel.tp_size,
                parallel.ep_rank,
                parallel.ep_size,
            );
        }
        debug!("Initializing GPU device {}", device_ordinal);
        let ctx = DeviceContext::new_with_device(device_ordinal)?;

        let config = DsV32Config::from_file(model_path)?;
        let num_layers_to_load = max_layers
            .map(|m| m.min(config.num_hidden_layers))
            .unwrap_or(config.num_hidden_layers);
        if is_primary_log_rank {
            info!(
                "Config: hidden={}, layers={} (loading {}), dense={}, moe={}, experts={}, heads={}, vocab={}",
                config.hidden_size,
                config.num_hidden_layers,
                num_layers_to_load,
                config.num_dense_layers(),
                config.num_moe_layers(),
                config.n_routed_experts,
                config.num_attention_heads,
                config.vocab_size,
            );
            info!(
                "MLA: q_lora_rank={}, kv_lora_rank={}, qk_nope={}, qk_rope={}, v={}",
                config.q_lora_rank,
                config.kv_lora_rank,
                config.qk_nope_head_dim,
                config.qk_rope_head_dim,
                config.v_head_dim,
            );
        }

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        if is_primary_log_rank {
            info!("Loading {} safetensor shard(s)", shard_paths.len());
        }
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;

        let t_gpu = Instant::now();

        // Embedding (bf16)
        debug!("Loading embeddings");
        let embed_tokens = load_tensor_2d(&ctx, &shards, &weight_map, "model.embed_tokens.weight")?;
        debug!(
            "embed_tokens: [{}, {}]",
            embed_tokens.rows, embed_tokens.cols
        );

        // LM head (bf16) — skip for partial loads to save memory
        let lm_head = if max_layers.is_some() {
            debug!("Partial load: skipping LM head");
            None
        } else if config.tie_word_embeddings {
            debug!("Using tied embeddings for LM head");
            None
        } else {
            debug!("Loading LM head");
            Some(load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                "lm_head.weight",
            )?)
        };

        // Layers
        if is_primary_log_rank {
            info!("Loading {} transformer layers", num_layers_to_load);
        }
        let mut layers = Vec::with_capacity(num_layers_to_load);
        for i in 0..num_layers_to_load {
            let prefix = format!("model.layers.{}", i);
            let attn_prefix = format!("{}.self_attn", prefix);

            // MLA weights (all layers have the same attention structure)
            let mla = MlaWeights {
                q_a_proj: load_fp8_matrix(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.q_a_proj.weight", attn_prefix),
                    &format!("{}.q_a_proj.weight_scale_inv", attn_prefix),
                )?,
                q_a_layernorm: load_1d_f32_as_bf16(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.q_a_layernorm.weight", attn_prefix),
                )?,
                q_b_proj: load_fp8_matrix(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.q_b_proj.weight", attn_prefix),
                    &format!("{}.q_b_proj.weight_scale_inv", attn_prefix),
                )?,
                kv_a_proj_with_mqa: load_fp8_matrix(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.kv_a_proj_with_mqa.weight", attn_prefix),
                    &format!("{}.kv_a_proj_with_mqa.weight_scale_inv", attn_prefix),
                )?,
                kv_a_layernorm: load_1d_f32_as_bf16(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.kv_a_layernorm.weight", attn_prefix),
                )?,
                kv_b_proj: load_fp8_matrix(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.kv_b_proj.weight", attn_prefix),
                    &format!("{}.kv_b_proj.weight_scale_inv", attn_prefix),
                )?,
                o_proj: load_fp8_matrix(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.o_proj.weight", attn_prefix),
                    &format!("{}.o_proj.weight_scale_inv", attn_prefix),
                )?,
            };

            // FFN: Dense or MoE
            let ffn = match config.layer_types[i] {
                FfnType::Dense => {
                    let mlp_prefix = format!("{}.mlp", prefix);
                    FfnWeights::Dense(DenseFFN {
                        gate_proj: load_fp8_matrix(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.gate_proj.weight", mlp_prefix),
                            &format!("{}.gate_proj.weight_scale_inv", mlp_prefix),
                        )?,
                        up_proj: load_fp8_matrix(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.up_proj.weight", mlp_prefix),
                            &format!("{}.up_proj.weight_scale_inv", mlp_prefix),
                        )?,
                        down_proj: load_fp8_matrix(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.down_proj.weight", mlp_prefix),
                            &format!("{}.down_proj.weight_scale_inv", mlp_prefix),
                        )?,
                    })
                }
                FfnType::MoE => {
                    let mlp_prefix = format!("{}.mlp", prefix);

                    // Gate (router) — bf16 weight + f32 bias
                    let gate = MoeGate {
                        weight: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.gate.weight", mlp_prefix),
                        )?,
                        e_score_correction_bias: load_1d_f32(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.gate.e_score_correction_bias", mlp_prefix),
                        )?,
                    };

                    // Routed experts — load only this rank's slice under EP
                    let (expert_start, num_local_experts) =
                        parallel.local_expert_range(config.n_routed_experts);
                    let mut experts = Vec::with_capacity(num_local_experts);
                    for e in expert_start..expert_start + num_local_experts {
                        let ep = format!("{}.experts.{}", mlp_prefix, e);
                        experts.push(ExpertFFN {
                            gate_proj: load_fp8_matrix(
                                &ctx,
                                &shards,
                                &weight_map,
                                &format!("{}.gate_proj.weight", ep),
                                &format!("{}.gate_proj.weight_scale_inv", ep),
                            )?,
                            up_proj: load_fp8_matrix(
                                &ctx,
                                &shards,
                                &weight_map,
                                &format!("{}.up_proj.weight", ep),
                                &format!("{}.up_proj.weight_scale_inv", ep),
                            )?,
                            down_proj: load_fp8_matrix(
                                &ctx,
                                &shards,
                                &weight_map,
                                &format!("{}.down_proj.weight", ep),
                                &format!("{}.down_proj.weight_scale_inv", ep),
                            )?,
                        });
                    }

                    // Shared expert
                    let sep = format!("{}.shared_experts", mlp_prefix);
                    let shared_expert = ExpertFFN {
                        gate_proj: load_fp8_matrix(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.gate_proj.weight", sep),
                            &format!("{}.gate_proj.weight_scale_inv", sep),
                        )?,
                        up_proj: load_fp8_matrix(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.up_proj.weight", sep),
                            &format!("{}.up_proj.weight_scale_inv", sep),
                        )?,
                        down_proj: load_fp8_matrix(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.down_proj.weight", sep),
                            &format!("{}.down_proj.weight_scale_inv", sep),
                        )?,
                    };

                    FfnWeights::MoE(MoeFfnWeights {
                        gate,
                        experts,
                        shared_expert,
                    })
                }
            };

            // Pre-compute absorbed MLA weights (W_UK, W_UV) from kv_b_proj
            let absorbed = compute_absorbed_weights(
                &ctx,
                &mla.kv_b_proj,
                config.num_attention_heads,
                config.qk_nope_head_dim,
                config.v_head_dim,
                config.kv_lora_rank,
                config.weight_block_size,
            )?;

            // NSA indexer weights (if model has indexer config)
            let indexer = if config.index_head_dim.is_some() {
                let idx_prefix = format!("{}.indexer", attn_prefix);
                Some(IndexerWeights {
                    wq_b: load_fp8_matrix(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.wq_b.weight", idx_prefix),
                        &format!("{}.wq_b.weight_scale_inv", idx_prefix),
                    )?,
                    wk: load_fp8_matrix(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.wk.weight", idx_prefix),
                        &format!("{}.wk.weight_scale_inv", idx_prefix),
                    )?,
                    k_norm_weight: load_1d_f32_as_bf16(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.k_norm.weight", idx_prefix),
                    )?,
                    k_norm_bias: load_1d_f32_as_bf16(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.k_norm.bias", idx_prefix),
                    )?,
                    weights_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.weights_proj.weight", idx_prefix),
                    )?,
                })
            } else {
                None
            };

            let block = TransformerBlock {
                input_layernorm: load_1d_f32_as_bf16(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                mla,
                absorbed,
                indexer,
                post_attention_layernorm: load_1d_f32_as_bf16(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                ffn,
            };

            let ffn_type = config.layer_types[i];
            if ffn_type == FfnType::MoE && parallel.is_ep_sharded() {
                let (start, count) = parallel.local_expert_range(config.n_routed_experts);
                debug!(
                    "Loaded layer {}/{}: {:?} (experts {}..{}, shared)",
                    i + 1,
                    num_layers_to_load,
                    ffn_type,
                    start,
                    start + count,
                );
            } else {
                debug!(
                    "Loaded layer {}/{}: {:?}",
                    i + 1,
                    num_layers_to_load,
                    ffn_type,
                );
            }
            layers.push(block);
        }

        // Final norm — skip for partial loads (it's in a late shard)
        let norm = if max_layers.is_some() {
            debug!("Partial load: using zeros for final norm");
            DeviceVec::zeros(&ctx, config.hidden_size)?
        } else {
            load_1d_f32_as_bf16(&ctx, &shards, &weight_map, "model.norm.weight")?
        };

        // Precompute YaRN RoPE cos/sin cache
        debug!(
            "Precomputing YaRN RoPE cache (head_dim={}, max_seq_len={})",
            config.qk_rope_head_dim, config.max_position_embeddings
        );
        let (cos_cache, sin_cache) = precompute_yarn_rope(
            &ctx,
            config.qk_rope_head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.yarn_beta_fast,
            config.yarn_beta_slow,
            config.yarn_factor,
            config.yarn_original_max_position_embeddings,
        )?;

        ctx.sync()?;
        let loaded_desc = if max_layers.is_some() {
            format!(
                "{}/{} layers (partial)",
                num_layers_to_load, config.num_hidden_layers
            )
        } else {
            "all layers".to_string()
        };
        if is_primary_log_rank {
            info!(
                "GPU transfer complete in {:.1}s ({})",
                t_gpu.elapsed().as_secs_f64(),
                loaded_desc,
            );
            info!("DeepSeek-V3.2 model loaded successfully");
        }

        Ok(Self {
            ctx,
            config,
            parallel,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            deep_ep_buffer: None,
        })
    }

    pub(crate) fn config(&self) -> &DsV32Config {
        &self.config
    }

    pub(crate) fn device_ctx(&self) -> &DeviceContext {
        &self.ctx
    }

    pub(crate) fn parallel(&self) -> &ParallelConfig {
        &self.parallel
    }

    pub(super) fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }
}
