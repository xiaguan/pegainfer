use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use log::{debug, info};
use std::collections::HashMap;
use std::time::Instant;

use super::config::{DsV3Config, FfnType, ParallelConfig};
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
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) ffn: FfnWeights,
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// DeepSeek-V3.2 model weights.
pub struct DsV3Model {
    pub(super) ctx: DeviceContext,
    pub(super) config: DsV3Config,
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

unsafe impl Send for DsV3Model {}
unsafe impl Sync for DsV3Model {}

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
/// DSV3 norm weights are stored as f32 in the checkpoint.
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

impl DsV3Model {
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
        info!(
            "Loading DeepSeek-V3.2 from: {} (device={}, tp={}/{}, ep={}/{})",
            model_path,
            device_ordinal,
            parallel.tp_rank,
            parallel.tp_size,
            parallel.ep_rank,
            parallel.ep_size,
        );
        debug!("Initializing GPU device {}", device_ordinal);
        let ctx = DeviceContext::new_with_device(device_ordinal)?;

        let config = DsV3Config::from_file(model_path)?;
        let num_layers_to_load = max_layers
            .map(|m| m.min(config.num_hidden_layers))
            .unwrap_or(config.num_hidden_layers);
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

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        info!("Loading {} safetensor shard(s)", shard_paths.len());
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
        info!("Loading {} transformer layers", num_layers_to_load);
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

            let block = TransformerBlock {
                input_layernorm: load_1d_f32_as_bf16(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                mla,
                absorbed,
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
        info!(
            "GPU transfer complete in {:.1}s ({})",
            t_gpu.elapsed().as_secs_f64(),
            loaded_desc,
        );
        info!("DeepSeek-V3.2 model loaded successfully");

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

    pub(crate) fn config(&self) -> &DsV3Config {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::dsv3::config::DsV3Config;

    fn get_dsv3_model_path() -> Option<String> {
        std::env::var("PEGAINFER_DSV3_MODEL_PATH").ok()
    }

    /// Load reference f32 binary from test_data/dsv3_ref/.
    fn load_ref_bin(name: &str) -> Vec<f32> {
        let path = format!("test_data/dsv3_ref/{}.bin", name);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "Failed to read reference file '{}': {}. Run tools/dsv3_ref/main.py first.",
                path, e
            )
        });
        assert_eq!(
            bytes.len() % 4,
            0,
            "Reference file '{}' is not a multiple of 4 bytes",
            name
        );
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn dsv3_config_parse() {
        let model_path = match get_dsv3_model_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipped: set PEGAINFER_DSV3_MODEL_PATH to run");
                return;
            }
        };
        let config = DsV3Config::from_file(&model_path).expect("config parse failed");

        assert_eq!(config.hidden_size, 7168);
        assert_eq!(config.num_hidden_layers, 61);
        assert_eq!(config.num_attention_heads, 128);
        assert_eq!(config.vocab_size, 129280);

        // MLA
        assert_eq!(config.q_lora_rank, 1536);
        assert_eq!(config.kv_lora_rank, 512);
        assert_eq!(config.qk_nope_head_dim, 128);
        assert_eq!(config.qk_rope_head_dim, 64);
        assert_eq!(config.v_head_dim, 128);
        assert_eq!(config.q_head_dim(), 192);
        assert_eq!(config.kv_a_proj_dim(), 576);
        assert_eq!(config.kv_b_proj_dim(), 128 * 256); // 32768
        assert_eq!(config.q_b_proj_dim(), 128 * 192); // 24576
        assert_eq!(config.o_proj_input_dim(), 128 * 128); // 16384

        // MoE
        assert_eq!(config.first_k_dense_replace, 3);
        assert_eq!(config.n_routed_experts, 256);
        assert_eq!(config.n_shared_experts, 1);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.moe_intermediate_size, 2048);
        assert_eq!(config.num_dense_layers(), 3);
        assert_eq!(config.num_moe_layers(), 58);

        // FP8
        assert_eq!(config.weight_block_size, [128, 128]);

        // Layer types
        assert_eq!(config.layer_types[0], super::super::config::FfnType::Dense);
        assert_eq!(config.layer_types[2], super::super::config::FfnType::Dense);
        assert_eq!(config.layer_types[3], super::super::config::FfnType::MoE);
        assert_eq!(config.layer_types[60], super::super::config::FfnType::MoE);

        eprintln!("Config parsed OK: softmax_mscale={}", config.softmax_mscale);
    }

    /// Load embedding + 4 layers (3 dense + 1 MoE) on a single GPU.
    /// Covers both FP8 matrix loading and MoE expert loading paths.
    #[test]
    #[ignore]
    fn dsv3_partial_load_4_layers() {
        let model_path = match get_dsv3_model_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipped: set PEGAINFER_DSV3_MODEL_PATH to run");
                return;
            }
        };
        crate::logging::init_stderr("info");

        let model =
            DsV3Model::from_safetensors_partial(&model_path, 0, 4).expect("partial load failed");

        assert_eq!(model.layers.len(), 4);
        assert_eq!(model.embed_tokens.rows, 129280);
        assert_eq!(model.embed_tokens.cols, 7168);

        // Layer 0: dense FFN
        match &model.layers[0].ffn {
            FfnWeights::Dense(d) => {
                assert_eq!(d.gate_proj.rows, 18432);
                assert_eq!(d.gate_proj.cols, 7168);
            }
            _ => panic!("Layer 0 should be Dense"),
        }

        // Layer 0 MLA shapes
        let mla = &model.layers[0].mla;
        assert_eq!(mla.q_a_proj.rows, 1536);
        assert_eq!(mla.q_a_proj.cols, 7168);
        assert_eq!(mla.q_b_proj.rows, 24576);
        assert_eq!(mla.q_b_proj.cols, 1536);
        assert_eq!(mla.kv_a_proj_with_mqa.rows, 576);
        assert_eq!(mla.kv_a_proj_with_mqa.cols, 7168);
        assert_eq!(mla.kv_b_proj.rows, 32768);
        assert_eq!(mla.kv_b_proj.cols, 512);
        assert_eq!(mla.o_proj.rows, 7168);
        assert_eq!(mla.o_proj.cols, 16384);

        // Layer 3: MoE FFN
        match &model.layers[3].ffn {
            FfnWeights::MoE(moe) => {
                assert_eq!(moe.experts.len(), 256);
                assert_eq!(moe.experts[0].gate_proj.rows, 2048);
                assert_eq!(moe.experts[0].gate_proj.cols, 7168);
                assert_eq!(moe.gate.weight.rows, 256);
                assert_eq!(moe.gate.weight.cols, 7168);
            }
            _ => panic!("Layer 3 should be MoE"),
        }

        // Check VRAM usage
        let (free, total) = cudarc::driver::result::mem_get_info().unwrap();
        let used_mb = (total - free) / (1024 * 1024);
        eprintln!("VRAM used after partial load: {} MB", used_mb);
    }

    /// Verify forward correctness: embedding → RMSNorm → FP8 quantize → FP8 GEMM (q_a_proj).
    /// Compares against reference outputs generated by tools/dsv3_ref/main.py.
    #[test]
    #[ignore]
    fn dsv3_forward_embedding_q_a_proj() {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;

        let model_path = match get_dsv3_model_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipped: set PEGAINFER_DSV3_MODEL_PATH to run");
                return;
            }
        };
        crate::logging::init_stderr("info");

        // Load reference data
        let ref_embedded = load_ref_bin("embedded");
        let ref_normed = load_ref_bin("normed");
        let ref_q_a_output = load_ref_bin("q_a_output");

        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("test_data/dsv3_ref/meta.json").expect("meta.json not found"),
        )
        .expect("meta.json parse failed");

        let token_ids: Vec<u32> = meta["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let seq_len = token_ids.len();
        let hidden_size = meta["hidden_size"].as_u64().unwrap() as usize;
        let q_lora_rank = meta["q_lora_rank"].as_u64().unwrap() as usize;
        let rms_norm_eps = meta["rms_norm_eps"].as_f64().unwrap() as f32;

        eprintln!(
            "Reference: seq_len={}, hidden_size={}, q_lora_rank={}",
            seq_len, hidden_size, q_lora_rank
        );

        // Load model (only 1 layer needed)
        let model =
            DsV3Model::from_safetensors_partial(&model_path, 0, 1).expect("partial load failed");
        let ctx = &model.ctx;

        // ============================
        // Step 1: Embedding lookup
        // ============================
        let token_ids_gpu = ctx
            .stream
            .clone_htod(&token_ids)
            .expect("H2D token_ids failed");
        let mut hidden =
            crate::tensor::HiddenStates::zeros(ctx, hidden_size, seq_len).expect("alloc failed");
        crate::ops::embedding_batch(ctx, &model.embed_tokens, &token_ids_gpu, &mut hidden)
            .expect("embedding_batch failed");
        ctx.sync().expect("sync failed");

        // Verify embedding
        let host_embedded: Vec<bf16> = ctx.stream.clone_dtoh(&hidden.data).expect("D2H failed");
        ctx.sync().expect("sync failed");
        let host_embedded_f32: Vec<f32> = host_embedded.iter().map(|x| x.to_f32()).collect();

        assert_eq!(host_embedded_f32.len(), ref_embedded.len());
        let embed_max_err = host_embedded_f32
            .iter()
            .zip(ref_embedded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Embedding max error: {:.6e}", embed_max_err);
        assert!(
            embed_max_err < 1e-3,
            "Embedding error too large: {:.6e}",
            embed_max_err
        );

        // ============================
        // Step 2: RMSNorm
        // ============================
        let mut normed =
            crate::tensor::HiddenStates::zeros(ctx, hidden_size, seq_len).expect("alloc failed");
        crate::ops::rms_norm_batch_into(
            ctx,
            &hidden,
            &model.layers[0].input_layernorm,
            rms_norm_eps,
            &mut normed,
        );
        ctx.sync().expect("sync failed");

        let host_normed: Vec<bf16> = ctx.stream.clone_dtoh(&normed.data).expect("D2H failed");
        ctx.sync().expect("sync failed");
        let host_normed_f32: Vec<f32> = host_normed.iter().map(|x| x.to_f32()).collect();

        assert_eq!(host_normed_f32.len(), ref_normed.len());
        let norm_max_err = host_normed_f32
            .iter()
            .zip(ref_normed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("RMSNorm max error: {:.6e}", norm_max_err);
        // bf16 RMSNorm vs f32 reference: up to ~2e-2 max error is expected
        assert!(
            norm_max_err < 2e-2,
            "RMSNorm error too large: {:.6e}",
            norm_max_err
        );

        // ============================
        // Step 3: FP8 quantize activation
        // ============================
        // normed is [hidden_dim * seq_len] bf16, stored as token_0[7168] token_1[7168] ...
        // For FP8 GEMM: A[M, K] where M=seq_len, K=hidden_size
        let m = seq_len as i32;
        let k = hidden_size as i32;
        let n = q_lora_rank as i32; // q_a_proj output dim

        // Allocate FP8 activation buffer and scale buffer
        let mut fp8_act: cudarc::driver::CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(seq_len * hidden_size)
            .expect("alloc fp8_act failed");
        let scale_k_chunks = ((hidden_size + 127) / 128) as usize;
        let padded_m = ((seq_len + 3) / 4) * 4;
        let mut scale_a: cudarc::driver::CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(scale_k_chunks * padded_m)
            .expect("alloc scale_a failed");

        {
            let (normed_ptr, _gn) = normed.data.device_ptr(&ctx.stream);
            let (fp8_ptr, _gf) = fp8_act.device_ptr_mut(&ctx.stream);
            let (scale_a_ptr, _gs) = scale_a.device_ptr_mut(&ctx.stream);

            unsafe {
                crate::ffi::fp8_quantize_1x128_cuda(
                    normed_ptr as *const crate::ffi::Half,
                    fp8_ptr as *mut u8,
                    scale_a_ptr as *mut f32,
                    m,
                    k,
                    ctx.stream.cu_stream(),
                );
            }
        }
        ctx.sync().expect("sync after fp8_quantize failed");

        // ============================
        // Step 4: FP8 GEMM — q_a_proj
        // ============================
        // D[M, N] = A[M, K] @ B[N, K]^T
        // A = fp8_act (just quantized), scale_a = per-token 1x128 scales
        // B = q_a_proj.data (from checkpoint), scale_b = q_a_proj.scale_inv
        let q_a = &model.layers[0].mla.q_a_proj;
        assert_eq!(q_a.rows as i32, n);
        assert_eq!(q_a.cols as i32, k);

        let mut output_bf16: cudarc::driver::CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(seq_len * q_lora_rank)
            .expect("alloc output failed");

        {
            let (fp8_ptr, _gf) = fp8_act.device_ptr(&ctx.stream);
            let (scale_a_ptr, _gs) = scale_a.device_ptr(&ctx.stream);
            let (b_ptr, _gb) = q_a.data.device_ptr(&ctx.stream);
            let (scale_b_ptr, _gsb) = q_a.scale_inv.device_ptr(&ctx.stream);
            let (out_ptr, _go) = output_bf16.device_ptr_mut(&ctx.stream);

            unsafe {
                crate::ffi::fp8_gemm_cuda(
                    fp8_ptr as *const u8,
                    scale_a_ptr as *const f32,
                    b_ptr as *const u8,
                    scale_b_ptr as *const f32,
                    out_ptr as *mut crate::ffi::Half,
                    m,
                    n,
                    k,
                    ctx.stream.cu_stream(),
                );
            }
        }
        ctx.sync().expect("sync after fp8_gemm failed");

        // ============================
        // Step 5: Compare with reference
        // ============================
        let host_output: Vec<bf16> = ctx.stream.clone_dtoh(&output_bf16).expect("D2H failed");
        ctx.sync().expect("sync failed");
        let host_output_f32: Vec<f32> = host_output.iter().map(|x| x.to_f32()).collect();

        assert_eq!(
            host_output_f32.len(),
            ref_q_a_output.len(),
            "Output size mismatch: got {}, expected {}",
            host_output_f32.len(),
            ref_q_a_output.len()
        );

        // Compute max absolute error and mean absolute error
        let mut max_err = 0.0f32;
        let mut sum_err = 0.0f64;
        let mut max_err_idx = 0;
        for (i, (a, b)) in host_output_f32
            .iter()
            .zip(ref_q_a_output.iter())
            .enumerate()
        {
            let err = (a - b).abs();
            if err > max_err {
                max_err = err;
                max_err_idx = i;
            }
            sum_err += err as f64;
        }
        let mean_err = sum_err / host_output_f32.len() as f64;

        let token_idx = max_err_idx / q_lora_rank;
        let dim_idx = max_err_idx % q_lora_rank;
        eprintln!(
            "q_a_proj output: max_err={:.6e} at [{}, {}] (got={:.6}, ref={:.6}), mean_err={:.6e}",
            max_err,
            token_idx,
            dim_idx,
            host_output_f32[max_err_idx],
            ref_q_a_output[max_err_idx],
            mean_err
        );

        // FP8 GEMM introduces quantization error — reference uses fp32 dequant + matmul,
        // our path does online fp8 quantize (activation) + fp8 GEMM with block scales.
        // Tolerance: max error < 1.0 (very generous for FP8), mean error < 0.1
        assert!(
            max_err < 1.0,
            "q_a_proj max error too large: {:.6e}",
            max_err
        );
        assert!(
            mean_err < 0.1,
            "q_a_proj mean error too large: {:.6e}",
            mean_err
        );

        eprintln!("Forward verification PASSED");
    }

    /// Phase 1 end-to-end: full 3-layer forward (MLA + Dense FFN).
    ///
    /// Compares hidden states after each layer against reference outputs
    /// generated by tools/dsv3_ref/verify_phase1.py.
    #[test]
    #[ignore]
    fn dsv3_forward_3_layers() {
        use half::bf16;

        let model_path = match get_dsv3_model_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipped: set PEGAINFER_DSV3_MODEL_PATH to run");
                return;
            }
        };
        crate::logging::init_stderr("info");

        // Check reference data exists
        let phase1_dir = "test_data/dsv3_phase1";
        let meta_path = format!("{}/meta.json", phase1_dir);
        if !std::path::Path::new(&meta_path).exists() {
            eprintln!(
                "Skipped: run 'cd tools/dsv3_ref && uv run verify_phase1.py \
                 --model-path <path> --output-dir ../../test_data/dsv3_phase1' first"
            );
            return;
        }

        // Load metadata
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        let token_id = meta["token_id"].as_u64().unwrap() as u32;
        let position = meta["position"].as_i64().unwrap() as i32;
        let num_layers = meta["num_layers"].as_u64().unwrap() as usize;
        let hidden_size = meta["hidden_size"].as_u64().unwrap() as usize;

        eprintln!(
            "Phase 1 verification: token_id={}, position={}, num_layers={}",
            token_id, position, num_layers
        );

        // Load reference hidden states
        let load_ref = |name: &str| -> Vec<f32> {
            let path = format!("{}/{}.bin", phase1_dir, name);
            let bytes =
                std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to read '{}': {}", path, e));
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        let ref_embedded = load_ref("embedded");
        let mut ref_layer_hidden = Vec::new();
        let mut ref_layer_attn = Vec::new();
        for i in 0..num_layers {
            ref_layer_hidden.push(load_ref(&format!("layer{}_hidden", i)));
            ref_layer_attn.push(load_ref(&format!("layer{}_hidden_after_attn", i)));
        }

        eprintln!(
            "Reference data loaded: {} layers, hidden_size={}",
            num_layers, hidden_size
        );

        // Load model (3 dense layers)
        let model = DsV3Model::from_safetensors_partial(&model_path, 0, num_layers)
            .expect("model load failed");
        let ctx = model.device_ctx();
        let config = model.config();

        assert_eq!(config.hidden_size, hidden_size);

        // Allocate KV pool (small — just 1 request)
        let kv_pool = super::super::mla_kv::MlaKvPool::new(
            ctx,
            num_layers,
            config.kv_lora_rank,
            config.qk_rope_head_dim,
            64, // page_size
            32, // num_pages — plenty for 1 token
        )
        .expect("KV pool alloc failed");
        let mut kv_state = kv_pool.alloc();

        // Allocate forward buffers
        let mut bufs = super::super::forward::MlaForwardBuffers::new(ctx, config, 1)
            .expect("buffer alloc failed");

        // Embedding
        let token_ids_gpu = ctx.stream.clone_htod(&[token_id]).expect("H2D failed");
        let mut hidden =
            crate::tensor::HiddenStates::zeros(ctx, hidden_size, 1).expect("alloc hidden failed");
        hidden.seq_len = 1;
        crate::ops::embedding_batch(ctx, &model.embed_tokens, &token_ids_gpu, &mut hidden)
            .expect("embedding failed");

        // Verify embedding
        {
            let host: Vec<bf16> = ctx.stream.clone_dtoh(&hidden.data).expect("D2H");
            ctx.sync().unwrap();
            let host_f32: Vec<f32> = host.iter().map(|x| x.to_f32()).collect();
            let (max_err, mean_err) = compute_errors(&host_f32[..hidden_size], &ref_embedded);
            eprintln!(
                "Embedding: max_err={:.6e}, mean_err={:.6e}",
                max_err, mean_err
            );
            assert!(max_err < 1e-3, "Embedding error too large: {:.6e}", max_err);
        }

        // Forward through layers
        let positions = [position];
        for layer_idx in 0..num_layers {
            eprintln!("\n--- Layer {} ---", layer_idx);

            let mut kv_refs: Vec<&mut super::super::mla_kv::MlaKvState> = vec![&mut kv_state];
            model
                .forward_layer(
                    layer_idx,
                    &mut hidden,
                    &mut kv_refs,
                    &positions,
                    &mut bufs,
                    &model.cos_cache,
                    &model.sin_cache,
                    &kv_pool,
                )
                .expect("forward_layer failed");

            ctx.sync().unwrap();

            // Compare hidden states after full layer (attn + FFN)
            let host: Vec<bf16> = ctx.stream.clone_dtoh(&hidden.data).expect("D2H");
            let host_f32: Vec<f32> = host.iter().map(|x| x.to_f32()).collect();
            let (max_err, mean_err) =
                compute_errors(&host_f32[..hidden_size], &ref_layer_hidden[layer_idx]);
            eprintln!(
                "Layer {} hidden (after FFN): max_err={:.6e}, mean_err={:.6e}",
                layer_idx, max_err, mean_err
            );

            // FP8 GEMM introduces quantization error at each projection.
            // Each layer has ~7 FP8 GEMMs (q_a, q_b, kv_a, o_proj, gate, up, down).
            // Error accumulates across layers. FP8 outliers (max_err) can be
            // large while mean_err stays low.
            let layer_max_tol = 5.0 * (layer_idx as f32 + 1.0);
            let layer_mean_tol = 0.5 * (layer_idx as f32 + 1.0);
            assert!(
                max_err < layer_max_tol,
                "Layer {} max_err={:.6e} exceeds tolerance {:.1}",
                layer_idx,
                max_err,
                layer_max_tol
            );
            assert!(
                mean_err < layer_mean_tol as f64,
                "Layer {} mean_err={:.6e} exceeds tolerance {:.1}",
                layer_idx,
                mean_err,
                layer_mean_tol
            );
        }

        eprintln!("\nPhase 1 forward verification (3 layers) PASSED");
    }

    fn compute_errors(actual: &[f32], expected: &[f32]) -> (f32, f64) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        let mut max_err = 0.0f32;
        let mut sum_err = 0.0f64;
        for (a, b) in actual.iter().zip(expected.iter()) {
            let err = (a - b).abs();
            if err > max_err {
                max_err = err;
            }
            sum_err += err as f64;
        }
        let mean_err = sum_err / actual.len() as f64;
        (max_err, mean_err)
    }
}
