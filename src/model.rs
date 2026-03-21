//! Qwen3 model: weights, forward pass, generation.

use anyhow::Result;
use fastrace::local::LocalSpan;
use log::{debug, info};
use std::time::Instant;

use rand::RngExt;
use rand::rngs::StdRng;

use cudarc::driver::CudaSlice;
use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;

use crate::decode_buffers::DecodeBuffers;
use crate::kv_cache::KVCache;
use crate::ops;
use crate::qwen3_config::Config;
use crate::sampler::{self, SamplingParams};
use crate::tensor::*;
use crate::weight_loader::*;

/// CUDA Graph state for decode path.
/// First decode call captures the graph; subsequent calls replay it.
struct CudaGraphState {
    graph: Option<CudaGraph>,
}

// SAFETY: CudaGraph contains raw CUDA pointers that are not Send by default.
// We only access the graph from the single inference thread that owns the model.
unsafe impl Send for CudaGraphState {}

#[derive(Clone, Copy, Debug)]
pub struct ModelRuntimeConfig {
    pub enable_cuda_graph: bool,
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
        }
    }
}

/// Attention layer weights
pub struct Attention {
    pub q_proj: DeviceMatrix,
    pub k_proj: DeviceMatrix,
    pub v_proj: DeviceMatrix,
    pub o_proj: DeviceMatrix,
    pub q_norm: DeviceVec,
    pub k_norm: DeviceVec,
}

/// MLP layer weights
pub struct MLP {
    pub gate_proj: DeviceMatrix,
    pub up_proj: DeviceMatrix,
    pub down_proj: DeviceMatrix,
}

/// Transformer block
pub struct TransformerBlock {
    pub input_layernorm: DeviceVec,
    pub attention: Attention,
    pub post_attention_layernorm: DeviceVec,
    pub mlp: MLP,
}

/// Qwen3 model
pub struct Qwen3Model {
    pub ctx: DeviceContext,
    pub config: Config,
    pub embed_tokens: DeviceMatrix,
    lm_head: Option<DeviceMatrix>,
    pub layers: Vec<TransformerBlock>,
    pub norm: DeviceVec,
    // RoPE cache on GPU - contiguous buffer [max_seq_len * head_dim]
    pub cos_cache: DeviceVec,
    pub sin_cache: DeviceVec,
    // Persistent decode state (reused across generate calls for CUDA Graph replay)
    decode_bufs: Option<DecodeBuffers>,
    kv_cache: Option<KVCache>,
    graph_state: CudaGraphState,
    enable_cuda_graph: bool,
}

/// Streaming generation summary for transport layers.
pub struct StreamingStats {
    pub emitted_tokens: usize,
    pub hit_eos: bool,
    pub consumer_dropped: bool,
}

/// Pre-allocated scratch buffers for one prefill forward pass.
/// Created once per prefill in `process_all_layers_batch`, eliminating
/// per-layer `cuMemAllocAsync` overhead (~11k calls / 88ms at seq=2048).
///
/// Buffer reuse across steps (all kernels serialized on a single stream):
///   `normed`  reused for `normed2`  (steps 1-4 done before step 8)
///   `o_buf`   reused for `mlp_out`  (step 7 done before step 12)
struct PrefillBuffers {
    /// Output ping-pong: layer writes result here; caller swaps with the incoming hidden.
    hidden_out: HiddenStates, // hidden_dim × seq_len
    normed: HiddenStates,      // hidden_dim × seq_len (reused for normed2)
    q_batch: HiddenStates,     // q_dim × seq_len
    k_batch: HiddenStates,     // kv_dim × seq_len
    v_batch: HiddenStates,     // kv_dim × seq_len
    o_buf: HiddenStates,       // hidden_dim × seq_len (reused for mlp_out)
    gate_out: HiddenStates,    // inter_dim × seq_len
    up_out: HiddenStates,      // inter_dim × seq_len
    act_out: HiddenStates,     // inter_dim × seq_len
    attn_output: HiddenStates, // q_dim × seq_len
}

impl PrefillBuffers {
    fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            normed: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, seq_len)?,
        })
    }
}

impl Qwen3Model {
    pub fn from_safetensors(model_path: &str) -> Result<Self> {
        Self::from_safetensors_with_runtime(model_path, ModelRuntimeConfig::default())
    }

    pub fn from_safetensors_with_runtime(
        model_path: &str,
        runtime: ModelRuntimeConfig,
    ) -> Result<Self> {
        info!("Loading model from: {}", model_path);
        debug!("Initializing GPU");
        let ctx = DeviceContext::new()?;

        let config = Config::from_file(model_path)?;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        debug!("Loading {} safetensor shard(s)", shard_paths.len());
        let mmaps = mmap_shards(&shard_paths)?;
        let shards: Vec<safetensors::SafeTensors> = mmaps
            .iter()
            .map(|m| {
                safetensors::SafeTensors::deserialize(m)
                    .map_err(|e| anyhow::anyhow!("Deserialize error: {}", e))
            })
            .collect::<Result<_>>()?;

        let t_gpu = Instant::now();
        debug!("Loading embeddings to GPU");
        let embed_tokens = load_tensor_2d(&ctx, &shards, &weight_map, "model.embed_tokens.weight")?;
        let lm_head = if config.tie_word_embeddings {
            debug!("Using tied input/output embeddings");
            None
        } else {
            debug!("Loading untied LM head to GPU");
            Some(load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                config.lm_head_tensor_name(),
            )?)
        };

        debug!(
            "Loading layers to GPU: num_layers={}",
            config.num_hidden_layers
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);

            let block = TransformerBlock {
                input_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                attention: Attention {
                    q_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.q_proj.weight", prefix),
                    )?,
                    k_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.k_proj.weight", prefix),
                    )?,
                    v_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.v_proj.weight", prefix),
                    )?,
                    o_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.o_proj.weight", prefix),
                    )?,
                    q_norm: load_tensor_1d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.q_norm.weight", prefix),
                    )?,
                    k_norm: load_tensor_1d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.k_norm.weight", prefix),
                    )?,
                },
                post_attention_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                mlp: MLP {
                    gate_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.gate_proj.weight", prefix),
                    )?,
                    up_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.up_proj.weight", prefix),
                    )?,
                    down_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.down_proj.weight", prefix),
                    )?,
                },
            };
            layers.push(block);
        }

        let norm = load_tensor_1d(&ctx, &shards, &weight_map, "model.norm.weight")?;

        debug!("Precomputing RoPE cache on GPU");
        let (cos_cache, sin_cache) =
            precompute_rope(&ctx, config.head_dim, 4096, config.rope_theta)?;

        ctx.sync()?;
        info!(
            "GPU transfer complete in {:.0}ms",
            t_gpu.elapsed().as_secs_f64() * 1e3
        );
        info!("GPU model loaded successfully");

        let model = Self {
            ctx,
            config,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            decode_bufs: None,
            kv_cache: None,
            graph_state: CudaGraphState { graph: None },
            enable_cuda_graph: runtime.enable_cuda_graph,
        };

        if model.enable_cuda_graph {
            debug!("Preloading decode-path Triton kernels before CUDA Graph capture");
            model.preload_decode_triton_kernels()?;
            debug!("Decode path CUDA Graph is enabled");
        } else {
            debug!("Decode path CUDA Graph is disabled");
        }

        Ok(model)
    }

    fn preload_decode_triton_kernels(&self) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let cache_len = self.config.num_key_value_heads * 4096 * self.config.head_dim;
        let dummy_token_id = 0_i32;
        let dummy_pos = 0_i32;
        let dummy_seq_len = 1_i32;

        let decode_meta = self
            .ctx
            .stream
            .clone_htod(&[dummy_token_id, dummy_pos, dummy_seq_len])
            .map_err(|e| anyhow::anyhow!("Preload decode_meta H2D failed: {}", e))?;
        let mut embed_out = DeviceVec::zeros(&self.ctx, hidden_size)?;
        ops::embedding_decode_into(&self.ctx, &self.embed_tokens, &decode_meta, &mut embed_out)?;

        let layer0 = &self.layers[0];
        let q = DeviceVec::zeros(&self.ctx, q_dim)?;
        let k = DeviceVec::zeros(&self.ctx, kv_dim)?;
        let v = DeviceVec::zeros(&self.ctx, kv_dim)?;
        let mut k_cache = DeviceVec::zeros(&self.ctx, cache_len)?;
        let mut v_cache = DeviceVec::zeros(&self.ctx, cache_len)?;
        let mut out = DeviceVec::zeros(&self.ctx, q_dim)?;

        let num_qheads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let num_kv_splits = 4usize;
        let mut partial_out = self
            .ctx
            .stream
            .alloc_zeros::<f32>(num_qheads * num_kv_splits * head_dim)
            .map_err(|e| anyhow::anyhow!("Alloc partial_out failed: {}", e))?;
        let mut partial_m = self
            .ctx
            .stream
            .alloc_zeros::<f32>(num_qheads * num_kv_splits)
            .map_err(|e| anyhow::anyhow!("Alloc partial_m failed: {}", e))?;
        let mut partial_l = self
            .ctx
            .stream
            .alloc_zeros::<f32>(num_qheads * num_kv_splits)
            .map_err(|e| anyhow::anyhow!("Alloc partial_l failed: {}", e))?;

        ops::fused_attention_decode_into(
            &self.ctx,
            &q,
            &k,
            &v,
            &layer0.attention.q_norm,
            &layer0.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &decode_meta,
            &mut k_cache,
            &mut v_cache,
            &mut out,
            &mut partial_out,
            &mut partial_m,
            &mut partial_l,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
        )?;

        self.ctx.sync()?;
        Ok(())
    }

    /// Forward pass returning final logits (for accuracy testing).
    pub fn forward_logits(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let mut kv_cache = KVCache::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
        );
        let start_pos = kv_cache.len();
        let hidden = self.get_embeddings_batch(token_ids)?;
        let hidden = self.process_all_layers_batch(hidden, start_pos, &mut kv_cache)?;
        let logits = self.compute_logits_batch(&hidden)?;
        logits.to_host(&self.ctx)
    }

    // ============================================================
    // Batched forward path (prefill)
    // ============================================================

    #[fastrace::trace(name = "get_embeddings_batch")]
    fn get_embeddings_batch(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        let seq_len = token_ids.len();
        let hidden_dim = self.config.hidden_size;

        // Copy token IDs to GPU
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(&token_ids_i32)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let mut out = HiddenStates::zeros(&self.ctx, hidden_dim, seq_len)?;
        ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids_gpu, &mut out)?;

        Ok(out)
    }

    #[fastrace::trace(name = "process_all_layers_batch")]
    fn process_all_layers_batch(
        &self,
        mut hidden: HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<HiddenStates> {
        let seq_len = hidden.seq_len;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let inter_dim = self.config.intermediate_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Allocate all intermediates once — eliminates ~11k cuMemAllocAsync calls.
        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            seq_len,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch(
                layer_idx,
                layer,
                &mut hidden,
                start_pos,
                kv_cache,
                &mut bufs,
            )?;
        }

        // Increment sequence length AFTER all layers processed
        for _ in 0..seq_len {
            kv_cache.increment_seq_len();
        }

        Ok(hidden)
    }

    fn compute_logits_batch(&self, hidden: &HiddenStates) -> Result<DeviceVec> {
        let last_hidden = ops::extract_vec(&self.ctx, hidden, hidden.seq_len - 1)?;
        let normed = ops::rms_norm(
            &self.ctx,
            &last_hidden,
            &self.norm,
            self.config.rms_norm_eps,
        )?;
        ops::linear(&self.ctx, &normed, self.output_projection())
    }

    fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    fn forward_layer_batch(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        kv_cache.init_if_needed(&self.ctx, head_dim)?;

        // 1. RMSNorm → bufs.normed
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        // 2. QKV projections → bufs.q_batch, bufs.k_batch, bufs.v_batch
        ops::gemm_into(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q_batch,
        )?;
        ops::gemm_into(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k_batch,
        )?;
        ops::gemm_into(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v_batch,
        )?;

        // 3. FlashAttention-2 (Triton) → bufs.attn_output
        let (k_cache_layer, v_cache_layer) = kv_cache.get_cache_mut(&self.ctx, layer_idx)?;
        ops::prefill_attention_batch(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            k_cache_layer,
            v_cache_layer,
            &mut bufs.attn_output,
            num_heads,
            num_kv_heads,
            head_dim,
            start_pos,
            self.config.rms_norm_eps,
        )?;

        // 4. O projection → bufs.o_buf (as o_batch)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        )?;

        // 5. Residual add: hidden_in + o_batch → bufs.hidden_out
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        // Swap: hidden = attn_residual, bufs.hidden_out = old hidden_in (now free)
        std::mem::swap(hidden, &mut bufs.hidden_out);

        // 6. MLP RMSNorm → bufs.normed (reused for normed2; steps 1-4 are done)
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        // 7. MLP: gate + up → act → down → bufs.o_buf (reused for mlp_out; step 5 is done)
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_proj,
            &bufs.normed,
            &mut bufs.gate_out,
        )?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.up_proj,
            &bufs.normed,
            &mut bufs.up_out,
        )?;
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        )?;

        // 8. Residual add: attn_residual + mlp_out → bufs.hidden_out (old hidden_in, free to overwrite)
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        // Swap: hidden = layer output, bufs.hidden_out = attn_residual (free next layer)
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }

    fn select_token(
        &self,
        logits: &DeviceVec,
        params: &SamplingParams,
        rng: &mut StdRng,
        sample_probs: Option<&mut CudaSlice<f32>>,
    ) -> Result<u32> {
        if params.is_greedy() {
            ops::argmax(&self.ctx, logits)
        } else if let Some(probs) = sample_probs {
            // GPU sampling: temperature → softmax → top-k → top-p → multinomial
            let random_val: f32 = rng.random();
            ops::gpu_sample(&self.ctx, logits, probs, params, random_val)
        } else {
            // CPU fallback (prefill first token — only called once)
            let logits_f32 = logits.to_host(&self.ctx)?;
            Ok(sampler::sample(&logits_f32, params, rng))
        }
    }

    // ============================================================
    // Zero-allocation decode path (pre-allocated buffers)
    // ============================================================

    /// Single decode step using pre-allocated buffers. Zero GPU allocation.
    /// With CUDA Graph: first call captures, subsequent calls replay.
    fn decode_one_token(
        &self,
        token_id: u32,
        kv_cache: &mut KVCache,
        bufs: &mut DecodeBuffers,
        graph_state: &mut CudaGraphState,
    ) -> Result<()> {
        let pos = kv_cache.len();
        let seq_len = pos + 1;

        // Ensure KV cache is initialized (no-op after first call)
        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        // Upload decode metadata to GPU (outside graph — values change each step)
        self.ctx
            .stream
            .memcpy_htod(
                &[token_id as i32, pos as i32, seq_len as i32],
                &mut bufs.decode_meta,
            )
            .map_err(|e| anyhow::anyhow!("H2D decode_meta failed: {}", e))?;

        if !self.enable_cuda_graph {
            self.decode_kernels(kv_cache, bufs)?;
            kv_cache.increment_seq_len();
            return Ok(());
        }

        match &graph_state.graph {
            Some(graph) => {
                // Replay captured graph
                graph
                    .launch()
                    .map_err(|e| anyhow::anyhow!("CUDA Graph launch failed: {}", e))?;
            }
            None => {
                // First call: capture the kernel sequence into a graph
                debug!("Capturing CUDA Graph for decode path...");
                self.ctx
                    .stream
                    .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                    .map_err(|e| anyhow::anyhow!("begin_capture failed: {}", e))?;

                self.decode_kernels(kv_cache, bufs)?;

                graph_state.graph = self
                    .ctx
                    .stream
                    .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                    .map_err(|e| anyhow::anyhow!("end_capture failed: {}", e))?;
                debug!("CUDA Graph captured successfully");

                // Capture only records — kernels don't execute. Launch to actually compute.
                if let Some(ref graph) = graph_state.graph {
                    graph
                        .launch()
                        .map_err(|e| anyhow::anyhow!("CUDA Graph first launch failed: {}", e))?;
                }
            }
        }

        kv_cache.increment_seq_len();
        Ok(())
    }

    /// Pure kernel sequence for decode — no CPU-GPU sync, no allocation.
    /// Called during graph capture and also replayed via CUDA Graph.
    fn decode_kernels(&self, kv_cache: &mut KVCache, bufs: &mut DecodeBuffers) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let num_layers = self.layers.len();

        // 1. Embedding (reads token_id from decode_meta[0])
        ops::embedding_decode_into(
            &self.ctx,
            &self.embed_tokens,
            &bufs.decode_meta,
            &mut bufs.hidden,
        )?;

        // 2. First layer input norm (standalone — no prior residual to fuse with)
        ops::rms_norm_into(
            &self.ctx,
            &bufs.hidden,
            &self.layers[0].input_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        // 3. All transformer layers
        //    Each layer assumes normed is already filled by the previous step.
        //    Post-MLP residual is fused with the next norm (next layer's input or final norm).
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.decode_layer_inner(layer_idx, layer, kv_cache, bufs)?;

            // Fused: hidden += mlp_out; normed = rms_norm(hidden, next_weight)
            let next_weight = if layer_idx + 1 < num_layers {
                &self.layers[layer_idx + 1].input_layernorm
            } else {
                &self.norm // final norm for last layer
            };
            ops::fused_add_rms_norm_into(
                &self.ctx,
                &mut bufs.hidden,
                &bufs.mlp_out,
                next_weight,
                eps,
                &mut bufs.normed,
            )?;
        }

        // 4. LM Head (normed already computed by last fused_add_rms_norm)
        ops::gemv(
            &self.ctx,
            self.output_projection(),
            &bufs.normed,
            &mut bufs.logits,
        )?;

        Ok(())
    }

    /// Inner decode layer: assumes normed is pre-filled, does NOT do post-MLP residual.
    /// Post-MLP residual + next norm is handled by the caller via fused_add_rms_norm.
    fn decode_layer_inner(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        kv_cache: &mut KVCache,
        bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        // normed is already filled (by caller)

        // QKV projection: normed → q, k, v (3× gemv)
        ops::gemv(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q,
        )?;
        ops::gemv(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k,
        )?;
        ops::gemv(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v,
        )?;

        // Fused Attention (decode variant): split-KV + reduce
        let (k_cache, v_cache) = kv_cache.get_cache_mut(&self.ctx, layer_idx)?;
        ops::fused_attention_decode_into(
            &self.ctx,
            &bufs.q,
            &bufs.k,
            &bufs.v,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.decode_meta,
            k_cache,
            v_cache,
            &mut bufs.attn_out,
            &mut bufs.partial_out,
            &mut bufs.partial_m,
            &mut bufs.partial_l,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
        )?;

        // O projection: attn_out → attn_proj
        ops::gemv(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.attn_proj,
        )?;

        // Fused residual + post-attention RMSNorm: hidden += attn_proj; normed = rms_norm(hidden)
        ops::fused_add_rms_norm_into(
            &self.ctx,
            &mut bufs.hidden,
            &bufs.attn_proj,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        // Fused MLP: normed → mlp_out (post-MLP residual handled by caller)
        ops::fused_mlp_into(
            &self.ctx,
            &bufs.normed,
            &layer.mlp.gate_proj,
            &layer.mlp.up_proj,
            &layer.mlp.down_proj,
            &mut bufs.mlp_act,
            &mut bufs.mlp_out,
        )?;

        Ok(())
    }

    /// Take decode buffers from self, lazily allocating on first use.
    fn take_decode_bufs(&mut self) -> Result<DecodeBuffers> {
        match self.decode_bufs.take() {
            Some(bufs) => Ok(bufs),
            None => DecodeBuffers::new(&self.ctx, &self.config),
        }
    }

    /// Take KV cache from self, lazily creating on first use. Resets seq_len for new request.
    fn take_kv_cache(&mut self) -> KVCache {
        match self.kv_cache.take() {
            Some(mut kv) => {
                kv.reset();
                kv
            }
            None => KVCache::new(
                self.config.num_hidden_layers,
                self.config.num_key_value_heads,
            ),
        }
    }

    /// Take graph state from self, leaving a fresh empty state.
    fn take_graph_state(&mut self) -> CudaGraphState {
        std::mem::replace(&mut self.graph_state, CudaGraphState { graph: None })
    }

    /// Generate tokens
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");
        let _span = LocalSpan::enter_with_local_parent("generate").with_properties(|| {
            [
                ("prompt_len", prompt_tokens.len().to_string()),
                ("max_new_tokens", max_new_tokens.to_string()),
            ]
        });

        // Process prompt and measure TTFT
        let ttft_start = Instant::now();
        let mut tokens = prompt_tokens.to_vec();
        let mut kv_cache = self.take_kv_cache();

        // Take persistent decode state early (needed for single-token prefill optimization)
        let mut bufs = self.take_decode_bufs()?;
        let mut graph_state = self.take_graph_state();

        let next_token = if prompt_tokens.len() == 1 {
            // Single-token prompt: use decode path (GEMV + fused kernels + CUDA Graph)
            // Avoids batch GEMM overhead, ~580 temp allocations, and non-fused MLP.
            let _span = LocalSpan::enter_with_local_parent("prefill_decode")
                .with_property(|| ("prompt_tokens", "1".to_string()));
            self.decode_one_token(prompt_tokens[0], &mut kv_cache, &mut bufs, &mut graph_state)?;
            self.select_token(&bufs.logits, params, rng, Some(&mut bufs.sample_probs))?
        } else {
            // Multi-token prompt: batch prefill (GEMM)
            let _span = LocalSpan::enter_with_local_parent("prefill")
                .with_property(|| ("prompt_tokens", prompt_tokens.len().to_string()));
            let start_pos = kv_cache.len();
            let hidden = self.get_embeddings_batch(&tokens)?;
            let hidden = self.process_all_layers_batch(hidden, start_pos, &mut kv_cache)?;
            let logits = self.compute_logits_batch(&hidden)?;
            self.select_token(&logits, params, rng, None)?
        };

        let ttft = ttft_start.elapsed();

        LocalSpan::add_property(|| ("ttft_ms", format!("{:.2}", ttft.as_secs_f64() * 1000.0)));

        debug!(
            "TTFT: {:.2}ms (prompt_len={})",
            ttft.as_secs_f64() * 1000.0,
            prompt_tokens.len()
        );

        if !params.ignore_eos && self.config.is_stop_token(next_token) {
            self.kv_cache = Some(kv_cache);
            return Ok(tokens);
        }

        tokens.push(next_token);

        // Reuse the decode state and captured CUDA graph from the first token.

        // Generate new tokens using pre-allocated buffers + CUDA Graph
        let tpot_start = Instant::now();
        let mut generated_count = 0;
        for i in 1..max_new_tokens {
            let next_token = {
                let _span = LocalSpan::enter_with_local_parent("decode_step")
                    .with_property(|| ("step", i.to_string()));
                self.decode_one_token(
                    *tokens.last().unwrap(),
                    &mut kv_cache,
                    &mut bufs,
                    &mut graph_state,
                )?;
                self.select_token(&bufs.logits, params, rng, Some(&mut bufs.sample_probs))?
            };

            if !params.ignore_eos && self.config.is_stop_token(next_token) {
                break;
            }

            tokens.push(next_token);
            generated_count += 1;
        }

        // Put persistent state back for next generate call
        self.decode_bufs = Some(bufs);
        self.kv_cache = Some(kv_cache);
        self.graph_state = graph_state;

        if generated_count > 0 {
            let tpot_total = tpot_start.elapsed();
            let tpot_avg = tpot_total.as_secs_f64() / generated_count as f64;
            LocalSpan::add_properties(|| {
                [
                    ("tpot_avg_ms", format!("{:.2}", tpot_avg * 1000.0)),
                    ("generated_tokens", generated_count.to_string()),
                    (
                        "tok_per_sec",
                        format!("{:.1}", generated_count as f64 / tpot_total.as_secs_f64()),
                    ),
                ]
            });
            debug!(
                "TPOT: {:.2}ms/tok (generated {} tokens in {:.2}ms, {:.1} tok/s)",
                tpot_avg * 1000.0,
                generated_count,
                tpot_total.as_secs_f64() * 1000.0,
                generated_count as f64 / tpot_total.as_secs_f64()
            );
        }

        Ok(tokens)
    }

    /// Like `generate`, but invokes `on_token` for each new token.
    ///
    /// Return `false` from `on_token` to stop generation early (e.g. client disconnected).
    pub fn generate_streaming_with_callback<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
        mut on_token: F,
    ) -> Result<StreamingStats>
    where
        F: FnMut(u32) -> bool,
    {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");
        let _span =
            LocalSpan::enter_with_local_parent("generate_streaming").with_properties(|| {
                [
                    ("prompt_len", prompt_tokens.len().to_string()),
                    ("max_new_tokens", max_new_tokens.to_string()),
                ]
            });

        let mut tokens = prompt_tokens.to_vec();
        let mut kv_cache = self.take_kv_cache();

        // Take persistent decode state early (needed for single-token prefill optimization)
        let mut bufs = self.take_decode_bufs()?;
        let mut graph_state = self.take_graph_state();

        // Prefill
        let ttft_start = Instant::now();
        let next_token = if prompt_tokens.len() == 1 {
            let _span = LocalSpan::enter_with_local_parent("prefill_decode")
                .with_property(|| ("prompt_tokens", "1".to_string()));
            self.decode_one_token(prompt_tokens[0], &mut kv_cache, &mut bufs, &mut graph_state)?;
            self.select_token(&bufs.logits, params, rng, Some(&mut bufs.sample_probs))?
        } else {
            let _span = LocalSpan::enter_with_local_parent("prefill")
                .with_property(|| ("prompt_tokens", prompt_tokens.len().to_string()));
            let start_pos = kv_cache.len();
            let hidden = self.get_embeddings_batch(&tokens)?;
            let hidden = self.process_all_layers_batch(hidden, start_pos, &mut kv_cache)?;
            let logits = self.compute_logits_batch(&hidden)?;
            self.select_token(&logits, params, rng, None)?
        };

        let ttft = ttft_start.elapsed();
        debug!(
            "TTFT: {:.2}ms (prompt_len={})",
            ttft.as_secs_f64() * 1000.0,
            prompt_tokens.len()
        );

        if !params.ignore_eos && self.config.is_stop_token(next_token) {
            self.kv_cache = Some(kv_cache);
            return Ok(StreamingStats {
                emitted_tokens: 0,
                hit_eos: true,
                consumer_dropped: false,
            });
        }

        tokens.push(next_token);
        let mut emitted_tokens = 1usize;
        if !on_token(next_token) {
            self.decode_bufs = Some(bufs);
            self.kv_cache = Some(kv_cache);
            self.graph_state = graph_state;
            return Ok(StreamingStats {
                emitted_tokens,
                hit_eos: false,
                consumer_dropped: true,
            });
        }

        // Decode using pre-allocated buffers + CUDA Graph
        let tpot_start = Instant::now();
        let mut generated_count = 0;
        let mut hit_eos = false;
        for i in 1..max_new_tokens {
            let next_token = {
                let _span = LocalSpan::enter_with_local_parent("decode_step")
                    .with_property(|| ("step", i.to_string()));
                self.decode_one_token(
                    *tokens.last().unwrap(),
                    &mut kv_cache,
                    &mut bufs,
                    &mut graph_state,
                )?;
                self.select_token(&bufs.logits, params, rng, Some(&mut bufs.sample_probs))?
            };

            if !params.ignore_eos && self.config.is_stop_token(next_token) {
                hit_eos = true;
                break;
            }

            tokens.push(next_token);
            generated_count += 1;
            emitted_tokens += 1;

            if !on_token(next_token) {
                self.decode_bufs = Some(bufs);
                self.kv_cache = Some(kv_cache);
                self.graph_state = graph_state;
                return Ok(StreamingStats {
                    emitted_tokens,
                    hit_eos: false,
                    consumer_dropped: true,
                });
            }
        }

        // Put persistent state back for next generate call
        self.decode_bufs = Some(bufs);
        self.kv_cache = Some(kv_cache);
        self.graph_state = graph_state;

        if generated_count > 0 {
            let tpot_total = tpot_start.elapsed();
            let tpot_avg = tpot_total.as_secs_f64() / generated_count as f64;
            debug!(
                "TPOT: {:.2}ms/tok (generated {} tokens in {:.2}ms, {:.1} tok/s)",
                tpot_avg * 1000.0,
                generated_count,
                tpot_total.as_secs_f64() * 1000.0,
                generated_count as f64 / tpot_total.as_secs_f64()
            );
        }

        Ok(StreamingStats {
            emitted_tokens,
            hit_eos,
            consumer_dropped: false,
        })
    }

    /// Like `generate`, but sends each new token through `tx` as it's produced.
    /// The caller receives tokens via the corresponding `mpsc::Receiver`.
    pub fn generate_streaming(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
        tx: tokio::sync::mpsc::UnboundedSender<u32>,
    ) -> Result<()> {
        let _ = self.generate_streaming_with_callback(
            prompt_tokens,
            max_new_tokens,
            params,
            rng,
            |token_id| tx.send(token_id).is_ok(),
        )?;
        Ok(())
    }
}
