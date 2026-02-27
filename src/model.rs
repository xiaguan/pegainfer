//! Qwen3 model: weights, forward pass, generation.

use anyhow::Result;
use fastrace::local::LocalSpan;
use log::info;
use safetensors::SafeTensors;
use std::fs;
use std::time::Instant;

use rand::rngs::StdRng;

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

impl Qwen3Model {
    pub fn from_safetensors(model_path: &str) -> Result<Self> {
        Self::from_safetensors_with_runtime(model_path, ModelRuntimeConfig::default())
    }

    pub fn from_safetensors_with_runtime(
        model_path: &str,
        runtime: ModelRuntimeConfig,
    ) -> Result<Self> {
        info!("Loading model from: {}", model_path);
        info!("Initializing GPU");
        let ctx = DeviceContext::new()?;

        let config = Config::from_file(model_path)?;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        info!("Loading {} safetensor shard(s)", shard_paths.len());
        let shard_data: Vec<Vec<u8>> = shard_paths
            .iter()
            .map(|p| {
                info!("Reading shard: {}", p);
                fs::read(p)
            })
            .collect::<std::io::Result<_>>()?;
        let shards: Vec<SafeTensors> = shard_data
            .iter()
            .map(|d| {
                SafeTensors::deserialize(d).map_err(|e| anyhow::anyhow!("Deserialize error: {}", e))
            })
            .collect::<Result<_>>()?;

        info!("Loading embeddings to GPU");
        let embed_tokens = load_tensor_2d(&ctx, &shards, &weight_map, "model.embed_tokens.weight")?;

        info!(
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

        info!("Precomputing RoPE cache on GPU");
        let (cos_cache, sin_cache) =
            precompute_rope(&ctx, config.head_dim, 4096, config.rope_theta)?;

        ctx.sync()?;
        info!("GPU model loaded successfully");
        if runtime.enable_cuda_graph {
            info!("Decode path CUDA Graph is enabled");
        } else {
            info!("Decode path CUDA Graph is disabled");
        }

        Ok(Self {
            ctx,
            config,
            embed_tokens,
            layers,
            norm,
            cos_cache,
            sin_cache,
            decode_bufs: None,
            kv_cache: None,
            graph_state: CudaGraphState { graph: None },
            enable_cuda_graph: runtime.enable_cuda_graph,
        })
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
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer_batch(layer_idx, layer, &hidden, start_pos, kv_cache)?;
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
        ops::linear(&self.ctx, &normed, &self.embed_tokens)
    }

    fn forward_layer_batch(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<HiddenStates> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let seq_len = hidden.seq_len;

        kv_cache.init_if_needed(&self.ctx, head_dim)?;

        // 1. RMSNorm (batched)
        let normed = ops::rms_norm_batch(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
        )?;

        // 2. QKV projection (GEMM — reads each weight matrix once for all tokens)
        let q_batch = ops::gemm(&self.ctx, &layer.attention.q_proj, &normed)?;
        let k_batch = ops::gemm(&self.ctx, &layer.attention.k_proj, &normed)?;
        let v_batch = ops::gemm(&self.ctx, &layer.attention.v_proj, &normed)?;

        // 3. Attention (per-token loop — attention is <1% of FLOPs for short sequences)
        let q_dim = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_batch = HiddenStates::zeros(&self.ctx, q_dim, seq_len)?;

        for i in 0..seq_len {
            let pos = start_pos + i;
            let current_seq_len = pos + 1;

            let q_i = ops::extract_vec(&self.ctx, &q_batch, i)?;
            let k_i = ops::extract_vec(&self.ctx, &k_batch, i)?;
            let v_i = ops::extract_vec(&self.ctx, &v_batch, i)?;

            let (k_cache_layer, v_cache_layer) = kv_cache.get_cache_mut(&self.ctx, layer_idx)?;

            let cos_pos = self.cos_cache.view(pos * head_dim, head_dim);
            let sin_pos = self.sin_cache.view(pos * head_dim, head_dim);

            let attn_out = ops::fused_attention(
                &self.ctx,
                &q_i,
                &k_i,
                &v_i,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &cos_pos,
                &sin_pos,
                k_cache_layer,
                v_cache_layer,
                num_heads,
                num_kv_heads,
                head_dim,
                pos,
                current_seq_len,
                scale,
                self.config.rms_norm_eps,
            )?;

            ops::write_vec(&self.ctx, &mut attn_batch, i, &attn_out)?;
        }

        // 4. O projection (GEMM)
        let o_batch = ops::gemm(&self.ctx, &layer.attention.o_proj, &attn_batch)?;

        // 5. Residual add
        let hidden = ops::add_batch(&self.ctx, hidden, &o_batch)?;

        // 6. MLP RMSNorm (batched)
        let normed2 = ops::rms_norm_batch(
            &self.ctx,
            &hidden,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
        )?;

        // 7. MLP (decomposed into 3 GEMMs + element-wise SiLU·mul)
        let gate_out = ops::gemm(&self.ctx, &layer.mlp.gate_proj, &normed2)?;
        let up_out = ops::gemm(&self.ctx, &layer.mlp.up_proj, &normed2)?;
        let act_out = ops::silu_mul_batch(&self.ctx, &gate_out, &up_out)?;
        let mlp_out = ops::gemm(&self.ctx, &layer.mlp.down_proj, &act_out)?;

        // 8. Residual add
        let hidden = ops::add_batch(&self.ctx, &hidden, &mlp_out)?;

        Ok(hidden)
    }

    fn select_token(
        &self,
        logits: &DeviceVec,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        if params.is_greedy() {
            ops::argmax(&self.ctx, logits)
        } else {
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
                info!("Capturing CUDA Graph for decode path...");
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
                info!("CUDA Graph captured successfully");
            }
        }

        kv_cache.increment_seq_len();
        Ok(())
    }

    /// Pure kernel sequence for decode — no CPU-GPU sync, no allocation.
    /// Called during graph capture and also replayed via CUDA Graph.
    fn decode_kernels(&self, kv_cache: &mut KVCache, bufs: &mut DecodeBuffers) -> Result<()> {
        // 1. Embedding (reads token_id from decode_meta[0])
        ops::embedding_decode_into(
            &self.ctx,
            &self.embed_tokens,
            &bufs.decode_meta,
            &mut bufs.hidden,
        )?;

        // 2. All transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.decode_layer(layer_idx, layer, kv_cache, bufs)?;
        }

        // 3. Final norm
        ops::rms_norm_into(
            &self.ctx,
            &bufs.hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        // 4. LM Head
        ops::gemv(
            &self.ctx,
            &self.embed_tokens,
            &bufs.normed,
            &mut bufs.logits,
        )?;

        Ok(())
    }

    fn decode_layer(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        kv_cache: &mut KVCache,
        bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        let scale = 1.0 / (self.config.head_dim as f32).sqrt();
        let eps = self.config.rms_norm_eps;

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        // Input RMSNorm: hidden → normed
        ops::rms_norm_into(
            &self.ctx,
            &bufs.hidden,
            &layer.input_layernorm,
            eps,
            &mut bufs.normed,
        )?;

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

        // Fused Attention (decode variant): reads pos/seq_len from decode_meta
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
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            scale,
            eps,
        )?;

        // O projection: attn_out → attn_proj
        ops::gemv(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.attn_proj,
        )?;

        // Residual: hidden += attn_proj
        ops::add_inplace(&self.ctx, &mut bufs.hidden, &bufs.attn_proj)?;

        // Post-attention RMSNorm: hidden → normed
        ops::rms_norm_into(
            &self.ctx,
            &bufs.hidden,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        // Fused MLP: normed → mlp_out
        ops::fused_mlp_into(
            &self.ctx,
            &bufs.normed,
            &layer.mlp.gate_proj,
            &layer.mlp.up_proj,
            &layer.mlp.down_proj,
            &mut bufs.mlp_act,
            &mut bufs.mlp_out,
        )?;

        // Residual: hidden += mlp_out
        ops::add_inplace(&self.ctx, &mut bufs.hidden, &bufs.mlp_out)?;

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

        let next_token = {
            let _span = LocalSpan::enter_with_local_parent("prefill")
                .with_property(|| ("prompt_tokens", prompt_tokens.len().to_string()));
            let start_pos = kv_cache.len();
            let hidden = self.get_embeddings_batch(&tokens)?;
            let hidden = self.process_all_layers_batch(hidden, start_pos, &mut kv_cache)?;
            let logits = self.compute_logits_batch(&hidden)?;
            self.select_token(&logits, params, rng)?
        };

        let ttft = ttft_start.elapsed();
        tokens.push(next_token);

        LocalSpan::add_property(|| ("ttft_ms", format!("{:.2}", ttft.as_secs_f64() * 1000.0)));

        info!(
            "TTFT: {:.2}ms (prompt_len={})",
            ttft.as_secs_f64() * 1000.0,
            prompt_tokens.len()
        );

        // Take persistent decode state from self (avoids borrow conflicts)
        let mut bufs = self.take_decode_bufs()?;
        let mut graph_state = self.take_graph_state();

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
                self.select_token(&bufs.logits, params, rng)?
            };

            if next_token == self.config.eos_token_id {
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
            info!(
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
        let _span =
            LocalSpan::enter_with_local_parent("generate_streaming").with_properties(|| {
                [
                    ("prompt_len", prompt_tokens.len().to_string()),
                    ("max_new_tokens", max_new_tokens.to_string()),
                ]
            });

        let mut tokens = prompt_tokens.to_vec();
        let mut kv_cache = self.take_kv_cache();

        // Prefill
        let ttft_start = Instant::now();
        let next_token = {
            let _span = LocalSpan::enter_with_local_parent("prefill")
                .with_property(|| ("prompt_tokens", prompt_tokens.len().to_string()));
            let start_pos = kv_cache.len();
            let hidden = self.get_embeddings_batch(&tokens)?;
            let hidden = self.process_all_layers_batch(hidden, start_pos, &mut kv_cache)?;
            let logits = self.compute_logits_batch(&hidden)?;
            self.select_token(&logits, params, rng)?
        };

        let ttft = ttft_start.elapsed();
        info!(
            "TTFT: {:.2}ms (prompt_len={})",
            ttft.as_secs_f64() * 1000.0,
            prompt_tokens.len()
        );

        tokens.push(next_token);
        let mut emitted_tokens = 1usize;
        if !on_token(next_token) {
            self.kv_cache = Some(kv_cache);
            return Ok(StreamingStats {
                emitted_tokens,
                hit_eos: false,
                consumer_dropped: true,
            });
        }

        // Take persistent decode state from self
        let mut bufs = self.take_decode_bufs()?;
        let mut graph_state = self.take_graph_state();

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
                self.select_token(&bufs.logits, params, rng)?
            };

            if next_token == self.config.eos_token_id {
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
            info!(
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
