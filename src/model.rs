//! Qwen3 model: weights, forward pass, generation.

use anyhow::Result;
use fastrace::local::LocalSpan;
use log::{debug, info};
use safetensors::SafeTensors;
use std::fs;
use std::time::Instant;

use rand::rngs::StdRng;

use crate::kv_cache::KVCache;
use crate::ops;
use crate::qwen3_config::Config;
use crate::sampler::{self, SamplingParams};
use crate::tensor::*;
use crate::weight_loader::*;

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
    // RoPE cache on GPU - precomputed for all positions
    pub cos_cache: Vec<DeviceVec>, // [position] -> (head_dim,)
    pub sin_cache: Vec<DeviceVec>,
}

/// Streaming generation summary for transport layers.
pub struct StreamingStats {
    pub emitted_tokens: usize,
    pub hit_eos: bool,
    pub consumer_dropped: bool,
}

impl Qwen3Model {
    pub fn from_safetensors(model_path: &str) -> Result<Self> {
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

        Ok(Self {
            ctx,
            config,
            embed_tokens,
            layers,
            norm,
            cos_cache,
            sin_cache,
        })
    }

    /// Forward pass for a single token
    #[fastrace::trace(name = "forward")]
    pub fn forward(&self, token_ids: &[u32], kv_cache: &mut KVCache) -> Result<u32> {
        LocalSpan::add_property(|| ("num_tokens", token_ids.len().to_string()));
        debug!("forward: num_tokens={}", token_ids.len());
        let start_pos = kv_cache.len();

        let mut hidden_states = self.get_embeddings(token_ids)?;
        hidden_states = self.process_all_layers(hidden_states, start_pos, kv_cache)?;
        let next_token = self.predict_next_token(&hidden_states)?;

        Ok(next_token)
    }

    /// Forward pass returning final logits (for accuracy testing)
    ///
    /// Similar to `forward()`, but returns the full logits vector instead of argmax.
    /// Used for numerical validation against reference implementations.
    pub fn forward_logits(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let mut kv_cache = KVCache::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
        );
        let start_pos = kv_cache.len();

        let mut hidden_states = self.get_embeddings(token_ids)?;
        hidden_states = self.process_all_layers(hidden_states, start_pos, &mut kv_cache)?;

        // Final norm - use LAST hidden state for prediction
        let last_hidden = ops::rms_norm(
            &self.ctx,
            hidden_states.last().unwrap(),
            &self.norm,
            self.config.rms_norm_eps,
        )?;

        // LM head: logits = embed_tokens @ hidden (tied weights)
        let logits = ops::linear(&self.ctx, &last_hidden, &self.embed_tokens)?;

        // Copy to host as f32
        logits.to_host(&self.ctx)
    }

    #[fastrace::trace(name = "get_embeddings")]
    fn get_embeddings(&self, token_ids: &[u32]) -> Result<Vec<DeviceVec>> {
        debug!("get_embeddings: num_tokens={}", token_ids.len());
        token_ids
            .iter()
            .map(|&id| ops::embedding(&self.ctx, &self.embed_tokens, id))
            .collect::<Result<Vec<_>>>()
    }

    #[fastrace::trace(name = "process_all_layers")]
    fn process_all_layers(
        &self,
        mut hidden_states: Vec<DeviceVec>,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Vec<DeviceVec>> {
        let num_tokens = hidden_states.len();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states =
                self.forward_layer(layer_idx, layer, &hidden_states, start_pos, kv_cache)?;
        }

        // Increment sequence length AFTER all layers processed
        for _ in 0..num_tokens {
            kv_cache.increment_seq_len();
        }

        Ok(hidden_states)
    }

    #[fastrace::trace(name = "predict_next_token")]
    fn predict_next_token(&self, hidden_states: &[DeviceVec]) -> Result<u32> {
        let logits = self.compute_logits(hidden_states)?;
        ops::argmax(&self.ctx, &logits)
    }

    fn compute_logits(&self, hidden_states: &[DeviceVec]) -> Result<DeviceVec> {
        let last_hidden = ops::rms_norm(
            &self.ctx,
            hidden_states.last().unwrap(),
            &self.norm,
            self.config.rms_norm_eps,
        )?;
        ops::linear(&self.ctx, &last_hidden, &self.embed_tokens)
    }

    fn forward_layer(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden_states: &[DeviceVec],
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Vec<DeviceVec>> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        // Initialize cache on first use
        kv_cache.init_if_needed(&self.ctx, head_dim)?;

        let mut outputs = Vec::with_capacity(hidden_states.len());

        for (i, h) in hidden_states.iter().enumerate() {
            let pos = start_pos + i;

            let normed = ops::rms_norm(
                &self.ctx,
                h,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
            )?;

            let (q, k, v) = ops::linear_qkv_batched(
                &self.ctx,
                &normed,
                &layer.attention.q_proj,
                &layer.attention.k_proj,
                &layer.attention.v_proj,
            )?;

            // Use pos (start_pos + i) for current position, not kv_cache.seq_len()
            // This is critical for prefill phase where we process multiple tokens
            let current_pos = pos;
            let seq_len = pos + 1;
            let scale = 1.0 / (head_dim as f32).sqrt();

            let (k_cache_layer, v_cache_layer) = kv_cache.get_cache_mut(&self.ctx, layer_idx)?;

            let attn_concat = ops::fused_attention(
                &self.ctx,
                &q,
                &k,
                &v,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &self.cos_cache[pos],
                &self.sin_cache[pos],
                k_cache_layer,
                v_cache_layer,
                num_heads,
                num_kv_heads,
                head_dim,
                current_pos,
                seq_len,
                scale,
                self.config.rms_norm_eps,
            )?;

            let attn_output = ops::linear(&self.ctx, &attn_concat, &layer.attention.o_proj)?;
            let h = ops::add(&self.ctx, h, &attn_output)?;

            let normed = ops::rms_norm(
                &self.ctx,
                &h,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;

            // MLP: fully fused (gate_proj @ x + up_proj @ x + SiLU + down_proj)
            let mlp_out = ops::fused_mlp(
                &self.ctx,
                &normed,
                &layer.mlp.gate_proj,
                &layer.mlp.up_proj,
                &layer.mlp.down_proj,
            )?;
            let h = ops::add(&self.ctx, &h, &mlp_out)?;

            outputs.push(h);
        }

        Ok(outputs)
    }

    // ============================================================
    // Batched forward path (prefill optimization)
    // ============================================================

    /// Batched forward pass: process all tokens with GEMM instead of per-token GEMV.
    /// Returns the predicted next token (argmax of last position's logits).
    #[fastrace::trace(name = "forward_batch")]
    pub fn forward_batch(&self, token_ids: &[u32], kv_cache: &mut KVCache) -> Result<u32> {
        LocalSpan::add_property(|| ("num_tokens", token_ids.len().to_string()));
        debug!("forward_batch: num_tokens={}", token_ids.len());
        let start_pos = kv_cache.len();

        let hidden = self.get_embeddings_batch(token_ids)?;
        let hidden = self.process_all_layers_batch(hidden, start_pos, kv_cache)?;
        let next_token = self.predict_next_token_batch(&hidden)?;

        Ok(next_token)
    }

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

    #[fastrace::trace(name = "predict_next_token_batch")]
    fn predict_next_token_batch(&self, hidden: &HiddenStates) -> Result<u32> {
        let logits = self.compute_logits_batch(hidden)?;
        ops::argmax(&self.ctx, &logits)
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

            let attn_out = ops::fused_attention(
                &self.ctx,
                &q_i,
                &k_i,
                &v_i,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &self.cos_cache[pos],
                &self.sin_cache[pos],
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

    /// Generate tokens
    pub fn generate(
        &self,
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
        let mut kv_cache = KVCache::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
        );

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

        // Generate new tokens and measure TPOT
        let tpot_start = Instant::now();
        let mut generated_count = 0;
        for i in 1..max_new_tokens {
            let next_token = {
                let _span = LocalSpan::enter_with_local_parent("decode_step")
                    .with_property(|| ("step", i.to_string()));
                let start_pos = kv_cache.len();
                let mut hidden_states = self.get_embeddings(&[*tokens.last().unwrap()])?;
                hidden_states = self.process_all_layers(hidden_states, start_pos, &mut kv_cache)?;
                let logits = self.compute_logits(&hidden_states)?;
                self.select_token(&logits, params, rng)?
            };

            if next_token == self.config.eos_token_id {
                break;
            }

            tokens.push(next_token);
            generated_count += 1;
        }

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
        &self,
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
        let mut kv_cache = KVCache::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
        );

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
            return Ok(StreamingStats {
                emitted_tokens,
                hit_eos: false,
                consumer_dropped: true,
            });
        }

        // Decode
        let tpot_start = Instant::now();
        let mut generated_count = 0;
        let mut hit_eos = false;
        for i in 1..max_new_tokens {
            let next_token = {
                let _span = LocalSpan::enter_with_local_parent("decode_step")
                    .with_property(|| ("step", i.to_string()));
                let start_pos = kv_cache.len();
                let mut hidden_states = self.get_embeddings(&[*tokens.last().unwrap()])?;
                hidden_states = self.process_all_layers(hidden_states, start_pos, &mut kv_cache)?;
                let logits = self.compute_logits(&hidden_states)?;
                self.select_token(&logits, params, rng)?
            };

            if next_token == self.config.eos_token_id {
                hit_eos = true;
                break;
            }

            tokens.push(next_token);
            generated_count += 1;
            emitted_tokens += 1;

            if !on_token(next_token) {
                return Ok(StreamingStats {
                    emitted_tokens,
                    hit_eos: false,
                    consumer_dropped: true,
                });
            }
        }

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
        &self,
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
