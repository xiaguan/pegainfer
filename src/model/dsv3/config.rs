use anyhow::Result;
use serde::Deserialize;
use std::fs;

/// Layer type: first `first_k_dense_replace` layers use dense FFN, the rest use MoE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FfnType {
    Dense,
    MoE,
}

/// FP8 block-wise quantization configuration.
#[derive(Debug, Deserialize)]
pub(crate) struct QuantizationConfig {
    /// "dynamic" activation quantization
    pub(crate) activation_scheme: String,
    /// "e4m3" float format
    pub(crate) fmt: String,
    /// "fp8"
    pub(crate) quant_method: String,
    /// Weight block size for block-wise quantization, e.g. [128, 128]
    pub(crate) weight_block_size: [usize; 2],
}

/// YaRN RoPE scaling configuration.
#[derive(Debug, Deserialize)]
struct RopeScaling {
    beta_fast: f64,
    beta_slow: f64,
    factor: f64,
    mscale: f64,
    mscale_all_dim: f64,
    original_max_position_embeddings: usize,
    #[serde(rename = "type")]
    scaling_type: String,
}

/// Raw config.json deserialization target.
#[derive(Debug, Deserialize)]
struct RawConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    bos_token_id: u32,
    eos_token_id: u32,
    tie_word_embeddings: bool,

    // MLA parameters
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,

    // MoE parameters
    first_k_dense_replace: usize,
    moe_layer_freq: usize,
    n_group: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    num_experts_per_tok: usize,
    scoring_func: String,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
    topk_group: usize,
    topk_method: String,
    moe_intermediate_size: usize,

    // Indexer (NSA) parameters
    #[serde(default)]
    index_head_dim: Option<usize>,
    #[serde(default)]
    index_n_heads: Option<usize>,
    #[serde(default)]
    index_topk: Option<usize>,

    // MTP
    #[serde(default)]
    num_nextn_predict_layers: Option<usize>,

    // Quantization
    #[serde(default)]
    quantization_config: Option<QuantizationConfig>,

    // RoPE
    #[serde(default)]
    rope_scaling: Option<RopeScaling>,
    max_position_embeddings: usize,
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    eos_token_id: EosTokenIds,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosTokenIds {
    Single(u32),
    Multiple(Vec<u32>),
}

/// DeepSeek-V3.2 model configuration.
#[derive(Debug)]
pub(crate) struct DsV3Config {
    // === Dimensions ===
    pub(crate) hidden_size: usize,
    /// Dense FFN intermediate size (layers 0..first_k_dense_replace)
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f32,

    // === MLA ===
    pub(crate) q_lora_rank: usize,
    pub(crate) kv_lora_rank: usize,
    pub(crate) qk_nope_head_dim: usize,
    pub(crate) qk_rope_head_dim: usize,
    pub(crate) v_head_dim: usize,

    // === MoE ===
    /// Number of leading dense FFN layers (typically 3)
    pub(crate) first_k_dense_replace: usize,
    pub(crate) n_routed_experts: usize,
    pub(crate) n_shared_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) n_group: usize,
    pub(crate) topk_group: usize,
    pub(crate) norm_topk_prob: bool,
    pub(crate) routed_scaling_factor: f32,
    /// Per-expert FFN intermediate size
    pub(crate) moe_intermediate_size: usize,

    // === RoPE ===
    pub(crate) rope_theta: f32,
    pub(crate) max_position_embeddings: usize,
    /// YaRN attention scaling factor applied to softmax scale
    pub(crate) softmax_mscale: f32,

    // === YaRN parameters ===
    pub(crate) yarn_beta_fast: f32,
    pub(crate) yarn_beta_slow: f32,
    pub(crate) yarn_factor: f32,
    pub(crate) yarn_original_max_position_embeddings: usize,

    // === FP8 ===
    /// Weight block size for dequantization, e.g. [128, 128]
    pub(crate) weight_block_size: [usize; 2],

    // === Token IDs ===
    pub(crate) bos_token_id: u32,
    pub(crate) stop_token_ids: Vec<u32>,
    pub(crate) tie_word_embeddings: bool,

    // === Layer layout ===
    pub(crate) layer_types: Vec<FfnType>,
}

impl DsV3Config {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let raw: RawConfig = serde_json::from_str(&content)?;

        anyhow::ensure!(
            raw.scoring_func == "sigmoid",
            "Only sigmoid scoring is supported, got: {}",
            raw.scoring_func
        );

        // Build layer type vector
        let layer_types: Vec<FfnType> = (0..raw.num_hidden_layers)
            .map(|i| {
                if i < raw.first_k_dense_replace {
                    FfnType::Dense
                } else {
                    FfnType::MoE
                }
            })
            .collect();

        // YaRN parameters and attention scaling factor
        let (
            softmax_mscale,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_factor,
            yarn_original_max_position_embeddings,
        ) = match &raw.rope_scaling {
            Some(rs) => {
                let mscale = yarn_get_mscale(rs.factor, rs.mscale);
                let mscale_all_dim = yarn_get_mscale(rs.factor, rs.mscale_all_dim);
                let attention_factor = if mscale_all_dim > 0.0 {
                    mscale / mscale_all_dim
                } else {
                    mscale
                };
                (
                    attention_factor,
                    rs.beta_fast as f32,
                    rs.beta_slow as f32,
                    rs.factor as f32,
                    rs.original_max_position_embeddings,
                )
            }
            None => (1.0, 32.0, 1.0, 1.0, raw.max_position_embeddings),
        };

        let weight_block_size = raw
            .quantization_config
            .as_ref()
            .map(|q| q.weight_block_size)
            .unwrap_or([128, 128]);

        let stop_token_ids = Self::load_stop_token_ids(model_path, raw.eos_token_id)?;

        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            vocab_size: raw.vocab_size,
            rms_norm_eps: raw.rms_norm_eps as f32,

            q_lora_rank: raw.q_lora_rank,
            kv_lora_rank: raw.kv_lora_rank,
            qk_nope_head_dim: raw.qk_nope_head_dim,
            qk_rope_head_dim: raw.qk_rope_head_dim,
            v_head_dim: raw.v_head_dim,

            first_k_dense_replace: raw.first_k_dense_replace,
            n_routed_experts: raw.n_routed_experts,
            n_shared_experts: raw.n_shared_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            n_group: raw.n_group,
            topk_group: raw.topk_group,
            norm_topk_prob: raw.norm_topk_prob,
            routed_scaling_factor: raw.routed_scaling_factor as f32,
            moe_intermediate_size: raw.moe_intermediate_size,

            rope_theta: raw.rope_theta as f32,
            max_position_embeddings: raw.max_position_embeddings,
            softmax_mscale,

            yarn_beta_fast,
            yarn_beta_slow,
            yarn_factor,
            yarn_original_max_position_embeddings,

            weight_block_size,

            bos_token_id: raw.bos_token_id,
            stop_token_ids,
            tie_word_embeddings: raw.tie_word_embeddings,

            layer_types,
        })
    }

    pub(crate) fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    /// Q head dimension = qk_nope_head_dim + qk_rope_head_dim
    pub(crate) fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// KV compressed projection output dim = kv_lora_rank + qk_rope_head_dim
    pub(crate) fn kv_a_proj_dim(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    /// kv_b_proj output per head = qk_nope_head_dim + v_head_dim
    pub(crate) fn kv_b_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.v_head_dim
    }

    /// Total kv_b_proj output dim = num_heads * (qk_nope_head_dim + v_head_dim)
    pub(crate) fn kv_b_proj_dim(&self) -> usize {
        self.num_attention_heads * self.kv_b_head_dim()
    }

    /// Total q_b_proj output dim = num_heads * q_head_dim
    pub(crate) fn q_b_proj_dim(&self) -> usize {
        self.num_attention_heads * self.q_head_dim()
    }

    /// Total output projection input dim = num_heads * v_head_dim
    pub(crate) fn o_proj_input_dim(&self) -> usize {
        self.num_attention_heads * self.v_head_dim
    }

    /// Number of dense FFN layers.
    pub(crate) fn num_dense_layers(&self) -> usize {
        self.first_k_dense_replace
    }

    /// Number of MoE layers.
    pub(crate) fn num_moe_layers(&self) -> usize {
        self.num_hidden_layers - self.first_k_dense_replace
    }

    /// Base softmax scale (before YaRN mscale).
    pub(crate) fn base_softmax_scale(&self) -> f32 {
        (self.q_head_dim() as f32).powf(-0.5) * self.softmax_mscale
    }

    fn load_stop_token_ids(model_path: &str, fallback_eos_token_id: u32) -> Result<Vec<u32>> {
        let generation_config_path = format!("{}/generation_config.json", model_path);
        match fs::read_to_string(&generation_config_path) {
            Ok(content) => {
                let generation_config: GenerationConfig = serde_json::from_str(&content)?;
                let mut stop_token_ids = match generation_config.eos_token_id {
                    EosTokenIds::Single(id) => vec![id],
                    EosTokenIds::Multiple(ids) => ids,
                };
                stop_token_ids.dedup();
                Ok(stop_token_ids)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Ok(vec![fallback_eos_token_id])
            }
            Err(err) => Err(err.into()),
        }
    }
}

/// YaRN attention scaling factor (from transformers modeling_rope_utils).
fn yarn_get_mscale(scale: f64, mscale: f64) -> f32 {
    if scale <= 1.0 {
        return 1.0;
    }
    (0.1 * mscale * scale.ln() + 1.0) as f32
}
