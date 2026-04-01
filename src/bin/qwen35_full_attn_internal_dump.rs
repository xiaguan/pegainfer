use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use pegainfer::model::Qwen35Model;
use serde::Serialize;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long)]
    model_path: String,
    #[arg(long)]
    tokens_file: PathBuf,
    #[arg(long)]
    prompt_len: usize,
    #[arg(long)]
    layer_idx: usize,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Serialize)]
struct InternalDump {
    prompt_len: usize,
    layer_idx: usize,
    input_layernorm_last: Vec<f32>,
    q_full_last: Vec<f32>,
    k_proj_last: Vec<f32>,
    v_proj_last: Vec<f32>,
    q_prepped_last: Vec<f32>,
    attn_pre_gate_last: Vec<f32>,
    gated_attn_last: Vec<f32>,
    o_proj_last: Vec<f32>,
}

fn load_tokens(path: &PathBuf) -> Result<Vec<u32>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read tokens file {}", path.display()))?;
    if let Ok(ids) = serde_json::from_str::<Vec<u32>>(&text) {
        return Ok(ids);
    }
    #[derive(serde::Deserialize)]
    struct Wrapper {
        token_ids: Vec<u32>,
    }
    let wrapper: Wrapper =
        serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(wrapper.token_ids)
}

fn main() -> Result<()> {
    pegainfer::logging::init_stderr("info");
    let cli = Cli::parse();
    let tokens = load_tokens(&cli.tokens_file)?;
    anyhow::ensure!(
        cli.prompt_len > 0 && cli.prompt_len <= tokens.len(),
        "prompt_len {} out of range for {} tokens",
        cli.prompt_len,
        tokens.len()
    );

    let model = Qwen35Model::from_safetensors_with_options(&cli.model_path, true)
        .context("failed to load Qwen3.5 model")?;
    let (
        input_layernorm_last,
        q_full_last,
        k_proj_last,
        v_proj_last,
        q_prepped_last,
        attn_pre_gate_last,
        gated_attn_last,
        o_proj_last,
    ) = model.debug_prefill_full_attention_internal_last_hidden(
        &tokens[..cli.prompt_len],
        cli.layer_idx,
    )?;

    let dump = InternalDump {
        prompt_len: cli.prompt_len,
        layer_idx: cli.layer_idx,
        input_layernorm_last,
        q_full_last,
        k_proj_last,
        v_proj_last,
        q_prepped_last,
        attn_pre_gate_last,
        gated_attn_last,
        o_proj_last,
    };

    let json = serde_json::to_string_pretty(&dump)?;
    if let Some(out) = cli.out {
        fs::write(&out, json).with_context(|| format!("failed to write {}", out.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}
