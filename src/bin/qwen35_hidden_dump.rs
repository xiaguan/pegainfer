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
    out: Option<PathBuf>,
}

#[derive(Serialize)]
struct HiddenDump {
    prompt_len: usize,
    layer_types: Vec<String>,
    embedding_last: Vec<f32>,
    layers_last: Vec<Vec<f32>>,
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
    let (embedding_last, layers_last, layer_types) =
        model.debug_prefill_last_hidden_by_layer(&tokens[..cli.prompt_len])?;

    let dump = HiddenDump {
        prompt_len: cli.prompt_len,
        layer_types,
        embedding_last,
        layers_last,
    };

    let json = serde_json::to_string_pretty(&dump)?;
    if let Some(out) = cli.out {
        fs::write(&out, json).with_context(|| format!("failed to write {}", out.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}
