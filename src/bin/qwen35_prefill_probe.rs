use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use pegainfer::model::Qwen35Model;
use pegainfer::sampler::SamplingParams;
use pegainfer::scheduler::{SchedulerRequest, TokenEvent};
use pegainfer::scheduler_qwen35;
use pegainfer::server_engine::TokenLogprob;
use pegainfer::tokenizer::Tokenizer;
use serde::Serialize;
use tokio::sync::mpsc;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long)]
    model_path: String,
    #[arg(long)]
    tokens_file: PathBuf,
    #[arg(long, value_delimiter = ',')]
    lengths: Vec<usize>,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Serialize)]
struct ProbeTop {
    id: u32,
    text: String,
    logprob: f32,
}

#[derive(Serialize)]
struct ProbeResult {
    prompt_len: usize,
    last_prompt_token: u32,
    generated_token: u32,
    generated_text: String,
    generated_logprob: Option<f32>,
    top_logprobs: Vec<ProbeTop>,
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
    let wrapper: Wrapper = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(wrapper.token_ids)
}

fn decode_one(tokenizer: &Tokenizer, id: u32) -> String {
    tokenizer
        .decode(&[id])
        .unwrap_or_else(|_| format!("<decode_error:{id}>"))
}

fn top_entries(tokenizer: &Tokenizer, lp: Option<TokenLogprob>) -> (Option<f32>, Vec<ProbeTop>) {
    match lp {
        Some(lp) => {
            let top = lp
                .top_logprobs
                .into_iter()
                .map(|(id, logprob)| ProbeTop {
                    id,
                    text: decode_one(tokenizer, id),
                    logprob,
                })
                .collect();
            (Some(lp.logprob), top)
        }
        None => (None, Vec::new()),
    }
}

fn main() -> Result<()> {
    pegainfer::logging::init_stderr("info");
    let cli = Cli::parse();
    let tokens = load_tokens(&cli.tokens_file)?;
    anyhow::ensure!(!cli.lengths.is_empty(), "at least one prompt length is required");

    let model = Qwen35Model::from_safetensors_with_options(&cli.model_path, true)
        .context("failed to load Qwen3.5 model")?;
    let tokenizer = Tokenizer::from_file(&cli.model_path).context("failed to load tokenizer")?;
    let handle = scheduler_qwen35::start_with_capacity(model, 42, 1)
        .context("failed to start Qwen3.5 scheduler")?;

    let mut results = Vec::with_capacity(cli.lengths.len());
    for &prompt_len in &cli.lengths {
        anyhow::ensure!(
            prompt_len > 0 && prompt_len <= tokens.len(),
            "prompt_len {} out of range for {} tokens",
            prompt_len,
            tokens.len()
        );
        let prompt_tokens = tokens[..prompt_len].to_vec();
        let last_prompt_token = *prompt_tokens.last().unwrap();
        let (token_tx, mut token_rx) = mpsc::unbounded_channel();
        handle
            .submit(SchedulerRequest {
                prompt_tokens,
                params: SamplingParams::default(),
                max_tokens: 1,
                token_tx,
                logprobs: cli.top_k,
                echo: false,
            })
            .context("failed to submit scheduler request")?;

        let mut generated_token = None;
        let mut generated_lp = None;
        loop {
            match token_rx.blocking_recv() {
                Some(TokenEvent::Token { id, logprob }) => {
                    generated_token = Some(id);
                    generated_lp = Some(logprob);
                }
                Some(TokenEvent::PromptTokens { .. }) => {}
                Some(TokenEvent::Finished { .. }) => break,
                None => anyhow::bail!("scheduler channel closed before Finished"),
            }
        }

        let generated_token = generated_token.context("no generated token received")?;
        let (generated_logprob, top_logprobs) = top_entries(&tokenizer, generated_lp.flatten());
        results.push(ProbeResult {
            prompt_len,
            last_prompt_token,
            generated_token,
            generated_text: decode_one(&tokenizer, generated_token),
            generated_logprob,
            top_logprobs,
        });
    }

    let json = serde_json::to_string_pretty(&results)?;
    if let Some(out) = cli.out {
        fs::write(&out, json).with_context(|| format!("failed to write {}", out.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}
