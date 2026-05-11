use std::{path::Path, sync::mpsc as std_mpsc, thread};

use anyhow::{Context, Result, bail};
use log::{info, warn};
use pegainfer_core::engine::{
    EngineHandle, EngineLoadOptions, FinishReason, GenerateRequest, TokenEvent,
};
use tokio::sync::mpsc;

use super::worker::{
    FullDirectRuntime, ensure_direct_decode_caches, load_full_direct_runtime,
    run_direct_decode_logits, run_prefill_logits_and_seed_decode_cache,
};
use crate::Config;

pub struct DirectGeneration {
    pub generated: Vec<u32>,
    pub finish_reason: FinishReason,
}

pub struct DeepSeekV4DirectGenerator {
    config: &'static Config,
    runtime: FullDirectRuntime,
}

impl DeepSeekV4DirectGenerator {
    pub fn from_model_dir(model_path: &Path) -> Result<Self> {
        let config = Box::leak(Box::new(Config::from_model_dir(model_path).with_context(
            || {
                format!(
                    "failed to load DeepSeek V4 config from {}",
                    model_path.display()
                )
            },
        )?));
        let runtime = load_full_direct_runtime(model_path, config)?;
        Ok(Self { config, runtime })
    }

    pub fn eos_token_id(&self) -> usize {
        self.config.eos_token_id
    }

    pub fn generate_greedy<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
        mut on_token: F,
    ) -> Result<DirectGeneration>
    where
        F: FnMut(u32) -> Result<()>,
    {
        if prompt_tokens.is_empty() {
            bail!("DeepSeek V4 request produced an empty prompt");
        }
        if max_new_tokens == 0 {
            return Ok(DirectGeneration {
                generated: Vec::new(),
                finish_reason: FinishReason::Length,
            });
        }

        ensure_direct_decode_caches(
            &mut self.runtime,
            self.config,
            prompt_tokens.len() + max_new_tokens,
        )?;

        let mut next_logits = run_prefill_logits_and_seed_decode_cache(
            &mut self.runtime,
            self.config,
            prompt_tokens,
        )?;
        let mut generated = Vec::with_capacity(max_new_tokens);

        for step in 0..max_new_tokens {
            let token = argmax_f32(&next_logits) as u32;
            if !ignore_eos && token as usize == self.config.eos_token_id {
                return Ok(DirectGeneration {
                    generated,
                    finish_reason: FinishReason::Stop,
                });
            }
            on_token(token)?;
            generated.push(token);
            if step + 1 == max_new_tokens {
                break;
            }
            next_logits =
                run_direct_decode_logits(&mut self.runtime, token, prompt_tokens.len() + step)?;
        }

        Ok(DirectGeneration {
            generated,
            finish_reason: FinishReason::Length,
        })
    }
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    if options.device_ordinals != (0..8).collect::<Vec<_>>() {
        bail!(
            "DeepSeek V4 MP8 currently requires device_ordinals=0..7, got {:?}",
            options.device_ordinals
        );
    }
    if options.enable_cuda_graph {
        warn!("DeepSeek V4 direct engine does not use CUDA graph yet");
    }
    let model_path = model_path.to_path_buf();
    let (submit_tx, mut submit_rx) = mpsc::unbounded_channel::<GenerateRequest>();
    let (init_tx, init_rx) = std_mpsc::channel::<Result<()>>();
    thread::Builder::new()
        .name("deepseek-v4-scheduler".into())
        .spawn(move || {
            let mut generator = match DeepSeekV4DirectGenerator::from_model_dir(&model_path) {
                Ok(generator) => {
                    let _ = init_tx.send(Ok(()));
                    generator
                }
                Err(err) => {
                    let _ = init_tx.send(Err(err));
                    return;
                }
            };
            info!("DeepSeek V4 scheduler ready");
            while let Some(req) = submit_rx.blocking_recv() {
                handle_request(&mut generator, req);
            }
            info!("DeepSeek V4 scheduler exiting");
        })
        .expect("failed to spawn DeepSeek V4 scheduler thread");
    init_rx
        .recv()
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 engine init channel closed: {err}"))??;
    Ok(EngineHandle::new(submit_tx))
}

fn handle_request(generator: &mut DeepSeekV4DirectGenerator, req: GenerateRequest) {
    let prompt_len = req.prompt_tokens.len();
    if req.echo {
        let _ = req.token_tx.send(TokenEvent::PromptTokens {
            ids: req.prompt_tokens.clone(),
            logprobs: vec![None; prompt_len],
        });
    }
    if req.params.temperature > 0.0 || req.params.top_k != -1 || req.params.top_p < 1.0 {
        reject_request(
            &req,
            prompt_len,
            format!(
                "DeepSeek V4 direct engine currently serves greedy decoding only; requested temperature={}, top_k={}, top_p={}",
                req.params.temperature, req.params.top_k, req.params.top_p
            ),
        );
        return;
    }
    if req.logprobs > 0 {
        reject_request(
            &req,
            prompt_len,
            "DeepSeek V4 direct engine does not return logprobs yet".to_string(),
        );
        return;
    }

    let token_tx = req.token_tx.clone();
    let result = generator.generate_greedy(
        &req.prompt_tokens,
        req.max_tokens,
        req.params.ignore_eos,
        |token| {
            token_tx
                .send(TokenEvent::Token {
                    id: token,
                    logprob: None,
                })
                .map_err(|_| anyhow::anyhow!("request receiver dropped"))?;
            Ok(())
        },
    );
    match result {
        Ok(generation) => {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: generation.finish_reason,
                prompt_tokens: prompt_len,
                completion_tokens: generation.generated.len(),
            });
        }
        Err(err) => {
            let message = format!("DeepSeek V4 direct request failed: {err:#}");
            warn!("{message}");
            let _ = req.token_tx.send(TokenEvent::Error {
                message,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
        }
    }
}

fn reject_request(req: &GenerateRequest, prompt_len: usize, reason: String) {
    warn!("{reason}");
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message: reason,
        prompt_tokens: prompt_len,
        completion_tokens: 0,
    });
}

fn argmax_f32(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best {
            best = value;
            best_idx = idx;
        }
    }
    best_idx
}
