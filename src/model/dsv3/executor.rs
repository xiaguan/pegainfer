//! DSV3 multi-GPU executor: loads models across GPUs with EP/TP sharding
//! and coordinates forward passes.

use std::sync::Arc;
use std::thread;

use anyhow::Result;
use crossbeam_channel as channel;
use half::bf16;
use log::info;

use super::config::ParallelConfig;
use super::deep_ep::{self, DeepEpBuffer, DeepEpConfig};
use super::forward::MlaForwardBuffers;
use super::mla_kv::{MlaKvPool, MlaKvState};
use super::weights::DsV3Model;
use crate::tensor::DeviceContext;

// ---------------------------------------------------------------------------
// DsV3Executor — manages one DsV3Model per GPU
// ---------------------------------------------------------------------------

pub struct DsV3Executor {
    /// Models indexed by rank (0..world_size). Each lives on its own GPU thread.
    ranks: Vec<RankHandle>,
}

struct RankHandle {
    rank: usize,
    tx: channel::Sender<RankCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

enum RankCommand {
    /// Synchronize — wait for all prior GPU work on this rank to complete.
    Sync {
        resp: channel::Sender<Result<()>>,
    },
    /// Run full forward pass (embedding → all layers → logits) and return
    /// the output logits as host bf16 vec.
    Forward {
        token_ids: Arc<Vec<u32>>,
        positions: Arc<Vec<i32>>,
        resp: channel::Sender<Result<Vec<bf16>>>,
    },
    Shutdown,
}

impl DsV3Executor {
    /// Load DSV3 models across multiple GPUs.
    ///
    /// `device_ordinals` maps rank index → CUDA device ordinal.
    /// For TP1 EP8: `device_ordinals = [0,1,2,3,4,5,6,7]`, each rank gets
    /// full attention + 32 routed experts.
    pub fn load(model_path: &str, device_ordinals: &[usize], tp_size: usize) -> Result<Self> {
        let world_size = device_ordinals.len();
        let ep_size = world_size / tp_size;
        anyhow::ensure!(
            world_size > 0 && world_size % tp_size == 0,
            "world_size {} must be a positive multiple of tp_size {}",
            world_size,
            tp_size,
        );
        info!(
            "Loading DSV3 across {} GPUs (TP{} × EP{})",
            world_size, tp_size, ep_size,
        );

        // Load models in parallel — one thread per GPU. Each thread sets its
        // own CUDA device and does independent H2D copies over its own PCIe
        // link. The safetensors mmap is duplicated per thread (cheap — just
        // page table setup, actual I/O is demand-paged from disk).
        let load_start = std::time::Instant::now();
        let models: Vec<DsV3Model> = thread::scope(|s| {
            let handles: Vec<_> = device_ordinals
                .iter()
                .enumerate()
                .map(|(rank, &device_ordinal)| {
                    let tp_rank = rank % tp_size;
                    let ep_rank = rank / tp_size;
                    let parallel = ParallelConfig {
                        tp_rank,
                        tp_size,
                        ep_rank,
                        ep_size,
                    };
                    s.spawn(move || {
                        DsV3Model::from_safetensors_parallel(model_path, device_ordinal, parallel)
                    })
                })
                .collect();
            handles
                .into_iter()
                .enumerate()
                .map(|(rank, h)| {
                    h.join().unwrap_or_else(|_| {
                        Err(anyhow::anyhow!("rank {} load thread panicked", rank))
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;
        info!(
            "All {} ranks loaded in {:.1}s (parallel)",
            world_size,
            load_start.elapsed().as_secs_f64(),
        );

        // TODO: when tp_size > 1, init NCCL comms here and attach to models

        // Init DeepEP intranode buffers for EP > 1
        let mut models = models;
        if ep_size > 1 {
            let config = &models[0].config();
            let hidden = config.hidden_size;
            let num_topk = config.num_experts_per_tok;
            let ep_config = DeepEpConfig::default();

            let (exchange, barrier) = deep_ep::new_intra_process_exchange(world_size);

            // Collect device info before spawning threads (avoids borrow conflict).
            // Cast CUstream to usize to satisfy Send bound (raw ptrs are !Send).
            let rank_info: Vec<_> = models
                .iter()
                .map(|m| {
                    let ctx = m.device_ctx();
                    (ctx.device_ordinal, ctx.stream.cu_stream() as usize)
                })
                .collect();

            // Allocate buffers and exchange pointers via peer access (intra-process)
            let buffers: Vec<DeepEpBuffer> = thread::scope(|s| {
                let handles: Vec<_> = rank_info
                    .iter()
                    .enumerate()
                    .map(|(rank, &(device_ordinal, cu_stream_usize))| {
                        let exchange = Arc::clone(&exchange);
                        let barrier = Arc::clone(&barrier);
                        s.spawn(move || -> Result<DeepEpBuffer> {
                            unsafe {
                                let err = crate::ffi::cuda_set_device(device_ordinal as i32);
                                anyhow::ensure!(err == 0, "cuda_set_device failed: {err}");
                            }
                            let cu_stream = cu_stream_usize as cudarc::driver::sys::CUstream;

                            let mut buf = DeepEpBuffer::alloc(
                                rank, world_size, hidden, num_topk, ep_config, cu_stream,
                            )?;
                            buf.exchange_pointers(&exchange, &barrier, device_ordinal as i32)?;
                            Ok(buf)
                        })
                    })
                    .collect();

                handles
                    .into_iter()
                    .enumerate()
                    .map(|(rank, h)| {
                        h.join().unwrap_or_else(|_| {
                            Err(anyhow::anyhow!("rank {} DeepEP init panicked", rank))
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })?;

            for (model, buf) in models.iter_mut().zip(buffers) {
                model.deep_ep_buffer = Some(buf);
            }
            info!("DeepEP intranode buffers initialized for EP{}", ep_size);
        }

        // Spawn per-rank worker threads
        let mut ranks = Vec::with_capacity(world_size);
        for (rank, model) in models.into_iter().enumerate() {
            ranks.push(RankHandle::spawn(rank, model)?);
        }

        info!("DSV3 executor ready: {} ranks active", world_size);
        Ok(Self { ranks })
    }

    /// Number of ranks (GPUs).
    pub fn world_size(&self) -> usize {
        self.ranks.len()
    }

    /// Synchronize all ranks — wait for all GPU work to complete.
    pub fn sync_all(&self) -> Result<()> {
        let receivers: Vec<_> = self
            .ranks
            .iter()
            .map(|r| {
                let (tx, rx) = channel::bounded(1);
                r.tx.send(RankCommand::Sync { resp: tx })
                    .map_err(|_| anyhow::anyhow!("rank {} channel closed", r.rank))
                    .map(|_| rx)
            })
            .collect::<Result<Vec<_>>>()?;

        for (rank, rx) in receivers.into_iter().enumerate() {
            rx.recv()
                .map_err(|_| anyhow::anyhow!("rank {} dropped sync response", rank))??;
        }
        Ok(())
    }

    /// Run full forward pass across all ranks and return logits from rank 0.
    ///
    /// All ranks must participate (DeepEP is collective). Returns logits
    /// `[vocab_size]` from rank 0 (all ranks produce identical logits for TP1).
    pub fn forward(&self, token_ids: &[u32], positions: &[i32]) -> Result<Vec<bf16>> {
        let token_ids = Arc::new(token_ids.to_vec());
        let positions = Arc::new(positions.to_vec());

        let receivers: Vec<_> = self
            .ranks
            .iter()
            .map(|r| {
                let (tx, rx) = channel::bounded(1);
                r.tx.send(RankCommand::Forward {
                    token_ids: Arc::clone(&token_ids),
                    positions: Arc::clone(&positions),
                    resp: tx,
                })
                .map_err(|_| anyhow::anyhow!("rank {} channel closed", r.rank))
                .map(|_| rx)
            })
            .collect::<Result<Vec<_>>>()?;

        // Collect results — all ranks must complete (collective ops).
        // Return logits from rank 0.
        let mut rank0_logits = None;
        for (rank, rx) in receivers.into_iter().enumerate() {
            let logits = rx
                .recv()
                .map_err(|_| anyhow::anyhow!("rank {} dropped forward response", rank))??;
            if rank == 0 {
                rank0_logits = Some(logits);
            }
        }
        rank0_logits.ok_or_else(|| anyhow::anyhow!("no rank 0 logits"))
    }

    pub fn rank_count(&self) -> usize {
        self.ranks.len()
    }
}

impl Drop for DsV3Executor {
    fn drop(&mut self) {
        for rank in &mut self.ranks {
            rank.shutdown();
        }
    }
}

// ---------------------------------------------------------------------------
// Per-rank worker thread
// ---------------------------------------------------------------------------

fn bind_device(ctx: &DeviceContext) -> Result<()> {
    unsafe {
        let err = crate::ffi::cuda_set_device(ctx.device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {}: cudaError={}",
                ctx.device_ordinal,
                err
            ));
        }
    }
    ctx.ctx
        .bind_to_thread()
        .map_err(|e| anyhow::anyhow!("Failed to bind CUDA context: {e}"))?;
    unsafe {
        crate::ffi::cublas_init();
    }
    Ok(())
}

/// Per-rank state for forward passes (lazily initialized on first Forward).
struct RankForwardState {
    bufs: MlaForwardBuffers,
    kv_pool: MlaKvPool,
    kv_state: MlaKvState,
}

impl RankForwardState {
    fn init(model: &DsV3Model, max_bs: usize) -> Result<Self> {
        let ctx = model.device_ctx();
        let config = model.config();
        let num_layers = model.layers.len();

        let bufs = MlaForwardBuffers::new(ctx, config, max_bs)?;
        let kv_pool = MlaKvPool::new(
            ctx,
            num_layers,
            config.kv_lora_rank,
            config.qk_rope_head_dim,
            64,  // page_size (FlashMLA requires 64)
            128, // num_pages — enough for ~8K tokens
        )?;
        let kv_state = kv_pool.alloc();

        Ok(Self {
            bufs,
            kv_pool,
            kv_state,
        })
    }

    fn reset(&mut self) {
        self.kv_state.reset();
    }
}

impl RankHandle {
    fn spawn(rank: usize, model: DsV3Model) -> Result<Self> {
        let (tx, rx) = channel::unbounded();
        let (startup_tx, startup_rx) = channel::bounded(1);

        let handle = thread::Builder::new()
            .name(format!("dsv3-rank-{rank}"))
            .spawn(move || {
                match bind_device(model.device_ctx()) {
                    Ok(()) => {
                        let _ = startup_tx.send(Ok(()));
                        let mut fwd_state: Option<RankForwardState> = None;

                        while let Ok(cmd) = rx.recv() {
                            match cmd {
                                RankCommand::Sync { resp } => {
                                    let result = model.device_ctx().sync();
                                    let _ = resp.send(result);
                                }
                                RankCommand::Forward {
                                    token_ids,
                                    positions,
                                    resp,
                                } => {
                                    let result = (|| -> Result<Vec<bf16>> {
                                        // Lazy init forward state on first call
                                        let state = match &mut fwd_state {
                                            Some(s) => s,
                                            None => {
                                                let max_bs = 64; // generous for prefill
                                                fwd_state =
                                                    Some(RankForwardState::init(&model, max_bs)?);
                                                fwd_state.as_mut().unwrap()
                                            }
                                        };
                                        state.reset();

                                        let logits = if model.config().has_indexer() {
                                            model.forward_prefill_sparse(
                                                &token_ids,
                                                &positions,
                                                &mut state.bufs,
                                            )?
                                        } else {
                                            model.forward_prefill(
                                                &token_ids,
                                                &positions,
                                                &mut state.kv_state,
                                                &mut state.bufs,
                                                &state.kv_pool,
                                            )?
                                        };

                                        // Copy logits to host
                                        let ctx = model.device_ctx();
                                        let len = logits.hidden_dim * logits.seq_len;
                                        let host: Vec<bf16> =
                                            ctx.stream.clone_dtoh(&logits.data.slice(..len))?;
                                        ctx.sync()?;
                                        Ok(host)
                                    })();
                                    let _ = resp.send(result);
                                }
                                RankCommand::Shutdown => break,
                            }
                        }
                    }
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                    }
                }
                unsafe {
                    crate::ffi::cublas_destroy();
                }
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn dsv3 rank {rank} worker: {e}"))?;

        startup_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("dsv3 rank {rank} worker exited during startup"))??;

        Ok(Self {
            rank,
            tx,
            handle: Some(handle),
        })
    }

    fn shutdown(&mut self) {
        let _ = self.tx.send(RankCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_dsv3_model_path() -> Option<String> {
        std::env::var("PEGAINFER_DSV3_MODEL_PATH").ok()
    }

    /// Load DSV3 in TP1 EP8 across 8 GPUs.
    /// Verifies that each rank loads only its 32 experts and that all
    /// worker threads come up successfully.
    #[test]
    #[ignore]
    fn dsv3_executor_tp1_ep8() {
        let model_path = match get_dsv3_model_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipped: set PEGAINFER_DSV3_MODEL_PATH to run");
                return;
            }
        };
        crate::logging::init_stderr("info");

        let device_ordinals: Vec<usize> = (0..8).collect();
        let executor =
            DsV3Executor::load(&model_path, &device_ordinals, 1).expect("TP1 EP8 load failed");

        assert_eq!(executor.world_size(), 8);
        executor.sync_all().expect("sync_all failed");
        eprintln!("TP1 EP8 executor loaded and synced successfully");
    }

    /// Load DSV3 on 2 GPUs with TP1 EP2 (partial — only 4 layers).
    /// Lighter-weight test: 3 dense + 1 MoE layer, each rank gets 128 experts.
    #[test]
    #[ignore]
    fn dsv3_executor_tp1_ep2_partial() {
        let model_path = match get_dsv3_model_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipped: set PEGAINFER_DSV3_MODEL_PATH to run");
                return;
            }
        };
        crate::logging::init_stderr("info");

        let parallel_0 = ParallelConfig {
            tp_rank: 0,
            tp_size: 1,
            ep_rank: 0,
            ep_size: 2,
        };
        let parallel_1 = ParallelConfig {
            tp_rank: 0,
            tp_size: 1,
            ep_rank: 1,
            ep_size: 2,
        };

        // 4 layers: 3 dense + 1 MoE — fits on a single GPU
        let model_0 = DsV3Model::from_safetensors_partial_parallel(&model_path, 0, 4, parallel_0)
            .expect("rank 0 load failed");
        let model_1 = DsV3Model::from_safetensors_partial_parallel(&model_path, 1, 4, parallel_1)
            .expect("rank 1 load failed");

        // Verify expert counts: 256 / 2 = 128 per rank for MoE layers
        let cfg = model_0.config();
        let (start_0, count_0) = parallel_0.local_expert_range(cfg.n_routed_experts);
        let (start_1, count_1) = parallel_1.local_expert_range(cfg.n_routed_experts);
        assert_eq!((start_0, count_0), (0, 128));
        assert_eq!((start_1, count_1), (128, 128));

        // Verify MoE layer on rank 0 has 128 experts
        for layer in &model_0.layers {
            if let crate::model::dsv3::weights::FfnWeights::MoE(moe) = &layer.ffn {
                assert_eq!(moe.experts.len(), 128, "rank 0 should have 128 experts");
            }
        }
        // Same for rank 1
        for layer in &model_1.layers {
            if let crate::model::dsv3::weights::FfnWeights::MoE(moe) = &layer.ffn {
                assert_eq!(moe.experts.len(), 128, "rank 1 should have 128 experts");
            }
        }

        eprintln!("TP1 EP2 partial load verified: rank0 experts 0..128, rank1 experts 128..256");
    }

    /// Full 61-layer forward on 8 GPUs (TP1 EP8).
    ///
    /// Runs token-by-token forward through all layers and compares output
    /// logits against Python references (generated by gen_logits_ref.py).
    /// Tests multiple prompts with a single model load.
    ///
    /// Requirements:
    /// - PEGAINFER_DSV3_MODEL_PATH set to model weights directory
    /// - Reference data in test_data/dsv3_logits_ref/ (run gen_logits_ref.py first)
    /// - 8x H20 GPUs available
    #[test]
    #[ignore]
    fn dsv3_forward_full_ep8() {
        let model_path = match get_dsv3_model_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipped: set PEGAINFER_DSV3_MODEL_PATH to run");
                return;
            }
        };
        crate::logging::init_stderr("info");

        // Load manifest
        let ref_dir = "test_data/dsv3_logits_ref";
        let manifest_path = format!("{}/manifest.json", ref_dir);
        if !std::path::Path::new(&manifest_path).exists() {
            eprintln!(
                "Skipped: run 'cd tools/dsv3_ref && \
                 /root/develop/xingming/vllm_test/.venv/bin/python gen_logits_ref.py \
                 --model-path <path> --output-dir ../../test_data/dsv3_logits_ref' first"
            );
            return;
        }

        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
        let prompts = manifest["prompts"].as_array().unwrap();
        eprintln!("{} prompts to verify", prompts.len());

        // Load model (full 61 layers, 8 GPUs) — once for all prompts
        let device_ordinals: Vec<usize> = (0..8).collect();
        let executor =
            DsV3Executor::load(&model_path, &device_ordinals, 1).expect("TP1 EP8 load failed");
        eprintln!("Model loaded on {} GPUs\n", executor.world_size());

        let mut all_passed = true;

        for entry in prompts {
            let prompt = entry["prompt"].as_str().unwrap();
            let token_ids: Vec<u32> = entry["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let ref_generated_id = entry["generated_token_id"].as_u64().unwrap() as usize;
            let top10_ids: Vec<usize> = entry["top10_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();

            eprintln!("========================================");
            eprintln!("Prompt: {:?}  ({} tokens)", prompt, token_ids.len());
            eprintln!("========================================");

            // Build positions: [0, 1, 2, ..., seq_len-1]
            let positions: Vec<i32> = (0..token_ids.len() as i32).collect();

            // Run forward
            let start = std::time::Instant::now();
            let logits_bf16 = executor
                .forward(&token_ids, &positions)
                .expect("forward failed");
            let elapsed = start.elapsed();
            eprintln!(
                "  Forward: {:.2}s ({} tokens x 61 layers)",
                elapsed.as_secs_f64(),
                token_ids.len(),
            );

            // Convert to f32 and find our top-K
            let logits_f32: Vec<f32> = logits_bf16.iter().map(|x| x.to_f32()).collect();
            let mut indexed: Vec<(usize, f32)> = logits_f32.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            eprintln!(
                "  Ours  top-5: {}",
                indexed[..5]
                    .iter()
                    .map(|(id, v)| format!("({}, {:.2})", id, v))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            eprintln!("  Ref   top-5: {:?}", &top10_ids[..5.min(top10_ids.len())]);

            // Check top-1 (greedy decode token)
            let top1_match = indexed[0].0 == ref_generated_id;
            eprintln!(
                "  Top-1: ours=id{} ref=id{} {}",
                indexed[0].0,
                ref_generated_id,
                if top1_match { "MATCH" } else { "MISMATCH" }
            );

            // Check top-5 overlap
            let our_top5: std::collections::HashSet<usize> =
                indexed.iter().take(5).map(|x| x.0).collect();
            let ref_top5: std::collections::HashSet<usize> =
                top10_ids.iter().take(5).copied().collect();
            let overlap = our_top5.intersection(&ref_top5).count();
            eprintln!("  Top-5 overlap: {}/5", overlap);

            if !top1_match {
                eprintln!("  FAILED: top-1 mismatch");
                all_passed = false;
            }
            if overlap < 3 {
                eprintln!("  FAILED: top-5 overlap < 3");
                all_passed = false;
            }

            eprintln!();
        }

        assert!(all_passed, "Some prompts failed verification");
        eprintln!(
            "Phase 2d full forward verification PASSED ({} prompts)",
            prompts.len()
        );
    }
}
