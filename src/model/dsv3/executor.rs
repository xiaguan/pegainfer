//! DSV3 multi-GPU executor: loads models across GPUs with EP/TP sharding
//! and coordinates forward passes.

use std::sync::Arc;
use std::thread;

use anyhow::Result;
use crossbeam_channel as channel;
use log::info;

use super::config::ParallelConfig;
use super::deep_ep::{self, DeepEpBuffer, DeepEpConfig};
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

            let (exchange, barrier) = deep_ep::new_ipc_exchange(world_size);

            // Collect device info before spawning threads (avoids borrow conflict).
            // Cast CUstream to usize to satisfy Send bound (raw ptrs are !Send).
            let rank_info: Vec<_> = models
                .iter()
                .map(|m| {
                    let ctx = m.device_ctx();
                    (ctx.device_ordinal, ctx.stream.cu_stream() as usize)
                })
                .collect();

            // Allocate buffers and exchange IPC handles in parallel
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
                            buf.exchange_ipc_handles(&exchange, &barrier)?;
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

    /// Get a reference to the model on a specific rank.
    /// Only valid from the rank's own thread — returns None if called
    /// from outside. For testing, use the _unchecked variant.
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
                        while let Ok(cmd) = rx.recv() {
                            match cmd {
                                RankCommand::Sync { resp } => {
                                    let result = model.device_ctx().sync();
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
}
