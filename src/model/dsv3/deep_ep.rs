//! DeepEP intranode buffer management for EP All-to-All.
//!
//! Manages NVLink IPC buffers, barrier signals, and host-mapped memory
//! for cross-GPU token dispatch/combine in MoE layers.

use std::ptr;
use std::sync::{Arc, Barrier, Mutex};

use anyhow::{Result, ensure};
use log::info;

use crate::ffi;

// DeepEP constants (from configs.cuh)
const NUM_MAX_NVL_PEERS: usize = 8;
const NUM_BUFFER_ALIGNMENT_BYTES: usize = 128;
const NUM_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const NUM_MAX_LOCAL_EXPERTS: usize = 1024;

/// DeepEP config for intranode dispatch/combine.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeepEpConfig {
    /// Number of SMs used for communication (even number, each 2 SMs = 1 channel).
    pub num_sms: i32,
    /// Max tokens that can be sent per chunk (dispatch config).
    pub num_max_nvl_chunked_send_tokens: i32,
    /// Max tokens that can be received per chunk.
    pub num_max_nvl_chunked_recv_tokens: i32,
}

impl Default for DeepEpConfig {
    fn default() -> Self {
        // Defaults for EP8 intranode (from DeepEP Python API)
        Self {
            num_sms: 20,
            num_max_nvl_chunked_send_tokens: 6,
            num_max_nvl_chunked_recv_tokens: 256,
        }
    }
}

impl DeepEpConfig {
    pub fn num_channels(&self) -> i32 {
        self.num_sms / 2
    }
}

/// Per-rank DeepEP buffer holding NVLink IPC buffer pointers, barrier signals,
/// and host-mapped memory for CPU-GPU synchronization.
pub(crate) struct DeepEpBuffer {
    rank: usize,
    num_ranks: usize,

    // NVLink buffer: local allocation + remote IPC-opened pointers
    nvl_buffer_local: *mut std::ffi::c_void,
    nvl_buffer_size: usize,
    // buffer_ptrs[i] = pointer to rank i's NVLink buffer (local or IPC-opened)
    buffer_ptrs: [*mut std::ffi::c_void; NUM_MAX_NVL_PEERS],
    // GPU-resident copy of buffer_ptrs
    buffer_ptrs_gpu: *mut *mut std::ffi::c_void,

    // Barrier signals: same pattern as buffer_ptrs
    barrier_local: *mut std::ffi::c_void,
    barrier_size: usize,
    barrier_ptrs: [*mut i32; NUM_MAX_NVL_PEERS],
    barrier_ptrs_gpu: *mut *mut i32,

    // Host-mapped memory for CPU-GPU sync (volatile reads from CPU side)
    moe_recv_counter_host: *mut i32,        // total recv token count
    moe_recv_counter_mapped: *mut i32,      // device pointer to same memory
    moe_recv_expert_counter_host: *mut i32, // per-expert recv counts
    moe_recv_expert_counter_mapped: *mut i32,

    // Workspace
    workspace: *mut std::ffi::c_void,
    workspace_size: usize,

    // Config
    pub config: DeepEpConfig,
}

// Raw pointers are Send if we manage lifetimes correctly.
unsafe impl Send for DeepEpBuffer {}
unsafe impl Sync for DeepEpBuffer {}

/// Shared state for IPC handle exchange between ranks.
pub(crate) struct IpcExchange {
    buffer_handles: Vec<ffi::CudaIpcMemHandle>,
    barrier_handles: Vec<ffi::CudaIpcMemHandle>,
}

/// Calculate NVLink buffer size for intranode dispatch/combine.
/// Mirrors `Config::get_nvl_buffer_size_hint()` from DeepEP Python API.
fn nvl_buffer_size_hint(
    num_ranks: usize,
    hidden: usize,
    num_topk: usize,
    config: &DeepEpConfig,
) -> usize {
    let num_channels = config.num_channels() as usize;
    let hidden_bytes = hidden * 2; // bf16

    // Per-rank metadata at buffer start
    let rank_prefix_matrix = num_ranks * num_ranks * 4; // int
    let expert_prefix = num_ranks * NUM_MAX_LOCAL_EXPERTS * 4;
    let metadata_size = rank_prefix_matrix + expert_prefix;

    // Per-channel ring buffers (send + recv, asymmetric)
    let tokens = config.num_max_nvl_chunked_recv_tokens as usize;
    let per_token_bytes = hidden_bytes                          // x data
        + 4                                                      // src_idx (int)
        + num_topk * 8                                           // topk_idx (int64)
        + num_topk * 4; // topk_weights (float)
    let channel_data = num_ranks * tokens * per_token_bytes;

    // Control buffers (head/tail indices, prefix sums)
    let channel_control = num_channels * num_ranks * 4 * 5; // 5 arrays of int

    let total = metadata_size + channel_data * num_channels + channel_control;

    // Add generous padding for alignment and internal bookkeeping
    let padded = total * 2 + NUM_WORKSPACE_BYTES;

    // Align to 128 bytes
    (padded + NUM_BUFFER_ALIGNMENT_BYTES - 1) / NUM_BUFFER_ALIGNMENT_BYTES
        * NUM_BUFFER_ALIGNMENT_BYTES
}

/// Barrier signal buffer size per rank.
fn barrier_signal_size(num_ranks: usize) -> usize {
    // Each rank needs num_ranks ints for the barrier protocol, padded to 128 bytes.
    let min = num_ranks * std::mem::size_of::<i32>();
    (min + NUM_BUFFER_ALIGNMENT_BYTES - 1) / NUM_BUFFER_ALIGNMENT_BYTES * NUM_BUFFER_ALIGNMENT_BYTES
}

impl DeepEpBuffer {
    /// Create a DeepEP buffer for one rank.
    ///
    /// Call from each rank's thread after setting the CUDA device.
    /// After all ranks have called this, call `exchange_ipc_handles` to
    /// complete the setup.
    pub fn alloc(
        rank: usize,
        num_ranks: usize,
        hidden: usize,
        num_topk: usize,
        config: DeepEpConfig,
        _stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        ensure!(
            num_ranks <= NUM_MAX_NVL_PEERS,
            "num_ranks > 8 not supported for intranode"
        );

        let nvl_size = nvl_buffer_size_hint(num_ranks, hidden, num_topk, &config);
        let bar_size = barrier_signal_size(num_ranks);

        // Allocate NVLink buffer
        let mut nvl_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut nvl_ptr, nvl_size) };
        ensure!(err == 0, "cudaMalloc NVLink buffer failed: {err}");
        let err = unsafe { ffi::cudaMemset(nvl_ptr, 0, nvl_size) };
        ensure!(err == 0, "cudaMemset NVLink buffer failed: {err}");

        // Allocate barrier signal buffer
        let mut bar_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut bar_ptr, bar_size) };
        ensure!(err == 0, "cudaMalloc barrier buffer failed: {err}");
        let err = unsafe { ffi::cudaMemset(bar_ptr, 0, bar_size) };
        ensure!(err == 0, "cudaMemset barrier buffer failed: {err}");

        // Allocate host-mapped memory for moe_recv_counter
        let mut recv_counter_host: *mut std::ffi::c_void = ptr::null_mut();
        let err = unsafe {
            ffi::cudaHostAlloc(
                &mut recv_counter_host,
                std::mem::size_of::<i32>(),
                ffi::CUDA_HOST_ALLOC_MAPPED,
            )
        };
        ensure!(err == 0, "cudaHostAlloc moe_recv_counter failed: {err}");
        let mut recv_counter_mapped: *mut std::ffi::c_void = ptr::null_mut();
        let err = unsafe {
            ffi::cudaHostGetDevicePointer(&mut recv_counter_mapped, recv_counter_host, 0)
        };
        ensure!(
            err == 0,
            "cudaHostGetDevicePointer moe_recv_counter failed: {err}"
        );

        // Allocate host-mapped memory for moe_recv_expert_counter
        let expert_counter_bytes = NUM_MAX_LOCAL_EXPERTS * std::mem::size_of::<i32>();
        let mut expert_counter_host: *mut std::ffi::c_void = ptr::null_mut();
        let err = unsafe {
            ffi::cudaHostAlloc(
                &mut expert_counter_host,
                expert_counter_bytes,
                ffi::CUDA_HOST_ALLOC_MAPPED,
            )
        };
        ensure!(
            err == 0,
            "cudaHostAlloc moe_recv_expert_counter failed: {err}"
        );
        let mut expert_counter_mapped: *mut std::ffi::c_void = ptr::null_mut();
        let err = unsafe {
            ffi::cudaHostGetDevicePointer(&mut expert_counter_mapped, expert_counter_host, 0)
        };
        ensure!(
            err == 0,
            "cudaHostGetDevicePointer moe_recv_expert_counter failed: {err}"
        );

        // Allocate workspace
        let mut ws_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut ws_ptr, NUM_WORKSPACE_BYTES) };
        ensure!(err == 0, "cudaMalloc workspace failed: {err}");
        let err = unsafe { ffi::cudaMemset(ws_ptr, 0, NUM_WORKSPACE_BYTES) };
        ensure!(err == 0, "cudaMemset workspace failed: {err}");

        let mut buffer_ptrs = [ptr::null_mut(); NUM_MAX_NVL_PEERS];
        buffer_ptrs[rank] = nvl_ptr;

        let mut barrier_ptrs: [*mut i32; NUM_MAX_NVL_PEERS] = [ptr::null_mut(); NUM_MAX_NVL_PEERS];
        barrier_ptrs[rank] = bar_ptr as *mut i32;

        info!(
            "Rank {}: DeepEP buffer allocated (nvl={:.1}MB, barrier={}B, workspace=32MB)",
            rank,
            nvl_size as f64 / (1024.0 * 1024.0),
            bar_size,
        );

        Ok(Self {
            rank,
            num_ranks,
            nvl_buffer_local: nvl_ptr,
            nvl_buffer_size: nvl_size,
            buffer_ptrs,
            buffer_ptrs_gpu: ptr::null_mut(),
            barrier_local: bar_ptr,
            barrier_size: bar_size,
            barrier_ptrs,
            barrier_ptrs_gpu: ptr::null_mut(),
            moe_recv_counter_host: recv_counter_host as *mut i32,
            moe_recv_counter_mapped: recv_counter_mapped as *mut i32,
            moe_recv_expert_counter_host: expert_counter_host as *mut i32,
            moe_recv_expert_counter_mapped: expert_counter_mapped as *mut i32,
            workspace: ws_ptr,
            workspace_size: NUM_WORKSPACE_BYTES,
            config,
        })
    }

    /// Exchange IPC handles between all ranks and open remote buffers.
    ///
    /// Must be called from each rank's thread concurrently. Uses the shared
    /// `exchange` mutex and `barrier` to coordinate.
    pub fn exchange_ipc_handles(
        &mut self,
        exchange: &Arc<Mutex<IpcExchange>>,
        barrier: &Arc<Barrier>,
    ) -> Result<()> {
        // Step 1: Get our IPC handles and store them in the shared exchange
        let mut nvl_handle = ffi::CudaIpcMemHandle {
            reserved: [0u8; 64],
        };
        let err = unsafe { ffi::cudaIpcGetMemHandle(&mut nvl_handle, self.nvl_buffer_local) };
        ensure!(err == 0, "cudaIpcGetMemHandle (NVL) failed: {err}");

        let mut bar_handle = ffi::CudaIpcMemHandle {
            reserved: [0u8; 64],
        };
        let err = unsafe { ffi::cudaIpcGetMemHandle(&mut bar_handle, self.barrier_local) };
        ensure!(err == 0, "cudaIpcGetMemHandle (barrier) failed: {err}");

        {
            let mut ex = exchange.lock().unwrap();
            ex.buffer_handles[self.rank] = nvl_handle;
            ex.barrier_handles[self.rank] = bar_handle;
        }

        // Step 2: Wait for all ranks to deposit their handles
        barrier.wait();

        // Step 3: Open remote handles
        let handles = {
            let ex = exchange.lock().unwrap();
            (ex.buffer_handles.clone(), ex.barrier_handles.clone())
        };

        for r in 0..self.num_ranks {
            if r == self.rank {
                continue;
            }
            // Open NVLink buffer
            let mut remote_ptr: *mut std::ffi::c_void = ptr::null_mut();
            let err = unsafe {
                ffi::cudaIpcOpenMemHandle(
                    &mut remote_ptr,
                    handles.0[r],
                    ffi::CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
                )
            };
            ensure!(
                err == 0,
                "cudaIpcOpenMemHandle (NVL, rank {r}) failed: {err}"
            );
            self.buffer_ptrs[r] = remote_ptr;

            // Open barrier signal buffer
            let mut remote_bar: *mut std::ffi::c_void = ptr::null_mut();
            let err = unsafe {
                ffi::cudaIpcOpenMemHandle(
                    &mut remote_bar,
                    handles.1[r],
                    ffi::CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
                )
            };
            ensure!(
                err == 0,
                "cudaIpcOpenMemHandle (barrier, rank {r}) failed: {err}"
            );
            self.barrier_ptrs[r] = remote_bar as *mut i32;
        }

        // Step 4: Upload pointer arrays to GPU
        let ptrs_size = NUM_MAX_NVL_PEERS * std::mem::size_of::<*mut std::ffi::c_void>();
        let mut gpu_buf_ptrs: *mut std::ffi::c_void = ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut gpu_buf_ptrs, ptrs_size) };
        ensure!(err == 0, "cudaMalloc buffer_ptrs_gpu failed: {err}");
        let err = unsafe {
            ffi::cudaMemcpy(
                gpu_buf_ptrs,
                self.buffer_ptrs.as_ptr() as *const std::ffi::c_void,
                ptrs_size,
                ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        ensure!(err == 0, "cudaMemcpy buffer_ptrs_gpu failed: {err}");
        self.buffer_ptrs_gpu = gpu_buf_ptrs as *mut *mut std::ffi::c_void;

        let bar_ptrs_size = NUM_MAX_NVL_PEERS * std::mem::size_of::<*mut i32>();
        let mut gpu_bar_ptrs: *mut std::ffi::c_void = ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut gpu_bar_ptrs, bar_ptrs_size) };
        ensure!(err == 0, "cudaMalloc barrier_ptrs_gpu failed: {err}");
        let err = unsafe {
            ffi::cudaMemcpy(
                gpu_bar_ptrs,
                self.barrier_ptrs.as_ptr() as *const std::ffi::c_void,
                bar_ptrs_size,
                ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        ensure!(err == 0, "cudaMemcpy barrier_ptrs_gpu failed: {err}");
        self.barrier_ptrs_gpu = gpu_bar_ptrs as *mut *mut i32;

        // Step 5: Wait for all ranks to finish opening handles
        barrier.wait();

        info!(
            "Rank {}: DeepEP IPC handles exchanged, {} peers connected",
            self.rank,
            self.num_ranks - 1
        );
        Ok(())
    }

    // ========================================================================
    // Accessors for forward pass
    // ========================================================================

    pub fn rank(&self) -> i32 {
        self.rank as i32
    }

    pub fn num_ranks(&self) -> i32 {
        self.num_ranks as i32
    }

    pub fn buffer_ptrs_gpu(&self) -> *mut *mut std::ffi::c_void {
        self.buffer_ptrs_gpu
    }

    pub fn barrier_signal_ptrs_gpu(&self) -> *mut *mut i32 {
        self.barrier_ptrs_gpu
    }

    pub fn moe_recv_counter_mapped(&self) -> *mut i32 {
        self.moe_recv_counter_mapped
    }

    pub fn moe_recv_expert_counter_mapped(&self) -> *mut i32 {
        self.moe_recv_expert_counter_mapped
    }

    /// Read the recv token count from host-mapped memory (volatile).
    /// Must be called after notify_dispatch + stream sync.
    pub fn read_recv_count(&self) -> i32 {
        unsafe { ptr::read_volatile(self.moe_recv_counter_host) }
    }

    /// Read per-expert recv token counts from host-mapped memory.
    pub fn read_expert_recv_counts(&self, num_local_experts: usize) -> Vec<i32> {
        let mut counts = vec![0i32; num_local_experts];
        for i in 0..num_local_experts {
            counts[i] = unsafe { ptr::read_volatile(self.moe_recv_expert_counter_host.add(i)) };
        }
        counts
    }

    /// Reset recv counters before each dispatch (write 0 via host pointer).
    pub fn reset_recv_counters(&self) {
        unsafe {
            ptr::write_volatile(self.moe_recv_counter_host, 0);
        }
    }
}

impl Drop for DeepEpBuffer {
    fn drop(&mut self) {
        unsafe {
            // Close IPC handles for remote ranks
            for r in 0..self.num_ranks {
                if r != self.rank {
                    if !self.buffer_ptrs[r].is_null() {
                        ffi::cudaIpcCloseMemHandle(self.buffer_ptrs[r]);
                    }
                    if !self.barrier_ptrs[r].is_null() {
                        ffi::cudaIpcCloseMemHandle(self.barrier_ptrs[r] as *mut std::ffi::c_void);
                    }
                }
            }
            // Free GPU pointer arrays
            if !self.buffer_ptrs_gpu.is_null() {
                ffi::cudaFree(self.buffer_ptrs_gpu as *mut std::ffi::c_void);
            }
            if !self.barrier_ptrs_gpu.is_null() {
                ffi::cudaFree(self.barrier_ptrs_gpu as *mut std::ffi::c_void);
            }
            // Free local allocations
            if !self.nvl_buffer_local.is_null() {
                ffi::cudaFree(self.nvl_buffer_local);
            }
            if !self.barrier_local.is_null() {
                ffi::cudaFree(self.barrier_local);
            }
            if !self.workspace.is_null() {
                ffi::cudaFree(self.workspace);
            }
            // Free host-mapped memory
            if !self.moe_recv_counter_host.is_null() {
                ffi::cudaFreeHost(self.moe_recv_counter_host as *mut std::ffi::c_void);
            }
            if !self.moe_recv_expert_counter_host.is_null() {
                ffi::cudaFreeHost(self.moe_recv_expert_counter_host as *mut std::ffi::c_void);
            }
        }
    }
}

/// Create IPC exchange state for a set of ranks.
pub(crate) fn new_ipc_exchange(num_ranks: usize) -> (Arc<Mutex<IpcExchange>>, Arc<Barrier>) {
    let exchange = Arc::new(Mutex::new(IpcExchange {
        buffer_handles: vec![
            ffi::CudaIpcMemHandle {
                reserved: [0u8; 64]
            };
            num_ranks
        ],
        barrier_handles: vec![
            ffi::CudaIpcMemHandle {
                reserved: [0u8; 64]
            };
            num_ranks
        ],
    }));
    let barrier = Arc::new(Barrier::new(num_ranks));
    (exchange, barrier)
}
