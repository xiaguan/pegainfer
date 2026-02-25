//! Device tensor types and CUDA context.

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::bf16;
use std::sync::Arc;

use crate::ffi;

/// CUDA device context holding context and stream.
pub struct DeviceContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
}

impl DeviceContext {
    pub fn new() -> Result<Self> {
        let ctx =
            CudaContext::new(0).map_err(|e| anyhow!("Failed to create CUDA context: {}", e))?;

        // Disable multi-stream event tracking before creating streams.
        // We use a single compute stream, so no cross-stream synchronization is needed.
        // This avoids stream.wait(event) calls that break CUDA Graph capture.
        // SAFETY: We only use one stream for all GPU work.
        unsafe {
            ctx.disable_event_tracking();
        }

        let stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("Failed to create CUDA stream: {}", e))?;

        // Initialize cuBLAS handle
        unsafe {
            ffi::cublas_init();
        }

        Ok(Self { ctx, stream })
    }

    /// Synchronize stream
    pub fn sync(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("Sync failed: {}", e))
    }
}

/// 1D device tensor (vector) — stored as bf16.
pub struct DeviceVec {
    pub data: CudaSlice<bf16>,
    pub len: usize,
}

impl DeviceVec {
    /// Create from host data (bf16)
    pub fn from_host(ctx: &DeviceContext, data: &[bf16]) -> Result<Self> {
        let gpu_data = ctx
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            len: data.len(),
        })
    }

    /// Create zeroed tensor
    pub fn zeros(ctx: &DeviceContext, len: usize) -> Result<Self> {
        let gpu_data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            len,
        })
    }

    /// Copy to host as f32 (for compatibility)
    pub fn to_host(&self, ctx: &DeviceContext) -> Result<Vec<f32>> {
        let host_f16 = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
        ctx.sync()?;
        Ok(host_f16.iter().map(|x| x.to_f32()).collect())
    }
}

/// An immutable view into a sub-range of a DeviceVec.
/// Borrows the parent's CudaSlice without copying.
pub struct DeviceVecView<'a> {
    pub data: cudarc::driver::CudaView<'a, bf16>,
    pub len: usize,
}

impl DeviceVec {
    /// Create an immutable sub-view: elements [offset..offset+len).
    pub fn view(&self, offset: usize, len: usize) -> DeviceVecView<'_> {
        assert!(
            offset + len <= self.len,
            "view out of bounds: {}+{} > {}",
            offset,
            len,
            self.len
        );
        DeviceVecView {
            data: self.data.slice(offset..offset + len),
            len,
        }
    }
}

impl Clone for DeviceVec {
    fn clone(&self) -> Self {
        Self {
            data: self.data.try_clone().unwrap(),
            len: self.len,
        }
    }
}

/// 2D device tensor (matrix) — stored in row-major order as bf16.
pub struct DeviceMatrix {
    pub data: CudaSlice<bf16>,
    pub rows: usize,
    pub cols: usize,
}

impl DeviceMatrix {
    /// Create from host data (row-major, bf16)
    pub fn from_host(ctx: &DeviceContext, data: &[bf16], rows: usize, cols: usize) -> Result<Self> {
        assert_eq!(data.len(), rows * cols);
        let gpu_data = ctx
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            rows,
            cols,
        })
    }
}

/// Batched hidden states: seq_len vectors of dim hidden_dim, stored contiguously.
/// Memory layout: [hidden_dim * seq_len] elements, token i at offset i * hidden_dim.
/// cuBLAS interprets as [hidden_dim, seq_len] column-major.
pub struct HiddenStates {
    pub data: CudaSlice<bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

impl HiddenStates {
    /// Create zeroed batch
    pub fn zeros(ctx: &DeviceContext, hidden_dim: usize, seq_len: usize) -> Result<Self> {
        let data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(hidden_dim * seq_len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        Ok(Self {
            data,
            hidden_dim,
            seq_len,
        })
    }
}
