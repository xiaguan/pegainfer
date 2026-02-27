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

    pub fn from_safetensors(
        ctx: &DeviceContext,
        data: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if data.len() != rows * cols * std::mem::size_of::<bf16>() {
            return Err(anyhow!(
                "Data length mismatch: expected {} bytes, got {} bytes",
                rows * cols * std::mem::size_of::<bf16>(),
                data.len()
            ));
        }
        // NOTE: This assumes a little-endian host. Safetensors are little-endian.
        // On a big-endian machine, this will be incorrect. A full solution would
        // involve byte-swapping.
        let slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), rows * cols) };
        let gpu_data = ctx
            .stream
            .clone_htod(slice)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn copy_matrix_to_host(ctx: &DeviceContext, matrix: &DeviceMatrix) -> Vec<bf16> {
        let host = ctx
            .stream
            .clone_dtoh(&matrix.data)
            .expect("D2H copy failed");
        ctx.sync().expect("CUDA sync failed");
        host
    }

    #[test]
    fn test_device_matrix_from_host_roundtrip() {
        let ctx = DeviceContext::new().expect("Failed to create CUDA context");
        let rows = 2;
        let cols = 3;
        let host = vec![
            bf16::from_f32(-1.5),
            bf16::from_f32(0.0),
            bf16::from_f32(2.25),
            bf16::from_f32(7.0),
            bf16::from_f32(-3.0),
            bf16::from_f32(0.5),
        ];

        let matrix =
            DeviceMatrix::from_host(&ctx, &host, rows, cols).expect("from_host should succeed");

        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);

        let got = copy_matrix_to_host(&ctx, &matrix);
        assert_eq!(got.len(), host.len());
        for (idx, (actual, expected)) in got.iter().zip(host.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "roundtrip mismatch at index {}",
                idx
            );
        }
    }

    #[test]
    fn test_device_matrix_from_safetensors_matches_from_host() {
        let ctx = DeviceContext::new().expect("Failed to create CUDA context");
        let rows = 3;
        let cols = 2;
        let host = vec![
            bf16::from_f32(-8.0),
            bf16::from_f32(-0.25),
            bf16::from_f32(1.0),
            bf16::from_f32(3.5),
            bf16::from_f32(9.0),
            bf16::from_f32(10.75),
        ];
        let safetensor_bytes: Vec<u8> = host
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let from_host =
            DeviceMatrix::from_host(&ctx, &host, rows, cols).expect("from_host should succeed");
        let from_safetensors = DeviceMatrix::from_safetensors(&ctx, &safetensor_bytes, rows, cols)
            .expect("from_safetensors should succeed");

        assert_eq!(from_safetensors.rows, from_host.rows);
        assert_eq!(from_safetensors.cols, from_host.cols);

        let host_out = copy_matrix_to_host(&ctx, &from_host);
        let safetensors_out = copy_matrix_to_host(&ctx, &from_safetensors);
        assert_eq!(host_out.len(), safetensors_out.len());
        for (idx, (a, b)) in host_out.iter().zip(safetensors_out.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "from_safetensors/from_host mismatch at index {}",
                idx
            );
        }
    }
}
