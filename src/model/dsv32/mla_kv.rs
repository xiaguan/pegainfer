//! MLA paged KV cache — per-layer layout for FlashMLA.
//!
//! FlashMLA dense decode expects `kcache: [num_blocks, page_size=64, h_kv=1, d_kv]`.
//! Each layer gets its own contiguous slice of the shared buffer so the kernel
//! can index pages directly via `block_table`.
//!
//! ```text
//! Buffer layout (layer-first):
//!   layer 0:  [page_0 | page_1 | ... | page_{N-1}]
//!   layer 1:  [page_0 | page_1 | ... | page_{N-1}]
//!   ...
//!   layer L:  [page_0 | page_1 | ... | page_{N-1}]
//!
//! Each page:  [page_size × kv_dim]  bf16
//!             = [64 × 576]  for DSV3.2 (c_KV 512 + k_R 64)
//! ```

use std::sync::Arc;

use anyhow::{Result, bail};
use cudarc::driver::CudaSlice;
use half::bf16;

use crate::page_pool::{OwnedPagePermit, PageId, PagePool};
use crate::tensor::DeviceContext;

/// Pure geometry — no GPU, no allocation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MlaKvLayout {
    pub(crate) page_size: usize,
    pub(crate) num_layers: usize,
    /// kv_lora_rank + qk_rope_head_dim (512 + 64 = 576 for DSV3.2).
    pub(crate) kv_dim: usize,
    /// kv_lora_rank — also FlashMLA's head_size_v (512 for DSV3.2).
    pub(crate) kv_lora_rank: usize,
    /// Elements per page per layer: page_size × kv_dim.
    pub(crate) page_kv_len: usize,
}

impl MlaKvLayout {
    pub(crate) fn new(
        num_layers: usize,
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
        page_size: usize,
    ) -> Self {
        let kv_dim = kv_lora_rank + qk_rope_head_dim;
        let page_kv_len = page_size * kv_dim;
        Self {
            page_size,
            num_layers,
            kv_dim,
            kv_lora_rank,
            page_kv_len,
        }
    }
}

struct MlaKvPoolInner {
    pool: PagePool,
    buffer: CudaSlice<bf16>,
    layout: MlaKvLayout,
    num_pages: usize,
    padding_permit: OwnedPagePermit,
}

/// Shared MLA KV backing storage with per-layer paged layout.
///
/// Owns a single GPU buffer in layer-first order. For layer `l`,
/// the contiguous `[num_pages, page_size, 1, kv_dim]` region starts at
/// `layer_offset(l)` bf16 elements from the base pointer.
#[derive(Clone)]
pub(crate) struct MlaKvPool {
    inner: Arc<MlaKvPoolInner>,
}

impl MlaKvPool {
    pub(crate) fn new(
        ctx: &DeviceContext,
        num_layers: usize,
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
        page_size: usize,
        num_pages: usize,
    ) -> Result<Self> {
        let layout = MlaKvLayout::new(num_layers, kv_lora_rank, qk_rope_head_dim, page_size);
        let total_elements = num_layers * num_pages * layout.page_kv_len;

        let buffer: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(total_elements)
            .map_err(|e| anyhow::anyhow!("MlaKvPool alloc failed: {e}"))?;

        let pool = PagePool::new(num_pages);
        let padding_permit = pool
            .try_acquire_many(1)
            .expect("pool must have at least 1 page for padding");

        Ok(Self {
            inner: Arc::new(MlaKvPoolInner {
                pool,
                buffer,
                layout,
                num_pages,
                padding_permit,
            }),
        })
    }

    pub(crate) fn alloc(&self) -> MlaKvState {
        MlaKvState {
            permit: self.inner.pool.try_acquire_many(0).expect("zero acquire"),
            seq_len: 0,
            pool: self.clone(),
        }
    }

    pub(crate) fn layout(&self) -> &MlaKvLayout {
        &self.inner.layout
    }

    pub(crate) fn num_pages(&self) -> usize {
        self.inner.num_pages
    }

    pub(crate) fn capacity_pages(&self) -> usize {
        self.inner.pool.capacity_pages()
    }

    pub(crate) fn available_pages(&self) -> usize {
        self.inner.pool.available_pages()
    }

    pub(crate) fn padding_page_id(&self) -> i32 {
        self.inner.padding_permit.pages()[0].index() as i32
    }

    /// Offset (in bf16 elements) to the start of layer `l`'s paged region.
    pub(crate) fn layer_offset(&self, layer: usize) -> usize {
        layer * self.inner.num_pages * self.inner.layout.page_kv_len
    }

    /// Get the underlying buffer reference (for kernel access).
    pub(crate) fn buffer(&self) -> &CudaSlice<bf16> {
        &self.inner.buffer
    }
}

/// Per-request MLA KV state.
pub(crate) struct MlaKvState {
    permit: OwnedPagePermit,
    seq_len: usize,
    pool: MlaKvPool,
}

fn pages_needed(token_count: usize, page_size: usize) -> usize {
    token_count.div_ceil(page_size)
}

impl MlaKvState {
    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub(crate) fn num_pages(&self) -> usize {
        self.permit.pages().len()
    }

    pub(crate) fn last_page_len(&self) -> usize {
        if self.seq_len == 0 {
            0
        } else {
            let rem = self.seq_len % self.pool.inner.layout.page_size;
            if rem == 0 {
                self.pool.inner.layout.page_size
            } else {
                rem
            }
        }
    }

    pub(crate) fn page_indices_i32(&self) -> Vec<i32> {
        self.permit
            .pages()
            .iter()
            .map(|p| p.index() as i32)
            .collect()
    }

    pub(crate) fn ensure_capacity(&mut self, token_count: usize) -> Result<()> {
        let needed = pages_needed(token_count, self.pool.inner.layout.page_size);
        let held = self.permit.len();
        if needed <= held {
            return Ok(());
        }
        let grow = needed - held;
        if !self.permit.try_grow(grow) {
            bail!(
                "MlaKvState: out of pages (need {grow} more, {} available)",
                self.pool.available_pages()
            );
        }
        Ok(())
    }

    pub(crate) fn advance(&mut self, count: usize) {
        self.seq_len += count;
    }

    pub(crate) fn desc(&self) -> MlaKvDesc<'_> {
        MlaKvDesc {
            layout: self.pool.inner.layout,
            buffer: &self.pool.inner.buffer,
            num_pages: self.pool.inner.num_pages,
            pages: self.permit.pages(),
            seq_len: self.seq_len,
            last_page_len: self.last_page_len(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.permit = self
            .pool
            .inner
            .pool
            .try_acquire_many(0)
            .expect("zero acquire");
        self.seq_len = 0;
    }
}

/// Kernel-facing metadata for one request's MLA KV.
pub(crate) struct MlaKvDesc<'a> {
    layout: MlaKvLayout,
    buffer: &'a CudaSlice<bf16>,
    num_pages: usize,
    pages: &'a [PageId],
    seq_len: usize,
    last_page_len: usize,
}

impl MlaKvDesc<'_> {
    pub(crate) fn layout(&self) -> &MlaKvLayout {
        &self.layout
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub(crate) fn last_page_len(&self) -> usize {
        self.last_page_len
    }

    pub(crate) fn num_pages(&self) -> usize {
        self.pages.len()
    }

    pub(crate) fn page_indices(&self) -> &[PageId] {
        self.pages
    }

    pub(crate) fn buffer(&self) -> &CudaSlice<bf16> {
        self.buffer
    }

    /// Offset (in bf16 elements) to layer `l`'s paged region.
    /// The region is `[total_pool_pages, page_size, 1, kv_dim]` contiguous.
    pub(crate) fn layer_offset(&self, layer: usize) -> usize {
        layer * self.num_pages * self.layout.page_kv_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pool(page_size: usize, num_pages: usize) -> MlaKvPool {
        let ctx = DeviceContext::new().expect("GPU required for mla_kv tests");
        // DSV3.2 dims: kv_lora_rank=512, qk_rope_head_dim=64, 61 layers
        MlaKvPool::new(&ctx, 61, 512, 64, page_size, num_pages).expect("MlaKvPool::new failed")
    }

    #[test]
    fn layout_geometry_dsv32() {
        let l = MlaKvLayout::new(61, 512, 64, 64);

        assert_eq!(l.kv_dim, 576);
        assert_eq!(l.kv_lora_rank, 512);
        assert_eq!(l.page_kv_len, 64 * 576); // 36864 elements per page per layer
        // Bytes per page per layer: 36864 * 2 = 73728 = 72 KB
        assert_eq!(l.page_kv_len * 2, 72 * 1024);
    }

    #[test]
    fn pool_buffer_size() {
        let ctx = DeviceContext::new().expect("GPU required");
        let num_pages = 320; // ~20K tokens
        let pool = MlaKvPool::new(&ctx, 61, 512, 64, 64, num_pages).expect("MlaKvPool::new failed");

        // Layer offsets
        let page_kv_len = 64 * 576; // 36864
        assert_eq!(pool.layer_offset(0), 0);
        assert_eq!(pool.layer_offset(1), num_pages * page_kv_len);
        assert_eq!(pool.layer_offset(61), 61 * num_pages * page_kv_len);
    }

    #[test]
    fn state_lifecycle() {
        // 5 pages: 1 padding, 4 available
        let pool = test_pool(64, 5);
        let mut kv = pool.alloc();

        assert_eq!(kv.seq_len(), 0);
        assert_eq!(kv.num_pages(), 0);

        // Prefill 100 tokens: needs 2 pages (100/64 = 1.5625)
        kv.ensure_capacity(100).unwrap();
        kv.advance(100);
        assert_eq!(kv.seq_len(), 100);
        assert_eq!(kv.num_pages(), 2);
        assert_eq!(kv.last_page_len(), 36); // 100 - 64 = 36
        assert_eq!(pool.available_pages(), 2);

        // Decode to 128: still 2 pages, last_page_len=64 (full)
        kv.ensure_capacity(128).unwrap();
        kv.advance(28);
        assert_eq!(kv.num_pages(), 2);
        assert_eq!(kv.last_page_len(), 64);

        // One more token: 3 pages
        kv.ensure_capacity(129).unwrap();
        kv.advance(1);
        assert_eq!(kv.num_pages(), 3);
        assert_eq!(kv.last_page_len(), 1);

        // Reset
        kv.reset();
        assert_eq!(kv.seq_len(), 0);
        assert_eq!(pool.available_pages(), 4);
    }

    #[test]
    fn desc_layer_offset() {
        let pool = test_pool(64, 10);
        let kv = pool.alloc();
        let desc = kv.desc();

        // Layer 0 starts at 0
        assert_eq!(desc.layer_offset(0), 0);
        // Layer 1 starts after all pages of layer 0
        // 10 pages × 64 tokens × 576 dims = 368640
        assert_eq!(desc.layer_offset(1), 10 * 64 * 576);
        assert_eq!(desc.layer_offset(1), 368_640);
    }
}
