use std::sync::Arc;

use anyhow::{Result, bail};
use cudarc::driver::CudaSlice;
use half::bf16;

use crate::page_pool::{OwnedPagePermit, PageId, PagePool};
use crate::tensor::DeviceContext;

/// Page-first geometry: dimensions and derived strides for one page.
///
/// Pure value type — no GPU, no allocation. Shared by `KvPool`, `KvState`, `KvDesc`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KvLayout {
    pub(crate) page_size: usize,
    pub(crate) num_layers: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    /// Elements in one K (or V) block: page_size × num_kv_heads × head_dim.
    pub(crate) kv_block_len: usize,
    /// Elements between layers within a page: 2 × kv_block_len (K then V).
    pub(crate) layer_stride: usize,
    /// Elements per page (all layers): num_layers × layer_stride.
    pub(crate) page_stride: usize,
}

impl KvLayout {
    pub(crate) fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
    ) -> Self {
        let kv_block_len = page_size * num_kv_heads * head_dim;
        let layer_stride = 2 * kv_block_len;
        let page_stride = num_layers * layer_stride;
        Self {
            page_size,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_block_len,
            layer_stride,
            page_stride,
        }
    }
}

struct KvPoolInner {
    pool: PagePool,
    buffer: CudaSlice<bf16>,
    layout: KvLayout,
}

/// Shared KV backing storage with page-based allocation.
///
/// Owns a single GPU buffer in page-first layout: each page is a contiguous
/// chunk holding all layers' K and V for `page_size` tokens.
///
/// ```text
/// page_i: [L0_K | L0_V | L1_K | L1_V | ... | Ln_K | Ln_V]
///          └─ page_size × num_kv_heads × head_dim each ─┘
/// ```
///
/// Cheaply clonable (Arc). Multiple `KvState`s share the same pool.
#[derive(Clone)]
pub(crate) struct KvPool {
    inner: Arc<KvPoolInner>,
}

impl KvPool {
    pub(crate) fn new(
        ctx: &DeviceContext,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        num_pages: usize,
    ) -> Result<Self> {
        let layout = KvLayout::new(num_layers, num_kv_heads, head_dim, page_size);
        let total_elements = num_pages * layout.page_stride;

        let buffer: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(total_elements)
            .map_err(|e| anyhow::anyhow!("KvPool alloc failed: {e}"))?;

        Ok(Self {
            inner: Arc::new(KvPoolInner {
                pool: PagePool::new(num_pages),
                buffer,
                layout,
            }),
        })
    }

    /// Create an empty per-request KV state.
    pub(crate) fn alloc(&self) -> KvState {
        KvState {
            permit: self.inner.pool.try_acquire_many(0).expect("zero acquire"),
            seq_len: 0,
            pool: self.clone(),
        }
    }

    pub(crate) fn layout(&self) -> &KvLayout {
        &self.inner.layout
    }

    pub(crate) fn capacity_pages(&self) -> usize {
        self.inner.pool.capacity_pages()
    }

    pub(crate) fn available_pages(&self) -> usize {
        self.inner.pool.available_pages()
    }
}

/// Per-request KV state. Parallels `RecurrentState` for linear attention.
///
/// Holds an RAII page permit that auto-returns pages on drop.
pub(crate) struct KvState {
    permit: OwnedPagePermit,
    seq_len: usize,
    pool: KvPool,
}

/// Pages needed to hold `token_count` tokens with the given `page_size`.
fn pages_needed(token_count: usize, page_size: usize) -> usize {
    token_count.div_ceil(page_size)
}

impl KvState {
    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Ensure capacity for at least `token_count` tokens total.
    pub(crate) fn ensure_capacity(&mut self, token_count: usize) -> Result<()> {
        let needed = pages_needed(token_count, self.pool.inner.layout.page_size);
        let held = self.permit.len();
        if needed <= held {
            return Ok(());
        }
        let grow = needed - held;
        if !self.permit.try_grow(grow) {
            bail!(
                "KvState: out of pages (need {grow} more, {} available)",
                self.pool.available_pages()
            );
        }
        Ok(())
    }

    /// Advance sequence length after writing tokens.
    pub(crate) fn advance(&mut self, count: usize) {
        self.seq_len += count;
    }

    /// Build kernel-facing metadata for this request's KV.
    pub(crate) fn desc(&self) -> KvDesc<'_> {
        let pages = self.permit.pages();
        let last_page_len = if self.seq_len == 0 {
            0
        } else {
            let rem = self.seq_len % self.pool.inner.layout.page_size;
            if rem == 0 {
                self.pool.inner.layout.page_size
            } else {
                rem
            }
        };
        KvDesc {
            layout: self.pool.inner.layout,
            buffer: &self.pool.inner.buffer,
            pages,
            seq_len: self.seq_len,
            last_page_len,
        }
    }

    /// Reset for a new request: return all pages, zero seq_len.
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

/// Kernel-facing metadata describing this request's paged KV layout.
///
/// Bundles pool geometry (strides) with request state (page list, seq_len).
/// Forward code passes this opaquely to FFI calls.
pub(crate) struct KvDesc<'a> {
    layout: KvLayout,
    buffer: &'a CudaSlice<bf16>,
    pages: &'a [PageId],
    seq_len: usize,
    last_page_len: usize,
}

impl KvDesc<'_> {
    pub(crate) fn layout(&self) -> &KvLayout {
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

    /// Page indices for this request (CPU-side).
    pub(crate) fn page_indices(&self) -> &[PageId] {
        self.pages
    }

    /// The backing buffer. Kernel integration will extract device pointers
    /// via `buffer.device_ptr(&stream)`.
    pub(crate) fn buffer(&self) -> &CudaSlice<bf16> {
        self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pool(page_size: usize, num_pages: usize) -> KvPool {
        let ctx = DeviceContext::new().expect("GPU required for kv_pool tests");
        // Tiny allocation: 1 layer, 1 head, dim=1 — just enough to exercise logic.
        KvPool::new(&ctx, 1, 1, 1, page_size, num_pages).expect("KvPool::new failed")
    }

    #[test]
    fn stride_geometry_qwen35() {
        // Qwen3.5-4B: 8 full attn layers, 4 KV heads, head_dim=128, page_size=16
        let l = KvLayout::new(8, 4, 128, 16);

        assert_eq!(l.kv_block_len, 16 * 4 * 128); // 8192 elements
        assert_eq!(l.layer_stride, 2 * 8192); // 16384 (K + V)
        assert_eq!(l.page_stride, 8 * 16384); // 131072 per page
        // bytes: 131072 * 2 = 262144 = 256 KB per page
        assert_eq!(l.page_stride * 2, 256 * 1024);
    }

    #[test]
    fn stride_geometry_qwen3() {
        // Qwen3-4B: 36 full attn layers, 4 KV heads, head_dim=128, page_size=16
        let l = KvLayout::new(36, 4, 128, 16);

        // 36 layers × 16384 = 589824 elements per page
        assert_eq!(l.page_stride, 589_824);
        // bytes: ~1.125 MB per page
    }

    #[test]
    fn kv_state_lifecycle() {
        let pool = test_pool(16, 4); // page_size=16, 4 pages → 64 tokens max
        let mut kv = pool.alloc();

        // Starts empty
        assert_eq!(kv.seq_len(), 0);
        let desc = kv.desc();
        assert_eq!(desc.num_pages(), 0);
        assert_eq!(desc.last_page_len(), 0);

        // Prefill 10 tokens: needs 1 page
        kv.ensure_capacity(10).unwrap();
        kv.advance(10);
        assert_eq!(kv.seq_len(), 10);
        let desc = kv.desc();
        assert_eq!(desc.num_pages(), 1);
        assert_eq!(desc.last_page_len(), 10);
        assert_eq!(pool.available_pages(), 3);

        // Decode to 16: still 1 page, last_page_len=16 (full)
        for _ in 10..16 {
            kv.ensure_capacity(kv.seq_len() + 1).unwrap();
            kv.advance(1);
        }
        let desc = kv.desc();
        assert_eq!(desc.num_pages(), 1);
        assert_eq!(desc.last_page_len(), 16);

        // One more token crosses page boundary: 2 pages
        kv.ensure_capacity(kv.seq_len() + 1).unwrap();
        kv.advance(1);
        assert_eq!(kv.seq_len(), 17);
        let desc = kv.desc();
        assert_eq!(desc.num_pages(), 2);
        assert_eq!(desc.last_page_len(), 1);
        assert_eq!(pool.available_pages(), 2);

        // Reset returns all pages
        kv.reset();
        assert_eq!(kv.seq_len(), 0);
        assert_eq!(pool.available_pages(), 4);
    }

    #[test]
    fn kv_state_out_of_pages() {
        let pool = test_pool(16, 2); // 2 pages → 32 tokens max
        let mut kv = pool.alloc();

        kv.ensure_capacity(32).unwrap(); // exactly fits
        assert!(kv.ensure_capacity(33).is_err()); // needs 3rd page
    }

    #[test]
    fn kv_state_drop_returns_pages() {
        let pool = test_pool(16, 4);
        {
            let mut kv = pool.alloc();
            kv.ensure_capacity(20).unwrap(); // 2 pages
            kv.advance(20);
            assert_eq!(pool.available_pages(), 2);
        }
        // Drop returns pages
        assert_eq!(pool.available_pages(), 4);
    }
}
