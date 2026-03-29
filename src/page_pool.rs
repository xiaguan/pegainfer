use std::sync::Arc;

use parking_lot::Mutex;

/// Shared fixed-page allocator with RAII-owned permits.
#[derive(Clone, Debug)]
pub(crate) struct PagePool {
    inner: Arc<PagePoolInner>,
}

#[derive(Debug)]
struct PagePoolInner {
    capacity_pages: usize,
    free_list: Mutex<Vec<PageId>>,
}

/// A set of pages leased from a [`PagePool`]. Pages return automatically on drop.
#[derive(Debug)]
pub(crate) struct OwnedPagePermit {
    inner: Arc<PagePoolInner>,
    pages: Vec<PageId>,
}

/// Opaque page identifier within a pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PageId(usize);

impl PagePool {
    pub(crate) fn new(capacity_pages: usize) -> Self {
        let free_list = (0..capacity_pages).rev().map(PageId).collect();
        Self {
            inner: Arc::new(PagePoolInner {
                capacity_pages,
                free_list: Mutex::new(free_list),
            }),
        }
    }

    pub(crate) fn try_acquire_many(&self, n: usize) -> Option<OwnedPagePermit> {
        if n == 0 {
            return Some(OwnedPagePermit {
                inner: Arc::clone(&self.inner),
                pages: Vec::new(),
            });
        }

        let mut free_list = self.inner.free_list.lock();
        if free_list.len() < n {
            return None;
        }

        let mut pages = Vec::with_capacity(n);
        for _ in 0..n {
            pages.push(free_list.pop().expect("free_list length checked above"));
        }

        Some(OwnedPagePermit {
            inner: Arc::clone(&self.inner),
            pages,
        })
    }

    pub(crate) fn capacity_pages(&self) -> usize {
        self.inner.capacity_pages
    }

    pub(crate) fn available_pages(&self) -> usize {
        self.inner.free_list.lock().len()
    }
}

impl OwnedPagePermit {
    pub(crate) fn pages(&self) -> &[PageId] {
        &self.pages
    }

    pub(crate) fn len(&self) -> usize {
        self.pages.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

impl Drop for OwnedPagePermit {
    fn drop(&mut self) {
        if self.pages.is_empty() {
            return;
        }

        let pages = std::mem::take(&mut self.pages);
        let mut free_list = self.inner.free_list.lock();
        free_list.extend(pages.into_iter().rev());
    }
}

#[cfg(test)]
mod tests {
    use super::{PageId, PagePool};

    #[test]
    fn zero_capacity_pool_never_acquires_pages() {
        let pool = PagePool::new(0);

        assert_eq!(pool.capacity_pages(), 0);
        assert_eq!(pool.available_pages(), 0);
        assert!(pool.try_acquire_many(1).is_none());
        assert!(
            pool.try_acquire_many(0)
                .expect("empty acquire should succeed")
                .is_empty()
        );
    }

    #[test]
    fn acquires_pages_and_restores_on_drop() {
        let pool = PagePool::new(4);
        assert_eq!(pool.capacity_pages(), 4);

        {
            let permit = pool.try_acquire_many(2).expect("acquire should succeed");
            assert_eq!(permit.len(), 2);
            assert_eq!(permit.pages(), &[PageId(0), PageId(1)]);
            assert_eq!(pool.available_pages(), 2);

            let permit_b = pool
                .try_acquire_many(1)
                .expect("second acquire should succeed");
            assert_eq!(permit_b.pages(), &[PageId(2)]);
            assert_eq!(pool.available_pages(), 1);
        }

        assert_eq!(pool.available_pages(), 4);
        let permit = pool
            .try_acquire_many(4)
            .expect("released pages should be reusable");
        assert_eq!(
            permit.pages(),
            &[PageId(0), PageId(1), PageId(2), PageId(3)]
        );
    }

    #[test]
    fn reports_insufficient_capacity_with_none() {
        let pool = PagePool::new(2);
        let _permit = pool.try_acquire_many(1).expect("acquire should succeed");

        assert!(pool.try_acquire_many(2).is_none());
    }
}
