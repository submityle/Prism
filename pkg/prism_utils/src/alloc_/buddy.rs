//! A buddy sub-allocator over an abstract region: [`BuddyAllocator`].
//!
//! # What it is
//! A buddy allocator carves a power-of-two region into blocks whose sizes are
//! `min_block << order`. A request is served from the smallest order that fits;
//! a too-large free block is split in half repeatedly until it reaches that
//! order. On free, a block is merged ("coalesced") with its buddy whenever the
//! buddy is also free and of the same order, all the way back up. This bounds
//! external fragmentation and keeps alloc/free logarithmic in the order count.
//!
//! # Offsets, not pointers
//! This allocator manages *offsets* into a caller-owned backing region (host
//! RAM, a `GPU` heap, a streaming buffer); it never dereferences memory itself.
//! Bookkeeping is entirely index-based (free lists of block indices), so the
//! whole module is safe code with no `unsafe`. [`allocate`](BuddyAllocator::allocate)
//! returns a byte offset into the region, or `None` when the region cannot
//! satisfy the request.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

/// A buddy allocator over a region of `min_block << max_order` bytes.
///
/// All offsets and sizes are measured in bytes; internally blocks are tracked
/// in units of `min_block`.
#[derive(Debug)]
pub struct BuddyAllocator {
    /// Size of the smallest allocatable block, in bytes (a power of two).
    min_block: usize,
    /// The region spans `min_block << max_order` bytes.
    max_order: usize,
    /// `free[o]` holds the block indices (in `min_block` units) of the free
    /// blocks of order `o`; a block of order `o` spans `1 << o` units.
    free: Vec<BTreeSet<usize>>,
    /// Bytes currently handed out (sum of rounded block sizes).
    allocated: usize,
}

impl BuddyAllocator {
    /// Create a buddy allocator managing `min_block << max_order` bytes.
    ///
    /// `min_block` must be a power of two and non-zero.
    ///
    /// # Panics
    /// Panics if `min_block` is zero or not a power of two.
    #[must_use]
    pub fn new(min_block: usize, max_order: usize) -> Self {
        assert!(
            min_block.is_power_of_two(),
            "min_block must be a non-zero power of two"
        );
        let mut free: Vec<BTreeSet<usize>> = (0..=max_order).map(|_| BTreeSet::new()).collect();
        // The whole region starts as one free block at the top order.
        free[max_order].insert(0);
        Self {
            min_block,
            max_order,
            free,
            allocated: 0,
        }
    }

    /// Total managed size in bytes.
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.min_block << self.max_order
    }

    /// Bytes currently allocated (rounded up to block sizes).
    #[must_use]
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated
    }

    /// Bytes not currently allocated.
    #[must_use]
    #[inline]
    pub fn free_bytes(&self) -> usize {
        self.capacity() - self.allocated
    }

    /// The smallest block order whose block size is `>= size` bytes.
    fn order_for(&self, size: usize) -> usize {
        let units = size.div_ceil(self.min_block).max(1);
        units.next_power_of_two().trailing_zeros() as usize
    }

    /// Allocate `size` bytes, returning a byte offset into the region naturally
    /// aligned to its (power-of-two) block size, or `None` if there is no fit.
    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        if size == 0 {
            return None;
        }
        let order = self.order_for(size);
        if order > self.max_order {
            return None;
        }

        // Find the smallest available order at least `order`.
        let mut avail = order;
        while avail <= self.max_order && self.free[avail].is_empty() {
            avail += 1;
        }
        if avail > self.max_order {
            return None;
        }

        // Take a block and split it down to `order`, freeing the right halves.
        let block = *self.free[avail].iter().next()?;
        self.free[avail].remove(&block);
        while avail > order {
            avail -= 1;
            let buddy = block + (1 << avail);
            self.free[avail].insert(buddy);
        }
        self.allocated += self.min_block << order;
        Some(block * self.min_block)
    }

    /// Free a block previously returned by [`allocate`](Self::allocate).
    ///
    /// `size` must be the same size that was passed to the matching
    /// `allocate`; the allocator derives the block order from it (buddy
    /// allocators recover the order from the request size, exactly like
    /// `dealloc(ptr, layout)`).
    ///
    /// # Panics
    /// Panics if `offset` is not aligned to the block size implied by `size`,
    /// which indicates a mismatched or corrupt free.
    pub fn deallocate(&mut self, offset: usize, size: usize) {
        if size == 0 {
            return;
        }
        let order = self.order_for(size);
        debug_assert!(order <= self.max_order, "freed size exceeds region");
        let mut block = offset / self.min_block;
        assert!(
            offset.is_multiple_of(self.min_block) && block.is_multiple_of(1 << order),
            "deallocate: offset {offset} is not aligned to its block"
        );
        self.allocated -= self.min_block << order;

        // Coalesce with the buddy while it is free and of the same order.
        let mut order = order;
        while order < self.max_order {
            let buddy = block ^ (1 << order);
            if self.free[order].remove(&buddy) {
                block = block.min(buddy);
                order += 1;
            } else {
                break;
            }
        }
        self.free[order].insert(block);
    }
}
