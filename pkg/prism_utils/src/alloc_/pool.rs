//! A fixed-size-block pool allocator ([`Pool`]).
//!
//! A pool hands out blocks that all share one [`Layout`]. Free blocks are
//! threaded onto an intrusive singly-linked free list stored *inside* the free
//! blocks themselves, so both allocation and deallocation are `O(1)` with no
//! per-block bookkeeping overhead. When the free list is empty a new backing
//! *chunk* of `blocks_per_chunk` blocks is carved from the global heap and its
//! blocks are pushed onto the free list.
//!
//! This is the classic "池 vs malloc" allocator from the roadmap: freeing a
//! block merely re-links it, and the next allocation of the same size reuses
//! that exact block, avoiding repeated trips through the global allocator.

extern crate alloc;

use alloc::vec::Vec;
use core::alloc::Layout;
use core::cell::RefCell;
use core::ptr::NonNull;

use super::{AllocError, Allocator};

/// A backing allocation carved into same-sized blocks.
struct Chunk {
    /// Base pointer of the chunk allocation.
    ptr: NonNull<u8>,
    /// The exact [`Layout`] used to allocate (and later free) the chunk.
    layout: Layout,
}

/// Interior, mutable bookkeeping guarded by a [`RefCell`] so [`Pool`] can
/// satisfy allocations through a shared `&self` reference.
struct PoolInner {
    /// Head of the intrusive free list, or `None` when exhausted.
    free_head: Option<NonNull<u8>>,
    /// Backing chunks, freed in bulk when the pool is dropped.
    chunks: Vec<Chunk>,
    /// Number of blocks currently handed out.
    live: usize,
}

/// A fixed-size-block pool allocator with `O(1)` allocate/free and automatic
/// chunk growth.
///
/// Every block satisfies the normalized block [`Layout`] chosen at
/// construction: its size is at least the requested size (and at least large
/// enough to hold the free-list link) and its alignment is at least the
/// requested alignment (and at least pointer alignment).
pub struct Pool {
    /// Per-block stride in bytes (also the block size and alignment-rounded).
    block_size: usize,
    /// Alignment shared by every block.
    block_align: usize,
    /// How many blocks each freshly grown chunk contains.
    blocks_per_chunk: usize,
    /// Mutable state behind interior mutability.
    inner: RefCell<PoolInner>,
}

impl Pool {
    /// Create a pool whose blocks satisfy `block` and that grows
    /// `blocks_per_chunk` blocks at a time.
    ///
    /// The effective block size/alignment are widened as needed so each block
    /// can hold the intrusive free-list pointer. `blocks_per_chunk` is clamped
    /// to at least `1`.
    pub fn new(block: Layout, blocks_per_chunk: usize) -> Self {
        let link_size = size_of::<*mut u8>();
        let link_align = align_of::<*mut u8>();

        let block_align = block.align().max(link_align);
        let raw_size = block.size().max(link_size).max(1);
        // Round the stride up to the alignment so every block in a chunk starts
        // on an aligned boundary.
        let block_size = raw_size.next_multiple_of(block_align);

        Self {
            block_size,
            block_align,
            blocks_per_chunk: blocks_per_chunk.max(1),
            inner: RefCell::new(PoolInner {
                free_head: None,
                chunks: Vec::new(),
                live: 0,
            }),
        }
    }

    /// The per-block stride in bytes (the usable block size).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// The alignment every block satisfies.
    pub fn block_align(&self) -> usize {
        self.block_align
    }

    /// Number of blocks currently handed out (not yet returned).
    pub fn live(&self) -> usize {
        self.inner.borrow().live
    }

    /// Number of backing chunks allocated so far.
    pub fn chunk_count(&self) -> usize {
        self.inner.borrow().chunks.len()
    }

    /// Whether `layout` can be served from this pool's blocks.
    pub fn fits(&self, layout: Layout) -> bool {
        layout.size() <= self.block_size && layout.align() <= self.block_align
    }

    /// Grow the pool by one chunk, pushing its blocks onto the free list.
    fn grow(&self, inner: &mut PoolInner) -> Result<(), AllocError> {
        let chunk_bytes = self
            .block_size
            .checked_mul(self.blocks_per_chunk)
            .ok_or(AllocError)?;
        let layout =
            Layout::from_size_align(chunk_bytes, self.block_align).map_err(|_| AllocError)?;

        #[expect(
            unsafe_code,
            reason = "chunk backing memory comes from the global allocator"
        )]
        // SAFETY: `chunk_bytes >= block_size >= 1`, so the layout is non-zero,
        // satisfying `alloc`'s precondition.
        let raw = unsafe { alloc::alloc::alloc(layout) };
        let base = NonNull::new(raw).ok_or(AllocError)?;

        // Thread each block onto the free list. Iterating forward keeps the
        // lowest-address block at the head, which makes reuse order easy to
        // reason about in tests.
        for i in (0..self.blocks_per_chunk).rev() {
            let offset = i * self.block_size;
            #[expect(
                unsafe_code,
                reason = "carving the chunk into blocks requires pointer arithmetic"
            )]
            // SAFETY: `offset < chunk_bytes`, so the resulting pointer stays
            // within the freshly allocated chunk; `base` is non-null, hence so
            // is the offset pointer.
            let block = unsafe { NonNull::new_unchecked(base.as_ptr().add(offset)) };
            #[expect(
                unsafe_code,
                reason = "the free list is stored intrusively inside free blocks"
            )]
            // SAFETY: `block` is aligned to `block_align >= align_of::<*mut u8>()`
            // and owns at least `block_size >= size_of::<*mut u8>()` bytes, so
            // writing the free-list link there is in-bounds and well-aligned.
            unsafe {
                write_link(block, inner.free_head);
            }
            inner.free_head = Some(block);
        }

        inner.chunks.push(Chunk { ptr: base, layout });
        Ok(())
    }

    /// Allocate one block, growing the pool if the free list is empty.
    pub fn allocate_block(&self) -> Result<NonNull<u8>, AllocError> {
        let mut inner = self.inner.borrow_mut();
        let head = match inner.free_head {
            Some(head) => head,
            None => {
                self.grow(&mut inner)?;
                inner.free_head.ok_or(AllocError)?
            }
        };
        #[expect(
            unsafe_code,
            reason = "popping the free list reads the intrusive link"
        )]
        // SAFETY: `head` came from the free list, where every node stores a
        // valid link written by `write_link`, so reading it back is sound.
        let next = unsafe { read_link(head) };
        inner.free_head = next;
        inner.live += 1;
        Ok(head)
    }

    /// Return a block previously obtained from [`allocate_block`].
    ///
    /// # Safety
    /// `ptr` must have come from `allocate_block` on *this* pool and must not
    /// have been freed already.
    #[expect(
        unsafe_code,
        reason = "returning a block re-links caller-owned memory into the free list"
    )]
    pub unsafe fn deallocate_block(&self, ptr: NonNull<u8>) {
        let mut inner = self.inner.borrow_mut();
        #[expect(
            unsafe_code,
            reason = "pushing onto the free list writes the intrusive link"
        )]
        // SAFETY: `ptr` is a live block from this pool (caller contract); it is
        // therefore aligned and large enough to hold the free-list link.
        unsafe {
            write_link(ptr, inner.free_head);
        }
        inner.free_head = Some(ptr);
        inner.live -= 1;
    }
}

impl Allocator for Pool {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            let ptr =
                NonNull::new(core::ptr::without_provenance_mut(layout.align())).ok_or(AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(ptr, 0));
        }
        if !self.fits(layout) {
            return Err(AllocError);
        }
        let ptr = self.allocate_block()?;
        Ok(NonNull::slice_from_raw_parts(ptr, self.block_size))
    }

    #[expect(
        unsafe_code,
        reason = "bridges the trait contract to the pool's block free list"
    )]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        // SAFETY: a non-zero-size `ptr` from `Pool::allocate` is exactly a block
        // from `allocate_block`, so forwarding it to `deallocate_block` upholds
        // that method's contract.
        unsafe {
            self.deallocate_block(ptr);
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let inner = self.inner.get_mut();
        for chunk in inner.chunks.drain(..) {
            #[expect(
                unsafe_code,
                reason = "chunk backing memory must be returned to the global allocator"
            )]
            // SAFETY: each `chunk` was allocated in `grow` with exactly
            // `chunk.layout` via `alloc::alloc::alloc`, and chunks are freed
            // exactly once here as the pool is torn down.
            unsafe {
                alloc::alloc::dealloc(chunk.ptr.as_ptr(), chunk.layout);
            }
        }
    }
}

/// Write the intrusive free-list link `next` into the block at `block`.
///
/// # Safety
/// `block` must point to a block that is at least `size_of::<*mut u8>()` bytes
/// and aligned to at least `align_of::<*mut u8>()`.
#[expect(
    unsafe_code,
    reason = "the free list is stored intrusively inside free blocks"
)]
unsafe fn write_link(block: NonNull<u8>, next: Option<NonNull<u8>>) {
    let slot: *mut *mut u8 = block.as_ptr().cast();
    let raw = match next {
        Some(p) => p.as_ptr(),
        None => core::ptr::null_mut(),
    };
    // SAFETY: by the contract `slot` is aligned for and large enough to hold a
    // `*mut u8`, and the pool owns this memory while it sits on the free list.
    unsafe {
        slot.write(raw);
    }
}

/// Read the intrusive free-list link out of the block at `block`.
///
/// # Safety
/// `block` must point to a block previously initialized by [`write_link`].
#[expect(
    unsafe_code,
    reason = "the free list is stored intrusively inside free blocks"
)]
unsafe fn read_link(block: NonNull<u8>) -> Option<NonNull<u8>> {
    let slot: *mut *mut u8 = block.as_ptr().cast();
    // SAFETY: by the contract `slot` holds a `*mut u8` previously written by
    // `write_link`, so reading it back yields that same (possibly null) link.
    let raw = unsafe { slot.read() };
    NonNull::new(raw)
}
