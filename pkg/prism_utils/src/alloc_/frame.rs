//! A linear bump / frame allocator ([`FrameAllocator`]).
//!
//! A frame allocator owns one contiguous backing buffer and serves requests by
//! bumping a cursor forward. Individual blocks are never freed one at a time;
//! instead [`FrameAllocator::reset`] rewinds the cursor to the start in `O(1)`,
//! reclaiming the entire frame at once. This is the standard per-frame scratch
//! allocator ("帧 reset") used for transient data that lives for exactly one
//! simulation/render frame.
//!
//! ## Lifetime contract
//! Because [`reset`](FrameAllocator::reset) takes `&mut self`, the borrow
//! checker guarantees no shared `&self` allocation borrow is outstanding across
//! a reset — satisfying the roadmap's "reset 前所有借用必须结束" invariant for
//! allocator-aware containers that borrow the frame.

extern crate alloc;

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::NonNull;

use super::{AllocError, Allocator};

/// Default backing-buffer alignment: enough for `u128`/SIMD-ish scalars.
const DEFAULT_ALIGN: usize = 16;

/// A linear bump allocator over a fixed backing buffer.
///
/// Allocation advances an internal cursor (kept in a [`Cell`] so allocation
/// works through `&self`); [`reset`](FrameAllocator::reset) rewinds it. The
/// backing buffer is freed when the allocator is dropped.
pub struct FrameAllocator {
    /// Base of the backing buffer.
    buf: NonNull<u8>,
    /// Total capacity of the backing buffer in bytes.
    capacity: usize,
    /// Exact [`Layout`] used to allocate (and later free) the buffer.
    layout: Layout,
    /// Bytes handed out so far, measured from `buf`.
    cursor: Cell<usize>,
}

impl FrameAllocator {
    /// Create a frame allocator with `capacity` bytes, aligned for scalars up
    /// to 16 bytes.
    ///
    /// # Panics
    /// Panics if the backing buffer cannot be allocated (out of memory) or if
    /// `capacity` overflows a valid [`Layout`].
    pub fn new(capacity: usize) -> Self {
        Self::with_align(capacity, DEFAULT_ALIGN)
    }

    /// Create a frame allocator with `capacity` bytes whose backing buffer is
    /// aligned to at least `align` (rounded up to a power of two as required by
    /// [`Layout`]).
    ///
    /// # Panics
    /// Panics if the buffer cannot be allocated or the layout is invalid.
    pub fn with_align(capacity: usize, align: usize) -> Self {
        let align = align.max(1);
        let layout = Layout::from_size_align(capacity.max(1), align)
            .expect("FrameAllocator: invalid capacity/alignment");

        #[expect(
            unsafe_code,
            reason = "the backing buffer comes from the global allocator"
        )]
        // SAFETY: `layout` has non-zero size (`capacity.max(1)`), satisfying the
        // precondition of `alloc`.
        let raw = unsafe { alloc::alloc::alloc(layout) };
        let Some(buf) = NonNull::new(raw) else {
            alloc::alloc::handle_alloc_error(layout)
        };

        Self {
            buf,
            capacity,
            layout,
            cursor: Cell::new(0),
        }
    }

    /// Total backing capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes currently handed out since the last [`reset`](Self::reset).
    pub fn used(&self) -> usize {
        self.cursor.get()
    }

    /// Bytes still available before the frame is exhausted.
    pub fn remaining(&self) -> usize {
        self.capacity - self.cursor.get()
    }

    /// Reclaim the entire frame in `O(1)`.
    ///
    /// Taking `&mut self` ensures (via the borrow checker) that no outstanding
    /// shared borrow of this allocator survives the reset. The backing memory
    /// is retained for reuse; only the cursor is rewound. Any raw pointers
    /// previously handed out are logically invalidated.
    pub fn reset(&mut self) {
        self.cursor.set(0);
    }
}

impl Allocator for FrameAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            let ptr =
                NonNull::new(core::ptr::without_provenance_mut(layout.align())).ok_or(AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(ptr, 0));
        }

        let align = layout.align();
        let base_addr = self.buf.as_ptr() as usize;
        let cursor = self.cursor.get();

        // Align the *absolute* address of the next block so correctness does
        // not depend on the backing buffer's own alignment exceeding `align`.
        let current_addr = base_addr.checked_add(cursor).ok_or(AllocError)?;
        let aligned_addr = current_addr
            .checked_add(align - 1)
            .ok_or(AllocError)?
            & !(align - 1);
        let padding = aligned_addr - current_addr;

        let new_cursor = cursor
            .checked_add(padding)
            .and_then(|c| c.checked_add(layout.size()))
            .ok_or(AllocError)?;
        if new_cursor > self.capacity {
            return Err(AllocError);
        }

        let block_offset = cursor + padding;
        #[expect(
            unsafe_code,
            reason = "bumping the cursor requires offsetting into the backing buffer"
        )]
        // SAFETY: `block_offset <= new_cursor <= capacity`, so the pointer lands
        // within the backing buffer; `buf` is non-null, so the result is too.
        let ptr = unsafe { NonNull::new_unchecked(self.buf.as_ptr().add(block_offset)) };
        self.cursor.set(new_cursor);
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    #[expect(
        unsafe_code,
        reason = "a bump allocator reclaims in bulk via reset, so per-block free is a no-op"
    )]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Intentionally empty: a frame allocator reclaims all blocks at once in
        // `reset`. Individual deallocation is a no-op.
    }
}

impl Drop for FrameAllocator {
    fn drop(&mut self) {
        #[expect(
            unsafe_code,
            reason = "the backing buffer must be returned to the global allocator"
        )]
        // SAFETY: `buf` was allocated in `with_align` with exactly `self.layout`
        // via `alloc::alloc::alloc`, and is freed exactly once here.
        unsafe {
            alloc::alloc::dealloc(self.buf.as_ptr(), self.layout);
        }
    }
}
