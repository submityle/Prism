//! A scoped bump-allocator stack ([`ScopeStack`]).
//!
//! Where [`FrameAllocator`](super::frame::FrameAllocator) reclaims its *entire*
//! buffer in one [`reset`](super::frame::FrameAllocator::reset), a scope stack
//! reclaims memory in **nested LIFO scopes**. Entering a scope records the
//! current cursor; leaving it (by dropping the returned [`Scope`] guard, or by
//! calling [`ScopeStack::rewind`]) rewinds the cursor to that mark in `O(1)`,
//! freeing every block allocated inside the scope at once with zero per-object
//! destructor cost.
//!
//! This matches the roadmap's "进入作用域 push 一个 bump 竞技场，退出 pop 整体
//! reset（嵌套安全）": a frame opens an outer scope, a system opens an inner
//! scope for its intermediate results, and so on, each nested lifetime peeling
//! off cleanly.
//!
//! ```
//! use prism_utils::alloc_::{Allocator, ScopeStack};
//! use core::alloc::Layout;
//!
//! let stack = ScopeStack::new(4096);
//! let l = Layout::from_size_align(64, 16).unwrap();
//!
//! let outer = stack.scope();
//! outer.allocate(l).unwrap();
//! {
//!     let inner = stack.scope();
//!     inner.allocate(l).unwrap();
//!     assert!(stack.used() >= 128);
//! } // `inner` drops: the inner 64 bytes are reclaimed.
//! assert!(stack.used() >= 64 && stack.used() < 128);
//! drop(outer); // everything is reclaimed.
//! assert_eq!(stack.used(), 0);
//! ```
//!
//! ## Lifetime / safety contract
//! Like every bump allocator here, [`ScopeStack`] hands out raw `NonNull<[u8]>`
//! pointers that are **not** lifetime-tied to the allocator, and its
//! [`deallocate`](Allocator::deallocate) is a no-op. Rewinding a scope
//! *logically invalidates* every pointer allocated after its mark, exactly as
//! `FrameAllocator::reset` does. Callers must not use those pointers past the
//! scope that produced them. Because [`Scope`] guards live on the stack, Rust's
//! drop order gives correct LIFO rewinding for the common nested-scope pattern;
//! out-of-order teardown is still memory-safe (a rewind never advances the
//! cursor) but reclaims eagerly, so prefer dropping inner scopes first.

extern crate alloc;

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::NonNull;

use super::{AllocError, Allocator};

/// Default backing-buffer alignment: enough for `u128` / SIMD-ish scalars.
const DEFAULT_ALIGN: usize = 16;

/// An opaque mark into a [`ScopeStack`] captured when a scope is entered.
///
/// Produced by [`ScopeStack::mark`] and consumed by [`ScopeStack::rewind`] for
/// callers that want manual (guard-free) control. The [`Scope`] RAII guard
/// wraps a mark and rewinds automatically on drop.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[must_use = "a ScopeMark does nothing until passed to ScopeStack::rewind"]
pub struct ScopeMark(usize);

impl ScopeMark {
    /// The byte offset this mark rewinds the stack to.
    #[must_use]
    pub fn offset(self) -> usize {
        self.0
    }
}

/// A nestable bump allocator whose memory is reclaimed in LIFO scopes.
///
/// Allocation advances an internal cursor (kept in a [`Cell`] so it works
/// through `&self`, allowing many [`Scope`] guards and containers to share one
/// stack). Entering a scope captures the cursor; leaving it rewinds the cursor.
pub struct ScopeStack {
    /// Base of the backing buffer.
    buf: NonNull<u8>,
    /// Total capacity of the backing buffer in bytes.
    capacity: usize,
    /// Exact [`Layout`] used to allocate (and later free) the buffer.
    layout: Layout,
    /// Bytes handed out so far, measured from `buf`.
    cursor: Cell<usize>,
    /// Largest `cursor` ever reached, for watermark/budget diagnostics.
    high_water: Cell<usize>,
}

impl ScopeStack {
    /// Create a scope stack with `capacity` bytes, aligned for scalars up to 16
    /// bytes.
    ///
    /// # Panics
    /// Panics if the backing buffer cannot be allocated (out of memory) or if
    /// `capacity` overflows a valid [`Layout`].
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self::with_align(capacity, DEFAULT_ALIGN)
    }

    /// Create a scope stack with `capacity` bytes whose backing buffer is
    /// aligned to at least `align` (rounded up to a power of two as required by
    /// [`Layout`]).
    ///
    /// # Panics
    /// Panics if the buffer cannot be allocated or the layout is invalid.
    #[must_use]
    pub fn with_align(capacity: usize, align: usize) -> Self {
        let align = align.max(1);
        let layout = Layout::from_size_align(capacity.max(1), align)
            .expect("ScopeStack: invalid capacity/alignment");

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
            high_water: Cell::new(0),
        }
    }

    /// Total backing capacity in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes currently handed out across all open scopes.
    #[must_use]
    pub fn used(&self) -> usize {
        self.cursor.get()
    }

    /// Bytes still available before the stack is exhausted.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.capacity - self.cursor.get()
    }

    /// The largest number of bytes ever simultaneously in use since creation.
    ///
    /// Useful for right-sizing the backing buffer: run a representative frame,
    /// then read the watermark to pick a capacity that never spills.
    #[must_use]
    pub fn high_water(&self) -> usize {
        self.high_water.get()
    }

    /// Capture the current cursor as a [`ScopeMark`] without opening a guard.
    ///
    /// Pair with [`rewind`](Self::rewind) for manual scope control. Most callers
    /// should prefer [`scope`](Self::scope), which rewinds automatically.
    pub fn mark(&self) -> ScopeMark {
        ScopeMark(self.cursor.get())
    }

    /// Rewind the cursor to a previously captured [`ScopeMark`], reclaiming
    /// every block allocated after it in `O(1)`.
    ///
    /// A rewind never advances the cursor: if `mark` lies *ahead* of the
    /// current cursor (which only happens under non-LIFO teardown) the cursor is
    /// left unchanged, keeping already-reclaimed memory free. The high-water
    /// mark is preserved.
    pub fn rewind(&self, mark: ScopeMark) {
        let target = mark.0.min(self.cursor.get());
        self.cursor.set(target);
    }

    /// Open a new RAII [`Scope`]: a guard that rewinds the stack to the current
    /// cursor when it drops.
    ///
    /// Nest scopes by calling this again while an outer guard is alive; drop the
    /// inner guard first for correct LIFO reclamation.
    #[must_use = "the returned Scope rewinds the stack when dropped; bind it to a name"]
    pub fn scope(&self) -> Scope<'_> {
        Scope {
            stack: self,
            mark: self.mark(),
        }
    }

    /// The shared allocation routine used by both [`ScopeStack`] and [`Scope`].
    fn bump(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            let ptr = NonNull::new(core::ptr::without_provenance_mut(layout.align()))
                .ok_or(AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(ptr, 0));
        }

        let align = layout.align();
        let base_addr = self.buf.as_ptr() as usize;
        let cursor = self.cursor.get();

        // Align the *absolute* address of the next block so correctness does not
        // depend on the backing buffer's own alignment exceeding `align`.
        let current_addr = base_addr.checked_add(cursor).ok_or(AllocError)?;
        let aligned_addr = current_addr.checked_add(align - 1).ok_or(AllocError)? & !(align - 1);
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
        if new_cursor > self.high_water.get() {
            self.high_water.set(new_cursor);
        }
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }
}

impl Allocator for ScopeStack {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.bump(layout)
    }

    #[expect(
        unsafe_code,
        reason = "a scope stack reclaims in bulk via rewind, so per-block free is a no-op"
    )]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Intentionally empty: memory is reclaimed per-scope via `rewind`.
    }
}

impl Drop for ScopeStack {
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

// A `ScopeStack` owns a heap buffer and uses interior mutability through a
// single-threaded `Cell`; it is intentionally neither `Send` nor `Sync`, which
// the `NonNull` field already enforces.

/// An RAII guard returned by [`ScopeStack::scope`] that rewinds the stack when
/// it drops.
///
/// Allocate through the guard (it implements [`Allocator`]) or directly through
/// the underlying [`ScopeStack`]; either way, dropping the guard reclaims every
/// block allocated after the guard was opened. Nest guards for nested
/// lifetimes.
#[must_use = "dropping the Scope immediately rewinds the stack; bind it to a name"]
pub struct Scope<'a> {
    stack: &'a ScopeStack,
    mark: ScopeMark,
}

impl Scope<'_> {
    /// Bytes allocated inside *this* scope so far (since it was opened).
    #[must_use]
    pub fn used(&self) -> usize {
        self.stack.cursor.get().saturating_sub(self.mark.0)
    }

    /// Bytes still available before the whole stack is exhausted.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.stack.remaining()
    }

    /// The backing [`ScopeStack`] this scope borrows, for opening nested scopes
    /// or querying totals.
    #[must_use]
    pub fn stack(&self) -> &ScopeStack {
        self.stack
    }

    /// Open a nested scope inside this one.
    #[must_use = "the returned Scope rewinds the stack when dropped; bind it to a name"]
    pub fn scope(&self) -> Scope<'_> {
        self.stack.scope()
    }
}

impl Allocator for Scope<'_> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.stack.bump(layout)
    }

    #[inline]
    #[expect(
        unsafe_code,
        reason = "forwards the no-op per-block free to the backing stack"
    )]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded unchanged to the backing stack, whose `deallocate`
        // is a no-op.
        unsafe { self.stack.deallocate(ptr, layout) }
    }
}

impl Drop for Scope<'_> {
    fn drop(&mut self) {
        self.stack.rewind(self.mark);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(size: usize) -> Layout {
        Layout::from_size_align(size, 16).unwrap()
    }

    #[test]
    fn nested_scopes_rewind_lifo() {
        let stack = ScopeStack::new(4096);
        assert_eq!(stack.used(), 0);

        let outer = stack.scope();
        outer.allocate(layout(64)).unwrap();
        let after_outer = stack.used();
        assert!(after_outer >= 64);

        {
            let inner = stack.scope();
            inner.allocate(layout(128)).unwrap();
            assert!(stack.used() >= after_outer + 128);
            assert!(inner.used() >= 128);
        } // inner drops

        assert_eq!(stack.used(), after_outer, "inner scope fully reclaimed");

        drop(outer);
        assert_eq!(stack.used(), 0, "outer scope fully reclaimed");
    }

    #[test]
    fn manual_mark_and_rewind() {
        let stack = ScopeStack::new(1024);
        let mark = stack.mark();
        stack.allocate(layout(200)).unwrap();
        assert!(stack.used() >= 200);
        stack.rewind(mark);
        assert_eq!(stack.used(), 0);
    }

    #[test]
    fn allocations_are_aligned_and_distinct() {
        let stack = ScopeStack::new(4096);
        let a = stack
            .allocate(Layout::from_size_align(1, 64).unwrap())
            .unwrap();
        let b = stack
            .allocate(Layout::from_size_align(1, 64).unwrap())
            .unwrap();
        assert_eq!(a.as_ptr() as *const u8 as usize % 64, 0);
        assert_eq!(b.as_ptr() as *const u8 as usize % 64, 0);
        assert_ne!(a.as_ptr() as *const u8, b.as_ptr() as *const u8);
    }

    #[test]
    fn exhaustion_returns_err_not_panic() {
        let stack = ScopeStack::new(128);
        assert!(stack.allocate(layout(64)).is_ok());
        // The next 128-byte block cannot fit in the remaining space.
        assert_eq!(stack.allocate(layout(128)), Err(AllocError));
    }

    #[test]
    fn zero_sized_request_never_bumps() {
        let stack = ScopeStack::new(64);
        let before = stack.used();
        let z = stack
            .allocate(Layout::from_size_align(0, 8).unwrap())
            .unwrap();
        assert_eq!(z.len(), 0);
        assert_eq!(stack.used(), before, "zero-sized request is free");
    }

    #[test]
    fn high_water_survives_rewind() {
        let stack = ScopeStack::new(4096);
        {
            let s = stack.scope();
            s.allocate(layout(512)).unwrap();
            assert!(stack.high_water() >= 512);
        }
        assert_eq!(stack.used(), 0);
        assert!(stack.high_water() >= 512, "watermark is not rewound");
    }

    #[test]
    fn out_of_order_rewind_never_advances_cursor() {
        // Defensive: capture an early mark, allocate, then rewind to a *later*
        // mark first — the cursor must not jump forward.
        let stack = ScopeStack::new(1024);
        let early = stack.mark();
        stack.allocate(layout(100)).unwrap();
        let late = stack.mark();
        stack.allocate(layout(100)).unwrap();

        stack.rewind(early); // jump back past both allocations
        let used = stack.used();
        assert_eq!(used, 0);
        stack.rewind(late); // `late` is ahead of the cursor now
        assert_eq!(stack.used(), 0, "rewind must never advance the cursor");
    }
}
