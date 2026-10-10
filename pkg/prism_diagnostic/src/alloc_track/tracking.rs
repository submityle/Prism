//! Header-free [`GlobalAlloc`] wrapper (zero per-allocation overhead).
//!
//! [`TrackingAllocator`] accounts every allocation and deallocation into the
//! shared process-global counters (live/peak/cumulative bytes, alloc/free
//! counts) and attributes *cumulative* bytes to the active per-callsite tag.
//! Because [`GlobalAlloc::dealloc`] is handed the original [`Layout`], the byte
//! accounting is *exact and symmetric* — not an estimate. It adds no bytes to
//! the allocation itself, so it is the zero-cost default; precise per-tag
//! **live** residency is provided by the sibling
//! [`LiveTrackingAllocator`](super::live::LiveTrackingAllocator).

#![expect(
    unsafe_code,
    reason = "implementing GlobalAlloc requires unsafe; this wrapper only \
              forwards to the inner allocator and touches atomics"
)]

use core::alloc::{GlobalAlloc, Layout};
use std::alloc::System;

use super::{record_alloc, record_free};

/// A [`GlobalAlloc`] wrapper that accounts allocations into process-global
/// counters while delegating the actual memory work to an inner allocator
/// (defaulting to the system allocator).
///
/// Install it as the program allocator:
///
/// ```ignore
/// use prism_diagnostic::alloc_track::TrackingAllocator;
/// use std::alloc::System;
///
/// #[global_allocator]
/// static GLOBAL: TrackingAllocator<System> = TrackingAllocator::new(System);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct TrackingAllocator<A: GlobalAlloc = System> {
    inner: A,
}

impl<A: GlobalAlloc> TrackingAllocator<A> {
    /// Wrap `inner`, accounting all traffic routed through it.
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }

    /// Borrow the wrapped allocator.
    pub fn inner(&self) -> &A {
        &self.inner
    }
}

// SAFETY: `TrackingAllocator` forwards every method to `self.inner`, a correct
// `GlobalAlloc`, with identical pointers and layouts. The accounting around the
// delegate calls only touches atomics and a thread-local `Cell` — it never
// allocates, deallocates, or reenters the allocator — so it cannot violate the
// `GlobalAlloc` contract. Returned pointers and alignment are exactly those the
// inner allocator produced.
unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is a valid non-zero layout per the caller contract;
        // forwarded unchanged to the inner allocator.
        let ptr = unsafe { self.inner.alloc(layout) };
        if !ptr.is_null() {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as `alloc`; forwarded unchanged.
        let ptr = unsafe { self.inner.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was returned by a previous `alloc`/`realloc` of this
        // allocator with this exact `layout`; forwarded unchanged to free it.
        unsafe { self.inner.dealloc(ptr, layout) };
        record_free(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr`/`layout` describe a current allocation of this
        // allocator and `new_size` is a valid size per the caller contract;
        // forwarded unchanged.
        let new_ptr = unsafe { self.inner.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            // On success the old block (of `layout.size()`) is released and a
            // `new_size` block is live.
            record_free(layout.size());
            record_alloc(new_size);
        }
        new_ptr
    }
}
