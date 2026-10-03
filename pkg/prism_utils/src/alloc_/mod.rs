//! # Allocators (`alloc_`)
//!
//! M2 of `prism_utils`: a Prism-native allocator abstraction plus two concrete
//! allocators and one allocator-aware container.
//!
//! The module name is `alloc_` (with a trailing underscore) to avoid colliding
//! with the `alloc` crate that the whole kernel is built on.
//!
//! ## Why a Prism trait instead of `core::alloc::Allocator`
//! The standard-library [`core::alloc::Allocator`] trait is still unstable and
//! would force the entire engine onto a nightly toolchain. Instead this module
//! defines its own [`Allocator`] trait with the same essential shape
//! ([`allocate`](Allocator::allocate) returning a `NonNull<[u8]>` and
//! [`deallocate`](Allocator::deallocate)), so container code can be written
//! against a stable abstraction today and bridged to the std trait later.
//!
//! ## Pieces
//! - [`Allocator`]: the trait every Prism allocator implements.
//! - [`AllocError`]: the zero-sized failure type returned on exhaustion.
//! - [`Global`]: a bridge to the global heap via [`alloc::alloc`].
//! - [`Pool`](crate::alloc_::pool::Pool): an `O(1)` fixed-size-block pool.
//! - [`FrameAllocator`](crate::alloc_::frame::FrameAllocator): a linear bump
//!   allocator with `O(1)` [`reset`](crate::alloc_::frame::FrameAllocator::reset).
//! - [`AllocBox`](crate::alloc_::boxed::AllocBox): a `Box`-like owner that is
//!   generic over any [`Allocator`], demonstrating "容器带分配器".
//!
//! Everything here is `no_std` + `alloc` compatible and uses only `core` and
//! `alloc`.

extern crate alloc;

use core::alloc::Layout;
use core::fmt;
use core::ptr::NonNull;

pub mod boxed;
pub mod frame;
pub mod pool;

pub use boxed::AllocBox;
pub use frame::FrameAllocator;
pub use pool::Pool;

/// The error returned when an [`Allocator`] cannot satisfy a request.
///
/// It carries no information beyond "the allocation failed", matching the
/// shape of the unstable `core::alloc::AllocError`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct AllocError;

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("memory allocation failed")
    }
}

/// A Prism-native allocation abstraction.
///
/// This is deliberately a *stable* trait modeled on the still-unstable
/// `core::alloc::Allocator`. An allocator hands out blocks of memory described
/// by a [`Layout`] and later reclaims them.
///
/// # Guarantees expected of implementors
/// - A successful [`allocate`](Allocator::allocate) returns a non-null,
///   correctly aligned block whose length is **at least** `layout.size()`
///   (the returned slice length is the usable size).
/// - A zero-sized `layout` must still succeed, returning a dangling-but-aligned
///   pointer and a zero-length slice; it must **not** touch the heap.
/// - Memory returned by `allocate` stays valid until it is passed to
///   [`deallocate`](Allocator::deallocate) (or until the allocator is dropped,
///   for arena-style allocators that reclaim in bulk).
pub trait Allocator {
    /// Allocate a block fitting `layout`.
    ///
    /// On success the returned slice has length `>= layout.size()` and its
    /// pointer is aligned to `layout.align()`. On exhaustion returns
    /// [`AllocError`].
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;

    /// Return a block previously obtained from this allocator.
    ///
    /// # Safety
    /// `ptr` must denote a block currently allocated by *this* allocator, and
    /// `layout` must be the same [`Layout`] that was used to allocate it. After
    /// this call the block must not be used again.
    #[expect(
        unsafe_code,
        reason = "deallocation is inherently unsafe: the caller owns the ptr/layout invariant"
    )]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

/// Forward [`Allocator`] through shared references so callers can pass
/// `&MyAllocator` wherever an owned `A: Allocator` is expected (e.g. sharing a
/// single [`Pool`] across many [`AllocBox`]es).
impl<A> Allocator for &A
where
    A: Allocator + ?Sized,
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate(layout)
    }

    #[inline]
    #[expect(
        unsafe_code,
        reason = "forwards the unchanged deallocation contract to the referent"
    )]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the caller upholds `deallocate`'s contract for `*self`, and we
        // forward `ptr`/`layout` unchanged to the referent allocator.
        unsafe { (**self).deallocate(ptr, layout) }
    }
}

/// An [`Allocator`] that bridges to the process-global heap via
/// [`alloc::alloc`].
///
/// This is the default backing allocator and the baseline that [`Pool`] and
/// [`FrameAllocator`] are benchmarked against ("池 vs malloc").
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Global;

impl Allocator for Global {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            // A zero-sized request never touches the heap; hand back a dangling
            // but correctly aligned pointer. `align` is a non-zero power of two,
            // so `NonNull::new` always succeeds.
            let ptr = NonNull::new(core::ptr::without_provenance_mut(layout.align()))
                .ok_or(AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(ptr, 0));
        }

        #[expect(
            unsafe_code,
            reason = "the global allocator entry point is an unsafe FFI-style call"
        )]
        // SAFETY: `layout.size()` is non-zero (checked above), which is the sole
        // precondition of `alloc::alloc::alloc`.
        let raw = unsafe { alloc::alloc::alloc(layout) };
        let ptr = NonNull::new(raw).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    #[expect(
        unsafe_code,
        reason = "pairs with `alloc::alloc::alloc`; caller guarantees ptr/layout"
    )]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            // Zero-sized allocations were never heap-backed; nothing to free.
            return;
        }
        #[expect(
            unsafe_code,
            reason = "the global free entry point is an unsafe FFI-style call"
        )]
        // SAFETY: the caller guarantees `ptr` came from `Global::allocate` with
        // this exact `layout`, and the zero-size case returned above, so the
        // block was produced by `alloc::alloc::alloc(layout)`.
        unsafe {
            alloc::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }
}
