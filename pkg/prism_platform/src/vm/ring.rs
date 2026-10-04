//! Magic ring buffer: a mirrored virtual-memory mapping (design doc §24.2
//! "环形流送堆").
//!
//! A [`MirroredRing`] reserves `2 * capacity` *virtual* bytes but backs them
//! with only `capacity` *physical* bytes, by mapping the same physical pages
//! twice at adjacent virtual addresses. Writing byte `i` and byte `capacity +
//! i` therefore touches the same physical page: a reader or writer that walks
//! off the end of the first half transparently wraps into the start of the
//! same data in the second half. Ring buffers built on top of it never split a
//! contiguous record across the wrap boundary and need no per-element modulo or
//! branch — a producer/consumer can `memcpy` a span that straddles the end in a
//! single call.
//!
//! This is the streaming-ring substrate called out in the design doc for audio
//! mix rings and asset streaming queues. It sits beside the plain
//! [`Reservation`](super::Reservation) page primitive and, like the rest of the
//! [`vm`](super) module, requires the `std` feature and reaches the OS through
//! dependency-free `extern` declarations (no `libc` crate).
//!
//! ## Backends
//! - **Apple (macOS, aarch64/x86-64)**: `mach_vm_allocate` reserves `2N`, then
//!   `mach_vm_remap` aliases the first `N` bytes over the second `N`
//!   (shared, non-copy). Verified on real hardware.
//! - **Other platforms**: honestly [`VmError::Unsupported`]. A real Linux
//!   (`memfd_create` + double `mmap`) and Windows (`CreateFileMapping` +
//!   `MapViewOfFileEx`) backend is PLANNED but intentionally not shipped
//!   unverified from this macOS host (see the design doc's honest-boundary
//!   note). Query [`MirroredRing::is_supported`] before relying on it.
//!
//! ## Lifecycle contract
//! A [`MirroredRing`] owns its whole `2N` mapping and releases it exactly once
//! on [`Drop`]. The usable capacity is `N` ([`capacity`](MirroredRing::capacity));
//! the pointer returned by [`as_mut_ptr`](MirroredRing::as_mut_ptr) is valid for
//! reads and writes across the full `2N` span, where the upper `N` bytes alias
//! the lower `N`. Coordinating concurrent producer/consumer access is the
//! caller's responsibility, exactly as for the raw pointers from a
//! [`Reservation`](super::Reservation).

use core::fmt;

use super::{backend, page_size, Region, Result, VmError};

/// An owned mirrored (magic) ring mapping of `2 * capacity` virtual bytes
/// backed by `capacity` physical bytes.
///
/// Create one with [`with_min_capacity`](Self::with_min_capacity). The two
/// halves alias the same physical pages, so a byte written at offset `i` is
/// also visible at offset `capacity + i`.
pub struct MirroredRing {
    region: Region,
}

#[expect(
    unsafe_code,
    reason = "a mirrored mapping is a self-contained OS mapping with no thread affinity, so it is Send + Sync like Reservation"
)]
// SAFETY: a `MirroredRing` owns a unique OS mapping identified by raw pointers
// that carry no thread affinity; moving it to another thread is sound
// (equivalent to moving a `Box`). It exposes no `&self` method that mutates
// Rust-visible state, so sharing `&self` is sound too — coordinating concurrent
// writes through the aliased pages is the caller's responsibility, exactly as
// for the raw pointers it hands out.
unsafe impl Send for MirroredRing {}

#[expect(
    unsafe_code,
    reason = "a mirrored mapping is a self-contained OS mapping with no thread affinity, so it is Send + Sync like Reservation"
)]
// SAFETY: see the `Send` impl above — `&MirroredRing` grants only read access to
// the stored pointer/length and never mutates shared Rust state.
unsafe impl Sync for MirroredRing {}

impl MirroredRing {
    /// Create a mirrored ring whose capacity is at least `min_bytes`, rounded
    /// up to a whole number of pages.
    ///
    /// Returns [`VmError::InvalidArgument`] for a zero size,
    /// [`VmError::OutOfMemory`] if the OS has no room, or
    /// [`VmError::Unsupported`] on platforms without a mirroring backend (check
    /// [`is_supported`](Self::is_supported) first).
    pub fn with_min_capacity(min_bytes: usize) -> Result<Self> {
        if min_bytes == 0 {
            return Err(VmError::InvalidArgument);
        }
        let rounded = super::round_up(min_bytes, page_size()).ok_or(VmError::InvalidArgument)?;
        Ok(Self {
            region: backend::mirror_map(rounded)?,
        })
    }

    /// Returns `true` if this build has a working mirrored-mapping backend.
    pub fn is_supported() -> bool {
        backend::MIRROR_SUPPORTED
    }

    /// Usable capacity in bytes (a whole number of pages). The mapping spans
    /// `2 * capacity()` virtual bytes.
    pub fn capacity(&self) -> usize {
        self.region.len
    }

    /// Returns `true` if the ring has zero capacity. A real ring is never
    /// empty, so this is always `false`; provided for the `len`/`is_empty`
    /// convention.
    pub fn is_empty(&self) -> bool {
        self.region.len == 0
    }

    /// Base pointer of the mirrored mapping. Valid for reads across the full
    /// `2 * capacity()` span; the upper half aliases the lower half.
    pub fn as_ptr(&self) -> *const u8 {
        self.region.ptr.cast_const()
    }

    /// Mutable base pointer of the mirrored mapping. Valid for reads and writes
    /// across the full `2 * capacity()` span; a write at offset `i` is visible
    /// at offset `capacity() + i` and vice versa.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.region.ptr
    }
}

impl fmt::Debug for MirroredRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MirroredRing")
            .field("ptr", &self.region.ptr)
            .field("capacity", &self.region.len)
            .finish()
    }
}

impl Drop for MirroredRing {
    fn drop(&mut self) {
        backend::mirror_release(self.region);
    }
}
