//! M3 virtual-memory layer: reserve / commit / protect / release of raw
//! address space, aligned reservations, guard pages, optional large (huge)
//! pages, and physical-memory information.
//!
//! This is the page-level substrate that `prism_utils`' allocators and the
//! engine's memory-budget system sit on (design doc §9, §22). It gives upper
//! crates a single, testable surface for *decoupling address-space reservation
//! from physical commit*, so a large buffer can own a contiguous virtual range
//! without paying for physical pages until they are actually touched.
//!
//! The whole module requires the `std` feature; it is absent in a `no_std`
//! build and the crate still compiles (mirroring [`crate::fs`] and
//! [`crate::thread`]). All OS access is reached through dependency-free
//! `extern` declarations resolved against the C runtime that `std` already
//! links — there is no `libc` crate — exactly like [`crate::thread::affinity`].
//!
//! ## Backends
//! - **Unix (Linux / macOS)**: `mmap`/`munmap`/`mprotect`/`madvise`/`sysconf`.
//!   Reservation is `mmap(PROT_NONE)`; commit is `mprotect` to the requested
//!   protection (physical pages arrive on first touch); decommit is
//!   `madvise(DONTNEED/FREE)` + `mprotect(PROT_NONE)`.
//! - **Windows**: `VirtualAlloc`/`VirtualFree`/`VirtualProtect` with
//!   `MEM_RESERVE`/`MEM_COMMIT`/`MEM_DECOMMIT`/`MEM_RELEASE`, plus
//!   `GlobalMemoryStatusEx`/`GetSystemInfo`/`GetLargePageMinimum`.
//! - **Other OSes**: every operation honestly returns
//!   [`VmError::Unsupported`].
//!
//! ## Honest capability gaps
//! - **Large/huge pages** are supported on Linux (`MAP_HUGETLB`, subject to the
//!   sysadmin's hugepage pool) and Windows (`MEM_LARGE_PAGES`, subject to the
//!   *Lock pages in memory* privilege). macOS exposes no portable
//!   transparent-huge-page reservation for user space, so
//!   [`Reservation::reserve_huge`] returns [`VmError::Unsupported`] there and
//!   [`large_page_size`] returns [`None`]. Query [`huge_pages_supported`]
//!   before relying on it.
//!
//! ## Lifecycle contract
//! A [`Reservation`] owns one OS mapping for its whole lifetime; its backing
//! range is released exactly once on [`Drop`]. Committing, decommitting, and
//! re-protecting sub-ranges never change which range is owned. Reserving,
//! committing, and releasing must stay paired — releasing (dropping) a region
//! while another part of the engine still reads or writes it is undefined
//! behavior, so ownership of a [`Reservation`] must track the lifetime of the
//! data placed inside it.

use core::fmt;

#[cfg(all(feature = "std", unix))]
#[path = "unix.rs"]
mod backend;

#[cfg(all(feature = "std", windows))]
#[path = "windows.rs"]
mod backend;

#[cfg(all(feature = "std", not(any(unix, windows))))]
mod backend {
    //! Fallback backend for platforms without a page-level VM API (e.g. wasm).
    //! Every operation is honestly [`super::VmError::Unsupported`].

    use super::{MemoryInfo, Region, Result, VmError};

    /// This build can reserve address space.
    pub(super) const SUPPORTED: bool = false;

    /// Reports whether large/huge pages can be requested on this platform.
    pub(super) fn huge_pages_supported() -> bool {
        false
    }

    /// OS page size, assumed 4 KiB where no query API exists.
    pub(super) fn page_size() -> usize {
        4096
    }

    /// Large-page size, unavailable on this platform.
    pub(super) fn large_page_size() -> Option<usize> {
        None
    }

    /// Physical-memory information, unavailable on this platform.
    pub(super) fn memory_info() -> Result<MemoryInfo> {
        Err(VmError::Unsupported)
    }

    /// Reserve address space (unsupported here).
    pub(super) fn reserve(_size: usize) -> Result<Region> {
        Err(VmError::Unsupported)
    }

    /// Reserve aligned address space (unsupported here).
    pub(super) fn reserve_aligned(_size: usize, _align: usize) -> Result<Region> {
        Err(VmError::Unsupported)
    }

    /// Reserve huge-page-backed address space (unsupported here).
    pub(super) fn reserve_huge(_size: usize) -> Result<Region> {
        Err(VmError::Unsupported)
    }

    /// Commit pages (unsupported here).
    pub(super) fn commit(_ptr: *mut u8, _len: usize, _prot: super::Protection) -> Result<()> {
        Err(VmError::Unsupported)
    }

    /// Decommit pages (unsupported here).
    pub(super) fn decommit(_ptr: *mut u8, _len: usize) -> Result<()> {
        Err(VmError::Unsupported)
    }

    /// Change page protection (unsupported here).
    pub(super) fn protect(_ptr: *mut u8, _len: usize, _prot: super::Protection) -> Result<()> {
        Err(VmError::Unsupported)
    }

    /// Release a mapping (no-op; nothing was ever reserved here).
    pub(super) fn release(_region: Region, _huge: bool) {}
}

/// Result type for every virtual-memory operation in this module.
pub type Result<T> = core::result::Result<T, VmError>;

/// Why a virtual-memory operation did not succeed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmError {
    /// This platform does not provide the requested capability (for example
    /// huge pages on macOS, or any VM primitive on wasm). No OS call was made
    /// that could partially succeed.
    Unsupported,
    /// An argument was invalid: zero size, a non-page-aligned offset or
    /// length, a sub-range outside the reservation, or an alignment that is not
    /// a power of two.
    InvalidArgument,
    /// The OS could not satisfy the request for lack of address space or
    /// physical memory (for example `mmap`/`VirtualAlloc` returning failure, or
    /// an empty hugepage pool).
    OutOfMemory,
    /// The OS rejected the request for another reason; carries the platform
    /// error code (`errno` on Unix, `GetLastError` on Windows).
    SystemError(i32),
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmError::Unsupported => {
                write!(f, "virtual-memory capability is unsupported on this platform")
            }
            VmError::InvalidArgument => write!(f, "invalid virtual-memory argument"),
            VmError::OutOfMemory => write!(f, "out of address space or physical memory"),
            VmError::SystemError(code) => {
                write!(f, "OS rejected virtual-memory request (code {code})")
            }
        }
    }
}

impl std::error::Error for VmError {}

/// Access protection applied to a committed page range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protection {
    /// No access; any read, write, or execute faults. Used for guard pages and
    /// freshly reserved (uncommitted) space.
    None,
    /// Read-only.
    Read,
    /// Read and write — the common case for data buffers.
    ReadWrite,
    /// Read and execute — for generated code pages (JIT), use with care.
    ReadExec,
    /// Read, write, and execute. Discouraged (W^X); offered for completeness.
    ReadWriteExec,
}

/// Physical-memory and page-size information for the running host.
///
/// Produced by [`memory_info`]. `available_physical` is a best-effort estimate
/// of memory that can be allocated without paging; its exact definition is
/// platform-specific (see [`memory_info`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryInfo {
    /// Total physical RAM installed, in bytes.
    pub total_physical: u64,
    /// Best-effort physical RAM currently available, in bytes.
    pub available_physical: u64,
    /// OS base page size in bytes (always a power of two, typically 4 KiB or
    /// 16 KiB).
    pub page_size: usize,
    /// Large/huge page size in bytes if the platform supports large pages,
    /// otherwise [`None`].
    pub large_page_size: Option<usize>,
}

/// Internal description of an owned OS mapping.
///
/// `base`/`base_len` identify the exact span handed to the release call (which
/// may be larger than the usable span for aligned reservations on Windows),
/// while `ptr`/`len` are the usable, correctly-aligned region exposed to
/// callers.
#[derive(Clone, Copy)]
pub(crate) struct Region {
    base: *mut u8,
    base_len: usize,
    ptr: *mut u8,
    len: usize,
}

/// Returns the OS base page size in bytes (a power of two).
pub fn page_size() -> usize {
    backend::page_size()
}

/// Returns the large/huge page size in bytes, or [`None`] if large pages are
/// unavailable on this platform.
pub fn large_page_size() -> Option<usize> {
    backend::large_page_size()
}

/// Returns `true` if this build can request large/huge pages.
///
/// Even when `true`, an individual [`Reservation::reserve_huge`] can still fail
/// at runtime (an empty Linux hugepage pool, or a missing Windows *Lock pages
/// in memory* privilege); it then returns [`VmError::OutOfMemory`] or
/// [`VmError::SystemError`].
pub fn huge_pages_supported() -> bool {
    backend::huge_pages_supported()
}

/// Returns `true` if this build has a working page-level VM backend
/// (reserve/commit/release). False only on platforms without such an API.
pub fn virtual_memory_supported() -> bool {
    backend::SUPPORTED
}

/// Queries total and available physical memory plus page sizes.
///
/// `total_physical` is installed RAM. `available_physical` is a best-effort
/// figure:
/// - **Linux**: `sysconf(_SC_AVPHYS_PAGES)` (pages not currently in use).
/// - **macOS**: free + inactive (+ speculative + purgeable) pages from
///   `host_statistics64`, i.e. memory reclaimable without swapping.
/// - **Windows**: `MEMORYSTATUSEX::ullAvailPhys`.
///
/// Returns [`VmError::Unsupported`] on platforms with no query API.
pub fn memory_info() -> Result<MemoryInfo> {
    backend::memory_info()
}

/// An owned reservation of contiguous virtual address space.
///
/// Create one with [`reserve`](Reservation::reserve),
/// [`reserve_aligned`](Reservation::reserve_aligned), or
/// [`reserve_huge`](Reservation::reserve_huge). A plain reservation starts
/// fully *reserved but uncommitted* (no physical pages, no access); call
/// [`commit`](Reservation::commit) to back a sub-range with physical pages at a
/// chosen [`Protection`]. A huge-page reservation is committed read-write up
/// front because large pages cannot be reserved without backing.
///
/// The entire range is released exactly once when the `Reservation` is dropped.
pub struct Reservation {
    region: Region,
    huge: bool,
}

#[expect(
    unsafe_code,
    reason = "a reservation is a self-contained OS mapping with no thread affinity, so it is Send + Sync like other handle types"
)]
// SAFETY: a `Reservation` owns a unique OS address-space mapping identified by
// raw pointers; those pointers carry no thread affinity, so transferring
// ownership to another thread is sound (equivalent to moving a `Box`). The type
// exposes no `&self` method that mutates Rust-visible memory, so sharing `&self`
// is also sound; coordinating overlapping commit/protect calls on the same
// sub-range is the caller's responsibility, exactly as it is for the raw
// pointers obtained from it.
unsafe impl Send for Reservation {}

#[expect(
    unsafe_code,
    reason = "a reservation is a self-contained OS mapping with no thread affinity, so it is Send + Sync like other handle types"
)]
// SAFETY: see the `Send` impl above — `&Reservation` grants only read access to
// the stored pointer/length and never mutates shared Rust state.
unsafe impl Sync for Reservation {}

impl Reservation {
    /// Reserve `size` bytes of address space (rounded up to the page size),
    /// uncommitted and inaccessible.
    ///
    /// Returns [`VmError::InvalidArgument`] for a zero size,
    /// [`VmError::OutOfMemory`] if the OS has no room, or
    /// [`VmError::Unsupported`] on platforms without a VM backend.
    pub fn reserve(size: usize) -> Result<Self> {
        if size == 0 {
            return Err(VmError::InvalidArgument);
        }
        let rounded = round_up(size, page_size()).ok_or(VmError::InvalidArgument)?;
        Ok(Self {
            region: backend::reserve(rounded)?,
            huge: false,
        })
    }

    /// Reserve `size` bytes whose base address is a multiple of `align`.
    ///
    /// `align` must be a power of two. Alignments at or below the page size are
    /// satisfied by an ordinary [`reserve`](Self::reserve), since every mapping
    /// is already page-aligned. The reservation is uncommitted and
    /// inaccessible.
    pub fn reserve_aligned(size: usize, align: usize) -> Result<Self> {
        if size == 0 || align == 0 || !align.is_power_of_two() {
            return Err(VmError::InvalidArgument);
        }
        let ps = page_size();
        let rounded = round_up(size, ps).ok_or(VmError::InvalidArgument)?;
        if align <= ps {
            return Ok(Self {
                region: backend::reserve(rounded)?,
                huge: false,
            });
        }
        Ok(Self {
            region: backend::reserve_aligned(rounded, align)?,
            huge: false,
        })
    }

    /// Reserve `size` bytes backed by large/huge pages, committed read-write.
    ///
    /// `size` is rounded up to the large-page size. Returns
    /// [`VmError::Unsupported`] where huge pages do not exist (macOS, wasm),
    /// [`VmError::OutOfMemory`] if the hugepage pool is empty, or
    /// [`VmError::SystemError`] if the OS refuses (for example a missing
    /// privilege). Check [`huge_pages_supported`] first.
    pub fn reserve_huge(size: usize) -> Result<Self> {
        if size == 0 {
            return Err(VmError::InvalidArgument);
        }
        if !huge_pages_supported() {
            return Err(VmError::Unsupported);
        }
        let lps = large_page_size().ok_or(VmError::Unsupported)?;
        let rounded = round_up(size, lps).ok_or(VmError::InvalidArgument)?;
        Ok(Self {
            region: backend::reserve_huge(rounded)?,
            huge: true,
        })
    }

    /// Base pointer of the usable reservation.
    pub fn as_ptr(&self) -> *const u8 {
        self.region.ptr.cast_const()
    }

    /// Mutable base pointer of the usable reservation.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.region.ptr
    }

    /// Usable length of the reservation in bytes (a multiple of the page size,
    /// or of the large-page size for a huge reservation).
    pub fn len(&self) -> usize {
        self.region.len
    }

    /// Returns `true` if the reservation has zero length. A real reservation is
    /// never empty, so this is always `false`; provided to satisfy the
    /// `len`/`is_empty` convention.
    pub fn is_empty(&self) -> bool {
        self.region.len == 0
    }

    /// Returns `true` if this reservation is backed by large/huge pages.
    pub fn is_huge(&self) -> bool {
        self.huge
    }

    /// Commit the sub-range `[offset, offset + len)` with the given protection,
    /// backing it with physical pages (on first touch for data pages).
    ///
    /// `offset` and `len` must be page-aligned and the range must lie inside
    /// the reservation. Huge reservations are already committed read-write;
    /// calling this on them only changes protection and is better expressed via
    /// [`protect`](Self::protect).
    pub fn commit(&self, offset: usize, len: usize, prot: Protection) -> Result<()> {
        let ptr = self.checked_subrange(offset, len)?;
        backend::commit(ptr, len, prot)
    }

    /// Decommit the sub-range `[offset, offset + len)`, returning its physical
    /// pages to the OS while keeping the address space reserved.
    ///
    /// `offset` and `len` must be page-aligned and inside the reservation.
    /// After this, the range reads back as zero on next commit; accessing it
    /// before re-committing faults.
    pub fn decommit(&self, offset: usize, len: usize) -> Result<()> {
        let ptr = self.checked_subrange(offset, len)?;
        backend::decommit(ptr, len)
    }

    /// Change the protection of the committed sub-range `[offset, offset +
    /// len)`.
    ///
    /// `offset` and `len` must be page-aligned and inside the reservation.
    pub fn protect(&self, offset: usize, len: usize, prot: Protection) -> Result<()> {
        let ptr = self.checked_subrange(offset, len)?;
        backend::protect(ptr, len, prot)
    }

    /// Install a no-access guard page of one page at `offset`.
    ///
    /// Any read, write, or execute touching the page faults immediately, giving
    /// a precise crash at an overrun boundary (the debug-build overflow/overrun
    /// detector described in the design doc). This is a portable no-access
    /// guard (`PROT_NONE` / `PAGE_NOACCESS`), not the one-shot Windows
    /// `PAGE_GUARD` semantic. `offset` must be page-aligned and leave room for
    /// one page inside the reservation.
    pub fn guard_page(&self, offset: usize) -> Result<()> {
        let ps = page_size();
        let ptr = self.checked_subrange(offset, ps)?;
        backend::protect(ptr, ps, Protection::None)
    }

    /// Validate that `[offset, offset + len)` is page-aligned and within the
    /// reservation, returning the absolute pointer to `offset`.
    fn checked_subrange(&self, offset: usize, len: usize) -> Result<*mut u8> {
        let ps = page_size();
        if len == 0 || !offset.is_multiple_of(ps) || !len.is_multiple_of(ps) {
            return Err(VmError::InvalidArgument);
        }
        let end = offset.checked_add(len).ok_or(VmError::InvalidArgument)?;
        if end > self.region.len {
            return Err(VmError::InvalidArgument);
        }
        // Pointer arithmetic stays within the single owned mapping.
        Ok(self.region.ptr.wrapping_add(offset))
    }
}

impl fmt::Debug for Reservation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reservation")
            .field("ptr", &self.region.ptr)
            .field("len", &self.region.len)
            .field("huge", &self.huge)
            .finish()
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        backend::release(self.region, self.huge);
    }
}

/// Round `v` up to the next multiple of the power-of-two `align`, or [`None`]
/// on overflow.
fn round_up(v: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    v.checked_add(align - 1).map(|s| s & !(align - 1))
}

#[cfg(test)]
mod tests;
