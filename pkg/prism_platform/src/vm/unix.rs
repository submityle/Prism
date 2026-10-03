//! Unix (Linux / macOS) virtual-memory backend.
//!
//! Reservation maps anonymous `PROT_NONE` space with `mmap`; commit raises the
//! protection with `mprotect` (physical pages fault in on first touch); decommit
//! drops the physical pages and re-arms `PROT_NONE` so the range faults until
//! re-committed. On Linux it uses `MADV_DONTNEED` + `mprotect` (which zero-fills
//! on the next fault); on macOS/BSD, where `MADV_FREE` only lazily reclaims and
//! leaves stale contents, it instead remaps fresh zero-fill anonymous pages with
//! `MAP_FIXED` to guarantee a zeroed re-commit. Release unmaps with `munmap`. Memory
//! information comes from `sysconf` (Linux) or `sysctlbyname` +
//! `host_statistics64` (macOS). Huge pages use `MAP_HUGETLB` on Linux and are
//! honestly unsupported on macOS, which has no portable user-space superpage
//! reservation.
//!
//! Every FFI call is a dependency-free `extern` declaration resolved against the
//! C runtime `std` already links (no `libc` crate), mirroring
//! [`crate::thread::affinity`].
#![expect(
    unsafe_code,
    reason = "page-level virtual memory requires mmap/munmap/mprotect/madvise/sysconf (and mach/sysctl on macOS) via dependency-free C FFI"
)]

use core::ffi::{c_long, c_void};

use super::{MemoryInfo, Protection, Region, Result, VmError};

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    fn sysconf(name: i32) -> c_long;
}

// Protection bits are identical on Linux and macOS.
const PROT_NONE: i32 = 0x0;
const PROT_READ: i32 = 0x1;
const PROT_WRITE: i32 = 0x2;
const PROT_EXEC: i32 = 0x4;

const MAP_PRIVATE: i32 = 0x2;
/// `MAP_FIXED` has the same value on Linux and macOS/BSD.
const MAP_FIXED: i32 = 0x10;

/// `errno` value for "cannot allocate memory" (identical on Linux and macOS).
const ENOMEM: i32 = 12;

#[cfg(target_os = "linux")]
mod sys {
    //! Linux-specific constants and `errno` access.
    pub(super) const MAP_ANON: i32 = 0x20; // MAP_ANONYMOUS
    pub(super) const MAP_HUGETLB: i32 = 0x40000;
    pub(super) const MADV_DECOMMIT: i32 = 4; // MADV_DONTNEED
    /// Linux `MADV_DONTNEED` guarantees zero-fill on the next fault, so the
    /// cheaper madvise+mprotect decommit path is correct here.
    pub(super) const DECOMMIT_BY_REMAP: bool = false;
    pub(super) const SC_PAGESIZE: i32 = 30;
    pub(super) const SC_PHYS_PAGES: i32 = 85;
    pub(super) const SC_AVPHYS_PAGES: i32 = 86;

    unsafe extern "C" {
        fn __errno_location() -> *mut i32;
    }

    pub(super) fn errno() -> i32 {
        // SAFETY: `__errno_location` returns a valid pointer to this thread's
        // `errno`, which we only read.
        unsafe { *__errno_location() }
    }
}

#[cfg(target_vendor = "apple")]
mod sys {
    //! macOS-specific constants and `errno` access.
    pub(super) const MAP_ANON: i32 = 0x1000;
    pub(super) const MADV_DECOMMIT: i32 = 5; // MADV_FREE
    /// macOS `MADV_FREE` only *lazily* reclaims pages and leaves their old
    /// contents visible until the kernel reuses them, so a re-commit could
    /// read stale bytes. Decommit by remapping fresh zero-fill anonymous pages
    /// instead, which guarantees a zeroed re-commit.
    pub(super) const DECOMMIT_BY_REMAP: bool = true;
    pub(super) const SC_PAGESIZE: i32 = 29;
    pub(super) const SC_PHYS_PAGES: i32 = 200;

    unsafe extern "C" {
        fn __error() -> *mut i32;
    }

    pub(super) fn errno() -> i32 {
        // SAFETY: `__error` returns a valid pointer to this thread's `errno`,
        // which we only read.
        unsafe { *__error() }
    }
}

#[cfg(not(any(target_os = "linux", target_vendor = "apple")))]
mod sys {
    //! Generic Unix fallback (BSDs and similar). Uses the common BSD-ish
    //! `MAP_ANON`/`MADV_FREE` values and reads `errno` via `__error`.
    pub(super) const MAP_ANON: i32 = 0x1000;
    pub(super) const MADV_DECOMMIT: i32 = 5; // MADV_FREE
    /// BSD `MADV_FREE` is lazy like macOS, so remap to guarantee zeroing.
    pub(super) const DECOMMIT_BY_REMAP: bool = true;
    pub(super) const SC_PAGESIZE: i32 = 29;
    pub(super) const SC_PHYS_PAGES: i32 = 200;

    unsafe extern "C" {
        fn __error() -> *mut i32;
    }

    pub(super) fn errno() -> i32 {
        // SAFETY: `__error` returns a valid pointer to this thread's `errno`.
        unsafe { *__error() }
    }
}

/// This build has a working page-level VM backend.
pub(super) const SUPPORTED: bool = true;

/// Translate a [`Protection`] into Unix `mprotect`/`mmap` protection bits.
fn prot_bits(prot: Protection) -> i32 {
    match prot {
        Protection::None => PROT_NONE,
        Protection::Read => PROT_READ,
        Protection::ReadWrite => PROT_READ | PROT_WRITE,
        Protection::ReadExec => PROT_READ | PROT_EXEC,
        Protection::ReadWriteExec => PROT_READ | PROT_WRITE | PROT_EXEC,
    }
}

/// Map an `mmap`/`mprotect` failure's `errno` into a [`VmError`].
fn os_error() -> VmError {
    let e = sys::errno();
    if e == ENOMEM {
        VmError::OutOfMemory
    } else {
        VmError::SystemError(e)
    }
}

/// Returns `true` if the `mmap` return value is `MAP_FAILED` (`(void*) -1`).
fn is_map_failed(p: *mut c_void) -> bool {
    p.addr() == usize::MAX
}

pub(super) fn page_size() -> usize {
    // SAFETY: `sysconf` is a pure query with no memory arguments.
    let v = unsafe { sysconf(sys::SC_PAGESIZE) };
    if v > 0 { v as usize } else { 4096 }
}

#[cfg(target_os = "linux")]
pub(super) fn large_page_size() -> Option<usize> {
    // Linux reports the default huge-page size in /proc/meminfo as
    // "Hugepagesize:   <kB> kB". Absent or zero means hugepages are not
    // configured in this kernel.
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Hugepagesize:") {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            if kb > 0 {
                return Some(kb * 1024);
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
pub(super) fn large_page_size() -> Option<usize> {
    // macOS and other Unixes expose no portable user-space huge-page
    // reservation, so we honestly report none.
    None
}

pub(super) fn huge_pages_supported() -> bool {
    cfg!(target_os = "linux") && large_page_size().is_some()
}

#[cfg(target_os = "linux")]
pub(super) fn memory_info() -> Result<MemoryInfo> {
    let ps = page_size();
    // SAFETY: each `sysconf` call is a pure query with no memory arguments.
    let phys = unsafe { sysconf(sys::SC_PHYS_PAGES) };
    let avail = unsafe { sysconf(sys::SC_AVPHYS_PAGES) };
    if phys <= 0 {
        return Err(os_error());
    }
    let total = (phys as u64).saturating_mul(ps as u64);
    let available = if avail > 0 {
        (avail as u64).saturating_mul(ps as u64).min(total)
    } else {
        total
    };
    Ok(MemoryInfo {
        total_physical: total,
        available_physical: available,
        page_size: ps,
        large_page_size: large_page_size(),
    })
}

#[cfg(not(target_os = "linux"))]
pub(super) fn memory_info() -> Result<MemoryInfo> {
    let ps = page_size();
    let total = apple_total_physical(ps);
    if total == 0 {
        return Err(VmError::Unsupported);
    }
    let available = apple_available_physical(ps).min(total);
    Ok(MemoryInfo {
        total_physical: total,
        available_physical: available,
        page_size: ps,
        large_page_size: None,
    })
}

#[cfg(not(target_os = "linux"))]
fn apple_total_physical(ps: usize) -> u64 {
    use core::ffi::c_char;

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> i32;
    }

    let mut val: u64 = 0;
    let mut len: usize = size_of::<u64>();
    let name = c"hw.memsize";
    // SAFETY: `name` is a valid NUL-terminated C string; we hand `sysctlbyname`
    // a writable `u64` out-parameter and its exact byte length, and no input
    // buffer. It writes at most `len` bytes into `val`.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            core::ptr::from_mut(&mut val).cast::<c_void>(),
            core::ptr::from_mut(&mut len),
            core::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && val > 0 {
        return val;
    }
    // Fallback: physical page count from sysconf.
    // SAFETY: `sysconf` is a pure query with no memory arguments.
    let phys = unsafe { sysconf(sys::SC_PHYS_PAGES) };
    if phys > 0 {
        (phys as u64).saturating_mul(ps as u64)
    } else {
        0
    }
}

#[cfg(not(target_os = "linux"))]
fn apple_available_physical(ps: usize) -> u64 {
    // `host_statistics64(HOST_VM_INFO64)` returns the Mach `vm_statistics64`
    // page counters. We treat free + inactive + speculative + purgeable pages
    // as reclaimable (available) memory. The counters live at fixed `u32`
    // indices in the returned array (free=0, inactive=2, purgeable=22,
    // speculative=23); a generous buffer tolerates kernel struct growth.
    const HOST_VM_INFO64: i32 = 4;

    unsafe extern "C" {
        fn mach_host_self() -> u32;
        fn host_statistics64(host: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
    }

    let mut buf = [0u32; 64];
    let mut count: u32 = 64;
    // SAFETY: `mach_host_self` yields this task's host port; we pass a writable
    // 64-word buffer and its capacity. `host_statistics64` writes at most
    // `count` 32-bit words into `buf` and updates `count` to the words written.
    let kr = unsafe {
        let host = mach_host_self();
        host_statistics64(
            host,
            HOST_VM_INFO64,
            buf.as_mut_ptr().cast::<i32>(),
            core::ptr::from_mut(&mut count),
        )
    };
    if kr != 0 || count < 3 {
        return 0;
    }
    let read = |i: usize| -> u64 {
        if (i as u32) < count {
            u64::from(buf[i])
        } else {
            0
        }
    };
    let pages = read(0) + read(2) + read(22) + read(23);
    pages.saturating_mul(ps as u64)
}

pub(super) fn reserve(size: usize) -> Result<Region> {
    // SAFETY: an anonymous `mmap` with `fd = -1` reads no memory; a null `addr`
    // lets the kernel choose the base. We validate the result against
    // `MAP_FAILED` before using it.
    let p = unsafe {
        mmap(
            core::ptr::null_mut(),
            size,
            PROT_NONE,
            MAP_PRIVATE | sys::MAP_ANON,
            -1,
            0,
        )
    };
    if is_map_failed(p) {
        return Err(os_error());
    }
    let ptr = p.cast::<u8>();
    Ok(Region {
        base: ptr,
        base_len: size,
        ptr,
        len: size,
    })
}

pub(super) fn reserve_aligned(size: usize, align: usize) -> Result<Region> {
    let total = size.checked_add(align).ok_or(VmError::InvalidArgument)?;
    // SAFETY: see `reserve`; this over-maps `size + align` anonymous bytes.
    let p = unsafe {
        mmap(
            core::ptr::null_mut(),
            total,
            PROT_NONE,
            MAP_PRIVATE | sys::MAP_ANON,
            -1,
            0,
        )
    };
    if is_map_failed(p) {
        return Err(os_error());
    }
    let base_addr = p.addr();
    let aligned_addr = (base_addr + (align - 1)) & !(align - 1);
    let front = aligned_addr - base_addr;
    let aligned = p.wrapping_byte_add(front);
    if front > 0 {
        // SAFETY: `p` is the mapping base and `front` bytes of slack precede the
        // aligned start; unmapping exactly that leading slice is valid.
        unsafe {
            munmap(p, front);
        }
    }
    let tail_len = total - front - size;
    if tail_len > 0 {
        let tail = aligned.wrapping_byte_add(size);
        // SAFETY: `tail` is the trailing slack within the same mapping and
        // `tail_len` its exact length.
        unsafe {
            munmap(tail, tail_len);
        }
    }
    let ptr = aligned.cast::<u8>();
    Ok(Region {
        base: ptr,
        base_len: size,
        ptr,
        len: size,
    })
}

#[cfg(target_os = "linux")]
pub(super) fn reserve_huge(size: usize) -> Result<Region> {
    // SAFETY: see `reserve`; `MAP_HUGETLB` requests huge pages, committed
    // read-write. Failure (empty pool) is reported via `errno`.
    let p = unsafe {
        mmap(
            core::ptr::null_mut(),
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | sys::MAP_ANON | sys::MAP_HUGETLB,
            -1,
            0,
        )
    };
    if is_map_failed(p) {
        return Err(os_error());
    }
    let ptr = p.cast::<u8>();
    Ok(Region {
        base: ptr,
        base_len: size,
        ptr,
        len: size,
    })
}

#[cfg(not(target_os = "linux"))]
pub(super) fn reserve_huge(_size: usize) -> Result<Region> {
    Err(VmError::Unsupported)
}

pub(super) fn commit(ptr: *mut u8, len: usize, prot: Protection) -> Result<()> {
    // SAFETY: `ptr`/`len` is a page-aligned sub-range of a live reservation
    // (validated by the caller); `mprotect` only changes its protection.
    let rc = unsafe { mprotect(ptr.cast::<c_void>(), len, prot_bits(prot)) };
    if rc == 0 { Ok(()) } else { Err(os_error()) }
}

pub(super) fn decommit(ptr: *mut u8, len: usize) -> Result<()> {
    if sys::DECOMMIT_BY_REMAP {
        // Atomically overwrite the sub-range with a fresh zero-fill anonymous
        // `PROT_NONE` mapping. `MAP_FIXED` drops the old physical pages and
        // guarantees the next commit reads back zero, which bare `MADV_FREE`
        // on macOS/BSD does not. The surrounding reservation is untouched.
        //
        // SAFETY: `ptr`/`len` is a validated page-aligned sub-range of a live
        // reservation owned by the caller. `MAP_FIXED` replaces exactly that
        // span; `fd = -1` with `MAP_ANON` reads no file. On success the range
        // stays a valid (inaccessible) part of the reservation.
        let addr = unsafe {
            mmap(
                ptr.cast::<c_void>(),
                len,
                PROT_NONE,
                MAP_FIXED | MAP_PRIVATE | sys::MAP_ANON,
                -1,
                0,
            )
        };
        return if is_map_failed(addr) { Err(os_error()) } else { Ok(()) };
    }
    // SAFETY: `ptr`/`len` is a validated page-aligned sub-range of a live
    // reservation. `madvise` drops the physical pages; `mprotect` re-arms
    // `PROT_NONE` so the range faults until re-committed.
    let advised = unsafe { madvise(ptr.cast::<c_void>(), len, sys::MADV_DECOMMIT) };
    if advised != 0 {
        return Err(os_error());
    }
    // SAFETY: same validated sub-range as above.
    let rc = unsafe { mprotect(ptr.cast::<c_void>(), len, PROT_NONE) };
    if rc == 0 { Ok(()) } else { Err(os_error()) }
}

pub(super) fn protect(ptr: *mut u8, len: usize, prot: Protection) -> Result<()> {
    // SAFETY: `ptr`/`len` is a validated page-aligned sub-range of a live
    // reservation; `mprotect` only changes its protection.
    let rc = unsafe { mprotect(ptr.cast::<c_void>(), len, prot_bits(prot)) };
    if rc == 0 { Ok(()) } else { Err(os_error()) }
}

pub(super) fn release(region: Region, _huge: bool) {
    // SAFETY: `region.base`/`region.base_len` is exactly the span this
    // reservation owns; `munmap` is called once, from `Drop`.
    unsafe {
        munmap(region.base.cast::<c_void>(), region.base_len);
    }
}
