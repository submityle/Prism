//! Unix (Linux / macOS / BSD) memory-mapping backend.
//!
//! Maps the file descriptor with `mmap(MAP_SHARED)` so writes propagate back to
//! the file, unmaps with `munmap`, and flushes dirty pages with
//! `msync(MS_SYNC)`. Arbitrary byte offsets are supported by aligning the
//! underlying mapping down to a page boundary and exposing the requested
//! sub-slice.
//!
//! Every FFI call is a dependency-free `extern` declaration resolved against
//! the C runtime that `std` already links (no `libc` crate), mirroring
//! [`crate::vm`].

use core::ffi::{c_int, c_void};
use std::fs::File;
use std::os::fd::AsRawFd;

use super::{MmapError, Result};

#[expect(
    unsafe_code,
    reason = "memory mapping requires the mmap/munmap/msync syscalls via dependency-free C FFI"
)]
unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64)
        -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn msync(addr: *mut c_void, len: usize, flags: c_int) -> c_int;
}

const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x1;

#[cfg(target_os = "linux")]
mod sys {
    use core::ffi::c_int;
    pub(super) const MS_SYNC: c_int = 4;
    #[expect(unsafe_code, reason = "errno accessor FFI declaration")]
    unsafe extern "C" {
        fn __errno_location() -> *mut c_int;
    }
    #[expect(unsafe_code, reason = "reading this thread's errno through its C accessor")]
    pub(super) fn errno() -> c_int {
        // SAFETY: `__errno_location` returns a valid pointer to this thread's
        // `errno`, which we only read.
        unsafe { *__errno_location() }
    }
}

#[cfg(not(target_os = "linux"))]
mod sys {
    use core::ffi::c_int;
    // macOS/BSD `MS_SYNC`.
    pub(super) const MS_SYNC: c_int = 0x0010;
    #[expect(unsafe_code, reason = "errno accessor FFI declaration")]
    unsafe extern "C" {
        fn __error() -> *mut c_int;
    }
    #[expect(unsafe_code, reason = "reading this thread's errno through its C accessor")]
    pub(super) fn errno() -> c_int {
        // SAFETY: `__error` returns a valid pointer to this thread's `errno`,
        // which we only read.
        unsafe { *__error() }
    }
}

/// This build maps files zero-copy through the real `mmap` primitive.
pub(super) const SUPPORTED: bool = true;

/// An owned Unix file mapping.
///
/// `base`/`base_len` are the page-aligned span handed to `munmap`; `data`/`len`
/// are the caller-visible sub-range (which may start part-way into the first
/// mapped page for a non-page-aligned offset).
pub(super) struct Mapping {
    base: *mut c_void,
    base_len: usize,
    data: *mut u8,
    len: usize,
    writable: bool,
    // Keep the descriptor owner alive for the mapping's lifetime. Unix keeps a
    // mapping valid after the fd is closed, but holding it is tidy and lets the
    // fallback backends share the same ownership shape.
    _file: File,
}

/// Map `[offset, offset + len)` of `file`, read-only or read-write.
#[expect(
    unsafe_code,
    reason = "mmap over a file descriptor is the core zero-copy mapping primitive"
)]
pub(super) fn map(file: File, offset: u64, len: usize, writable: bool) -> Result<Mapping> {
    let ps = crate::vm::page_size() as u64;
    let aligned = offset & !(ps - 1);
    let delta = (offset - aligned) as usize;
    let map_len = delta.checked_add(len).ok_or(MmapError::InvalidArgument)?;
    let prot = PROT_READ | if writable { PROT_WRITE } else { 0 };
    let fd = file.as_raw_fd();
    // SAFETY: `fd` is a live descriptor owned by `file`; `aligned` is
    // page-aligned as `mmap` requires; `map_len` is the validated mapping
    // length. A null `addr` lets the kernel choose the base. The result is
    // checked against `MAP_FAILED` before any use.
    let p = unsafe { mmap(core::ptr::null_mut(), map_len, prot, MAP_SHARED, fd, aligned as i64) };
    if p.addr() == usize::MAX {
        return Err(MmapError::System(sys::errno()));
    }
    let data = p.cast::<u8>().wrapping_add(delta);
    Ok(Mapping {
        base: p,
        base_len: map_len,
        data,
        len,
        writable,
        _file: file,
    })
}

impl Mapping {
    pub(super) fn as_ptr(&self) -> *const u8 {
        self.data.cast_const()
    }

    pub(super) fn as_mut_ptr(&self) -> *mut u8 {
        self.data
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    #[expect(
        unsafe_code,
        reason = "msync flushes the mapped file-backed pages to disk"
    )]
    pub(super) fn flush(&self) -> Result<()> {
        if !self.writable {
            return Ok(());
        }
        // SAFETY: `base`/`base_len` is exactly the live mapping this value
        // owns; `msync` only writes its dirty pages back to the file.
        let rc = unsafe { msync(self.base, self.base_len, sys::MS_SYNC) };
        if rc == 0 {
            Ok(())
        } else {
            Err(MmapError::System(sys::errno()))
        }
    }
}

impl Drop for Mapping {
    #[expect(unsafe_code, reason = "munmap releases the mapping exactly once on drop")]
    fn drop(&mut self) {
        // SAFETY: `base`/`base_len` is the exact span returned by `mmap`;
        // `munmap` is called once, here, at end of life.
        unsafe {
            munmap(self.base, self.base_len);
        }
    }
}
