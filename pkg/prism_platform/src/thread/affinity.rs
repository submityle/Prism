//! Best-effort CPU-core pinning for the current thread.
//!
//! Pinning a thread to a fixed core improves cache locality and reduces
//! cross-core migration for latency-sensitive worker loops. Support is
//! inherently platform-dependent:
//!
//! - **Linux**: implemented via the `sched_setaffinity` system call (reached
//!   through a dependency-free `extern "C"` declaration resolved against the C
//!   runtime that `std` already links; no `libc` crate).
//! - **Windows**: implemented via `SetThreadAffinityMask` from `kernel32`.
//! - **macOS**: *not supported*. macOS exposes no portable per-core pinning
//!   API — `THREAD_AFFINITY_POLICY` is only an advisory hint on Intel and is a
//!   no-op on Apple Silicon — so this crate honestly returns
//!   [`AffinityError::Unsupported`] rather than faking success. On macOS the
//!   recommended mechanism is Quality-of-Service classes, which do not pin.
//! - **Other OSes**: return [`AffinityError::Unsupported`].
//!
//! Use [`affinity_supported`] to query whether pinning can succeed on this
//! build before relying on it.

use core::fmt;

/// Why a thread-affinity request did not take effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffinityError {
    /// This operating system does not provide per-core thread pinning (for
    /// example macOS), so no pinning was attempted.
    Unsupported,
    /// A requested core index was empty or out of the representable range.
    InvalidCore,
    /// The OS rejected the request; carries the platform error code
    /// (`errno` on Linux, `GetLastError` on Windows).
    SystemError(i32),
}

impl fmt::Display for AffinityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AffinityError::Unsupported => {
                write!(f, "thread affinity is unsupported on this platform")
            }
            AffinityError::InvalidCore => write!(f, "invalid or empty CPU core selection"),
            AffinityError::SystemError(code) => {
                write!(f, "OS rejected affinity request (code {code})")
            }
        }
    }
}

impl std::error::Error for AffinityError {}

/// Pin the calling thread to a single logical core.
///
/// Returns [`Ok`] once the affinity mask is applied, or an [`AffinityError`]
/// (notably [`AffinityError::Unsupported`] on macOS and other platforms that
/// cannot pin).
pub fn set_current_thread_affinity(core: usize) -> Result<(), AffinityError> {
    imp::set_mask(&[core])
}

/// Pin the calling thread to the set of logical cores in `cores`.
///
/// An empty slice yields [`AffinityError::InvalidCore`]. Returns
/// [`AffinityError::Unsupported`] on platforms that cannot pin.
pub fn set_current_thread_affinity_mask(cores: &[usize]) -> Result<(), AffinityError> {
    imp::set_mask(cores)
}

/// Returns `true` if this build can actually pin threads to cores.
pub fn affinity_supported() -> bool {
    imp::SUPPORTED
}

#[cfg(target_os = "linux")]
#[expect(
    unsafe_code,
    reason = "CPU affinity requires the sched_setaffinity syscall via a dependency-free C FFI declaration"
)]
mod imp {
    use super::AffinityError;

    /// Number of 64-bit words in the kernel CPU-set mask (1024 CPUs).
    const WORDS: usize = 16;

    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
        fn __errno_location() -> *mut i32;
    }

    pub(super) const SUPPORTED: bool = true;

    pub(super) fn set_mask(cores: &[usize]) -> Result<(), AffinityError> {
        if cores.is_empty() {
            return Err(AffinityError::InvalidCore);
        }
        let mut mask = [0u64; WORDS];
        for &core in cores {
            if core >= WORDS * 64 {
                return Err(AffinityError::InvalidCore);
            }
            mask[core / 64] |= 1u64 << (core % 64);
        }
        // SAFETY: `mask` is a fully-initialized `[u64; WORDS]` living for the
        // whole call; we pass its exact byte length and a read-only pointer the
        // kernel only reads. `pid = 0` targets the calling thread.
        let rc = unsafe { sched_setaffinity(0, core::mem::size_of_val(&mask), mask.as_ptr()) };
        if rc == 0 {
            Ok(())
        } else {
            // SAFETY: `__errno_location` returns a valid pointer to this
            // thread's `errno`, which we only read.
            let errno = unsafe { *__errno_location() };
            Err(AffinityError::SystemError(errno))
        }
    }
}

#[cfg(target_os = "windows")]
#[expect(
    unsafe_code,
    reason = "CPU affinity requires SetThreadAffinityMask from the Win32 API via a dependency-free FFI declaration"
)]
mod imp {
    use super::AffinityError;

    /// Pointer-sized unsigned integer, matching Win32 `DWORD_PTR`.
    type DwordPtr = usize;
    /// Opaque Win32 `HANDLE`.
    type Handle = *mut core::ffi::c_void;

    unsafe extern "system" {
        fn GetCurrentThread() -> Handle;
        fn SetThreadAffinityMask(thread: Handle, mask: DwordPtr) -> DwordPtr;
        fn GetLastError() -> u32;
    }

    pub(super) const SUPPORTED: bool = true;

    pub(super) fn set_mask(cores: &[usize]) -> Result<(), AffinityError> {
        if cores.is_empty() {
            return Err(AffinityError::InvalidCore);
        }
        let bits = core::mem::size_of::<DwordPtr>() * 8;
        let mut mask: DwordPtr = 0;
        for &core in cores {
            if core >= bits {
                return Err(AffinityError::InvalidCore);
            }
            mask |= 1 << core;
        }
        // SAFETY: `GetCurrentThread` returns a pseudo-handle valid for the
        // calling thread; `SetThreadAffinityMask` only reads the handle and the
        // scalar mask. Both are plain FFI calls with no borrowed memory.
        let prev = unsafe {
            let handle = GetCurrentThread();
            SetThreadAffinityMask(handle, mask)
        };
        if prev != 0 {
            Ok(())
        } else {
            // SAFETY: `GetLastError` only reads this thread's last-error code.
            let code = unsafe { GetLastError() };
            Err(AffinityError::SystemError(code as i32))
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
mod imp {
    use super::AffinityError;

    pub(super) const SUPPORTED: bool = false;

    pub(super) fn set_mask(cores: &[usize]) -> Result<(), AffinityError> {
        if cores.is_empty() {
            return Err(AffinityError::InvalidCore);
        }
        Err(AffinityError::Unsupported)
    }
}
