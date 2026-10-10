//! Unix (Linux / macOS / BSD) dynamic-loader backend.
//!
//! Wraps `dlopen` / `dlsym` / `dlclose` / `dlerror` with
//! `RTLD_NOW | RTLD_LOCAL`: symbols are resolved eagerly at load so a broken
//! library fails fast, and the symbols stay local to this handle rather than
//! polluting the global namespace. This is the real, tested backend on the
//! host.
//!
//! Every FFI call is a dependency-free `extern` declaration resolved against
//! the C runtime that `std` already links (no `libc` crate), mirroring
//! [`crate::fs::mmap`] and [`crate::vm`].

use alloc::ffi::CString;
use core::ffi::{c_char, c_int, c_void};
use core::ptr::NonNull;
use std::os::unix::ffi::OsStrExt as _;
use std::path::Path;

use super::{DynlibError, Result};

#[expect(
    unsafe_code,
    reason = "dlopen/dlsym/dlclose/dlerror are the POSIX dynamic-loader entry points, declared as dependency-free C FFI"
)]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *mut c_char;
}

const RTLD_NOW: c_int = 0x2;

// `RTLD_LOCAL` is 0 on Linux/Android but 0x4 on macOS/BSD.
#[cfg(any(target_os = "linux", target_os = "android"))]
const RTLD_LOCAL: c_int = 0x0;
#[cfg(not(any(target_os = "linux", target_os = "android")))]
const RTLD_LOCAL: c_int = 0x4;

/// This build loads dynamic libraries through the real `dlopen` loader.
pub(super) const SUPPORTED: bool = true;

/// An owned `dlopen` handle.
pub(super) struct Handle {
    ptr: NonNull<c_void>,
}

/// Drain `dlerror` into an owned string (empty if the loader reported nothing).
#[expect(
    unsafe_code,
    reason = "dlerror returns a transient C string owned by the loader; we only read it"
)]
fn take_dlerror() -> String {
    // SAFETY: `dlerror` returns either null or a pointer to a NUL-terminated C
    // string valid until the next loader call on this thread; we copy it out
    // immediately into an owned `String`.
    unsafe {
        let p = dlerror();
        if p.is_null() {
            String::new()
        } else {
            core::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// `dlopen(path, RTLD_NOW | RTLD_LOCAL)`.
#[expect(
    unsafe_code,
    reason = "dlopen loads the shared object and runs its initializers"
)]
pub(super) fn open(path: &Path) -> Result<Handle> {
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| DynlibError::InvalidName(path.display().to_string()))?;
    // Clear any stale error before the call so `take_dlerror` is meaningful.
    let _ = take_dlerror();
    // SAFETY: `c_path` is a valid NUL-terminated path for the duration of the
    // call; the flags are the documented RTLD constants. The returned pointer
    // is checked for null before being wrapped.
    let raw = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
    match NonNull::new(raw) {
        Some(ptr) => Ok(Handle { ptr }),
        None => {
            let msg = take_dlerror();
            Err(DynlibError::System(if msg.is_empty() {
                format!("dlopen failed for {}", path.display())
            } else {
                msg
            }))
        }
    }
}

/// `dlsym(handle, name)`.
#[expect(
    unsafe_code,
    reason = "dlsym resolves a symbol address within the loaded object"
)]
pub(super) fn symbol(handle: &Handle, name: &str) -> Result<NonNull<c_void>> {
    let c_name = CString::new(name).map_err(|_| DynlibError::InvalidName(name.to_string()))?;
    let _ = take_dlerror();
    // SAFETY: `handle.ptr` is a live `dlopen` handle owned by `handle`;
    // `c_name` is a valid NUL-terminated symbol name for the call. The result
    // is validated against null (and `dlerror`) before use.
    let raw = unsafe { dlsym(handle.ptr.as_ptr(), c_name.as_ptr()) };
    match NonNull::new(raw) {
        Some(ptr) => Ok(ptr),
        None => {
            let msg = take_dlerror();
            if msg.is_empty() {
                Err(DynlibError::SymbolNotFound(name.to_string()))
            } else {
                Err(DynlibError::System(msg))
            }
        }
    }
}

/// `dlclose(handle)`.
#[expect(unsafe_code, reason = "dlclose unloads the shared object exactly once")]
pub(super) fn close(handle: Handle) -> Result<()> {
    // SAFETY: `handle.ptr` is a live handle consumed by value here, so
    // `dlclose` runs exactly once for it.
    let rc = unsafe { dlclose(handle.ptr.as_ptr()) };
    if rc == 0 {
        Ok(())
    } else {
        let msg = take_dlerror();
        Err(DynlibError::System(if msg.is_empty() {
            "dlclose failed".to_string()
        } else {
            msg
        }))
    }
}
