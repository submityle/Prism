//! Windows dynamic-loader backend.
//!
//! Wraps `LoadLibraryW` / `GetProcAddress` / `FreeLibrary` from `kernel32`.
//! Paths are passed as UTF-16; symbol names as ANSI (the `GetProcAddress`
//! convention). Compiled behind `cfg(windows)`.
//!
//! **Status: written but not yet validated on a Windows host.** The code path
//! is honest (no stubs) but should be treated as provisional until exercised on
//! real Windows, mirroring the Windows backends in [`crate::fs::mmap`] and
//! [`crate::vm`].

use core::ffi::{c_void, CStr};
use core::ptr::NonNull;
use std::ffi::CString;
use std::os::windows::ffi::OsStrExt as _;
use std::path::Path;

use super::{DynlibError, Result};

type Hmodule = *mut c_void;
type Farproc = *mut c_void;

#[expect(
    unsafe_code,
    reason = "LoadLibraryW/GetProcAddress/FreeLibrary/GetLastError are the Win32 loader entry points, declared as dependency-free FFI"
)]
unsafe extern "system" {
    fn LoadLibraryW(filename: *const u16) -> Hmodule;
    fn GetProcAddress(module: Hmodule, name: *const u8) -> Farproc;
    fn FreeLibrary(module: Hmodule) -> i32;
    fn GetLastError() -> u32;
}

/// This build loads dynamic libraries through the real Win32 loader.
pub(super) const SUPPORTED: bool = true;

/// An owned `LoadLibraryW` module handle.
pub(super) struct Handle {
    module: NonNull<c_void>,
}

/// Encode a path as a NUL-terminated UTF-16 sequence for `LoadLibraryW`.
fn wide(path: &Path) -> Vec<u16> {
    path.as_os_str().encode_wide().chain(core::iter::once(0)).collect()
}

/// `LoadLibraryW(path)`.
#[expect(unsafe_code, reason = "LoadLibraryW maps the module and runs its entry point")]
pub(super) fn open(path: &Path) -> Result<Handle> {
    let wpath = wide(path);
    // SAFETY: `wpath` is a valid NUL-terminated UTF-16 string for the call.
    // The returned handle is checked for null before being wrapped.
    let module = unsafe { LoadLibraryW(wpath.as_ptr()) };
    match NonNull::new(module) {
        Some(module) => Ok(Handle { module }),
        None => {
            // SAFETY: `GetLastError` takes no arguments and reads thread-local
            // error state.
            let code = unsafe { GetLastError() };
            Err(DynlibError::System(format!(
                "LoadLibraryW failed for {} (error {code})",
                path.display()
            )))
        }
    }
}

/// `GetProcAddress(module, name)`.
#[expect(unsafe_code, reason = "GetProcAddress resolves an exported symbol address")]
pub(super) fn symbol(handle: &Handle, name: &str) -> Result<NonNull<c_void>> {
    let c_name = CString::new(name).map_err(|_| DynlibError::InvalidName(name.to_string()))?;
    let _ = CStr::from_bytes_with_nul(c_name.as_bytes_with_nul());
    // SAFETY: `handle.module` is a live module handle; `c_name` is a valid
    // NUL-terminated ANSI symbol name. The result is validated against null.
    let proc = unsafe { GetProcAddress(handle.module.as_ptr(), c_name.as_ptr().cast::<u8>()) };
    match NonNull::new(proc) {
        Some(ptr) => Ok(ptr),
        None => Err(DynlibError::SymbolNotFound(name.to_string())),
    }
}

/// `FreeLibrary(module)`.
#[expect(unsafe_code, reason = "FreeLibrary unloads the module exactly once")]
pub(super) fn close(handle: Handle) -> Result<()> {
    // SAFETY: `handle.module` is a live handle consumed by value, so
    // `FreeLibrary` runs exactly once for it.
    let ok = unsafe { FreeLibrary(handle.module.as_ptr()) };
    if ok != 0 {
        Ok(())
    } else {
        // SAFETY: see `open`.
        let code = unsafe { GetLastError() };
        Err(DynlibError::System(format!("FreeLibrary failed (error {code})")))
    }
}
