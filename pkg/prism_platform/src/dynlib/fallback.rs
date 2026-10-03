//! Fallback dynamic-loader backend for targets without a runtime loader
//! (for example wasm).
//!
//! There is no `dlopen`-equivalent here, so every operation honestly reports
//! [`DynlibError::Unsupported`] and [`super::supported`] returns `false`. This
//! keeps the facade compiling and callable everywhere while never pretending a
//! library was loaded.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::path::Path;

use super::{DynlibError, Result};

/// This build cannot load dynamic libraries at runtime.
pub(super) const SUPPORTED: bool = false;

/// A handle that can never be constructed on this backend.
pub(super) enum Handle {}

/// Always fails: there is no loader on this target.
pub(super) fn open(_path: &Path) -> Result<Handle> {
    Err(DynlibError::Unsupported)
}

/// Unreachable in practice: no [`Handle`] value can exist to pass in.
pub(super) fn symbol(handle: &Handle, _name: &str) -> Result<NonNull<c_void>> {
    match *handle {}
}

/// Unreachable in practice: no [`Handle`] value can exist to pass in.
pub(super) fn close(handle: Handle) -> Result<()> {
    match handle {}
}
