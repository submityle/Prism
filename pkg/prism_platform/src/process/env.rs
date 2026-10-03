//! Environment-variable access (design doc §11 环境变量).
//!
//! Read / iterate / set / remove process environment variables, used for
//! configuration overrides such as `PRISM_LOG` or `PRISM_RENDER_BACKEND`.
//!
//! ## Reads are safe; writes are not
//! Reading and iterating the environment are ordinary safe functions. Mutating
//! it ([`set_var`] / [`remove_var`]) is **`unsafe`** on this crate's Rust 2024
//! edition — and genuinely hazardous — because the process environment is
//! global mutable state with no internal synchronization: a write that races
//! with a read (including reads performed inside the C library by other threads,
//! e.g. `getenv`, `localtime`, or name resolution) is undefined behavior. The
//! facade preserves that contract instead of hiding it; the caller must ensure
//! no other thread touches the environment concurrently. The intended use is at
//! single-threaded startup, before worker threads are spawned.
//!
//! Both the raw [`OsString`] forms and lossy [`String`] conveniences are
//! provided, mirroring [`crate::process::args`].

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

/// Read an environment variable as a raw [`OsString`], preserving non-Unicode
/// bytes. Returns [`None`] if the variable is unset.
#[must_use]
pub fn var_os<K: AsRef<OsStr>>(key: K) -> Option<OsString> {
    std::env::var_os(key)
}

/// Read an environment variable as a lossily-decoded [`String`].
///
/// Non-Unicode bytes are replaced with `U+FFFD`. Returns [`None`] if the
/// variable is unset. For byte-exact values, use [`var_os`].
#[must_use]
pub fn var<K: AsRef<OsStr>>(key: K) -> Option<String> {
    std::env::var_os(key).map(|value| value.to_string_lossy().into_owned())
}

/// Returns `true` if the named environment variable is set (even if empty).
#[must_use]
pub fn is_set<K: AsRef<OsStr>>(key: K) -> bool {
    std::env::var_os(key).is_some()
}

/// Iterate every environment variable as raw `(OsString, OsString)` pairs.
pub fn vars_os() -> impl Iterator<Item = (OsString, OsString)> {
    std::env::vars_os()
}

/// Iterate every environment variable as lossily-decoded `(String, String)`
/// pairs.
pub fn vars() -> impl Iterator<Item = (String, String)> {
    std::env::vars_os().map(|(key, value)| {
        (
            key.to_string_lossy().into_owned(),
            value.to_string_lossy().into_owned(),
        )
    })
}

/// Set an environment variable for this process.
///
/// # Safety
/// The process environment is global mutable state shared with the C library
/// and every thread. The caller must guarantee that no other thread reads or
/// writes the environment (including indirectly via libc functions such as
/// `getenv`, `setlocale`, `localtime`, or `getaddrinfo`) for the duration of
/// this call. Violating this is undefined behavior. In practice: only mutate
/// the environment during single-threaded startup, before spawning workers.
#[expect(
    unsafe_code,
    reason = "std::env::set_var is unsafe on edition 2024; the global-env mutation hazard is forwarded to this facade's caller"
)]
pub unsafe fn set_var<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {
    // SAFETY: the caller upholds this function's documented contract that no
    // other thread accesses the environment concurrently; we simply forward to
    // the std primitive under that same precondition.
    unsafe { std::env::set_var(key, value) }
}

/// Remove an environment variable from this process.
///
/// # Safety
/// Same contract as [`set_var`]: the caller must guarantee no concurrent
/// environment access from any thread (including libc internals) for the
/// duration of the call. Violating this is undefined behavior.
#[expect(
    unsafe_code,
    reason = "std::env::remove_var is unsafe on edition 2024; the global-env mutation hazard is forwarded to this facade's caller"
)]
pub unsafe fn remove_var<K: AsRef<OsStr>>(key: K) {
    // SAFETY: the caller upholds this function's documented contract that no
    // other thread accesses the environment concurrently; we simply forward to
    // the std primitive under that same precondition.
    unsafe { std::env::remove_var(key) }
}

/// The process's current working directory.
///
/// # Errors
/// Propagates any [`std::io::Error`] from [`std::env::current_dir`] (for
/// example when the directory has been removed or is inaccessible).
pub fn current_dir() -> std::io::Result<PathBuf> {
    std::env::current_dir()
}

/// Change the process's current working directory.
///
/// This is process-global state; prefer per-child `cwd` configuration
/// ([`crate::process::Command::current_dir`]) over mutating the parent's cwd.
///
/// # Errors
/// Propagates any [`std::io::Error`] from [`std::env::set_current_dir`] (for
/// example when the path does not exist or is not a directory).
pub fn set_current_dir<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<()> {
    std::env::set_current_dir(path)
}
