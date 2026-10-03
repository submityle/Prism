//! Command-line arguments (`argv`).
//!
//! Raw process arguments for the engine's CLI / configuration layer
//! (design doc §11 命令行解析原料). Windows wide-character `argv` is decoded by
//! `std` before it reaches here, so this facade is uniform across platforms.
//!
//! Two flavors are offered:
//! - [`args_os`] yields [`std::ffi::OsString`], preserving bytes that are not
//!   valid Unicode (the correct choice for paths passed on the command line).
//! - [`args`] yields lossily-decoded [`String`]s (convenient for flags and
//!   human-facing text) and never panics on non-Unicode input, unlike
//!   [`std::env::args`].

use std::ffi::OsString;
use std::path::PathBuf;

/// Iterate the process arguments as [`OsString`]s, preserving non-Unicode
/// bytes.
///
/// The first element is conventionally the program name/path as the OS passed
/// it (which may be empty or not an absolute path); use [`executable_path`] for
/// the resolved executable location.
pub fn args_os() -> impl Iterator<Item = OsString> {
    std::env::args_os()
}

/// Iterate the process arguments as lossily-decoded [`String`]s.
///
/// Non-Unicode bytes are replaced with `U+FFFD`. Unlike [`std::env::args`],
/// this never panics on invalid Unicode. For byte-exact arguments (e.g. file
/// paths), use [`args_os`].
pub fn args() -> impl Iterator<Item = String> {
    std::env::args_os().map(|arg| arg.to_string_lossy().into_owned())
}

/// Collect the process arguments into a [`Vec`] of lossily-decoded
/// [`String`]s.
#[must_use]
pub fn to_vec() -> Vec<String> {
    args().collect()
}

/// The number of process arguments, including the program name at index 0.
#[must_use]
pub fn count() -> usize {
    std::env::args_os().count()
}

/// The resolved absolute path of the running executable.
///
/// Unlike `argv[0]`, this is resolved by the OS and does not depend on how the
/// program was invoked.
///
/// # Errors
/// Propagates any [`std::io::Error`] from [`std::env::current_exe`] (for
/// example when the executable has been unlinked).
pub fn executable_path() -> std::io::Result<PathBuf> {
    std::env::current_exe()
}
