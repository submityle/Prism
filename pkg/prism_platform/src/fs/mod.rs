//! Cross-platform filesystem facade (desktop: Linux / macOS / Windows).
//!
//! This module is a thin, ergonomic wrapper over [`std::fs`] and
//! [`std::path`]. It gives the engine a single, testable surface for file,
//! directory, path, and standard-directory operations so upper crates never
//! reach for [`std::fs`] directly (see the single-entry invariant in the
//! design doc).
//!
//! The whole module requires the `std` feature; filesystem access is
//! impossible in a `no_std` build. Under `no_std` the module is absent and the
//! crate still compiles (the OS probe layer in M0 stays `core`-only).
//!
//! Every fallible operation returns [`Result`] and never panics on I/O errors.
//!
//! ## Layout
//! - [`file`]: whole-file read/write plus [`file::OpenOptions`] and
//!   [`file::Metadata`].
//! - [`dir`]: directory create/remove/list plus [`dir::walk`] and
//!   [`dir::DirEntry`].
//! - [`path`]: lexical [`path::normalize`] and small [`std::path::Path`]
//!   helpers.
//! - [`dirs`]: per-OS standard directories (config/data/cache/home/temp).

pub mod dir;
pub mod dirs;
pub mod file;
pub mod path;

use core::fmt;
use std::path::PathBuf;

/// Result type for every filesystem operation in this crate.
pub type Result<T> = core::result::Result<T, FsError>;

/// Error returned by filesystem operations.
///
/// Wraps [`std::io::Error`] for underlying I/O failures and adds a few
/// higher-level variants (invalid paths, unavailable standard directories) so
/// callers can match without string parsing.
#[derive(Debug)]
pub enum FsError {
    /// An underlying I/O error from [`std::fs`] or [`std::io`].
    Io(std::io::Error),
    /// A path could not be interpreted (for example non-UTF-8 where UTF-8 was
    /// required, or an empty path).
    InvalidPath(String),
    /// A requested standard directory could not be resolved on this host
    /// (missing environment variable or unsupported OS). The payload names the
    /// directory kind, for example `"config_dir"`.
    StandardDirUnavailable(&'static str),
}

impl fmt::Display for FsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FsError::Io(err) => write!(f, "filesystem I/O error: {err}"),
            FsError::InvalidPath(msg) => write!(f, "invalid path: {msg}"),
            FsError::StandardDirUnavailable(kind) => {
                write!(f, "standard directory unavailable: {kind}")
            }
        }
    }
}

impl std::error::Error for FsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FsError::Io(err) => Some(err),
            FsError::InvalidPath(_) | FsError::StandardDirUnavailable(_) => None,
        }
    }
}

impl From<std::io::Error> for FsError {
    fn from(err: std::io::Error) -> Self {
        FsError::Io(err)
    }
}

impl FsError {
    /// Returns the underlying [`std::io::Error`] if this is an I/O error.
    pub fn as_io(&self) -> Option<&std::io::Error> {
        match self {
            FsError::Io(err) => Some(err),
            _ => None,
        }
    }
}

pub use dir::{create_dir, create_dir_all, read_dir, remove_dir, remove_dir_all, walk, DirEntry};
pub use file::{Metadata, OpenOptions};

/// Re-export of [`std::path::PathBuf`] for convenience at the module root.
pub type OwnedPath = PathBuf;

#[cfg(test)]
mod tests;
