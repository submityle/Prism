//! Error and result types for the Slang toolchain.

use core::fmt;
use std::path::PathBuf;

/// Result alias used throughout this crate.
pub type SlangResult<T> = Result<T, SlangError>;

/// Failures that can occur while locating, invoking, or interpreting `slangc`.
#[derive(Debug)]
#[non_exhaustive]
pub enum SlangError {
    /// The `slangc` binary could not be located via env override or `PATH`.
    ToolchainNotFound {
        /// Human-readable description of the lookup that was attempted.
        searched: String,
    },
    /// A `slangc` invocation failed to spawn or exited non-zero.
    CompileFailed {
        /// The status code reported by the process, if any.
        code: Option<i32>,
        /// Captured standard error output.
        stderr: String,
    },
    /// A required output artifact was not produced by `slangc`.
    MissingArtifact {
        /// Path that was expected to exist after compilation.
        path: PathBuf,
    },
    /// Reflection JSON could not be parsed into the ABI model.
    ReflectionParse {
        /// Explanation of what was malformed or unsupported.
        detail: String,
    },
    /// An I/O error while reading sources or writing artifacts.
    Io {
        /// The path involved in the failing operation, if known.
        path: Option<PathBuf>,
        /// The underlying error message.
        detail: String,
    },
}

impl fmt::Display for SlangError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlangError::ToolchainNotFound { searched } => {
                write!(f, "slangc not found ({searched})")
            }
            SlangError::CompileFailed { code, stderr } => match code {
                Some(code) => write!(f, "slangc exited with code {code}: {stderr}"),
                None => write!(f, "slangc terminated by signal: {stderr}"),
            },
            SlangError::MissingArtifact { path } => {
                write!(f, "expected slangc artifact was not produced: {}", path.display())
            }
            SlangError::ReflectionParse { detail } => {
                write!(f, "failed to parse Slang reflection: {detail}")
            }
            SlangError::Io { path, detail } => match path {
                Some(path) => write!(f, "io error at {}: {detail}", path.display()),
                None => write!(f, "io error: {detail}"),
            },
        }
    }
}

impl std::error::Error for SlangError {}

impl SlangError {
    /// Build an [`SlangError::Io`] from a path context and an underlying error.
    pub fn io(path: impl Into<PathBuf>, err: &std::io::Error) -> Self {
        SlangError::Io {
            path: Some(path.into()),
            detail: err.to_string(),
        }
    }
}
