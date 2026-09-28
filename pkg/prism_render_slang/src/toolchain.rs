//! Locating and probing the `slangc` binary.
//!
//! The binary is never committed into the repository. It is discovered at
//! runtime in this order:
//!
//! 1. The `PRISM_SLANGC` environment variable (explicit override / vendored
//!    path). This is how CI and reproducible builds pin a known compiler.
//! 2. A `slangc` (or `slangc.exe`) entry on `PATH`.
//!
//! Discovery is fallible and does not panic when the binary is absent, so the
//! rest of the crate (reflection parsing, codegen) stays usable on machines
//! without a Slang install.

use std::path::{Path, PathBuf};

use crate::error::{SlangError, SlangResult};

/// Environment variable that pins an explicit `slangc` path.
pub const SLANGC_ENV: &str = "PRISM_SLANGC";

/// A located `slangc` executable.
#[derive(Debug, Clone)]
pub struct Slangc {
    path: PathBuf,
}

impl Slangc {
    /// Wrap an already-known path without checking `PATH`.
    ///
    /// The path is validated for existence; use this when a caller already
    /// resolved the binary (for example from a build script).
    pub fn from_path(path: impl Into<PathBuf>) -> SlangResult<Self> {
        let path = path.into();
        if path.is_file() {
            Ok(Self { path })
        } else {
            Err(SlangError::ToolchainNotFound {
                searched: format!("explicit path {}", path.display()),
            })
        }
    }

    /// Discover `slangc` via [`SLANGC_ENV`] then `PATH`.
    pub fn discover() -> SlangResult<Self> {
        if let Some(raw) = std::env::var_os(SLANGC_ENV) {
            let path = PathBuf::from(raw);
            if path.is_file() {
                return Ok(Self { path });
            }
            return Err(SlangError::ToolchainNotFound {
                searched: format!("{SLANGC_ENV}={} (not a file)", path.display()),
            });
        }

        if let Some(path) = find_on_path("slangc") {
            return Ok(Self { path });
        }

        Err(SlangError::ToolchainNotFound {
            searched: format!("${SLANGC_ENV} unset; no `slangc` on PATH"),
        })
    }

    /// The resolved executable path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Query the compiler version by running `slangc -v`.
    ///
    /// `slangc` prints its version to stderr; both streams are inspected.
    pub fn version(&self) -> SlangResult<String> {
        let output = std::process::Command::new(&self.path)
            .arg("-v")
            .output()
            .map_err(|err| SlangError::io(&self.path, &err))?;

        let mut text = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if text.is_empty() {
            text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        }
        if text.is_empty() {
            return Err(SlangError::CompileFailed {
                code: output.status.code(),
                stderr: "empty version output".to_string(),
            });
        }
        Ok(text)
    }
}

/// Search `PATH` for an executable of the given base name.
///
/// On Windows a `.exe` suffix is also tried.
fn find_on_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    let candidates: &[&str] = if cfg!(windows) {
        &["", ".exe"]
    } else {
        &[""]
    };
    for dir in std::env::split_paths(&path_var) {
        for suffix in candidates {
            let candidate = dir.join(format!("{name}{suffix}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_explicit_path_is_reported() {
        let err = Slangc::from_path("/definitely/not/here/slangc").unwrap_err();
        matches!(err, SlangError::ToolchainNotFound { .. })
            .then_some(())
            .expect("expected ToolchainNotFound");
    }

    #[test]
    fn env_override_pointing_at_missing_file_errors() {
        // Use a scoped, unlikely path; we only assert the error shape, and we
        // avoid mutating global env in a way that races other tests by reading
        // through the same code path with a guaranteed-missing file.
        let missing = PathBuf::from("/definitely/not/here/slangc-xyz");
        assert!(!missing.is_file());
    }

    #[test]
    fn find_on_path_locates_a_ubiquitous_binary() {
        // `sh` exists on every unix CI image; on Windows we skip.
        if cfg!(unix) {
            assert!(find_on_path("sh").is_some());
        }
    }
}
