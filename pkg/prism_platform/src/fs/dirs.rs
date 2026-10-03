//! Per-OS standard directories, resolved with `std` and environment variables.
//!
//! Dependency-free (no `dirs` crate): Linux follows the XDG Base Directory
//! spec, macOS uses the `~/Library` conventions, and Windows uses the
//! `%APPDATA%` / `%LOCALAPPDATA%` / `%USERPROFILE%` variables. On an
//! unsupported OS the getters return [`FsError::StandardDirUnavailable`].
//!
//! These resolve base locations only; applications should append their own
//! vendor/app subdirectory before creating files.

use std::path::PathBuf;

use crate::fs::{FsError, Result};

/// Read an environment variable as a path, treating empty values as unset.
fn env_path(key: &str) -> Option<PathBuf> {
    match std::env::var_os(key) {
        Some(value) if !value.is_empty() => Some(PathBuf::from(value)),
        _ => None,
    }
}

/// The current user's home directory.
///
/// Uses `$HOME` on Unix and `%USERPROFILE%` on Windows.
pub fn home_dir() -> Result<PathBuf> {
    #[cfg(unix)]
    {
        env_path("HOME").ok_or(FsError::StandardDirUnavailable("home_dir"))
    }
    #[cfg(windows)]
    {
        env_path("USERPROFILE").ok_or(FsError::StandardDirUnavailable("home_dir"))
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err(FsError::StandardDirUnavailable("home_dir"))
    }
}

/// Resolve `home_dir()` joined with the given relative components.
#[cfg(target_os = "macos")]
fn home_subdir(parts: &[&str], kind: &'static str) -> Result<PathBuf> {
    let mut base = home_dir().map_err(|_| FsError::StandardDirUnavailable(kind))?;
    for part in parts {
        base.push(part);
    }
    Ok(base)
}

/// Resolve an XDG directory: `$key` if set, otherwise `$HOME/<fallback>`.
#[cfg(target_os = "linux")]
fn xdg_dir(key: &str, fallback: &str, kind: &'static str) -> Result<PathBuf> {
    if let Some(path) = env_path(key) {
        return Ok(path);
    }
    let mut base = home_dir().map_err(|_| FsError::StandardDirUnavailable(kind))?;
    base.push(fallback);
    Ok(base)
}

/// Directory for user-specific configuration files.
///
/// Linux: `$XDG_CONFIG_HOME` or `~/.config`. macOS:
/// `~/Library/Application Support`. Windows: `%APPDATA%`.
pub fn config_dir() -> Result<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        xdg_dir("XDG_CONFIG_HOME", ".config", "config_dir")
    }
    #[cfg(target_os = "macos")]
    {
        home_subdir(&["Library", "Application Support"], "config_dir")
    }
    #[cfg(target_os = "windows")]
    {
        env_path("APPDATA").ok_or(FsError::StandardDirUnavailable("config_dir"))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Err(FsError::StandardDirUnavailable("config_dir"))
    }
}

/// Directory for user-specific application data.
///
/// Linux: `$XDG_DATA_HOME` or `~/.local/share`. macOS:
/// `~/Library/Application Support`. Windows: `%APPDATA%`.
pub fn data_dir() -> Result<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        xdg_dir("XDG_DATA_HOME", ".local/share", "data_dir")
    }
    #[cfg(target_os = "macos")]
    {
        home_subdir(&["Library", "Application Support"], "data_dir")
    }
    #[cfg(target_os = "windows")]
    {
        env_path("APPDATA").ok_or(FsError::StandardDirUnavailable("data_dir"))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Err(FsError::StandardDirUnavailable("data_dir"))
    }
}

/// Directory for user-specific non-essential cached data.
///
/// Linux: `$XDG_CACHE_HOME` or `~/.cache`. macOS: `~/Library/Caches`.
/// Windows: `%LOCALAPPDATA%`.
pub fn cache_dir() -> Result<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        xdg_dir("XDG_CACHE_HOME", ".cache", "cache_dir")
    }
    #[cfg(target_os = "macos")]
    {
        home_subdir(&["Library", "Caches"], "cache_dir")
    }
    #[cfg(target_os = "windows")]
    {
        env_path("LOCALAPPDATA").ok_or(FsError::StandardDirUnavailable("cache_dir"))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Err(FsError::StandardDirUnavailable("cache_dir"))
    }
}

/// The system temporary-file directory (via [`std::env::temp_dir`]).
pub fn temp_dir() -> PathBuf {
    std::env::temp_dir()
}

/// Absolute path to the currently running executable.
pub fn current_exe() -> Result<PathBuf> {
    Ok(std::env::current_exe()?)
}

/// The process's current working directory.
pub fn current_dir() -> Result<PathBuf> {
    Ok(std::env::current_dir()?)
}
