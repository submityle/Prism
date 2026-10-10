//! Backend error and diagnostic types (§19).
//!
//! The backend never panics on recoverable platform failures; it surfaces them
//! as [`BackendError`] so the host can log, retry, or degrade. Only truly
//! unrecoverable invariant violations are allowed to abort.

use core::fmt;

/// A recoverable failure raised while driving the winit backend.
#[derive(Debug)]
pub enum BackendError {
    /// The OS event loop could not be constructed.
    EventLoopCreation(String),
    /// A window could not be created from the requested attributes.
    WindowCreation(String),
    /// A command referenced a window id with no live platform window.
    UnknownWindow,
    /// The backend was asked to create a window that already exists.
    DuplicateWindow,
    /// A command is not supported on the current platform; carries a human
    /// readable reason for diagnostics.
    Unsupported(&'static str),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EventLoopCreation(why) => write!(f, "failed to create event loop: {why}"),
            Self::WindowCreation(why) => write!(f, "failed to create window: {why}"),
            Self::UnknownWindow => write!(f, "command referenced an unknown window"),
            Self::DuplicateWindow => write!(f, "window already exists"),
            Self::Unsupported(what) => write!(f, "unsupported on this platform: {what}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Convenience result alias for backend operations.
pub type BackendResult<T> = Result<T, BackendError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_human_readable() {
        assert_eq!(
            BackendError::UnknownWindow.to_string(),
            "command referenced an unknown window"
        );
        assert_eq!(
            BackendError::Unsupported("exclusive fullscreen").to_string(),
            "unsupported on this platform: exclusive fullscreen"
        );
    }
}
