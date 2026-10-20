//! Application state and exit contracts.

use alloc::string::String;
use core::fmt;

/// Application lifecycle state, including restart and suspension transitions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplicationState {
    /// Constructed but no host resources have been acquired.
    Created,
    /// Startup is acquiring host resources.
    Starting,
    /// The application can advance frames.
    Running,
    /// Host presentation is suspended while application state is retained.
    Suspended,
    /// Shutdown is releasing host resources.
    Stopping,
    /// Shutdown completed successfully.
    Stopped,
    /// A lifecycle operation failed; [`crate::EditorApplication::recover`] may
    /// rebuild the host when the failure is recoverable.
    Failed,
}

/// Reason the application was asked to stop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExitReason {
    /// The caller explicitly requested shutdown.
    Requested,
    /// A managed window closed.
    WindowClosed,
    /// A bounded run reached its frame limit.
    FrameLimit,
    /// A host or GPU failure made continued execution impossible.
    Failure,
}

/// Public exit state, separate from the lifecycle state so a pending request
/// can be observed before resources are torn down.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExitStatus {
    /// No exit has been requested.
    NotRequested,
    /// Shutdown has been requested but not completed.
    Requested(ExitReason),
    /// Shutdown completed.
    Completed(ExitReason),
    /// The application stopped because a lifecycle operation failed.
    Failed,
}

/// Compact report returned after a successful shutdown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShutdownReport {
    /// Reason recorded by the application.
    pub reason: ExitReason,
    /// Frame at which shutdown began.
    pub frame: u64,
}

/// Error from an invalid application lifecycle operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidStateError {
    /// Operation that was attempted.
    pub operation: String,
    /// State in which it was attempted.
    pub state: ApplicationState,
}

impl InvalidStateError {
    /// Creates a state error.
    pub fn new(operation: impl Into<String>, state: ApplicationState) -> Self {
        Self {
            operation: operation.into(),
            state,
        }
    }
}

impl fmt::Display for InvalidStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot {} while application is {:?}",
            self.operation, self.state
        )
    }
}

impl core::error::Error for InvalidStateError {}
