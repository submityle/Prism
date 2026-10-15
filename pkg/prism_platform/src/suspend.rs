//! Platform suspend/resume event values and capability reporting.
//!
//! This module deliberately contains no event loop and no OS callback. A
//! platform backend owns the callback and injects a timestamped
//! [`SuspendEvent`] here. Keeping the value type in the platform crate gives
//! `prism_app` and window backends one stable contract without making the time
//! kernel depend on an OS or a window library.

use crate::clock::MonotonicNanos;

/// Whether the current target has a built-in process-lifecycle callback path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlatformSuspendCapability {
    /// The host event loop can report process suspend/resume edges directly.
    /// This is the winit lifecycle path on iOS and Android.
    ProcessLifecycle,
    /// No portable process-suspend watcher is compiled in, but a native host
    /// can inject the same event through [`SuspendEvent`]. This is the honest
    /// desktop status until native sleep/power notifications are integrated.
    HostCallbackOnly,
    /// The target has no known suspend/resume event path in this crate.
    Unsupported,
}

/// Return the suspend/resume capability of the current compilation target.
#[must_use]
pub const fn suspend_capability() -> PlatformSuspendCapability {
    if cfg!(any(target_os = "ios", target_os = "android")) {
        PlatformSuspendCapability::ProcessLifecycle
    } else if cfg!(any(target_os = "macos", target_os = "windows")) {
        PlatformSuspendCapability::HostCallbackOnly
    } else {
        PlatformSuspendCapability::Unsupported
    }
}

/// The edge carried by a platform suspend/resume callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuspendEventKind {
    /// The host is suspending the process or its process lifecycle.
    Suspended,
    /// The host resumed the process or its process lifecycle.
    Resumed,
}

/// A real host lifecycle edge with the monotonic timestamp observed at the
/// callback boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuspendEvent {
    kind: SuspendEventKind,
    timestamp: MonotonicNanos,
}

impl SuspendEvent {
    /// Construct a suspend event from a host callback timestamp.
    #[must_use]
    pub const fn suspended(timestamp: MonotonicNanos) -> Self {
        Self {
            kind: SuspendEventKind::Suspended,
            timestamp,
        }
    }

    /// Construct a resume event from a host callback timestamp.
    #[must_use]
    pub const fn resumed(timestamp: MonotonicNanos) -> Self {
        Self {
            kind: SuspendEventKind::Resumed,
            timestamp,
        }
    }

    /// The edge kind.
    #[must_use]
    pub const fn kind(self) -> SuspendEventKind {
        self.kind
    }

    /// The monotonic timestamp captured at the host callback boundary.
    #[must_use]
    pub const fn timestamp(self) -> MonotonicNanos {
        self.timestamp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_preserves_kind_and_timestamp() {
        let suspended = SuspendEvent::suspended(MonotonicNanos(11));
        let resumed = SuspendEvent::resumed(MonotonicNanos(29));
        assert_eq!(suspended.kind(), SuspendEventKind::Suspended);
        assert_eq!(suspended.timestamp(), MonotonicNanos(11));
        assert_eq!(resumed.kind(), SuspendEventKind::Resumed);
        assert_eq!(resumed.timestamp(), MonotonicNanos(29));
    }

    #[test]
    fn capability_is_explicit_for_each_supported_target_family() {
        let capability = suspend_capability();
        #[cfg(any(target_os = "ios", target_os = "android"))]
        assert_eq!(capability, PlatformSuspendCapability::ProcessLifecycle);
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        assert_eq!(capability, PlatformSuspendCapability::HostCallbackOnly);
        #[cfg(not(any(
            target_os = "ios",
            target_os = "android",
            target_os = "macos",
            target_os = "windows"
        )))]
        assert_eq!(capability, PlatformSuspendCapability::Unsupported);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_compiles_with_host_callback_only_capability() {
        assert_eq!(
            suspend_capability(),
            PlatformSuspendCapability::HostCallbackOnly
        );
    }

    #[cfg(target_os = "ios")]
    #[test]
    fn ios_compiles_with_process_lifecycle_capability() {
        assert_eq!(
            suspend_capability(),
            PlatformSuspendCapability::ProcessLifecycle
        );
    }

    #[cfg(target_os = "android")]
    #[test]
    fn android_compiles_with_process_lifecycle_capability() {
        assert_eq!(
            suspend_capability(),
            PlatformSuspendCapability::ProcessLifecycle
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_compiles_with_host_callback_only_capability() {
        assert_eq!(
            suspend_capability(),
            PlatformSuspendCapability::HostCallbackOnly
        );
    }
}
