//! The backend → kernel command result channel.
//!
//! When `desired != realized` (the OS clamps a size, downgrades a refresh
//! rate, denies fullscreen, ...), the backend reports the real outcome and
//! reason here instead of silently pretending the request landed. The kernel
//! must treat the reported [`RealizedWindowDelta`] — not the requested values
//! — as truth (`prism_window_refactor_zh.md` §4.3 law 1; winit design §5.3).

use crate::command::CommandSequence;
use crate::geometry::{PhysicalPosition, PhysicalSize};
use crate::mode::{PresentMode, WindowMode};
use crate::window::WindowId;

/// A platform capability a command may require.
///
/// Reported inside [`CommandStatus::Unsupported`] so the kernel can pick a
/// documented degrade path rather than guessing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Capability {
    /// More than one top-level window.
    MultiWindow,
    /// HDR output.
    Hdr,
    /// Variable refresh rate.
    Vrr,
    /// Exclusive (mode-switching) fullscreen.
    ExclusiveFullscreen,
    /// Borderless fullscreen.
    BorderlessFullscreen,
    /// Per-pixel transparency.
    Transparency,
    /// Always-on-top / window level control.
    AlwaysOnTop,
    /// Confining the cursor to the window.
    CursorConfine,
    /// Locking the cursor in place (relative mouse-look).
    CursorLock,
    /// IME enable/position control.
    ImeControl,
    /// Programmatic window drag/resize.
    DragWindow,
    /// Content protection (capture exclusion).
    ContentProtection,
    /// A specific present mode.
    PresentMode(PresentMode),
}

/// Why the platform adjusted a request instead of honoring it exactly.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AdjustmentReason {
    /// The requested inner size was clamped to platform/monitor bounds.
    SizeClamped,
    /// The requested position was clamped onto a valid monitor area.
    PositionClamped,
    /// The requested refresh rate was unavailable; a nearby one was used.
    RefreshRateClamped,
    /// Exclusive fullscreen was downgraded to borderless.
    FullscreenDowngraded,
    /// The requested present mode was substituted with a supported one.
    PresentModeSubstituted,
    /// The requested video mode was replaced with the closest match.
    VideoModeSubstituted,
}

/// Why the platform refused a command outright.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PlatformDenial {
    /// The action requires an active user gesture (e.g. fullscreen on web).
    NeedsUserGesture,
    /// The window must be foreground/active first (focus-stealing policy).
    ForegroundActivationRequired,
    /// The OS/user policy forbids the action.
    PolicyRestricted,
    /// A required OS permission has not been granted.
    PermissionDenied,
}

/// A hard failure executing a command.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WindowBackendError {
    /// Window creation failed.
    CreationFailed,
    /// The target window id does not exist in the backend.
    UnknownWindow,
    /// The render surface was lost and the command could not apply.
    SurfaceLost,
    /// A generic OS error occurred.
    OsError,
    /// The command timed out waiting on the OS.
    Timeout,
}

/// The outcome of a single command.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CommandStatus {
    /// Fully applied and realized.
    Applied,
    /// Accepted; the realized value will arrive via a later window event.
    AcceptedPending,
    /// Already in the requested state; nothing changed.
    NoChange,
    /// Applied, but the platform adjusted the request (see reason).
    Adjusted(AdjustmentReason),
    /// The platform does not support the required capability.
    Unsupported(Capability),
    /// The platform refused the command (see denial).
    Denied(PlatformDenial),
    /// The command failed to execute.
    Failed(WindowBackendError),
}

impl CommandStatus {
    /// Whether the command had its full intended effect (exactly or already).
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Applied | Self::NoChange)
    }

    /// Whether a later window event is expected to finish realizing this.
    #[must_use]
    pub const fn is_pending(self) -> bool {
        matches!(self, Self::AcceptedPending)
    }

    /// Whether the command did not land (unsupported, denied, or failed).
    #[must_use]
    pub const fn is_rejected(self) -> bool {
        matches!(
            self,
            Self::Unsupported(_) | Self::Denied(_) | Self::Failed(_)
        )
    }
}

/// What the OS actually changed, reported back so the kernel can reconcile
/// its Realized state with reality. Every field is optional: `None` means
/// "this facet was not touched by this command".
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct RealizedWindowDelta {
    /// New realized inner size, if it changed.
    pub inner_size: Option<PhysicalSize>,
    /// New realized outer position, if it changed.
    pub outer_position: Option<PhysicalPosition>,
    /// New realized scale factor (milli), if it changed.
    pub scale_factor_milli: Option<u32>,
    /// New realized present mode, if it changed.
    pub present_mode: Option<PresentMode>,
    /// New realized window mode, if it changed.
    pub window_mode: Option<WindowMode>,
    /// New realized visibility, if it changed.
    pub visible: Option<bool>,
    /// New realized focus, if it changed.
    pub focused: Option<bool>,
}

impl RealizedWindowDelta {
    /// An empty delta (nothing changed).
    pub const EMPTY: Self = Self {
        inner_size: None,
        outer_position: None,
        scale_factor_milli: None,
        present_mode: None,
        window_mode: None,
        visible: None,
        focused: None,
    };

    /// Whether no facet changed.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.inner_size.is_none()
            && self.outer_position.is_none()
            && self.scale_factor_milli.is_none()
            && self.present_mode.is_none()
            && self.window_mode.is_none()
            && self.visible.is_none()
            && self.focused.is_none()
    }
}

/// The backend's reply to one [`WindowCommandEnvelope`](crate::command::WindowCommandEnvelope).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WindowCommandResult {
    /// Echoes the command's sequence for correlation.
    pub sequence: CommandSequence,
    /// The affected window (`None` if a `Create` failed before id assignment).
    pub window: Option<WindowId>,
    /// The outcome.
    pub status: CommandStatus,
    /// What actually landed on the OS.
    pub realized_delta: RealizedWindowDelta,
}

impl WindowCommandResult {
    /// A clean success with no realized delta (e.g. [`CommandStatus::NoChange`]).
    #[must_use]
    pub const fn simple(
        sequence: CommandSequence,
        window: WindowId,
        status: CommandStatus,
    ) -> Self {
        Self {
            sequence,
            window: Some(window),
            status,
            realized_delta: RealizedWindowDelta::EMPTY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_classification() {
        assert!(CommandStatus::Applied.is_success());
        assert!(CommandStatus::NoChange.is_success());
        assert!(!CommandStatus::AcceptedPending.is_success());
        assert!(CommandStatus::AcceptedPending.is_pending());
        assert!(CommandStatus::Unsupported(Capability::Hdr).is_rejected());
        assert!(CommandStatus::Denied(PlatformDenial::NeedsUserGesture).is_rejected());
        assert!(CommandStatus::Failed(WindowBackendError::OsError).is_rejected());
        assert!(!CommandStatus::Adjusted(AdjustmentReason::SizeClamped).is_rejected());
    }

    #[test]
    fn empty_delta_detection() {
        assert!(RealizedWindowDelta::EMPTY.is_empty());
        let d = RealizedWindowDelta {
            inner_size: Some(PhysicalSize::new(100, 100)),
            ..RealizedWindowDelta::EMPTY
        };
        assert!(!d.is_empty());
    }

    #[test]
    fn simple_result_has_empty_delta() {
        let r =
            WindowCommandResult::simple(CommandSequence(3), WindowId(2), CommandStatus::Applied);
        assert_eq!(r.sequence, CommandSequence(3));
        assert_eq!(r.window, Some(WindowId(2)));
        assert!(r.realized_delta.is_empty());
    }
}
