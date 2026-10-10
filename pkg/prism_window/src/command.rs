//! The kernel/app → backend command channel.
//!
//! The ABI is strictly two-directional: events in ([`WindowEventEnvelope`]),
//! commands out ([`WindowCommandEnvelope`]). Commands are the *only* way the
//! kernel asks the OS to change a window; the backend acknowledges each one on
//! the result channel ([`WindowCommandResult`](crate::result::WindowCommandResult)).
//! Commands are a cold path, so they may own allocations (e.g. [`Arc<str>`] for
//! titles); hot-path events stay `Copy`.
//!
//! The command vocabulary is owned by the winit backend design (§5.2); this
//! kernel module mirrors it in backend-agnostic kernel types.
//!
//! [`WindowEventEnvelope`]: crate::envelope::WindowEventEnvelope

use alloc::sync::Arc;

use crate::capabilities::HdrConfig;
use crate::cursor::CursorOptions;
use crate::envelope::MonotonicTimestamp;
use crate::geometry::{PhysicalPosition, PhysicalSize};
use crate::mode::{PresentMode, WindowLevel, WindowMode, WindowTheme};
use crate::monitor::{MonitorId, VideoMode};
use crate::window::{WindowAttributes, WindowId};

/// A monotonic command sequence number, paired 1:1 with a
/// [`WindowCommandResult`](crate::result::WindowCommandResult).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct CommandSequence(pub u64);

impl CommandSequence {
    /// The first command sequence value.
    pub const FIRST: Self = Self(0);

    /// The raw value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Allocates monotonic [`CommandSequence`]s on the simulation side.
///
/// One sequencer per backend connection; the backend echoes the sequence back
/// in the matching result so the kernel can correlate command and outcome.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct CommandSequencer {
    next: u64,
}

impl CommandSequencer {
    /// A fresh sequencer starting at [`CommandSequence::FIRST`].
    #[must_use]
    pub const fn new() -> Self {
        Self { next: 0 }
    }

    /// Returns the next sequence and advances the counter.
    #[must_use]
    pub fn next_sequence(&mut self) -> CommandSequence {
        let seq = CommandSequence(self.next);
        self.next = self.next.wrapping_add(1);
        seq
    }

    /// Builds a fully-stamped command envelope targeting `target`.
    #[must_use]
    pub fn envelope(
        &mut self,
        issued_at: MonotonicTimestamp,
        target: WindowTarget,
        command: WindowCommand,
    ) -> WindowCommandEnvelope {
        WindowCommandEnvelope {
            sequence: self.next_sequence(),
            issued_at,
            target,
            command,
        }
    }
}

/// The window a command addresses.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum WindowTarget {
    /// An existing, realized window.
    Existing(WindowId),
    /// A [`WindowCommand::Create`] carrying the id the kernel pre-allocated for
    /// the new window (so results can be correlated before the OS confirms).
    New(WindowId),
}

impl WindowTarget {
    /// The underlying [`WindowId`] regardless of variant.
    #[must_use]
    pub const fn window_id(self) -> WindowId {
        match self {
            Self::Existing(id) | Self::New(id) => id,
        }
    }

    /// Whether this targets a not-yet-created window.
    #[must_use]
    pub const fn is_new(self) -> bool {
        matches!(self, Self::New(_))
    }
}

/// Which edge or corner a programmatic resize-drag pulls.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ResizeDirection {
    /// Drag the right edge.
    East,
    /// Drag the top edge.
    North,
    /// Drag the top-right corner.
    NorthEast,
    /// Drag the top-left corner.
    NorthWest,
    /// Drag the bottom edge.
    South,
    /// Drag the bottom-right corner.
    SouthEast,
    /// Drag the bottom-left corner.
    SouthWest,
    /// Drag the left edge.
    West,
}

/// How insistently to request user attention.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum AttentionKind {
    /// A gentle hint (e.g. taskbar flash once).
    #[default]
    Informational,
    /// A persistent, critical request (e.g. keep flashing).
    Critical,
}

/// A fullscreen/windowed mode request, including the display to target.
///
/// `monitor`/`video_mode` are only consulted for the fullscreen variants; for
/// windowed mode they are ignored.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct WindowModeRequest {
    /// The desired window mode.
    pub mode: WindowMode,
    /// The monitor to go fullscreen on (`None` = current/primary).
    pub monitor: Option<MonitorId>,
    /// The exact video mode for exclusive fullscreen (`None` = backend picks).
    pub video_mode: Option<VideoMode>,
}

impl WindowModeRequest {
    /// A plain windowed request.
    pub const WINDOWED: Self = Self {
        mode: WindowMode::Windowed,
        monitor: None,
        video_mode: None,
    };

    /// A mode request with no specific monitor/video-mode selection.
    #[must_use]
    pub const fn simple(mode: WindowMode) -> Self {
        Self {
            mode,
            monitor: None,
            video_mode: None,
        }
    }
}

/// A control request for the platform IME (text composition).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ImeRequest {
    /// Enable (`true`) or disable (`false`) IME input for the window.
    SetAllowed(bool),
    /// Place the candidate/composition box at a physical rectangle in the
    /// window (so the OS positions the IME UI correctly).
    SetArea {
        /// Top-left of the text cursor area, in physical pixels.
        position: PhysicalPosition,
        /// Size of the text cursor area, in physical pixels.
        size: PhysicalSize,
    },
}

/// A single backend command. Fine-grained so the backend can realize exactly
/// one OS mutation and report its precise outcome.
///
/// Not `Copy`: cold-path payloads such as [`WindowCommand::SetTitle`] own
/// allocations. Clone is cheap (the title is an [`Arc<str>`]).
#[derive(Clone, PartialEq, Debug)]
pub enum WindowCommand {
    /// Create a window from the desired attributes (target is a `New` id).
    Create(WindowAttributes),
    /// Destroy the target window.
    Destroy,
    /// Set the title bar / taskbar text.
    SetTitle(Arc<str>),
    /// Show or hide the window.
    SetVisible(bool),
    /// Move the window's top-left to a desktop position (physical pixels).
    SetOuterPosition(PhysicalPosition),
    /// Request a new inner (drawable) size; the OS may clamp or reject it.
    RequestInnerSize(PhysicalSize),
    /// Set/clear the minimum inner size constraint.
    SetMinInnerSize(Option<PhysicalSize>),
    /// Set/clear the maximum inner size constraint.
    SetMaxInnerSize(Option<PhysicalSize>),
    /// Allow or forbid user resizing.
    SetResizable(bool),
    /// Show or hide OS decorations (title bar, borders).
    SetDecorations(bool),
    /// Change the desktop stacking level.
    SetLevel(WindowLevel),
    /// Change windowed/fullscreen mode (and target display).
    SetMode(WindowModeRequest),
    /// Force a theme, or `None` to follow the system.
    SetTheme(Option<WindowTheme>),
    /// Change the swapchain present mode (v-sync behavior).
    SetPresentMode(PresentMode),
    /// Update cursor appearance, visibility, and grab mode.
    SetCursor(CursorOptions),
    /// Control the IME.
    SetIme(ImeRequest),
    /// Configure HDR output (or disable it).
    SetHdr(HdrConfig),
    /// Ask the OS to focus/activate the window (may require a user gesture).
    RequestFocus,
    /// Request user attention, or `None` to cancel an outstanding request.
    RequestAttention(Option<AttentionKind>),
    /// Ask for a redraw of the window surface.
    RequestRedraw,
    /// Start an OS-driven window move (follows the pointer until release).
    BeginDragWindow,
    /// Start an OS-driven resize-drag in a direction.
    BeginResizeDrag(ResizeDirection),
    /// Mark surface contents protected (excluded from screen capture).
    SetContentProtected(bool),
}

impl WindowCommand {
    /// Builds a [`WindowCommand::SetTitle`] from any string-like value.
    #[must_use]
    pub fn set_title(title: &str) -> Self {
        Self::SetTitle(Arc::from(title))
    }

    /// Whether this command creates a window (its target must be a `New` id).
    #[must_use]
    pub const fn is_create(&self) -> bool {
        matches!(self, Self::Create(_))
    }

    /// Whether this command destroys a window (collapses later commands for
    /// the same target during coalescing; see refactor §4.3).
    #[must_use]
    pub const fn is_destroy(&self) -> bool {
        matches!(self, Self::Destroy)
    }
}

/// A command addressed to a specific window, with sequence and issue time.
#[derive(Clone, PartialEq, Debug)]
pub struct WindowCommandEnvelope {
    /// Monotonic sequence, paired with the result channel.
    pub sequence: CommandSequence,
    /// When the command was issued (monotonic timeline).
    pub issued_at: MonotonicTimestamp,
    /// The window the command targets (`New` for `Create`).
    pub target: WindowTarget,
    /// The command payload.
    pub command: WindowCommand,
}

impl WindowCommandEnvelope {
    /// Builds a command envelope.
    #[must_use]
    pub const fn new(
        sequence: CommandSequence,
        issued_at: MonotonicTimestamp,
        target: WindowTarget,
        command: WindowCommand,
    ) -> Self {
        Self {
            sequence,
            issued_at,
            target,
            command,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequencer_is_monotonic() {
        let mut seq = CommandSequencer::new();
        assert_eq!(seq.next_sequence(), CommandSequence(0));
        assert_eq!(seq.next_sequence(), CommandSequence(1));
        assert_eq!(seq.next_sequence(), CommandSequence(2));
    }

    #[test]
    fn sequencer_builds_envelope_with_increasing_sequence() {
        let mut seq = CommandSequencer::new();
        let e1 = seq.envelope(
            MonotonicTimestamp::from_nanos(1),
            WindowTarget::Existing(WindowId(1)),
            WindowCommand::RequestRedraw,
        );
        let e2 = seq.envelope(
            MonotonicTimestamp::from_nanos(2),
            WindowTarget::Existing(WindowId(1)),
            WindowCommand::SetVisible(false),
        );
        assert_eq!(e1.sequence, CommandSequence(0));
        assert_eq!(e2.sequence, CommandSequence(1));
        assert!(e1.sequence < e2.sequence);
    }

    #[test]
    fn set_title_uses_arc_and_is_cheap_to_clone() {
        let cmd = WindowCommand::set_title("Prism Editor");
        let cloned = cmd.clone();
        assert_eq!(cmd, cloned);
        match cmd {
            WindowCommand::SetTitle(s) => assert_eq!(&*s, "Prism Editor"),
            _ => panic!("expected SetTitle"),
        }
    }

    #[test]
    fn target_helpers() {
        let new = WindowTarget::New(WindowId(7));
        let existing = WindowTarget::Existing(WindowId(7));
        assert!(new.is_new());
        assert!(!existing.is_new());
        assert_eq!(new.window_id(), WindowId(7));
        assert_eq!(existing.window_id(), WindowId(7));
    }

    #[test]
    fn create_and_destroy_classification() {
        assert!(WindowCommand::Create(WindowAttributes::default()).is_create());
        assert!(WindowCommand::Destroy.is_destroy());
        assert!(!WindowCommand::RequestRedraw.is_create());
        assert!(!WindowCommand::RequestRedraw.is_destroy());
    }

    #[test]
    fn mode_request_constructors() {
        assert_eq!(WindowModeRequest::WINDOWED.mode, WindowMode::Windowed);
        let fs = WindowModeRequest::simple(WindowMode::Fullscreen);
        assert_eq!(fs.mode, WindowMode::Fullscreen);
        assert!(fs.monitor.is_none());
    }
}
