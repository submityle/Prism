//! Input event types shared across hit testing, dispatch, and gestures.
//!
//! These types are intentionally small and `Copy` where practical so that
//! events can be forwarded cheaply through the capture/target/bubble pipeline
//! and fed to multiple gesture recognizers without allocation.

use crate::geometry::Point;

/// Stable identifier for a node that participates in input handling.
///
/// The input system is engine-agnostic: callers assign a `NodeId` to each
/// interactive node when they build the hit-test tree, and the same id is
/// used to register dispatch handlers and gesture recognizers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u64);

impl NodeId {
    /// Creates a `NodeId` from a raw integer.
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the underlying raw integer.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Identifier distinguishing concurrent pointers for multi-touch input.
///
/// A mouse typically uses a single stable `PointerId`, while touch input
/// assigns a distinct id per finger so recognizers such as pinch can track
/// each contact independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PointerId(u64);

impl PointerId {
    /// Creates a `PointerId` from a raw integer.
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the underlying raw integer.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Mouse-style button associated with a pointer event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerButton {
    /// The primary button (usually left mouse or a touch contact).
    Primary,
    /// The secondary button (usually right mouse).
    Secondary,
    /// The middle button (usually the scroll wheel press).
    Middle,
}

/// Keyboard modifier state captured at the time of an event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modifiers {
    /// Whether a shift key is held.
    pub shift: bool,
    /// Whether a control key is held.
    pub ctrl: bool,
    /// Whether an alt / option key is held.
    pub alt: bool,
    /// Whether a meta / command / super key is held.
    pub meta: bool,
}

impl Modifiers {
    /// Returns modifiers with no keys held.
    pub const fn none() -> Self {
        Self {
            shift: false,
            ctrl: false,
            alt: false,
            meta: false,
        }
    }

    /// Returns a copy with the shift flag set to `value`.
    pub const fn with_shift(mut self, value: bool) -> Self {
        self.shift = value;
        self
    }

    /// Returns a copy with the control flag set to `value`.
    pub const fn with_ctrl(mut self, value: bool) -> Self {
        self.ctrl = value;
        self
    }

    /// Returns a copy with the alt flag set to `value`.
    pub const fn with_alt(mut self, value: bool) -> Self {
        self.alt = value;
        self
    }

    /// Returns a copy with the meta flag set to `value`.
    pub const fn with_meta(mut self, value: bool) -> Self {
        self.meta = value;
        self
    }

    /// Returns `true` when any modifier key is held.
    pub const fn any(self) -> bool {
        self.shift || self.ctrl || self.alt || self.meta
    }
}

/// Kind of pointer event, mirroring the standard pointer lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerKind {
    /// A pointer became active (mouse button down or touch start).
    Down,
    /// A pointer stopped being active (mouse button up or touch end).
    Up,
    /// A pointer moved while active.
    Move,
    /// A pointer entered a node's area.
    Enter,
    /// A pointer left a node's area.
    Leave,
    /// Input for a pointer was cancelled (for example by the system).
    Cancel,
}

/// A single pointer event with its position, timing, and modifier state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointerEvent {
    /// The pointer that produced the event.
    pub pointer: PointerId,
    /// The kind of event.
    pub kind: PointerKind,
    /// Position in layout units, in the same space as hit-test rectangles.
    pub position: Point<f32>,
    /// Button associated with the event.
    pub button: PointerButton,
    /// Monotonic timestamp in milliseconds, used by timed gestures.
    pub timestamp_ms: u64,
    /// Keyboard modifiers held when the event occurred.
    pub modifiers: Modifiers,
}

impl PointerEvent {
    /// Creates a pointer event with the primary button and no modifiers.
    pub const fn new(
        pointer: PointerId,
        kind: PointerKind,
        position: Point<f32>,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            pointer,
            kind,
            position,
            button: PointerButton::Primary,
            timestamp_ms,
            modifiers: Modifiers::none(),
        }
    }

    /// Returns a copy with the button set to `button`.
    pub const fn with_button(mut self, button: PointerButton) -> Self {
        self.button = button;
        self
    }

    /// Returns a copy with the modifiers set to `modifiers`.
    pub const fn with_modifiers(mut self, modifiers: Modifiers) -> Self {
        self.modifiers = modifiers;
        self
    }
}

/// A keyboard key, enough to drive focus traversal and activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyCode {
    /// The tab key, used to move focus forward (or backward with shift).
    Tab,
    /// The enter / return key.
    Enter,
    /// The escape key.
    Escape,
    /// The space bar.
    Space,
    /// The up arrow key.
    ArrowUp,
    /// The down arrow key.
    ArrowDown,
    /// The left arrow key.
    ArrowLeft,
    /// The right arrow key.
    ArrowRight,
    /// A printable character key.
    Char(char),
    /// Any other key, identified by a platform-specific scan code.
    Other(u32),
}

/// A keyboard event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyEvent {
    /// The key involved.
    pub code: KeyCode,
    /// Whether the key was pressed (`true`) or released (`false`).
    pub pressed: bool,
    /// Keyboard modifiers held when the event occurred.
    pub modifiers: Modifiers,
    /// Monotonic timestamp in milliseconds.
    pub timestamp_ms: u64,
}

impl KeyEvent {
    /// Creates a key-press event with no modifiers.
    pub const fn press(code: KeyCode, timestamp_ms: u64) -> Self {
        Self {
            code,
            pressed: true,
            modifiers: Modifiers::none(),
            timestamp_ms,
        }
    }

    /// Creates a key-release event with no modifiers.
    pub const fn release(code: KeyCode, timestamp_ms: u64) -> Self {
        Self {
            code,
            pressed: false,
            modifiers: Modifiers::none(),
            timestamp_ms,
        }
    }

    /// Returns a copy with the modifiers set to `modifiers`.
    pub const fn with_modifiers(mut self, modifiers: Modifiers) -> Self {
        self.modifiers = modifiers;
        self
    }
}

/// Propagation phase during which a handler runs.
///
/// Dispatch walks the hit path in three phases: `Capture` from the root down
/// toward the target, `Target` at the hit node itself, then `Bubble` back up
/// toward the root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    /// Travelling from the root toward the target.
    Capture,
    /// At the target node.
    Target,
    /// Travelling from the target back toward the root.
    Bubble,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_and_pointer_ids_round_trip() {
        assert_eq!(NodeId::new(7).get(), 7);
        assert_eq!(PointerId::new(3).get(), 3);
    }

    #[test]
    fn modifiers_builder_and_any() {
        let m = Modifiers::none().with_shift(true).with_meta(true);
        assert!(m.shift && m.meta);
        assert!(!m.ctrl && !m.alt);
        assert!(m.any());
        assert!(!Modifiers::none().any());
    }

    #[test]
    fn pointer_event_builders() {
        let e = PointerEvent::new(
            PointerId::new(1),
            PointerKind::Down,
            Point::new(2.0, 3.0),
            100,
        )
        .with_button(PointerButton::Secondary)
        .with_modifiers(Modifiers::none().with_ctrl(true));
        assert_eq!(e.button, PointerButton::Secondary);
        assert!(e.modifiers.ctrl);
        assert_eq!(e.timestamp_ms, 100);
    }

    #[test]
    fn key_event_helpers() {
        let p = KeyEvent::press(KeyCode::Tab, 5).with_modifiers(Modifiers::none().with_shift(true));
        assert!(p.pressed);
        assert!(p.modifiers.shift);
        let r = KeyEvent::release(KeyCode::Enter, 6);
        assert!(!r.pressed);
    }
}
