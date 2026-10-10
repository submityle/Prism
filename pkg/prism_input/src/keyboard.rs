//! Physical keyboard keys, modifier state, and keyboard events.
//!
//! [`KeyCode`] identifies a key by its physical position (layout-independent),
//! mirroring the W3C `UI Events` `code` values so a backend can map platform
//! scancodes without guessing the user's layout.

use crate::event::ButtonState;

/// A physical keyboard key, identified by position rather than the character it
/// produces under the active layout.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[non_exhaustive]
#[expect(
    missing_docs,
    reason = "each variant names the physical key it represents; per-variant \
              docs would be pure restatement of the identifier"
)]
pub enum KeyCode {
    // Letters (physical positions on a US QWERTY keyboard).
    KeyA,
    KeyB,
    KeyC,
    KeyD,
    KeyE,
    KeyF,
    KeyG,
    KeyH,
    KeyI,
    KeyJ,
    KeyK,
    KeyL,
    KeyM,
    KeyN,
    KeyO,
    KeyP,
    KeyQ,
    KeyR,
    KeyS,
    KeyT,
    KeyU,
    KeyV,
    KeyW,
    KeyX,
    KeyY,
    KeyZ,
    // Number row.
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
    // Function row.
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    // Whitespace and editing.
    Escape,
    Tab,
    Space,
    Backspace,
    Enter,
    Delete,
    Insert,
    Home,
    End,
    PageUp,
    PageDown,
    // Arrows.
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    // Modifiers.
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    SuperLeft,
    SuperRight,
    CapsLock,
    // Punctuation (US layout positions).
    Minus,
    Equal,
    BracketLeft,
    BracketRight,
    Backslash,
    Semicolon,
    Quote,
    Backquote,
    Comma,
    Period,
    Slash,
    // Numpad.
    Numpad0,
    Numpad1,
    Numpad2,
    Numpad3,
    Numpad4,
    Numpad5,
    Numpad6,
    Numpad7,
    Numpad8,
    Numpad9,
    NumpadAdd,
    NumpadSubtract,
    NumpadMultiply,
    NumpadDivide,
    NumpadEnter,
    NumpadDecimal,
}

impl KeyCode {
    /// Whether this key is one of the modifier keys (shift/control/alt/super).
    #[must_use]
    pub const fn is_modifier(self) -> bool {
        matches!(
            self,
            Self::ShiftLeft
                | Self::ShiftRight
                | Self::ControlLeft
                | Self::ControlRight
                | Self::AltLeft
                | Self::AltRight
                | Self::SuperLeft
                | Self::SuperRight
        )
    }
}

/// The current state of the modifier keys, derived from held keys.
///
/// Each flag is set when either the left or right variant of the modifier is
/// held, which is what gameplay and UI shortcuts usually care about.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ModifiersState {
    /// Either shift key is held.
    pub shift: bool,
    /// Either control key is held.
    pub control: bool,
    /// Either alt/option key is held.
    pub alt: bool,
    /// Either super (Windows/Command) key is held.
    pub super_key: bool,
}

impl ModifiersState {
    /// Whether no modifier is held.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        !self.shift && !self.control && !self.alt && !self.super_key
    }
}

/// A keyboard key-state change event.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct KeyboardInput {
    /// The physical key.
    pub key_code: KeyCode,
    /// Whether the key went down or up.
    pub state: ButtonState,
    /// Whether this is an auto-repeat while the key was already held.
    pub repeat: bool,
}
