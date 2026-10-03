//! Mouse buttons, motion, and wheel input.

use crate::event::ButtonState;

/// A mouse button.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum MouseButton {
    /// The primary (usually left) button.
    Left,
    /// The secondary (usually right) button.
    Right,
    /// The middle button / wheel click.
    Middle,
    /// The first extra button (often "back").
    Back,
    /// The second extra button (often "forward").
    Forward,
    /// Any other button, identified by its platform code.
    Other(u16),
}

/// A mouse button state-change event.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MouseButtonInput {
    /// Which button changed.
    pub button: MouseButton,
    /// Whether it went down or up.
    pub state: ButtonState,
}

/// Relative mouse motion since the last event, in logical pixels.
///
/// This is the raw delta, independent of the cursor position, suitable for
/// camera look where the cursor may be grabbed.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct MouseMotion {
    /// Horizontal delta (right is positive).
    pub delta_x: f32,
    /// Vertical delta (down is positive).
    pub delta_y: f32,
}

/// The unit a [`MouseWheel`] delta is expressed in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MouseScrollUnit {
    /// Delta is in lines (classic notched wheels).
    Line,
    /// Delta is in logical pixels (precise/trackpad scrolling).
    Pixel,
}

/// A mouse wheel scroll event.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MouseWheel {
    /// Whether the deltas are lines or pixels.
    pub unit: MouseScrollUnit,
    /// Horizontal scroll amount.
    pub x: f32,
    /// Vertical scroll amount.
    pub y: f32,
}
