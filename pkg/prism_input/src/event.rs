//! The press/release state carried by digital-input events and the unified
//! [`InputEvent`] stream.
//!
//! Backends translate OS/windowing events into these variants and feed them to
//! the input state. Keeping a single ordered event type lets higher layers
//! record, replay, or route raw input without caring about its origin.

use crate::gamepad::{GamepadAxis, GamepadButton, GamepadConnection, GamepadId};
use crate::keyboard::KeyboardInput;
use crate::mouse::{MouseButtonInput, MouseMotion, MouseWheel};
use crate::touch::TouchInput;

/// Whether a digital input was pressed or released by an event.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ButtonState {
    /// The button went down.
    Pressed,
    /// The button went up.
    Released,
}

impl ButtonState {
    /// Whether this state represents a press.
    #[must_use]
    pub const fn is_pressed(self) -> bool {
        matches!(self, Self::Pressed)
    }
}

/// A single raw input event from any device.
///
/// Backends emit these in arrival order; the input state applies them to update
/// its [`ButtonInput`](crate::ButtonInput) / [`Axis`](crate::Axis) stores.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum InputEvent {
    /// A keyboard key changed state.
    Keyboard(KeyboardInput),
    /// A mouse button changed state.
    MouseButton(MouseButtonInput),
    /// The mouse moved by a relative delta.
    MouseMotion(MouseMotion),
    /// The mouse wheel scrolled.
    MouseWheel(MouseWheel),
    /// A touch point changed.
    Touch(TouchInput),
    /// A gamepad button changed state.
    GamepadButton {
        /// Which gamepad.
        gamepad: GamepadId,
        /// Which button.
        button: GamepadButton,
        /// The new state.
        state: ButtonState,
    },
    /// A gamepad analog axis moved.
    GamepadAxis {
        /// Which gamepad.
        gamepad: GamepadId,
        /// Which axis.
        axis: GamepadAxis,
        /// The new raw value.
        value: f32,
    },
    /// A gamepad connected or disconnected.
    GamepadConnection {
        /// Which gamepad.
        gamepad: GamepadId,
        /// The connection transition.
        connection: GamepadConnection,
    },
}
