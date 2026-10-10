//! # `prism_input`
//!
//! Prism's input kernel: a deterministic, backend-agnostic model of input
//! state. Windowing/OS backends live in separate crates and translate platform
//! events into the unified [`InputEvent`] stream; this crate turns that stream
//! into queryable state and does nothing platform-specific.
//!
//! It is a pure, deterministic, `no_std + alloc` type system with no `unsafe`.
//! Ordered collections ([`BTreeSet`](alloc::collections::BTreeSet) /
//! [`BTreeMap`](alloc::collections::BTreeMap)) give stable iteration across runs
//! and platforms, which matters for replay and lockstep networking.
//!
//! ## Layout
//! - Generic cores: [`button`] ([`ButtonInput<T>`]), [`axis`] ([`Axis<T>`]).
//! - Devices: [`keyboard`], [`mouse`], [`gamepad`], [`touch`].
//! - Unified stream: [`event`] ([`InputEvent`], [`ButtonState`]).
//! - ABI envelope: [`envelope`] ([`InputEventEnvelope`], [`MonotonicTimestamp`],
//!   [`PlatformInputStamp`]), [`capabilities`] ([`InputBackendCapabilities`]),
//!   and [`registry`] (deterministic drain-to-batch sorting).
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

pub mod axis;
pub mod button;
pub mod capabilities;
pub mod envelope;
pub mod event;
pub mod gamepad;
pub mod keyboard;
pub mod mouse;
pub mod registry;
pub mod touch;

pub use axis::Axis;
pub use button::ButtonInput;
pub use capabilities::InputBackendCapabilities;
pub use envelope::{
    InputDeviceId, InputEventEnvelope, InputEventSequence, InputSource, MonotonicTimestamp,
    PlatformInputStamp,
};
pub use event::{ButtonState, InputEvent};
pub use gamepad::{
    radial_deadzone, AxisSettings, ButtonSettings, GamepadAxis, GamepadButton, GamepadConnection,
    GamepadId, GamepadSettings,
};
pub use keyboard::{KeyCode, KeyboardInput, ModifiersState};
pub use mouse::{MouseButton, MouseButtonInput, MouseMotion, MouseScrollUnit, MouseWheel};
pub use registry::{sort_envelopes, InputEnvelopeQueue};
pub use touch::{Touch, TouchInput, TouchPhase, Touches};

/// The common types most consumers import.
pub mod prelude {
    pub use crate::axis::Axis;
    pub use crate::button::ButtonInput;
    pub use crate::capabilities::InputBackendCapabilities;
    pub use crate::envelope::{
        InputDeviceId, InputEventEnvelope, InputEventSequence, InputSource, MonotonicTimestamp,
        PlatformInputStamp,
    };
    pub use crate::event::{ButtonState, InputEvent};
    pub use crate::gamepad::{
        GamepadAxis, GamepadButton, GamepadConnection, GamepadId, GamepadSettings,
    };
    pub use crate::keyboard::{KeyCode, KeyboardInput, ModifiersState};
    pub use crate::mouse::{
        MouseButton, MouseButtonInput, MouseMotion, MouseScrollUnit, MouseWheel,
    };
    pub use crate::registry::{sort_envelopes, InputEnvelopeQueue};
    pub use crate::touch::{Touch, TouchInput, TouchPhase, Touches};
}

#[cfg(test)]
mod tests;
