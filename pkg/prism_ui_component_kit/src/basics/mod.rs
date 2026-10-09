//! Basic, structural controls: the primitives every other family builds on.
//!
//! Everything here emits a data-only [`Element`](prism_ui::Element) and attaches
//! only kit class names (styled by [`crate::preset`]). Controls are plain
//! [`Component`](prism_ui_component::Component)s, so they compose and test
//! without a running runtime.

pub mod button;

pub use button::{Button, ButtonProps};
