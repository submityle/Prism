//! Basic, structural controls: the primitives every other family builds on.
//!
//! Everything here emits a data-only [`Element`](prism_ui::Element) and attaches
//! only kit class names (styled by this module's [`register_styles`]). Controls
//! are plain [`Component`](prism_ui_component::Component)s, so they compose and
//! test without a running runtime.

use crate::preset::StyleSheet;

pub mod button;

pub use button::{Button, ButtonProps};

/// Registers every `basics/` control's token-backed classes into `sheet`.
pub fn register_styles(sheet: &mut StyleSheet) {
    button::register_styles(sheet);
}
