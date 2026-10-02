//! Separable layer **blend modes** for compositing decoded textures.
//!
//! See [`modes`] for the W3C/PDF separable blend-mode family (multiply, screen,
//! overlay, dodge/burn, hard/soft light, difference, ...) plus an `RGBA8`
//! layer compositor, and [`nonseparable`] for the whole-colour
//! Hue/Saturation/Color/Luminosity modes.

mod modes;
mod nonseparable;

pub use modes::{blend_channel, blend_rgba8, BlendMode};
pub use nonseparable::{blend_nonseparable, blend_nonseparable_rgba8, NonSeparableBlendMode};
