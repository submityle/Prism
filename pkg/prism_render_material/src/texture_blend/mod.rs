//! Separable layer **blend modes** for compositing decoded textures.
//!
//! See [`modes`] for the W3C/PDF separable blend-mode family (multiply, screen,
//! overlay, dodge/burn, hard/soft light, difference, ...) plus an `RGBA8`
//! layer compositor.

mod modes;

pub use modes::{blend_channel, blend_rgba8, BlendMode};
