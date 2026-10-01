//! Non-layout visual properties resolved for a node.
//!
//! Layout decides *where* a box is; [`PaintStyle`] decides how it *looks*. The
//! runtime diffs this struct between frames so the backend is only told about
//! visual properties that actually changed.

use prism_ui_style::Color;

/// The resolved visual appearance of a node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaintStyle {
    /// Background fill color.
    pub background_color: Option<Color>,
    /// Foreground / text color.
    pub color: Option<Color>,
    /// Border color.
    pub border_color: Option<Color>,
    /// Border corner radius, in logical pixels.
    pub border_radius: f32,
    /// Border thickness, in logical pixels.
    pub border_width: f32,
    /// Opacity in `0.0..=1.0`.
    pub opacity: f32,
    /// Font size, in logical pixels (text nodes).
    pub font_size: f32,
    /// Font weight (e.g. `400.0` normal, `700.0` bold).
    pub font_weight: f32,
}

impl Default for PaintStyle {
    fn default() -> Self {
        Self {
            background_color: None,
            color: None,
            border_color: None,
            border_radius: 0.0,
            border_width: 0.0,
            opacity: 1.0,
            font_size: 16.0,
            font_weight: 400.0,
        }
    }
}
