//! Linear RGBA color used for clear values and blend constants.

/// A linear, non-premultiplied RGBA color with `f64` channels.
///
/// Clear values and blend constants are specified in the render target's own
/// space; this type does not perform sRGB encoding. Channels are not clamped
/// so that extended-range (HDR) clear values round-trip exactly.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Color {
    /// The red channel.
    pub r: f64,
    /// The green channel.
    pub g: f64,
    /// The blue channel.
    pub b: f64,
    /// The alpha channel.
    pub a: f64,
}

impl Color {
    /// Opaque black (`0, 0, 0, 1`).
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    /// Opaque white (`1, 1, 1, 1`).
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);
    /// Fully transparent (`0, 0, 0, 0`).
    pub const TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);

    /// Creates a color from explicit channels.
    #[must_use]
    pub const fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }

    /// Creates an opaque color (`a = 1`).
    #[must_use]
    pub const fn rgb(r: f64, g: f64, b: f64) -> Self {
        Self::new(r, g, b, 1.0)
    }

    /// The channels as a `[r, g, b, a]` array.
    #[must_use]
    pub const fn to_array(self) -> [f64; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::TRANSPARENT
    }
}

impl From<[f64; 4]> for Color {
    fn from([r, g, b, a]: [f64; 4]) -> Self {
        Self::new(r, g, b, a)
    }
}
