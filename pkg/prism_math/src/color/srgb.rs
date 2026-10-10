//! Non-linear sRGB RGBA ([`Srgba`]) — the storage/display color.

use crate::color::linear::LinearRgba;

/// A color in **non-linear sRGB** (gamma-encoded) with straight alpha. This is
/// how colors are stored in `*_SRGB` textures, image files, and CSS. Decode to
/// [`LinearRgba`] before doing any arithmetic on it.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
#[repr(C)]
pub struct Srgba {
    /// Red (gamma-encoded, `0..=1`).
    pub red: f32,
    /// Green (gamma-encoded, `0..=1`).
    pub green: f32,
    /// Blue (gamma-encoded, `0..=1`).
    pub blue: f32,
    /// Alpha (linear, `0..=1`).
    pub alpha: f32,
}

impl Srgba {
    /// Opaque black.
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    /// Opaque white.
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);

    /// Construct from components.
    #[inline]
    pub const fn new(red: f32, green: f32, blue: f32, alpha: f32) -> Self {
        Self {
            red,
            green,
            blue,
            alpha,
        }
    }

    /// Construct an opaque color (`alpha = 1`).
    #[inline]
    pub const fn rgb(red: f32, green: f32, blue: f32) -> Self {
        Self::new(red, green, blue, 1.0)
    }

    /// Build from 8-bit sRGB components (`0..=255`), opaque.
    #[inline]
    pub fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self::from_u8a(r, g, b, 255)
    }

    /// Build from 8-bit sRGB + alpha components (`0..=255`).
    #[inline]
    pub fn from_u8a(r: u8, g: u8, b: u8, a: u8) -> Self {
        const INV: f32 = 1.0 / 255.0;
        Self::new(
            r as f32 * INV,
            g as f32 * INV,
            b as f32 * INV,
            a as f32 * INV,
        )
    }

    /// Quantize to 8-bit `[r, g, b, a]`, rounding and clamping to `0..=255`.
    #[inline]
    pub fn to_u8_array(self) -> [u8; 4] {
        #[inline]
        fn q(c: f32) -> u8 {
            let v = c.clamp(0.0, 1.0) * 255.0 + 0.5;
            // `v` is already in `[0.5, 255.5]`; the floor via `as u8` saturates.
            v as u8
        }
        [q(self.red), q(self.green), q(self.blue), q(self.alpha)]
    }

    /// Components as `[r, g, b, a]`.
    #[inline]
    pub const fn to_array(self) -> [f32; 4] {
        [self.red, self.green, self.blue, self.alpha]
    }

    /// Build from `[r, g, b, a]`.
    #[inline]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }

    /// Decode to [`LinearRgba`].
    #[inline]
    pub fn to_linear(self) -> LinearRgba {
        LinearRgba::from_srgb(self)
    }

    /// Encode a [`LinearRgba`] into sRGB.
    #[inline]
    pub fn from_linear(c: LinearRgba) -> Self {
        c.to_srgb()
    }
}

impl From<LinearRgba> for Srgba {
    #[inline]
    fn from(c: LinearRgba) -> Self {
        c.to_srgb()
    }
}
