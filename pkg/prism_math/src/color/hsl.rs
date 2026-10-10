//! HSL / HSV cylindrical color ([`Hsla`], [`Hsva`]).
//!
//! Both models are defined over **non-linear sRGB** components (the usual
//! convention for color pickers), so conversions here round-trip through
//! [`Srgba`], not [`crate::color::LinearRgba`]. Hue is in degrees `[0, 360)`;
//! the remaining axes are in `[0, 1]`.

use crate::color::srgb::Srgba;
use crate::float::f32 as mf;

/// A color in **HSL** (hue / saturation / lightness) with straight alpha.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
#[repr(C)]
pub struct Hsla {
    /// Hue in degrees `[0, 360)`.
    pub hue: f32,
    /// Saturation `[0, 1]`.
    pub saturation: f32,
    /// Lightness `[0, 1]`.
    pub lightness: f32,
    /// Alpha (linear).
    pub alpha: f32,
}

/// A color in **HSV** (hue / saturation / value) with straight alpha.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
#[repr(C)]
pub struct Hsva {
    /// Hue in degrees `[0, 360)`.
    pub hue: f32,
    /// Saturation `[0, 1]`.
    pub saturation: f32,
    /// Value/brightness `[0, 1]`.
    pub value: f32,
    /// Alpha (linear).
    pub alpha: f32,
}

/// Decompose sRGB into (max, min, chroma, hue-in-degrees).
#[inline]
fn rgb_to_hue(c: Srgba) -> (f32, f32, f32, f32) {
    let (r, g, b) = (c.red, c.green, c.blue);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let chroma = max - min;
    let hue = if chroma == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / chroma).rem_euclid(6.0))
    } else if max == g {
        60.0 * ((b - r) / chroma + 2.0)
    } else {
        60.0 * ((r - g) / chroma + 4.0)
    };
    (max, min, chroma, hue)
}

/// Reconstruct sRGB from hue/chroma plus a per-channel offset `m`.
#[inline]
fn hue_to_rgb(hue: f32, chroma: f32, m: f32, alpha: f32) -> Srgba {
    let h = hue.rem_euclid(360.0) / 60.0;
    let x = chroma * (1.0 - mf::abs(h.rem_euclid(2.0) - 1.0));
    let (r1, g1, b1) = match h as u32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    Srgba::new(r1 + m, g1 + m, b1 + m, alpha)
}

impl Hsla {
    /// Construct from components.
    #[inline]
    pub const fn new(hue: f32, saturation: f32, lightness: f32, alpha: f32) -> Self {
        Self {
            hue,
            saturation,
            lightness,
            alpha,
        }
    }

    /// Convert from non-linear [`Srgba`].
    #[inline]
    pub fn from_srgb(c: Srgba) -> Self {
        let (max, min, chroma, hue) = rgb_to_hue(c);
        let lightness = 0.5 * (max + min);
        let saturation = if lightness <= 0.0 || lightness >= 1.0 {
            0.0
        } else {
            chroma / (1.0 - mf::abs(2.0 * lightness - 1.0))
        };
        Self::new(hue, saturation, lightness, c.alpha)
    }

    /// Convert to non-linear [`Srgba`].
    #[inline]
    pub fn to_srgb(self) -> Srgba {
        let chroma = (1.0 - mf::abs(2.0 * self.lightness - 1.0)) * self.saturation;
        let m = self.lightness - 0.5 * chroma;
        hue_to_rgb(self.hue, chroma, m, self.alpha)
    }
}

impl Hsva {
    /// Construct from components.
    #[inline]
    pub const fn new(hue: f32, saturation: f32, value: f32, alpha: f32) -> Self {
        Self {
            hue,
            saturation,
            value,
            alpha,
        }
    }

    /// Convert from non-linear [`Srgba`].
    #[inline]
    pub fn from_srgb(c: Srgba) -> Self {
        let (max, _min, chroma, hue) = rgb_to_hue(c);
        let value = max;
        let saturation = if value <= 0.0 { 0.0 } else { chroma / value };
        Self::new(hue, saturation, value, c.alpha)
    }

    /// Convert to non-linear [`Srgba`].
    #[inline]
    pub fn to_srgb(self) -> Srgba {
        let chroma = self.value * self.saturation;
        let m = self.value - chroma;
        hue_to_rgb(self.hue, chroma, m, self.alpha)
    }
}

impl From<Srgba> for Hsla {
    #[inline]
    fn from(c: Srgba) -> Self {
        Self::from_srgb(c)
    }
}

impl From<Hsla> for Srgba {
    #[inline]
    fn from(c: Hsla) -> Self {
        c.to_srgb()
    }
}

impl From<Srgba> for Hsva {
    #[inline]
    fn from(c: Srgba) -> Self {
        Self::from_srgb(c)
    }
}

impl From<Hsva> for Srgba {
    #[inline]
    fn from(c: Hsva) -> Self {
        c.to_srgb()
    }
}
