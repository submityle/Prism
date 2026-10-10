//! Linear-light RGBA ([`LinearRgba`]) — the rendering/blending hub color.

use crate::color::oklab::Oklaba;
use crate::color::srgb::Srgba;
use crate::color::transfer::{linear_to_srgb, srgb_to_linear};
use crate::color::xyz::Xyza;
use crate::Vec4;

/// A color in **linear-light** sRGB primaries with a straight (non-premultiplied)
/// alpha. This is the correct space for adding, averaging, and lighting colors;
/// store/display colors as [`Srgba`] instead.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
#[repr(C)]
pub struct LinearRgba {
    /// Red (linear light).
    pub red: f32,
    /// Green (linear light).
    pub green: f32,
    /// Blue (linear light).
    pub blue: f32,
    /// Alpha (linear, `0` transparent .. `1` opaque).
    pub alpha: f32,
}

impl LinearRgba {
    /// Opaque black.
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    /// Opaque white.
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);
    /// Fully transparent (all zero).
    pub const TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);

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

    /// Reinterpret as a [`Vec4`] (`x=r, y=g, z=b, w=a`).
    #[inline]
    pub const fn to_vec4(self) -> Vec4 {
        Vec4::new(self.red, self.green, self.blue, self.alpha)
    }

    /// Clamp every component (including alpha) to `[0, 1]`.
    #[inline]
    pub fn clamped(self) -> Self {
        Self::new(
            self.red.clamp(0.0, 1.0),
            self.green.clamp(0.0, 1.0),
            self.blue.clamp(0.0, 1.0),
            self.alpha.clamp(0.0, 1.0),
        )
    }

    /// Encode to non-linear [`Srgba`] for storage/display.
    #[inline]
    pub fn to_srgb(self) -> Srgba {
        Srgba::new(
            linear_to_srgb(self.red),
            linear_to_srgb(self.green),
            linear_to_srgb(self.blue),
            self.alpha,
        )
    }

    /// Decode a non-linear [`Srgba`] into linear light.
    #[inline]
    pub fn from_srgb(c: Srgba) -> Self {
        Self::new(
            srgb_to_linear(c.red),
            srgb_to_linear(c.green),
            srgb_to_linear(c.blue),
            c.alpha,
        )
    }

    /// Convert to CIE 1931 [`Xyza`] (D65 white point).
    #[inline]
    pub fn to_xyz(self) -> Xyza {
        let (r, g, b) = (self.red, self.green, self.blue);
        Xyza::new(
            0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b,
            0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b,
            0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b,
            self.alpha,
        )
    }

    /// Convert from CIE 1931 [`Xyza`] (D65 white point).
    #[inline]
    pub fn from_xyz(c: Xyza) -> Self {
        let (x, y, z) = (c.x, c.y, c.z);
        Self::new(
            3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z,
            -0.969_266 * x + 1.876_010_8 * y + 0.041_556_0 * z,
            0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z,
            c.alpha,
        )
    }

    /// Convert to perceptually-uniform [`Oklaba`].
    #[inline]
    pub fn to_oklab(self) -> Oklaba {
        Oklaba::from_linear(self)
    }

    /// Convert from perceptually-uniform [`Oklaba`].
    #[inline]
    pub fn from_oklab(c: Oklaba) -> Self {
        c.to_linear()
    }

    /// Component-wise linear interpolation in linear-light space.
    ///
    /// This is the physically-correct space for blending; `t` is not clamped.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Self::new(
            self.red + (rhs.red - self.red) * t,
            self.green + (rhs.green - self.green) * t,
            self.blue + (rhs.blue - self.blue) * t,
            self.alpha + (rhs.alpha - self.alpha) * t,
        )
    }

    /// `true` if every component is finite (no `NaN`/`inf`).
    #[inline]
    pub fn is_finite(self) -> bool {
        self.red.is_finite()
            && self.green.is_finite()
            && self.blue.is_finite()
            && self.alpha.is_finite()
    }
}

impl From<Srgba> for LinearRgba {
    #[inline]
    fn from(c: Srgba) -> Self {
        Self::from_srgb(c)
    }
}

impl From<Xyza> for LinearRgba {
    #[inline]
    fn from(c: Xyza) -> Self {
        Self::from_xyz(c)
    }
}

impl From<Oklaba> for LinearRgba {
    #[inline]
    fn from(c: Oklaba) -> Self {
        c.to_linear()
    }
}
