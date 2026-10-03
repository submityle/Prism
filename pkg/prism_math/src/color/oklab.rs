//! `OkLab` / `OkLCh` perceptually-uniform color ([`Oklaba`], [`Oklcha`]).
//!
//! `OkLab` (Björn Ottosson, 2020) is a perceptual color space in which Euclidean
//! distance tracks perceived difference far better than sRGB or CIELAB, making
//! it the preferred space for gradients and color mixing. `OkLCh` is its
//! cylindrical (lightness / chroma / hue) form.

use crate::color::linear::LinearRgba;
use crate::float::f32 as mf;

/// A color in **`OkLab`** with a straight alpha.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
#[repr(C)]
pub struct Oklaba {
    /// Perceptual lightness `L` (`0` black .. `1` white).
    pub l: f32,
    /// Green–red axis `a`.
    pub a: f32,
    /// Blue–yellow axis `b`.
    pub b: f32,
    /// Alpha (linear).
    pub alpha: f32,
}

/// A color in **`OkLCh`** (`OkLab` in cylindrical lightness/chroma/hue form).
#[derive(Clone, Copy, PartialEq, Default, Debug)]
#[repr(C)]
pub struct Oklcha {
    /// Perceptual lightness `L`.
    pub l: f32,
    /// Chroma (radial distance from the neutral axis, `>= 0`).
    pub chroma: f32,
    /// Hue angle in radians.
    pub hue: f32,
    /// Alpha (linear).
    pub alpha: f32,
}

impl Oklaba {
    /// Construct from components.
    #[inline]
    pub const fn new(l: f32, a: f32, b: f32, alpha: f32) -> Self {
        Self { l, a, b, alpha }
    }

    /// Components as `[l, a, b, alpha]`.
    #[inline]
    pub const fn to_array(self) -> [f32; 4] {
        [self.l, self.a, self.b, self.alpha]
    }

    /// Build from `[l, a, b, alpha]`.
    #[inline]
    pub const fn from_array(v: [f32; 4]) -> Self {
        Self::new(v[0], v[1], v[2], v[3])
    }

    /// Convert from linear-light sRGB.
    #[inline]
    pub fn from_linear(c: LinearRgba) -> Self {
        let (r, g, b) = (c.red, c.green, c.blue);
        let l = 0.412_221_47 * r + 0.536_332_55 * g + 0.051_445_995 * b;
        let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
        let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;
        let l_ = mf::cbrt(l);
        let m_ = mf::cbrt(m);
        let s_ = mf::cbrt(s);
        Self::new(
            0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
            1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
            0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
            c.alpha,
        )
    }

    /// Convert to linear-light sRGB.
    #[inline]
    pub fn to_linear(self) -> LinearRgba {
        let l_ = self.l + 0.396_337_78 * self.a + 0.215_803_76 * self.b;
        let m_ = self.l - 0.105_561_346 * self.a - 0.063_854_17 * self.b;
        let s_ = self.l - 0.089_484_18 * self.a - 1.291_485_5 * self.b;
        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;
        LinearRgba::new(
            4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
            -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
            -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
            self.alpha,
        )
    }

    /// Convert to the cylindrical [`Oklcha`] form.
    #[inline]
    pub fn to_oklch(self) -> Oklcha {
        let chroma = mf::sqrt(self.a * self.a + self.b * self.b);
        let hue = mf::atan2(self.b, self.a);
        Oklcha::new(self.l, chroma, hue, self.alpha)
    }

    /// Perceptually-uniform interpolation in `OkLab` space (`t` not clamped).
    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Self::new(
            self.l + (rhs.l - self.l) * t,
            self.a + (rhs.a - self.a) * t,
            self.b + (rhs.b - self.b) * t,
            self.alpha + (rhs.alpha - self.alpha) * t,
        )
    }
}

impl Oklcha {
    /// Construct from components.
    #[inline]
    pub const fn new(l: f32, chroma: f32, hue: f32, alpha: f32) -> Self {
        Self { l, chroma, hue, alpha }
    }

    /// Convert to the Cartesian [`Oklaba`] form.
    #[inline]
    pub fn to_oklab(self) -> Oklaba {
        let (sin, cos) = mf::sin_cos(self.hue);
        Oklaba::new(self.l, self.chroma * cos, self.chroma * sin, self.alpha)
    }
}

impl From<Oklaba> for Oklcha {
    #[inline]
    fn from(c: Oklaba) -> Self {
        c.to_oklch()
    }
}

impl From<Oklcha> for Oklaba {
    #[inline]
    fn from(c: Oklcha) -> Self {
        c.to_oklab()
    }
}
