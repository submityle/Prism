//! IEEE 754 `binary16` (half precision) scalar and small storage vectors.
//!
//! [`F16`] is a 16-bit floating-point value kept as raw bits. Conversions to
//! and from `f32` are exact in the `f16` -> `f32` direction and
//! round-to-nearest-even in the `f32` -> `f16` direction, with correct
//! handling of subnormals, infinities, and `NaN` (per IEEE 754). The companion
//! [`F16Vec2`], [`F16Vec3`], and [`F16Vec4`] types are compact storage vectors
//! intended for vertex streams, lightmaps, and other bandwidth-sensitive data
//! that is decoded to `f32` for computation.
//!
//! `f16` has no native arithmetic here: operators convert to `f32`, compute,
//! and round the result back. This keeps a single, auditable rounding path.

use crate::vec::{Vec2, Vec3, Vec4};
use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A half-precision IEEE 754 `binary16` scalar stored as raw bits.
///
/// The layout is a transparent `u16`, so slices of [`F16`] alias `u16` slices
/// for zero-copy upload to a `GPU`.
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct F16(u16);

impl F16 {
    /// Positive zero.
    pub const ZERO: Self = Self(0x0000);
    /// Negative zero.
    pub const NEG_ZERO: Self = Self(0x8000);
    /// One.
    pub const ONE: Self = Self(0x3C00);
    /// Negative one.
    pub const NEG_ONE: Self = Self(0xBC00);
    /// Positive infinity.
    pub const INFINITY: Self = Self(0x7C00);
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(0xFC00);
    /// A quiet `NaN`.
    pub const NAN: Self = Self(0x7E00);
    /// The largest finite value (65504).
    pub const MAX: Self = Self(0x7BFF);
    /// The smallest (most negative) finite value (-65504).
    pub const MIN: Self = Self(0xFBFF);
    /// The smallest positive normal value (`2^-14`).
    pub const MIN_POSITIVE: Self = Self(0x0400);
    /// Machine epsilon: the difference between `1.0` and the next larger value.
    pub const EPSILON: Self = Self(0x1400);

    /// Construct from raw `binary16` bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Return the raw `binary16` bits.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert an `f32` to `f16`, rounding to nearest with ties to even.
    ///
    /// Overflow saturates to the correctly signed infinity; subnormals and
    /// signed zeros are preserved; `NaN` maps to a quiet `NaN`.
    #[inline]
    #[must_use]
    pub const fn from_f32(value: f32) -> Self {
        Self(f32_to_f16_bits(value))
    }

    /// Convert this `f16` to an `f32` exactly.
    #[inline]
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        f16_bits_to_f32(self.0)
    }

    /// True if this value is `NaN`.
    #[inline]
    #[must_use]
    pub const fn is_nan(self) -> bool {
        (self.0 & 0x7C00) == 0x7C00 && (self.0 & 0x03FF) != 0
    }

    /// True if this value is positive or negative infinity.
    #[inline]
    #[must_use]
    pub const fn is_infinite(self) -> bool {
        (self.0 & 0x7FFF) == 0x7C00
    }

    /// True if this value is neither infinite nor `NaN`.
    #[inline]
    #[must_use]
    pub const fn is_finite(self) -> bool {
        (self.0 & 0x7C00) != 0x7C00
    }

    /// The absolute value.
    #[inline]
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0 & 0x7FFF)
    }

    /// Linear interpolation, computed in `f32` and rounded back to `f16`.
    #[inline]
    #[must_use]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        let a = self.to_f32();
        Self::from_f32(a + (rhs.to_f32() - a) * t)
    }
}

/// Round-to-nearest-even `f32` -> `binary16` bit conversion.
///
/// This follows the classic branch-based reference (sign/exponent/mantissa
/// decomposition with explicit subnormal and rounding handling) so the result
/// is bit-exact against a widening oracle.
#[inline]
const fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = x & 0x8000_0000;
    let exp = x & 0x7F80_0000;
    let man = x & 0x007F_FFFF;

    // Infinity or NaN: all exponent bits set.
    if exp == 0x7F80_0000 {
        // A non-zero mantissa is NaN; force the quiet bit and keep high bits.
        let nan_bit = if man == 0 { 0 } else { 0x0200 };
        return ((sign >> 16) | 0x7C00 | nan_bit | (man >> 13)) as u16;
    }

    let half_sign = sign >> 16;
    // Rebias the exponent: f32 bias 127 -> f16 bias 15.
    let unbiased = ((exp >> 23) as i32) - 127;
    let half_exp = unbiased + 15;

    // Overflow of the exponent saturates to infinity.
    if half_exp >= 0x1F {
        return (half_sign | 0x7C00) as u16;
    }

    // Subnormal (or zero) half result.
    if half_exp <= 0 {
        // Shift beyond the representable range rounds to zero.
        if 14 - half_exp > 24 {
            return half_sign as u16;
        }
        // Restore the implicit leading one, then shift into place.
        let man = man | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_man = man >> shift;
        // Round to nearest, ties to even.
        let round_bit = 1u32 << (13 - half_exp);
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
            half_man += 1;
        }
        return (half_sign | half_man) as u16;
    }

    // Normalized half result.
    let half_exp_bits = (half_exp as u32) << 10;
    let half_man = man >> 13;
    let round_bit = 0x0000_1000;
    if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
        // Rounding up may carry into the exponent, which is intentional.
        return ((half_sign | half_exp_bits | half_man) + 1) as u16;
    }
    (half_sign | half_exp_bits | half_man) as u16
}

/// Exact `binary16` -> `f32` bit conversion.
#[inline]
const fn f16_bits_to_f32(i: u16) -> f32 {
    // Signed zero fast path.
    if i & 0x7FFF == 0 {
        return f32::from_bits((i as u32) << 16);
    }

    let half_sign = (i & 0x8000) as u32;
    let half_exp = (i & 0x7C00) as u32;
    let half_man = (i & 0x03FF) as u32;
    let sign = half_sign << 16;

    // Infinity or NaN.
    if half_exp == 0x7C00 {
        if half_man == 0 {
            return f32::from_bits(sign | 0x7F80_0000);
        }
        return f32::from_bits(sign | 0x7FC0_0000 | (half_man << 13));
    }

    // Subnormal: renormalize into an f32 normal.
    if half_exp == 0 {
        let e = (half_man as u16).leading_zeros() - 6;
        let exp = (127 - 15 - e) << 23;
        let man = (half_man << (14 + e)) & 0x7F_FFFF;
        return f32::from_bits(sign | exp | man);
    }

    // Normalized: rebias and widen the mantissa.
    let unbiased = ((half_exp >> 10) as i32) - 15;
    let exp = ((unbiased + 127) as u32) << 23;
    let man = half_man << 13;
    f32::from_bits(sign | exp | man)
}

impl From<f32> for F16 {
    #[inline]
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}
impl From<F16> for f32 {
    #[inline]
    fn from(v: F16) -> Self {
        v.to_f32()
    }
}

impl PartialEq for F16 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Compare as `f32` so `NaN != NaN` and `+0 == -0`.
        self.to_f32() == other.to_f32()
    }
}

impl PartialOrd for F16 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl Neg for F16 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(self.0 ^ 0x8000)
    }
}

macro_rules! impl_f16_binop {
    ($trait:ident, $method:ident) => {
        impl $trait for F16 {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                Self::from_f32($trait::$method(self.to_f32(), rhs.to_f32()))
            }
        }
    };
}
impl_f16_binop!(Add, add);
impl_f16_binop!(Sub, sub);
impl_f16_binop!(Mul, mul);
impl_f16_binop!(Div, div);

impl core::fmt::Debug for F16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "F16({})", self.to_f32())
    }
}
impl core::fmt::Display for F16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

/// A 2-component half-precision storage vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct F16Vec2 {
    /// X component.
    pub x: F16,
    /// Y component.
    pub y: F16,
}

/// A 3-component half-precision storage vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct F16Vec3 {
    /// X component.
    pub x: F16,
    /// Y component.
    pub y: F16,
    /// Z component.
    pub z: F16,
}

/// A 4-component half-precision storage vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct F16Vec4 {
    /// X component.
    pub x: F16,
    /// Y component.
    pub y: F16,
    /// Z component.
    pub z: F16,
    /// W component.
    pub w: F16,
}

impl F16Vec2 {
    /// Build from raw half components.
    #[inline]
    #[must_use]
    pub const fn new(x: F16, y: F16) -> Self {
        Self { x, y }
    }
    /// Round an `f32` [`Vec2`] to half precision.
    #[inline]
    #[must_use]
    pub const fn from_vec2(v: Vec2) -> Self {
        Self {
            x: F16::from_f32(v.x),
            y: F16::from_f32(v.y),
        }
    }
    /// Widen back to an `f32` [`Vec2`].
    #[inline]
    #[must_use]
    pub const fn to_vec2(self) -> Vec2 {
        Vec2::new(self.x.to_f32(), self.y.to_f32())
    }
}

impl F16Vec3 {
    /// Build from raw half components.
    #[inline]
    #[must_use]
    pub const fn new(x: F16, y: F16, z: F16) -> Self {
        Self { x, y, z }
    }
    /// Round an `f32` [`Vec3`] to half precision.
    #[inline]
    #[must_use]
    pub const fn from_vec3(v: Vec3) -> Self {
        Self {
            x: F16::from_f32(v.x),
            y: F16::from_f32(v.y),
            z: F16::from_f32(v.z),
        }
    }
    /// Widen back to an `f32` [`Vec3`].
    #[inline]
    #[must_use]
    pub const fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x.to_f32(), self.y.to_f32(), self.z.to_f32())
    }
}

impl F16Vec4 {
    /// Build from raw half components.
    #[inline]
    #[must_use]
    pub const fn new(x: F16, y: F16, z: F16, w: F16) -> Self {
        Self { x, y, z, w }
    }
    /// Round an `f32` [`Vec4`] to half precision.
    #[inline]
    #[must_use]
    pub const fn from_vec4(v: Vec4) -> Self {
        Self {
            x: F16::from_f32(v.x),
            y: F16::from_f32(v.y),
            z: F16::from_f32(v.z),
            w: F16::from_f32(v.w),
        }
    }
    /// Widen back to an `f32` [`Vec4`].
    #[inline]
    #[must_use]
    pub const fn to_vec4(self) -> Vec4 {
        Vec4::new(
            self.x.to_f32(),
            self.y.to_f32(),
            self.z.to_f32(),
            self.w.to_f32(),
        )
    }
}

impl From<Vec2> for F16Vec2 {
    #[inline]
    fn from(v: Vec2) -> Self {
        Self::from_vec2(v)
    }
}
impl From<F16Vec2> for Vec2 {
    #[inline]
    fn from(v: F16Vec2) -> Self {
        v.to_vec2()
    }
}
impl From<Vec3> for F16Vec3 {
    #[inline]
    fn from(v: Vec3) -> Self {
        Self::from_vec3(v)
    }
}
impl From<F16Vec3> for Vec3 {
    #[inline]
    fn from(v: F16Vec3) -> Self {
        v.to_vec3()
    }
}
impl From<Vec4> for F16Vec4 {
    #[inline]
    fn from(v: Vec4) -> Self {
        Self::from_vec4(v)
    }
}
impl From<F16Vec4> for Vec4 {
    #[inline]
    fn from(v: F16Vec4) -> Self {
        v.to_vec4()
    }
}
