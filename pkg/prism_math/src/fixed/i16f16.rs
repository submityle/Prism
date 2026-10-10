//! The lightweight fixed-point scalar type [`I16F16`] (signed Q16.16).
//!
//! `I16F16` stores a value as a raw [`i32`] interpreted as `raw / 2^16`
//! (16 integer bits, 16 fractional bits — the design's *Q16.16 angle* format,
//! see design doc §6/§13). It trades range and precision for half the storage
//! of [`Fixed`](super::Fixed) and is handy for compact angles, normalized
//! parameters, and gameplay scalars where Q32.32 is overkill.
//!
//! Like [`Fixed`](super::Fixed), all arithmetic is exact integer math with a
//! documented saturating/rounding policy, so results are bit-identical across
//! platforms. Transcendental functions live on [`Fixed`](super::Fixed); widen
//! with [`I16F16::to_fixed`] to use them, then narrow back with
//! [`I16F16::from_fixed`].

use super::Fixed;
use core::cmp::Ordering;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// Number of fractional bits in [`I16F16`] (Q16.16).
pub const FRAC_BITS: u32 = 16;

/// A signed Q16.16 fixed-point scalar backed by an [`i32`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct I16F16 {
    raw: i32,
}

/// Clamp a wide intermediate down to the [`i32`] raw range.
#[inline]
const fn saturate_i64(v: i64) -> i32 {
    if v > i32::MAX as i64 {
        i32::MAX
    } else if v < i32::MIN as i64 {
        i32::MIN
    } else {
        v as i32
    }
}

impl I16F16 {
    /// The scaling factor `2^16` as a raw integer.
    pub const ONE_BITS: i32 = 1 << FRAC_BITS;

    /// Zero.
    pub const ZERO: Self = Self { raw: 0 };
    /// One.
    pub const ONE: Self = Self {
        raw: Self::ONE_BITS,
    };
    /// Negative one.
    pub const NEG_ONE: Self = Self {
        raw: -Self::ONE_BITS,
    };
    /// One half (`0.5`).
    pub const HALF: Self = Self {
        raw: Self::ONE_BITS >> 1,
    };
    /// The most negative representable value.
    pub const MIN: Self = Self { raw: i32::MIN };
    /// The most positive representable value.
    pub const MAX: Self = Self { raw: i32::MAX };
    /// The smallest positive step between representable values (`2^-16`).
    pub const EPSILON: Self = Self { raw: 1 };

    /// Reinterpret a raw Q16.16 integer as an [`I16F16`].
    #[inline]
    pub const fn from_bits(raw: i32) -> Self {
        Self { raw }
    }
    /// Return the raw Q16.16 integer.
    #[inline]
    pub const fn to_bits(self) -> i32 {
        self.raw
    }
    /// Construct from a whole integer, saturating on overflow.
    #[inline]
    pub const fn from_int(n: i32) -> Self {
        Self {
            raw: saturate_i64((n as i64) << FRAC_BITS),
        }
    }
    /// Truncate toward zero to a whole integer.
    #[inline]
    pub const fn to_int(self) -> i32 {
        self.raw >> FRAC_BITS
    }

    /// Widen to a [`Fixed`] (exact: Q16.16 is a sub-lattice of Q32.32).
    #[inline]
    pub const fn to_fixed(self) -> Fixed {
        // Shift the fractional point from 16 to 32 bits.
        Fixed::from_bits((self.raw as i64) << (super::scalar::FRAC_BITS - FRAC_BITS))
    }
    /// Narrow from a [`Fixed`] (round to nearest, ties toward `+∞`; saturating).
    #[inline]
    pub const fn from_fixed(x: Fixed) -> Self {
        let shift = super::scalar::FRAC_BITS - FRAC_BITS;
        let rounded = (x.to_bits() + (1 << (shift - 1))) >> shift;
        Self {
            raw: saturate_i64(rounded),
        }
    }

    /// Construct from an `f64` (round to nearest). Not on the deterministic
    /// path; prefer [`from_bits`](Self::from_bits)/[`from_int`](Self::from_int).
    #[inline]
    pub fn from_f64(x: f64) -> Self {
        let r = libm::round(x * (Self::ONE_BITS as f64));
        if r >= i32::MAX as f64 {
            Self::MAX
        } else if r <= i32::MIN as f64 {
            Self::MIN
        } else {
            Self { raw: r as i32 }
        }
    }
    /// Lossy conversion to `f64` (display/debug only).
    #[inline]
    pub fn to_f64(self) -> f64 {
        (self.raw as f64) / (Self::ONE_BITS as f64)
    }
    /// Lossy conversion to `f32` (display/debug only).
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// True if exactly zero.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.raw == 0
    }

    /// Absolute value (saturating).
    #[inline]
    pub const fn abs(self) -> Self {
        if self.raw < 0 {
            Self {
                raw: self.raw.saturating_neg(),
            }
        } else {
            self
        }
    }
    /// Minimum of two values.
    #[inline]
    pub const fn min(self, rhs: Self) -> Self {
        if self.raw <= rhs.raw {
            self
        } else {
            rhs
        }
    }
    /// Maximum of two values.
    #[inline]
    pub const fn max(self, rhs: Self) -> Self {
        if self.raw >= rhs.raw {
            self
        } else {
            rhs
        }
    }
    /// Clamp into `[lo, hi]`.
    #[inline]
    pub const fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Saturating addition.
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self {
            raw: self.raw.saturating_add(rhs.raw),
        }
    }
    /// Saturating subtraction.
    #[inline]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            raw: self.raw.saturating_sub(rhs.raw),
        }
    }
    /// Saturating multiplication (round-to-nearest intermediate).
    #[inline]
    pub const fn saturating_mul(self, rhs: Self) -> Self {
        let p = (self.raw as i64) * (rhs.raw as i64);
        let rounded = (p + (1 << (FRAC_BITS - 1))) >> FRAC_BITS;
        Self {
            raw: saturate_i64(rounded),
        }
    }
    /// Saturating division (truncates toward zero; divide-by-zero saturates by
    /// sign, with `0/0 == 0`).
    #[inline]
    pub const fn saturating_div(self, rhs: Self) -> Self {
        if rhs.raw == 0 {
            if self.raw > 0 {
                Self::MAX
            } else if self.raw < 0 {
                Self::MIN
            } else {
                Self::ZERO
            }
        } else {
            let num = (self.raw as i64) << FRAC_BITS;
            Self {
                raw: saturate_i64(num / (rhs.raw as i64)),
            }
        }
    }
    /// Saturating negation.
    #[inline]
    pub const fn saturating_neg(self) -> Self {
        Self {
            raw: self.raw.saturating_neg(),
        }
    }
    /// Checked multiplication, `None` on overflow.
    #[inline]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        let p = (self.raw as i64) * (rhs.raw as i64);
        let rounded = (p + (1 << (FRAC_BITS - 1))) >> FRAC_BITS;
        if rounded > i32::MAX as i64 || rounded < i32::MIN as i64 {
            None
        } else {
            Some(Self {
                raw: rounded as i32,
            })
        }
    }
}

impl PartialOrd for I16F16 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for I16F16 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.raw.cmp(&other.raw)
    }
}

impl Add for I16F16 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }
}
impl Sub for I16F16 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }
}
impl Mul for I16F16 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.saturating_mul(rhs)
    }
}
impl Div for I16F16 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.saturating_div(rhs)
    }
}
impl Rem for I16F16 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Self {
            raw: if rhs.raw == 0 { 0 } else { self.raw % rhs.raw },
        }
    }
}
impl Neg for I16F16 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.saturating_neg()
    }
}
impl AddAssign for I16F16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for I16F16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl MulAssign for I16F16 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl DivAssign for I16F16 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl RemAssign for I16F16 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl core::fmt::Debug for I16F16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I16F16({} [raw={}])", self.to_f64(), self.raw)
    }
}

impl From<i16> for I16F16 {
    #[inline]
    fn from(n: i16) -> Self {
        Self::from_int(n as i32)
    }
}
