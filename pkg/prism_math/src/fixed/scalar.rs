//! The core fixed-point scalar type [`Fixed`] (signed Q32.32).
//!
//! `Fixed` stores a value as a raw [`i64`] interpreted as `raw / 2^32`, giving
//! 32 integer bits and 32 fractional bits (the design's *Q32.32 position*
//! format, see design doc §6/§13). Every arithmetic result is a deterministic
//! function of the integer `raw` bits alone: there is **no hardware float on
//! the arithmetic path**, so the same inputs produce bit-identical outputs on
//! every platform. This is the numerical bedrock of Prism's "four-way
//! determinism" (ECS / tasks / time / transform).
//!
//! ## Overflow policy
//! The `core::ops` operators ([`Add`], [`Sub`], [`Mul`], [`Div`], [`Neg`]) are
//! **saturating**: a result outside `[Fixed::MIN, Fixed::MAX]` clamps to the
//! nearest bound instead of wrapping or panicking. Saturation is fully
//! deterministic and avoids silent wrap-around desyncs. Explicit
//! `wrapping_*`, `checked_*`, and `saturating_*` methods are provided when a
//! different policy is wanted.
//!
//! ## Rounding
//! Multiplication rounds the 64-bit intermediate to the nearest Q32.32 value
//! (ties toward `+∞`). Division truncates toward zero. Both choices are fixed
//! and documented so they stay bit-stable across versions.
//!
//! ## Float conversions
//! [`Fixed::from_f64`]/[`Fixed::from_f32`] and the `to_f*` methods exist for
//! authoring, I/O, and debugging. They touch hardware float and therefore are
//! **not** part of the deterministic path — never feed a float conversion
//! result straight into a networked simulation step and expect cross-platform
//! bit-exactness. Use [`Fixed::from_bits`]/[`Fixed::from_int`] for exact,
//! deterministic construction.

use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

/// Number of fractional bits in [`Fixed`] (Q32.32).
pub const FRAC_BITS: u32 = 32;

/// A signed Q32.32 fixed-point scalar backed by an [`i64`].
///
/// The stored value is `to_bits() as f64 / 2f64.powi(32)` conceptually, but all
/// arithmetic is performed in exact integer form. See the [module
/// docs](self) for the overflow, rounding, and determinism contract.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Fixed {
    raw: i64,
}

/// Clamp a wide Q32.32 intermediate down to the [`i64`] raw range.
#[inline]
const fn saturate_i128(v: i128) -> i64 {
    if v > i64::MAX as i128 {
        i64::MAX
    } else if v < i64::MIN as i128 {
        i64::MIN
    } else {
        v as i64
    }
}

impl Fixed {
    /// The scaling factor `2^32` as a raw integer (`Fixed::ONE.to_bits()`).
    pub const ONE_BITS: i64 = 1 << FRAC_BITS;

    /// Zero.
    pub const ZERO: Self = Self { raw: 0 };
    /// One.
    pub const ONE: Self = Self { raw: Self::ONE_BITS };
    /// Negative one.
    pub const NEG_ONE: Self = Self { raw: -Self::ONE_BITS };
    /// One half (`0.5`).
    pub const HALF: Self = Self { raw: Self::ONE_BITS >> 1 };
    /// Two.
    pub const TWO: Self = Self { raw: Self::ONE_BITS << 1 };
    /// The most negative representable value (`-2^31`).
    pub const MIN: Self = Self { raw: i64::MIN };
    /// The most positive representable value (`≈ 2^31 - ε`).
    pub const MAX: Self = Self { raw: i64::MAX };
    /// The smallest positive step between representable values (`2^-32`).
    pub const EPSILON: Self = Self { raw: 1 };

    /// π (ratio of a circle's circumference to its diameter).
    pub const PI: Self = Self { raw: 13_493_037_705 };
    /// τ = 2π.
    pub const TAU: Self = Self { raw: 26_986_075_409 };
    /// π/2.
    pub const FRAC_PI_2: Self = Self { raw: 6_746_518_852 };
    /// π/4.
    pub const FRAC_PI_4: Self = Self { raw: 3_373_259_426 };
    /// Natural logarithm of 2.
    pub const LN_2: Self = Self { raw: 2_977_044_472 };
    /// Euler's number `e`.
    pub const E: Self = Self { raw: 11_674_931_555 };

    // -- construction ------------------------------------------------------

    /// Reinterpret a raw Q32.32 integer as a [`Fixed`] (exact, deterministic).
    #[inline]
    pub const fn from_bits(raw: i64) -> Self {
        Self { raw }
    }

    /// Return the raw Q32.32 integer (exact, deterministic).
    #[inline]
    pub const fn to_bits(self) -> i64 {
        self.raw
    }

    /// Construct from a whole integer, saturating on overflow.
    #[inline]
    pub const fn from_int(n: i64) -> Self {
        // n * 2^32 can overflow i64, so widen first.
        Self { raw: saturate_i128((n as i128) << FRAC_BITS) }
    }

    /// Truncate toward zero to a whole integer.
    #[inline]
    pub const fn to_int(self) -> i64 {
        self.raw >> FRAC_BITS
    }

    /// Construct from an `f64` (round to nearest). **Not** on the deterministic
    /// path; see the [module docs](self).
    #[inline]
    pub fn from_f64(x: f64) -> Self {
        let scaled = x * (Self::ONE_BITS as f64);
        let r = libm::round(scaled);
        if r >= i64::MAX as f64 {
            Self::MAX
        } else if r <= i64::MIN as f64 {
            Self::MIN
        } else {
            Self { raw: r as i64 }
        }
    }

    /// Construct from an `f32` (round to nearest). **Not** on the deterministic
    /// path; see the [module docs](self).
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        Self::from_f64(x as f64)
    }

    /// Lossy conversion to `f64` (for display/debug, not the deterministic
    /// path).
    #[inline]
    pub fn to_f64(self) -> f64 {
        (self.raw as f64) / (Self::ONE_BITS as f64)
    }

    /// Lossy conversion to `f32` (for display/debug, not the deterministic
    /// path).
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    // -- predicates --------------------------------------------------------

    /// True if exactly zero.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.raw == 0
    }
    /// True if strictly greater than zero.
    #[inline]
    pub const fn is_positive(self) -> bool {
        self.raw > 0
    }
    /// True if strictly less than zero.
    #[inline]
    pub const fn is_negative(self) -> bool {
        self.raw < 0
    }

    // -- sign / rounding ---------------------------------------------------

    /// Absolute value (saturating: `MIN.abs() == MAX`).
    #[inline]
    pub const fn abs(self) -> Self {
        if self.raw < 0 {
            Self { raw: self.raw.saturating_neg() }
        } else {
            self
        }
    }

    /// `-1`, `0`, or `+1` depending on sign.
    #[inline]
    pub const fn signum(self) -> Self {
        if self.raw > 0 {
            Self::ONE
        } else if self.raw < 0 {
            Self::NEG_ONE
        } else {
            Self::ZERO
        }
    }

    /// Largest integer `≤ self`.
    #[inline]
    pub const fn floor(self) -> Self {
        Self { raw: self.raw & !(Self::ONE_BITS - 1) }
    }

    /// Smallest integer `≥ self`.
    #[inline]
    pub const fn ceil(self) -> Self {
        let frac_mask = Self::ONE_BITS - 1;
        if self.raw & frac_mask == 0 {
            self
        } else {
            Self { raw: (self.raw & !frac_mask).saturating_add(Self::ONE_BITS) }
        }
    }

    /// Round to the nearest integer (ties toward `+∞`).
    #[inline]
    pub const fn round(self) -> Self {
        Self { raw: (self.raw.saturating_add(Self::ONE_BITS >> 1)) & !(Self::ONE_BITS - 1) }
    }

    /// Truncate toward zero to an integer value.
    #[inline]
    pub const fn trunc(self) -> Self {
        Self { raw: (self.raw / Self::ONE_BITS) * Self::ONE_BITS }
    }

    /// Fractional part `self - self.trunc()` (keeps the sign of `self`).
    #[inline]
    pub const fn fract(self) -> Self {
        Self { raw: self.raw - self.trunc().raw }
    }

    // -- min / max / clamp -------------------------------------------------

    /// Minimum of two values.
    #[inline]
    pub const fn min(self, rhs: Self) -> Self {
        if self.raw <= rhs.raw { self } else { rhs }
    }
    /// Maximum of two values.
    #[inline]
    pub const fn max(self, rhs: Self) -> Self {
        if self.raw >= rhs.raw { self } else { rhs }
    }
    /// Clamp into `[lo, hi]`.
    #[inline]
    pub const fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    // -- wrapping arithmetic ----------------------------------------------

    /// Wrapping addition (modular on the raw `i64`).
    #[inline]
    pub const fn wrapping_add(self, rhs: Self) -> Self {
        Self { raw: self.raw.wrapping_add(rhs.raw) }
    }
    /// Wrapping subtraction (modular on the raw `i64`).
    #[inline]
    pub const fn wrapping_sub(self, rhs: Self) -> Self {
        Self { raw: self.raw.wrapping_sub(rhs.raw) }
    }
    /// Wrapping multiplication (round-to-nearest, modular on overflow).
    #[inline]
    pub const fn wrapping_mul(self, rhs: Self) -> Self {
        let p = (self.raw as i128) * (rhs.raw as i128);
        let rounded = (p + (1 << (FRAC_BITS - 1))) >> FRAC_BITS;
        Self { raw: rounded as i64 }
    }

    // -- saturating arithmetic --------------------------------------------

    /// Saturating addition.
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self { raw: self.raw.saturating_add(rhs.raw) }
    }
    /// Saturating subtraction.
    #[inline]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self { raw: self.raw.saturating_sub(rhs.raw) }
    }
    /// Saturating multiplication (round-to-nearest intermediate).
    #[inline]
    pub const fn saturating_mul(self, rhs: Self) -> Self {
        let p = (self.raw as i128) * (rhs.raw as i128);
        let rounded = (p + (1 << (FRAC_BITS - 1))) >> FRAC_BITS;
        Self { raw: saturate_i128(rounded) }
    }
    /// Saturating division (truncates toward zero; divide-by-zero saturates to
    /// `MAX`/`MIN` by sign, with `0/0 == 0`).
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
            let num = (self.raw as i128) << FRAC_BITS;
            Self { raw: saturate_i128(num / (rhs.raw as i128)) }
        }
    }
    /// Saturating negation (`MIN` negates to `MAX`).
    #[inline]
    pub const fn saturating_neg(self) -> Self {
        Self { raw: self.raw.saturating_neg() }
    }

    // -- checked arithmetic -----------------------------------------------

    /// Checked addition, `None` on overflow.
    #[inline]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        match self.raw.checked_add(rhs.raw) {
            Some(raw) => Some(Self { raw }),
            None => None,
        }
    }
    /// Checked subtraction, `None` on overflow.
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        match self.raw.checked_sub(rhs.raw) {
            Some(raw) => Some(Self { raw }),
            None => None,
        }
    }
    /// Checked multiplication, `None` on overflow.
    #[inline]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        let p = (self.raw as i128) * (rhs.raw as i128);
        let rounded = (p + (1 << (FRAC_BITS - 1))) >> FRAC_BITS;
        if rounded > i64::MAX as i128 || rounded < i64::MIN as i128 {
            None
        } else {
            Some(Self { raw: rounded as i64 })
        }
    }
    /// Checked division, `None` on divide-by-zero or overflow.
    #[inline]
    pub const fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.raw == 0 {
            return None;
        }
        let num = (self.raw as i128) << FRAC_BITS;
        let q = num / (rhs.raw as i128);
        if q > i64::MAX as i128 || q < i64::MIN as i128 {
            None
        } else {
            Some(Self { raw: q as i64 })
        }
    }

    /// Reciprocal `1 / self` (saturating).
    #[inline]
    pub const fn recip(self) -> Self {
        Self::ONE.saturating_div(self)
    }
}

// -- ordering ---------------------------------------------------------------

impl PartialOrd for Fixed {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Fixed {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.raw.cmp(&other.raw)
    }
}

// -- operators (saturating) -------------------------------------------------

impl Add for Fixed {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }
}
impl Sub for Fixed {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }
}
impl Mul for Fixed {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.saturating_mul(rhs)
    }
}
impl Div for Fixed {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.saturating_div(rhs)
    }
}
impl Rem for Fixed {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        // Exact: raw remainder shares the Q32.32 scale.
        Self { raw: if rhs.raw == 0 { 0 } else { self.raw % rhs.raw } }
    }
}
impl Neg for Fixed {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.saturating_neg()
    }
}

impl AddAssign for Fixed {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for Fixed {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl MulAssign for Fixed {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl DivAssign for Fixed {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl RemAssign for Fixed {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl core::fmt::Debug for Fixed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Show both the decimal approximation and the exact raw bits so that
        // determinism bugs are debuggable without hiding the authoritative
        // integer representation.
        write!(f, "Fixed({} [raw={}])", self.to_f64(), self.raw)
    }
}

impl From<i32> for Fixed {
    #[inline]
    fn from(n: i32) -> Self {
        Self::from_int(n as i64)
    }
}
