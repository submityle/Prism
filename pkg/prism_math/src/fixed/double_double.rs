//! Double-double extended precision (design doc §24.4).
//!
//! A [`DoubleDouble`] represents a real value as an unevaluated sum of two
//! non-overlapping `f64` lanes, `hi + lo`, where `|lo| <= 0.5 * ulp(hi)`. This
//! yields roughly 106 bits of significand (about 31 decimal digits) without
//! needing a `f128` type or software big-float.
//!
//! It is the extended-precision companion to the compensated-sum helpers in
//! [`compensated`](super::compensated): where [`KahanSum`](super::KahanSum)
//! keeps a *long reduction* accurate, [`DoubleDouble`] keeps *individual
//! arithmetic* (`+ - * /`, `sqrt`) accurate across a whole expression. The two
//! target the same §24.4 use cases — extreme big-world coordinate accumulation
//! and high-precision offline baking — where plain `f64` drift becomes visible.
//!
//! ## Relationship to the fixed-point path
//! Like the compensated sums, this is a **single-platform, low-drift** aid, not
//! a cross-platform bit-exact substitute for [`Fixed`](super::Fixed). The
//! error-free transforms below are exact *per IEEE-754 rules*, but they lean on
//! correctly-rounded `f64` add/multiply and a fused multiply-add
//! (via [`libm::fma`]), so bit-exact reproducibility holds only across
//! platforms that honor those same IEEE operations. When cross-machine lockstep
//! determinism is required, use [`Fixed`](super::Fixed) instead.
//!
//! ## Algorithms
//! The building blocks are the classic Dekker/Knuth error-free transforms
//! (`two_sum`, `quick_two_sum`, `two_prod`) and the Hida–Li–Bailey QD-library
//! formulas for the composite operations. All are branch-light and allocation
//! free.

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Knuth `two_sum`: returns `(s, e)` with `s = fl(a + b)` and `s + e = a + b`
/// exactly. Works for any ordering of magnitudes.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// Dekker `quick_two_sum`: returns `(s, e)` with `s + e = a + b` exactly,
/// **requiring** `|a| >= |b|`.
#[inline]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let err = b - (s - a);
    (s, err)
}

/// `two_prod` via fused multiply-add: returns `(p, e)` with `p = fl(a * b)` and
/// `p + e = a * b` exactly.
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    // `fma(a, b, -p)` is the exact rounding error of `a * b`.
    let err = libm::fma(a, b, -p);
    (p, err)
}

/// An extended-precision real number stored as two non-overlapping `f64` lanes.
///
/// The value represented is exactly `hi + lo` with the invariant
/// `|lo| <= 0.5 * ulp(hi)` maintained by every constructor and operator.
#[derive(Clone, Copy)]
pub struct DoubleDouble {
    hi: f64,
    lo: f64,
}

impl DoubleDouble {
    /// The additive identity, `0`.
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    /// The multiplicative identity, `1`.
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    /// Construct from a single `f64` (exact; `lo` is zero).
    #[inline]
    #[must_use]
    pub const fn from_f64(value: f64) -> Self {
        Self {
            hi: value,
            lo: 0.0,
        }
    }

    /// Construct from two lanes, renormalizing so the result satisfies the
    /// non-overlap invariant. Use this when `hi`/`lo` come from an arbitrary
    /// split that may not already be normalized.
    #[inline]
    #[must_use]
    pub fn from_lanes(hi: f64, lo: f64) -> Self {
        let (hi, lo) = quick_two_sum(hi, lo);
        Self { hi, lo }
    }

    /// The leading (high-order) `f64` lane.
    #[inline]
    #[must_use]
    pub const fn hi(self) -> f64 {
        self.hi
    }

    /// The trailing (low-order) `f64` lane.
    #[inline]
    #[must_use]
    pub const fn lo(self) -> f64 {
        self.lo
    }

    /// Round to the nearest `f64` (the leading lane already carries the
    /// correctly-rounded value).
    #[inline]
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        self.hi
    }

    /// Negation.
    #[inline]
    #[must_use]
    pub const fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    /// Absolute value.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        if self.hi < 0.0 { self.neg() } else { self }
    }

    /// `true` if both lanes are zero.
    #[inline]
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.hi == 0.0 && self.lo == 0.0
    }

    /// `true` if either lane is `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.hi.is_nan() || self.lo.is_nan()
    }

    /// Add a plain `f64`, keeping extended precision.
    #[inline]
    #[must_use]
    pub fn add_f64(self, b: f64) -> Self {
        let (s1, s2) = two_sum(self.hi, b);
        let s2 = s2 + self.lo;
        let (hi, lo) = quick_two_sum(s1, s2);
        Self { hi, lo }
    }

    /// Multiply by a plain `f64`, keeping extended precision.
    #[inline]
    #[must_use]
    pub fn mul_f64(self, b: f64) -> Self {
        let (p1, p2) = two_prod(self.hi, b);
        let p2 = p2 + self.lo * b;
        let (hi, lo) = quick_two_sum(p1, p2);
        Self { hi, lo }
    }

    /// Add two [`DoubleDouble`] values (IEEE-accurate QD addition).
    #[inline]
    #[must_use]
    pub fn add_dd(self, b: Self) -> Self {
        let (s1, s2) = two_sum(self.hi, b.hi);
        let (t1, t2) = two_sum(self.lo, b.lo);
        let s2 = s2 + t1;
        let (s1, s2) = quick_two_sum(s1, s2);
        let s2 = s2 + t2;
        let (hi, lo) = quick_two_sum(s1, s2);
        Self { hi, lo }
    }

    /// Subtract two [`DoubleDouble`] values.
    #[inline]
    #[must_use]
    pub fn sub_dd(self, b: Self) -> Self {
        self.add_dd(b.neg())
    }

    /// Multiply two [`DoubleDouble`] values.
    #[inline]
    #[must_use]
    pub fn mul_dd(self, b: Self) -> Self {
        let (p1, p2) = two_prod(self.hi, b.hi);
        // Cross terms are an order of magnitude smaller; `fma`-exact `p2`
        // already captures the `hi*hi` rounding error.
        let p2 = p2 + (self.hi * b.lo + self.lo * b.hi);
        let (hi, lo) = quick_two_sum(p1, p2);
        Self { hi, lo }
    }

    /// Square (slightly cheaper than [`mul_dd`](Self::mul_dd) with itself).
    #[inline]
    #[must_use]
    pub fn sqr(self) -> Self {
        let (p1, p2) = two_prod(self.hi, self.hi);
        let p2 = p2 + 2.0 * self.hi * self.lo;
        let (hi, lo) = quick_two_sum(p1, p2);
        Self { hi, lo }
    }

    /// Divide two [`DoubleDouble`] values (QD accurate division: three
    /// correcting Newton quotients).
    #[inline]
    #[must_use]
    pub fn div_dd(self, b: Self) -> Self {
        let q1 = self.hi / b.hi;
        let r = self.sub_dd(b.mul_f64(q1));
        let q2 = r.hi / b.hi;
        let r = r.sub_dd(b.mul_f64(q2));
        let q3 = r.hi / b.hi;
        let (hi, lo) = quick_two_sum(q1, q2);
        Self { hi, lo }.add_f64(q3)
    }

    /// Square root (Karp's method: one Newton step in extended precision).
    ///
    /// Returns [`ZERO`](Self::ZERO) for a zero input and a `NaN`
    /// [`DoubleDouble`] for a negative input.
    #[inline]
    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }
        if self.hi < 0.0 {
            return Self::from_f64(f64::NAN);
        }
        // `x` approximates 1/sqrt(a) in plain `f64`.
        let x = 1.0 / libm::sqrt(self.hi);
        let ax = self.hi * x;
        // Correction: sqrt(a) ≈ ax + (a - ax^2) * x / 2, evaluated so the
        // residual is formed in extended precision.
        let diff = self.sub_dd(Self::from_two_prod(ax, ax));
        let e = diff.hi * x * 0.5;
        Self::from_f64(ax).add_f64(e)
    }

    /// Exact product of two `f64` values as a normalized [`DoubleDouble`].
    #[inline]
    #[must_use]
    pub fn from_two_prod(a: f64, b: f64) -> Self {
        let (hi, lo) = two_prod(a, b);
        Self { hi, lo }
    }

    /// Total ordering helper over the represented real value.
    #[inline]
    #[must_use]
    fn cmp_value(self, other: Self) -> Option<Ordering> {
        match self.hi.partial_cmp(&other.hi) {
            Some(Ordering::Equal) => self.lo.partial_cmp(&other.lo),
            non_equal => non_equal,
        }
    }
}

impl Default for DoubleDouble {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<f64> for DoubleDouble {
    #[inline]
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<f32> for DoubleDouble {
    #[inline]
    fn from(value: f32) -> Self {
        Self::from_f64(f64::from(value))
    }
}

impl From<i32> for DoubleDouble {
    #[inline]
    fn from(value: i32) -> Self {
        Self::from_f64(f64::from(value))
    }
}

impl PartialEq for DoubleDouble {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.hi == other.hi && self.lo == other.lo
    }
}

impl PartialOrd for DoubleDouble {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp_value(*other)
    }
}

impl Add for DoubleDouble {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.add_dd(rhs)
    }
}

impl Sub for DoubleDouble {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.sub_dd(rhs)
    }
}

impl Mul for DoubleDouble {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_dd(rhs)
    }
}

impl Div for DoubleDouble {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.div_dd(rhs)
    }
}

impl Neg for DoubleDouble {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        DoubleDouble::neg(self)
    }
}

impl AddAssign for DoubleDouble {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add_dd(rhs);
    }
}

impl SubAssign for DoubleDouble {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub_dd(rhs);
    }
}

impl MulAssign for DoubleDouble {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_dd(rhs);
    }
}

impl DivAssign for DoubleDouble {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = self.div_dd(rhs);
    }
}

impl fmt::Debug for DoubleDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DoubleDouble({:e} + {:e})", self.hi, self.lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_bits_f64_loses() {
        // 1e16 has ulp == 2, so `1e16 + 1.0` rounds back to 1e16 in plain f64.
        let plain = 1e16_f64 + 1.0;
        assert_eq!(plain, 1e16_f64, "f64 loses the +1 as expected");

        let dd = DoubleDouble::from_f64(1e16).add_f64(1.0);
        assert_eq!(dd.hi(), 1e16);
        assert_eq!(dd.lo(), 1.0, "double-double keeps the low bit");
        // Subtracting the big part recovers exactly 1.0.
        let back = dd.sub_dd(DoubleDouble::from_f64(1e16));
        assert_eq!(back.to_f64(), 1.0);
    }

    #[test]
    fn sum_of_tenths_beats_f64() {
        // 0.1 is not representable; summing ten of them drifts in plain f64.
        let mut plain = 0.0_f64;
        let mut dd = DoubleDouble::ZERO;
        let tenth = DoubleDouble::from_f64(0.1);
        for _ in 0..10 {
            plain += 0.1;
            dd += tenth;
        }
        // dd's residual to 1.0 is far smaller than f64's.
        let dd_err = libm::fabs(dd.sub_dd(DoubleDouble::ONE).to_f64());
        let f64_err = libm::fabs(plain - 1.0);
        // dd accumulates the ten inexact f64(0.1) terms *without adding its own
        // rounding error*, so its residual (bounded by the input error) is no
        // worse than the plain-f64 accumulation and strictly better here.
        assert!(
            dd_err < f64_err,
            "dd_err={dd_err:e} should beat f64_err={f64_err:e}"
        );
    }

    #[test]
    fn mul_div_roundtrip_is_tight() {
        let a = DoubleDouble::from_f64(3.0).div_dd(DoubleDouble::from_f64(7.0));
        let b = DoubleDouble::from_f64(11.0).div_dd(DoubleDouble::from_f64(13.0));
        let back = a.mul_dd(b).div_dd(b);
        let err = libm::fabs(back.sub_dd(a).to_f64()) / libm::fabs(a.to_f64());
        assert!(err < 1e-28, "mul/div roundtrip rel err={err:e}");
    }

    #[test]
    fn one_third_times_three() {
        let third = DoubleDouble::ONE.div_dd(DoubleDouble::from_f64(3.0));
        let whole = third.mul_f64(3.0);
        let residual = libm::fabs(whole.sub_dd(DoubleDouble::ONE).to_f64());
        assert!(residual < 1e-30, "1/3*3 residual={residual:e}");
    }

    #[test]
    fn sqrt_two_squared_is_two() {
        let two = DoubleDouble::from_f64(2.0);
        let root = two.sqrt();
        // The high lane must agree with the plain-f64 square root.
        assert_eq!(root.to_f64(), libm::sqrt(2.0));
        // The defining property sqrt(2)^2 == 2 must hold to full dd precision,
        // which plain f64 (residual ~1e-16) cannot achieve.
        let squared = root.sqr();
        let residual = libm::fabs(squared.sub_dd(two).to_f64());
        assert!(residual < 1e-30, "sqrt(2)^2 residual={residual:e}");
    }

    #[test]
    fn sqrt_zero_and_negative() {
        assert!(DoubleDouble::ZERO.sqrt().is_zero());
        assert!(DoubleDouble::from_f64(-1.0).sqrt().is_nan());
    }

    #[test]
    fn ordering_uses_both_lanes() {
        let a = DoubleDouble::from_f64(1e16).add_f64(1.0);
        let b = DoubleDouble::from_f64(1e16).add_f64(2.0);
        assert!(a < b, "low lane must break the tie when his are equal");
        assert_eq!(a, a);
        assert!(DoubleDouble::from_f64(-1.0) < DoubleDouble::ZERO);
    }

    #[test]
    fn add_is_commutative_and_exact_for_small_ints() {
        let a = DoubleDouble::from(5);
        let b = DoubleDouble::from(7);
        assert_eq!((a + b).to_f64(), 12.0);
        assert_eq!((a + b), (b + a));
        assert_eq!((a - b).to_f64(), -2.0);
    }
}
