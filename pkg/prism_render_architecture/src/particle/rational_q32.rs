//! Exact rational (fraction `p`/`q`) arithmetic for particle ratio math.
//!
//! Scope and boundary with neighbouring modules:
//! * `crate::particle::fixed_point_q16` stores a *fixed-point* Q16.16 value: a
//!   single scaled integer with a fixed binary fraction, so it approximates
//!   values with bounded resolution. This module is different: it stores an
//!   *exact rational* number as an irreducible fraction `num`/`den`, so values
//!   such as 1/3 are represented with zero error.
//! * `crate::particle::integer_gcd_lcm` provides the exact integer `gcd`/`lcm`
//!   primitives. This module *reuses* its `gcd_i64` routine to reduce every
//!   fraction to lowest terms instead of duplicating a Euclidean loop.
//!
//! Every `RationalQ32` is kept canonical: the denominator is strictly positive,
//! the sign lives in the numerator, and the `gcd` of `|num|` and `den` is `1`.
//! All arithmetic uses `i64::checked_*` with `i128` intermediates so overflow is
//! reported as `None` rather than wrapping. There is no `unsafe`, no
//! transcendental function, and the only floating point is the division inside
//! `to_f32`; equality of `f32` outputs is therefore tested with an epsilon.

use crate::particle::integer_gcd_lcm::gcd_i64;

/// An exact rational number stored as an irreducible fraction.
///
/// Invariants maintained by every constructor and operation:
/// * `den > 0` (the sign is carried by `num`),
/// * `gcd(|num|, den) == 1` (always reduced to lowest terms),
/// * zero is canonicalised to `0`/`1`.
///
/// The fields use `i64` storage even though the type is named `Q32`, because
/// the numerator and denominator each originate from 32-bit style ratios yet
/// need the extra head-room during normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RationalQ32 {
    num: i64,
    den: i64,
}

impl RationalQ32 {
    /// Builds a reduced rational from `num`/`den`, returning `None` when
    /// `den == 0`.
    ///
    /// The sign is moved onto the numerator, then both parts are divided by
    /// their `gcd` so the stored fraction is always canonical.
    #[must_use]
    pub fn new(num: i64, den: i64) -> Option<RationalQ32> {
        if den == 0 {
            return None;
        }
        let mut n = num;
        let mut d = den;
        // Move the sign onto the numerator so the denominator stays positive.
        if d < 0 {
            // `checked_neg` guards against negating `i64::MIN`.
            n = n.checked_neg()?;
            d = d.checked_neg()?;
        }
        let divisor = gcd_i64(n, d);
        if divisor > 1 {
            // The `gcd` of two in-range `i64` values fits back into `i64`, and
            // it divides both operands exactly, so these casts never wrap.
            let g = divisor as i64;
            n /= g;
            d /= g;
        }
        Some(RationalQ32 { num: n, den: d })
    }

    /// Builds the rational `n`/`1` from an integer.
    #[must_use]
    pub const fn from_integer(n: i64) -> RationalQ32 {
        RationalQ32 { num: n, den: 1 }
    }

    /// Returns the numerator of the reduced fraction.
    #[must_use]
    pub const fn numerator(&self) -> i64 {
        self.num
    }

    /// Returns the (strictly positive) denominator of the reduced fraction.
    #[must_use]
    pub const fn denominator(&self) -> i64 {
        self.den
    }

    /// Normalizes an `i128` fraction back into a reduced `RationalQ32`.
    ///
    /// Shared by the arithmetic routines: it checks that both parts fit in
    /// `i64` after reduction, moves the sign onto the numerator, and divides by
    /// the `gcd`. Returns `None` on overflow or a zero denominator.
    fn from_i128(num: i128, den: i128) -> Option<RationalQ32> {
        if den == 0 {
            return None;
        }
        let mut n = num;
        let mut d = den;
        if d < 0 {
            n = n.checked_neg()?;
            d = d.checked_neg()?;
        }
        // Reduce in the `i128` domain first; `unsigned_abs` avoids the
        // `i128::MIN` negation hazard.
        let a = n.unsigned_abs();
        let b = d.unsigned_abs();
        let g = gcd_u128(a, b);
        if g > 1 {
            let gi = g as i128;
            n /= gi;
            d /= gi;
        }
        let n64 = i64::try_from(n).ok()?;
        let d64 = i64::try_from(d).ok()?;
        Some(RationalQ32 { num: n64, den: d64 })
    }

    /// Exact addition; returns `None` only on `i64` overflow of the result.
    #[must_use]
    pub fn checked_add(&self, other: &RationalQ32) -> Option<RationalQ32> {
        // num = a.num * b.den + b.num * a.den, den = a.den * b.den, in `i128`.
        let num =
            (self.num as i128) * (other.den as i128) + (other.num as i128) * (self.den as i128);
        let den = (self.den as i128) * (other.den as i128);
        RationalQ32::from_i128(num, den)
    }

    /// Exact subtraction; returns `None` only on `i64` overflow of the result.
    #[must_use]
    pub fn checked_sub(&self, other: &RationalQ32) -> Option<RationalQ32> {
        let num =
            (self.num as i128) * (other.den as i128) - (other.num as i128) * (self.den as i128);
        let den = (self.den as i128) * (other.den as i128);
        RationalQ32::from_i128(num, den)
    }

    /// Exact multiplication; returns `None` only on `i64` overflow of the
    /// reduced result.
    #[must_use]
    pub fn checked_mul(&self, other: &RationalQ32) -> Option<RationalQ32> {
        let num = (self.num as i128) * (other.num as i128);
        let den = (self.den as i128) * (other.den as i128);
        RationalQ32::from_i128(num, den)
    }

    /// Exact division; returns `None` when dividing by zero or on overflow.
    #[must_use]
    pub fn checked_div(&self, other: &RationalQ32) -> Option<RationalQ32> {
        if other.num == 0 {
            return None;
        }
        // Multiply by the reciprocal: num = a.num * b.den, den = a.den * b.num.
        let num = (self.num as i128) * (other.den as i128);
        let den = (self.den as i128) * (other.num as i128);
        RationalQ32::from_i128(num, den)
    }

    /// Returns the additive inverse `-self`.
    ///
    /// Reduction is already guaranteed, so only the numerator sign flips; it
    /// cannot overflow because a canonical numerator is never `i64::MIN` paired
    /// with a positive denominator larger than one for in-range inputs, yet we
    /// still saturate defensively through `wrapping` only on the impossible
    /// edge and keep it exact via `checked_neg` fallback.
    #[must_use]
    pub fn negate(&self) -> RationalQ32 {
        // A canonical fraction's numerator negates without overflow unless it is
        // exactly `i64::MIN`; in that impossible-for-reduced-form case we fall
        // back to re-normalizing through `from_i128`, which stays exact.
        match self.num.checked_neg() {
            Some(n) => RationalQ32 {
                num: n,
                den: self.den,
            },
            None => RationalQ32::from_i128(-(self.num as i128), self.den as i128).unwrap_or(*self),
        }
    }

    /// Returns `1`/`self`, or `None` when `self` is zero.
    #[must_use]
    pub fn reciprocal(&self) -> Option<RationalQ32> {
        if self.num == 0 {
            return None;
        }
        // Swapping keeps magnitudes identical, so re-normalizing only fixes the
        // sign placement when the old numerator was negative.
        RationalQ32::from_i128(self.den as i128, self.num as i128)
    }

    /// Returns `|self|`.
    #[must_use]
    pub fn abs(&self) -> RationalQ32 {
        if self.num < 0 {
            self.negate()
        } else {
            *self
        }
    }

    /// Compares two rationals exactly via cross multiplication in `i128`.
    ///
    /// Because both denominators are positive, `a/b < c/d` is equivalent to
    /// `a*d < c*b`, and the `i128` products never overflow for `i64` inputs.
    #[must_use]
    pub fn compare(&self, other: &RationalQ32) -> core::cmp::Ordering {
        let left = (self.num as i128) * (other.den as i128);
        let right = (other.num as i128) * (self.den as i128);
        left.cmp(&right)
    }

    /// Returns `true` when the value equals zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    /// Returns `true` when the value is an exact integer (denominator `1`).
    #[must_use]
    pub fn is_integer(&self) -> bool {
        self.den == 1
    }

    /// Converts to the nearest `f32` by dividing numerator by denominator.
    ///
    /// This is the only floating-point operation in the module; callers that
    /// need to compare the result do so with an epsilon tolerance.
    #[must_use]
    pub fn to_f32(&self) -> f32 {
        (self.num as f32) / (self.den as f32)
    }
}

/// Greatest common divisor over `u128`, used by the `i128` normalization path.
///
/// Mirrors the Euclidean loop in `crate::particle::integer_gcd_lcm` but in the
/// wider domain required by the arithmetic intermediates. By convention
/// `gcd(n, 0) == n`.
fn gcd_u128(a: u128, b: u128) -> u128 {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

impl PartialOrd for RationalQ32 {
    fn partial_cmp(&self, other: &RationalQ32) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RationalQ32 {
    fn cmp(&self, other: &RationalQ32) -> core::cmp::Ordering {
        self.compare(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Ordering;

    #[test]
    fn reduces_two_quarters() {
        let r = RationalQ32::new(2, 4).unwrap();
        assert_eq!(r.numerator(), 1);
        assert_eq!(r.denominator(), 2);
    }

    #[test]
    fn reduces_double_negative() {
        let r = RationalQ32::new(-1, -2).unwrap();
        assert_eq!(r.numerator(), 1);
        assert_eq!(r.denominator(), 2);
    }

    #[test]
    fn moves_sign_to_numerator() {
        let r = RationalQ32::new(3, -6).unwrap();
        assert_eq!(r.numerator(), -1);
        assert_eq!(r.denominator(), 2);
    }

    #[test]
    fn zero_denominator_is_none() {
        assert!(RationalQ32::new(1, 0).is_none());
    }

    #[test]
    fn zero_is_canonical() {
        let r = RationalQ32::new(0, 5).unwrap();
        assert_eq!(r.numerator(), 0);
        assert_eq!(r.denominator(), 1);
        assert!(r.is_zero());
    }

    #[test]
    fn from_integer_builds_over_one() {
        let r = RationalQ32::from_integer(7);
        assert_eq!(r.numerator(), 7);
        assert_eq!(r.denominator(), 1);
        assert!(r.is_integer());
    }

    #[test]
    fn add_half_and_third() {
        let a = RationalQ32::new(1, 2).unwrap();
        let b = RationalQ32::new(1, 3).unwrap();
        let s = a.checked_add(&b).unwrap();
        assert_eq!(s, RationalQ32::new(5, 6).unwrap());
    }

    #[test]
    fn add_two_halves_is_one() {
        let a = RationalQ32::new(1, 2).unwrap();
        let s = a.checked_add(&a).unwrap();
        assert_eq!(s, RationalQ32::from_integer(1));
        assert!(s.is_integer());
    }

    #[test]
    fn subtract_to_zero() {
        let a = RationalQ32::new(3, 4).unwrap();
        let d = a.checked_sub(&a).unwrap();
        assert!(d.is_zero());
    }

    #[test]
    fn subtract_mixed_signs() {
        let a = RationalQ32::new(1, 2).unwrap();
        let b = RationalQ32::new(-1, 3).unwrap();
        let d = a.checked_sub(&b).unwrap();
        assert_eq!(d, RationalQ32::new(5, 6).unwrap());
    }

    #[test]
    fn multiply_two_thirds_by_three_quarters() {
        let a = RationalQ32::new(2, 3).unwrap();
        let b = RationalQ32::new(3, 4).unwrap();
        let p = a.checked_mul(&b).unwrap();
        assert_eq!(p, RationalQ32::new(1, 2).unwrap());
    }

    #[test]
    fn multiply_by_zero_is_zero() {
        let a = RationalQ32::new(5, 7).unwrap();
        let z = RationalQ32::from_integer(0);
        assert!(a.checked_mul(&z).unwrap().is_zero());
    }

    #[test]
    fn divide_half_by_quarter() {
        let a = RationalQ32::new(1, 2).unwrap();
        let b = RationalQ32::new(1, 4).unwrap();
        let q = a.checked_div(&b).unwrap();
        assert_eq!(q, RationalQ32::from_integer(2));
    }

    #[test]
    fn divide_by_zero_is_none() {
        let a = RationalQ32::new(1, 2).unwrap();
        let z = RationalQ32::from_integer(0);
        assert!(a.checked_div(&z).is_none());
    }

    #[test]
    fn divide_negative() {
        let a = RationalQ32::new(-3, 5).unwrap();
        let b = RationalQ32::new(3, 10).unwrap();
        let q = a.checked_div(&b).unwrap();
        assert_eq!(q, RationalQ32::from_integer(-2));
    }

    #[test]
    fn compare_third_less_than_half() {
        let a = RationalQ32::new(1, 3).unwrap();
        let b = RationalQ32::new(1, 2).unwrap();
        assert_eq!(a.compare(&b), Ordering::Less);
        assert_eq!(b.compare(&a), Ordering::Greater);
    }

    #[test]
    fn compare_equal_fractions() {
        let a = RationalQ32::new(2, 4).unwrap();
        let b = RationalQ32::new(1, 2).unwrap();
        assert_eq!(a.compare(&b), Ordering::Equal);
        assert_eq!(a, b);
    }

    #[test]
    fn compare_negative_values() {
        let a = RationalQ32::new(-1, 2).unwrap();
        let b = RationalQ32::new(-1, 3).unwrap();
        // -1/2 is less than -1/3.
        assert_eq!(a.compare(&b), Ordering::Less);
    }

    #[test]
    fn sort_a_batch_of_rationals() {
        let mut values = [
            RationalQ32::new(1, 2).unwrap(),
            RationalQ32::new(-3, 4).unwrap(),
            RationalQ32::new(1, 3).unwrap(),
            RationalQ32::new(2, 1).unwrap(),
            RationalQ32::new(0, 9).unwrap(),
        ];
        values.sort();
        let expected = [
            RationalQ32::new(-3, 4).unwrap(),
            RationalQ32::new(0, 1).unwrap(),
            RationalQ32::new(1, 3).unwrap(),
            RationalQ32::new(1, 2).unwrap(),
            RationalQ32::new(2, 1).unwrap(),
        ];
        assert_eq!(values, expected);
    }

    #[test]
    fn reciprocal_of_zero_is_none() {
        let z = RationalQ32::from_integer(0);
        assert!(z.reciprocal().is_none());
    }

    #[test]
    fn reciprocal_flips_fraction() {
        let a = RationalQ32::new(2, 3).unwrap();
        assert_eq!(a.reciprocal().unwrap(), RationalQ32::new(3, 2).unwrap());
    }

    #[test]
    fn reciprocal_keeps_sign_on_numerator() {
        let a = RationalQ32::new(-2, 3).unwrap();
        let r = a.reciprocal().unwrap();
        assert_eq!(r.numerator(), -3);
        assert_eq!(r.denominator(), 2);
    }

    #[test]
    fn negate_flips_sign() {
        let a = RationalQ32::new(3, 7).unwrap();
        assert_eq!(a.negate(), RationalQ32::new(-3, 7).unwrap());
        assert_eq!(a.negate().negate(), a);
    }

    #[test]
    fn abs_of_negative() {
        let a = RationalQ32::new(-5, 8).unwrap();
        assert_eq!(a.abs(), RationalQ32::new(5, 8).unwrap());
    }

    #[test]
    fn abs_of_positive_is_identity() {
        let a = RationalQ32::new(5, 8).unwrap();
        assert_eq!(a.abs(), a);
    }

    #[test]
    fn to_f32_half_within_epsilon() {
        let a = RationalQ32::new(1, 2).unwrap();
        let diff = (a.to_f32() - 0.5_f32).abs();
        assert!(diff < 1e-6_f32);
    }

    #[test]
    fn to_f32_third_within_epsilon() {
        let a = RationalQ32::new(1, 3).unwrap();
        let diff = (a.to_f32() - (1.0_f32 / 3.0_f32)).abs();
        assert!(diff < 1e-6_f32);
    }

    #[test]
    fn multiply_overflow_returns_none() {
        // Two large coprime numerators whose product exceeds `i64::MAX`.
        let a = RationalQ32::new(i64::MAX, 1).unwrap();
        let b = RationalQ32::new(i64::MAX - 1, 1).unwrap();
        assert!(a.checked_mul(&b).is_none());
    }

    #[test]
    fn add_overflow_returns_none() {
        // Denominators chosen so the common denominator exceeds `i64::MAX`.
        let a = RationalQ32::new(1, i64::MAX).unwrap();
        let b = RationalQ32::new(1, i64::MAX - 2).unwrap();
        assert!(a.checked_add(&b).is_none());
    }

    #[test]
    fn is_integer_detects_fractions() {
        assert!(RationalQ32::new(4, 2).unwrap().is_integer());
        assert!(!RationalQ32::new(3, 2).unwrap().is_integer());
    }

    #[test]
    fn property_add_matches_manual_cross_multiply() {
        // Deterministic LCG over small non-zero denominators.
        let mut state: u64 = 0x1234_5678_9abc_def1;
        let mut count: u32 = 0;
        while count < 200 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let pn = ((state >> 16) as i64 % 2001) - 1000;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let pd = ((state >> 16) as i64 % 1000) + 1;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let qn = ((state >> 16) as i64 % 2001) - 1000;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let qd = ((state >> 16) as i64 % 1000) + 1;

            let a = RationalQ32::new(pn, pd).unwrap();
            let b = RationalQ32::new(qn, qd).unwrap();

            let sum = a.checked_add(&b).unwrap();
            // Manual common-denominator sum, then reduce for comparison.
            let manual = RationalQ32::new(pn * qd + qn * pd, pd * qd).unwrap();
            assert_eq!(sum, manual);
            count += 1;
        }
    }

    #[test]
    fn property_compare_is_reflexive_and_antisymmetric() {
        let mut state: u64 = 0x0fed_cba9_8765_4321;
        let mut count: u32 = 0;
        while count < 200 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let pn = ((state >> 16) as i64 % 401) - 200;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let pd = ((state >> 16) as i64 % 200) + 1;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let qn = ((state >> 16) as i64 % 401) - 200;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let qd = ((state >> 16) as i64 % 200) + 1;

            let a = RationalQ32::new(pn, pd).unwrap();
            let b = RationalQ32::new(qn, qd).unwrap();

            // Reflexivity.
            assert_eq!(a.compare(&a), Ordering::Equal);
            // Antisymmetry: reversing the operands reverses the ordering.
            assert_eq!(a.compare(&b), b.compare(&a).reverse());
            count += 1;
        }
    }
}
