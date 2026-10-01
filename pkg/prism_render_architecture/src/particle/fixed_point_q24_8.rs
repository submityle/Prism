//! Signed `Q24.8` fixed-point arithmetic: a deterministic, pure-integer number
//! type for particle simulation state that must agree bit-for-bit between the
//! `CPU` reference path and a future `GPU` kernel.
//!
//! # Representation
//!
//! A [`Q24_8`] wraps a single [`i32`] (`raw`) whose top 24 bits hold the signed
//! integer part and whose low 8 bits hold the fractional part. The stored
//! integer `r` represents the real value `r / 256`. The scale factor is
//! `2^8 = 256` ([`SCALE`]) and the fractional bit count is [`SCALE_BITS`]
//! (`8`), so one least-significant bit (`LSB`) is `1 / 256 = 0.00390625`.
//!
//! # Why fixed point
//!
//! Floating point rounds differently across compilers, `SIMD` widths, and the
//! `CPU`/`GPU` divide, which breaks lock-step determinism. `Q24.8` is a plain
//! two's-complement [`i32`] under the hood, so every add, subtract, multiply,
//! and shift is exact and portable. No floating point appears anywhere in this
//! module and no transcendental function is used.
//!
//! # API shape
//!
//! Operations are exposed as free functions (`add`, `mul`, …) rather than
//! inherent `add`/`sub`/`mul`/`neg` methods or the standard arithmetic traits,
//! so callers pick the exact overflow policy at each call site and the type
//! never silently shadows the standard operators.
//!
//! # Rounding
//!
//! All truncating conversions and shifts use arithmetic right shift, which
//! rounds toward negative infinity (`floor`), **not** toward zero. For example
//! [`to_int_trunc`] of `-2.5` yields `-3`, matching [`floor`]. Keep this in
//! mind when porting code that assumes C-style truncation toward zero.

/// A signed `Q24.8` fixed-point number backed by one [`i32`].
///
/// The wrapped integer [`raw`](Self::raw) represents the real value
/// `raw / 256`. See the [module documentation](self) for the full contract.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Q24_8 {
    /// The raw two's-complement integer value; the real value is `raw / 256`.
    pub raw: i32,
}

/// Number of fractional bits in the representation (`8`).
pub const SCALE_BITS: i32 = 8;

/// The fixed-point scale factor `2^8 = 256`.
pub const SCALE: i32 = 1 << SCALE_BITS;

/// The constant `0.0` in `Q24.8` (`raw == 0`).
pub const ZERO: Q24_8 = Q24_8 { raw: 0 };

/// The constant `1.0` in `Q24.8` (`raw == 256`).
pub const ONE: Q24_8 = Q24_8 { raw: SCALE };

/// The constant `0.5` in `Q24.8` (`raw == 128`).
pub const HALF: Q24_8 = Q24_8 { raw: SCALE >> 1 };

// Compile-time enshrinement of the hard references for the core constants.
const _: () = {
    const { assert!(SCALE == 256) };
    const { assert!(SCALE_BITS == 8) };
    const { assert!(ZERO.raw == 0) };
    const { assert!(ONE.raw == 256) };
    const { assert!(HALF.raw == 128) };
};

/// Builds a `Q24.8` from an integer, i.e. `i.0` with zero fractional part.
///
/// Computes `raw = i * 256`. The caller must keep `i` inside the representable
/// integer range `[-8388608, 8388607]`; values outside that range overflow.
pub fn from_int(i: i32) -> Q24_8 {
    Q24_8 { raw: i * SCALE }
}

/// Truncates a `Q24.8` to an [`i32`], rounding toward negative infinity.
///
/// This is an arithmetic right shift by [`SCALE_BITS`], so the result is the
/// mathematical `floor` of the real value, **not** truncation toward zero. For
/// instance `to_int_trunc` of `-2.5` is `-3`.
pub fn to_int_trunc(x: Q24_8) -> i32 {
    x.raw >> SCALE_BITS
}

/// Builds a `Q24.8` approximating the ratio `num / den`.
///
/// Computes `((num as i64) << 8) / (den as i64)` and narrows back to [`i32`].
/// The division truncates toward zero (standard integer division semantics on
/// the widened value), so `from_ratio(1, 3)` yields `raw == 85`.
///
/// Panics if `den == 0`.
pub fn from_ratio(num: i32, den: i32) -> Q24_8 {
    let scaled = (num as i64) << SCALE_BITS;
    Q24_8 {
        raw: (scaled / (den as i64)) as i32,
    }
}

/// Adds two `Q24.8` values by adding their raw integers.
///
/// Overflow wraps in two's complement, matching `GPU` integer hardware. Use
/// [`saturating_add`] for clamping behaviour instead.
pub fn add(a: Q24_8, b: Q24_8) -> Q24_8 {
    Q24_8 {
        raw: a.raw.wrapping_add(b.raw),
    }
}

/// Adds two `Q24.8` values, clamping to the representable [`i32`] extremes.
pub fn saturating_add(a: Q24_8, b: Q24_8) -> Q24_8 {
    Q24_8 {
        raw: a.raw.saturating_add(b.raw),
    }
}

/// Subtracts `b` from `a` by subtracting their raw integers (wrapping).
pub fn sub(a: Q24_8, b: Q24_8) -> Q24_8 {
    Q24_8 {
        raw: a.raw.wrapping_sub(b.raw),
    }
}

/// Subtracts `b` from `a`, clamping to the representable [`i32`] extremes.
pub fn saturating_sub(a: Q24_8, b: Q24_8) -> Q24_8 {
    Q24_8 {
        raw: a.raw.saturating_sub(b.raw),
    }
}

/// Negates a `Q24.8` value (wrapping at [`i32::MIN`]).
pub fn neg(a: Q24_8) -> Q24_8 {
    Q24_8 {
        raw: a.raw.wrapping_neg(),
    }
}

/// Multiplies two `Q24.8` values using an [`i64`] intermediate.
///
/// Computes `((a.raw as i64) * (b.raw as i64)) >> 8` and narrows back to
/// [`i32`]. The arithmetic right shift rounds the product toward negative
/// infinity.
pub fn mul(a: Q24_8, b: Q24_8) -> Q24_8 {
    let product = (a.raw as i64) * (b.raw as i64);
    Q24_8 {
        raw: (product >> SCALE_BITS) as i32,
    }
}

/// Divides `a` by `b` using an [`i64`] intermediate.
///
/// Computes `((a.raw as i64) << 8) / (b.raw as i64)` and narrows back to
/// [`i32`]. The division truncates toward zero on the widened value.
///
/// Panics if `b.raw == 0`.
pub fn div(a: Q24_8, b: Q24_8) -> Q24_8 {
    let scaled = (a.raw as i64) << SCALE_BITS;
    Q24_8 {
        raw: (scaled / (b.raw as i64)) as i32,
    }
}

/// Rounds a `Q24.8` down to the nearest whole integer (toward negative
/// infinity), returning the result still in `Q24.8`.
///
/// Clears the low [`SCALE_BITS`] fractional bits via `raw & !0xFF`, which is
/// exact `floor` for both positive and negative values.
pub fn floor(x: Q24_8) -> Q24_8 {
    Q24_8 {
        raw: x.raw & !(SCALE - 1),
    }
}

/// Returns the fractional part of a `Q24.8` value in the range `[0, 1)`.
///
/// Computes `raw & 0xFF`, which is `x - floor(x)` and therefore always a
/// non-negative fraction, consistent with [`floor`]'s round-toward-negative-
/// infinity semantics.
pub fn fract(x: Q24_8) -> Q24_8 {
    Q24_8 {
        raw: x.raw & (SCALE - 1),
    }
}

/// Returns the absolute value of a `Q24.8` number (wrapping at [`i32::MIN`]).
pub fn abs(x: Q24_8) -> Q24_8 {
    Q24_8 {
        raw: x.raw.wrapping_abs(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Core constants / hard references ---------------------------------

    #[test]
    fn constants_have_expected_raw() {
        assert_eq!(ZERO.raw, 0);
        assert_eq!(ONE.raw, 256);
        assert_eq!(HALF.raw, 128);
        assert_eq!(SCALE, 256);
        assert_eq!(SCALE_BITS, 8);
    }

    #[test]
    fn from_int_one() {
        assert_eq!(from_int(1).raw, 256);
    }

    #[test]
    fn from_int_negative_one() {
        assert_eq!(from_int(-1).raw, -256);
    }

    #[test]
    fn from_int_zero() {
        assert_eq!(from_int(0).raw, 0);
    }

    #[test]
    fn from_int_five_roundtrip() {
        assert_eq!(to_int_trunc(from_int(5)), 5);
    }

    #[test]
    fn to_int_trunc_positive() {
        assert_eq!(to_int_trunc(from_int(123)), 123);
    }

    // --- Multiplication ----------------------------------------------------

    #[test]
    fn mul_three_by_four_raw() {
        assert_eq!(mul(from_int(3), from_int(4)).raw, 3072);
    }

    #[test]
    fn mul_three_by_four_to_int() {
        assert_eq!(to_int_trunc(mul(from_int(3), from_int(4))), 12);
    }

    #[test]
    fn mul_half_by_half_is_quarter() {
        assert_eq!(mul(HALF, HALF).raw, 64);
    }

    #[test]
    fn mul_identity_leaves_value_unchanged() {
        let v = from_ratio(7, 3);
        assert_eq!(mul(v, ONE).raw, v.raw);
        assert_eq!(mul(ONE, v).raw, v.raw);
    }

    #[test]
    fn mul_by_zero_is_zero() {
        assert_eq!(mul(from_int(42), ZERO).raw, 0);
    }

    #[test]
    fn mul_is_commutative() {
        let a = from_ratio(5, 4);
        let b = from_ratio(9, 7);
        assert_eq!(mul(a, b).raw, mul(b, a).raw);
    }

    #[test]
    fn mul_negative() {
        assert_eq!(mul(from_int(-3), from_int(4)).raw, -3072);
    }

    // --- Addition / subtraction -------------------------------------------

    #[test]
    fn add_one_plus_one_raw() {
        assert_eq!(add(ONE, ONE).raw, 512);
    }

    #[test]
    fn add_is_commutative() {
        let a = from_int(3);
        let b = from_ratio(1, 2);
        assert_eq!(add(a, b).raw, add(b, a).raw);
    }

    #[test]
    fn add_is_associative() {
        let a = from_int(2);
        let b = from_ratio(1, 2);
        let c = from_ratio(1, 4);
        assert_eq!(add(add(a, b), c).raw, add(a, add(b, c)).raw);
    }

    #[test]
    fn add_zero_identity() {
        let v = from_ratio(13, 5);
        assert_eq!(add(v, ZERO).raw, v.raw);
    }

    #[test]
    fn sub_basic() {
        assert_eq!(sub(from_int(5), from_int(2)).raw, 768);
    }

    #[test]
    fn sub_to_negative() {
        assert_eq!(sub(from_int(2), from_int(5)).raw, -768);
    }

    #[test]
    fn add_then_sub_roundtrip() {
        let a = from_ratio(7, 4);
        let b = from_ratio(3, 8);
        assert_eq!(sub(add(a, b), b).raw, a.raw);
    }

    #[test]
    fn saturating_add_clamps_high() {
        let big = Q24_8 { raw: i32::MAX };
        assert_eq!(saturating_add(big, ONE).raw, i32::MAX);
    }

    #[test]
    fn saturating_sub_clamps_low() {
        let small = Q24_8 { raw: i32::MIN };
        assert_eq!(saturating_sub(small, ONE).raw, i32::MIN);
    }

    // --- Negation ----------------------------------------------------------

    #[test]
    fn neg_three_raw() {
        assert_eq!(neg(from_int(3)).raw, -768);
    }

    #[test]
    fn neg_zero_is_zero() {
        assert_eq!(neg(ZERO).raw, 0);
    }

    #[test]
    fn neg_twice_is_identity() {
        let v = from_ratio(11, 3);
        assert_eq!(neg(neg(v)).raw, v.raw);
    }

    // --- Division ----------------------------------------------------------

    #[test]
    fn div_one_by_two_is_half() {
        assert_eq!(div(ONE, from_int(2)).raw, 128);
    }

    #[test]
    fn div_mul_inverse_exact() {
        let a = from_int(12);
        let b = from_int(3);
        assert_eq!(div(mul(a, b), b).raw, a.raw);
    }

    #[test]
    fn div_by_one_identity() {
        let v = from_ratio(17, 4);
        assert_eq!(div(v, ONE).raw, v.raw);
    }

    #[test]
    fn div_negative() {
        assert_eq!(div(from_int(-1), from_int(2)).raw, -128);
    }

    #[test]
    #[should_panic]
    fn div_by_zero_panics() {
        let _ = div(ONE, ZERO);
    }

    // --- from_ratio truncation --------------------------------------------

    #[test]
    fn from_ratio_one_half() {
        assert_eq!(from_ratio(1, 2).raw, 128);
    }

    #[test]
    fn from_ratio_one_quarter() {
        assert_eq!(from_ratio(1, 4).raw, 64);
    }

    #[test]
    fn from_ratio_one_third_truncates() {
        assert_eq!(from_ratio(1, 3).raw, 85);
    }

    #[test]
    #[should_panic]
    fn from_ratio_zero_denominator_panics() {
        let _ = from_ratio(1, 0);
    }

    // --- floor / trunc / fract --------------------------------------------

    #[test]
    fn floor_two_and_half() {
        assert_eq!(floor(from_ratio(5, 2)).raw, 512);
    }

    #[test]
    fn floor_negative_two_and_half() {
        assert_eq!(floor(from_ratio(-5, 2)).raw, -768);
    }

    #[test]
    fn to_int_trunc_negative_is_floor() {
        assert_eq!(to_int_trunc(from_ratio(-5, 2)), -3);
    }

    #[test]
    fn floor_of_integer_is_unchanged() {
        assert_eq!(floor(from_int(4)).raw, 1024);
    }

    #[test]
    fn fract_positive_in_range() {
        let f = fract(from_ratio(5, 2));
        assert_eq!(f.raw, 128);
        assert!(f.raw >= 0 && f.raw < SCALE);
    }

    #[test]
    fn fract_negative_in_range() {
        let f = fract(from_ratio(-5, 2));
        assert_eq!(f.raw, 128);
        assert!(f.raw >= 0 && f.raw < SCALE);
    }

    #[test]
    fn floor_plus_fract_reconstructs() {
        let x = from_ratio(-5, 2);
        assert_eq!(add(floor(x), fract(x)).raw, x.raw);
    }

    // --- abs ---------------------------------------------------------------

    #[test]
    fn abs_of_negative_seven() {
        assert_eq!(abs(neg(from_int(7))).raw, 1792);
    }

    #[test]
    fn abs_of_positive_unchanged() {
        assert_eq!(abs(from_int(7)).raw, 1792);
    }

    #[test]
    fn abs_of_zero_is_zero() {
        assert_eq!(abs(ZERO).raw, 0);
    }

    // --- boundary values ---------------------------------------------------

    #[test]
    fn half_raw_is_128() {
        assert_eq!(HALF.raw, 128);
    }

    #[test]
    fn one_minus_half_is_half() {
        assert_eq!(sub(ONE, HALF).raw, HALF.raw);
    }

    #[test]
    fn half_plus_half_is_one() {
        assert_eq!(add(HALF, HALF).raw, ONE.raw);
    }

    #[test]
    fn type_is_copy_and_eq() {
        let a = from_int(3);
        let b = a;
        assert_eq!(a, b);
    }
}
