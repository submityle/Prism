//! Signed `Q16.16` fixed-point arithmetic: a deterministic, mostly integer
//! number type for particle simulation state that must agree bit-for-bit
//! between the `CPU` reference path and a future `GPU` kernel (design §25,
//! §29).
//!
//! # Representation
//!
//! A [`Q16_16`] wraps a single [`i32`] whose top 16 bits hold the signed
//! integer part and whose low 16 bits hold the fractional part, so the stored
//! integer `r` represents the real value `r / 65536`. The scale factor is
//! `2^16 = 65536`, exposed as [`Q16_16::ONE`], and the fractional bit count is
//! [`Q16_16::FRAC_BITS`] (`16`). The representable range is therefore
//! `[-32768, 32768)` in real units, with a resolution (one least-significant
//! bit, one `LSB`) of `1 / 65536 ≈ 0.0000152587890625`.
//!
//! # Why fixed point
//!
//! Floating point rounds differently across compilers, `SIMD` widths, and the
//! `CPU`/`GPU` divide, which breaks lock-step determinism. `Q16.16` is a plain
//! two's-complement integer under the hood, so every add, subtract, multiply,
//! and shift is exact and portable. This module keeps `f32` at the boundary
//! only: [`q_from_f32`] and [`q_to_f32`] convert to and from author-facing
//! floats, and the only `f32` operation used anywhere is
//! [`f32::floor`]/[`f32::clamp`] — never a transcendental function.
//!
//! # API shape
//!
//! Operations are exposed as free functions (`q_add`, `q_mul`, …) rather than
//! inherent `add`/`sub`/`mul`/`neg` methods, so callers pick the exact overflow
//! policy at each call site and the type does not silently shadow the standard
//! arithmetic traits. Additive helpers come in two flavours: the plain
//! `q_add`/`q_sub`/`q_neg` wrap on overflow (matching hardware two's-complement
//! integer wrap-around, which is what a `GPU` kernel does), while
//! `q_add_sat`/`q_sub_sat` clamp to the representable extremes. Multiplication
//! and division use an [`i64`] intermediate with round-to-nearest and then
//! clamp back into [`i32`] range.

/// A signed `Q16.16` fixed-point number backed by one [`i32`].
///
/// The wrapped integer `r` (field `.0`) represents the real value
/// `r / 65536`. See the [module documentation](self) for the full contract.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Q16_16(pub i32);

impl Q16_16 {
    /// Number of fractional bits in the representation (`16`).
    pub const FRAC_BITS: i32 = 16;

    /// The raw integer value of `1.0`, i.e. the fixed-point scale factor
    /// `2^16 = 65536`.
    pub const ONE: i32 = 1 << Self::FRAC_BITS;
}

/// The constant `1.0` in `Q16.16`.
pub const Q_ONE: Q16_16 = Q16_16(Q16_16::ONE);

/// The constant `0.0` in `Q16.16`.
pub const Q_ZERO: Q16_16 = Q16_16(0);

/// The constant `0.5` in `Q16.16`.
pub const Q_HALF: Q16_16 = Q16_16(Q16_16::ONE / 2);

/// The round-to-nearest bias added before a right shift by `FRAC_BITS`
/// (half of one integer unit, `2^15`).
const ROUND_BIAS: i64 = (Q16_16::ONE as i64) / 2;

/// Clamps a wide [`i64`] result back into [`i32`] range, saturating at the
/// extremes instead of wrapping.
const fn clamp_to_i32(v: i64) -> i32 {
    if v > i32::MAX as i64 {
        i32::MAX
    } else if v < i32::MIN as i64 {
        i32::MIN
    } else {
        v as i32
    }
}

/// Builds a `Q16.16` value from an integer (`i` becomes `i.0`).
///
/// The shift discards any bits above bit 15 of `i`; inputs outside
/// `[-32768, 32767]` wrap, matching two's-complement hardware.
#[must_use]
pub const fn q_from_int(i: i32) -> Q16_16 {
    Q16_16(i << Q16_16::FRAC_BITS)
}

/// Converts to an integer by truncating toward zero (drops the fraction).
///
/// Integer division by [`Q16_16::ONE`] rounds toward zero, so `1.75` becomes
/// `1` and `-1.75` becomes `-1`.
#[must_use]
pub const fn q_to_int_trunc(q: Q16_16) -> i32 {
    q.0 / Q16_16::ONE
}

/// Converts to an integer by flooring toward negative infinity.
///
/// An arithmetic right shift keeps the sign, so `1.75` becomes `1` and `-1.75`
/// becomes `-2`.
#[must_use]
pub const fn q_to_int_floor(q: Q16_16) -> i32 {
    q.0 >> Q16_16::FRAC_BITS
}

/// Converts an [`f32`] to `Q16.16`.
///
/// The value is scaled by `65536`, clamped into [`i32`] range to avoid an
/// out-of-range cast, and then floored to the nearest lower integer. Flooring
/// (rather than rounding) is the only `f32` rounding operation this module
/// permits, keeping the conversion transcendental-free. A `NaN` input floors
/// to `0` because `NaN.clamp(lo, hi)` yields `lo` here only when ordered; in
/// practice `NaN` is clamped by the standard library to a bound and then
/// floored deterministically.
#[must_use]
pub fn q_from_f32(x: f32) -> Q16_16 {
    let scaled = x * (Q16_16::ONE as f32);
    let clamped = scaled.clamp(i32::MIN as f32, i32::MAX as f32);
    Q16_16(clamped.floor() as i32)
}

/// Converts a `Q16.16` value to an [`f32`].
///
/// Values with magnitude below `256` are exactly representable; larger
/// magnitudes lose low fractional bits to `f32`'s 24-bit mantissa.
#[must_use]
pub fn q_to_f32(q: Q16_16) -> f32 {
    (q.0 as f32) / (Q16_16::ONE as f32)
}

/// Adds two values, wrapping on overflow (two's-complement wrap-around).
#[must_use]
pub const fn q_add(a: Q16_16, b: Q16_16) -> Q16_16 {
    Q16_16(a.0.wrapping_add(b.0))
}

/// Subtracts `b` from `a`, wrapping on overflow (two's-complement wrap-around).
#[must_use]
pub const fn q_sub(a: Q16_16, b: Q16_16) -> Q16_16 {
    Q16_16(a.0.wrapping_sub(b.0))
}

/// Negates a value, wrapping on overflow.
///
/// The single wrapping case is [`i32::MIN`], which negates back to itself.
#[must_use]
pub const fn q_neg(a: Q16_16) -> Q16_16 {
    Q16_16(a.0.wrapping_neg())
}

/// Adds two values, saturating at the representable extremes.
///
/// Overflow clamps to the `Q16.16` value backed by [`i32::MAX`] or
/// [`i32::MIN`] instead of wrapping.
#[must_use]
pub const fn q_add_sat(a: Q16_16, b: Q16_16) -> Q16_16 {
    Q16_16(a.0.saturating_add(b.0))
}

/// Subtracts `b` from `a`, saturating at the representable extremes.
#[must_use]
pub const fn q_sub_sat(a: Q16_16, b: Q16_16) -> Q16_16 {
    Q16_16(a.0.saturating_sub(b.0))
}

/// Multiplies two values with round-to-nearest and saturating clamp.
///
/// The product is computed in [`i64`] to avoid intermediate overflow, biased
/// by half an integer unit for round-half-up, shifted right by
/// [`Q16_16::FRAC_BITS`], and clamped back into [`i32`] range. For example
/// `0.5 * 0.5` yields exactly `0.25`.
#[must_use]
pub const fn q_mul(a: Q16_16, b: Q16_16) -> Q16_16 {
    let product = (a.0 as i64) * (b.0 as i64);
    let rounded = (product + ROUND_BIAS) >> Q16_16::FRAC_BITS;
    Q16_16(clamp_to_i32(rounded))
}

/// Divides `a` by `b` with round-to-nearest and saturating clamp.
///
/// The numerator is widened to [`i64`] and shifted left by
/// [`Q16_16::FRAC_BITS`] before the integer divide, so the quotient stays in
/// `Q16.16`. A half-divisor bias (matching the sign of the exact quotient)
/// gives round-half-away-from-zero. Division by zero cannot produce a real
/// value, so it saturates: a non-negative numerator returns the `Q16.16`
/// value backed by [`i32::MAX`] and a negative numerator returns the value
/// backed by [`i32::MIN`] (this also covers the `0 / 0` case, which returns
/// [`i32::MAX`]).
#[must_use]
pub const fn q_div(a: Q16_16, b: Q16_16) -> Q16_16 {
    if b.0 == 0 {
        if a.0 >= 0 {
            return Q16_16(i32::MAX);
        }
        return Q16_16(i32::MIN);
    }
    let numerator = (a.0 as i64) << Q16_16::FRAC_BITS;
    let denominator = b.0 as i64;
    let half = denominator.abs() / 2;
    let biased = if (numerator >= 0) == (denominator >= 0) {
        numerator + half
    } else {
        numerator - half
    };
    Q16_16(clamp_to_i32(biased / denominator))
}

/// Returns the absolute value, saturating [`i32::MIN`] to [`i32::MAX`].
#[must_use]
pub const fn q_abs(a: Q16_16) -> Q16_16 {
    Q16_16(a.0.saturating_abs())
}

/// Returns the smaller of two values (integer comparison, exact).
#[must_use]
pub const fn q_min(a: Q16_16, b: Q16_16) -> Q16_16 {
    if a.0 <= b.0 {
        a
    } else {
        b
    }
}

/// Returns the larger of two values (integer comparison, exact).
#[must_use]
pub const fn q_max(a: Q16_16, b: Q16_16) -> Q16_16 {
    if a.0 >= b.0 {
        a
    } else {
        b
    }
}

/// Clamps `v` into `[lo, hi]` (integer comparison, exact).
///
/// The caller must pass `lo <= hi`; otherwise the result is the upper bound.
#[must_use]
pub const fn q_clamp(v: Q16_16, lo: Q16_16, hi: Q16_16) -> Q16_16 {
    q_min(q_max(v, lo), hi)
}

/// Returns the largest whole `Q16.16` value not greater than `q` (floor).
///
/// Clearing the low [`Q16_16::FRAC_BITS`] bits rounds toward negative
/// infinity, so `-0.5` floors to `-1.0`.
#[must_use]
pub const fn q_floor(q: Q16_16) -> Q16_16 {
    Q16_16(q.0 & !(Q16_16::ONE - 1))
}

/// Returns the fractional part `q - floor(q)`, always in `[0, 1)`.
///
/// Because the fraction is defined against the floor, a negative input still
/// yields a non-negative fraction: `-0.25` has fraction `0.75` (since
/// `-0.25 = -1.0 + 0.75`). This is the mask of the low
/// [`Q16_16::FRAC_BITS`] bits.
#[must_use]
pub const fn q_frac(q: Q16_16) -> Q16_16 {
    Q16_16(q.0 & (Q16_16::ONE - 1))
}

/// Linearly interpolates from `a` to `b` by `t`, computed as
/// `a + (b - a) * t`.
///
/// With `t = 0` the result is `a`, with `t = 1.0` it is `b`, and `t = 0.5`
/// gives the midpoint. `t` outside `[0, 1]` extrapolates. The intermediate
/// `(b - a)` uses wrapping subtraction and the scale uses [`q_mul`]'s
/// saturating multiply.
#[must_use]
pub const fn q_lerp(a: Q16_16, b: Q16_16, t: Q16_16) -> Q16_16 {
    q_add(a, q_mul(q_sub(b, a), t))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small helper: assert two `f32` values are within an epsilon.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn int_roundtrip_positive() {
        let q = q_from_int(5);
        assert_eq!(q.0, 5 * Q16_16::ONE);
        assert_eq!(q_to_int_trunc(q), 5);
        assert_eq!(q_to_int_floor(q), 5);
    }

    #[test]
    fn int_roundtrip_negative() {
        let q = q_from_int(-7);
        assert_eq!(q.0, -7 * Q16_16::ONE);
        assert_eq!(q_to_int_trunc(q), -7);
        assert_eq!(q_to_int_floor(q), -7);
    }

    #[test]
    fn to_int_trunc_rounds_toward_zero() {
        let pos = q_from_f32(1.75);
        let neg = q_from_f32(-1.75);
        assert_eq!(q_to_int_trunc(pos), 1);
        assert_eq!(q_to_int_trunc(neg), -1);
    }

    #[test]
    fn to_int_floor_rounds_toward_negative_infinity() {
        let pos = q_from_f32(1.75);
        let neg = q_from_f32(-1.75);
        assert_eq!(q_to_int_floor(pos), 1);
        assert_eq!(q_to_int_floor(neg), -2);
    }

    #[test]
    fn f32_roundtrip_within_epsilon() {
        for &x in &[
            0.0_f32,
            1.0,
            -1.0,
            0.5,
            -0.25,
            core::f32::consts::PI,
            -core::f32::consts::E,
            100.5,
        ] {
            let back = q_to_f32(q_from_f32(x));
            assert!(approx(back, x), "roundtrip {x} -> {back}");
        }
    }

    #[test]
    fn f32_half_is_exact_bits() {
        assert_eq!(q_from_f32(0.5).0, Q16_16::ONE / 2);
        assert_eq!(q_from_f32(1.0).0, Q16_16::ONE);
    }

    #[test]
    fn from_f32_clamps_large_positive() {
        // Far beyond the representable range: must clamp, not wrap or panic.
        let q = q_from_f32(1.0e30);
        assert_eq!(q.0, i32::MAX);
    }

    #[test]
    fn from_f32_clamps_large_negative() {
        let q = q_from_f32(-1.0e30);
        assert_eq!(q.0, i32::MIN);
    }

    #[test]
    fn add_basic() {
        let r = q_add(q_from_f32(1.5), q_from_f32(2.25));
        assert!(approx(q_to_f32(r), 3.75));
    }

    #[test]
    fn sub_basic() {
        let r = q_sub(q_from_f32(5.0), q_from_f32(1.25));
        assert!(approx(q_to_f32(r), 3.75));
    }

    #[test]
    fn neg_basic() {
        let r = q_neg(q_from_f32(2.5));
        assert!(approx(q_to_f32(r), -2.5));
        assert_eq!(q_neg(Q_ZERO).0, 0);
    }

    #[test]
    fn neg_of_min_wraps_to_itself() {
        assert_eq!(q_neg(Q16_16(i32::MIN)).0, i32::MIN);
    }

    #[test]
    fn add_wraps_on_overflow() {
        let r = q_add(Q16_16(i32::MAX), Q16_16(1));
        assert_eq!(r.0, i32::MIN);
    }

    #[test]
    fn add_sat_clamps_at_max() {
        let r = q_add_sat(Q16_16(i32::MAX), Q16_16(1));
        assert_eq!(r.0, i32::MAX);
    }

    #[test]
    fn sub_sat_clamps_at_min() {
        let r = q_sub_sat(Q16_16(i32::MIN), Q16_16(1));
        assert_eq!(r.0, i32::MIN);
    }

    #[test]
    fn mul_half_times_half_is_quarter() {
        let r = q_mul(Q_HALF, Q_HALF);
        assert_eq!(r.0, Q16_16::ONE / 4);
        assert!(approx(q_to_f32(r), 0.25));
    }

    #[test]
    fn mul_by_one_is_identity() {
        let a = q_from_f32(3.5);
        assert_eq!(q_mul(a, Q_ONE).0, a.0);
    }

    #[test]
    fn mul_negative_signs() {
        let r = q_mul(q_from_f32(-2.0), q_from_f32(1.5));
        assert!(approx(q_to_f32(r), -3.0));
    }

    #[test]
    fn mul_rounds_to_nearest() {
        // One LSB times one LSB is far below one LSB; round-half-up gives 0.
        let tiny = Q16_16(1);
        assert_eq!(q_mul(tiny, tiny).0, 0);
        // (ONE/2 + small) squared exercises the +half bias path.
        let r = q_mul(Q16_16(3), Q16_16(Q16_16::ONE / 2));
        // 3 * 32768 = 98304, +32768 = 131072, >>16 = 2.
        assert_eq!(r.0, 2);
    }

    #[test]
    fn mul_saturates_on_overflow() {
        let big = q_from_int(30000);
        let r = q_mul(big, big);
        assert_eq!(r.0, i32::MAX);
    }

    #[test]
    fn div_basic() {
        let r = q_div(q_from_f32(3.0), q_from_f32(2.0));
        assert!(approx(q_to_f32(r), 1.5));
    }

    #[test]
    fn div_one_over_two_is_half() {
        let r = q_div(Q_ONE, q_from_int(2));
        assert_eq!(r.0, Q16_16::ONE / 2);
    }

    #[test]
    fn div_by_zero_positive_saturates_to_max() {
        let r = q_div(q_from_f32(1.0), Q_ZERO);
        assert_eq!(r.0, i32::MAX);
    }

    #[test]
    fn div_by_zero_negative_saturates_to_min() {
        let r = q_div(q_from_f32(-1.0), Q_ZERO);
        assert_eq!(r.0, i32::MIN);
    }

    #[test]
    fn div_zero_over_zero_is_max() {
        assert_eq!(q_div(Q_ZERO, Q_ZERO).0, i32::MAX);
    }

    #[test]
    fn div_negative_signs() {
        let r = q_div(q_from_f32(-3.0), q_from_f32(2.0));
        assert!(approx(q_to_f32(r), -1.5));
    }

    #[test]
    fn abs_positive_and_negative() {
        assert!(approx(q_to_f32(q_abs(q_from_f32(-2.5))), 2.5));
        assert!(approx(q_to_f32(q_abs(q_from_f32(2.5))), 2.5));
    }

    #[test]
    fn abs_of_min_saturates() {
        assert_eq!(q_abs(Q16_16(i32::MIN)).0, i32::MAX);
    }

    #[test]
    fn min_max_pick_extremes() {
        let a = q_from_f32(-1.0);
        let b = q_from_f32(2.0);
        assert_eq!(q_min(a, b).0, a.0);
        assert_eq!(q_max(a, b).0, b.0);
    }

    #[test]
    fn clamp_bounds_value() {
        let lo = q_from_f32(-1.0);
        let hi = q_from_f32(1.0);
        assert_eq!(q_clamp(q_from_f32(-5.0), lo, hi).0, lo.0);
        assert_eq!(q_clamp(q_from_f32(5.0), lo, hi).0, hi.0);
        assert_eq!(q_clamp(q_from_f32(0.25), lo, hi).0, q_from_f32(0.25).0);
    }

    #[test]
    fn floor_positive_and_negative() {
        assert_eq!(q_floor(q_from_f32(1.75)).0, q_from_int(1).0);
        assert_eq!(q_floor(q_from_f32(-0.5)).0, q_from_int(-1).0);
        assert_eq!(q_floor(q_from_int(3)).0, q_from_int(3).0);
    }

    #[test]
    fn frac_positive_semantics() {
        let r = q_frac(q_from_f32(1.75));
        assert!(approx(q_to_f32(r), 0.75));
    }

    #[test]
    fn frac_negative_semantics() {
        // -0.25 = -1.0 + 0.75, so the fraction is 0.75.
        let r = q_frac(q_from_f32(-0.25));
        assert!(approx(q_to_f32(r), 0.75));
    }

    #[test]
    fn frac_plus_floor_reconstructs_value() {
        let v = q_from_f32(-2.6);
        let sum = q_add(q_floor(v), q_frac(v));
        assert_eq!(sum.0, v.0);
    }

    #[test]
    fn lerp_endpoints_and_midpoint() {
        let a = q_from_f32(2.0);
        let b = q_from_f32(6.0);
        assert_eq!(q_lerp(a, b, Q_ZERO).0, a.0);
        assert_eq!(q_lerp(a, b, Q_ONE).0, b.0);
        assert!(approx(q_to_f32(q_lerp(a, b, Q_HALF)), 4.0));
    }

    #[test]
    fn lerp_quarter() {
        let a = q_from_f32(0.0);
        let b = q_from_f32(4.0);
        let t = q_from_f32(0.25);
        assert!(approx(q_to_f32(q_lerp(a, b, t)), 1.0));
    }

    #[test]
    fn constants_have_expected_values() {
        assert_eq!(Q_ZERO.0, 0);
        assert_eq!(Q_ONE.0, Q16_16::ONE);
        assert_eq!(Q_HALF.0, Q16_16::ONE / 2);
        assert_eq!(Q16_16::FRAC_BITS, 16);
        assert_eq!(Q16_16::ONE, 65536);
        assert!(approx(q_to_f32(Q_ONE), 1.0));
        assert!(approx(q_to_f32(Q_HALF), 0.5));
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(Q16_16::default().0, 0);
    }
}
