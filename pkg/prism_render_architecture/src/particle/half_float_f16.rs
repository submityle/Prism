//! `IEEE` 754 binary16 (half precision) <-> `f32` conversion, implemented
//! entirely with integer bit manipulation so the result is bit-exact and fully
//! verifiable on the `CPU` reference path before it is mirrored in a `GPU`
//! kernel (design §25, §29).
//!
//! The forward conversion `f32_to_f16_bits` rounds using round-to-nearest-even
//! (`RNE`), the tie-breaking rule mandated by `IEEE` 754. It handles the full
//! range of edge cases: subnormal (denormal) results, gradual underflow to a
//! signed zero, overflow to infinity, `NaN` payloads (preserving the quiet
//! bit), signed zeros, and the exponent-bias change from 127 to 15. All logic
//! operates on the raw `u32` bit pattern returned by `f32::to_bits`; no `f32`
//! arithmetic or comparison is performed.
//!
//! The reverse conversion `f16_bits_to_f32` expands a binary16 bit pattern back
//! into an `f32`, normalizing subnormal mantissas with an integer
//! `leading_zeros` count and reconstructing infinities, `NaN`s, and signed
//! zeros, then returns `f32::from_bits`.

/// Positive infinity in binary16 (sign 0, exponent all ones, mantissa 0).
pub const F16_POSITIVE_INFINITY: u16 = 0x7c00;

/// Negative infinity in binary16 (sign 1, exponent all ones, mantissa 0).
pub const F16_NEGATIVE_INFINITY: u16 = 0xfc00;

/// Positive zero in binary16.
pub const F16_POSITIVE_ZERO: u16 = 0x0000;

/// Negative zero in binary16.
pub const F16_NEGATIVE_ZERO: u16 = 0x8000;

/// A canonical quiet `NaN` in binary16 (exponent all ones, quiet bit set).
pub const F16_QUIET_NAN: u16 = 0x7e00;

/// Returns `true` when `h` encodes a `NaN` (exponent all ones, mantissa
/// non-zero).
pub const fn f16_is_nan(h: u16) -> bool {
    (h & 0x7c00) == 0x7c00 && (h & 0x03ff) != 0
}

/// Returns `true` when `h` encodes either positive or negative infinity.
pub const fn f16_is_inf(h: u16) -> bool {
    (h & 0x7fff) == 0x7c00
}

/// Returns the sign bit of `h`: `true` for negative values (including negative
/// zero and negative infinity).
pub const fn f16_sign(h: u16) -> bool {
    (h & 0x8000) != 0
}

/// Converts an `f32` to the binary16 bit pattern, rounding to nearest with ties
/// to even (`RNE`).
///
/// The implementation reads `x.to_bits()` and works purely on the integer bit
/// pattern: no `f32` arithmetic or comparison takes place.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    // Sign bit relocated from `f32` bit 31 to binary16 bit 15.
    let sign = ((bits >> 16) & 0x0000_8000) as u16;
    // Raw `f32` exponent field (bias 127).
    let raw_exp = ((bits >> 23) & 0xff) as i32;
    // 23-bit `f32` significand (no implicit leading one).
    let mantissa = bits & 0x007f_ffff;

    // `NaN` or infinity: `f32` exponent field is all ones.
    if raw_exp == 0xff {
        if mantissa != 0 {
            // `NaN`: carry the top significand bits down and force the quiet
            // bit (binary16 mantissa most significant bit, the `MSB`) so the
            // result stays a quiet `NaN` with a non-zero mantissa.
            let quiet = ((mantissa >> 13) as u16) | 0x0200;
            return sign | 0x7c00 | quiet;
        }
        return sign | 0x7c00;
    }

    // Re-bias the exponent for binary16 (127 -> 15).
    let exp = raw_exp - 112;

    if exp >= 0x1f {
        // Overflow: finite magnitude too large for binary16.
        return sign | 0x7c00;
    }

    if exp <= 0 {
        if exp < -10 {
            // Far below the smallest subnormal: flush to a signed zero.
            return sign;
        }
        // Subnormal result: restore the implicit leading one, then shift the
        // 24-bit significand right into the 10-bit field with `RNE` rounding.
        let m = mantissa | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let truncated = m >> shift;
        let round_bit = (m >> (shift - 1)) & 1;
        let sticky = (m & ((1u32 << (shift - 1)) - 1)) != 0;
        let mut result = truncated;
        if round_bit == 1 && (sticky || (truncated & 1) == 1) {
            result += 1;
        }
        return sign | result as u16;
    }

    // Normalized result: drop the low 13 significand bits with `RNE`. The least
    // significant retained bit (`LSB`) decides ties.
    let truncated = mantissa >> 13;
    let round_bit = (mantissa >> 12) & 1;
    let sticky = (mantissa & 0x0fff) != 0;
    let mut result = ((exp as u32) << 10) | truncated;
    if round_bit == 1 && (sticky || (truncated & 1) == 1) {
        // A carry out of the mantissa correctly increments the exponent field.
        result += 1;
    }
    if result >= 0x7c00 {
        // Rounding pushed the magnitude up to infinity.
        return sign | 0x7c00;
    }
    sign | result as u16
}

/// Converts a binary16 bit pattern to the corresponding `f32`.
///
/// Subnormal (denormal) inputs are normalized with an integer `leading_zeros`
/// count; infinities, `NaN`s (quiet bit preserved), and signed zeros are
/// reconstructed exactly. The result is produced with `f32::from_bits`.
pub fn f16_bits_to_f32(h: u16) -> f32 {
    // Sign bit relocated from binary16 bit 15 to `f32` bit 31.
    let sign = ((h as u32) & 0x0000_8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;

    if exp == 0x1f {
        if mant == 0 {
            // Infinity.
            return f32::from_bits(sign | 0x7f80_0000);
        }
        // `NaN`: binary16 mantissa bit 9 maps to `f32` mantissa bit 22, the
        // quiet bit, so the quiet status is preserved.
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }

    if exp == 0 {
        if mant == 0 {
            // Signed zero.
            return f32::from_bits(sign);
        }
        // Subnormal: normalize the mantissa using the integer leading-zero
        // count. The highest set bit sits at index `31 - lz`.
        let lz = mant.leading_zeros();
        let f32_exp = 134 - lz;
        let f32_mant = (mant << (lz - 8)) & 0x007f_ffff;
        return f32::from_bits(sign | (f32_exp << 23) | f32_mant);
    }

    // Normalized: re-bias the exponent (15 -> 127) and widen the mantissa.
    let f32_exp = exp + 112;
    let f32_mant = mant << 13;
    f32::from_bits(sign | (f32_exp << 23) | f32_mant)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for the rare `f32` comparison expressed as a magnitude; most
    // assertions instead compare raw bit patterns to sidestep `f32` equality.
    const TOL: f32 = 1e-12;

    #[test]
    fn forward_one() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
    }

    #[test]
    fn forward_two() {
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
    }

    #[test]
    fn forward_half() {
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
    }

    #[test]
    fn forward_negative_two() {
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
    }

    #[test]
    fn forward_max_finite() {
        // 65504 is the largest finite binary16 value: (2 - 2^-10) * 2^15.
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
    }

    #[test]
    fn forward_positive_zero() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
    }

    #[test]
    fn forward_negative_zero() {
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    }

    #[test]
    fn forward_positive_infinity() {
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
    }

    #[test]
    fn forward_negative_infinity() {
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
    }

    #[test]
    fn forward_min_normal() {
        // Smallest positive normal binary16 value: 2^-14, bits 0x3f000000.
        let min_normal = f32::from_bits(0x3880_0000);
        assert_eq!(f32_to_f16_bits(min_normal), 0x0400);
    }

    #[test]
    fn forward_min_subnormal() {
        // Smallest positive subnormal binary16 value: 2^-24, bits 0x33800000.
        let min_subnormal = f32::from_bits(0x3380_0000);
        assert_eq!(f32_to_f16_bits(min_subnormal), 0x0001);
    }

    #[test]
    fn forward_rne_tie_rounds_down_to_even() {
        // 1 + 2^-11 is the exact midpoint between 0x3c00 and 0x3c01; 0x3c00 is
        // even, so `RNE` rounds down.
        let midpoint = f32::from_bits(0x3f80_1000);
        assert_eq!(f32_to_f16_bits(midpoint), 0x3c00);
    }

    #[test]
    fn forward_rne_tie_rounds_up_to_even() {
        // 1 + 3*2^-11 is the exact midpoint between 0x3c01 and 0x3c02; 0x3c02 is
        // even, so `RNE` rounds up.
        let midpoint = f32::from_bits(0x3f80_3000);
        assert_eq!(f32_to_f16_bits(midpoint), 0x3c02);
    }

    #[test]
    fn forward_overflow_to_infinity() {
        assert_eq!(f32_to_f16_bits(70000.0), 0x7c00);
    }

    #[test]
    fn forward_negative_overflow_to_infinity() {
        assert_eq!(f32_to_f16_bits(-70000.0), 0xfc00);
    }

    #[test]
    fn forward_underflow_to_zero() {
        // 2^-26 is far below the smallest subnormal: flush to zero.
        let tiny = f32::from_bits(0x32800000);
        assert_eq!(f32_to_f16_bits(tiny), 0x0000);
    }

    #[test]
    fn forward_half_subnormal_ties_to_even_zero() {
        // 2^-25 is exactly halfway between 0 and 2^-24; `RNE` rounds to the even
        // representable value, which is zero.
        let half_min = f32::from_bits(0x3300_0000);
        assert_eq!(f32_to_f16_bits(half_min), 0x0000);
    }

    #[test]
    fn forward_just_above_half_subnormal_rounds_up() {
        // Slightly more than 2^-25 must round up to the smallest subnormal.
        let above = f32::from_bits(0x3300_0001);
        assert_eq!(f32_to_f16_bits(above), 0x0001);
    }

    #[test]
    fn forward_nan_is_nan() {
        let out = f32_to_f16_bits(f32::NAN);
        assert!(f16_is_nan(out));
    }

    #[test]
    fn forward_nan_preserves_quiet_bit() {
        // A signalling-looking payload still yields a quiet `NaN`.
        let signalling = f32::from_bits(0x7f80_0001);
        let out = f32_to_f16_bits(signalling);
        assert!(f16_is_nan(out));
        assert_ne!(out & 0x0200, 0);
    }

    #[test]
    fn reverse_one() {
        assert_eq!(f16_bits_to_f32(0x3c00).to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn reverse_half() {
        assert_eq!(f16_bits_to_f32(0x3800).to_bits(), 0.5f32.to_bits());
    }

    #[test]
    fn reverse_negative_two() {
        assert_eq!(f16_bits_to_f32(0xc000).to_bits(), (-2.0f32).to_bits());
    }

    #[test]
    fn reverse_min_subnormal() {
        // 2^-24 == f32 bits 0x33800000.
        assert_eq!(f16_bits_to_f32(0x0001).to_bits(), 0x3380_0000);
    }

    #[test]
    fn reverse_largest_subnormal() {
        // 1023 * 2^-24, normalized to f32 bits 0x387fc000.
        assert_eq!(f16_bits_to_f32(0x03ff).to_bits(), 0x387f_c000);
    }

    #[test]
    fn reverse_min_normal() {
        // 2^-14 == f32 bits 0x38800000.
        assert_eq!(f16_bits_to_f32(0x0400).to_bits(), 0x3880_0000);
    }

    #[test]
    fn reverse_max_finite() {
        // 65504 == f32 bits 0x477fe000.
        assert_eq!(f16_bits_to_f32(0x7bff).to_bits(), 0x477f_e000);
    }

    #[test]
    fn reverse_infinities() {
        assert!(f16_bits_to_f32(F16_POSITIVE_INFINITY).is_infinite());
        assert!(f16_bits_to_f32(F16_POSITIVE_INFINITY) > 0.0 - TOL);
        assert!(f16_bits_to_f32(F16_NEGATIVE_INFINITY).is_infinite());
        assert!(f16_sign(F16_NEGATIVE_INFINITY));
    }

    #[test]
    fn reverse_nan_is_nan() {
        assert!(f16_bits_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn reverse_signed_zero_sign_bits() {
        assert_eq!(f16_bits_to_f32(0x0000).to_bits(), 0x0000_0000);
        assert_eq!(f16_bits_to_f32(0x8000).to_bits(), 0x8000_0000);
    }

    #[test]
    fn roundtrip_known_patterns_bit_exact() {
        // Every finite and infinite binary16 pattern below survives a
        // half -> f32 -> half round trip unchanged.
        let cases: [u16; 11] = [
            0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x4000, 0x7bff, 0xc000, 0x7c00, 0xfc00,
        ];
        for &h in &cases {
            let widened = f16_bits_to_f32(h);
            assert_eq!(f32_to_f16_bits(widened), h);
        }
    }

    #[test]
    fn roundtrip_exact_f32_values_bit_exact() {
        // Values exactly representable in binary16 recover their original
        // f32 bit pattern after f32 -> half -> f32.
        let values: [f32; 6] = [1.0, 2.0, 0.5, -2.0, 0.25, 65504.0];
        for &v in &values {
            let back = f16_bits_to_f32(f32_to_f16_bits(v));
            assert_eq!(back.to_bits(), v.to_bits());
        }
    }

    #[test]
    fn roundtrip_all_subnormals_bit_exact() {
        // Exhaustively check every positive subnormal pattern.
        for h in 0x0001u16..=0x03ff {
            let back = f32_to_f16_bits(f16_bits_to_f32(h));
            assert_eq!(back, h);
        }
    }

    #[test]
    fn predicates_classify_correctly() {
        assert!(f16_is_inf(F16_POSITIVE_INFINITY));
        assert!(f16_is_inf(F16_NEGATIVE_INFINITY));
        assert!(!f16_is_inf(F16_QUIET_NAN));
        assert!(!f16_is_inf(0x3c00));

        assert!(f16_is_nan(F16_QUIET_NAN));
        assert!(f16_is_nan(0x7c01));
        assert!(!f16_is_nan(F16_POSITIVE_INFINITY));
        assert!(!f16_is_nan(0x0000));

        assert!(f16_sign(0x8000));
        assert!(f16_sign(0xc000));
        assert!(!f16_sign(0x0000));
        assert!(!f16_sign(0x3c00));
    }

    #[test]
    fn negative_subnormal_roundtrips() {
        let widened = f16_bits_to_f32(0x8001);
        assert!(f16_sign(0x8001));
        assert_eq!(f32_to_f16_bits(widened), 0x8001);
    }
}
