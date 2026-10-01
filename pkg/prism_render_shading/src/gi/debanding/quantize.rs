//! Quantization + dithered quantization — CPU golden reference.
//!
//! The final debanding step maps a continuous value to one of the discrete
//! levels available at a given bit depth. For `bits` bits there are
//! `L = 2^bits − 1` *intervals* between `L + 1` reconstruction levels, so the
//! quantization step is `q = 1 / L` and round-to-nearest quantization is
//! `round(v · L) / L`. Direct quantization of a smooth gradient produces the
//! visible contours ("banding") that dithering exists to hide.
//!
//! Adding a sub-step dither *before* rounding converts the deterministic
//! staircase error into noise whose local average tracks the true value. With
//! an ideal dither the expected quantized value equals the input (no DC bias),
//! so a smooth gradient reads as smooth again once the eye averages neighbours:
//!
//! ```text
//! dithered = quantize(value + dither · q,  bits)
//! ```
//!
//! where `dither` is the (dimensionless) perturbation produced by
//! [`crate::gi::debanding::ordered`] or [`crate::gi::debanding::noise_dither`],
//! scaled here by one quantization step `q`.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, IO, GPU, global state, or `unsafe`.
//! * `bits` is clamped to `[MIN_BITS, MAX_BITS]`; values are sanitized to
//!   finite and clamped to `[0, 1]` (display-referred, normalized signals).
//! * Round-to-nearest uses [`bevy_math::ops::round`] (round half away from
//!   zero); all outputs are finite and in `[0, 1]`.
//!
//! # References
//! * L. Schuchman, "Dither Signals and Their Effect on Quantization Noise",
//!   IEEE Trans. Communication Technology, 1964.
//! * J. Vanderkooy & S. P. Lipshitz, "Dither in Digital Audio", JAES 1987
//!   (why dither linearizes quantization; TPDF decorrelation).

use bevy_math::ops;

/// Minimum supported bit depth (a single interval, two levels).
pub const MIN_BITS: u32 = 1;

/// Maximum supported bit depth (`2¹⁶ − 1` intervals fit exactly in `f32`).
pub const MAX_BITS: u32 = 16;

/// Replaces a non-finite scalar with `fallback`.
#[inline]
#[must_use]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Clamps a requested bit depth to the supported `[MIN_BITS, MAX_BITS]` range.
#[inline]
#[must_use]
pub fn clamp_bits(bits: u32) -> u32 {
    bits.clamp(MIN_BITS, MAX_BITS)
}

/// Returns the number of quantization intervals `L = 2^bits − 1`.
///
/// `bits` is clamped to `[MIN_BITS, MAX_BITS]`, so the result is in
/// `[1, 65535]` and always `≥ 1` (the divide in [`quantize_step`] is safe).
#[inline]
#[must_use]
pub fn levels(bits: u32) -> f32 {
    let bits = clamp_bits(bits);
    // `1u32 << bits` is exact for `bits <= 16`; subtract one for interval count.
    ((1u32 << bits) - 1) as f32
}

/// Returns the quantization step `q = 1 / (2^bits − 1)`.
///
/// `bits` is clamped, so `q` is finite and in `(0, 1]`.
#[inline]
#[must_use]
pub fn quantize_step(bits: u32) -> f32 {
    1.0 / levels(bits)
}

/// Round-to-nearest quantization of `value` to `bits` bits.
///
/// Sanitizes and clamps `value` to `[0, 1]`, then returns
/// `round(value · L) / L` with `L = 2^bits − 1`. The result lands exactly on a
/// reconstruction level, is finite, and lies in `[0, 1]`.
#[inline]
#[must_use]
pub fn quantize(value: f32, bits: u32) -> f32 {
    let l = levels(bits);
    let v = finite_or(value, 0.0).clamp(0.0, 1.0);
    (ops::round(v * l) / l).clamp(0.0, 1.0)
}

/// Dithered quantization: `quantize(value + dither · q, bits)`.
///
/// `dither` is the dimensionless perturbation from an ordered or noise source
/// (typically `[-0.5, 0.5)` for one-LSB RPDF dither, or `[-1, 1]` for TPDF).
/// It is scaled by one quantization step `q` and added before rounding, so the
/// local average of many dithered pixels tracks the true `value` instead of
/// snapping to a single level. Inputs are sanitized; the result is finite and
/// in `[0, 1]`.
#[inline]
#[must_use]
pub fn quantize_dithered(value: f32, dither: f32, bits: u32) -> f32 {
    let q = quantize_step(bits);
    let d = finite_or(dither, 0.0);
    quantize(finite_or(value, 0.0) + d * q, bits)
}

/// Returns the signed quantization error `quantize(value) − value`.
///
/// Positive when quantization rounds up, negative when it rounds down. With
/// `value` clamped to `[0, 1]`, the magnitude never exceeds half a step
/// `q / 2`.
#[inline]
#[must_use]
pub fn quantization_error(value: f32, bits: u32) -> f32 {
    let v = finite_or(value, 0.0).clamp(0.0, 1.0);
    quantize(v, bits) - v
}

/// Returns the signed error of a single *dithered* sample vs. the true value.
///
/// `quantize_dithered(value, dither, bits) − value`. Over many decorrelated
/// dither draws this averages to ~zero for an unbiased dither, which is the
/// property the module-level and `tests` verifications check.
#[inline]
#[must_use]
pub fn dithered_error(value: f32, dither: f32, bits: u32) -> f32 {
    let v = finite_or(value, 0.0).clamp(0.0, 1.0);
    quantize_dithered(v, dither, bits) - v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::debanding::noise_dither::tpdf_dither;
    use crate::gi::debanding::ordered::bayer_threshold;

    /// Step and level counts match the bit-depth definition.
    #[test]
    fn levels_and_step() {
        assert_eq!(levels(1), 1.0);
        assert_eq!(levels(8), 255.0);
        assert_eq!(levels(16), 65535.0);
        assert!((quantize_step(8) - 1.0 / 255.0).abs() < 1.0e-9);
    }

    /// Out-of-range bit depths clamp into `[MIN_BITS, MAX_BITS]`.
    #[test]
    fn bit_clamping() {
        assert_eq!(clamp_bits(0), MIN_BITS);
        assert_eq!(clamp_bits(99), MAX_BITS);
        // `levels` must stay finite and positive even for a `0`-bit request.
        assert!(levels(0) >= 1.0);
        assert!(quantize_step(0).is_finite());
    }

    /// Quantization lands on reconstruction levels and preserves endpoints.
    #[test]
    fn quantize_snaps_to_levels() {
        assert_eq!(quantize(0.0, 8), 0.0);
        assert_eq!(quantize(1.0, 8), 1.0);
        let l = levels(8);
        for &v in &[0.1_f32, 0.25, 0.5, 0.73, 0.9] {
            let q = quantize(v, 8);
            let k = q * l;
            assert!((k - ops::round(k)).abs() < 1.0e-3, "not on a level: {q}");
            assert!((0.0..=1.0).contains(&q));
        }
    }

    /// Quantization error magnitude never exceeds half a step.
    #[test]
    fn error_bounded_by_half_step() {
        let half = 0.5 * quantize_step(8) + 1.0e-6;
        for i in 0..=1000 {
            let v = i as f32 / 1000.0;
            assert!(quantization_error(v, 8).abs() <= half, "v={v}");
        }
    }

    /// Non-finite inputs are sanitized to finite, in-range outputs.
    #[test]
    fn robust_to_non_finite() {
        assert!(quantize(f32::NAN, 8).is_finite());
        assert_eq!(quantize(-5.0, 8), 0.0);
        assert_eq!(quantize(5.0, 8), 1.0);
        assert!(quantize_dithered(f32::INFINITY, f32::NAN, 8).is_finite());
    }

    /// Adding a dither cannot push the output outside `[0, 1]`.
    #[test]
    fn dithered_output_in_range() {
        for &v in &[0.0_f32, 0.004, 0.5, 0.996, 1.0] {
            for &d in &[-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
                let q = quantize_dithered(v, d, 8);
                assert!((0.0..=1.0).contains(&q), "v={v} d={d} q={q}");
            }
        }
    }

    /// Bayer-dithered values average back to the true value (no DC bias),
    /// beating undithered quantization on a hard-to-represent gradient value.
    #[test]
    fn ordered_dither_removes_dc_bias() {
        // A value that sits far from any 4-bit level maximizes undithered bias.
        let value = 0.3_f32;
        let bits = 4;
        let mut sum = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..8u32 {
            for x in 0..8u32 {
                let d = bayer_threshold(x, y, 8);
                sum += quantize_dithered(value, d, bits) as f64;
                n += 1.0;
            }
        }
        let dithered_mean = sum / n;
        let undithered = quantize(value, bits) as f64;
        let dithered_bias = (dithered_mean - value as f64).abs();
        let undithered_bias = (undithered - value as f64).abs();
        assert!(
            dithered_bias < undithered_bias,
            "dither should reduce bias: dithered={dithered_bias} undithered={undithered_bias}"
        );
        assert!(dithered_bias < 0.01, "dithered bias too large: {dithered_bias}");
    }

    /// TPDF-dithered values converge to the true value over a large tile.
    #[test]
    fn tpdf_dither_converges() {
        let value = 0.42_f32;
        let bits = 5;
        let mut sum = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..128u32 {
            for x in 0..128u32 {
                let d = tpdf_dither(x, y, 7);
                sum += quantize_dithered(value, d, bits) as f64;
                n += 1.0;
            }
        }
        let mean = sum / n;
        assert!((mean - value as f64).abs() < 0.005, "mean={mean}");
    }

    /// `dithered_error` is consistent with `quantize_dithered`.
    #[test]
    fn dithered_error_consistency() {
        let v = 0.6_f32;
        let d = 0.25_f32;
        let e = dithered_error(v, d, 8);
        assert!((e - (quantize_dithered(v, d, 8) - v)).abs() < 1.0e-7);
    }
}
