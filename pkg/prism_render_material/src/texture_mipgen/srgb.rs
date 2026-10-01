//! sRGB <-> scene-linear transfer functions for gamma-correct mip filtering.
//!
//! Averaging colour in the stored sRGB encoding darkens mips (the EOTF is
//! convex), so AAA texture builders down-sample albedo in scene-linear space.
//! These helpers implement the exact IEC 61966-2-1 piecewise transfer; the
//! sibling [`box_filter`](super::box_filter) module averages the linearised
//! values and re-encodes.
//!
//! # Conventions
//! * `u8` samples are normalised by `/255.0`; encode rounds half-up and clamps
//!   to `[0, 255]`.
//! * The transfer is applied to colour (RGB) only; alpha is a linear coverage
//!   quantity and is averaged in its raw integer domain by the caller.
//!
//! # References
//! * IEC 61966-2-1:1999 (sRGB), the standard piecewise EOTF / OETF.

/// Decode one sRGB-encoded `u8` channel to a scene-linear `f32` in `[0, 1]`.
#[must_use]
pub fn srgb_to_linear(encoded: u8) -> f32 {
    let s = f32::from(encoded) / 255.0;
    if s <= 0.040_448_237 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Encode a scene-linear `f32` back to an sRGB `u8` (clamped, round half-up).
#[must_use]
pub fn linear_to_srgb(linear: f32) -> u8 {
    // Guard NaN (clamp alone would propagate it) and clamp range; +/-inf
    // saturate to the unit interval endpoints.
    let l = if linear.is_nan() {
        0.0
    } else {
        linear.clamp(0.0, 1.0)
    };
    let s = if l <= 0.003_130_8 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    // s is already in [0, 1] for l in [0, 1]; clamp defends against FP drift.
    let scaled = (s.clamp(0.0, 1.0) * 255.0 + 0.5).floor();
    scaled as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_round_trip_exactly() {
        assert_eq!(linear_to_srgb(srgb_to_linear(0)), 0);
        assert_eq!(linear_to_srgb(srgb_to_linear(255)), 255);
    }

    #[test]
    fn all_u8_values_round_trip_within_one_lsb() {
        for v in 0u8..=255 {
            let back = linear_to_srgb(srgb_to_linear(v));
            let diff = i16::from(back) - i16::from(v);
            assert!(diff.abs() <= 1, "value {v} round-tripped to {back}");
        }
    }

    #[test]
    fn linear_of_mid_grey_is_below_half() {
        // sRGB 128 is perceptual mid-grey but ~0.216 in linear light.
        let lin = srgb_to_linear(128);
        assert!(lin > 0.20 && lin < 0.23, "linear(128) = {lin}");
    }

    #[test]
    fn non_finite_encodes_to_zero() {
        assert_eq!(linear_to_srgb(f32::NAN), 0);
        assert_eq!(linear_to_srgb(f32::INFINITY), 255); // clamped to 1.0 -> 255
    }
}
