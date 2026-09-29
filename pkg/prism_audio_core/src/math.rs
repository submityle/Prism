//! Scalar type and fundamental DSP math helpers.
//!
//! All functions here are branch-light and safe to call from the real-time
//! audio thread; none allocate.

use bevy_math::ops;
use core::f32::consts::PI;

/// The audio sample scalar used throughout the engine.
///
/// 32-bit float is the industry-standard internal mixing format: it provides
/// ~144 dB of dynamic range and lets intermediate buses exceed 0 dBFS without
/// clipping until the final limiter stage.
pub type Sample = f32;

/// Smallest linear gain that is still treated as audible.
///
/// Gains at or below this threshold convert to [`f32::NEG_INFINITY`] decibels
/// (i.e. silence), matching the behaviour of professional metering.
pub const MIN_AUDIBLE_GAIN: Sample = 1.0e-7;

/// Converts a decibel value into a linear amplitude multiplier.
///
/// `0 dB` maps to `1.0`, `-6.02 dB` maps to ~`0.5`, and
/// [`f32::NEG_INFINITY`] maps to `0.0` (true silence).
///
/// # Examples
///
/// ```
/// # use prism_audio_core::math::db_to_linear;
/// assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
/// assert!((db_to_linear(-6.020599) - 0.5).abs() < 1e-4);
/// assert_eq!(db_to_linear(f32::NEG_INFINITY), 0.0);
/// ```
#[inline]
#[must_use]
pub fn db_to_linear(db: Sample) -> Sample {
    if db == f32::NEG_INFINITY {
        0.0
    } else {
        // 10^(db/20) computed via exp2 for speed and accuracy.
        exp10(db * 0.05)
    }
}

/// Converts a linear amplitude multiplier into decibels.
///
/// Values at or below [`MIN_AUDIBLE_GAIN`] map to [`f32::NEG_INFINITY`].
///
/// # Examples
///
/// ```
/// # use prism_audio_core::math::linear_to_db;
/// assert!((linear_to_db(1.0)).abs() < 1e-4);
/// assert!((linear_to_db(0.5) + 6.020599).abs() < 1e-3);
/// assert_eq!(linear_to_db(0.0), f32::NEG_INFINITY);
/// ```
#[inline]
#[must_use]
pub fn linear_to_db(linear: Sample) -> Sample {
    if linear <= MIN_AUDIBLE_GAIN {
        f32::NEG_INFINITY
    } else {
        20.0 * ops::log10(linear)
    }
}

/// Flushes denormalized (subnormal) floats to zero.
///
/// Feedback structures such as reverbs and IIR filters can decay into the
/// subnormal range, where some CPUs incur a large per-operation penalty.
/// Flushing keeps the real-time thread deterministic.
///
/// # Examples
///
/// ```
/// # use prism_audio_core::math::flush_denormal;
/// assert_eq!(flush_denormal(0.0), 0.0);
/// assert_eq!(flush_denormal(1.0e-40), 0.0);
/// assert_eq!(flush_denormal(0.5), 0.5);
/// ```
#[inline]
#[must_use]
pub fn flush_denormal(x: Sample) -> Sample {
    if x.abs() < f32::MIN_POSITIVE { 0.0 } else { x }
}

/// Clamps a sample into the closed range `[-1.0, 1.0]`.
///
/// Used as a last-resort hard clip; production signal chains should prefer a
/// proper limiter node instead.
#[inline]
#[must_use]
pub fn hard_clip(x: Sample) -> Sample {
    x.clamp(-1.0, 1.0)
}

/// Equal-power stereo pan gains for a pan position in `[-1.0, 1.0]`.
///
/// Returns `(left_gain, right_gain)`. At center (`0.0`) both channels receive
/// `1/sqrt(2)` so the perceived loudness is constant across the stereo field,
/// which is the standard constant-power pan law.
///
/// # Examples
///
/// ```
/// # use prism_audio_core::math::equal_power_pan;
/// let (l, r) = equal_power_pan(0.0);
/// assert!((l - r).abs() < 1e-6);
/// let (l, r) = equal_power_pan(-1.0);
/// assert!(l > 0.99 && r < 1e-3);
/// ```
#[inline]
#[must_use]
pub fn equal_power_pan(pan: Sample) -> (Sample, Sample) {
    // Map pan [-1, 1] to angle [0, pi/2].
    let normalized = (pan.clamp(-1.0, 1.0) + 1.0) * 0.5;
    let angle = normalized * (PI * 0.5);
    let (sin, cos) = ops::sin_cos(angle);
    (cos, sin)
}

/// Linearly interpolates between `a` and `b` by `t` (clamped to `[0, 1]`).
#[inline]
#[must_use]
pub fn lerp(a: Sample, b: Sample, t: Sample) -> Sample {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Computes `10^x` deterministically via [`bevy_math::ops`].
#[inline]
fn exp10(x: Sample) -> Sample {
    ops::exp(x * core::f32::consts::LN_10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_round_trip() {
        for &db in &[-48.0, -12.0, -6.0, 0.0, 6.0] {
            let lin = db_to_linear(db);
            let back = linear_to_db(lin);
            assert!((back - db).abs() < 1e-3, "db={db} back={back}");
        }
    }

    #[test]
    fn silence_maps_to_neg_inf() {
        assert_eq!(db_to_linear(f32::NEG_INFINITY), 0.0);
        assert_eq!(linear_to_db(0.0), f32::NEG_INFINITY);
    }

    #[test]
    fn pan_is_constant_power() {
        for i in 0..=20 {
            let pan = -1.0 + (i as f32) / 10.0;
            let (l, r) = equal_power_pan(pan);
            let power = l * l + r * r;
            assert!((power - 1.0).abs() < 1e-5, "pan={pan} power={power}");
        }
    }

    #[test]
    fn denormals_flush() {
        assert_eq!(flush_denormal(1.0e-40), 0.0);
        assert_eq!(flush_denormal(-1.0e-40), 0.0);
        assert_eq!(flush_denormal(0.25), 0.25);
    }
}
