//! Firefly (fireflies / outlier) clamping for the sampler.
//!
//! Unbiased path tracing occasionally returns a single sample with enormous
//! energy: a near-grazing specular bounce that happens to connect to a small
//! bright emitter, a low-probability caustic path, or a `BSDF`/light sampling
//! combination with a tiny probability density in the denominator. Averaged
//! over a finite sample budget these rare spikes survive as isolated bright
//! pixels — "fireflies" — that are far more objectionable than the surrounding
//! noise and that converge away only very slowly.
//!
//! Firefly clamping trades a small, bounded bias for a large reduction in that
//! high-frequency noise by capping how much energy any one sample may
//! contribute. Production offline and real-time path tracers expose exactly
//! this control (an indirect-luminance or max-sample clamp). The clamp here
//! scales an over-bright sample down so its luminance equals the configured
//! maximum while preserving its chromaticity, so a clamped highlight keeps its
//! colour and only loses excess brightness.
//!
//! The default policy is [`FireflyClamp::Off`], which passes every sample
//! through unchanged and keeps the estimator exactly unbiased; clamping is
//! strictly opt-in.

use super::Vec3;

/// Luma weight for the red channel (standard luma coefficients, summing to one
/// so a unit-white sample has unit luminance).
const LUMA_R: f32 = 0.2126;
/// Luma weight for the green channel (see [`LUMA_R`]).
const LUMA_G: f32 = 0.7152;
/// Luma weight for the blue channel (see [`LUMA_R`]).
const LUMA_B: f32 = 0.0722;

/// Relative luminance of a linear `RGB` radiance value.
fn luminance(value: Vec3) -> f32 {
    LUMA_R * value.x + LUMA_G * value.y + LUMA_B * value.z
}

/// Per-sample firefly clamp policy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum FireflyClamp {
    /// No clamping: every sample passes through unchanged, keeping the estimate
    /// exactly unbiased.
    #[default]
    Off,
    /// Clamp any sample whose luminance exceeds `max_luminance` down to that
    /// luminance, preserving chromaticity. A non-positive maximum disables the
    /// clamp (treated as [`FireflyClamp::Off`]).
    MaxLuminance(f32),
}

impl FireflyClamp {
    /// Applies the clamp to one radiance sample.
    ///
    /// Returns the sample unchanged when clamping is off, when the maximum is
    /// non-positive, or when the sample luminance is already at or below the
    /// maximum. Otherwise scales the sample uniformly so its luminance equals
    /// the maximum, which preserves the ratio between channels (its hue and
    /// saturation) and only removes the excess brightness.
    #[must_use]
    pub fn apply(self, value: Vec3) -> Vec3 {
        match self {
            Self::Off => value,
            Self::MaxLuminance(max) => {
                let luma = luminance(value);
                // When `luma > max` the maximum is strictly positive and the
                // luminance is strictly larger, so the divisor is safe.
                if max <= 0.0 || luma <= max {
                    value
                } else {
                    value.scale(max / luma)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Luminance helper mirrored in the test module for readable assertions.
    fn luma(value: Vec3) -> f32 {
        luminance(value)
    }

    #[test]
    fn off_is_identity() {
        let sample = Vec3::new(100.0, 2.0, 0.3);
        assert_eq!(FireflyClamp::Off.apply(sample), sample);
    }

    #[test]
    fn non_positive_maximum_is_identity() {
        let sample = Vec3::new(10.0, 20.0, 30.0);
        assert_eq!(FireflyClamp::MaxLuminance(0.0).apply(sample), sample);
        assert_eq!(FireflyClamp::MaxLuminance(-1.0).apply(sample), sample);
    }

    #[test]
    fn below_threshold_is_unchanged() {
        let sample = Vec3::splat(0.4);
        // Grey luminance equals the channel value (weights sum to one).
        assert_eq!(FireflyClamp::MaxLuminance(1.0).apply(sample), sample);
    }

    #[test]
    fn above_threshold_is_scaled_to_exact_luminance() {
        let sample = Vec3::new(12.0, 4.0, 1.0);
        let clamped = FireflyClamp::MaxLuminance(2.0).apply(sample);
        assert!(
            (luma(clamped) - 2.0).abs() < 1.0e-4,
            "clamped luminance {} must equal the maximum",
            luma(clamped)
        );
        // Clamping must never raise luminance.
        assert!(luma(clamped) <= luma(sample));
    }

    #[test]
    fn chromaticity_is_preserved_under_clamping() {
        let sample = Vec3::new(12.0, 4.0, 1.0);
        let clamped = FireflyClamp::MaxLuminance(2.0).apply(sample);
        // Equal uniform scaling keeps every channel ratio identical.
        let scale = clamped.x / sample.x;
        assert!((clamped.y / sample.y - scale).abs() < 1.0e-5);
        assert!((clamped.z / sample.z - scale).abs() < 1.0e-5);
    }

    #[test]
    fn black_sample_is_unchanged() {
        let sample = Vec3::ZERO;
        assert_eq!(FireflyClamp::MaxLuminance(2.0).apply(sample), sample);
    }
}
