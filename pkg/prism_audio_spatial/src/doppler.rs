//! Classic Doppler pitch-shift model for a moving source and a (near-)
//! stationary observer.
//!
//! The Doppler effect describes the change in observed frequency when a sound
//! source and listener move relative to one another: approaching sources are
//! pitched up and receding sources are pitched down. This module implements the
//! textbook single-degree-of-freedom approximation
//!
//! ```text
//! f_observed = f_source * c / (c + v_radial)
//! ```
//!
//! where `c` is the speed of sound and `v_radial` is the component of the
//! source's velocity along the source -> listener line. Following the
//! convention established by [`crate::geometry`], `v_radial` is **positive when
//! the source is receding** from the listener (which lowers the pitch,
//! `ratio < 1`) and negative when it is approaching (which raises the pitch,
//! `ratio > 1`). A `scale` term exaggerates or attenuates the effect for
//! artistic control.
//!
//! # Determinism
//!
//! This module performs only comparisons, additions, multiplications, and a
//! single division on [`Sample`] values; it invokes no transcendental
//! functions and therefore needs nothing from [`bevy_math::ops`]. The math is
//! bit-reproducible across targets.
//!
//! # Provenance
//!
//! The formula above is the classic (stationary-observer / moving-source)
//! Doppler relation from standard acoustics textbooks. This module contains
//! **no Unreal Engine, Unity, Godot, Wwise, or FMOD source or derived code**;
//! it is implemented purely from publicly documented physics/DSP knowledge.

use prism_audio_core::math::Sample;

/// Speed of sound in dry air at ~20 degrees Celsius, in metres per second.
///
/// This is the standard reference value used as the default `c` in the Doppler
/// relation.
pub const SPEED_OF_SOUND_MPS: Sample = 343.0;

/// Computes the raw Doppler frequency ratio `f_observed / f_source`.
///
/// The ratio is
///
/// ```text
/// ratio = c / (c + scale * radial_velocity)
/// ```
///
/// with `c = speed_of_sound.max(1e-3)` so the propagation speed can never be
/// zero or negative. Per the crate convention, `radial_velocity` is **positive
/// when the source is receding** (ratio `< 1`, pitched down) and negative when
/// approaching (ratio `> 1`, pitched up). The `scale` factor multiplies the
/// radial velocity to exaggerate (`> 1`) or soften (`< 1`) the effect; a
/// `scale` of `0` disables it and returns `1.0`.
///
/// # Degenerate denominator
///
/// If the source recedes at or faster than the speed of sound (a supersonic
/// tail), the denominator `c + scale * radial_velocity` can reach zero or turn
/// negative, which would otherwise produce a division by zero (`inf`/`NaN`) or
/// an inverted, negative ratio. To keep the output finite and positive the
/// denominator is clamped to a small positive floor of `c * 0.05`.
///
/// # Examples
///
/// ```
/// # use prism_audio_spatial::doppler::{doppler_ratio, SPEED_OF_SOUND_MPS};
/// // A stationary source: no shift.
/// assert!((doppler_ratio(0.0, SPEED_OF_SOUND_MPS, 1.0) - 1.0).abs() < 1e-6);
/// // Receding source: pitched down.
/// assert!(doppler_ratio(30.0, SPEED_OF_SOUND_MPS, 1.0) < 1.0);
/// // Approaching source: pitched up.
/// assert!(doppler_ratio(-30.0, SPEED_OF_SOUND_MPS, 1.0) > 1.0);
/// ```
#[must_use]
#[inline]
pub fn doppler_ratio(radial_velocity: Sample, speed_of_sound: Sample, scale: Sample) -> f32 {
    let c = speed_of_sound.max(1.0e-3);
    let denominator = c + scale * radial_velocity;
    // Clamp the denominator to a small positive floor so a supersonic receding
    // source cannot make it zero or negative (which would yield inf/NaN or an
    // inverted ratio).
    let floor = c * 0.05;
    let denominator = denominator.max(floor);
    c / denominator
}

/// Configurable Doppler model turning a listener-relative radial velocity into
/// a clamped pitch multiplier.
///
/// All fields are plain configuration; construct via [`Doppler::default`] and
/// tweak, or build one directly.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Doppler {
    /// Speed of sound in metres per second used as `c` in the ratio. Clamped to
    /// a small positive value internally, so non-positive settings are safe.
    pub speed_of_sound: Sample,
    /// Multiplier applied to the radial velocity before the ratio is computed.
    /// `1.0` is physically accurate; larger values exaggerate the effect and
    /// `0.0` disables it entirely.
    pub scale: Sample,
    /// Symmetric bound on the returned pitch multiplier: the result is clamped
    /// to `[1.0 / max_ratio, max_ratio]`. Values below `1.0` are treated as
    /// `1.0` (no shift allowed).
    pub max_ratio: Sample,
}

impl Default for Doppler {
    #[inline]
    fn default() -> Self {
        Self { speed_of_sound: SPEED_OF_SOUND_MPS, scale: 1.0, max_ratio: 2.0 }
    }
}

impl Doppler {
    /// Computes the perceived pitch multiplier for a source with the given
    /// listener-relative `radial_velocity` (metres per second, positive when
    /// receding).
    ///
    /// This evaluates [`doppler_ratio`] with this model's `speed_of_sound` and
    /// `scale`, then clamps the result to `[1.0 / max_ratio, max_ratio]` where
    /// `max_ratio` is first floored at `1.0`. The clamp bounds the audible
    /// pitch excursion and guarantees a finite, positive multiplier even for
    /// extreme (e.g. supersonic) velocities.
    ///
    /// # Examples
    ///
    /// ```
    /// # use prism_audio_spatial::doppler::Doppler;
    /// let doppler = Doppler::default();
    /// // Stationary source: unity pitch.
    /// assert!((doppler.pitch_ratio(0.0) - 1.0).abs() < 1e-6);
    /// // Extreme approach is clamped to `max_ratio`.
    /// assert!((doppler.pitch_ratio(-1.0e6) - doppler.max_ratio).abs() < 1e-4);
    /// ```
    #[must_use]
    #[inline]
    pub fn pitch_ratio(&self, radial_velocity: Sample) -> f32 {
        let max_ratio = self.max_ratio.max(1.0);
        let min_ratio = 1.0 / max_ratio;
        let ratio = doppler_ratio(radial_velocity, self.speed_of_sound, self.scale);
        ratio.clamp(min_ratio, max_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for floating-point comparisons in these tests.
    const EPS: f32 = 1.0e-5;

    #[test]
    fn zero_radial_velocity_is_unity() {
        assert!((doppler_ratio(0.0, SPEED_OF_SOUND_MPS, 1.0) - 1.0).abs() < EPS);
        assert!((Doppler::default().pitch_ratio(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn receding_source_lowers_pitch() {
        // Positive radial velocity = receding => ratio < 1 (pitched down).
        let ratio = doppler_ratio(40.0, SPEED_OF_SOUND_MPS, 1.0);
        assert!(ratio < 1.0, "receding ratio should be < 1, got {ratio}");
        assert!(ratio > 0.0);
        assert!(Doppler::default().pitch_ratio(40.0) < 1.0);
    }

    #[test]
    fn approaching_source_raises_pitch() {
        // Negative radial velocity = approaching => ratio > 1 (pitched up).
        let ratio = doppler_ratio(-40.0, SPEED_OF_SOUND_MPS, 1.0);
        assert!(ratio > 1.0, "approaching ratio should be > 1, got {ratio}");
        assert!(Doppler::default().pitch_ratio(-40.0) > 1.0);
    }

    #[test]
    fn large_velocity_is_clamped_by_max_ratio() {
        let doppler = Doppler::default();
        let max = doppler.max_ratio;
        let min = 1.0 / max;

        // Extreme approach saturates at the upper bound.
        let up = doppler.pitch_ratio(-1.0e6);
        assert!((up - max).abs() < 1e-4, "expected {max}, got {up}");

        // Extreme recession saturates at the lower bound.
        let down = doppler.pitch_ratio(1.0e6);
        assert!((down - min).abs() < 1e-4, "expected {min}, got {down}");
    }

    #[test]
    fn scale_amplifies_and_softens_effect() {
        let v = 30.0;
        let strong = doppler_ratio(v, SPEED_OF_SOUND_MPS, 2.0);
        let normal = doppler_ratio(v, SPEED_OF_SOUND_MPS, 1.0);
        let weak = doppler_ratio(v, SPEED_OF_SOUND_MPS, 0.5);

        // All are receding (< 1). Larger scale pushes the ratio further from 1.
        assert!(strong < normal, "strong {strong} should be < normal {normal}");
        assert!(normal < weak, "normal {normal} should be < weak {weak}");
        assert!(weak < 1.0);
    }

    #[test]
    fn zero_scale_disables_effect() {
        assert!((doppler_ratio(123.0, SPEED_OF_SOUND_MPS, 0.0) - 1.0).abs() < EPS);
        assert!((doppler_ratio(-123.0, SPEED_OF_SOUND_MPS, 0.0) - 1.0).abs() < EPS);

        let doppler = Doppler { scale: 0.0, ..Doppler::default() };
        assert!((doppler.pitch_ratio(500.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn degenerate_denominator_stays_finite_and_positive() {
        // Supersonic recession would drive c + scale*v to zero/negative.
        let ratio = doppler_ratio(SPEED_OF_SOUND_MPS, SPEED_OF_SOUND_MPS, 1.0);
        assert!(ratio.is_finite(), "ratio should be finite, got {ratio}");
        assert!(ratio > 0.0, "ratio should be positive, got {ratio}");

        // Far beyond the speed of sound: denominator clamped to the floor.
        let extreme = doppler_ratio(1.0e6, SPEED_OF_SOUND_MPS, 10.0);
        assert!(extreme.is_finite());
        assert!(extreme > 0.0);

        // The model's clamp also keeps things bounded and finite.
        let clamped = Doppler::default().pitch_ratio(1.0e6);
        assert!(clamped.is_finite());
        assert!(clamped > 0.0);
    }

    #[test]
    fn non_positive_speed_of_sound_is_safe() {
        // A zero/negative c must not divide by zero; it is floored internally.
        let ratio = doppler_ratio(10.0, 0.0, 1.0);
        assert!(ratio.is_finite());
        assert!(ratio > 0.0);
    }

    #[test]
    fn max_ratio_below_one_is_treated_as_unity() {
        // max_ratio < 1.0 is floored to 1.0, collapsing the clamp to [1, 1].
        let doppler = Doppler { max_ratio: 0.25, ..Doppler::default() };
        assert!((doppler.pitch_ratio(-100.0) - 1.0).abs() < EPS);
        assert!((doppler.pitch_ratio(100.0) - 1.0).abs() < EPS);
    }
}
