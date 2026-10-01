//! Interaural time difference `ITD` prediction from a spherical-head model.
//!
//! The interaural time difference is the small arrival-time disparity, in
//! seconds, between a sound reaching the near ear and the far ear. Together
//! with the interaural level difference it is one of the two dominant binaural
//! cues the auditory system uses to localise sources in the horizontal plane
//! (the duplex theory): the ITD dominates below roughly `1.5` kHz, where the
//! wavelength is long enough for phase to be unambiguous.
//!
//! This module predicts the ITD analytically from the geometry of a rigid
//! sphere of radius `head_radius_m` in a medium with sound speed
//! `speed_of_sound`. It offers the two classical closed-form predictors:
//!
//! # Model
//!
//! Let `a` be the head radius, `c` the speed of sound, and `phi` the source
//! azimuth in radians (`0` straight ahead, positive toward one ear). The
//! azimuth is first wrapped to `[-PI, PI]`.
//!
//! * **Woodworth ray model** (frequency-independent, geometric). A ray grazes
//!   the sphere, so the extra path to the far ear is the chord plus the arc
//!   subtended on the surface:
//!
//!   ```text
//!   ITD(phi) = a / c * (phi + sin(phi))        for |phi| <= PI / 2
//!   ```
//!
//!   For sources behind the interaural axis (`|phi| > PI / 2`) the geometry is
//!   mirror-symmetric about the `+-90` degree axis, so the model folds the
//!   back hemisphere onto the front using the supplement `PI - |phi|`:
//!
//!   ```text
//!   ITD(phi) = sign(phi) * a / c * ((PI - |phi|) + sin(PI - |phi|))
//!   ```
//!
//!   This makes the ITD rise from `0` straight ahead to its maximum at `+-90`
//!   degrees and fall back to `0` directly behind (`+-180` degrees), matching
//!   the front-back ambiguity of a symmetric sphere. The peak value is
//!
//!   ```text
//!   max_itd = a / c * (PI / 2 + 1)
//!   ```
//!
//! * **Kuhn asymptotic model** (frequency-dependent). Kuhn (1977) showed the
//!   ray model over-predicts and that the true ITD approaches two distinct
//!   sinusoidal asymptotes. At low frequencies (below about `500` Hz) the
//!   phase-delay ITD is
//!
//!   ```text
//!   itd_low(phi) = 3 * a * sin(phi) / c
//!   ```
//!
//!   and at high frequencies (above about `3` kHz) the group-delay ITD settles
//!   to
//!
//!   ```text
//!   itd_high(phi) = 2 * a * sin(phi) / c
//!   ```
//!
//!   Both are odd in `sin(phi)`, so they are naturally front-back symmetric and
//!   vanish straight ahead and directly behind.
//!
//! # Relationship
//!
//! This is a predictive geometric / asymptotic model and is distinct from the
//! measurement-based spatial descriptors in this crate:
//! [`crate::spatial_impression`] measures the interaural *cross-correlation*
//! `IACC` from a pair of binaural impulse responses (an observed similarity,
//! not a predicted delay), and [`crate::panner`] applies *amplitude* panning
//! laws rather than time differences. The ITD predicted here can be used to
//! drive a fractional-sample delay line for binaural rendering, complementing
//! the amplitude cues produced elsewhere. All share the [`Sample`] scalar from
//! [`prism_audio_core`].
//!
//! # Real-time contract
//!
//! Every predictor is a branch-light scalar function that performs no heap
//! allocation, locking, or panicking, so it is safe to call per source per
//! block (or even per sample) on the audio thread. Non-finite azimuths and a
//! non-positive radius or sound speed return `0`.
//!
//! # Provenance
//!
//! This is a textbook implementation of the Woodworth spherical-head ray model
//! and the Kuhn low/high-frequency ITD asymptotes. It is pure classic DSP with
//! no AI or ML. It is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or
//! derived code**; it is implemented purely from those publicly documented
//! acoustic models.

use bevy_math::ops;
use core::f32::consts::{FRAC_PI_2, PI, TAU};

use prism_audio_core::math::Sample;

/// Default head radius in metres (the widely used average of `8.75` cm).
pub const DEFAULT_HEAD_RADIUS_M: Sample = 0.0875;

/// Default speed of sound in metres per second (dry air near `20` degrees
/// Celsius).
pub const DEFAULT_SPEED_OF_SOUND: Sample = 343.0;

/// Multiplier on `a * sin(phi) / c` for the Kuhn low-frequency asymptote.
pub const KUHN_LOW_FACTOR: Sample = 3.0;

/// Multiplier on `a * sin(phi) / c` for the Kuhn high-frequency asymptote.
pub const KUHN_HIGH_FACTOR: Sample = 2.0;

/// Replaces a non-finite value with zero so a bad input cannot poison the
/// prediction.
#[inline]
fn finite(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
}

/// Wraps an angle in radians to the principal interval `[-PI, PI]`.
#[inline]
fn wrap_pi(phi: Sample) -> Sample {
    phi - TAU * ops::round(phi / TAU)
}

/// Spherical-head interaural-time-difference predictor.
///
/// Stores the head radius and the speed of sound, then exposes the Woodworth
/// ray model and the Kuhn low/high-frequency asymptotes as functions of source
/// azimuth.
///
/// # Examples
///
/// ```
/// use core::f32::consts::FRAC_PI_2;
/// use prism_audio_spatial::interaural_time_difference::InterauralTimeDifference;
///
/// let itd = InterauralTimeDifference::default();
/// // Straight ahead there is no interaural delay.
/// assert!(itd.woodworth_itd(0.0).abs() < 1e-9);
/// // At 90 degrees to one ear the Woodworth model reaches its maximum.
/// let right = itd.woodworth_itd(FRAC_PI_2);
/// assert!((right - itd.max_itd()).abs() < 1e-9);
/// // ... and the opposite side is the exact negative.
/// assert!((itd.woodworth_itd(-FRAC_PI_2) + right).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct InterauralTimeDifference {
    head_radius_m: Sample,
    speed_of_sound: Sample,
}

impl Default for InterauralTimeDifference {
    #[inline]
    fn default() -> Self {
        Self {
            head_radius_m: DEFAULT_HEAD_RADIUS_M,
            speed_of_sound: DEFAULT_SPEED_OF_SOUND,
        }
    }
}

impl InterauralTimeDifference {
    /// Builds a predictor for a head of radius `head_radius_m` metres in a
    /// medium with sound speed `speed_of_sound` metres per second.
    #[inline]
    #[must_use]
    pub fn new(head_radius_m: Sample, speed_of_sound: Sample) -> Self {
        Self {
            head_radius_m,
            speed_of_sound,
        }
    }

    /// Returns the configured head radius in metres.
    #[inline]
    #[must_use]
    pub fn head_radius_m(&self) -> Sample {
        self.head_radius_m
    }

    /// Returns the configured speed of sound in metres per second.
    #[inline]
    #[must_use]
    pub fn speed_of_sound(&self) -> Sample {
        self.speed_of_sound
    }

    /// Ratio `a / c`, or `0` if either value is non-positive or non-finite.
    #[inline]
    fn radius_over_speed(&self) -> Sample {
        let a = finite(self.head_radius_m);
        let c = finite(self.speed_of_sound);
        if a <= 0.0 || c <= 0.0 { 0.0 } else { a / c }
    }

    /// Predicts the Woodworth ray-model ITD, in seconds, for a source at
    /// azimuth `azimuth_rad` (`0` straight ahead, positive toward one ear).
    ///
    /// The result is positive for sources on the leading side, negative on the
    /// opposite side, `0` straight ahead or directly behind, and peaks at
    /// [`max_itd`](Self::max_itd) at `+-90` degrees.
    #[must_use]
    pub fn woodworth_itd(&self, azimuth_rad: Sample) -> Sample {
        let ratio = self.radius_over_speed();
        if ratio == 0.0 {
            return 0.0;
        }
        let phi = wrap_pi(finite(azimuth_rad));
        let abs = phi.abs();
        // Fold the rear hemisphere onto the front via the supplement so the
        // delay is symmetric about the +-90 degree interaural axis.
        let folded = if abs <= FRAC_PI_2 { abs } else { PI - abs };
        let magnitude = ratio * (folded + ops::sin(folded));
        if phi < 0.0 { -magnitude } else { magnitude }
    }

    /// Predicts the Kuhn low-frequency (below roughly `500` Hz) ITD, in
    /// seconds, for a source at azimuth `azimuth_rad`.
    #[must_use]
    pub fn kuhn_itd_low(&self, azimuth_rad: Sample) -> Sample {
        KUHN_LOW_FACTOR * self.radius_over_speed() * ops::sin(wrap_pi(finite(azimuth_rad)))
    }

    /// Predicts the Kuhn high-frequency (above roughly `3` kHz) ITD, in
    /// seconds, for a source at azimuth `azimuth_rad`.
    #[must_use]
    pub fn kuhn_itd_high(&self, azimuth_rad: Sample) -> Sample {
        KUHN_HIGH_FACTOR * self.radius_over_speed() * ops::sin(wrap_pi(finite(azimuth_rad)))
    }

    /// Returns the maximum Woodworth ITD, in seconds, attained at `+-90`
    /// degrees: `a / c * (PI / 2 + 1)`.
    #[must_use]
    pub fn max_itd(&self) -> Sample {
        self.radius_over_speed() * (FRAC_PI_2 + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn straight_ahead_is_zero() {
        let itd = InterauralTimeDifference::default();
        assert!(approx(itd.woodworth_itd(0.0), 0.0, 1e-9));
        assert!(approx(itd.kuhn_itd_low(0.0), 0.0, 1e-9));
        assert!(approx(itd.kuhn_itd_high(0.0), 0.0, 1e-9));
    }

    #[test]
    fn woodworth_peaks_at_ninety_degrees() {
        let itd = InterauralTimeDifference::default();
        assert!(approx(itd.woodworth_itd(FRAC_PI_2), itd.max_itd(), 1e-9));
    }

    #[test]
    fn woodworth_is_odd() {
        let itd = InterauralTimeDifference::default();
        for &phi in &[0.1, 0.6, 1.2, 2.0, 2.9] {
            assert!(approx(itd.woodworth_itd(-phi), -itd.woodworth_itd(phi), 1e-9));
        }
    }

    #[test]
    fn woodworth_vanishes_directly_behind() {
        let itd = InterauralTimeDifference::default();
        assert!(approx(itd.woodworth_itd(PI), 0.0, 1e-6));
        assert!(approx(itd.woodworth_itd(-PI), 0.0, 1e-6));
    }

    #[test]
    fn woodworth_front_back_symmetry() {
        // 60 degrees and its supplement 120 degrees share the same magnitude.
        let itd = InterauralTimeDifference::default();
        let front = itd.woodworth_itd(PI / 3.0);
        let back = itd.woodworth_itd(2.0 * PI / 3.0);
        assert!(approx(front, back, 1e-6));
    }

    #[test]
    fn woodworth_monotonic_on_front_quadrant() {
        let itd = InterauralTimeDifference::default();
        let mut prev = -1.0;
        let mut phi = 0.0;
        while phi <= FRAC_PI_2 {
            let v = itd.woodworth_itd(phi);
            assert!(v > prev);
            prev = v;
            phi += 0.05;
        }
    }

    #[test]
    fn woodworth_within_peak_bound() {
        let itd = InterauralTimeDifference::default();
        let peak = itd.max_itd();
        let mut phi = -PI;
        while phi <= PI {
            let v = itd.woodworth_itd(phi);
            assert!(v.abs() <= peak + 1e-6);
            phi += 0.03;
        }
    }

    #[test]
    fn kuhn_low_matches_closed_form_at_ninety() {
        let itd = InterauralTimeDifference::new(0.0875, 343.0);
        let expected = KUHN_LOW_FACTOR * 0.0875 / 343.0; // sin(90 deg) = 1
        assert!(approx(itd.kuhn_itd_low(FRAC_PI_2), expected, 1e-9));
    }

    #[test]
    fn kuhn_high_matches_closed_form_at_ninety() {
        let itd = InterauralTimeDifference::new(0.0875, 343.0);
        let expected = KUHN_HIGH_FACTOR * 0.0875 / 343.0;
        assert!(approx(itd.kuhn_itd_high(FRAC_PI_2), expected, 1e-9));
    }

    #[test]
    fn kuhn_low_exceeds_high_for_same_side() {
        let itd = InterauralTimeDifference::default();
        let low = itd.kuhn_itd_low(0.7);
        let high = itd.kuhn_itd_high(0.7);
        assert!(low > high);
        assert!(approx(low / high, 1.5, 1e-5)); // 3 / 2
    }

    #[test]
    fn kuhn_is_odd() {
        let itd = InterauralTimeDifference::default();
        for &phi in &[0.2, 0.9, 1.4] {
            assert!(approx(itd.kuhn_itd_low(-phi), -itd.kuhn_itd_low(phi), 1e-9));
            assert!(approx(itd.kuhn_itd_high(-phi), -itd.kuhn_itd_high(phi), 1e-9));
        }
    }

    #[test]
    fn kuhn_vanishes_directly_behind() {
        let itd = InterauralTimeDifference::default();
        assert!(approx(itd.kuhn_itd_low(PI), 0.0, 1e-6));
        assert!(approx(itd.kuhn_itd_high(PI), 0.0, 1e-6));
    }

    #[test]
    fn max_itd_closed_form_and_magnitude() {
        let itd = InterauralTimeDifference::new(0.0875, 343.0);
        let expected = 0.0875 / 343.0 * (FRAC_PI_2 + 1.0);
        assert!(approx(itd.max_itd(), expected, 1e-9));
        // Human ITD maxima are near 0.6 - 0.7 ms.
        assert!(itd.max_itd() > 0.0006 && itd.max_itd() < 0.0008);
    }

    #[test]
    fn azimuth_wraps_by_full_turn() {
        let itd = InterauralTimeDifference::default();
        assert!(approx(
            itd.woodworth_itd(0.7),
            itd.woodworth_itd(0.7 + TAU),
            1e-6
        ));
        assert!(approx(itd.kuhn_itd_low(0.7), itd.kuhn_itd_low(0.7 - TAU), 1e-6));
    }

    #[test]
    fn non_finite_inputs_are_safe() {
        let itd = InterauralTimeDifference::default();
        assert!(approx(itd.woodworth_itd(Sample::NAN), 0.0, 1e-12));
        assert!(approx(itd.kuhn_itd_low(Sample::INFINITY), 0.0, 1e-12));
        assert!(approx(itd.kuhn_itd_high(Sample::NEG_INFINITY), 0.0, 1e-12));
    }

    #[test]
    fn degenerate_geometry_returns_zero() {
        let zero_c = InterauralTimeDifference::new(0.0875, 0.0);
        assert!(approx(zero_c.woodworth_itd(FRAC_PI_2), 0.0, 1e-12));
        assert!(approx(zero_c.max_itd(), 0.0, 1e-12));
        let zero_a = InterauralTimeDifference::new(0.0, 343.0);
        assert!(approx(zero_a.kuhn_itd_low(FRAC_PI_2), 0.0, 1e-12));
        let neg = InterauralTimeDifference::new(-0.1, 343.0);
        assert!(approx(neg.woodworth_itd(FRAC_PI_2), 0.0, 1e-12));
    }

    #[test]
    fn accessors_and_default() {
        let itd = InterauralTimeDifference::default();
        assert!(approx(itd.head_radius_m(), DEFAULT_HEAD_RADIUS_M, 1e-12));
        assert!(approx(itd.speed_of_sound(), DEFAULT_SPEED_OF_SOUND, 1e-12));
        let custom = InterauralTimeDifference::new(0.09, 340.0);
        assert!(approx(custom.head_radius_m(), 0.09, 1e-12));
        assert!(approx(custom.speed_of_sound(), 340.0, 1e-12));
    }
}
