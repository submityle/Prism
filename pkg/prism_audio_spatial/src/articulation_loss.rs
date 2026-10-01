//! Peutz articulation loss of consonants (`%ALcons`): a statistical prediction
//! of speech intelligibility from room geometry.
//!
//! Articulation loss of consonants is the classic Peutz-Klein measure of how
//! many consonants a listener fails to recognize in a reverberant space,
//! expressed as a percentage. Low values mean clear speech; values above
//! roughly `15` percent are generally considered unintelligible. Unlike a
//! measured parameter, `%ALcons` is predicted analytically from the room
//! volume, the mid-frequency reverberation time, the source-to-listener
//! distance, and the source directivity factor.
//!
//! # Model
//!
//! Let `r` be the listener distance in metres, `V` the room volume in cubic
//! metres, `RT` the mid-frequency reverberation time in seconds (typically the
//! average of the `500` Hz and `1000` Hz octave bands), and `Q` the source
//! directivity factor (`1` for an omnidirectional source). The reverberation
//! radius (critical distance) is
//! `r_c = PEUTZ_CRITICAL_DISTANCE_CONSTANT * sqrt(Q * V / RT)`, using the same
//! `0.057` coefficient as [`crate::room_acoustics::critical_distance`] extended
//! with the directivity factor.
//!
//! Peutz's empirical formula has two regimes about the `R_LIMIT = 3.16`
//! critical-distance boundary:
//!
//! - distance-dominated (`r <= R_LIMIT * r_c`):
//!   `%ALcons = 200 * r^2 * RT^2 / (V * Q)`,
//! - reverberation-saturated (`r > R_LIMIT * r_c`): `%ALcons = 9 * RT`.
//!
//! The result is clamped to `[0, MAX_ALCONS]`. An optional convenience mapping
//! converts `%ALcons` to an approximate speech transmission index via the
//! published empirical relation `STI = 0.9482 - 0.1845 * ln(%ALcons)`, clamped
//! to `[0, 1]`.
//!
//! # Real-time contract
//!
//! This is a control-rate, offline analytical estimator. It performs no heap
//! allocation, is not a per-sample callback, and must not run on an audio
//! thread. It never panics: a non-positive volume, reverberation time, or
//! directivity factor, a negative distance, or any non-finite input returns the
//! safe sentinel [`MAX_ALCONS`] (worst-case intelligibility). Intermediate
//! products are accumulated in `f64` and read back as [`Sample`]. All square
//! root and logarithm math routes through [`bevy_math::ops`].
//!
//! # Relationship
//!
//! This module predicts intelligibility from room statistics and geometry. It
//! complements [`crate::speech_transmission_index`], which measures
//! intelligibility from the modulation transfer function of a recorded impulse
//! response: the two are independent modelling paths to the same perceptual
//! concept, with different inputs and formulas, and neither duplicates the
//! other. The optional [`alcons_to_sti`] mapping is a published empirical
//! conversion only and is unrelated to the modulation-transfer computation in
//! that module. [`crate::room_acoustics`] supplies Sabine reverberation time
//! and the directivity-free critical distance that can serve as inputs here,
//! but this module keeps its own free functions and does not reuse that
//! module's cached state. All share the [`Sample`] scalar from
//! [`prism_audio_core`].
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published Peutz-Klein
//! articulation-loss formula and its empirical speech-transmission-index
//! mapping. It is pure classic DSP with no AI or ML. It is engine-agnostic and
//! contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code**; it is implemented purely
//! from that publicly documented acoustics literature.

use bevy_math::ops;

use prism_audio_core::math::Sample;

/// Critical-distance boundary multiplier separating the distance-dominated and
/// reverberation-saturated regimes of the Peutz formula.
pub const R_LIMIT: Sample = 3.16;

/// Maximum reported articulation loss as a percentage. Degenerate inputs and
/// saturated predictions clamp here.
pub const MAX_ALCONS: Sample = 100.0;

/// Reverberation-radius coefficient, matching the `0.057` constant used by
/// [`crate::room_acoustics::critical_distance`] and extended with the source
/// directivity factor `Q`.
pub const PEUTZ_CRITICAL_DISTANCE_CONSTANT: Sample = 0.057;

/// Smallest denominator treated as non-zero, matching the convention in
/// [`crate::room_acoustics`].
const MIN_DIVISOR: Sample = 1.0e-9;

/// Reverberation radius (critical distance) in metres for the Peutz model:
/// `r_c = PEUTZ_CRITICAL_DISTANCE_CONSTANT * sqrt(Q * V / RT)`.
///
/// Returns `0` for a non-positive volume, reverberation time, or directivity
/// factor, or any non-finite input. Unlike
/// [`crate::room_acoustics::critical_distance`] this variant includes the
/// directivity factor `Q`.
fn peutz_critical_distance_m(volume_m3: Sample, reverberation_time_s: Sample, directivity_q: Sample) -> Sample {
    if !volume_m3.is_finite()
        || !reverberation_time_s.is_finite()
        || !directivity_q.is_finite()
        || volume_m3 <= 0.0
        || reverberation_time_s <= MIN_DIVISOR
        || directivity_q <= 0.0
    {
        return 0.0;
    }
    PEUTZ_CRITICAL_DISTANCE_CONSTANT * ops::sqrt(directivity_q * volume_m3 / reverberation_time_s)
}

/// Reports whether every input is finite and physically valid.
#[inline]
fn inputs_valid(distance_m: Sample, volume_m3: Sample, reverberation_time_s: Sample, directivity_q: Sample) -> bool {
    distance_m.is_finite()
        && volume_m3.is_finite()
        && reverberation_time_s.is_finite()
        && directivity_q.is_finite()
        && distance_m >= 0.0
        && volume_m3 > MIN_DIVISOR
        && reverberation_time_s > 0.0
        && directivity_q > MIN_DIVISOR
}

/// Computes the articulation loss percentage and whether the prediction is in
/// the distance-dominated regime. Degenerate inputs return `(MAX_ALCONS, false)`.
fn compute(distance_m: Sample, volume_m3: Sample, reverberation_time_s: Sample, directivity_q: Sample) -> (Sample, bool) {
    if !inputs_valid(distance_m, volume_m3, reverberation_time_s, directivity_q) {
        return (MAX_ALCONS, false);
    }

    let r_c = peutz_critical_distance_m(volume_m3, reverberation_time_s, directivity_q);
    let boundary = R_LIMIT * r_c;
    // distance_m is finite and non-negative, boundary is finite non-negative.
    let distance_limited = distance_m <= boundary;

    let r = f64::from(distance_m);
    let v = f64::from(volume_m3);
    let rt = f64::from(reverberation_time_s);
    let q = f64::from(directivity_q);

    let alcons = if distance_limited {
        200.0 * r * r * rt * rt / (v * q)
    } else {
        9.0 * rt
    };

    let value = alcons as Sample;
    let clamped = if value.is_finite() {
        value.clamp(0.0, MAX_ALCONS)
    } else {
        MAX_ALCONS
    };
    (clamped, distance_limited)
}

/// Computes Peutz articulation loss of consonants `%ALcons` from room geometry.
///
/// `distance_m` is the source-to-listener distance in metres, `volume_m3` the
/// room volume in cubic metres, `reverberation_time_s` the mid-frequency
/// reverberation time in seconds, and `directivity_q` the source directivity
/// factor (`1` for an omnidirectional source). Degenerate inputs return
/// [`MAX_ALCONS`].
#[must_use]
pub fn articulation_loss_percent(distance_m: Sample, volume_m3: Sample, reverberation_time_s: Sample, directivity_q: Sample) -> Sample {
    compute(distance_m, volume_m3, reverberation_time_s, directivity_q).0
}

/// Maps an articulation loss percentage to an approximate speech transmission
/// index via the published empirical relation
/// `STI = 0.9482 - 0.1845 * ln(%ALcons)`, clamped to `[0, 1]`.
///
/// This is a convenience conversion only; a non-positive or non-finite
/// `alcons_percent` returns `1` (perfect intelligibility at zero loss).
#[must_use]
pub fn alcons_to_sti(alcons_percent: Sample) -> Sample {
    if !alcons_percent.is_finite() || alcons_percent <= 0.0 {
        return 1.0;
    }
    // ln via ops; the published relation is expressed in natural logarithm.
    let sti = 0.9482 - 0.1845 * ops::ln(alcons_percent);
    if sti.is_finite() { sti.clamp(0.0, 1.0) } else { 0.0 }
}

/// Peutz articulation-loss prediction with its derived intelligibility index.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ArticulationLoss {
    /// Articulation loss of consonants as a percentage in `[0, MAX_ALCONS]`.
    pub alcons_percent: Sample,
    /// Approximate speech transmission index from [`alcons_to_sti`], in `[0, 1]`.
    pub equivalent_sti: Sample,
    /// Whether the prediction is in the distance-dominated regime
    /// (`r <= R_LIMIT * r_c`) rather than the reverberation-saturated plateau.
    pub is_distance_limited: bool,
}

impl ArticulationLoss {
    /// Predicts articulation loss from room geometry.
    ///
    /// Degenerate inputs report [`MAX_ALCONS`], an equivalent index of `0`, and
    /// `is_distance_limited = false`.
    ///
    /// ```
    /// use prism_audio_spatial::articulation_loss::ArticulationLoss;
    ///
    /// // A listener 5 m from an omnidirectional source in a 1000 m^3 room with
    /// // a 1 s reverberation time: %ALcons = 200 * 25 * 1 / 1000 = 5 percent.
    /// let loss = ArticulationLoss::from_room(5.0, 1000.0, 1.0, 1.0);
    /// assert!((loss.alcons_percent - 5.0).abs() < 1e-3);
    /// assert!(loss.is_distance_limited);
    /// assert!(loss.equivalent_sti > 0.0 && loss.equivalent_sti < 1.0);
    /// ```
    #[must_use]
    pub fn from_room(distance_m: Sample, volume_m3: Sample, reverberation_time_s: Sample, directivity_q: Sample) -> Self {
        let (alcons_percent, is_distance_limited) =
            compute(distance_m, volume_m3, reverberation_time_s, directivity_q);
        Self {
            alcons_percent,
            equivalent_sti: alcons_to_sti(alcons_percent),
            is_distance_limited,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn known_value_matches_hand_calculation() {
        // V=1000, RT=1, Q=1, r=5: r_c = 0.057*sqrt(1000) = 1.8025,
        // boundary = 3.16*1.8025 = 5.696, so r=5 is distance-limited.
        // %ALcons = 200 * 25 * 1 / (1000 * 1) = 5.0.
        let a = articulation_loss_percent(5.0, 1000.0, 1.0, 1.0);
        assert!(approx(a, 5.0, 1e-3), "alcons {a}");
    }

    #[test]
    fn near_source_has_low_loss() {
        let near = articulation_loss_percent(0.5, 1000.0, 1.0, 1.0);
        let far = articulation_loss_percent(4.0, 1000.0, 1.0, 1.0);
        assert!(near < far, "near {near} far {far}");
        assert!(near < 1.0, "near {near}");
    }

    #[test]
    fn far_source_saturates_to_plateau() {
        // Far beyond the critical-distance boundary the loss saturates at 9*RT.
        let rt = 1.5;
        let a = articulation_loss_percent(500.0, 1000.0, rt, 1.0);
        assert!(approx(a, 9.0 * rt, 1e-3), "alcons {a}");
    }

    #[test]
    fn doubling_distance_quadruples_loss_in_distance_region() {
        // Both distances are inside the distance-dominated region.
        let a = articulation_loss_percent(2.0, 1000.0, 1.0, 1.0);
        let b = articulation_loss_percent(4.0, 1000.0, 1.0, 1.0);
        assert!(approx(b / a, 4.0, 1e-3), "a {a} b {b}");
    }

    #[test]
    fn is_distance_limited_flips_at_boundary() {
        // r_c = 1.8025, boundary = 5.696. Just inside vs just outside.
        let inside = ArticulationLoss::from_room(5.0, 1000.0, 1.0, 1.0);
        let outside = ArticulationLoss::from_room(6.0, 1000.0, 1.0, 1.0);
        assert!(inside.is_distance_limited);
        assert!(!outside.is_distance_limited);
    }

    #[test]
    fn non_positive_volume_is_sentinel() {
        assert_eq!(articulation_loss_percent(5.0, 0.0, 1.0, 1.0), MAX_ALCONS);
        assert_eq!(articulation_loss_percent(5.0, -10.0, 1.0, 1.0), MAX_ALCONS);
    }

    #[test]
    fn non_positive_rt_is_sentinel() {
        assert_eq!(articulation_loss_percent(5.0, 1000.0, 0.0, 1.0), MAX_ALCONS);
        assert_eq!(articulation_loss_percent(5.0, 1000.0, -1.0, 1.0), MAX_ALCONS);
    }

    #[test]
    fn non_positive_q_is_sentinel() {
        assert_eq!(articulation_loss_percent(5.0, 1000.0, 1.0, 0.0), MAX_ALCONS);
    }

    #[test]
    fn negative_distance_is_sentinel() {
        assert_eq!(articulation_loss_percent(-1.0, 1000.0, 1.0, 1.0), MAX_ALCONS);
    }

    #[test]
    fn non_finite_inputs_are_safe() {
        assert_eq!(
            articulation_loss_percent(Sample::NAN, 1000.0, 1.0, 1.0),
            MAX_ALCONS
        );
        assert_eq!(
            articulation_loss_percent(5.0, Sample::INFINITY, 1.0, 1.0),
            MAX_ALCONS
        );
    }

    #[test]
    fn alcons_to_sti_is_monotone_decreasing_and_clamped() {
        let s1 = alcons_to_sti(5.0);
        let s2 = alcons_to_sti(10.0);
        let s3 = alcons_to_sti(50.0);
        assert!(s1 > s2 && s2 > s3, "s1 {s1} s2 {s2} s3 {s3}");
        // Zero loss maps to perfect intelligibility, huge loss clamps to zero.
        assert_eq!(alcons_to_sti(0.0), 1.0);
        let big = alcons_to_sti(1.0e9);
        assert!((0.0..=1.0).contains(&big), "big {big}");
    }

    #[test]
    fn critical_distance_tracks_volume_and_rt() {
        let small_v = peutz_critical_distance_m(500.0, 1.0, 1.0);
        let large_v = peutz_critical_distance_m(2000.0, 1.0, 1.0);
        assert!(large_v > small_v, "small {small_v} large {large_v}");
        let short_rt = peutz_critical_distance_m(1000.0, 0.5, 1.0);
        let long_rt = peutz_critical_distance_m(1000.0, 2.0, 1.0);
        assert!(short_rt > long_rt, "short {short_rt} long {long_rt}");
    }

    #[test]
    fn from_room_matches_free_functions() {
        let loss = ArticulationLoss::from_room(3.0, 1500.0, 1.2, 2.0);
        assert!(approx(
            loss.alcons_percent,
            articulation_loss_percent(3.0, 1500.0, 1.2, 2.0),
            1e-6
        ));
        assert!(approx(
            loss.equivalent_sti,
            alcons_to_sti(loss.alcons_percent),
            1e-6
        ));
    }

    #[test]
    fn default_is_zero() {
        let loss = ArticulationLoss::default();
        assert_eq!(loss.alcons_percent, 0.0);
        assert_eq!(loss.equivalent_sti, 0.0);
        assert!(!loss.is_distance_limited);
    }

    #[test]
    fn constants_are_stable() {
        assert_eq!(R_LIMIT, 3.16);
        assert_eq!(MAX_ALCONS, 100.0);
        assert_eq!(PEUTZ_CRITICAL_DISTANCE_CONSTANT, 0.057);
    }

    #[test]
    fn result_is_clamped_to_range() {
        // A huge distance in a tiny room with long RT would overflow without the
        // clamp; it must stay within [0, MAX_ALCONS].
        let a = articulation_loss_percent(50.0, 1.0, 10.0, 0.001);
        assert!((0.0..=MAX_ALCONS).contains(&a), "alcons {a}");
    }
}
