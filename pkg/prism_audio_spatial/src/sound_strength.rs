//! ISO 3382-1 sound strength `G` from measured impulse responses.
//!
//! Sound strength (also called relative sound level) is the energy gain a room
//! provides to a source relative to the free field. It is defined as the
//! logarithmic ratio of the energy of a room impulse response to the energy
//! that the same source produces in a free field at `10 m`. A large positive
//! `G` means the room strongly supports and amplifies the source; a value near
//! `0` means the room adds little over the free field.
//!
//! # Model
//!
//! Let `p[n]` be the measured room impulse response and `E_ref` the reference
//! energy of the same source measured in a free field at `10 m` (supplied by
//! the caller). With `t = n / sample_rate` the arrival time in seconds and the
//! early-to-late boundary at `80 ms`:
//!
//! - Total strength `G = 10 * log10(sum_{all n} p[n]^2 / E_ref)` decibels.
//! - Early strength `G_early = 10 * log10(sum_{0 <= t <= 80 ms} p^2 / E_ref)`.
//! - Late strength `G_late = 10 * log10(sum_{80 ms < t <= end} p^2 / E_ref)`.
//!
//! The `80 ms` boundary is shared with [`crate::room_clarity`] via
//! [`crate::room_clarity::EARLY_LATE_SPLIT_80_MS`].
//!
//! # Relationship
//!
//! This module complements [`crate::room_clarity`] (clarity, definition,
//! reverberation times) and [`crate::spatial_impression`] (lateral energy and
//! interaural correlation), and the geometry-based reverberation estimate in
//! [`crate::room_acoustics`]. Clarity answers how distinct the sound is, spatial
//! impression answers how wide and enveloping it is, and this module answers how
//! loud the room makes the source. All share the [`Sample`] scalar from
//! [`prism_audio_core`], reuse the `80 ms` split constant from
//! [`crate::room_clarity`], and none reimplements another.
//!
//! # Real-time contract
//!
//! These are control-rate, offline estimators: each accepts a whole impulse
//! response and performs no heap allocation. They are not per-sample callbacks
//! and must not run on an audio thread. They never panic: empty, all-zero,
//! non-finite, non-positive reference energy, or non-positive sample-rate
//! inputs return safe defaults ([`MIN_STRENGTH_DB`] or `0`). All logarithmic and
//! length math routes through [`bevy_math::ops`].
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published ISO 3382-1 sound
//! strength parameter `G`. It is pure classic DSP with no AI or ML. It is
//! engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented standard.

use core::f32::consts::LN_10;

use bevy_math::ops;

use prism_audio_core::math::Sample;

use crate::room_clarity::EARLY_LATE_SPLIT_80_MS;

/// Safe sentinel in decibels returned when a measured energy is effectively
/// zero, avoiding negative infinity from the logarithm.
pub const MIN_STRENGTH_DB: Sample = -100.0;

/// Energies below this threshold are treated as silence.
const ENERGY_FLOOR: Sample = 1e-20;

/// Sanitises a sample, mapping non-finite values to `0`.
#[inline]
fn finite(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
}

/// Accumulates the energy `sum p[n]^2` of a slice, ignoring non-finite samples.
fn energy(slice: &[Sample]) -> Sample {
    let mut sum = 0.0;
    for &x in slice {
        let v = finite(x);
        sum += v * v;
    }
    sum
}

/// Converts a time in milliseconds to a sample index, clamped to `[0, len]`.
fn ms_to_index(ms: Sample, sample_rate: Sample, len: usize) -> usize {
    let x = ms / 1000.0 * sample_rate;
    if !x.is_finite() || x <= 0.0 {
        return 0;
    }
    let idx = ops::round(x) as usize;
    idx.min(len)
}

/// Converts an energy ratio to decibels, `10 * log10(energy / reference)`.
///
/// Returns [`MIN_STRENGTH_DB`] when the energy is below [`ENERGY_FLOOR`] or the
/// reference energy is not strictly positive.
fn energy_to_strength_db(energy: Sample, reference_energy: Sample) -> Sample {
    if !reference_energy.is_finite() || reference_energy <= 0.0 {
        return MIN_STRENGTH_DB;
    }
    #[expect(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "negated comparison keeps NaN energies on the safe branch, unlike the suggested direct comparison"
    )]
    if !(energy > ENERGY_FLOOR) {
        return MIN_STRENGTH_DB;
    }
    let db = 10.0 * ops::ln(energy / reference_energy) / LN_10;
    if db.is_finite() { db } else { MIN_STRENGTH_DB }
}

/// Computes the total sound strength `G` in decibels relative to the free-field
/// reference energy.
///
/// `G = 10 * log10(sum_{all n} p[n]^2 / reference_energy)`. An empty or silent
/// response, or a non-positive or non-finite `reference_energy`, returns
/// [`MIN_STRENGTH_DB`].
#[must_use]
pub fn sound_strength_db(response: &[Sample], reference_energy: Sample) -> Sample {
    if response.is_empty() {
        return MIN_STRENGTH_DB;
    }
    energy_to_strength_db(energy(response), reference_energy)
}

/// Computes the early sound strength `G_early` in decibels.
///
/// `G_early = 10 * log10(sum_{0 <= t <= 80 ms} p^2 / reference_energy)`. An
/// empty or silent early window, a non-positive or non-finite `reference_energy`,
/// or a non-positive or non-finite `sample_rate` returns [`MIN_STRENGTH_DB`].
#[must_use]
pub fn early_sound_strength_db(
    response: &[Sample],
    reference_energy: Sample,
    sample_rate: Sample,
) -> Sample {
    if response.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return MIN_STRENGTH_DB;
    }
    let split = ms_to_index(EARLY_LATE_SPLIT_80_MS, sample_rate, response.len());
    energy_to_strength_db(energy(&response[..split]), reference_energy)
}

/// Computes the late sound strength `G_late` in decibels.
///
/// `G_late = 10 * log10(sum_{80 ms < t <= end} p^2 / reference_energy)`. An
/// empty or silent late window, a non-positive or non-finite `reference_energy`,
/// or a non-positive or non-finite `sample_rate` returns [`MIN_STRENGTH_DB`].
#[must_use]
pub fn late_sound_strength_db(
    response: &[Sample],
    reference_energy: Sample,
    sample_rate: Sample,
) -> Sample {
    if response.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return MIN_STRENGTH_DB;
    }
    let split = ms_to_index(EARLY_LATE_SPLIT_80_MS, sample_rate, response.len());
    energy_to_strength_db(energy(&response[split..]), reference_energy)
}

/// The ISO 3382-1 sound-strength parameters of a measured room.
///
/// All fields are in decibels relative to the free-field reference energy at
/// `10 m`. A silent window reports [`MIN_STRENGTH_DB`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SoundStrength {
    /// Total sound strength `G` in decibels.
    pub g_db: Sample,
    /// Early sound strength `G_early` (`0` to `80 ms`) in decibels.
    pub g_early_db: Sample,
    /// Late sound strength `G_late` (`80 ms` to the end) in decibels.
    pub g_late_db: Sample,
}

impl Default for SoundStrength {
    fn default() -> Self {
        Self {
            g_db: 0.0,
            g_early_db: 0.0,
            g_late_db: 0.0,
        }
    }
}

impl SoundStrength {
    /// Computes the total, early, and late sound strengths from a measured
    /// impulse response in a single bounded scan.
    ///
    /// `reference_energy` is the free-field reference energy of the same source
    /// at `10 m`. Empty, silent, non-finite, non-positive reference energy, or
    /// non-positive sample-rate inputs yield [`MIN_STRENGTH_DB`] in the affected
    /// fields.
    ///
    /// ```
    /// use prism_audio_spatial::sound_strength::SoundStrength;
    ///
    /// // A unit impulse with a modest decaying tail, referenced to a free-field
    /// // energy of 1.0 at 10 m.
    /// let mut rir = vec![0.0f32; 48_000];
    /// rir[0] = 1.0;
    /// for n in 1..rir.len() {
    ///     rir[n] = (-(n as f32) / 4_000.0).exp() * 0.1;
    /// }
    /// let strength = SoundStrength::from_impulse_response(&rir, 1.0, 48_000.0);
    /// assert!(strength.g_db >= strength.g_early_db);
    /// assert!(strength.g_db.is_finite());
    /// ```
    #[must_use]
    pub fn from_impulse_response(
        response: &[Sample],
        reference_energy: Sample,
        sample_rate: Sample,
    ) -> Self {
        if response.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Self {
                g_db: sound_strength_db(response, reference_energy),
                g_early_db: MIN_STRENGTH_DB,
                g_late_db: MIN_STRENGTH_DB,
            };
        }
        let split = ms_to_index(EARLY_LATE_SPLIT_80_MS, sample_rate, response.len());
        let early_energy = energy(&response[..split]);
        let late_energy = energy(&response[split..]);
        let total_energy = early_energy + late_energy;
        Self {
            g_db: energy_to_strength_db(total_energy, reference_energy),
            g_early_db: energy_to_strength_db(early_energy, reference_energy),
            g_late_db: energy_to_strength_db(late_energy, reference_energy),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    const SR: Sample = 48_000.0;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    fn exp_response(len: usize, tau: Sample) -> Vec<Sample> {
        let mut out = vec![0.0; len];
        for (n, x) in out.iter_mut().enumerate() {
            *x = ops::exp(-(n as Sample) / tau);
        }
        out
    }

    #[test]
    fn dirac_strength_matches_closed_form() {
        // Single unit impulse: total energy = 1, so G = 10 log10(1 / E_ref).
        let rir = [1.0, 0.0, 0.0, 0.0];
        let reference = 2.0;
        let expected = 10.0 * ops::ln(1.0 / reference) / LN_10;
        assert!(approx(sound_strength_db(&rir, reference), expected, 1e-6));
    }

    #[test]
    fn doubling_energy_adds_about_three_db() {
        let rir = exp_response(8_000, 1_000.0);
        let single = sound_strength_db(&rir, 1.0);
        // Scaling amplitude by sqrt(2) doubles the energy.
        let scaled: Vec<Sample> = rir.iter().map(|&x| x * core::f32::consts::SQRT_2).collect();
        let doubled = sound_strength_db(&scaled, 1.0);
        assert!(approx(doubled - single, 3.0103, 1e-2), "delta {}", doubled - single);
    }

    #[test]
    fn early_plus_late_energy_equals_total() {
        let rir = exp_response(8_000, 1_500.0);
        let reference = 1.0;
        let g = sound_strength_db(&rir, reference);
        let g_early = early_sound_strength_db(&rir, reference, SR);
        let g_late = late_sound_strength_db(&rir, reference, SR);
        // Convert each dB back to a linear energy ratio and sum.
        let lin = |db: Sample| ops::exp(db * LN_10 / 10.0);
        let total_from_parts = lin(g_early) + lin(g_late);
        let expected = 10.0 * ops::ln(total_from_parts) / LN_10;
        assert!(approx(g, expected, 1e-3), "g {g} expected {expected}");
    }

    #[test]
    fn farther_field_reduces_strength() {
        // A weaker (farther) response has lower G against the same reference.
        let near = exp_response(8_000, 1_500.0);
        let far: Vec<Sample> = near.iter().map(|&x| x * 0.5).collect();
        assert!(sound_strength_db(&far, 1.0) < sound_strength_db(&near, 1.0));
    }

    #[test]
    fn larger_reference_reduces_strength() {
        let rir = exp_response(4_000, 1_000.0);
        assert!(sound_strength_db(&rir, 10.0) < sound_strength_db(&rir, 1.0));
    }

    #[test]
    fn empty_response_is_safe() {
        let empty: [Sample; 0] = [];
        assert_eq!(sound_strength_db(&empty, 1.0), MIN_STRENGTH_DB);
        assert_eq!(early_sound_strength_db(&empty, 1.0, SR), MIN_STRENGTH_DB);
        assert_eq!(late_sound_strength_db(&empty, 1.0, SR), MIN_STRENGTH_DB);
    }

    #[test]
    fn all_zero_response_is_safe() {
        let z = vec![0.0; 4_000];
        assert_eq!(sound_strength_db(&z, 1.0), MIN_STRENGTH_DB);
        let s = SoundStrength::from_impulse_response(&z, 1.0, SR);
        assert_eq!(s.g_db, MIN_STRENGTH_DB);
        assert_eq!(s.g_early_db, MIN_STRENGTH_DB);
        assert_eq!(s.g_late_db, MIN_STRENGTH_DB);
    }

    #[test]
    fn non_finite_samples_are_safe() {
        let rir = [Sample::NAN, Sample::INFINITY, 1.0, 0.0];
        let g = sound_strength_db(&rir, 1.0);
        assert!(g.is_finite());
        // Only the finite 1.0 contributes: energy = 1.
        assert!(approx(g, 0.0, 1e-6));
    }

    #[test]
    fn non_positive_reference_is_safe() {
        let rir = exp_response(1_000, 500.0);
        assert_eq!(sound_strength_db(&rir, 0.0), MIN_STRENGTH_DB);
        assert_eq!(sound_strength_db(&rir, -1.0), MIN_STRENGTH_DB);
        assert_eq!(sound_strength_db(&rir, Sample::NAN), MIN_STRENGTH_DB);
    }

    #[test]
    fn non_positive_sample_rate_is_safe() {
        let rir = exp_response(1_000, 500.0);
        assert_eq!(early_sound_strength_db(&rir, 1.0, 0.0), MIN_STRENGTH_DB);
        assert_eq!(late_sound_strength_db(&rir, 1.0, -1.0), MIN_STRENGTH_DB);
        assert_eq!(early_sound_strength_db(&rir, 1.0, Sample::NAN), MIN_STRENGTH_DB);
    }

    #[test]
    fn from_impulse_response_matches_free_functions() {
        let rir = exp_response(8_000, 2_000.0);
        let reference = 1.5;
        let s = SoundStrength::from_impulse_response(&rir, reference, SR);
        assert!(approx(s.g_db, sound_strength_db(&rir, reference), 1e-4));
        assert!(approx(s.g_early_db, early_sound_strength_db(&rir, reference, SR), 1e-4));
        assert!(approx(s.g_late_db, late_sound_strength_db(&rir, reference, SR), 1e-4));
    }

    #[test]
    fn late_strength_of_early_only_response_is_floor() {
        // A response confined to the first 80 ms has no late energy.
        let split = ms_to_index(EARLY_LATE_SPLIT_80_MS, SR, usize::MAX);
        let mut rir = vec![0.0; split + 2_000];
        for x in rir.iter_mut().take(split / 2) {
            *x = 1.0;
        }
        assert_eq!(late_sound_strength_db(&rir, 1.0, SR), MIN_STRENGTH_DB);
        assert!(early_sound_strength_db(&rir, 1.0, SR) > MIN_STRENGTH_DB);
    }

    #[test]
    fn default_is_all_zero() {
        let s = SoundStrength::default();
        assert_eq!(s.g_db, 0.0);
        assert_eq!(s.g_early_db, 0.0);
        assert_eq!(s.g_late_db, 0.0);
    }

    #[test]
    fn early_window_boundary_is_eighty_ms() {
        // Place energy just before and just after the 80 ms boundary.
        let split = ms_to_index(EARLY_LATE_SPLIT_80_MS, SR, usize::MAX);
        let mut rir = vec![0.0; split + 100];
        rir[split - 1] = 1.0;
        rir[split] = 1.0;
        // Early window [..split] catches index split-1; late window [split..]
        // catches index split. Both have energy 1.
        let reference = 1.0;
        assert!(approx(early_sound_strength_db(&rir, reference, SR), 0.0, 1e-6));
        assert!(approx(late_sound_strength_db(&rir, reference, SR), 0.0, 1e-6));
    }

    #[test]
    fn strength_scales_with_reference_in_db() {
        // Halving the reference energy adds 10 log10(2) ~= 3.0103 dB.
        let rir = exp_response(4_000, 1_000.0);
        let g1 = sound_strength_db(&rir, 1.0);
        let g2 = sound_strength_db(&rir, 0.5);
        assert!(approx(g2 - g1, 3.0103, 1e-3), "delta {}", g2 - g1);
    }
}
