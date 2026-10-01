//! ISO 3382-1 late lateral sound level `LJ` from a measured lateral response.
//!
//! The late lateral sound level quantifies the strength of late-arriving sound
//! that reaches a listener from the sides, which governs listener envelopment
//! (`LEV`): the sensation of being surrounded by the reverberant field. It is
//! measured from a figure-of-eight (lateral) impulse response whose null points
//! along the source-listener axis, and is expressed as the decibel ratio of its
//! late energy to the same free-field reference energy used by sound strength.
//!
//! # Model
//!
//! Let `pL[n]` be the lateral figure-of-eight impulse response and
//! `t = n / sample_rate` the arrival time in seconds. Let `E_ref` be the
//! reference energy the source produces in a free field at `10 m` (supplied by
//! the caller, identical in meaning to the reference energy of
//! [`crate::sound_strength`]). With the late window starting at `80 ms`:
//!
//! - `LJ = 10 * log10(sum_{80 ms < t <= end} pL^2 / E_ref)` decibels.
//!
//! Larger `LJ` means more late lateral energy relative to the free field, hence
//! a stronger sense of envelopment. The `80 ms` boundary is shared with
//! [`crate::room_clarity`] via [`crate::room_clarity::EARLY_LATE_SPLIT_80_MS`].
//!
//! # Relationship
//!
//! This module complements the other spatial parameters in the crate without
//! duplicating them:
//!
//! - [`crate::spatial_impression`] measures the early (`0` to `80 ms`) lateral
//!   energy *fraction* (`LF`/`LFC`, dimensionless `0` to `1`) relating to
//!   apparent source width.
//! - [`crate::sound_strength`] measures *omnidirectional* level `G` in decibels
//!   relative to the free field, with no directivity.
//! - This module measures the *late lateral* energy *level* in decibels,
//!   relating to listener envelopment.
//!
//! The window, dimensionality, and perceptual meaning differ in each case. All
//! share the [`Sample`] scalar from [`prism_audio_core`], reuse the `80 ms`
//! split constant from [`crate::room_clarity`], and share the free-field
//! reference-energy convention of [`crate::sound_strength`]; none reimplements
//! another.
//!
//! # Real-time contract
//!
//! This is a control-rate, offline estimator: it accepts a whole impulse
//! response and performs no heap allocation. It is not a per-sample callback and
//! must not run on an audio thread. It never panics: empty, all-zero,
//! non-finite, non-positive reference energy, or non-positive sample-rate
//! inputs return the safe sentinel [`NO_LATE_LATERAL_DB`]. All logarithmic and
//! length math routes through [`bevy_math::ops`].
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published ISO 3382-1 late
//! lateral sound level parameter `LJ`. It is pure classic DSP with no AI or ML.
//! It is engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise,
//! FMOD, Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented standard.

use core::f32::consts::LN_10;

use bevy_math::ops;

use prism_audio_core::math::Sample;

use crate::room_clarity::EARLY_LATE_SPLIT_80_MS;

/// Start of the late lateral window in milliseconds. Shared with the `80 ms`
/// early-to-late split of [`crate::room_clarity`].
pub const LATE_LATERAL_START_MS: Sample = EARLY_LATE_SPLIT_80_MS;

/// Safe sentinel in decibels returned when no late lateral level is available
/// (degenerate input, a silent late window, or a non-positive reference
/// energy), avoiding negative infinity from the logarithm.
pub const NO_LATE_LATERAL_DB: Sample = -100.0;

/// Energies below this threshold are treated as silence.
const ENERGY_FLOOR: Sample = 1e-20;

/// Sanitises a sample, mapping non-finite values to `0`.
#[inline]
fn finite(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
}

/// Accumulates the energy `sum pL[n]^2` over a slice as an `f64` accumulator,
/// ignoring non-finite samples.
fn energy(slice: &[Sample]) -> f64 {
    let mut sum = 0.0_f64;
    for &x in slice {
        let v = f64::from(finite(x));
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

/// Converts a late lateral energy and the free-field reference energy to
/// decibels, `10 * log10(energy / reference)`.
///
/// Returns [`NO_LATE_LATERAL_DB`] when the reference energy is not strictly
/// positive, the late energy is below [`ENERGY_FLOOR`], or the ratio is not
/// finite.
fn energy_to_db(late_energy: f64, reference_energy: Sample) -> Sample {
    if !reference_energy.is_finite() || reference_energy <= 0.0 {
        return NO_LATE_LATERAL_DB;
    }
    #[expect(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "negated comparison keeps NaN energies on the safe branch, unlike the suggested direct comparison"
    )]
    if !(late_energy > f64::from(ENERGY_FLOOR)) {
        return NO_LATE_LATERAL_DB;
    }
    let ratio = (late_energy / f64::from(reference_energy)) as Sample;
    let db = 10.0 * ops::ln(ratio) / LN_10;
    if db.is_finite() { db } else { NO_LATE_LATERAL_DB }
}

/// Computes the late lateral sound level `LJ` in decibels from a lateral
/// figure-of-eight impulse response.
///
/// `LJ = 10 * log10(sum_{80 ms < t <= end} pL^2 / reference_energy)`, where
/// `reference_energy` is the free-field reference energy of the same source at
/// `10 m` (see [`crate::sound_strength`]). An empty or silent late window, a
/// non-positive or non-finite `reference_energy`, or a non-positive or
/// non-finite `sample_rate` returns [`NO_LATE_LATERAL_DB`].
#[must_use]
pub fn late_lateral_sound_level_db(
    figure_eight: &[Sample],
    reference_energy: Sample,
    sample_rate: Sample,
) -> Sample {
    if figure_eight.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return NO_LATE_LATERAL_DB;
    }
    let split = ms_to_index(LATE_LATERAL_START_MS, sample_rate, figure_eight.len());
    energy_to_db(energy(&figure_eight[split..]), reference_energy)
}

/// The ISO 3382-1 late lateral sound level `LJ` of a measured lateral response.
///
/// The field is in decibels relative to the free-field reference energy at
/// `10 m`. A silent late window reports [`NO_LATE_LATERAL_DB`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LateLateralSoundLevel {
    /// Late lateral sound level `LJ` in decibels.
    pub lj_db: Sample,
}

impl LateLateralSoundLevel {
    /// Computes the late lateral sound level from a lateral figure-of-eight
    /// impulse response at `sample_rate`.
    ///
    /// `reference_energy` is the free-field reference energy of the same source
    /// at `10 m`. Degenerate inputs return the safe default
    /// ([`NO_LATE_LATERAL_DB`]).
    ///
    /// ```
    /// use prism_audio_spatial::late_lateral_sound_level::LateLateralSoundLevel;
    ///
    /// // A lateral response with late energy after 80 ms yields a finite LJ.
    /// let sr = 48_000.0f32;
    /// let mut pl = vec![0.0f32; 48_000];
    /// pl[(0.120 * sr) as usize] = 0.5;
    /// let lj = LateLateralSoundLevel::from_responses(&pl, 1.0, sr);
    /// assert!(lj.lj_db > -40.0);
    /// assert!(lj.lj_db.is_finite());
    /// ```
    #[must_use]
    pub fn from_responses(
        figure_eight: &[Sample],
        reference_energy: Sample,
        sample_rate: Sample,
    ) -> Self {
        Self {
            lj_db: late_lateral_sound_level_db(figure_eight, reference_energy, sample_rate),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const SR: Sample = 48_000.0;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    fn ms_index(ms: Sample) -> usize {
        ops::round(ms / 1000.0 * SR) as usize
    }

    /// A one-second lateral response with an optional single reflection placed
    /// in the late window (after `80 ms`).
    fn late_lateral_ir(late_amp: Sample) -> Vec<Sample> {
        let mut pl = vec![0.0; SR as usize];
        if late_amp != 0.0 {
            pl[ms_index(150.0)] = late_amp;
        }
        pl
    }

    #[test]
    fn late_lateral_start_is_eighty_ms() {
        assert_eq!(LATE_LATERAL_START_MS, 80.0);
        assert_eq!(LATE_LATERAL_START_MS, EARLY_LATE_SPLIT_80_MS);
    }

    #[test]
    fn known_energy_ratio_matches_db() {
        // Late reflection of amplitude 0.5 (energy 0.25) against reference 1.0
        // => LJ = 10 * log10(0.25) = -6.0206 dB.
        let lj = late_lateral_sound_level_db(&late_lateral_ir(0.5), 1.0, SR);
        assert!(approx(lj, -6.020_6, 1e-3), "lj {lj}");
        // Amplitude 0.25 (energy 0.0625) => 10 * log10(0.0625) = -12.0412 dB.
        let lj2 = late_lateral_sound_level_db(&late_lateral_ir(0.25), 1.0, SR);
        assert!(approx(lj2, -12.041_2, 1e-3), "lj2 {lj2}");
    }

    #[test]
    fn stronger_late_lateral_raises_lj() {
        let weak = late_lateral_sound_level_db(&late_lateral_ir(0.1), 1.0, SR);
        let strong = late_lateral_sound_level_db(&late_lateral_ir(0.5), 1.0, SR);
        assert!(strong > weak, "strong {strong} weak {weak}");
    }

    #[test]
    fn larger_reference_lowers_lj() {
        let pl = late_lateral_ir(0.5);
        assert!(
            late_lateral_sound_level_db(&pl, 10.0, SR)
                < late_lateral_sound_level_db(&pl, 1.0, SR)
        );
    }

    #[test]
    fn reference_scales_in_db() {
        // Halving the reference energy adds 10 log10(2) ~= 3.0103 dB.
        let pl = late_lateral_ir(0.5);
        let lj1 = late_lateral_sound_level_db(&pl, 1.0, SR);
        let lj2 = late_lateral_sound_level_db(&pl, 0.5, SR);
        assert!(approx(lj2 - lj1, 3.0103, 1e-3), "delta {}", lj2 - lj1);
    }

    #[test]
    fn empty_response_is_sentinel() {
        let empty: [Sample; 0] = [];
        assert_eq!(
            late_lateral_sound_level_db(&empty, 1.0, SR),
            NO_LATE_LATERAL_DB
        );
        assert_eq!(
            LateLateralSoundLevel::from_responses(&empty, 1.0, SR).lj_db,
            NO_LATE_LATERAL_DB
        );
    }

    #[test]
    fn all_zero_response_is_sentinel() {
        let pl = vec![0.0; SR as usize];
        assert_eq!(
            late_lateral_sound_level_db(&pl, 1.0, SR),
            NO_LATE_LATERAL_DB
        );
    }

    #[test]
    fn zero_sample_rate_is_sentinel() {
        let pl = late_lateral_ir(0.5);
        assert_eq!(late_lateral_sound_level_db(&pl, 1.0, 0.0), NO_LATE_LATERAL_DB);
        assert_eq!(
            late_lateral_sound_level_db(&pl, 1.0, -1.0),
            NO_LATE_LATERAL_DB
        );
        assert_eq!(
            late_lateral_sound_level_db(&pl, 1.0, Sample::NAN),
            NO_LATE_LATERAL_DB
        );
    }

    #[test]
    fn non_positive_reference_is_sentinel() {
        let pl = late_lateral_ir(0.5);
        assert_eq!(late_lateral_sound_level_db(&pl, 0.0, SR), NO_LATE_LATERAL_DB);
        assert_eq!(late_lateral_sound_level_db(&pl, -1.0, SR), NO_LATE_LATERAL_DB);
        assert_eq!(
            late_lateral_sound_level_db(&pl, Sample::NAN, SR),
            NO_LATE_LATERAL_DB
        );
    }

    #[test]
    fn non_finite_samples_are_safe() {
        let mut pl = late_lateral_ir(0.5);
        pl[0] = Sample::NAN;
        pl[ms_index(150.0)] = Sample::INFINITY;
        let lj = late_lateral_sound_level_db(&pl, 1.0, SR);
        // The infinite late sample is dropped, leaving a silent late window.
        assert_eq!(lj, NO_LATE_LATERAL_DB);
    }

    #[test]
    fn early_only_response_has_no_late_lateral() {
        // Energy confined to the first 80 ms leaves the late window silent.
        let split = ms_to_index(LATE_LATERAL_START_MS, SR, usize::MAX);
        let mut pl = vec![0.0; split + 2_000];
        for x in pl.iter_mut().take(split / 2) {
            *x = 1.0;
        }
        assert_eq!(late_lateral_sound_level_db(&pl, 1.0, SR), NO_LATE_LATERAL_DB);
    }

    #[test]
    fn late_window_boundary_is_eighty_ms() {
        // Place energy just before and just after the 80 ms boundary.
        let split = ms_to_index(LATE_LATERAL_START_MS, SR, usize::MAX);
        let mut pl = vec![0.0; split + 100];
        pl[split - 1] = 1.0; // early side, excluded
        pl[split] = 1.0; // late side, included
        // Only the index at split contributes to the late window (energy 1).
        assert!(approx(late_lateral_sound_level_db(&pl, 1.0, SR), 0.0, 1e-6));
    }

    #[test]
    fn short_ir_within_early_window_is_sentinel() {
        let pl = vec![1.0, 0.5, 0.25];
        assert_eq!(late_lateral_sound_level_db(&pl, 1.0, SR), NO_LATE_LATERAL_DB);
    }

    #[test]
    fn from_responses_matches_free_function() {
        let pl = late_lateral_ir(0.4);
        let lj = LateLateralSoundLevel::from_responses(&pl, 1.5, SR);
        assert!(approx(lj.lj_db, late_lateral_sound_level_db(&pl, 1.5, SR), 1e-6));
    }

    #[test]
    fn default_is_zero() {
        let d = LateLateralSoundLevel::default();
        assert_eq!(d.lj_db, 0.0);
    }
}
