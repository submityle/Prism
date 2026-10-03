//! Wet/dry encoding: the direct-to-reverberant ratio of an impulse response
//! and the reverb send gain derived from it.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the "wet/dry ratio -> reverb wet amount" measure of design
//! section 43. The direct-to-reverberant ratio (`DRR`) compares direct energy
//! with the reverberant tail; the wet send gain is a bounded, monotonically
//! decreasing function of that ratio.

use bevy_math::ops;

/// Decibel value returned when the reverberant tail is empty (a perfectly dry
/// path).
pub const MAX_DRR_DB: f32 = 60.0;
/// Decibel value returned when the direct energy is absent (a fully wet path).
pub const MIN_DRR_DB: f32 = -60.0;

/// Direct-to-reverberant ratio in decibels: `10 * log10(direct / reverb)`.
///
/// Degenerate inputs saturate to [`MAX_DRR_DB`] (no reverb) or [`MIN_DRR_DB`]
/// (no direct energy) rather than producing an infinity.
#[must_use]
pub fn drr_db(direct: f32, reverberant: f32) -> f32 {
    if reverberant <= 0.0 {
        return MAX_DRR_DB;
    }
    if direct <= 0.0 {
        return MIN_DRR_DB;
    }
    (10.0 * ops::log10(direct / reverberant)).clamp(MIN_DRR_DB, MAX_DRR_DB)
}

/// Maps a `DRR` in decibels to a wet send gain in `[0, 1]`.
///
/// The map is linear across `[-range_db, +range_db]`: a strongly direct path
/// (`DRR = +range_db`) sends no reverb, an equal split (`DRR = 0`) sends half,
/// and a strongly reverberant path (`DRR = -range_db`) sends full.
#[must_use]
pub fn wet_gain_from_drr(drr_db: f32, range_db: f32) -> f32 {
    let range = range_db.max(1.0e-3);
    (0.5 - 0.5 * drr_db / range).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn drr_saturates_on_degenerate_inputs() {
        assert!(approx(drr_db(1.0, 0.0), MAX_DRR_DB, 1e-6));
        assert!(approx(drr_db(0.0, 1.0), MIN_DRR_DB, 1e-6));
    }

    #[test]
    fn equal_energy_is_zero_db_and_half_wet() {
        assert!(approx(drr_db(2.0, 2.0), 0.0, 1e-5));
        assert!(approx(wet_gain_from_drr(0.0, 20.0), 0.5, 1e-6));
    }

    #[test]
    fn wet_gain_is_monotonically_decreasing_in_drr() {
        let wet_reverberant = wet_gain_from_drr(-20.0, 20.0);
        let wet_balanced = wet_gain_from_drr(0.0, 20.0);
        let wet_direct = wet_gain_from_drr(20.0, 20.0);
        assert!(approx(wet_reverberant, 1.0, 1e-6));
        assert!(approx(wet_direct, 0.0, 1e-6));
        assert!(wet_reverberant > wet_balanced && wet_balanced > wet_direct);
    }
}
