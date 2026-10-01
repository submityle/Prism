//! ISO 3382-1 Annex C stage support parameters (`ST_early` and `ST_late`).
//!
//! Stage support quantifies how well a stage enclosure returns a performer's
//! own sound back to them, which governs the ease of ensemble playing and of
//! hearing oneself. It is measured from a single room impulse response taken
//! with source and receiver `1 m` apart on the stage, and expressed as the
//! decibel ratio of reflected energy in a time window to the direct energy in
//! the first `10 ms`.
//!
//! # Model
//!
//! Let `p[n]` be the stage impulse response and `t = n / sample_rate` the
//! arrival time in seconds. Three energy windows are integrated:
//!
//! - Direct: `E_direct = sum_{0 <= t <= 10 ms} p^2`.
//! - Early:  `E_early  = sum_{20 ms <= t <= 100 ms} p^2`.
//! - Late:   `E_late   = sum_{100 ms < t <= 1000 ms} p^2`.
//!
//! The stage support parameters are the logarithmic ratios:
//!
//! - `ST_early = 10 * log10(E_early / E_direct)` decibels.
//! - `ST_late  = 10 * log10(E_late / E_direct)` decibels.
//!
//! Window boundaries are sample indices `round(ms * 1e-3 * sample_rate)`,
//! clamped to the response length. More reflected energy relative to the
//! direct sound yields a larger (less negative) support value.
//!
//! # Relationship
//!
//! This module complements the listener-side parameters elsewhere in the
//! crate: [`crate::room_clarity`] (clarity, definition, reverberation times),
//! [`crate::center_time`] (energy centre of gravity), and
//! [`crate::sound_strength`] (room gain `G`). Those describe the sound arriving
//! at an audience seat, while stage support describes the sound returned to a
//! performer on stage. All share the [`Sample`] scalar from
//! [`prism_audio_core`] and none reimplements another.
//!
//! # Real-time contract
//!
//! These are control-rate, offline estimators: each accepts a whole impulse
//! response and performs no heap allocation. They are not per-sample callbacks
//! and must not run on an audio thread. They never panic: empty, all-zero,
//! non-finite, zero sample-rate, out-of-range windows, or a non-positive
//! direct-window energy all return the safe sentinel `0`. All logarithmic and
//! length math routes through [`bevy_math::ops`].
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published ISO 3382-1
//! Annex C stage support parameters `ST_early` and `ST_late`. It is pure
//! classic DSP with no AI or ML. It is engine-agnostic and contains **no
//! Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance
//! Audio source or derived code**; it is implemented purely from that publicly
//! documented standard.

use core::f32::consts::LN_10;

use bevy_math::ops;

use prism_audio_core::math::Sample;

/// End of the direct-sound window in milliseconds (`0` to this value).
pub const DIRECT_WINDOW_END_MS: Sample = 10.0;

/// Start of the early-reflection window in milliseconds.
pub const EARLY_WINDOW_START_MS: Sample = 20.0;

/// End of the early-reflection window in milliseconds.
pub const EARLY_WINDOW_END_MS: Sample = 100.0;

/// Start of the late-reflection window in milliseconds.
pub const LATE_WINDOW_START_MS: Sample = 100.0;

/// End of the late-reflection window in milliseconds.
pub const LATE_WINDOW_END_MS: Sample = 1000.0;

/// Safe sentinel in decibels returned when no support information is available
/// (degenerate input or a non-positive direct-window energy).
pub const NO_SUPPORT_DB: Sample = 0.0;

/// Energies below this threshold are treated as silence.
const ENERGY_FLOOR: f64 = 1e-20;

/// Sanitises a sample, mapping non-finite values to `0`.
#[inline]
fn finite(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
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

/// Accumulates the energy `sum p[n]^2` over `response[start..end]` as an `f64`
/// accumulator, ignoring non-finite samples. An empty or inverted range yields
/// `0`.
fn window_energy(response: &[Sample], start: usize, end: usize) -> f64 {
    if start >= end || start >= response.len() {
        return 0.0;
    }
    let hi = end.min(response.len());
    let mut sum = 0.0_f64;
    for &x in &response[start..hi] {
        let v = f64::from(finite(x));
        sum += v * v;
    }
    sum
}

/// Converts a window energy and the direct energy to decibels,
/// `10 * log10(window / direct)`.
///
/// Returns [`NO_SUPPORT_DB`] when the direct energy is below [`ENERGY_FLOOR`]
/// or the ratio is not finite.
fn ratio_to_db(window_energy: f64, direct_energy: f64) -> Sample {
    #[expect(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "negated comparison keeps NaN energies on the safe branch, unlike the suggested direct comparison"
    )]
    if !(direct_energy > ENERGY_FLOOR) {
        return NO_SUPPORT_DB;
    }
    if window_energy <= 0.0 {
        return NO_SUPPORT_DB;
    }
    let db = 10.0 * ops::ln((window_energy / direct_energy) as Sample) / LN_10;
    if db.is_finite() { db } else { NO_SUPPORT_DB }
}

/// Computes the stage support parameter `ST_early` in decibels: the ratio of
/// early-reflection energy (`20 ms` to `100 ms`) to direct energy (`0` to
/// `10 ms`).
///
/// Degenerate inputs (empty, zero sample rate, silent direct window) return
/// [`NO_SUPPORT_DB`].
#[must_use]
pub fn stage_support_early_db(ir: &[Sample], sample_rate: u32) -> Sample {
    if ir.is_empty() || sample_rate == 0 {
        return NO_SUPPORT_DB;
    }
    let sr = sample_rate as Sample;
    let len = ir.len();
    let direct = window_energy(ir, 0, ms_to_index(DIRECT_WINDOW_END_MS, sr, len));
    let early = window_energy(
        ir,
        ms_to_index(EARLY_WINDOW_START_MS, sr, len),
        ms_to_index(EARLY_WINDOW_END_MS, sr, len),
    );
    ratio_to_db(early, direct)
}

/// Computes the stage support parameter `ST_late` in decibels: the ratio of
/// late-reflection energy (`100 ms` to `1000 ms`) to direct energy (`0` to
/// `10 ms`).
///
/// Degenerate inputs (empty, zero sample rate, silent direct window) return
/// [`NO_SUPPORT_DB`].
#[must_use]
pub fn stage_support_late_db(ir: &[Sample], sample_rate: u32) -> Sample {
    if ir.is_empty() || sample_rate == 0 {
        return NO_SUPPORT_DB;
    }
    let sr = sample_rate as Sample;
    let len = ir.len();
    let direct = window_energy(ir, 0, ms_to_index(DIRECT_WINDOW_END_MS, sr, len));
    let late = window_energy(
        ir,
        ms_to_index(LATE_WINDOW_START_MS, sr, len),
        ms_to_index(LATE_WINDOW_END_MS, sr, len),
    );
    ratio_to_db(late, direct)
}

/// The ISO 3382-1 Annex C stage support parameters.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StageSupport {
    /// Early stage support `ST_early` in decibels.
    pub st_early_db: Sample,
    /// Late stage support `ST_late` in decibels.
    pub st_late_db: Sample,
}

impl StageSupport {
    /// Computes both stage support parameters from a stage impulse response at
    /// `sample_rate`. Degenerate inputs return the safe default (both `0`).
    ///
    /// ```
    /// use prism_audio_spatial::stage_support::StageSupport;
    ///
    /// // A direct sound followed by a strong early reflection at 40 ms raises
    /// // ST_early above the silent sentinel.
    /// let sr = 48_000u32;
    /// let mut ir = vec![0.0f32; 48_000];
    /// ir[0] = 1.0;
    /// ir[(0.040 * sr as f32) as usize] = 0.5;
    /// let support = StageSupport::from_impulse_response(&ir, sr);
    /// assert!(support.st_early_db > -40.0);
    /// ```
    #[must_use]
    pub fn from_impulse_response(ir: &[Sample], sample_rate: u32) -> Self {
        if ir.is_empty() || sample_rate == 0 {
            return Self::default();
        }
        let sr = sample_rate as Sample;
        let len = ir.len();
        let direct = window_energy(ir, 0, ms_to_index(DIRECT_WINDOW_END_MS, sr, len));
        let early = window_energy(
            ir,
            ms_to_index(EARLY_WINDOW_START_MS, sr, len),
            ms_to_index(EARLY_WINDOW_END_MS, sr, len),
        );
        let late = window_energy(
            ir,
            ms_to_index(LATE_WINDOW_START_MS, sr, len),
            ms_to_index(LATE_WINDOW_END_MS, sr, len),
        );
        Self {
            st_early_db: ratio_to_db(early, direct),
            st_late_db: ratio_to_db(late, direct),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const SR: u32 = 48_000;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    fn ms_index(ms: Sample) -> usize {
        ops::round(ms / 1000.0 * SR as Sample) as usize
    }

    /// A one-second response with a unit direct spike and optional single
    /// reflections placed in the early and late windows.
    fn staged_ir(early_amp: Sample, late_amp: Sample) -> Vec<Sample> {
        let mut ir = vec![0.0; SR as usize];
        ir[0] = 1.0;
        if early_amp != 0.0 {
            ir[ms_index(40.0)] = early_amp;
        }
        if late_amp != 0.0 {
            ir[ms_index(300.0)] = late_amp;
        }
        ir
    }

    #[test]
    fn window_boundary_constants_are_correct() {
        assert_eq!(DIRECT_WINDOW_END_MS, 10.0);
        assert_eq!(EARLY_WINDOW_START_MS, 20.0);
        assert_eq!(EARLY_WINDOW_END_MS, 100.0);
        assert_eq!(LATE_WINDOW_START_MS, 100.0);
        assert_eq!(LATE_WINDOW_END_MS, 1000.0);
    }

    #[test]
    fn pure_direct_has_no_support() {
        let mut ir = vec![0.0; SR as usize];
        ir[0] = 1.0;
        let support = StageSupport::from_impulse_response(&ir, SR);
        // No early or late energy, so both windows return the sentinel.
        assert_eq!(support.st_early_db, NO_SUPPORT_DB);
        assert_eq!(support.st_late_db, NO_SUPPORT_DB);
    }

    #[test]
    fn strong_early_reflection_raises_st_early() {
        let weak = stage_support_early_db(&staged_ir(0.1, 0.0), SR);
        let strong = stage_support_early_db(&staged_ir(0.5, 0.0), SR);
        assert!(strong > weak, "strong {strong} weak {weak}");
    }

    #[test]
    fn strong_late_reflection_raises_st_late() {
        let weak = stage_support_late_db(&staged_ir(0.0, 0.1), SR);
        let strong = stage_support_late_db(&staged_ir(0.0, 0.5), SR);
        assert!(strong > weak, "strong {strong} weak {weak}");
    }

    #[test]
    fn known_energy_ratio_matches_db() {
        // Direct spike of amplitude 1 (energy 1), early reflection of amplitude
        // 0.5 (energy 0.25) => ST_early = 10 * log10(0.25) = -6.0206 dB.
        let st_early = stage_support_early_db(&staged_ir(0.5, 0.0), SR);
        assert!(approx(st_early, -6.020_6, 1e-3), "st_early {st_early}");
        // Late reflection of amplitude 0.25 (energy 0.0625) =>
        // 10 * log10(0.0625) = -12.0412 dB.
        let st_late = stage_support_late_db(&staged_ir(0.0, 0.25), SR);
        assert!(approx(st_late, -12.041_2, 1e-3), "st_late {st_late}");
    }

    #[test]
    fn empty_ir_is_default() {
        let empty: [Sample; 0] = [];
        assert_eq!(stage_support_early_db(&empty, SR), NO_SUPPORT_DB);
        assert_eq!(stage_support_late_db(&empty, SR), NO_SUPPORT_DB);
        assert_eq!(
            StageSupport::from_impulse_response(&empty, SR),
            StageSupport::default()
        );
    }

    #[test]
    fn all_zero_ir_is_sentinel() {
        let ir = vec![0.0; SR as usize];
        let support = StageSupport::from_impulse_response(&ir, SR);
        assert_eq!(support.st_early_db, NO_SUPPORT_DB);
        assert_eq!(support.st_late_db, NO_SUPPORT_DB);
    }

    #[test]
    fn zero_sample_rate_is_sentinel() {
        let ir = staged_ir(0.5, 0.3);
        assert_eq!(stage_support_early_db(&ir, 0), NO_SUPPORT_DB);
        assert_eq!(stage_support_late_db(&ir, 0), NO_SUPPORT_DB);
        assert_eq!(
            StageSupport::from_impulse_response(&ir, 0),
            StageSupport::default()
        );
    }

    #[test]
    fn non_finite_samples_are_safe() {
        let mut ir = staged_ir(0.5, 0.3);
        ir[0] = Sample::NAN;
        ir[ms_index(40.0)] = Sample::INFINITY;
        ir[ms_index(300.0)] = Sample::NEG_INFINITY;
        let support = StageSupport::from_impulse_response(&ir, SR);
        assert!(support.st_early_db.is_finite());
        assert!(support.st_late_db.is_finite());
    }

    #[test]
    fn short_ir_within_direct_window_is_safe() {
        // Only a few samples: all windows beyond the direct one are empty.
        let ir = vec![1.0, 0.5, 0.25];
        let support = StageSupport::from_impulse_response(&ir, SR);
        assert_eq!(support.st_early_db, NO_SUPPORT_DB);
        assert_eq!(support.st_late_db, NO_SUPPORT_DB);
    }

    #[test]
    fn silent_direct_window_is_sentinel() {
        // Energy only in the early window, no direct energy to normalise by.
        let mut ir = vec![0.0; SR as usize];
        ir[ms_index(40.0)] = 0.5;
        assert_eq!(stage_support_early_db(&ir, SR), NO_SUPPORT_DB);
    }

    #[test]
    fn from_ir_matches_free_functions() {
        let ir = staged_ir(0.4, 0.2);
        let support = StageSupport::from_impulse_response(&ir, SR);
        assert!(approx(support.st_early_db, stage_support_early_db(&ir, SR), 1e-6));
        assert!(approx(support.st_late_db, stage_support_late_db(&ir, SR), 1e-6));
    }

    #[test]
    fn default_is_zero() {
        let d = StageSupport::default();
        assert_eq!(d.st_early_db, 0.0);
        assert_eq!(d.st_late_db, 0.0);
    }

    #[test]
    fn louder_direct_lowers_support() {
        // Doubling the direct amplitude (quadrupling direct energy) lowers the
        // early support by 10 * log10(4) = 6.0206 dB.
        let base = stage_support_early_db(&staged_ir(0.5, 0.0), SR);
        let mut loud = staged_ir(0.5, 0.0);
        loud[0] = 2.0;
        let louder = stage_support_early_db(&loud, SR);
        assert!(approx(base - louder, 6.020_6, 1e-3), "base {base} louder {louder}");
    }
}
