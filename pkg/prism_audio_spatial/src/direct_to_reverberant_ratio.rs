//! Direct-to-reverberant ratio `DRR` from a measured room impulse response
//! (Zahorik).
//!
//! The direct-to-reverberant ratio compares the energy of the direct sound
//! path to the energy of everything that arrives afterwards. It is a primary
//! cue for auditory distance perception: a high ratio makes a source sound
//! near, while a low ratio makes it sound far away inside a reverberant space.
//! The ratio is expressed in decibels.
//!
//! # Model
//!
//! Let `p[n]` be the broadband room impulse response and `sample_rate` its
//! sampling rate in hertz. The direct-sound index `n_d` is the first sample
//! index that attains the global peak magnitude. A narrow symmetric window of
//! half-width `w = round(DIRECT_WINDOW_MS / 1000 * sample_rate)` samples around
//! `n_d` isolates the direct path:
//!
//! - `E_direct = sum_{n_d - w <= n <= n_d + w} p^2`,
//! - `E_total  = sum_n p^2`,
//! - `E_reverberant = E_total - E_direct`.
//!
//! The ratio is `DRR = 10 * log10(E_direct / E_reverberant)` decibels, clamped
//! to `[-MAX_DRR_DB, MAX_DRR_DB]`.
//!
//! # Relationship
//!
//! This module complements [`crate::room_clarity`] and
//! [`crate::useful_to_detrimental_ratio`] without duplicating them. `C50` and
//! `C80` and `U50` and `U80` split the response at a fixed `50` or `80` ms
//! boundary measured from time zero and quantify clarity and intelligibility.
//! The direct-to-reverberant ratio instead centres a very narrow window
//! (`DIRECT_WINDOW_MS`, around `2.5` ms) on the detected direct-sound peak to
//! isolate the direct path for distance perception, so its window position and
//! purpose differ fundamentally and its energy integration is not shared.
//! [`crate::center_time`] reports the energy centre of gravity and
//! [`crate::initial_time_delay_gap`] reports the gap to the first reflection;
//! both are distinct single-number parameters. The statistical, geometry-based
//! direct-to-reverberant ratio on [`crate::reverberant_field::ReverberantField`]
//! is a theoretical function of source distance and directivity, whereas this
//! module measures the ratio directly from a recorded impulse response. All
//! share the [`Sample`] scalar from [`prism_audio_core`].
//!
//! # Real-time contract
//!
//! This is a control-rate, offline estimator: it accepts a whole impulse
//! response and performs no heap allocation. It is not a per-sample callback
//! and must not run on an audio thread. It never panics: empty, all-zero,
//! non-finite, or non-positive sample-rate inputs return the safe sentinel
//! [`NO_DRR_DB`]. All logarithmic and length math routes through
//! [`bevy_math::ops`].
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published direct-to-
//! reverberant ratio distance cue (Zahorik). It is pure classic DSP with no AI
//! or ML. It is engine-agnostic and contains **no Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**; it is implemented purely from that publicly documented acoustics
//! literature.

use core::f32::consts::LN_10;

use bevy_math::ops;

use prism_audio_core::math::Sample;

/// Half-width, in milliseconds, of the window centred on the direct-sound peak
/// that isolates the direct path from the reverberant tail.
pub const DIRECT_WINDOW_MS: Sample = 2.5;

/// Safe sentinel in decibels returned when no direct-to-reverberant ratio is
/// available (degenerate input, a silent response, or a non-positive
/// sample-rate), avoiding negative infinity from the logarithm.
pub const NO_DRR_DB: Sample = -100.0;

/// Maximum reported ratio magnitude in decibels. A response with no measurable
/// reverberant energy clamps to this value instead of producing infinity.
pub const MAX_DRR_DB: Sample = 100.0;

/// Energies below this threshold are treated as silence.
const ENERGY_FLOOR: Sample = 1e-20;

/// Returns a sample, mapping non-finite values to `0`.
#[inline]
fn finite(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
}

/// Returns the magnitude of a sample, mapping non-finite values to `0`.
#[inline]
fn finite_abs(x: Sample) -> Sample {
    if x.is_finite() { x.abs() } else { 0.0 }
}

/// Accumulates the energy `sum p[n]^2` over a slice as an `f64` accumulator,
/// ignoring non-finite samples.
fn energy(slice: &[Sample]) -> f64 {
    let mut sum = 0.0_f64;
    for &x in slice {
        let v = f64::from(finite(x));
        sum += v * v;
    }
    sum
}

/// Converts the direct half-window from milliseconds to a sample count,
/// returning `0` for non-finite or non-positive rates.
fn window_half_samples(sample_rate: Sample) -> usize {
    let x = DIRECT_WINDOW_MS / 1000.0 * sample_rate;
    if !x.is_finite() || x < 0.0 {
        return 0;
    }
    ops::round(x) as usize
}

/// Computes the direct-to-reverberant ratio and the detected direct-sound
/// index. Degenerate inputs return `(NO_DRR_DB, 0)`.
fn compute(ir: &[Sample], sample_rate: Sample) -> (Sample, usize) {
    if ir.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return (NO_DRR_DB, 0);
    }

    // Direct sound: the first index attaining the global peak magnitude.
    let mut peak = 0.0;
    let mut direct_index = 0usize;
    for (i, &x) in ir.iter().enumerate() {
        let mag = finite_abs(x);
        if mag > peak {
            peak = mag;
            direct_index = i;
        }
    }
    if peak <= 0.0 {
        return (NO_DRR_DB, 0);
    }

    let len = ir.len();
    let half = window_half_samples(sample_rate);
    let lo = direct_index.saturating_sub(half);
    let hi = direct_index.saturating_add(half).saturating_add(1).min(len);

    let direct = energy(&ir[lo..hi]);
    let total = energy(ir);
    // total is a finite non-negative energy sum, so <= is a clean guard.
    if total <= f64::from(ENERGY_FLOOR) {
        return (NO_DRR_DB, direct_index);
    }
    let reverberant = (total - direct).max(0.0);
    // reverberant is a finite non-negative energy, so <= is a clean guard.
    if reverberant <= f64::from(ENERGY_FLOOR) {
        // No measurable reverberant energy: clamp to the maximum ratio.
        return (MAX_DRR_DB, direct_index);
    }
    let ratio = (direct / reverberant) as Sample;
    let db = 10.0 * ops::ln(ratio) / LN_10;
    if !db.is_finite() {
        return (NO_DRR_DB, direct_index);
    }
    (db.clamp(-MAX_DRR_DB, MAX_DRR_DB), direct_index)
}

/// Computes the direct-to-reverberant ratio `DRR` in decibels from a broadband
/// room impulse response.
///
/// The direct-sound peak is located, a window of half-width
/// [`DIRECT_WINDOW_MS`] milliseconds around it isolates the direct energy, and
/// `DRR = 10 * log10(E_direct / E_reverberant)` is returned, clamped to
/// `[-MAX_DRR_DB, MAX_DRR_DB]`. Empty, all-zero, non-finite, or non-positive
/// sample-rate inputs return [`NO_DRR_DB`]; a response with no reverberant
/// energy returns [`MAX_DRR_DB`].
#[must_use]
pub fn direct_to_reverberant_ratio_db(ir: &[Sample], sample_rate: Sample) -> Sample {
    compute(ir, sample_rate).0
}

/// The direct-to-reverberant ratio of a measured room impulse response.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DirectToReverberantRatio {
    /// Direct-to-reverberant ratio in decibels. Degenerate inputs report
    /// [`NO_DRR_DB`].
    pub drr_db: Sample,
    /// Sample index of the detected direct-sound peak.
    pub direct_arrival_index: usize,
}

impl DirectToReverberantRatio {
    /// Computes the direct-to-reverberant ratio from a broadband room impulse
    /// response sampled at `sample_rate` hertz.
    ///
    /// Degenerate inputs report [`NO_DRR_DB`] with a direct index of `0`.
    ///
    /// ```
    /// use prism_audio_spatial::direct_to_reverberant_ratio::DirectToReverberantRatio;
    ///
    /// // A strong direct hit with a weak reverberant tail reads near (high ratio).
    /// let sr = 48_000.0f32;
    /// let mut ir = vec![0.0f32; 48_000];
    /// ir[0] = 1.0;
    /// ir[24_000] = 0.1;
    /// let drr = DirectToReverberantRatio::from_impulse_response(&ir, sr);
    /// assert!(drr.drr_db > 0.0);
    /// assert_eq!(drr.direct_arrival_index, 0);
    /// ```
    #[must_use]
    pub fn from_impulse_response(ir: &[Sample], sample_rate: Sample) -> Self {
        let (drr_db, direct_arrival_index) = compute(ir, sample_rate);
        Self {
            drr_db,
            direct_arrival_index,
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

    /// A one-second response with a direct hit at index `0` and one reverberant
    /// sample placed well beyond the direct window.
    fn ir_with_reverb(direct: Sample, reverb_amp: Sample, reverb_index: usize) -> Vec<Sample> {
        let mut ir = vec![0.0; SR as usize];
        ir[0] = direct;
        if reverb_amp != 0.0 {
            ir[reverb_index] = reverb_amp;
        }
        ir
    }

    #[test]
    fn known_ratio_matches_db() {
        // Direct energy 1.0, reverberant energy 0.25:
        // DRR = 10 log10(1 / 0.25) = 6.0206 dB.
        let ir = ir_with_reverb(1.0, 0.5, 20_000);
        let drr = direct_to_reverberant_ratio_db(&ir, SR);
        assert!(approx(drr, 6.020_6, 1e-3), "drr {drr}");
    }

    #[test]
    fn direct_peak_located_correctly() {
        let mut ir = vec![0.0; SR as usize];
        ir[1000] = 0.8;
        ir[5000] = 1.0; // global peak
        ir[20_000] = 0.3;
        let d = DirectToReverberantRatio::from_impulse_response(&ir, SR);
        assert_eq!(d.direct_arrival_index, 5000);
    }

    #[test]
    fn window_width_scales_with_sample_rate() {
        // A reflection 100 samples after the peak is inside the 2.5 ms window
        // at 48 kHz (120 samples) but outside it at 8 kHz (20 samples).
        let mut ir = vec![0.0; 48_000];
        ir[0] = 1.0;
        ir[100] = 0.5;
        let near = direct_to_reverberant_ratio_db(&ir, 48_000.0);
        let far = direct_to_reverberant_ratio_db(&ir, 8_000.0);
        // At 48 kHz the reflection counts as direct, so no reverberant energy.
        assert!(approx(near, MAX_DRR_DB, 1e-6), "near {near}");
        // At 8 kHz the reflection counts as reverberant, giving a finite ratio.
        assert!(far.is_finite() && far < MAX_DRR_DB, "far {far}");
    }

    #[test]
    fn all_within_direct_window_clamps_max() {
        // A single direct impulse has no reverberant energy at all.
        let mut ir = vec![0.0; SR as usize];
        ir[10] = 1.0;
        let drr = direct_to_reverberant_ratio_db(&ir, SR);
        assert!(approx(drr, MAX_DRR_DB, 1e-6), "drr {drr}");
    }

    #[test]
    fn more_reverberant_energy_lowers_ratio() {
        let low_reverb = ir_with_reverb(1.0, 0.2, 20_000);
        let high_reverb = ir_with_reverb(1.0, 0.8, 20_000);
        let a = direct_to_reverberant_ratio_db(&low_reverb, SR);
        let b = direct_to_reverberant_ratio_db(&high_reverb, SR);
        assert!(a > b, "a {a} b {b}");
    }

    #[test]
    fn reflection_just_inside_window_is_direct() {
        // 2.5 ms at 48 kHz is 120 samples; index 120 is inside the window.
        let mut ir = vec![0.0; SR as usize];
        ir[0] = 1.0;
        ir[120] = 0.5;
        let drr = direct_to_reverberant_ratio_db(&ir, SR);
        assert!(approx(drr, MAX_DRR_DB, 1e-6), "drr {drr}");
    }

    #[test]
    fn empty_response_is_sentinel() {
        let empty: [Sample; 0] = [];
        assert_eq!(direct_to_reverberant_ratio_db(&empty, SR), NO_DRR_DB);
        assert_eq!(
            DirectToReverberantRatio::from_impulse_response(&empty, SR).drr_db,
            NO_DRR_DB
        );
    }

    #[test]
    fn all_zero_response_is_sentinel() {
        let ir = vec![0.0; SR as usize];
        assert_eq!(direct_to_reverberant_ratio_db(&ir, SR), NO_DRR_DB);
    }

    #[test]
    fn zero_sample_rate_is_sentinel() {
        let ir = ir_with_reverb(1.0, 0.5, 20_000);
        assert_eq!(direct_to_reverberant_ratio_db(&ir, 0.0), NO_DRR_DB);
    }

    #[test]
    fn nan_sample_rate_is_sentinel() {
        let ir = ir_with_reverb(1.0, 0.5, 20_000);
        assert_eq!(direct_to_reverberant_ratio_db(&ir, Sample::NAN), NO_DRR_DB);
    }

    #[test]
    fn non_finite_samples_are_safe() {
        let mut ir = ir_with_reverb(1.0, 0.5, 20_000);
        ir[5] = Sample::NAN;
        ir[30_000] = Sample::INFINITY;
        let drr = direct_to_reverberant_ratio_db(&ir, SR);
        assert!(drr.is_finite());
    }

    #[test]
    fn from_impulse_response_matches_free_function() {
        let ir = ir_with_reverb(1.0, 0.4, 20_000);
        let d = DirectToReverberantRatio::from_impulse_response(&ir, SR);
        assert!(approx(d.drr_db, direct_to_reverberant_ratio_db(&ir, SR), 1e-6));
    }

    #[test]
    fn default_is_zero() {
        let d = DirectToReverberantRatio::default();
        assert_eq!(d.drr_db, 0.0);
        assert_eq!(d.direct_arrival_index, 0);
    }

    #[test]
    fn direct_window_ms_constant_is_stable() {
        assert_eq!(DIRECT_WINDOW_MS, 2.5);
        assert_eq!(NO_DRR_DB, -100.0);
        assert_eq!(MAX_DRR_DB, 100.0);
    }

    #[test]
    fn result_is_clamped_to_range() {
        let ir = ir_with_reverb(1.0, 0.4, 20_000);
        let drr = direct_to_reverberant_ratio_db(&ir, SR);
        assert!((-MAX_DRR_DB..=MAX_DRR_DB).contains(&drr), "drr {drr}");
    }
}
