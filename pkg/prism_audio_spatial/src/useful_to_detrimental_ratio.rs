//! Useful-to-detrimental sound ratio `U50`/`U80` from a measured impulse
//! response (Bradley).
//!
//! The useful-to-detrimental sound ratio extends the clarity measures `C50`
//! and `C80` by placing a background-noise term in the denominator. Early
//! arriving sound is treated as useful for intelligibility, while late sound
//! together with steady background noise is treated as detrimental. The ratio
//! is expressed in decibels for a speech split of `50 ms` (`U50`) and a music
//! split of `80 ms` (`U80`).
//!
//! # Model
//!
//! Let `p[n]` be the broadband room impulse response and `t = n / sample_rate`
//! the arrival time in seconds. For a split time `t_s` (`50 ms` or `80 ms`),
//! with early, late, and total energies
//!
//! - `E_early = sum_{0 <= t < t_s} p^2`,
//! - `E_late  = sum_{t_s <= t <= end} p^2`,
//! - `E_total = E_early + E_late`,
//!
//! the detrimental background-noise energy for a signal-to-noise ratio
//! `snr_db` is `E_noise = E_total * 10^(-snr_db / 10)`. The ratio is
//!
//! - `U = 10 * log10(E_early / (E_late + E_noise))` decibels,
//!
//! clamped to `[-MAX_CLARITY_DB, MAX_CLARITY_DB]` using
//! [`crate::room_clarity::MAX_CLARITY_DB`]. A non-finite `snr_db` is treated as
//! an infinite signal-to-noise ratio, giving `E_noise = 0` and recovering the
//! noiseless clarity value `C_t`.
//!
//! # Relationship
//!
//! This module complements [`crate::room_clarity`] without duplicating it.
//! `C50`/`C80` are the noiseless early-to-late energy ratios; `U50`/`U80` add
//! the background-noise term `E_noise` to the detrimental denominator. As
//! `snr_db` tends to positive infinity, `E_noise` tends to `0` and `U_t`
//! converges to the identically clamped `C_t` from
//! [`crate::room_clarity::clarity_db`]. The early-to-late split constants
//! [`crate::room_clarity::EARLY_LATE_SPLIT_50_MS`] and
//! [`crate::room_clarity::EARLY_LATE_SPLIT_80_MS`] and the clamp
//! [`crate::room_clarity::MAX_CLARITY_DB`] are reused; the noiseless energy
//! integration is not reimplemented. All share the [`Sample`] scalar from
//! [`prism_audio_core`].
//!
//! # Real-time contract
//!
//! This is a control-rate, offline estimator: it accepts a whole impulse
//! response and performs no heap allocation. It is not a per-sample callback
//! and must not run on an audio thread. It never panics: empty, all-zero,
//! non-finite, or non-positive sample-rate inputs return the safe sentinel
//! [`NO_USEFUL_RATIO_DB`]. All logarithmic, exponential, and length math routes
//! through [`bevy_math::ops`].
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published Bradley (1986)
//! useful-to-detrimental sound ratio. It is pure classic DSP with no AI or ML.
//! It is engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise,
//! FMOD, Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented acoustics literature.

use core::f32::consts::LN_10;

use bevy_math::ops;

use prism_audio_core::math::Sample;

use crate::room_clarity::{EARLY_LATE_SPLIT_50_MS, EARLY_LATE_SPLIT_80_MS, MAX_CLARITY_DB};

/// Safe sentinel in decibels returned when no useful-to-detrimental ratio is
/// available (degenerate input, a silent response, or a non-positive
/// sample-rate), avoiding negative infinity from the logarithm.
pub const NO_USEFUL_RATIO_DB: Sample = -100.0;

/// Energies below this threshold are treated as silence.
const ENERGY_FLOOR: Sample = 1e-20;

/// Sanitises a sample, mapping non-finite values to `0`.
#[inline]
fn finite(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
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

/// Converts a time in milliseconds to a sample index, clamped to `[0, len]`.
fn ms_to_index(ms: Sample, sample_rate: Sample, len: usize) -> usize {
    let x = ms / 1000.0 * sample_rate;
    if !x.is_finite() || x <= 0.0 {
        return 0;
    }
    let idx = ops::round(x) as usize;
    idx.min(len)
}

/// Computes the useful-to-detrimental sound ratio `U` in decibels from a
/// broadband room impulse response.
///
/// `U = 10 * log10(E_early / (E_late + E_noise))` with
/// `E_noise = E_total * 10^(-snr_db / 10)`, split at `split_ms` milliseconds.
/// Pass [`crate::room_clarity::EARLY_LATE_SPLIT_50_MS`] for `U50` or
/// [`crate::room_clarity::EARLY_LATE_SPLIT_80_MS`] for `U80`. A non-finite
/// `snr_db` is treated as noiseless (`E_noise = 0`). The result is clamped to
/// `[-MAX_CLARITY_DB, MAX_CLARITY_DB]`. Empty, all-zero, non-finite, or
/// non-positive sample-rate inputs return [`NO_USEFUL_RATIO_DB`].
#[must_use]
pub fn useful_to_detrimental_ratio_db(
    ir: &[Sample],
    split_ms: Sample,
    snr_db: Sample,
    sample_rate: Sample,
) -> Sample {
    if ir.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return NO_USEFUL_RATIO_DB;
    }
    let split = ms_to_index(split_ms, sample_rate, ir.len());
    let early = energy(&ir[..split]);
    let late = energy(&ir[split..]);
    let total = early + late;
    // total is a finite non-negative energy sum, so <= is a clean guard.
    if total <= f64::from(ENERGY_FLOOR) {
        return NO_USEFUL_RATIO_DB;
    }
    // A non-finite snr is an infinite signal-to-noise ratio: no noise floor.
    let noise = if snr_db.is_finite() {
        total * f64::from(ops::powf(10.0, -snr_db / 10.0))
    } else {
        0.0
    };
    let denom = late + noise;
    // denom is a finite non-negative energy sum, so <= is a clean guard.
    if denom <= f64::from(ENERGY_FLOOR) {
        // No detrimental energy at all: clamp to the maximum useful ratio.
        return MAX_CLARITY_DB;
    }
    let ratio = (early / denom) as Sample;
    let db = 10.0 * ops::ln(ratio) / LN_10;
    if !db.is_finite() {
        return NO_USEFUL_RATIO_DB;
    }
    db.clamp(-MAX_CLARITY_DB, MAX_CLARITY_DB)
}

/// The Bradley useful-to-detrimental sound ratios `U50` and `U80` of a measured
/// room impulse response at a given signal-to-noise ratio.
///
/// Both fields are in decibels. Degenerate inputs report
/// [`NO_USEFUL_RATIO_DB`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct UsefulToDetrimental {
    /// Useful-to-detrimental ratio `U50` (speech, `50 ms` split) in decibels.
    pub u50_db: Sample,
    /// Useful-to-detrimental ratio `U80` (music, `80 ms` split) in decibels.
    pub u80_db: Sample,
}

impl UsefulToDetrimental {
    /// Computes `U50` and `U80` from a broadband room impulse response at
    /// `sample_rate`, for a background signal-to-noise ratio `snr_db`.
    ///
    /// A non-finite `snr_db` is treated as noiseless. Degenerate inputs return
    /// [`NO_USEFUL_RATIO_DB`] in both fields.
    ///
    /// ```
    /// use prism_audio_spatial::useful_to_detrimental_ratio::UsefulToDetrimental;
    ///
    /// // An early-dominated response has a high useful-to-detrimental ratio.
    /// let sr = 48_000.0f32;
    /// let mut ir = vec![0.0f32; 48_000];
    /// ir[0] = 1.0;
    /// ir[(0.120 * sr) as usize] = 0.1;
    /// let u = UsefulToDetrimental::from_impulse_response(&ir, 30.0, sr);
    /// assert!(u.u50_db > 0.0);
    /// assert!(u.u50_db.is_finite());
    /// ```
    #[must_use]
    pub fn from_impulse_response(ir: &[Sample], snr_db: Sample, sample_rate: Sample) -> Self {
        Self {
            u50_db: useful_to_detrimental_ratio_db(ir, EARLY_LATE_SPLIT_50_MS, snr_db, sample_rate),
            u80_db: useful_to_detrimental_ratio_db(ir, EARLY_LATE_SPLIT_80_MS, snr_db, sample_rate),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::room_clarity::clarity_db;
    use alloc::vec;
    use alloc::vec::Vec;

    const SR: Sample = 48_000.0;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    fn ms_index(ms: Sample) -> usize {
        ops::round(ms / 1000.0 * SR) as usize
    }

    /// A one-second impulse response with a direct hit and one late reflection.
    fn clarity_ir(direct: Sample, late_amp: Sample, late_ms: Sample) -> Vec<Sample> {
        let mut ir = vec![0.0; SR as usize];
        ir[0] = direct;
        if late_amp != 0.0 {
            ir[ms_index(late_ms)] = late_amp;
        }
        ir
    }

    #[test]
    fn high_snr_matches_clarity_50() {
        // At a very high SNR the noise term vanishes and U50 -> C50.
        let ir = clarity_ir(1.0, 0.3, 120.0);
        let u = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 200.0, SR);
        let c = clarity_db(&ir, SR, EARLY_LATE_SPLIT_50_MS);
        assert!(approx(u, c, 1e-3), "u {u} c {c}");
    }

    #[test]
    fn high_snr_matches_clarity_80() {
        let ir = clarity_ir(1.0, 0.3, 120.0);
        let u = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_80_MS, 200.0, SR);
        let c = clarity_db(&ir, SR, EARLY_LATE_SPLIT_80_MS);
        assert!(approx(u, c, 1e-3), "u {u} c {c}");
    }

    #[test]
    fn non_finite_snr_is_noiseless() {
        // A non-finite SNR is treated as infinite SNR, matching clarity exactly.
        let ir = clarity_ir(1.0, 0.3, 120.0);
        let u = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, Sample::INFINITY, SR);
        let c = clarity_db(&ir, SR, EARLY_LATE_SPLIT_50_MS);
        assert!(approx(u, c, 1e-3), "u {u} c {c}");
        let u_nan = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, Sample::NAN, SR);
        assert!(approx(u_nan, c, 1e-3), "u_nan {u_nan} c {c}");
    }

    #[test]
    fn lower_snr_lowers_ratio() {
        // Decreasing SNR raises the noise floor and lowers U monotonically.
        let ir = clarity_ir(1.0, 0.3, 120.0);
        let high = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 40.0, SR);
        let mid = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 10.0, SR);
        let low = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 0.0, SR);
        assert!(high > mid, "high {high} mid {mid}");
        assert!(mid > low, "mid {mid} low {low}");
    }

    #[test]
    fn u50_differs_from_u80_window() {
        // A reflection between 50 ms and 80 ms is late for U50 but early for U80.
        let ir = clarity_ir(1.0, 0.5, 65.0);
        let u50 = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 60.0, SR);
        let u80 = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_80_MS, 60.0, SR);
        assert!(u80 > u50, "u80 {u80} u50 {u50}");
    }

    #[test]
    fn known_ratio_matches_db() {
        // Direct energy 1.0, late energy 0.25, high SNR (noise negligible):
        // U = 10 log10(1 / 0.25) = 6.0206 dB.
        let ir = clarity_ir(1.0, 0.5, 120.0);
        let u = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 200.0, SR);
        assert!(approx(u, 6.020_6, 1e-3), "u {u}");
    }

    #[test]
    fn result_is_clamped_to_max() {
        // A pure direct hit with no late energy and infinite SNR clamps to max.
        let mut ir = vec![0.0; SR as usize];
        ir[0] = 1.0;
        let u = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, Sample::INFINITY, SR);
        assert!(approx(u, MAX_CLARITY_DB, 1e-6), "u {u}");
    }

    #[test]
    fn empty_response_is_sentinel() {
        let empty: [Sample; 0] = [];
        assert_eq!(
            useful_to_detrimental_ratio_db(&empty, EARLY_LATE_SPLIT_50_MS, 30.0, SR),
            NO_USEFUL_RATIO_DB
        );
        assert_eq!(
            UsefulToDetrimental::from_impulse_response(&empty, 30.0, SR).u50_db,
            NO_USEFUL_RATIO_DB
        );
    }

    #[test]
    fn all_zero_response_is_sentinel() {
        let ir = vec![0.0; SR as usize];
        assert_eq!(
            useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 30.0, SR),
            NO_USEFUL_RATIO_DB
        );
    }

    #[test]
    fn zero_sample_rate_is_sentinel() {
        let ir = clarity_ir(1.0, 0.3, 120.0);
        assert_eq!(
            useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 30.0, 0.0),
            NO_USEFUL_RATIO_DB
        );
        assert_eq!(
            useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 30.0, Sample::NAN),
            NO_USEFUL_RATIO_DB
        );
    }

    #[test]
    fn short_ir_within_early_window_is_handled() {
        // A short IR entirely in the early window has no late energy; with a
        // finite SNR the noise floor keeps the ratio finite and clamped.
        let ir = vec![1.0, 0.5, 0.25];
        let u = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 20.0, SR);
        assert!(u.is_finite());
        assert!((-MAX_CLARITY_DB..=MAX_CLARITY_DB).contains(&u), "u {u}");
    }

    #[test]
    fn non_finite_samples_are_safe() {
        let mut ir = clarity_ir(1.0, 0.3, 120.0);
        ir[1] = Sample::NAN;
        ir[ms_index(200.0)] = Sample::INFINITY;
        let u = useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 30.0, SR);
        assert!(u.is_finite());
    }

    #[test]
    fn from_impulse_response_matches_free_function() {
        let ir = clarity_ir(1.0, 0.3, 120.0);
        let u = UsefulToDetrimental::from_impulse_response(&ir, 25.0, SR);
        assert!(approx(
            u.u50_db,
            useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_50_MS, 25.0, SR),
            1e-6
        ));
        assert!(approx(
            u.u80_db,
            useful_to_detrimental_ratio_db(&ir, EARLY_LATE_SPLIT_80_MS, 25.0, SR),
            1e-6
        ));
    }

    #[test]
    fn default_is_zero() {
        let d = UsefulToDetrimental::default();
        assert_eq!(d.u50_db, 0.0);
        assert_eq!(d.u80_db, 0.0);
    }
}
