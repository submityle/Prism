//! ISO 3382 room-acoustic clarity and energy-ratio parameters from a measured
//! room impulse response.
//!
//! This module turns a single broadband room impulse response (`RIR`, a slice
//! of samples plus a sample rate) into the objective parameters standardised by
//! ISO 3382-1 for concert halls and rooms. All of them derive from the
//! Schroeder backward energy integral: starting from the tail of the squared
//! impulse response and accumulating towards the front yields a smooth energy
//! decay curve (`EDC`) that is far less noisy than the raw squared response.
//!
//! The parameters computed here are:
//!
//! - **`EDC`** (energy decay curve): the Schroeder backward integral,
//!   normalised so the start is `0` decibels.
//! - **`C50`** and **`C80`** (clarity): `10 * log10(early / late)` with the
//!   early/late split at 50 milliseconds (speech) or 80 milliseconds (music),
//!   in decibels.
//! - **`D50`** (definition, Deutlichkeit): the early-to-total energy ratio with
//!   the split at 50 milliseconds, a value in `[0, 1]`.
//! - **`Ts`** (centre time): the first moment of the squared response,
//!   `sum(t * h^2) / sum(h^2)`, in seconds.
//! - **`EDT`** (early decay time): the `-60` decibel time extrapolated from a
//!   least-squares fit of the `EDC` over its `0` to `-10` decibel segment.
//! - **`T20`** and **`T30`** (reverberation time): the `-60` decibel time
//!   extrapolated from least-squares fits over the `-5` to `-25` and `-5` to
//!   `-35` decibel segments respectively.
//!
//! # Model
//!
//! Let `h[n]` be the impulse response and `e[n] = h[n]^2` its instantaneous
//! energy. The Schroeder backward integral is
//! `EDC[n] = sum_{m >= n} e[m]`, which is non-increasing in `n`. Normalising by
//! the total energy `EDC[0]` and taking `10 * log10(.)` gives a decay curve
//! that starts at `0` decibels. A straight line is fitted (ordinary least
//! squares) to this curve over a chosen decibel window; its slope in decibels
//! per second is extrapolated to a full `60` decibel drop to obtain the
//! reverberation time, so a `10`, `20`, or `30` decibel window scales to `60`
//! decibels by the familiar factors of six, three, and two.
//!
//! Clarity and definition integrate `e[n]` directly: the early window runs from
//! the first sample up to the split sample, the late window runs from the split
//! sample to the end, and `C = 10 * log10(early / late)`, `D = early / total`.
//!
//! # Relationship
//!
//! This module is the measured-response counterpart to
//! [`crate::room_acoustics`]. That module predicts the broadband reverberation
//! time and Schroeder frequency from room geometry and absorption (Sabine,
//! Eyring); this module measures clarity, definition, centre time, and the
//! `EDT`/`T20`/`T30` reverberation times directly from a room impulse response.
//! The two are complementary: geometry drives a prediction, a measurement
//! validates it. Both share the [`Sample`] scalar defined in
//! [`prism_audio_core`]; neither reimplements the other.
//!
//! # Real-time contract
//!
//! The analysis functions are control-rate, offline estimators: they accept a
//! whole impulse response and may perform a single bounded heap allocation to
//! hold the energy decay curve (noted on [`energy_decay_curve`]). They are not
//! per-sample callbacks and must not run on an audio thread. They never panic:
//! empty, all-zero, or non-finite inputs return safe defaults (`0`, an empty
//! curve, or a clamped sentinel). All transcendental math routes through
//! [`bevy_math::ops`].
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published ISO 3382-1
//! objective room-acoustic parameters and of M. R. Schroeder's backward
//! integration method ("New method of measuring reverberation time", JASA
//! 1965). It is pure classic DSP with no AI or ML. It is engine-agnostic and
//! contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code**; it is implemented purely
//! from those publicly documented standards and papers.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::LN_10;

use prism_audio_core::math::Sample;

/// Early/late split used for `C50` and `D50`, in milliseconds (speech).
pub const EARLY_LATE_SPLIT_50_MS: Sample = 50.0;

/// Early/late split used for `C80`, in milliseconds (music).
pub const EARLY_LATE_SPLIT_80_MS: Sample = 80.0;

/// Upper decibel bound of the `EDT` fit window (closest to the start).
pub const EDT_UPPER_DB: Sample = 0.0;

/// Lower decibel bound of the `EDT` fit window.
pub const EDT_LOWER_DB: Sample = -10.0;

/// Upper decibel bound of the `T20` fit window.
pub const T20_UPPER_DB: Sample = -5.0;

/// Lower decibel bound of the `T20` fit window.
pub const T20_LOWER_DB: Sample = -25.0;

/// Upper decibel bound of the `T30` fit window.
pub const T30_UPPER_DB: Sample = -5.0;

/// Lower decibel bound of the `T30` fit window.
pub const T30_LOWER_DB: Sample = -35.0;

/// Clarity magnitude cap in decibels. Clarity is clamped to this range so a
/// degenerate (silent early or late) window yields a finite sentinel.
pub const MAX_CLARITY_DB: Sample = 100.0;

/// Absolute total-energy floor below which a response is treated as silent.
const ENERGY_FLOOR: Sample = 1e-20;

/// Normalised energy ratio floor, giving an `EDC` floor of about `-100`
/// decibels so the curve stays finite and monotonic.
const RATIO_FLOOR: Sample = 1e-10;

/// Smallest divisor used to keep least-squares and ratio math finite.
const MIN_DIVISOR: Sample = 1e-12;

/// Computes the Schroeder energy decay curve of a room impulse response, in
/// decibels, normalised so the first value is `0` decibels.
///
/// The returned vector has the same length as `rir`. It is non-increasing and
/// floored at about `-100` decibels. An empty response, or one whose total
/// energy is below [`ENERGY_FLOOR`] (silence), returns an empty vector.
///
/// This performs a single heap allocation of one `Sample` per input sample and
/// is an offline, control-rate analysis step, not a per-sample callback.
#[must_use]
pub fn energy_decay_curve(rir: &[Sample]) -> Vec<Sample> {
    let n = rir.len();
    if n == 0 {
        return Vec::new();
    }
    let mut edc = vec![0.0; n];

    // Backward integration: accumulate squared energy from the tail forward.
    let mut acc = 0.0;
    for (slot, &x) in edc.iter_mut().zip(rir.iter()).rev() {
        let e = x * x;
        if e.is_finite() {
            acc += e;
        }
        *slot = acc;
    }

    let total = edc[0];
    #[expect(clippy::neg_cmp_op_on_partial_ord, reason = "negated comparison keeps NaN totals on the safe branch, unlike the suggested direct comparison")]
    if !(total > ENERGY_FLOOR) {
        return Vec::new();
    }

    for v in &mut edc {
        let ratio = (*v / total).max(RATIO_FLOOR);
        *v = 10.0 * ops::ln(ratio) / LN_10;
    }
    edc
}

/// Returns the index of the early/late split sample for `split_ms` at
/// `sample_rate`, clamped into `[0, len]`.
fn split_index(split_ms: Sample, sample_rate: Sample, len: usize) -> usize {
    let x = split_ms / 1000.0 * sample_rate;
    if !x.is_finite() || x <= 0.0 {
        return 0;
    }
    let idx = ops::round(x) as usize;
    idx.min(len)
}

/// Converts an early and late energy pair into a clarity value in decibels,
/// clamped to `[-MAX_CLARITY_DB, MAX_CLARITY_DB]`.
fn clarity_from_energies(early: Sample, late: Sample) -> Sample {
    #[expect(clippy::neg_cmp_op_on_partial_ord, reason = "negated comparison keeps NaN totals on the safe branch, unlike the suggested direct comparison")]
    if !(early > ENERGY_FLOOR) {
        return -MAX_CLARITY_DB;
    }
    #[expect(clippy::neg_cmp_op_on_partial_ord, reason = "negated comparison keeps NaN totals on the safe branch, unlike the suggested direct comparison")]
    if !(late > ENERGY_FLOOR) {
        return MAX_CLARITY_DB;
    }
    (10.0 * ops::ln(early / late) / LN_10).clamp(-MAX_CLARITY_DB, MAX_CLARITY_DB)
}

/// Accumulates the total energy, the early energy up to `split`, and the
/// first-moment (centre-time) numerator of a response in a single pass.
fn energy_summary(rir: &[Sample], sample_rate: Sample, split: usize) -> (Sample, Sample, Sample) {
    let mut total = 0.0;
    let mut early = 0.0;
    let mut moment = 0.0;
    for (i, &x) in rir.iter().enumerate() {
        let e = x * x;
        if !e.is_finite() {
            continue;
        }
        total += e;
        if i < split {
            early += e;
        }
        moment += (i as Sample / sample_rate) * e;
    }
    (total, early, moment)
}

/// Computes the clarity in decibels, `10 * log10(early / late)`, with the
/// early/late boundary at `split_ms` milliseconds.
///
/// Pass [`EARLY_LATE_SPLIT_50_MS`] for `C50` or [`EARLY_LATE_SPLIT_80_MS`] for
/// `C80`. An empty response, a non-positive or non-finite sample rate, or a
/// silent early or late window returns a clamped sentinel in
/// `[-MAX_CLARITY_DB, MAX_CLARITY_DB]`.
#[must_use]
pub fn clarity_db(rir: &[Sample], sample_rate: Sample, split_ms: Sample) -> Sample {
    if rir.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return 0.0;
    }
    let split = split_index(split_ms, sample_rate, rir.len());
    let (total, early, _) = energy_summary(rir, sample_rate, split);
    let late = (total - early).max(0.0);
    clarity_from_energies(early, late)
}

/// Computes the definition `D50`, the ratio of early (first 50 milliseconds) to
/// total energy, in `[0, 1]`.
///
/// An empty response, a non-positive or non-finite sample rate, or a silent
/// response returns `0`.
#[must_use]
pub fn definition(rir: &[Sample], sample_rate: Sample) -> Sample {
    if rir.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return 0.0;
    }
    let split = split_index(EARLY_LATE_SPLIT_50_MS, sample_rate, rir.len());
    let (total, early, _) = energy_summary(rir, sample_rate, split);
    #[expect(clippy::neg_cmp_op_on_partial_ord, reason = "negated comparison keeps NaN totals on the safe branch, unlike the suggested direct comparison")]
    if !(total > ENERGY_FLOOR) {
        return 0.0;
    }
    (early / total).clamp(0.0, 1.0)
}

/// Computes the centre time `Ts`, the first moment of the squared response
/// `sum(t * h^2) / sum(h^2)`, in seconds.
///
/// An empty response, a non-positive or non-finite sample rate, or a silent
/// response returns `0`.
#[must_use]
pub fn center_time_s(rir: &[Sample], sample_rate: Sample) -> Sample {
    if rir.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return 0.0;
    }
    let (total, _, moment) = energy_summary(rir, sample_rate, 0);
    #[expect(clippy::neg_cmp_op_on_partial_ord, reason = "negated comparison keeps NaN totals on the safe branch, unlike the suggested direct comparison")]
    if !(total > ENERGY_FLOOR) {
        return 0.0;
    }
    moment / total
}

/// Fits a straight line to the energy decay curve over `[lower_db, upper_db]`
/// and extrapolates its slope to a `-60` decibel drop, returning the time in
/// seconds. Returns `0` when the window has fewer than two points or the fit is
/// not decaying.
fn fit_decay_time(edc_db: &[Sample], sample_rate: Sample, upper_db: Sample, lower_db: Sample) -> Sample {
    let mut n_pts = 0usize;
    let mut sum_t = 0.0;
    let mut sum_y = 0.0;
    let mut sum_tt = 0.0;
    let mut sum_ty = 0.0;
    for (i, &y) in edc_db.iter().enumerate() {
        if y.is_finite() && y <= upper_db && y >= lower_db {
            let t = i as Sample / sample_rate;
            n_pts += 1;
            sum_t += t;
            sum_y += y;
            sum_tt += t * t;
            sum_ty += t * y;
        }
    }
    if n_pts < 2 {
        return 0.0;
    }
    let n = n_pts as Sample;
    let denom = n * sum_tt - sum_t * sum_t;
    if denom.abs() < MIN_DIVISOR {
        return 0.0;
    }
    let slope = (n * sum_ty - sum_t * sum_y) / denom;
    if slope >= -MIN_DIVISOR {
        return 0.0;
    }
    let rt = -60.0 / slope;
    if rt.is_finite() && rt >= 0.0 { rt } else { 0.0 }
}

/// Computes the early decay time `EDT`: the `-60` decibel time extrapolated
/// from a least-squares fit of the energy decay curve over its `0` to `-10`
/// decibel segment. Returns `0` for a degenerate or insufficiently decaying
/// response.
#[must_use]
pub fn early_decay_time_s(rir: &[Sample], sample_rate: Sample) -> Sample {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return 0.0;
    }
    let edc = energy_decay_curve(rir);
    fit_decay_time(&edc, sample_rate, EDT_UPPER_DB, EDT_LOWER_DB)
}

/// Computes the `T20` reverberation time: the `-60` decibel time extrapolated
/// from a least-squares fit of the energy decay curve over its `-5` to `-25`
/// decibel segment. Returns `0` for a degenerate or insufficiently decaying
/// response.
#[must_use]
pub fn reverberation_time_t20_s(rir: &[Sample], sample_rate: Sample) -> Sample {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return 0.0;
    }
    let edc = energy_decay_curve(rir);
    fit_decay_time(&edc, sample_rate, T20_UPPER_DB, T20_LOWER_DB)
}

/// Computes the `T30` reverberation time: the `-60` decibel time extrapolated
/// from a least-squares fit of the energy decay curve over its `-5` to `-35`
/// decibel segment. Returns `0` for a degenerate or insufficiently decaying
/// response.
#[must_use]
pub fn reverberation_time_t30_s(rir: &[Sample], sample_rate: Sample) -> Sample {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return 0.0;
    }
    let edc = energy_decay_curve(rir);
    fit_decay_time(&edc, sample_rate, T30_UPPER_DB, T30_LOWER_DB)
}

/// The ISO 3382 objective clarity and reverberation parameters of a room
/// impulse response.
///
/// Build one with [`RoomClarity::from_impulse_response`], which computes the
/// energy decay curve once and reuses it for all of the fitted reverberation
/// times. Every field is finite; degenerate inputs yield zeros (and clamped
/// clarity sentinels).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RoomClarity {
    /// Clarity `C50` in decibels (early/late split at 50 milliseconds).
    pub c50_db: Sample,
    /// Clarity `C80` in decibels (early/late split at 80 milliseconds).
    pub c80_db: Sample,
    /// Definition `D50`, the early-to-total energy ratio in `[0, 1]`.
    pub d50: Sample,
    /// Centre time `Ts` in seconds.
    pub center_time_s: Sample,
    /// Early decay time `EDT` in seconds.
    pub edt_s: Sample,
    /// Reverberation time `T20` in seconds.
    pub t20_s: Sample,
    /// Reverberation time `T30` in seconds.
    pub t30_s: Sample,
}

impl Default for RoomClarity {
    fn default() -> Self {
        Self {
            c50_db: 0.0,
            c80_db: 0.0,
            d50: 0.0,
            center_time_s: 0.0,
            edt_s: 0.0,
            t20_s: 0.0,
            t30_s: 0.0,
        }
    }
}

impl RoomClarity {
    /// Computes every objective parameter from a room impulse response.
    ///
    /// The energy decay curve is computed once and shared by the `EDT`, `T20`,
    /// and `T30` fits. An empty response, a non-positive or non-finite sample
    /// rate, or a silent response returns [`RoomClarity::default`] (all zeros).
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::room_clarity::RoomClarity;
    ///
    /// // Synthesise an exponentially decaying impulse response.
    /// let sample_rate = 48_000.0f32;
    /// let mut rir = vec![0.0f32; 48_000];
    /// for (n, x) in rir.iter_mut().enumerate() {
    ///     *x = (-(n as f32) / 6_000.0).exp();
    /// }
    ///
    /// let clarity = RoomClarity::from_impulse_response(&rir, sample_rate);
    /// assert!(clarity.c80_db.is_finite());
    /// assert!(clarity.t30_s > 0.0);
    /// assert!((0.0..=1.0).contains(&clarity.d50));
    /// ```
    #[must_use]
    pub fn from_impulse_response(rir: &[Sample], sample_rate: Sample) -> Self {
        if rir.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Self::default();
        }

        let n = rir.len();
        let split50 = split_index(EARLY_LATE_SPLIT_50_MS, sample_rate, n);
        let split80 = split_index(EARLY_LATE_SPLIT_80_MS, sample_rate, n);

        // Single pass for total energy, early energies, and the centre-time
        // moment. The 80 ms early sum includes the 50 ms early sum, so running
        // two cheap counters avoids a second scan.
        let mut total = 0.0;
        let mut early50 = 0.0;
        let mut early80 = 0.0;
        let mut moment = 0.0;
        for (i, &x) in rir.iter().enumerate() {
            let e = x * x;
            if !e.is_finite() {
                continue;
            }
            total += e;
            if i < split50 {
                early50 += e;
            }
            if i < split80 {
                early80 += e;
            }
            moment += (i as Sample / sample_rate) * e;
        }

        #[expect(clippy::neg_cmp_op_on_partial_ord, reason = "negated comparison keeps NaN totals on the safe branch, unlike the suggested direct comparison")]
        if !(total > ENERGY_FLOOR) {
            return Self::default();
        }

        let late50 = (total - early50).max(0.0);
        let late80 = (total - early80).max(0.0);
        let c50_db = clarity_from_energies(early50, late50);
        let c80_db = clarity_from_energies(early80, late80);
        let d50 = (early50 / total).clamp(0.0, 1.0);
        let center_time_s = moment / total;

        let edc = energy_decay_curve(rir);
        let edt_s = fit_decay_time(&edc, sample_rate, EDT_UPPER_DB, EDT_LOWER_DB);
        let t20_s = fit_decay_time(&edc, sample_rate, T20_UPPER_DB, T20_LOWER_DB);
        let t30_s = fit_decay_time(&edc, sample_rate, T30_UPPER_DB, T30_LOWER_DB);

        Self {
            c50_db,
            c80_db,
            d50,
            center_time_s,
            edt_s,
            t20_s,
            t30_s,
        }
    }

    /// The definition `D50` expressed as a percentage in `[0, 100]`.
    #[must_use]
    pub fn definition_percent(&self) -> Sample {
        self.d50 * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::LN_10 as TEST_LN_10;

    const SR: Sample = 48_000.0;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    /// Builds an exponentially decaying impulse response `exp(-n / tau)`.
    fn exp_rir(len: usize, tau: Sample) -> Vec<Sample> {
        let mut rir = vec![0.0; len];
        for (n, x) in rir.iter_mut().enumerate() {
            *x = ops::exp(-(n as Sample) / tau);
        }
        rir
    }

    #[test]
    fn edc_first_sample_is_zero_db() {
        let rir = exp_rir(8_000, 2_000.0);
        let edc = energy_decay_curve(&rir);
        assert_eq!(edc.len(), rir.len());
        assert!(approx(edc[0], 0.0, 1e-5));
    }

    #[test]
    fn edc_is_monotonic_non_increasing() {
        let rir = exp_rir(8_000, 2_000.0);
        let edc = energy_decay_curve(&rir);
        for w in edc.windows(2) {
            assert!(w[1] <= w[0] + 1e-5, "edc rose: {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn empty_rir_is_safe() {
        let rir: [Sample; 0] = [];
        assert!(energy_decay_curve(&rir).is_empty());
        assert_eq!(clarity_db(&rir, SR, EARLY_LATE_SPLIT_80_MS), 0.0);
        assert_eq!(definition(&rir, SR), 0.0);
        assert_eq!(center_time_s(&rir, SR), 0.0);
        assert_eq!(early_decay_time_s(&rir, SR), 0.0);
        assert_eq!(reverberation_time_t30_s(&rir, SR), 0.0);
        assert_eq!(RoomClarity::from_impulse_response(&rir, SR), RoomClarity::default());
    }

    #[test]
    fn all_zero_rir_is_safe() {
        let rir = [0.0; 4_000];
        assert!(energy_decay_curve(&rir).is_empty());
        assert_eq!(definition(&rir, SR), 0.0);
        assert_eq!(center_time_s(&rir, SR), 0.0);
        assert_eq!(reverberation_time_t20_s(&rir, SR), 0.0);
        let rc = RoomClarity::from_impulse_response(&rir, SR);
        assert_eq!(rc, RoomClarity::default());
    }

    #[test]
    fn non_finite_rir_is_safe() {
        let mut rir = exp_rir(8_000, 2_000.0);
        rir[10] = Sample::NAN;
        rir[20] = Sample::INFINITY;
        let rc = RoomClarity::from_impulse_response(&rir, SR);
        assert!(rc.c50_db.is_finite());
        assert!(rc.c80_db.is_finite());
        assert!(rc.d50.is_finite());
        assert!(rc.center_time_s.is_finite());
        assert!(rc.t30_s.is_finite());
    }

    #[test]
    fn zero_sample_rate_is_safe() {
        let rir = exp_rir(4_000, 1_000.0);
        assert_eq!(clarity_db(&rir, 0.0, EARLY_LATE_SPLIT_50_MS), 0.0);
        assert_eq!(definition(&rir, 0.0), 0.0);
        assert_eq!(center_time_s(&rir, Sample::NAN), 0.0);
        assert_eq!(RoomClarity::from_impulse_response(&rir, -1.0), RoomClarity::default());
    }

    #[test]
    fn dirac_has_unit_definition_and_high_clarity() {
        let mut rir = [0.0; 4_000];
        rir[0] = 1.0;
        assert!(approx(definition(&rir, SR), 1.0, 1e-6));
        // No late energy -> clarity saturates to the positive cap.
        assert!(approx(clarity_db(&rir, SR, EARLY_LATE_SPLIT_50_MS), MAX_CLARITY_DB, 1e-6));
    }

    #[test]
    fn exponential_decay_t30_matches_analytic() {
        let tau = 6_000.0;
        let rir = exp_rir(48_000, tau);
        let t30 = reverberation_time_t30_s(&rir, SR);
        // For h = exp(-n / tau) the exact -60 dB time is 3 * ln(10) * tau / sr.
        let expected = 3.0 * TEST_LN_10 * tau / SR;
        assert!(approx(t30, expected, expected * 0.03), "t30 {t30} vs {expected}");
    }

    #[test]
    fn edt_matches_t30_for_exponential() {
        let rir = exp_rir(48_000, 6_000.0);
        let edt = early_decay_time_s(&rir, SR);
        let t30 = reverberation_time_t30_s(&rir, SR);
        assert!(approx(edt, t30, t30 * 0.05), "edt {edt} vs t30 {t30}");
    }

    #[test]
    fn t20_matches_t30_for_exponential() {
        let rir = exp_rir(48_000, 6_000.0);
        let t20 = reverberation_time_t20_s(&rir, SR);
        let t30 = reverberation_time_t30_s(&rir, SR);
        assert!(approx(t20, t30, t30 * 0.05), "t20 {t20} vs t30 {t30}");
    }

    #[test]
    fn longer_decay_gives_larger_edt() {
        let fast = exp_rir(48_000, 3_000.0);
        let slow = exp_rir(48_000, 9_000.0);
        let edt_fast = early_decay_time_s(&fast, SR);
        let edt_slow = early_decay_time_s(&slow, SR);
        assert!(edt_slow > edt_fast, "slow {edt_slow} fast {edt_fast}");
    }

    #[test]
    fn faster_decay_gives_higher_clarity() {
        let fast = exp_rir(48_000, 3_000.0);
        let slow = exp_rir(48_000, 12_000.0);
        let c_fast = clarity_db(&fast, SR, EARLY_LATE_SPLIT_80_MS);
        let c_slow = clarity_db(&slow, SR, EARLY_LATE_SPLIT_80_MS);
        assert!(c_fast > c_slow, "fast {c_fast} slow {c_slow}");
    }

    #[test]
    fn c80_matches_independent_summation() {
        let rir = exp_rir(48_000, 6_000.0);
        let split = split_index(EARLY_LATE_SPLIT_80_MS, SR, rir.len());
        let mut early = 0.0;
        let mut late = 0.0;
        for (i, &x) in rir.iter().enumerate() {
            let e = x * x;
            if i < split {
                early += e;
            } else {
                late += e;
            }
        }
        let expected = 10.0 * ops::ln(early / late) / LN_10;
        let got = clarity_db(&rir, SR, EARLY_LATE_SPLIT_80_MS);
        assert!(approx(got, expected, 1e-3), "got {got} vs {expected}");
    }

    #[test]
    fn center_time_increases_with_decay() {
        let fast = exp_rir(48_000, 3_000.0);
        let slow = exp_rir(48_000, 9_000.0);
        let ts_fast = center_time_s(&fast, SR);
        let ts_slow = center_time_s(&slow, SR);
        assert!(ts_slow > ts_fast, "slow {ts_slow} fast {ts_fast}");
        assert!(ts_fast > 0.0);
    }

    #[test]
    fn definition_in_unit_interval() {
        let rir = exp_rir(48_000, 6_000.0);
        let d50 = definition(&rir, SR);
        assert!((0.0..=1.0).contains(&d50));
    }

    #[test]
    fn clarity_matches_definition_relationship() {
        // C50 = 10 log10(D50 / (1 - D50)).
        let rir = exp_rir(48_000, 6_000.0);
        let d50 = definition(&rir, SR);
        let c50 = clarity_db(&rir, SR, EARLY_LATE_SPLIT_50_MS);
        let expected = 10.0 * ops::ln(d50 / (1.0 - d50)) / LN_10;
        assert!(approx(c50, expected, 1e-2), "c50 {c50} vs {expected}");
    }

    #[test]
    fn from_impulse_response_matches_free_functions() {
        let rir = exp_rir(48_000, 6_000.0);
        let rc = RoomClarity::from_impulse_response(&rir, SR);
        assert!(approx(rc.c50_db, clarity_db(&rir, SR, EARLY_LATE_SPLIT_50_MS), 1e-3));
        assert!(approx(rc.c80_db, clarity_db(&rir, SR, EARLY_LATE_SPLIT_80_MS), 1e-3));
        assert!(approx(rc.d50, definition(&rir, SR), 1e-5));
        assert!(approx(rc.center_time_s, center_time_s(&rir, SR), 1e-5));
        assert!(approx(rc.edt_s, early_decay_time_s(&rir, SR), 1e-5));
        assert!(approx(rc.t20_s, reverberation_time_t20_s(&rir, SR), 1e-5));
        assert!(approx(rc.t30_s, reverberation_time_t30_s(&rir, SR), 1e-5));
    }

    #[test]
    fn definition_percent_matches_ratio() {
        let rir = exp_rir(48_000, 6_000.0);
        let rc = RoomClarity::from_impulse_response(&rir, SR);
        assert!(approx(rc.definition_percent(), rc.d50 * 100.0, 1e-5));
    }

}
