//! ISO 3382-1 spatial-impression parameters from measured impulse responses:
//! the early lateral energy fractions `LF` and `LFC`, and the interaural
//! cross-correlation coefficient `IACC`.
//!
//! Where [`crate::room_clarity`] quantifies how distinct and reverberant a room
//! sounds, this module quantifies how spacious and enveloping it sounds. Both
//! families of parameters are standardised by ISO 3382-1 for concert halls and
//! rooms and are derived here directly from measured impulse responses.
//!
//! Two listener-side spatial attributes are captured:
//!
//! - **Apparent source width** is driven by the early lateral energy that
//!   arrives from the sides. It is measured by the lateral energy fraction
//!   `LF` (also written `JLF`) and its cosine-weighted variant `LFC`, computed
//!   from a coincident omnidirectional response `p` and a figure-of-eight
//!   (lateral) response `pL`.
//! - **Listener envelopment** is driven by how dissimilar the two ear signals
//!   are. It is measured by the interaural cross-correlation coefficient
//!   `IACC`, computed from a binaural pair of responses.
//!
//! # Model
//!
//! Let `p[n]` be the omnidirectional impulse response, `pL[n]` the
//! figure-of-eight response aligned with the lateral axis, and `t = n /
//! sample_rate` the arrival time in seconds. With the early window ending at
//! `80 ms` and the lateral window starting at `5 ms`:
//!
//! - `LF = sum_{5 ms <= t <= 80 ms} pL[n]^2 / sum_{0 <= t <= 80 ms} p[n]^2`.
//! - `LFC = sum_{5 ms <= t <= 80 ms} |pL[n] * p[n]| / sum_{0 <= t <= 80 ms} p[n]^2`.
//!
//! For a binaural pair `l[n]`, `r[n]` and a time window `[t1, t2]`, the
//! normalised interaural cross-correlation function at lag `tau` is
//! `IACF(tau) = sum l[t] r[t + tau] / sqrt(sum l[t]^2 * sum r[t]^2)`, where the
//! energy sums run over the window, and `IACC = max_{|tau| <= 1 ms} |IACF(tau)|`.
//! Three windows are reported: `IACC_early` (`0` to `80 ms`), `IACC_late`
//! (`80 ms` to the end), and `IACC_all` (the whole response).
//!
//! # Relationship
//!
//! This module is the spatial-impression counterpart to
//! [`crate::room_clarity`], which covers clarity, definition, centre time, and
//! the reverberation times, and to [`crate::room_acoustics`], which predicts
//! broadband reverberation from geometry. Clarity answers how clear the sound
//! is; this module answers how wide and enveloping it is. All three share the
//! [`Sample`] scalar from [`prism_audio_core`] and none reimplements another.
//!
//! # Real-time contract
//!
//! These functions are control-rate, offline estimators: each accepts a whole
//! impulse response and may perform a single bounded heap allocation. They are
//! not per-sample callbacks and must not run on an audio thread. They never
//! panic: empty, all-zero, non-finite, or non-positive-sample-rate inputs
//! return safe defaults (`0`). All transcendental and length math routes
//! through [`bevy_math::ops`].
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published ISO 3382-1
//! spatial-impression parameters (lateral energy fractions and interaural
//! cross-correlation). It is pure classic DSP with no AI or ML. It is
//! engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented standard.


use bevy_math::ops;

use prism_audio_core::math::Sample;

/// Start of the lateral energy integration window, in milliseconds.
pub const LF_EARLY_START_MS: Sample = 5.0;

/// End of the early energy window shared by `LF`, `LFC`, and `IACC_early`, in
/// milliseconds.
pub const EARLY_WINDOW_END_MS: Sample = 80.0;

/// Maximum interaural lag searched for `IACC`, in milliseconds.
pub const IACC_MAX_LAG_MS: Sample = 1.0;

/// Absolute energy floor below which a window is treated as silent.
const ENERGY_FLOOR: Sample = 1e-20;

/// Returns `x` when it is finite, otherwise `0`, so non-finite samples drop out
/// of every sum without poisoning it.
#[inline]
fn finite(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
}

/// Converts a millisecond offset to a non-negative sample index, returning `0`
/// for non-finite or non-positive inputs.
fn ms_to_samples(ms: Sample, sample_rate: Sample) -> usize {
    let x = ms / 1000.0 * sample_rate;
    if !x.is_finite() || x <= 0.0 {
        return 0;
    }
    ops::round(x) as usize
}

/// Clamps a `[start_ms, end_ms]` window to `[0, len]` sample indices, returning
/// an empty `(start, start)` range when the window is degenerate.
fn window_bounds(start_ms: Sample, end_ms: Sample, sample_rate: Sample, len: usize) -> (usize, usize) {
    let start = ms_to_samples(start_ms, sample_rate).min(len);
    let end = ms_to_samples(end_ms, sample_rate).min(len);
    if end > start { (start, end) } else { (start, start) }
}

/// Sums the squared (finite) samples of `slice`.
fn energy(slice: &[Sample]) -> Sample {
    let mut acc = 0.0;
    for &x in slice {
        let v = finite(x);
        acc += v * v;
    }
    acc
}

/// Computes the early lateral energy fraction `LF` (also written `JLF`).
///
/// `omni` is the coincident omnidirectional impulse response and
/// `figure_eight` the lateral figure-of-eight response, both at the same
/// `sample_rate` and ideally the same length (the shorter length is used).
/// `LF = sum_{5..80 ms} figure_eight^2 / sum_{0..80 ms} omni^2`, clamped to
/// `[0, 1]`.
///
/// Empty responses, a non-positive or non-finite sample rate, or a silent
/// omnidirectional early window return `0`.
#[must_use]
pub fn lateral_energy_fraction(omni: &[Sample], figure_eight: &[Sample], sample_rate: Sample) -> Sample {
    if omni.is_empty() || figure_eight.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return 0.0;
    }
    let n = omni.len().min(figure_eight.len());
    let (lat_start, lat_end) = window_bounds(LF_EARLY_START_MS, EARLY_WINDOW_END_MS, sample_rate, n);
    let (_, early_end) = window_bounds(0.0, EARLY_WINDOW_END_MS, sample_rate, n);

    let numerator = energy(&figure_eight[lat_start..lat_end]);
    let denominator = energy(&omni[..early_end]);
    #[expect(clippy::neg_cmp_op_on_partial_ord, reason = "negated comparison keeps NaN totals on the safe branch, unlike the suggested direct comparison")]
    if !(denominator > ENERGY_FLOOR) {
        return 0.0;
    }
    (numerator / denominator).clamp(0.0, 1.0)
}

/// Computes the cosine-weighted lateral energy fraction `LFC` (Kleiner).
///
/// Uses the absolute cross product of the lateral and omnidirectional
/// responses, `LFC = sum_{5..80 ms} |figure_eight * omni| / sum_{0..80 ms}
/// omni^2`, clamped to `[0, 1]`.
///
/// Empty responses, a non-positive or non-finite sample rate, or a silent
/// omnidirectional early window return `0`.
#[must_use]
pub fn lateral_energy_fraction_cosine(omni: &[Sample], figure_eight: &[Sample], sample_rate: Sample) -> Sample {
    if omni.is_empty() || figure_eight.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return 0.0;
    }
    let n = omni.len().min(figure_eight.len());
    let (lat_start, lat_end) = window_bounds(LF_EARLY_START_MS, EARLY_WINDOW_END_MS, sample_rate, n);
    let (_, early_end) = window_bounds(0.0, EARLY_WINDOW_END_MS, sample_rate, n);

    let mut numerator = 0.0;
    for (&a, &b) in omni[lat_start..lat_end].iter().zip(&figure_eight[lat_start..lat_end]) {
        numerator += (finite(a) * finite(b)).abs();
    }
    let denominator = energy(&omni[..early_end]);
    #[expect(clippy::neg_cmp_op_on_partial_ord, reason = "negated comparison keeps NaN totals on the safe branch, unlike the suggested direct comparison")]
    if !(denominator > ENERGY_FLOOR) {
        return 0.0;
    }
    (numerator / denominator).clamp(0.0, 1.0)
}

/// Computes the interaural cross-correlation coefficient `IACC` over the time
/// window `[window_start_ms, window_end_ms]`.
///
/// `IACC = max_{|tau| <= 1 ms} |IACF(tau)|`, where
/// `IACF(tau) = sum l[t] r[t + tau] / sqrt(sum l^2 * sum r^2)` and the energy
/// sums run over the window. The result is clamped to `[0, 1]`.
///
/// Empty responses, a non-positive or non-finite sample rate, a degenerate
/// window, or a silent window return `0`.
#[must_use]
pub fn interaural_cross_correlation(
    left: &[Sample],
    right: &[Sample],
    sample_rate: Sample,
    window_start_ms: Sample,
    window_end_ms: Sample,
) -> Sample {
    if left.is_empty() || right.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return 0.0;
    }
    let n = left.len().min(right.len());
    let (start, end) = window_bounds(window_start_ms, window_end_ms, sample_rate, n);
    if end <= start {
        return 0.0;
    }

    let energy_left = energy(&left[start..end]);
    let energy_right = energy(&right[start..end]);
    let norm = energy_left * energy_right;
    #[expect(clippy::neg_cmp_op_on_partial_ord, reason = "negated comparison keeps NaN totals on the safe branch, unlike the suggested direct comparison")]
    if !(norm > ENERGY_FLOOR) {
        return 0.0;
    }
    let denom = ops::sqrt(norm);

    let max_lag = ms_to_samples(IACC_MAX_LAG_MS, sample_rate) as isize;
    let mut best = 0.0;
    for lag in -max_lag..=max_lag {
        let mut cross = 0.0;
        #[expect(clippy::needless_range_loop, reason = "the index t drives left[t] while right is read at the lag-offset index j, so a single-slice iterator does not apply")]
        for t in start..end {
            let j = t as isize + lag;
            if j < 0 || j as usize >= n {
                continue;
            }
            cross += finite(left[t]) * finite(right[j as usize]);
        }
        let coeff = (cross / denom).abs();
        if coeff > best {
            best = coeff;
        }
    }
    best.clamp(0.0, 1.0)
}

/// The ISO 3382-1 spatial-impression parameters of a measured room.
///
/// Build one with [`SpatialImpression::from_responses`]. The lateral fractions
/// come from a coincident omnidirectional and figure-of-eight pair; the three
/// `IACC` values come from a binaural pair. Every field is finite; degenerate
/// inputs yield zeros.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialImpression {
    /// Early lateral energy fraction `LF` (`JLF`), in `[0, 1]`.
    pub lf: Sample,
    /// Cosine-weighted lateral energy fraction `LFC`, in `[0, 1]`.
    pub lfc: Sample,
    /// Early interaural cross-correlation `IACC` over `0` to `80 ms`.
    pub iacc_early: Sample,
    /// Late interaural cross-correlation `IACC` over `80 ms` to the end.
    pub iacc_late: Sample,
    /// Full-length interaural cross-correlation `IACC`.
    pub iacc_all: Sample,
}

impl Default for SpatialImpression {
    fn default() -> Self {
        Self {
            lf: 0.0,
            lfc: 0.0,
            iacc_early: 0.0,
            iacc_late: 0.0,
            iacc_all: 0.0,
        }
    }
}

impl SpatialImpression {
    /// Computes every spatial-impression parameter from measured responses.
    ///
    /// `omni` and `figure_eight` are the coincident omnidirectional and lateral
    /// responses used for `LF`/`LFC`; `left` and `right` are the binaural pair
    /// used for the three `IACC` windows. All share `sample_rate`. Empty
    /// responses, a non-positive or non-finite sample rate, or silent inputs
    /// yield [`SpatialImpression::default`] (all zeros) for the affected
    /// parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::spatial_impression::SpatialImpression;
    ///
    /// let sample_rate = 48_000.0f32;
    /// let mut omni = vec![0.0f32; 8_000];
    /// let mut figure_eight = vec![0.0f32; 8_000];
    /// omni[0] = 1.0; // direct sound (front, no lateral energy)
    /// omni[1_000] = 0.5; // a lateral early reflection
    /// figure_eight[1_000] = 0.5;
    ///
    /// let left = omni.clone();
    /// let right = omni.clone();
    ///
    /// let si = SpatialImpression::from_responses(&omni, &figure_eight, &left, &right, sample_rate);
    /// assert!((0.0..=1.0).contains(&si.lf));
    /// assert!((si.iacc_all - 1.0).abs() < 1e-4); // identical ears -> fully correlated
    /// ```
    #[must_use]
    pub fn from_responses(
        omni: &[Sample],
        figure_eight: &[Sample],
        left: &[Sample],
        right: &[Sample],
        sample_rate: Sample,
    ) -> Self {
        let lf = lateral_energy_fraction(omni, figure_eight, sample_rate);
        let lfc = lateral_energy_fraction_cosine(omni, figure_eight, sample_rate);

        let n = left.len().min(right.len());
        let duration_ms = if sample_rate.is_finite() && sample_rate > 0.0 {
            n as Sample / sample_rate * 1000.0 + 1.0
        } else {
            0.0
        };

        let iacc_early = interaural_cross_correlation(left, right, sample_rate, 0.0, EARLY_WINDOW_END_MS);
        let iacc_late =
            interaural_cross_correlation(left, right, sample_rate, EARLY_WINDOW_END_MS, duration_ms);
        let iacc_all = interaural_cross_correlation(left, right, sample_rate, 0.0, duration_ms);

        Self {
            lf,
            lfc,
            iacc_early,
            iacc_late,
            iacc_all,
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

    /// Builds an exponentially decaying response `exp(-n / tau)`.
    fn exp_response(len: usize, tau: Sample) -> Vec<Sample> {
        let mut out = vec![0.0; len];
        for (n, x) in out.iter_mut().enumerate() {
            *x = ops::exp(-(n as Sample) / tau);
        }
        out
    }

    /// A simple deterministic pseudo-random generator for decorrelated noise.
    fn lcg_noise(len: usize, seed: u32) -> Vec<Sample> {
        let mut state = seed;
        let mut out = vec![0.0; len];
        for x in &mut out {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as Sample / (1u32 << 24) as Sample;
            *x = unit * 2.0 - 1.0;
        }
        out
    }

    #[test]
    fn direct_only_has_zero_lateral_fraction() {
        // A single front arrival puts all energy in omni, none in figure-eight.
        let mut omni = vec![0.0; 8_000];
        omni[0] = 1.0;
        let figure_eight = vec![0.0; 8_000];
        assert_eq!(lateral_energy_fraction(&omni, &figure_eight, SR), 0.0);
    }

    #[test]
    fn purely_lateral_fraction_approaches_one() {
        // After 5 ms the omni and figure-eight carry the same lateral energy.
        let n = 8_000;
        let mut omni = vec![0.0; n];
        let mut figure_eight = vec![0.0; n];
        let start = ms_to_samples(LF_EARLY_START_MS, SR);
        for i in start..n {
            omni[i] = 0.5;
            figure_eight[i] = 0.5;
        }
        let lf = lateral_energy_fraction(&omni, &figure_eight, SR);
        assert!(lf > 0.95, "lf {lf}");
        assert!(lf <= 1.0);
    }

    #[test]
    fn lateral_fraction_in_unit_interval() {
        let omni = exp_response(8_000, 2_000.0);
        let figure_eight = exp_response(8_000, 3_000.0);
        let lf = lateral_energy_fraction(&omni, &figure_eight, SR);
        assert!((0.0..=1.0).contains(&lf), "lf {lf}");
    }

    #[test]
    fn lateral_fraction_cosine_in_unit_interval() {
        let omni = exp_response(8_000, 2_000.0);
        let figure_eight = exp_response(8_000, 2_000.0);
        let lfc = lateral_energy_fraction_cosine(&omni, &figure_eight, SR);
        assert!((0.0..=1.0).contains(&lfc), "lfc {lfc}");
    }

    #[test]
    fn empty_inputs_are_safe() {
        let empty: [Sample; 0] = [];
        let some = exp_response(4_000, 1_000.0);
        assert_eq!(lateral_energy_fraction(&empty, &some, SR), 0.0);
        assert_eq!(lateral_energy_fraction_cosine(&empty, &some, SR), 0.0);
        assert_eq!(interaural_cross_correlation(&empty, &some, SR, 0.0, 80.0), 0.0);
        let si = SpatialImpression::from_responses(&empty, &empty, &empty, &empty, SR);
        assert_eq!(si, SpatialImpression::default());
    }

    #[test]
    fn all_zero_inputs_are_safe() {
        let z = vec![0.0; 4_000];
        assert_eq!(lateral_energy_fraction(&z, &z, SR), 0.0);
        assert_eq!(lateral_energy_fraction_cosine(&z, &z, SR), 0.0);
        assert_eq!(interaural_cross_correlation(&z, &z, SR, 0.0, 80.0), 0.0);
        let si = SpatialImpression::from_responses(&z, &z, &z, &z, SR);
        assert_eq!(si, SpatialImpression::default());
    }

    #[test]
    fn non_finite_inputs_are_safe() {
        let mut omni = exp_response(8_000, 2_000.0);
        let mut figure_eight = exp_response(8_000, 2_000.0);
        omni[100] = Sample::NAN;
        figure_eight[200] = Sample::INFINITY;
        let si = SpatialImpression::from_responses(&omni, &figure_eight, &omni, &figure_eight, SR);
        assert!(si.lf.is_finite());
        assert!(si.lfc.is_finite());
        assert!(si.iacc_early.is_finite());
        assert!(si.iacc_late.is_finite());
        assert!(si.iacc_all.is_finite());
    }

    #[test]
    fn zero_sample_rate_is_safe() {
        let r = exp_response(4_000, 1_000.0);
        assert_eq!(lateral_energy_fraction(&r, &r, 0.0), 0.0);
        assert_eq!(lateral_energy_fraction_cosine(&r, &r, Sample::NAN), 0.0);
        assert_eq!(interaural_cross_correlation(&r, &r, -1.0, 0.0, 80.0), 0.0);
        assert_eq!(
            SpatialImpression::from_responses(&r, &r, &r, &r, 0.0),
            SpatialImpression::default()
        );
    }

    #[test]
    fn identical_ears_are_fully_correlated() {
        let r = exp_response(8_000, 2_000.0);
        let iacc = interaural_cross_correlation(&r, &r, SR, 0.0, 80.0);
        assert!(approx(iacc, 1.0, 1e-4), "iacc {iacc}");
    }

    #[test]
    fn antiphase_ears_are_fully_correlated_in_magnitude() {
        let r = exp_response(8_000, 2_000.0);
        let inverted: Vec<Sample> = r.iter().map(|&x| -x).collect();
        let iacc = interaural_cross_correlation(&r, &inverted, SR, 0.0, 80.0);
        assert!(approx(iacc, 1.0, 1e-4), "iacc {iacc}");
    }

    #[test]
    fn independent_noise_has_low_iacc() {
        let left = lcg_noise(16_000, 1);
        let right = lcg_noise(16_000, 7);
        let iacc = interaural_cross_correlation(&left, &right, SR, 0.0, 300.0);
        assert!(iacc < 0.5, "iacc {iacc}");
    }

    #[test]
    fn iacc_in_unit_interval() {
        let left = lcg_noise(16_000, 3);
        let right = lcg_noise(16_000, 9);
        let iacc = interaural_cross_correlation(&left, &right, SR, 0.0, 300.0);
        assert!((0.0..=1.0).contains(&iacc), "iacc {iacc}");
    }

    #[test]
    fn delayed_copy_peaks_within_lag_window() {
        // right is left delayed by one sample; the correlation peak at lag 1
        // keeps IACC near unity because one sample is well inside +/-1 ms.
        let base = exp_response(8_000, 2_000.0);
        let mut right = vec![0.0; base.len()];
        right[1..].copy_from_slice(&base[..base.len() - 1]);
        let iacc = interaural_cross_correlation(&base, &right, SR, 0.0, 80.0);
        assert!(iacc > 0.99, "iacc {iacc}");
    }

    #[test]
    fn early_and_late_windows_differ() {
        // Correlated early arrivals, decorrelated late tail.
        let n = 32_000;
        let early_end = ms_to_samples(EARLY_WINDOW_END_MS, SR);
        let mut left = vec![0.0; n];
        let mut right = vec![0.0; n];
        for i in 0..early_end {
            let v = ops::exp(-(i as Sample) / 1_000.0);
            left[i] = v;
            right[i] = v;
        }
        let ln = lcg_noise(n, 11);
        let rn = lcg_noise(n, 23);
        for i in early_end..n {
            left[i] = ln[i] * 0.1;
            right[i] = rn[i] * 0.1;
        }
        let early = interaural_cross_correlation(&left, &right, SR, 0.0, EARLY_WINDOW_END_MS);
        let late = interaural_cross_correlation(&left, &right, SR, EARLY_WINDOW_END_MS, 700.0);
        assert!(early > 0.99, "early {early}");
        assert!(late < early, "late {late} early {early}");
    }

    #[test]
    fn from_responses_matches_free_functions() {
        let omni = exp_response(16_000, 2_000.0);
        let figure_eight = exp_response(16_000, 3_000.0);
        let left = lcg_noise(16_000, 5);
        let right = lcg_noise(16_000, 6);
        let si = SpatialImpression::from_responses(&omni, &figure_eight, &left, &right, SR);
        assert!(approx(si.lf, lateral_energy_fraction(&omni, &figure_eight, SR), 1e-6));
        assert!(approx(si.lfc, lateral_energy_fraction_cosine(&omni, &figure_eight, SR), 1e-6));
        assert!(approx(
            si.iacc_early,
            interaural_cross_correlation(&left, &right, SR, 0.0, EARLY_WINDOW_END_MS),
            1e-6
        ));
    }

    #[test]
    fn default_is_all_zero() {
        let si = SpatialImpression::default();
        assert_eq!(si.lf, 0.0);
        assert_eq!(si.lfc, 0.0);
        assert_eq!(si.iacc_early, 0.0);
        assert_eq!(si.iacc_late, 0.0);
        assert_eq!(si.iacc_all, 0.0);
    }
}
