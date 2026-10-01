//! ISO 3382-1 octave-band reverberation-time spectrum with Beranek bass ratio
//! (`BR`) and treble ratio (`TR`).
//!
//! Real rooms do not reverberate equally at all frequencies: soft furnishings
//! absorb highs, so the reverberation time usually falls with frequency. This
//! module measures the per-octave-band reverberation time from a single
//! broadband room impulse response and condenses the low- and high-frequency
//! behaviour into Beranek's bass ratio (warmth) and treble ratio (brilliance).
//!
//! # Model
//!
//! The response is band-pass filtered into the eight ISO octave bands centred
//! at `OCTAVE_BAND_CENTERS` (63 Hz to 8 kHz). Each band uses a cascade of two
//! identical `RBJ` constant-peak-gain band-pass biquads (a fourth-order
//! response, unity gain at the centre). The band signal is backward integrated
//! (Schroeder) into an energy decay curve in decibels, normalised to `0` dB at
//! the start and floored near `-100` dB. A least-squares line is fitted over
//! the `-5` dB to `-25` dB segment and its slope extrapolated to a `-60` dB
//! drop, giving the per-band reverberation time (the `-60` dB time, equivalent
//! to scaling the `20` dB fit span by three).
//!
//! From the band times `T(f)` the ratios are
//! `BR = (T(125) + T(250)) / (T(500) + T(1000))` and
//! `TR = (T(2000) + T(4000)) / (T(500) + T(1000))`.
//!
//! # Relationship
//!
//! This module extends the broadband `T30`/`EDT` of [`crate::room_clarity`]
//! into a per-octave-band spectrum and adds Beranek's `BR`/`TR`; it does not
//! re-expose the broadband values. It complements [`crate::center_time`] (the
//! energy centre of gravity), [`crate::sound_strength`] (the energy level
//! `G`), and [`crate::initial_time_delay_gap`] (the intimacy gap). The octave
//! centres are shared with [`crate::material_library`] to stay aligned. All
//! share the [`Sample`] scalar from [`prism_audio_core`] and none reimplements
//! another.
//!
//! # Real-time contract
//!
//! This is a control-rate, offline estimator. Outside the per-sample filter
//! loop it performs a single bounded heap allocation of one scratch buffer the
//! length of the response, reused across all bands. It is not a per-sample
//! callback and must not run on an audio thread. It never panics: empty,
//! all-zero, non-finite, non-positive sample-rate, or degenerate-fit inputs
//! return the safe default `0`.
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published ISO 3382-1
//! octave-band reverberation time together with Beranek's bass- and
//! treble-ratio warmth and brilliance criteria, using the publicly documented
//! Robert Bristow-Johnson (`RBJ`) biquad band-pass cookbook formulas. It is
//! pure classic DSP with no AI or ML. It is engine-agnostic and contains **no
//! Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance
//! Audio source or derived code**; it is implemented purely from those
//! publicly documented standards and formulas.

use crate::material_library::{OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT};
use alloc::{vec, vec::Vec};
use bevy_math::ops;
use core::f32::consts::{LN_2, LN_10, PI};
use prism_audio_core::math::Sample;

/// Upper decibel bound of the per-band reverberation-time fit window.
pub const T30_FIT_UPPER_DB: Sample = -5.0;

/// Lower decibel bound of the per-band reverberation-time fit window.
pub const T30_FIT_LOWER_DB: Sample = -25.0;

/// Absolute total-energy floor below which a band is treated as silent.
const ENERGY_FLOOR: f64 = 1e-20;

/// Normalised energy ratio floor, giving a decay-curve floor near `-100` dB.
const RATIO_FLOOR: Sample = 1e-10;

/// Smallest divisor used to keep the least-squares math finite.
const MIN_DIVISOR: Sample = 1e-12;

/// Band-pass bandwidth in octaves (one octave per band).
const BANDWIDTH_OCTAVES: Sample = 1.0;

/// A normalised second-order section (biquad), `a0` folded to `1`.
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: Sample,
    b1: Sample,
    b2: Sample,
    a1: Sample,
    a2: Sample,
}

/// Running direct-form-II transposed state for one biquad.
#[derive(Debug, Clone, Copy, Default)]
struct BiquadState {
    s1: Sample,
    s2: Sample,
}

impl BiquadState {
    #[inline]
    fn tick(&mut self, coeffs: &Biquad, x: Sample) -> Sample {
        let y = coeffs.b0 * x + self.s1;
        self.s1 = coeffs.b1 * x - coeffs.a1 * y + self.s2;
        self.s2 = coeffs.b2 * x - coeffs.a2 * y;
        y
    }
}

/// Hyperbolic sine via the exponential, for the `RBJ` bandwidth term.
#[inline]
fn sinh(x: Sample) -> Sample {
    0.5 * (ops::exp(x) - ops::exp(-x))
}

/// Computes the `RBJ` constant-peak-gain band-pass biquad for `center_hz` at
/// `sample_rate`, or `None` for a degenerate band (non-finite, non-positive, or
/// at or above the Nyquist frequency).
fn bandpass_coeffs(center_hz: Sample, sample_rate: Sample) -> Option<Biquad> {
    if !center_hz.is_finite()
        || !sample_rate.is_finite()
        || center_hz <= 0.0
        || sample_rate <= 0.0
        || center_hz * 2.0 >= sample_rate
    {
        return None;
    }
    let w0 = 2.0 * PI * center_hz / sample_rate;
    let sin_w0 = ops::sin(w0);
    let cos_w0 = ops::cos(w0);
    if sin_w0.abs() < MIN_DIVISOR {
        return None;
    }
    let alpha = sin_w0 * sinh(0.5 * LN_2 * BANDWIDTH_OCTAVES * w0 / sin_w0);
    let a0 = 1.0 + alpha;
    if a0.abs() < MIN_DIVISOR {
        return None;
    }
    let inv = 1.0 / a0;
    Some(Biquad {
        b0: alpha * inv,
        b1: 0.0,
        b2: -alpha * inv,
        a1: -2.0 * cos_w0 * inv,
        a2: (1.0 - alpha) * inv,
    })
}

/// Band-pass filters `rir` into `out` using two cascaded identical biquads
/// (a fourth-order response). Non-finite inputs are treated as `0`.
fn filter_band(rir: &[Sample], coeffs: &Biquad, out: &mut [Sample]) {
    let mut s1 = BiquadState::default();
    let mut s2 = BiquadState::default();
    for (slot, &x) in out.iter_mut().zip(rir.iter()) {
        let xf = if x.is_finite() { x } else { 0.0 };
        let y1 = s1.tick(coeffs, xf);
        *slot = s2.tick(coeffs, y1);
    }
}

/// Overwrites `buf` (a band signal) with its Schroeder energy decay curve in
/// decibels, normalised to `0` dB at the start. Returns `false` when the band
/// is silent.
fn energy_decay_curve_in_place(buf: &mut [Sample]) -> bool {
    let mut acc = 0.0_f64;
    for slot in buf.iter_mut().rev() {
        let v = f64::from(*slot);
        acc += v * v;
        *slot = acc as Sample;
    }
    let total = acc;
    #[expect(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "negated comparison keeps NaN totals on the safe silent branch"
    )]
    if !(total > ENERGY_FLOOR) {
        return false;
    }
    let inv_total = 1.0 / total;
    for slot in buf.iter_mut() {
        let ratio = ((f64::from(*slot) * inv_total) as Sample).max(RATIO_FLOOR);
        *slot = 10.0 * ops::ln(ratio) / LN_10;
    }
    true
}

/// Fits a line to the decay curve `edc_db` over `[T30_FIT_LOWER_DB,
/// T30_FIT_UPPER_DB]` and extrapolates its slope to a `-60` dB drop, returning
/// the time in seconds, or `0` for a degenerate or non-decaying fit.
fn fit_decay_time(edc_db: &[Sample], sample_rate: Sample) -> Sample {
    let mut n_pts = 0usize;
    let mut sum_t = 0.0;
    let mut sum_y = 0.0;
    let mut sum_tt = 0.0;
    let mut sum_ty = 0.0;
    for (i, &y) in edc_db.iter().enumerate() {
        if y.is_finite() && (T30_FIT_LOWER_DB..=T30_FIT_UPPER_DB).contains(&y) {
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

/// Computes the per-octave-band reverberation times, in seconds, for the eight
/// bands centred at `OCTAVE_BAND_CENTERS`.
///
/// Each band is band-pass filtered, Schroeder integrated, and fitted over the
/// `-5` dB to `-25` dB segment, extrapolating to `-60` dB. Degenerate inputs
/// (empty, non-positive or non-finite sample rate, silence, non-decaying fit)
/// yield `0` for the affected bands.
#[must_use]
pub fn octave_band_reverberation_times(
    rir: &[Sample],
    sample_rate: u32,
) -> [Sample; OCTAVE_BAND_COUNT] {
    let mut out = [0.0; OCTAVE_BAND_COUNT];
    let n = rir.len();
    if n == 0 || sample_rate == 0 {
        return out;
    }
    let sr = sample_rate as Sample;
    let mut work: Vec<Sample> = vec![0.0; n];
    for (band, &center_hz) in OCTAVE_BAND_CENTERS.iter().enumerate() {
        let Some(coeffs) = bandpass_coeffs(center_hz, sr) else {
            continue;
        };
        filter_band(rir, &coeffs, &mut work);
        if !energy_decay_curve_in_place(&mut work) {
            continue;
        }
        out[band] = fit_decay_time(&work, sr);
    }
    out
}

/// Returns the ratio `(numerator_a + numerator_b) / (denom_a + denom_b)`, or
/// `0` when the denominator is not positive.
fn safe_ratio(num_a: Sample, num_b: Sample, den_a: Sample, den_b: Sample) -> Sample {
    let denom = den_a + den_b;
    if denom <= 0.0 { 0.0 } else { (num_a + num_b) / denom }
}

/// Computes Beranek's bass ratio from a per-band reverberation-time spectrum:
/// `BR = (T(125) + T(250)) / (T(500) + T(1000))`. Returns `0` when the
/// mid-band denominator is not positive.
#[must_use]
pub fn bass_ratio(t_per_band: &[Sample; OCTAVE_BAND_COUNT]) -> Sample {
    // Band indices: 125 Hz = 1, 250 Hz = 2, 500 Hz = 3, 1 kHz = 4.
    safe_ratio(t_per_band[1], t_per_band[2], t_per_band[3], t_per_band[4])
}

/// Computes Beranek's treble ratio from a per-band reverberation-time spectrum:
/// `TR = (T(2000) + T(4000)) / (T(500) + T(1000))`. Returns `0` when the
/// mid-band denominator is not positive.
#[must_use]
pub fn treble_ratio(t_per_band: &[Sample; OCTAVE_BAND_COUNT]) -> Sample {
    // Band indices: 2 kHz = 5, 4 kHz = 6, 500 Hz = 3, 1 kHz = 4.
    safe_ratio(t_per_band[5], t_per_band[6], t_per_band[3], t_per_band[4])
}

/// The octave-band reverberation-time spectrum of a measured room with its
/// Beranek bass and treble ratios.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReverberationSpectrum {
    /// Per-octave-band reverberation time in seconds, aligned with
    /// `OCTAVE_BAND_CENTERS` (63 Hz to 8 kHz).
    pub t30_per_band: [Sample; OCTAVE_BAND_COUNT],
    /// Beranek bass ratio (warmth): low-band time relative to mid-band time.
    pub bass_ratio: Sample,
    /// Beranek treble ratio (brilliance): high-band time relative to mid-band
    /// time.
    pub treble_ratio: Sample,
}

impl Default for ReverberationSpectrum {
    fn default() -> Self {
        Self {
            t30_per_band: [0.0; OCTAVE_BAND_COUNT],
            bass_ratio: 0.0,
            treble_ratio: 0.0,
        }
    }
}

impl ReverberationSpectrum {
    /// Computes the octave-band reverberation spectrum and Beranek ratios from a
    /// broadband room impulse response.
    ///
    /// Degenerate inputs yield all-zero bands and ratios.
    ///
    /// ```
    /// use prism_audio_spatial::reverberation_spectrum::ReverberationSpectrum;
    ///
    /// // An exponentially decaying 1 kHz tone excites the 1 kHz band.
    /// let sr = 48_000u32;
    /// let tau = 6_000.0f32; // decay time constant in samples
    /// let mut rir = vec![0.0f32; 48_000];
    /// for (n, x) in rir.iter_mut().enumerate() {
    ///     let t = n as f32;
    ///     let env = (-t / tau).exp();
    ///     let phase = 2.0 * core::f32::consts::PI * 1_000.0 * t / sr as f32;
    ///     *x = env * phase.sin();
    /// }
    /// let spec = ReverberationSpectrum::from_impulse_response(&rir, sr);
    /// assert!(spec.t30_per_band[4] > 0.0); // 1 kHz band has a positive T30
    /// ```
    #[must_use]
    pub fn from_impulse_response(rir: &[Sample], sample_rate: u32) -> Self {
        let t30_per_band = octave_band_reverberation_times(rir, sample_rate);
        Self {
            t30_per_band,
            bass_ratio: bass_ratio(&t30_per_band),
            treble_ratio: treble_ratio(&t30_per_band),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const SR: u32 = 48_000;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    /// Builds an exponentially decaying tone at `freq_hz` of `len` samples.
    fn decaying_tone(freq_hz: Sample, tau: Sample, len: usize) -> Vec<Sample> {
        (0..len)
            .map(|n| {
                let t = n as Sample;
                let env = ops::exp(-t / tau);
                let phase = 2.0 * PI * freq_hz * t / SR as Sample;
                env * ops::sin(phase)
            })
            .collect()
    }

    /// Steady-state magnitude of a biquad cascade at `freq_hz` driven by a sine.
    fn steady_state_gain(coeffs: &Biquad, freq_hz: Sample, len: usize) -> Sample {
        let mut s1 = BiquadState::default();
        let mut s2 = BiquadState::default();
        let mut peak: Sample = 0.0;
        for n in 0..len {
            let phase = 2.0 * PI * freq_hz * n as Sample / SR as Sample;
            let x = ops::sin(phase);
            let y1 = s1.tick(coeffs, x);
            let y = s2.tick(coeffs, y1);
            // Measure the peak only in the second half, after the transient.
            if n * 2 >= len {
                peak = peak.max(y.abs());
            }
        }
        peak
    }

    #[test]
    fn bandpass_unity_gain_at_center() {
        let coeffs = bandpass_coeffs(1_000.0, SR as Sample).unwrap();
        let gain = steady_state_gain(&coeffs, 1_000.0, 8_000);
        // Two cascaded unity-peak sections stay close to unity at the centre.
        assert!(approx(gain, 1.0, 0.05), "center gain {gain}");
    }

    #[test]
    fn bandpass_attenuates_far_from_center() {
        let coeffs = bandpass_coeffs(1_000.0, SR as Sample).unwrap();
        let far = steady_state_gain(&coeffs, 8_000.0, 8_000);
        assert!(far < 0.2, "far gain {far} should be well below unity");
    }

    #[test]
    fn exponential_tone_matches_analytic_t60() {
        let tau = 4_000.0; // samples
        let rir = decaying_tone(1_000.0, tau, 48_000);
        let bands = octave_band_reverberation_times(&rir, SR);
        let expected = 3.0 * LN_10 * tau / SR as Sample;
        // 1 kHz is band index 4.
        assert!(
            approx(bands[4], expected, 0.15 * expected),
            "T60 {} expected {expected}",
            bands[4]
        );
    }

    /// Total band energy after cascaded band-pass filtering of `rir`.
    fn band_energy(rir: &[Sample], center_hz: Sample) -> Sample {
        let coeffs = bandpass_coeffs(center_hz, SR as Sample).unwrap();
        let mut work = vec![0.0; rir.len()];
        filter_band(rir, &coeffs, &mut work);
        work.iter().map(|&x| x * x).sum()
    }

    #[test]
    fn adjacent_band_leakage_is_small() {
        // A 1 kHz decaying tone must deposit far more energy in its own band
        // (index 4) than in a non-adjacent band such as 8 kHz (index 7). The
        // reverberation time itself is slope-based and envelope-driven, so it
        // is amplitude-independent; band isolation shows up in band energy.
        let rir = decaying_tone(1_000.0, 4_000.0, 48_000);
        let e_center = band_energy(&rir, 1_000.0);
        let e_far = band_energy(&rir, 8_000.0);
        assert!(e_center > 0.0, "center energy {e_center}");
        assert!(
            e_center > e_far * 100.0,
            "center {e_center} should dominate far {e_far}"
        );
    }

    #[test]
    fn all_zero_response_is_zero() {
        let rir = vec![0.0; 8_000];
        let bands = octave_band_reverberation_times(&rir, SR);
        assert!(bands.iter().all(|&b| b == 0.0));
        let spec = ReverberationSpectrum::from_impulse_response(&rir, SR);
        assert_eq!(spec, ReverberationSpectrum::default());
    }

    #[test]
    fn empty_response_is_zero() {
        let empty: [Sample; 0] = [];
        let bands = octave_band_reverberation_times(&empty, SR);
        assert!(bands.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn zero_sample_rate_is_zero() {
        let rir = decaying_tone(1_000.0, 4_000.0, 8_000);
        let bands = octave_band_reverberation_times(&rir, 0);
        assert!(bands.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn non_finite_samples_are_safe() {
        let mut rir = decaying_tone(1_000.0, 4_000.0, 16_000);
        rir[10] = Sample::NAN;
        rir[20] = Sample::INFINITY;
        let bands = octave_band_reverberation_times(&rir, SR);
        assert!(bands.iter().all(|&b| b.is_finite() && b >= 0.0));
    }

    #[test]
    fn bass_ratio_definition() {
        let mut t = [0.0; OCTAVE_BAND_COUNT];
        t[1] = 2.0; // 125 Hz
        t[2] = 2.0; // 250 Hz
        t[3] = 1.0; // 500 Hz
        t[4] = 1.0; // 1 kHz
        assert!(approx(bass_ratio(&t), 2.0, 1e-6));
    }

    #[test]
    fn treble_ratio_definition() {
        let mut t = [0.0; OCTAVE_BAND_COUNT];
        t[5] = 0.5; // 2 kHz
        t[6] = 0.5; // 4 kHz
        t[3] = 1.0; // 500 Hz
        t[4] = 1.0; // 1 kHz
        assert!(approx(treble_ratio(&t), 0.5, 1e-6));
    }

    #[test]
    fn ratio_zero_denominator_is_zero() {
        let t = [0.0; OCTAVE_BAND_COUNT];
        assert_eq!(bass_ratio(&t), 0.0);
        assert_eq!(treble_ratio(&t), 0.0);
    }

    #[test]
    fn from_impulse_response_matches_free_functions() {
        let rir = decaying_tone(1_000.0, 4_000.0, 24_000);
        let spec = ReverberationSpectrum::from_impulse_response(&rir, SR);
        let bands = octave_band_reverberation_times(&rir, SR);
        assert_eq!(spec.t30_per_band, bands);
        assert!(approx(spec.bass_ratio, bass_ratio(&bands), 1e-6));
        assert!(approx(spec.treble_ratio, treble_ratio(&bands), 1e-6));
    }

    #[test]
    fn default_is_all_zero() {
        let d = ReverberationSpectrum::default();
        assert!(d.t30_per_band.iter().all(|&b| b == 0.0));
        assert_eq!(d.bass_ratio, 0.0);
        assert_eq!(d.treble_ratio, 0.0);
    }

    #[test]
    fn degenerate_band_above_nyquist_is_none() {
        // At a low sample rate the 8 kHz band is at or above Nyquist.
        assert!(bandpass_coeffs(8_000.0, 10_000.0).is_none());
        assert!(bandpass_coeffs(-1.0, SR as Sample).is_none());
        assert!(bandpass_coeffs(Sample::NAN, SR as Sample).is_none());
    }
}
