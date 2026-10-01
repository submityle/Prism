//! IEC 60268-16 speech transmission index (`STI`) from a room impulse
//! response, via the Schroeder modulation-transfer-function method.
//!
//! Reverberation and noise smear the slow amplitude modulations that carry
//! speech, reducing intelligibility. The speech transmission index condenses
//! this loss into a single number in `[0, 1]` by measuring how well each
//! octave band preserves a set of low-frequency amplitude modulations, then
//! combining the per-band results with the publicly tabulated weighting
//! factors of the standard.
//!
//! # Model
//!
//! The response is band-pass filtered into the seven octave bands centred at
//! `OCTAVE_CENTERS_HZ` (125 Hz to 8 kHz). Each band uses a cascade of two
//! identical `RBJ` constant-peak-gain band-pass biquads (a fourth-order,
//! unity-centre-gain response). For each band the squared envelope `h^2(t)`
//! is the energy flow, and for each of the `MODULATION_COUNT` one-third-octave
//! modulation frequencies `MODULATION_FREQS_HZ` (0.63 Hz to 12.5 Hz) the
//! modulation transfer `m(F)` is the normalised magnitude of its Fourier
//! component (Schroeder 1981):
//! `m(F) = |sum_t h^2(t) e^{-j 2 pi F t}| / sum_t h^2(t)`.
//! An optional noise term scales each `m` by `1 / (1 + 10^(-SNR/10))`.
//!
//! Each `m` becomes an apparent signal-to-noise ratio
//! `SNR_app = 10 log10(m / (1 - m))`, clamped to `[-15, +15]` dB. The
//! transmission index of a band is the mean of `(SNR_app + 15) / 30` over the
//! modulation frequencies, giving the modulation transfer index `MTI` in
//! `[0, 1]`. The overall index applies the male-speech weights
//! `STI = sum_k alpha_k MTI_k - sum_k beta_k sqrt(MTI_k MTI_{k+1})`,
//! clamped to `[0, 1]`. A qualitative `StiRating` follows the standard bands.
//!
//! # Relationship
//!
//! This module complements the energy-ratio clarity of [`crate::room_clarity`]
//! (`C50`/`D50`), the energy centre of gravity of [`crate::center_time`], and
//! the echo audibility of [`crate::echo_criterion`]: those report energy
//! ratios, a centroid, and an echo slope, while this reports a
//! modulation-transfer-based speech intelligibility index. All share the
//! [`Sample`] scalar from [`prism_audio_core`] and none reimplements another.
//!
//! # Real-time contract
//!
//! This is a control-rate, offline estimator. Outside the per-sample filter
//! loop it performs a single bounded heap allocation of one scratch buffer the
//! length of the response, reused across all bands. It is not a per-sample
//! callback and must not run on an audio thread. It never panics: empty,
//! all-zero, non-finite, or non-positive sample-rate inputs return the safe
//! default `0`.
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published IEC 60268-16
//! speech transmission index using Schroeder's (1981) modulation transfer
//! function from an impulse response and the publicly documented Robert
//! Bristow-Johnson (`RBJ`) biquad band-pass cookbook formulas. It is pure
//! classic DSP with no AI or ML. It is engine-agnostic and contains **no
//! Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance
//! Audio source or derived code**; it is implemented purely from those
//! publicly documented standards and formulas.

use alloc::{vec, vec::Vec};
use bevy_math::ops;
use core::f32::consts::{LN_2, LN_10, PI};
use prism_audio_core::math::Sample;

/// Number of octave bands used by the index (125 Hz to 8 kHz).
pub const OCTAVE_COUNT: usize = 7;

/// Number of one-third-octave modulation frequencies per band.
pub const MODULATION_COUNT: usize = 14;

/// Octave-band centre frequencies in hertz.
pub const OCTAVE_CENTERS_HZ: [Sample; OCTAVE_COUNT] =
    [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];

/// One-third-octave modulation frequencies in hertz.
pub const MODULATION_FREQS_HZ: [Sample; MODULATION_COUNT] = [
    0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5,
];

/// IEC 60268-16 male-speech band weights `alpha_k` (publicly tabulated).
pub const MALE_ALPHA: [Sample; OCTAVE_COUNT] =
    [0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173];

/// IEC 60268-16 male-speech redundancy weights `beta_k` (publicly tabulated).
pub const MALE_BETA: [Sample; OCTAVE_COUNT - 1] =
    [0.085, 0.078, 0.065, 0.011, 0.047, 0.095];

/// Apparent signal-to-noise clamp limit in decibels.
pub const APPARENT_SNR_LIMIT_DB: Sample = 15.0;

/// Band-pass bandwidth in octaves (one octave per band).
const BANDWIDTH_OCTAVES: Sample = 1.0;

/// Absolute total-energy floor below which a band is treated as silent.
const ENERGY_FLOOR: f64 = 1e-20;

/// Smallest divisor used to keep the biquad math finite.
const MIN_DIVISOR: Sample = 1e-12;

/// Modulation-index clamp bound keeping `m / (1 - m)` finite.
const MTF_LIMIT: Sample = 1.0 - 1e-4;

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
/// `sample_rate`, or `None` for a degenerate band (non-finite, non-positive,
/// or at or above the Nyquist frequency).
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

/// Noise scaling `1 / (1 + 10^(-SNR/10))`; `1` for a non-finite `snr_db`.
fn noise_factor(snr_db: Sample) -> f64 {
    if !snr_db.is_finite() {
        return 1.0;
    }
    let ratio = f64::from(ops::powf(10.0, -snr_db / 10.0));
    1.0 / (1.0 + ratio)
}

/// Modulation transfer `m(F)` for one band signal at `modulation_hz`, scaled
/// by `noise`. Returns `0` for a silent band. The result is clamped to
/// `[0, MTF_LIMIT]`.
fn modulation_index(band: &[Sample], sample_rate: f64, modulation_hz: Sample, noise: f64) -> Sample {
    let mut total = 0.0_f64;
    for &x in band {
        let e = f64::from(x) * f64::from(x);
        total += e;
    }
    #[expect(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "negated comparison keeps NaN totals on the safe silent branch"
    )]
    if !(total > ENERGY_FLOOR) {
        return 0.0;
    }
    let mut re = 0.0_f64;
    let mut im = 0.0_f64;
    let inv_sr = 1.0 / sample_rate;
    for (i, &x) in band.iter().enumerate() {
        let e = f64::from(x) * f64::from(x);
        let phase = 2.0 * f64::from(PI) * f64::from(modulation_hz) * (i as f64) * inv_sr;
        let phase = phase as Sample;
        re += e * f64::from(ops::cos(phase));
        im += e * f64::from(ops::sin(phase));
    }
    let magnitude = ops::sqrt((re * re + im * im) as Sample);
    let m = (magnitude / total as Sample) * noise as Sample;
    m.clamp(0.0, MTF_LIMIT)
}

/// Modulation transfer index of one band: the mean of `(SNR_app + 15) / 30`
/// over the modulation frequencies, in `[0, 1]`.
fn band_mti(band: &[Sample], sample_rate: f64, noise: f64) -> Sample {
    let mut sum = 0.0_f64;
    for &fm in &MODULATION_FREQS_HZ {
        let m = modulation_index(band, sample_rate, fm, noise);
        let one_minus = (1.0 - m).max(MIN_DIVISOR);
        let snr = (10.0 * ops::ln(m / one_minus) / LN_10)
            .clamp(-APPARENT_SNR_LIMIT_DB, APPARENT_SNR_LIMIT_DB);
        sum += f64::from((snr + APPARENT_SNR_LIMIT_DB) / (2.0 * APPARENT_SNR_LIMIT_DB));
    }
    (sum / MODULATION_COUNT as f64) as Sample
}

/// Qualitative speech intelligibility rating per IEC 60268-16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum StiRating {
    /// `STI < 0.3`.
    #[default]
    Bad,
    /// `0.3 <= STI < 0.45`.
    Poor,
    /// `0.45 <= STI < 0.6`.
    Fair,
    /// `0.6 <= STI < 0.75`.
    Good,
    /// `STI >= 0.75`.
    Excellent,
}

impl StiRating {
    /// Maps an `STI` value in `[0, 1]` to its qualitative rating.
    #[must_use]
    pub fn from_sti(sti: Sample) -> Self {
        if sti < 0.3 {
            Self::Bad
        } else if sti < 0.45 {
            Self::Poor
        } else if sti < 0.6 {
            Self::Fair
        } else if sti < 0.75 {
            Self::Good
        } else {
            Self::Excellent
        }
    }
}

/// Computes the speech transmission index `STI` for a room impulse response.
///
/// `snr_db` is the per-band signal-to-noise ratio; pass a large value (for
/// example `100.0`) for an effectively noise-free estimate. Degenerate inputs
/// (empty, non-positive sample rate, silence) return `0`.
#[must_use]
pub fn speech_transmission_index(ir: &[Sample], sample_rate: u32, snr_db: Sample) -> Sample {
    SpeechTransmissionIndex::from_ir(ir, sample_rate, snr_db).sti
}

/// The speech transmission index, its per-band modulation transfer indices,
/// and the qualitative rating.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpeechTransmissionIndex {
    /// Overall speech transmission index in `[0, 1]`.
    pub sti: Sample,
    /// Per-band modulation transfer indices `MTI_k`, each in `[0, 1]`.
    pub mti_per_band: [Sample; OCTAVE_COUNT],
    /// Qualitative rating derived from `sti`.
    pub rating: StiRating,
}

impl Default for SpeechTransmissionIndex {
    fn default() -> Self {
        Self {
            sti: 0.0,
            mti_per_band: [0.0; OCTAVE_COUNT],
            rating: StiRating::Bad,
        }
    }
}

impl SpeechTransmissionIndex {
    /// Computes the index from an impulse response at `sample_rate` with a
    /// per-band `snr_db`. Degenerate inputs return the safe default.
    ///
    /// ```
    /// use prism_audio_spatial::speech_transmission_index::{
    ///     SpeechTransmissionIndex, StiRating,
    /// };
    ///
    /// // A near-anechoic response (a single direct arrival) is highly
    /// // intelligible.
    /// let sr = 48_000u32;
    /// let mut ir = vec![0.0f32; 4_800];
    /// ir[0] = 1.0;
    /// let result = SpeechTransmissionIndex::from_ir(&ir, sr, 100.0);
    /// assert!(result.sti > 0.9);
    /// assert_eq!(result.rating, StiRating::Excellent);
    /// ```
    #[must_use]
    pub fn from_ir(ir: &[Sample], sample_rate: u32, snr_db: Sample) -> Self {
        let len = ir.len();
        if len == 0 || sample_rate == 0 {
            return Self::default();
        }
        let sr_f32 = sample_rate as Sample;
        let sr_f64 = f64::from(sample_rate);
        let noise = noise_factor(snr_db);

        let mut scratch: Vec<Sample> = vec![0.0; len];
        let mut mti_per_band = [0.0; OCTAVE_COUNT];
        for (band_index, &center) in OCTAVE_CENTERS_HZ.iter().enumerate() {
            if let Some(coeffs) = bandpass_coeffs(center, sr_f32) {
                filter_band(ir, &coeffs, &mut scratch);
                mti_per_band[band_index] = band_mti(&scratch, sr_f64, noise);
            }
        }

        let mut sti = 0.0;
        for (alpha, &mti) in MALE_ALPHA.iter().zip(mti_per_band.iter()) {
            sti += alpha * mti;
        }
        for (beta, pair) in MALE_BETA.iter().zip(mti_per_band.windows(2)) {
            sti -= beta * ops::sqrt((pair[0] * pair[1]).max(0.0));
        }
        let sti = sti.clamp(0.0, 1.0);

        Self {
            sti,
            mti_per_band,
            rating: StiRating::from_sti(sti),
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

    /// Direct spike plus an exponentially decaying reverberant noise tail with
    /// the given amplitude time constant (in samples). A physically meaningful
    /// room impulse response is band-limited noise under a decaying envelope,
    /// so the tail carries broadband energy whose squared envelope smears the
    /// speech modulations. A longer time constant means more smearing. The
    /// noise is a deterministic xorshift sequence for reproducible tests.
    fn reverberant_ir(tail_tau_samples: Sample, len: usize) -> Vec<Sample> {
        let mut ir = vec![0.0; len];
        ir[0] = 1.0;
        if tail_tau_samples > 0.0 {
            let mut state = 0x1234_5678u32;
            for (n, slot) in ir.iter_mut().enumerate().skip(1) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let noise = (state as Sample / u32::MAX as Sample) * 2.0 - 1.0;
                let env = ops::exp(-(n as Sample) / tail_tau_samples);
                *slot = 0.3 * noise * env;
            }
        }
        ir
    }

    #[test]
    fn pure_direct_is_excellent() {
        let mut ir = vec![0.0; 4_800];
        ir[0] = 1.0;
        let result = SpeechTransmissionIndex::from_ir(&ir, SR, 100.0);
        assert!(result.sti > 0.9, "sti {}", result.sti);
        assert_eq!(result.rating, StiRating::Excellent);
    }

    #[test]
    fn reverberation_lowers_sti() {
        let dry = speech_transmission_index(&reverberant_ir(0.0, 24_000), SR, 100.0);
        let wet = speech_transmission_index(&reverberant_ir(6_000.0, 24_000), SR, 100.0);
        assert!(wet < dry, "wet {wet} dry {dry}");
    }

    #[test]
    fn extreme_reverberation_is_poor_or_worse() {
        let sti = speech_transmission_index(&reverberant_ir(24_000.0, 48_000), SR, 100.0);
        assert!(sti < 0.45, "sti {sti}");
    }

    #[test]
    fn lower_snr_lowers_sti() {
        let ir = reverberant_ir(3_000.0, 24_000);
        let high = speech_transmission_index(&ir, SR, 30.0);
        let mid = speech_transmission_index(&ir, SR, 5.0);
        let low = speech_transmission_index(&ir, SR, -5.0);
        assert!(mid < high, "mid {mid} high {high}");
        assert!(low < mid, "low {low} mid {mid}");
    }

    #[test]
    fn empty_ir_is_default() {
        let empty: [Sample; 0] = [];
        let result = SpeechTransmissionIndex::from_ir(&empty, SR, 100.0);
        assert_eq!(result, SpeechTransmissionIndex::default());
        assert_eq!(speech_transmission_index(&empty, SR, 100.0), 0.0);
    }

    #[test]
    fn all_zero_ir_is_zero() {
        let ir = vec![0.0; 9_600];
        let result = SpeechTransmissionIndex::from_ir(&ir, SR, 100.0);
        assert_eq!(result.sti, 0.0);
        assert_eq!(result.rating, StiRating::Bad);
    }

    #[test]
    fn non_finite_samples_are_safe() {
        let mut ir = reverberant_ir(3_000.0, 24_000);
        ir[5] = Sample::NAN;
        ir[9] = Sample::INFINITY;
        ir[13] = Sample::NEG_INFINITY;
        let sti = speech_transmission_index(&ir, SR, 100.0);
        assert!(sti.is_finite() && (0.0..=1.0).contains(&sti), "sti {sti}");
    }

    #[test]
    fn zero_sample_rate_is_zero() {
        let ir = reverberant_ir(3_000.0, 24_000);
        assert_eq!(speech_transmission_index(&ir, 0, 100.0), 0.0);
    }

    #[test]
    fn rating_boundaries_are_correct() {
        assert_eq!(StiRating::from_sti(0.0), StiRating::Bad);
        assert_eq!(StiRating::from_sti(0.29), StiRating::Bad);
        assert_eq!(StiRating::from_sti(0.3), StiRating::Poor);
        assert_eq!(StiRating::from_sti(0.44), StiRating::Poor);
        assert_eq!(StiRating::from_sti(0.45), StiRating::Fair);
        assert_eq!(StiRating::from_sti(0.59), StiRating::Fair);
        assert_eq!(StiRating::from_sti(0.6), StiRating::Good);
        assert_eq!(StiRating::from_sti(0.74), StiRating::Good);
        assert_eq!(StiRating::from_sti(0.75), StiRating::Excellent);
        assert_eq!(StiRating::from_sti(1.0), StiRating::Excellent);
    }

    #[test]
    fn modulation_index_in_unit_interval() {
        let ir = reverberant_ir(3_000.0, 24_000);
        let coeffs = bandpass_coeffs(1_000.0, SR as Sample).unwrap();
        let mut band = vec![0.0; ir.len()];
        filter_band(&ir, &coeffs, &mut band);
        for &fm in &MODULATION_FREQS_HZ {
            let m = modulation_index(&band, f64::from(SR), fm, 1.0);
            assert!((0.0..=MTF_LIMIT).contains(&m), "m {m} at {fm} Hz");
        }
    }

    #[test]
    fn mti_values_in_unit_interval() {
        let result = SpeechTransmissionIndex::from_ir(&reverberant_ir(6_000.0, 24_000), SR, 20.0);
        for &mti in &result.mti_per_band {
            assert!((0.0..=1.0).contains(&mti), "mti {mti}");
        }
    }

    #[test]
    fn weight_arrays_have_standard_lengths_and_sum() {
        assert_eq!(MALE_ALPHA.len(), OCTAVE_COUNT);
        assert_eq!(MALE_BETA.len(), OCTAVE_COUNT - 1);
        assert_eq!(MODULATION_FREQS_HZ.len(), MODULATION_COUNT);
        assert_eq!(OCTAVE_CENTERS_HZ.len(), OCTAVE_COUNT);
        // All bands transferring perfectly (MTI = 1) must give exactly STI = 1,
        // which requires sum(alpha) - sum(beta) = 1.
        let alpha_sum: Sample = MALE_ALPHA.iter().sum();
        let beta_sum: Sample = MALE_BETA.iter().sum();
        assert!(approx(alpha_sum - beta_sum, 1.0, 1e-6), "alpha-beta {}", alpha_sum - beta_sum);
    }

    #[test]
    fn from_ir_matches_free_function() {
        let ir = reverberant_ir(4_000.0, 24_000);
        let s = SpeechTransmissionIndex::from_ir(&ir, SR, 25.0);
        assert!(approx(s.sti, speech_transmission_index(&ir, SR, 25.0), 1e-6));
    }

    #[test]
    fn default_is_zero_and_bad() {
        let d = SpeechTransmissionIndex::default();
        assert_eq!(d.sti, 0.0);
        assert_eq!(d.mti_per_band, [0.0; OCTAVE_COUNT]);
        assert_eq!(d.rating, StiRating::Bad);
    }
}
