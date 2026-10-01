//! Dietsch-Kraak echo criterion `EK` (echo audibility) from a room impulse
//! response.
//!
//! A strong, isolated late reflection is heard as a discrete echo rather than
//! as useful reverberation. The Dietsch-Kraak echo criterion predicts this
//! audibility from a single broadband room impulse response by measuring how
//! fast the energy centre of gravity (the build-up function) moves across a
//! short perceptual window.
//!
//! # Model
//!
//! Let `p(t)` be the impulse response. The build-up function is the running
//! first temporal moment weighted by `|p(t)|^n`:
//! `ts(tau) = (integral_0^tau |p(t)|^n * t dt) / (integral_0^tau |p(t)|^n dt)`.
//! Discretised with `t_i = i / sample_rate`, the integrals become prefix sums,
//! so `ts[k] = (sum_{i<=k} t_i * |p_i|^n) / (sum_{i<=k} |p_i|^n)` seconds. The
//! common sample-interval factor cancels in the ratio.
//!
//! The echo criterion is the largest rise of the build-up function across one
//! window of width `delta_tau_E`:
//! `EK = max_tau (ts(tau) - ts(tau - delta_tau_E)) / delta_tau_E`.
//! A smoothly decaying response moves its centroid slowly (small `EK`), while
//! an isolated late reflection shifts the centroid abruptly (large `EK`).
//!
//! Two listening modes use different exponents, windows, and thresholds:
//! speech uses `n = 2/3`, `delta_tau_E = 9 ms`, perceptible above `EK = 1.0`;
//! music uses `n = 1`, `delta_tau_E = 14 ms`, perceptible above `EK = 1.5`
//! (roughly the level at which half of listeners report an echo).
//!
//! # Relationship
//!
//! This module complements [`crate::center_time`], which reports the single
//! energy centre of gravity `Ts` of the whole response; here the running
//! build-up function `ts(tau)` is reused only as an intermediate, and the
//! reported quantity is its maximum windowed slope, an echo-audibility
//! criterion rather than a single instant. It also complements the clarity and
//! reverberation measures in [`crate::room_clarity`] and the per-band spectrum
//! in [`crate::reverberation_spectrum`]. All share the [`Sample`] scalar from
//! [`prism_audio_core`] and none reimplements another.
//!
//! # Real-time contract
//!
//! This is a control-rate, offline estimator. It performs a single bounded heap
//! allocation of one scratch buffer the length of the response and no
//! per-sample allocation. It is not a per-sample callback and must not run on
//! an audio thread. It never panics: empty, all-zero, non-finite, or
//! non-positive sample-rate inputs return the safe default `0` (not
//! perceptible).
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published Dietsch and
//! Kraak (1986) echo criterion. It is pure classic DSP with no AI or ML. It is
//! engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented acoustics literature.

use alloc::{vec, vec::Vec};
use bevy_math::ops;
use prism_audio_core::math::Sample;

/// Speech sliding-window width `delta_tau_E` in milliseconds.
pub const SPEECH_WINDOW_MS: Sample = 9.0;

/// Music sliding-window width `delta_tau_E` in milliseconds.
pub const MUSIC_WINDOW_MS: Sample = 14.0;

/// Speech perceptibility threshold: `EK` above this is heard as an echo.
pub const SPEECH_THRESHOLD: Sample = 1.0;

/// Music perceptibility threshold: `EK` above this is heard as an echo.
pub const MUSIC_THRESHOLD: Sample = 1.5;

/// Speech weighting exponent `n` applied to `|p(t)|`.
pub const SPEECH_EXPONENT: Sample = 2.0 / 3.0;

/// Music weighting exponent `n` applied to `|p(t)|`.
pub const MUSIC_EXPONENT: Sample = 1.0;

/// Listening mode selecting the Dietsch-Kraak exponent, window, and threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum EchoMode {
    /// Speech: `n = 2/3`, `delta_tau_E = 9 ms`, threshold `1.0`.
    #[default]
    Speech,
    /// Music: `n = 1`, `delta_tau_E = 14 ms`, threshold `1.5`.
    Music,
}

impl EchoMode {
    /// Sliding-window width `delta_tau_E` in milliseconds.
    #[must_use]
    pub fn window_ms(self) -> Sample {
        match self {
            Self::Speech => SPEECH_WINDOW_MS,
            Self::Music => MUSIC_WINDOW_MS,
        }
    }

    /// Perceptibility threshold on `EK`.
    #[must_use]
    pub fn threshold(self) -> Sample {
        match self {
            Self::Speech => SPEECH_THRESHOLD,
            Self::Music => MUSIC_THRESHOLD,
        }
    }

    /// Weighting exponent `n` applied to `|p(t)|`.
    #[must_use]
    pub fn exponent(self) -> Sample {
        match self {
            Self::Speech => SPEECH_EXPONENT,
            Self::Music => MUSIC_EXPONENT,
        }
    }
}

/// Computes the Dietsch-Kraak echo criterion `EK` for a room impulse response.
///
/// Returns the maximum windowed slope of the build-up function. Degenerate
/// inputs (empty, non-positive sample rate, silence) return `0`. Non-finite
/// samples are treated as `0`.
#[must_use]
pub fn echo_criterion(ir: &[Sample], sample_rate: u32, mode: EchoMode) -> Sample {
    let len = ir.len();
    if len == 0 || sample_rate == 0 {
        return 0.0;
    }
    let exponent = mode.exponent();
    let inv_sr = 1.0_f64 / f64::from(sample_rate);

    // Running build-up function ts[k], in seconds, from prefix moments.
    let mut build_up: Vec<f64> = vec![0.0; len];
    let mut moment = 0.0_f64; // sum t_i * |p_i|^n
    let mut weight = 0.0_f64; // sum |p_i|^n
    for (i, &sample) in ir.iter().enumerate() {
        let magnitude = if sample.is_finite() { sample.abs() } else { 0.0 };
        let w = f64::from(ops::powf(magnitude, exponent));
        moment += (i as f64 * inv_sr) * w;
        weight += w;
        build_up[i] = if weight > 0.0 { moment / weight } else { 0.0 };
    }

    // Window width in seconds and samples (at least one sample).
    let window_s = f64::from(mode.window_ms()) * 1.0e-3;
    let sr = sample_rate as Sample;
    let window_samples = (ops::round(mode.window_ms() * 1.0e-3 * sr) as usize).max(1);
    if window_samples >= len {
        return 0.0;
    }

    // Maximum positive rise of the build-up function across one window.
    let mut ek = 0.0_f64;
    for k in window_samples..len {
        let slope = (build_up[k] - build_up[k - window_samples]) / window_s;
        if slope > ek {
            ek = slope;
        }
    }
    ek as Sample
}

/// The Dietsch-Kraak echo criterion and its perceptibility verdict for one
/// listening mode.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EchoCriterion {
    /// Echo criterion value `EK` (dimensionless build-up slope).
    pub ek: Sample,
    /// Whether `EK` exceeds the mode threshold (an echo is likely audible).
    pub perceptible: bool,
    /// The listening mode used for the exponent, window, and threshold.
    pub mode: EchoMode,
}

impl Default for EchoCriterion {
    fn default() -> Self {
        Self {
            ek: 0.0,
            perceptible: false,
            mode: EchoMode::Speech,
        }
    }
}

impl EchoCriterion {
    /// Computes the echo criterion from an impulse response in a listening mode.
    ///
    /// Degenerate inputs yield `ek = 0` and `perceptible = false`.
    ///
    /// ```
    /// use prism_audio_spatial::echo_criterion::{EchoCriterion, EchoMode};
    ///
    /// // Direct sound at t = 0 plus a strong, isolated echo at 100 ms.
    /// let sr = 48_000u32;
    /// let mut ir = vec![0.0f32; 9_600];
    /// ir[0] = 1.0;
    /// ir[4_800] = 0.9;
    /// let result = EchoCriterion::from_ir(&ir, sr, EchoMode::Music);
    /// assert!(result.perceptible); // a late, strong reflection is audible
    /// ```
    #[must_use]
    pub fn from_ir(ir: &[Sample], sample_rate: u32, mode: EchoMode) -> Self {
        let ek = echo_criterion(ir, sample_rate, mode);
        Self {
            ek,
            perceptible: ek > mode.threshold(),
            mode,
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

    /// Builds an impulse response with a direct spike and one delayed echo.
    fn direct_plus_echo(delay_ms: Sample, echo_amp: Sample, len: usize) -> Vec<Sample> {
        let mut ir = vec![0.0; len];
        ir[0] = 1.0;
        let idx = ops::round(delay_ms * 1.0e-3 * SR as Sample) as usize;
        if idx < len {
            ir[idx] = echo_amp;
        }
        ir
    }

    #[test]
    fn pure_direct_impulse_has_zero_ek() {
        let mut ir = vec![0.0; 4_800];
        ir[0] = 1.0;
        let ek_music = echo_criterion(&ir, SR, EchoMode::Music);
        let ek_speech = echo_criterion(&ir, SR, EchoMode::Speech);
        assert!(approx(ek_music, 0.0, 1e-6), "music ek {ek_music}");
        assert!(approx(ek_speech, 0.0, 1e-6), "speech ek {ek_speech}");
        assert!(!EchoCriterion::from_ir(&ir, SR, EchoMode::Music).perceptible);
    }

    #[test]
    fn strong_delayed_echo_is_perceptible() {
        let ir = direct_plus_echo(100.0, 0.9, 9_600);
        let music = EchoCriterion::from_ir(&ir, SR, EchoMode::Music);
        let speech = EchoCriterion::from_ir(&ir, SR, EchoMode::Speech);
        assert!(music.ek > MUSIC_THRESHOLD, "music ek {}", music.ek);
        assert!(speech.ek > SPEECH_THRESHOLD, "speech ek {}", speech.ek);
        assert!(music.perceptible && speech.perceptible);
    }

    #[test]
    fn stronger_echo_increases_ek() {
        let weak = echo_criterion(&direct_plus_echo(50.0, 0.3, 9_600), SR, EchoMode::Music);
        let mid = echo_criterion(&direct_plus_echo(50.0, 0.6, 9_600), SR, EchoMode::Music);
        let strong = echo_criterion(&direct_plus_echo(50.0, 0.9, 9_600), SR, EchoMode::Music);
        assert!(weak < mid, "weak {weak} mid {mid}");
        assert!(mid < strong, "mid {mid} strong {strong}");
    }

    #[test]
    fn later_echo_increases_ek() {
        let near = echo_criterion(&direct_plus_echo(30.0, 0.8, 9_600), SR, EchoMode::Music);
        let far = echo_criterion(&direct_plus_echo(90.0, 0.8, 9_600), SR, EchoMode::Music);
        assert!(near < far, "near {near} far {far}");
    }

    #[test]
    fn exponent_affects_result() {
        // A mixed response is weighted differently by n = 2/3 vs n = 1, so the
        // two modes generally yield different criterion values.
        let ir = direct_plus_echo(60.0, 0.5, 9_600);
        let speech = echo_criterion(&ir, SR, EchoMode::Speech);
        let music = echo_criterion(&ir, SR, EchoMode::Music);
        assert!(speech > 0.0 && music > 0.0);
        assert!(!approx(speech, music, 1e-4), "speech {speech} music {music}");
    }

    #[test]
    fn mode_parameters_differ() {
        assert!(approx(EchoMode::Speech.window_ms(), 9.0, 1e-6));
        assert!(approx(EchoMode::Music.window_ms(), 14.0, 1e-6));
        assert!(approx(EchoMode::Speech.threshold(), 1.0, 1e-6));
        assert!(approx(EchoMode::Music.threshold(), 1.5, 1e-6));
        assert!(approx(EchoMode::Speech.exponent(), 2.0 / 3.0, 1e-6));
        assert!(approx(EchoMode::Music.exponent(), 1.0, 1e-6));
    }

    #[test]
    fn smooth_decay_is_not_perceptible() {
        // An exponentially decaying envelope moves its centroid slowly.
        let tau = 2_400.0; // samples
        let ir: Vec<Sample> = (0..24_000)
            .map(|n| ops::exp(-(n as Sample) / tau))
            .collect();
        let music = EchoCriterion::from_ir(&ir, SR, EchoMode::Music);
        let speech = EchoCriterion::from_ir(&ir, SR, EchoMode::Speech);
        assert!(!music.perceptible, "music ek {}", music.ek);
        assert!(!speech.perceptible, "speech ek {}", speech.ek);
    }

    #[test]
    fn empty_ir_is_zero() {
        let empty: [Sample; 0] = [];
        assert_eq!(echo_criterion(&empty, SR, EchoMode::Music), 0.0);
        let c = EchoCriterion::from_ir(&empty, SR, EchoMode::Music);
        assert_eq!(c.ek, 0.0);
        assert!(!c.perceptible);
    }

    #[test]
    fn all_zero_ir_is_zero() {
        let ir = vec![0.0; 9_600];
        assert_eq!(echo_criterion(&ir, SR, EchoMode::Speech), 0.0);
    }

    #[test]
    fn zero_sample_rate_is_zero() {
        let ir = direct_plus_echo(50.0, 0.9, 9_600);
        assert_eq!(echo_criterion(&ir, 0, EchoMode::Music), 0.0);
    }

    #[test]
    fn non_finite_samples_are_safe() {
        let mut ir = direct_plus_echo(50.0, 0.9, 9_600);
        ir[10] = Sample::NAN;
        ir[20] = Sample::INFINITY;
        ir[30] = Sample::NEG_INFINITY;
        let ek = echo_criterion(&ir, SR, EchoMode::Music);
        assert!(ek.is_finite() && ek >= 0.0, "ek {ek}");
    }

    #[test]
    fn window_longer_than_response_is_zero() {
        // Only a few samples: shorter than any mode window.
        let ir = vec![1.0, 0.0, 0.5, 0.2];
        assert_eq!(echo_criterion(&ir, SR, EchoMode::Music), 0.0);
    }

    #[test]
    fn perceptible_matches_threshold() {
        let ir = direct_plus_echo(100.0, 0.9, 9_600);
        let c = EchoCriterion::from_ir(&ir, SR, EchoMode::Music);
        assert_eq!(c.perceptible, c.ek > c.mode.threshold());
    }

    #[test]
    fn from_ir_matches_free_function() {
        let ir = direct_plus_echo(70.0, 0.6, 9_600);
        let c = EchoCriterion::from_ir(&ir, SR, EchoMode::Speech);
        assert!(approx(c.ek, echo_criterion(&ir, SR, EchoMode::Speech), 1e-6));
        assert_eq!(c.mode, EchoMode::Speech);
    }

    #[test]
    fn default_is_zero_and_not_perceptible() {
        let d = EchoCriterion::default();
        assert_eq!(d.ek, 0.0);
        assert!(!d.perceptible);
        assert_eq!(d.mode, EchoMode::Speech);
    }
}
