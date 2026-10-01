//! ISO 3382-1 centre time `Ts` from a measured room impulse response.
//!
//! Centre time is the time of the centre of gravity (first temporal moment) of
//! the squared impulse response. It is a single-number objective measure of the
//! balance between clarity and reverberance: a small `Ts` indicates a clear,
//! direct-dominated sound, while a large `Ts` indicates a reverberant sound. It
//! avoids the hard early-to-late split used by the clarity ratios.
//!
//! # Model
//!
//! Let `p[n]` be the measured room impulse response and `t_n = n / sample_rate`
//! the arrival time in seconds. The centre time is
//! `Ts = sum_n (t_n * p[n]^2) / sum_n p[n]^2` seconds, usually reported in
//! milliseconds. The numerator and denominator are accumulated in double
//! precision to preserve accuracy over long responses, then read back as a
//! [`Sample`].
//!
//! # Relationship
//!
//! This module complements [`crate::room_clarity`] (the clarity, definition, and
//! reverberation-time ratios), [`crate::sound_strength`] (the energy-level `G`),
//! and [`crate::spatial_impression`] (lateral energy and interaural
//! correlation), as well as the geometry-based estimate in
//! [`crate::room_acoustics`]. Where those express ratios or levels, this module
//! reports the single energy centre-of-gravity instant. All share the
//! [`Sample`] scalar from [`prism_audio_core`] and none reimplements another.
//!
//! # Real-time contract
//!
//! This is a control-rate, offline estimator: it accepts a whole impulse
//! response and performs no heap allocation. It is not a per-sample callback and
//! must not run on an audio thread. It never panics: empty, all-zero,
//! non-finite, or non-positive sample-rate inputs return the safe default `0`.
//!
//! # Provenance
//!
//! This is a textbook implementation of the publicly published ISO 3382-1
//! centre-time parameter `Ts`. It is pure classic DSP with no AI or ML. It is
//! engine-agnostic and contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented standard.

use prism_audio_core::math::Sample;

/// Sanitises a sample, mapping non-finite values to `0`.
#[inline]
fn finite(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
}

/// Accumulates the centre-time numerator `sum t_n * p^2` and denominator
/// `sum p^2` in double precision, returning `(numerator, denominator)`.
fn moments(response: &[Sample], sample_rate: u32) -> (f64, f64) {
    let inv_sr = 1.0_f64 / f64::from(sample_rate);
    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    for (n, &sample) in response.iter().enumerate() {
        let v = f64::from(finite(sample));
        let energy = v * v;
        let t = n as f64 * inv_sr;
        numerator += t * energy;
        denominator += energy;
    }
    (numerator, denominator)
}

/// Computes the centre time `Ts` in seconds.
///
/// `Ts = sum_n (t_n * p[n]^2) / sum_n p[n]^2` with `t_n = n / sample_rate`. An
/// empty or silent response, or a sample rate of `0`, returns `0`.
#[must_use]
pub fn center_time_seconds(response: &[Sample], sample_rate: u32) -> Sample {
    if response.is_empty() || sample_rate == 0 {
        return 0.0;
    }
    let (numerator, denominator) = moments(response, sample_rate);
    if denominator <= 0.0 {
        return 0.0;
    }
    let ts = numerator / denominator;
    if ts.is_finite() { ts as Sample } else { 0.0 }
}

/// Computes the centre time `Ts` in milliseconds (seconds times `1000`).
///
/// Returns `0` for the same degenerate inputs as [`center_time_seconds`].
#[must_use]
pub fn center_time_ms(response: &[Sample], sample_rate: u32) -> Sample {
    center_time_seconds(response, sample_rate) * 1000.0
}

/// The ISO 3382-1 centre time of a measured room, in both seconds and
/// milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CenterTime {
    /// Centre time `Ts` in seconds.
    pub ts_seconds: Sample,
    /// Centre time `Ts` in milliseconds (`ts_seconds * 1000`).
    pub ts_ms: Sample,
}

impl Default for CenterTime {
    fn default() -> Self {
        Self {
            ts_seconds: 0.0,
            ts_ms: 0.0,
        }
    }
}

impl CenterTime {
    /// Computes the centre time from a measured impulse response in a single
    /// bounded scan.
    ///
    /// Empty, silent, non-finite, or non-positive sample-rate inputs yield `0`
    /// for both fields.
    ///
    /// ```
    /// use prism_audio_spatial::center_time::CenterTime;
    ///
    /// // Two equal-energy impulses: the centre of gravity is their midpoint.
    /// let mut rir = vec![0.0f32; 48_000];
    /// rir[0] = 1.0;
    /// rir[24_000] = 1.0;
    /// let ct = CenterTime::from_impulse_response(&rir, 48_000);
    /// // Midpoint of t = 0 s and t = 0.5 s is 0.25 s.
    /// assert!((ct.ts_seconds - 0.25).abs() < 1e-4);
    /// assert!((ct.ts_ms - 250.0).abs() < 1e-1);
    /// ```
    #[must_use]
    pub fn from_impulse_response(response: &[Sample], sample_rate: u32) -> Self {
        let ts_seconds = center_time_seconds(response, sample_rate);
        Self {
            ts_seconds,
            ts_ms: ts_seconds * 1000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    const SR: u32 = 48_000;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn dirac_at_origin_has_zero_center_time() {
        let rir = [1.0, 0.0, 0.0, 0.0];
        assert!(approx(center_time_seconds(&rir, SR), 0.0, 1e-9));
    }

    #[test]
    fn single_delayed_impulse_equals_its_arrival_time() {
        let mut rir = vec![0.0; 10_000];
        let delay = 4_800usize; // 100 ms at 48 kHz
        rir[delay] = 1.0;
        let expected = delay as Sample / SR as Sample;
        assert!(approx(center_time_seconds(&rir, SR), expected, 1e-6));
    }

    #[test]
    fn two_equal_impulses_center_at_midpoint() {
        let mut rir = vec![0.0; 20_000];
        rir[0] = 1.0;
        rir[12_000] = 1.0;
        let expected = 0.5 * (12_000 as Sample / SR as Sample);
        assert!(approx(center_time_seconds(&rir, SR), expected, 1e-6));
    }

    #[test]
    fn unequal_impulses_weight_toward_higher_energy() {
        // A strong impulse at t=0 and a weak one later: Ts closer to 0.
        let mut rir = vec![0.0; 20_000];
        rir[0] = 2.0; // energy 4
        rir[10_000] = 1.0; // energy 1
        let late_t = 10_000 as Sample / SR as Sample;
        let expected = (0.0 * 4.0 + late_t * 1.0) / 5.0;
        assert!(approx(center_time_seconds(&rir, SR), expected, 1e-6));
    }

    #[test]
    fn empty_response_is_zero() {
        let empty: [Sample; 0] = [];
        assert_eq!(center_time_seconds(&empty, SR), 0.0);
        assert_eq!(center_time_ms(&empty, SR), 0.0);
    }

    #[test]
    fn all_zero_response_is_zero() {
        let z = vec![0.0; 4_000];
        assert_eq!(center_time_seconds(&z, SR), 0.0);
        let ct = CenterTime::from_impulse_response(&z, SR);
        assert_eq!(ct.ts_seconds, 0.0);
        assert_eq!(ct.ts_ms, 0.0);
    }

    #[test]
    fn zero_sample_rate_is_zero() {
        let rir = [1.0, 0.5, 0.25];
        assert_eq!(center_time_seconds(&rir, 0), 0.0);
        assert_eq!(center_time_ms(&rir, 0), 0.0);
    }

    #[test]
    fn non_finite_samples_are_safe() {
        let mut rir = vec![0.0; 10_000];
        rir[0] = Sample::NAN;
        rir[1] = Sample::INFINITY;
        rir[4_800] = 1.0; // the only finite non-zero contributor
        let ts = center_time_seconds(&rir, SR);
        assert!(ts.is_finite());
        let expected = 4_800 as Sample / SR as Sample;
        assert!(approx(ts, expected, 1e-6), "ts {ts} expected {expected}");
    }

    #[test]
    fn ms_is_seconds_times_thousand() {
        let mut rir = vec![0.0; 10_000];
        rir[4_800] = 1.0;
        let s = center_time_seconds(&rir, SR);
        let ms = center_time_ms(&rir, SR);
        assert!(approx(ms, s * 1000.0, 1e-6));
    }

    #[test]
    fn default_is_all_zero() {
        let ct = CenterTime::default();
        assert_eq!(ct.ts_seconds, 0.0);
        assert_eq!(ct.ts_ms, 0.0);
    }

    #[test]
    fn from_impulse_response_matches_free_functions() {
        let mut rir = vec![0.0; 16_000];
        for (n, x) in rir.iter_mut().enumerate() {
            *x = ops_exp_decay(n);
        }
        let ct = CenterTime::from_impulse_response(&rir, SR);
        assert!(approx(ct.ts_seconds, center_time_seconds(&rir, SR), 1e-6));
        assert!(approx(ct.ts_ms, center_time_ms(&rir, SR), 1e-4));
    }

    #[test]
    fn longer_decay_has_larger_center_time() {
        let fast: Vec<Sample> = (0..16_000).map(|n| decay(n, 1_000.0)).collect();
        let slow: Vec<Sample> = (0..16_000).map(|n| decay(n, 4_000.0)).collect();
        assert!(center_time_seconds(&slow, SR) > center_time_seconds(&fast, SR));
    }

    fn decay(n: usize, tau: Sample) -> Sample {
        bevy_math::ops::exp(-(n as Sample) / tau)
    }

    fn ops_exp_decay(n: usize) -> Sample {
        decay(n, 2_000.0)
    }
}
