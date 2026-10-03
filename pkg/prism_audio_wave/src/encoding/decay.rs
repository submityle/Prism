//! Decay-time encoding: Schroeder backward integration of the impulse
//! response into an energy decay curve, and a least-squares reverberation-time
//! estimate from it.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the "`RT60` / `EDC` decay time -> reverb wet amount" measure of
//! design section 43. The energy decay curve (`EDC`) is the Schroeder integral
//! of the squared pressure; the reverberation time `RT60` is read from the
//! slope of its decibel curve, extrapolated to a 60 `dB` drop.

use alloc::vec;
use alloc::vec::Vec;
use bevy_math::ops;

use crate::solver::ImpulseResponse;

/// Lower decibel bound of the slope-fitting region (relative to onset energy).
pub const FIT_UPPER_DB: f32 = -5.0;
/// Upper decibel bound of the slope-fitting region (relative to onset energy).
pub const FIT_LOWER_DB: f32 = -35.0;

/// Computes the Schroeder energy decay curve from `onset`, in decibels
/// relative to the energy at `onset`.
///
/// The returned curve starts at `0` `dB` and decreases monotonically. It is
/// empty when the response carries no energy after `onset`.
#[must_use]
pub fn schroeder_edc_db(ir: &ImpulseResponse, onset: usize) -> Vec<f32> {
    let samples = ir.samples();
    let start = onset.min(samples.len());
    let tail = &samples[start..];
    let n = tail.len();
    if n == 0 {
        return Vec::new();
    }
    let mut edc = vec![0.0_f32; n];
    let mut acc = 0.0_f32;
    for i in (0..n).rev() {
        acc += tail[i] * tail[i];
        edc[i] = acc;
    }
    let e0 = edc[0];
    if e0 <= 0.0 {
        return Vec::new();
    }
    for value in &mut edc {
        *value = 10.0 * ops::log10((*value / e0).max(1.0e-12));
    }
    edc
}

/// Estimates the reverberation time `RT60` in seconds from a decibel `edc`
/// curve sampled at `sample_rate` Hz.
///
/// A least-squares line is fitted to the samples lying in the
/// `[FIT_LOWER_DB, FIT_UPPER_DB]` band and extrapolated to a 60 `dB` decay.
/// Returns `0` when the curve is too short or too flat to fit.
#[must_use]
pub fn rt60_from_edc(edc: &[f32], sample_rate: f32) -> f32 {
    if edc.len() < 2 || sample_rate <= 0.0 {
        return 0.0;
    }
    // Collect (sample_index, db) pairs inside the fit band.
    let mut n = 0.0_f32;
    let mut sum_x = 0.0_f32;
    let mut sum_y = 0.0_f32;
    let mut sum_xx = 0.0_f32;
    let mut sum_xy = 0.0_f32;
    for (i, &db) in edc.iter().enumerate() {
        if (FIT_LOWER_DB..=FIT_UPPER_DB).contains(&db) {
            let x = i as f32;
            n += 1.0;
            sum_x += x;
            sum_y += db;
            sum_xx += x * x;
            sum_xy += x * db;
        }
    }
    if n < 2.0 {
        return 0.0;
    }
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() <= f32::EPSILON {
        return 0.0;
    }
    // Slope in dB per sample; a decaying curve has a negative slope.
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    if slope >= 0.0 {
        return 0.0;
    }
    let samples_for_60db = -60.0 / slope;
    samples_for_60db / sample_rate
}

/// Convenience wrapper: the `RT60` of `ir` measured from `onset`.
#[must_use]
pub fn rt60(ir: &ImpulseResponse, onset: usize) -> f32 {
    let edc = schroeder_edc_db(ir, onset);
    rt60_from_edc(&edc, ir.sample_rate())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn exp_decay_ir(sample_rate: f32, len: usize, a: f32) -> ImpulseResponse {
        let mut p = vec![0.0_f32; len];
        for (n, v) in p.iter_mut().enumerate() {
            *v = ops::exp(-a * n as f32);
        }
        ImpulseResponse::new(sample_rate, p)
    }

    #[test]
    fn edc_is_monotonically_decreasing() {
        let ir = exp_decay_ir(1000.0, 500, 0.02);
        let edc = schroeder_edc_db(&ir, 0);
        assert!(!edc.is_empty());
        assert!(approx(edc[0], 0.0, 1e-6));
        for w in edc.windows(2) {
            assert!(w[1] <= w[0] + 1e-4, "edc rose: {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn rt60_matches_analytic_exponential() {
        // p[n] = exp(-a n) -> EDC dB slope = -20 a / ln(10) per sample.
        // RT60 = 3 ln(10) / (a * sample_rate).
        let sr = 1000.0_f32;
        let a = 0.01_f32;
        let ir = exp_decay_ir(sr, 1000, a);
        let got = rt60(&ir, 0);
        let expected = 3.0 * ops::ln(10.0) / (a * sr);
        assert!(
            approx(got, expected, expected * 0.05),
            "rt60 {got} vs expected {expected}"
        );
    }

    #[test]
    fn faster_decay_has_shorter_rt60() {
        let sr = 2000.0_f32;
        let slow = rt60(&exp_decay_ir(sr, 2000, 0.005), 0);
        let fast = rt60(&exp_decay_ir(sr, 2000, 0.02), 0);
        assert!(fast < slow, "fast {fast} should be shorter than slow {slow}");
    }

    #[test]
    fn silent_response_has_zero_rt60() {
        let ir = ImpulseResponse::silent(1000.0, 64);
        assert_eq!(rt60(&ir, 0), 0.0);
    }
}
