//! Time-domain impulse responses: the raw solver output sampled at a probe
//! cell, plus the small set of time/energy queries the perceptual encoder
//! ([`crate::encoding`]) builds on.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Represents the "time-domain impulse response" sampled on the probe grid in
//! design section 43, before it is reduced to perceptual parameters. Pressure
//! is a plain real signal uniformly sampled at the solver time step.

use alloc::vec;
use alloc::vec::Vec;
use bevy_math::ops;

/// A uniformly sampled pressure impulse response at one probe cell.
///
/// `sample_rate` is the reciprocal of the solver time step; `pressure[n]` is
/// the pressure at time `n / sample_rate` seconds after the source fires.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ImpulseResponse {
    sample_rate: f32,
    pressure: Vec<f32>,
}

impl ImpulseResponse {
    /// Builds an impulse response from a pressure signal and its sample rate
    /// (Hz). The sample rate is clamped to a small positive floor.
    #[must_use]
    pub fn new(sample_rate: f32, pressure: Vec<f32>) -> Self {
        Self {
            sample_rate: sample_rate.max(1.0e-3),
            pressure,
        }
    }

    /// A silent response of `len` samples at `sample_rate` Hz.
    #[must_use]
    pub fn silent(sample_rate: f32, len: usize) -> Self {
        Self::new(sample_rate, vec![0.0; len])
    }

    /// Sampling rate in Hz.
    #[must_use]
    #[inline]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Number of samples.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.pressure.len()
    }

    /// Returns `true` when the response holds no samples.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pressure.is_empty()
    }

    /// The pressure samples.
    #[must_use]
    #[inline]
    pub fn samples(&self) -> &[f32] {
        &self.pressure
    }

    /// Mutable access to the pressure samples (used while a solve fills the
    /// buffer).
    #[must_use]
    #[inline]
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.pressure
    }

    /// Duration of the response in seconds.
    #[must_use]
    #[inline]
    pub fn duration_s(&self) -> f32 {
        self.pressure.len() as f32 / self.sample_rate
    }

    /// Total energy, the sum of squared pressure samples.
    #[must_use]
    pub fn total_energy(&self) -> f32 {
        self.pressure.iter().map(|p| p * p).sum()
    }

    /// Energy within the half-open sample window `[start, end)`, clamped to the
    /// valid range.
    #[must_use]
    pub fn window_energy(&self, start: usize, end: usize) -> f32 {
        let end = end.min(self.pressure.len());
        if start >= end {
            return 0.0;
        }
        self.pressure[start..end].iter().map(|p| p * p).sum()
    }

    /// Converts a time in milliseconds to the nearest sample index.
    #[must_use]
    pub fn sample_at_ms(&self, ms: f32) -> usize {
        let idx = ops::round(ms * 1.0e-3 * self.sample_rate);
        if idx <= 0.0 {
            0
        } else {
            idx as usize
        }
    }

    /// Index of the first sample whose magnitude reaches `threshold` times the
    /// peak magnitude, i.e. the onset / direct arrival. Returns `0` for a
    /// silent response.
    #[must_use]
    pub fn onset_index(&self, threshold: f32) -> usize {
        let peak = self
            .pressure
            .iter()
            .fold(0.0_f32, |m, p| m.max(p.abs()));
        if peak <= 0.0 {
            return 0;
        }
        let limit = peak * threshold.clamp(0.0, 1.0);
        for (i, p) in self.pressure.iter().enumerate() {
            if p.abs() >= limit {
                return i;
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn energy_windows_add_up() {
        let ir = ImpulseResponse::new(1000.0, vec![0.0, 1.0, 2.0, 2.0, 1.0]);
        let total = ir.total_energy();
        let a = ir.window_energy(0, 3);
        let b = ir.window_energy(3, 5);
        assert!(approx(a + b, total, 1e-5));
        assert!(approx(total, 10.0, 1e-5));
    }

    #[test]
    fn onset_finds_first_significant_sample() {
        let ir = ImpulseResponse::new(1000.0, vec![0.0, 0.0, 0.01, 1.0, 0.2]);
        assert_eq!(ir.onset_index(0.5), 3);
        assert_eq!(ir.onset_index(0.001), 2);
    }

    #[test]
    fn silent_response_has_no_energy() {
        let ir = ImpulseResponse::silent(48_000.0, 16);
        assert_eq!(ir.total_energy(), 0.0);
        assert_eq!(ir.onset_index(0.5), 0);
        assert!(approx(ir.duration_s(), 16.0 / 48_000.0, 1e-9));
    }

    #[test]
    fn sample_at_ms_rounds() {
        let ir = ImpulseResponse::silent(1000.0, 100);
        assert_eq!(ir.sample_at_ms(10.0), 10);
        assert_eq!(ir.sample_at_ms(-5.0), 0);
    }
}
