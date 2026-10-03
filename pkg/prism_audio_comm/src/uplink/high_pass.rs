//! DC removal and low-frequency rumble suppression for the uplink chain.
//!
//! Microphone captures carry a DC bias and sub-100 Hz rumble (handling noise,
//! HVAC, breath pops) that waste headroom and destabilise the adaptive echo
//! canceller downstream. This module removes them with two classic,
//! fully-specified options:
//!
//! * a first-order one-pole / one-zero DC blocker, and
//! * a second-order Butterworth high-pass biquad using the RBJ Audio EQ
//!   Cookbook coefficient formulas.
//!
//! Both run allocation-free in the real-time thread and expose per-sample and
//! per-block entry points.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the DC/high-pass stage of design section 45.2 (uplink
//! pre-processing). Built on the sample scalar and deterministic math of
//! `prism_audio_core`; it is the first stage of [`crate::uplink::UplinkChain`].

use bevy_math::ops;
use core::f32::consts::PI;
use prism_audio_core::math::{flush_denormal, Sample};

/// Filter order selection for [`HighPass`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum HighPassOrder {
    /// First-order one-pole / one-zero DC blocker (about 6 dB per octave).
    First,
    /// Second-order Butterworth high-pass (about 12 dB per octave).
    Second,
}

/// A fixed-coefficient high-pass filter with first- or second-order response.
///
/// The filter is realised as a transposed Direct Form II biquad; the
/// first-order mode simply loads a degenerate biquad whose second-order taps
/// are zero. Coefficients are recomputed only when [`HighPass::set_cutoff`] or
/// [`HighPass::new`] is called, never in the hot path.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HighPass {
    order: HighPassOrder,
    sample_rate: Sample,
    cutoff_hz: Sample,
    // Normalised biquad coefficients (a0 divided out).
    b0: Sample,
    b1: Sample,
    b2: Sample,
    a1: Sample,
    a2: Sample,
    // Transposed Direct Form II state.
    z1: Sample,
    z2: Sample,
}

impl HighPass {
    /// Creates a high-pass filter at `cutoff_hz` for the given `sample_rate`.
    ///
    /// The cutoff is clamped into the open interval `(0, nyquist)` so the
    /// coefficient math is always well defined.
    #[must_use]
    pub fn new(sample_rate: Sample, cutoff_hz: Sample, order: HighPassOrder) -> Self {
        let mut filter = Self {
            order,
            sample_rate: sample_rate.max(1.0),
            cutoff_hz: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        };
        filter.set_cutoff(cutoff_hz);
        filter
    }

    /// Returns the configured cutoff frequency in hertz.
    #[must_use]
    pub fn cutoff_hz(&self) -> Sample {
        self.cutoff_hz
    }

    /// Recomputes the coefficients for a new `cutoff_hz`.
    ///
    /// The internal delay state is preserved so the filter can be retuned
    /// without a discontinuity in slowly-varying scenarios.
    pub fn set_cutoff(&mut self, cutoff_hz: Sample) {
        let nyquist = self.sample_rate * 0.5;
        let fc = cutoff_hz.clamp(1.0, nyquist - 1.0);
        self.cutoff_hz = fc;
        match self.order {
            HighPassOrder::First => self.load_first_order(fc),
            HighPassOrder::Second => self.load_second_order(fc),
        }
    }

    /// Loads a first-order DC-blocker as a degenerate biquad.
    ///
    /// The analog prototype `H(s) = s / (s + wc)` is mapped with the bilinear
    /// transform, giving `y[n] = a * (x[n] - x[n-1]) + c * y[n-1]`.
    fn load_first_order(&mut self, fc: Sample) {
        let k = ops::tan(PI * fc / self.sample_rate);
        let norm = 1.0 / (1.0 + k);
        self.b0 = norm;
        self.b1 = -norm;
        self.b2 = 0.0;
        self.a1 = (k - 1.0) * norm;
        self.a2 = 0.0;
    }

    /// Loads a second-order Butterworth high-pass (RBJ cookbook, `Q = 1/sqrt(2)`).
    fn load_second_order(&mut self, fc: Sample) {
        let w0 = 2.0 * PI * fc / self.sample_rate;
        let (sin_w0, cos_w0) = ops::sin_cos(w0);
        let q = core::f32::consts::FRAC_1_SQRT_2;
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha;
        let inv_a0 = 1.0 / a0;
        self.b0 = ((1.0 + cos_w0) * 0.5) * inv_a0;
        self.b1 = (-(1.0 + cos_w0)) * inv_a0;
        self.b2 = ((1.0 + cos_w0) * 0.5) * inv_a0;
        self.a1 = (-2.0 * cos_w0) * inv_a0;
        self.a2 = (1.0 - alpha) * inv_a0;
    }

    /// Clears the delay memory so the next sample starts from rest.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Filters a single sample through the transposed Direct Form II structure.
    #[inline]
    #[must_use]
    pub fn process_sample(&mut self, x: Sample) -> Sample {
        let y = self.b0 * x + self.z1;
        self.z1 = flush_denormal(self.b1 * x - self.a1 * y + self.z2);
        self.z2 = flush_denormal(self.b2 * x - self.a2 * y);
        y
    }

    /// Filters `block` in place.
    pub fn process_block(&mut self, block: &mut [Sample]) {
        for sample in block.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-4;

    fn rms(block: &[Sample]) -> Sample {
        let sum: Sample = block.iter().map(|&v| v * v).sum();
        ops::sqrt(sum / block.len() as Sample)
    }

    #[test]
    fn first_order_removes_dc() {
        let mut hp = HighPass::new(48_000.0, 80.0, HighPassOrder::First);
        let mut block = [1.0; 4096];
        hp.process_block(&mut block);
        // After the transient the output should settle near zero for pure DC.
        let tail = &block[2048..];
        assert!(rms(tail) < 1e-2, "dc tail rms={}", rms(tail));
    }

    #[test]
    fn second_order_passes_high_frequency() {
        let sr = 48_000.0;
        let mut hp = HighPass::new(sr, 80.0, HighPassOrder::Second);
        let freq = 4_000.0;
        let mut block = [0.0; 4096];
        for (n, s) in block.iter_mut().enumerate() {
            *s = ops::sin(2.0 * PI * freq * n as Sample / sr);
        }
        hp.process_block(&mut block);
        let tail = &block[2048..];
        // A 4 kHz tone well above an 80 Hz cutoff should pass near unity.
        assert!((rms(tail) - core::f32::consts::FRAC_1_SQRT_2).abs() < 5e-2);
    }

    #[test]
    fn second_order_attenuates_low_frequency() {
        let sr = 48_000.0;
        let mut hp = HighPass::new(sr, 200.0, HighPassOrder::Second);
        let freq = 20.0;
        let mut block = [0.0; 8192];
        for (n, s) in block.iter_mut().enumerate() {
            *s = ops::sin(2.0 * PI * freq * n as Sample / sr);
        }
        hp.process_block(&mut block);
        let tail = &block[4096..];
        // 20 Hz is a decade below a 200 Hz cutoff: strongly attenuated.
        assert!(rms(tail) < 0.1, "low tail rms={}", rms(tail));
    }

    #[test]
    fn reset_clears_state() {
        let mut hp = HighPass::new(48_000.0, 100.0, HighPassOrder::Second);
        let mut block = [0.5; 64];
        hp.process_block(&mut block);
        hp.reset();
        let first = hp.process_sample(0.0);
        assert!(first.abs() < EPS);
    }
}
