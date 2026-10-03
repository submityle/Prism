//! Small shared classic-DSP primitives used across the procedural modules.
//!
//! These are deliberately tiny, `Copy`, allocation-free building blocks so they
//! can be embedded directly inside real-time voices: a one-pole smoothing
//! filter used for spectral tilt and friction colouring, and the standard
//! helpers that convert a decay half-life into a per-sample pole radius for the
//! modal resonators. All transcendental math is routed through
//! [`bevy_math::ops`] so results are deterministic across platforms.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The one-pole filter
//! and the `t60`/half-life-to-radius relations are standard, publicly
//! documented signal-processing formulas.
//!
//! # Relationship
//! Shared by the modal (section 47.2), continuous-contact (section 47.3), and
//! granular (section 47.4) modules; complements
//! [`prism_audio_core::param::Smoothed`] with a direct one-pole colour filter
//! rather than a parameter ramp.

use bevy_math::ops;

use prism_audio_core::math::{flush_denormal, Sample};

/// `2 * PI`, the radian span of a full cycle.
pub const TWO_PI: Sample = core::f32::consts::TAU;

/// Natural logarithm of `2`, used to turn a half-life into a decay rate.
pub const LN_2: Sample = core::f32::consts::LN_2;

/// Clamps a frequency into the open audible band for a given sample rate.
///
/// The result is always strictly below Nyquist and at least `1 Hz`, keeping
/// resonator and oscillator phase increments numerically well behaved.
#[inline]
#[must_use]
pub fn clamp_frequency(freq_hz: Sample, sample_rate: u32) -> Sample {
    let nyquist = 0.5 * sample_rate.max(1) as Sample;
    if freq_hz.is_finite() {
        freq_hz.clamp(1.0, nyquist * 0.999)
    } else {
        (nyquist * 0.5).min(nyquist * 0.999)
    }
}

/// Converts an amplitude half-life in seconds to a per-sample decay radius.
///
/// A resonator whose output envelope halves every `half_life_s` seconds has a
/// per-sample multiplier `r = 2^(-1 / (half_life_s * fs))`, returned here. A
/// non-positive half-life yields an immediate (`0`) radius so a mode with no
/// ring decays in a single sample instead of exploding.
#[inline]
#[must_use]
pub fn half_life_to_radius(half_life_s: Sample, sample_rate: u32) -> Sample {
    let fs = sample_rate.max(1) as Sample;
    if !half_life_s.is_finite() || half_life_s <= 0.0 {
        return 0.0;
    }
    let samples = half_life_s * fs;
    // r = 2^(-1/samples) = exp(-ln2 / samples).
    ops::exp(-LN_2 / samples).clamp(0.0, 0.999_999)
}

/// A first-order (one-pole) filter usable as a low-pass or, via complement, a
/// high-pass colour stage.
///
/// The pole coefficient is derived from a cutoff frequency with the standard
/// bilinear-free approximation `a = exp(-2*pi*fc/fs)`. [`OnePole::low`] returns
/// the smoothed (low-passed) output; [`OnePole::high`] returns the input minus
/// that output, i.e. the complementary high-pass. State is denormal-flushed so
/// the filter cannot stall.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OnePole {
    coeff: Sample,
    state: Sample,
}

impl OnePole {
    /// Creates a one-pole filter tuned to `cutoff_hz` at `sample_rate`.
    #[inline]
    #[must_use]
    pub fn new(cutoff_hz: Sample, sample_rate: u32) -> Self {
        let mut filter = Self {
            coeff: 0.0,
            state: 0.0,
        };
        filter.set_cutoff(cutoff_hz, sample_rate);
        filter
    }

    /// Retunes the filter to a new cutoff without disturbing its state.
    #[inline]
    pub fn set_cutoff(&mut self, cutoff_hz: Sample, sample_rate: u32) {
        let fs = sample_rate.max(1) as Sample;
        let fc = cutoff_hz.clamp(1.0, 0.499 * fs);
        self.coeff = ops::exp(-TWO_PI * fc / fs).clamp(0.0, 0.999_999);
    }

    /// Resets the internal state to silence.
    #[inline]
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Processes one sample and returns the low-passed output.
    #[inline]
    pub fn low(&mut self, input: Sample) -> Sample {
        let x = if input.is_finite() { input } else { 0.0 };
        self.state = flush_denormal(x + self.coeff * (self.state - x));
        self.state
    }

    /// Processes one sample and returns the complementary high-passed output.
    #[inline]
    pub fn high(&mut self, input: Sample) -> Sample {
        let x = if input.is_finite() { input } else { 0.0 };
        x - self.low(x)
    }
}

/// Linearly interpolates between `a` and `b` by `t` (clamped to `[0, 1]`).
#[inline]
#[must_use]
pub fn lerp(a: Sample, b: Sample, t: Sample) -> Sample {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Smoothstep easing of `t` into `[0, 1]` (`3t^2 - 2t^3`).
///
/// Used to shape grain and fade envelopes without discontinuous slopes.
#[inline]
#[must_use]
pub fn smoothstep(t: Sample) -> Sample {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radius_halves_over_half_life() {
        let fs = 48_000;
        let r = half_life_to_radius(1.0, fs);
        // After `fs` samples the envelope should be ~0.5.
        let decayed = ops::powf(r, fs as Sample);
        assert!((decayed - 0.5).abs() < 1e-3, "decayed={decayed}");
    }

    #[test]
    fn non_positive_half_life_is_dead() {
        assert_eq!(half_life_to_radius(0.0, 48_000), 0.0);
        assert_eq!(half_life_to_radius(-1.0, 48_000), 0.0);
    }

    #[test]
    fn one_pole_low_passes_dc_through() {
        let mut f = OnePole::new(1_000.0, 48_000);
        let mut y = 0.0;
        for _ in 0..2_000 {
            y = f.low(1.0);
        }
        assert!((y - 1.0).abs() < 1e-2, "y={y}");
    }

    #[test]
    fn one_pole_high_blocks_dc() {
        let mut f = OnePole::new(1_000.0, 48_000);
        let mut y = 0.0;
        for _ in 0..2_000 {
            y = f.high(1.0);
        }
        assert!(y.abs() < 1e-2, "y={y}");
    }

    #[test]
    fn frequency_clamp_below_nyquist() {
        let fc = clamp_frequency(1.0e9, 48_000);
        assert!(fc < 24_000.0);
        assert!(clamp_frequency(-5.0, 48_000) >= 1.0);
    }

    #[test]
    fn smoothstep_endpoints() {
        assert_eq!(smoothstep(-1.0), 0.0);
        assert_eq!(smoothstep(2.0), 1.0);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-6);
    }
}
