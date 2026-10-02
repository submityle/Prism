//! High-order windowed-sinc resampler (mastering / near-field grade).
//!
//! The highest grade on the quality ladder. It reuses the exact streaming
//! engine of [`crate::polyphase_sinc`] but with a much longer kernel (more
//! taps) and a finer prototype grid, which pushes the stop-band rejection and
//! pass-band flatness well past what the default grade needs. It is reserved
//! for near-field or mastering paths where the extra tap count is affordable.
//!
//! Delegating to the shared engine keeps a single, well-tested resampling core:
//! the only differences are the construction-time kernel length, the prototype
//! resolution, and the advertised [`ResampleQuality`].
//!
//! # Provenance
//!
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code, and no AI/ML.
//! Classic DSP only. The long windowed-sinc kernel is a standard multirate
//! construction; all math routes through [`bevy_math::ops`].
//!
//! # Relationship
//!
//! Downstream of [`prism_audio_core`],
//! [`crate::fractional_delay`], and [`crate::polyphase_sinc`] (whose
//! [`PolyphaseSincResampler`](crate::polyphase_sinc::PolyphaseSincResampler) it
//! wraps). Implements [`crate::resampler::Resampler`] at the
//! [`ResampleQuality::HighOrderSinc`] grade, the top rung above
//! [`crate::polyphase_sinc::PolyphaseSincResampler`].

use prism_audio_core::math::Sample;

use crate::polyphase_sinc::PolyphaseSincResampler;
use crate::resampler::{ResampleProgress, ResampleQuality, Resampler};

/// High-order (long-kernel) windowed-sinc resampler.
///
/// A thin wrapper over [`PolyphaseSincResampler`] configured with a 64-tap
/// one-sided kernel and a 512x prototype grid. Construct, set the ratio, and
/// stream exactly as with the default grade.
#[derive(Clone, Debug)]
pub struct HighOrderSincResampler {
    /// The underlying engine, built with the long-kernel parameters.
    inner: PolyphaseSincResampler,
}

impl HighOrderSincResampler {
    /// Builds the high-order resampler (`half = 64`, oversample `512`, minimum
    /// ratio `1/8`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: PolyphaseSincResampler::with_params(
                64,
                512,
                1.0 / 8.0,
                ResampleQuality::HighOrderSinc,
            ),
        }
    }

    /// The one-sided tap count of the underlying kernel.
    #[inline]
    #[must_use]
    pub fn half(&self) -> usize {
        self.inner.half()
    }
}

impl Default for HighOrderSincResampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Resampler for HighOrderSincResampler {
    fn quality(&self) -> ResampleQuality {
        self.inner.quality()
    }

    fn ratio(&self) -> Sample {
        self.inner.ratio()
    }

    fn set_ratio(&mut self, ratio: Sample) {
        self.inner.set_ratio(ratio);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) -> ResampleProgress {
        self.inner.process(input, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractional_delay::close;
    use alloc::vec;
    use alloc::vec::Vec;
    use bevy_math::ops;
    use core::f32::consts::TAU;

    fn run(res: &mut HighOrderSincResampler, input: &[Sample]) -> Vec<Sample> {
        let mut out = Vec::new();
        let mut scratch = vec![0.0 as Sample; 1024];
        let mut offset = 0;
        loop {
            let prog = res.process(&input[offset..], &mut scratch);
            out.extend_from_slice(&scratch[..prog.produced]);
            offset += prog.consumed;
            if prog.produced == 0 && prog.consumed == 0 {
                break;
            }
            if offset >= input.len() && prog.produced == 0 {
                break;
            }
        }
        out
    }

    fn sine(freq: Sample, rate: Sample, n: usize) -> Vec<Sample> {
        (0..n)
            .map(|i| ops::sin(TAU * freq * i as Sample / rate))
            .collect()
    }

    fn rms(x: &[Sample]) -> Sample {
        if x.is_empty() {
            return 0.0;
        }
        let s: Sample = x.iter().map(|&v| v * v).sum();
        ops::sqrt(s / x.len() as Sample)
    }

    #[test]
    fn grade_is_high_order() {
        let res = HighOrderSincResampler::new();
        assert_eq!(res.quality(), ResampleQuality::HighOrderSinc);
        assert_eq!(res.half(), 64);
    }

    #[test]
    fn upsample_preserves_passband_energy() {
        let mut res = HighOrderSincResampler::new();
        res.set_ratio(2.0);
        let input = sine(1_000.0, 48_000.0, 8192);
        let out = run(&mut res, &input);
        let a = rms(&input[1024..input.len() - 1024]);
        let b = rms(&out[2048..out.len() - 2048]);
        assert!(ops::abs(a - b) < 0.03, "a={a} b={b}");
    }

    #[test]
    fn downsample_rejects_images() {
        let mut res = HighOrderSincResampler::new();
        res.set_ratio(0.5);
        let input = sine(16_000.0, 48_000.0, 16384);
        let out = run(&mut res, &input);
        let residual = rms(&out[1024..out.len() - 1024]);
        assert!(residual < 0.05, "residual = {residual}");
    }

    #[test]
    fn determinism_and_reset() {
        let input = sine(640.0, 48_000.0, 8192);
        let mut a = HighOrderSincResampler::new();
        a.set_ratio(1.3);
        let first = run(&mut a, &input);
        a.reset();
        a.set_ratio(1.3);
        let second = run(&mut a, &input);
        assert_eq!(first.len(), second.len());
        for (x, y) in first.iter().zip(second.iter()) {
            assert!(close(*x, *y));
        }
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut res = HighOrderSincResampler::new();
        res.set_ratio(1.5);
        let mut input = sine(1000.0, 48_000.0, 4096);
        input[500] = Sample::NAN;
        let out = run(&mut res, &input);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
