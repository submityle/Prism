//! Linear-interpolation resampler (far-field / LOD grade).
//!
//! The cheapest grade on the quality ladder: each output sample is a two-point
//! linear blend of the two input samples bracketing the fractional read
//! position. It introduces a gentle high-frequency roll-off and does not reject
//! aliasing when decimating, so it is reserved for distant or low-priority
//! voices where CPU budget matters more than fidelity. It shares the streaming
//! contract and phase-continuity guarantees of the higher grades: the
//! fractional read position carries across blocks, so a swept ratio produces no
//! clicks.
//!
//! # Provenance
//!
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code, and no AI/ML.
//! Classic DSP only. Linear interpolation is elementary signal processing.
//!
//! # Relationship
//!
//! Downstream of [`prism_audio_core`] and [`crate::fractional_delay`] (reuses
//! [`linear_interp`](crate::fractional_delay::linear_interp)). Implements
//! [`crate::resampler::Resampler`] at the [`ResampleQuality::Linear`] grade,
//! the lowest rung beneath
//! [`crate::polyphase_sinc::PolyphaseSincResampler`].

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use prism_audio_core::math::Sample;

use crate::fractional_delay::{linear_interp, sanitize};
use crate::resampler::{ResampleProgress, ResampleQuality, Resampler, clamp_ratio};

/// Number of past samples retained between blocks (only one is needed for the
/// left side of a two-point blend; a small margin keeps rebasing simple).
const HISTORY: usize = 4;

/// Two-point linear-interpolation resampler.
#[derive(Clone, Debug)]
pub struct LinearResampler {
    /// Current ratio (`output_rate / input_rate`).
    ratio: Sample,
    /// Input advance per output sample (`1 / ratio`).
    step: Sample,
    /// Retained input samples preceding the current block.
    history: Vec<Sample>,
    /// Scratch used to rebase `history` without allocating.
    hist_scratch: Vec<Sample>,
    /// Fractional read position relative to the current block's `input[0]`.
    pos: Sample,
}

impl LinearResampler {
    /// Builds a linear resampler at unity ratio.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ratio: 1.0,
            step: 1.0,
            history: vec![0.0; HISTORY],
            hist_scratch: vec![0.0; HISTORY],
            pos: 0.0,
        }
    }

    /// Reads the input stream at integer index `i`, where negative indices come
    /// from the retained history. Out-of-range indices read as silence.
    #[inline]
    #[must_use]
    fn sample_at(&self, input: &[Sample], i: isize) -> Sample {
        if i >= 0 {
            let i = i as usize;
            if i < input.len() {
                sanitize(input[i])
            } else {
                0.0
            }
        } else {
            let h = self.history.len() as isize + i;
            if h >= 0 && (h as usize) < self.history.len() {
                self.history[h as usize]
            } else {
                0.0
            }
        }
    }

    /// Rebases the history so the next block's `input[0]` is the current index
    /// `consumed`.
    fn rebase(&mut self, input: &[Sample], consumed: usize) {
        let h = self.history.len();
        let start = consumed as isize - h as isize;
        let mut k = 0;
        while k < h {
            self.hist_scratch[k] = self.sample_at(input, start + k as isize);
            k += 1;
        }
        core::mem::swap(&mut self.history, &mut self.hist_scratch);
    }
}

impl Default for LinearResampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Resampler for LinearResampler {
    fn quality(&self) -> ResampleQuality {
        ResampleQuality::Linear
    }

    fn ratio(&self) -> Sample {
        self.ratio
    }

    fn set_ratio(&mut self, ratio: Sample) {
        let ratio = clamp_ratio(ratio);
        self.ratio = ratio;
        self.step = 1.0 / ratio;
    }

    fn reset(&mut self) {
        for v in &mut self.history {
            *v = 0.0;
        }
        self.pos = 0.0;
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) -> ResampleProgress {
        let n = input.len() as isize;
        let mut produced = 0;
        while produced < output.len() {
            let base = ops::floor(self.pos);
            let right = base as isize + 1;
            if right > n - 1 {
                break;
            }
            let frac = self.pos - base;
            let a = self.sample_at(input, base as isize);
            let b = self.sample_at(input, right);
            output[produced] = linear_interp(a, b, frac);
            produced += 1;
            self.pos += self.step;
        }
        // The newest sample still needed for the left side is floor(pos); drop
        // everything strictly before it.
        let drop = ops::floor(self.pos);
        let consumed = if drop > 0.0 {
            (drop as usize).min(input.len())
        } else {
            0
        };
        self.rebase(input, consumed);
        self.pos -= consumed as Sample;
        ResampleProgress { consumed, produced }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractional_delay::close;

    fn run(res: &mut LinearResampler, input: &[Sample]) -> Vec<Sample> {
        let mut out = Vec::new();
        let mut scratch = vec![0.0 as Sample; 256];
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

    #[test]
    fn ramp_upsample_is_linear() {
        // A linear ramp resampled 2x must stay a linear ramp (linear interp is
        // exact for affine signals).
        let mut res = LinearResampler::new();
        res.set_ratio(2.0);
        let input: Vec<Sample> = (0..64).map(|i| i as Sample).collect();
        let out = run(&mut res, &input);
        // Steady-state spacing between outputs is ~0.5 the input spacing.
        for w in out[4..out.len() - 4].windows(2) {
            assert!(close(w[1] - w[0], 0.5));
        }
    }

    #[test]
    fn output_length_tracks_ratio() {
        let mut res = LinearResampler::new();
        res.set_ratio(2.0);
        let input: Vec<Sample> = (0..1000).map(|i| (i % 7) as Sample).collect();
        let out = run(&mut res, &input);
        // About twice as many samples (minus boundary slack).
        assert!(out.len() > 1900 && out.len() <= 2000);
    }

    #[test]
    fn determinism_two_instances_match() {
        let input: Vec<Sample> = (0..2048).map(|i| ((i * 13 % 97) as Sample) / 97.0).collect();
        let mut a = LinearResampler::new();
        let mut b = LinearResampler::new();
        a.set_ratio(0.6);
        b.set_ratio(0.6);
        let oa = run(&mut a, &input);
        let ob = run(&mut b, &input);
        assert_eq!(oa.len(), ob.len());
        for (x, y) in oa.iter().zip(ob.iter()) {
            assert!(close(*x, *y));
        }
    }

    #[test]
    fn reset_reproduces_output() {
        let input: Vec<Sample> = (0..2048).map(|i| ((i * 5 % 31) as Sample) / 31.0).collect();
        let mut r = LinearResampler::new();
        r.set_ratio(1.25);
        let first = run(&mut r, &input);
        r.reset();
        r.set_ratio(1.25);
        let second = run(&mut r, &input);
        assert_eq!(first.len(), second.len());
        for (x, y) in first.iter().zip(second.iter()) {
            assert!(close(*x, *y));
        }
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut res = LinearResampler::new();
        res.set_ratio(1.5);
        let mut input: Vec<Sample> = (0..1024).map(|i| ops::sin(i as Sample)).collect();
        input[10] = Sample::NAN;
        input[20] = Sample::INFINITY;
        let out = run(&mut res, &input);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
