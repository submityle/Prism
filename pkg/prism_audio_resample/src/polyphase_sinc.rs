//! Polyphase Blackman-windowed sinc FIR resampler (default quality grade).
//!
//! This is the crate's workhorse sample-rate converter. A single-sided
//! windowed-sinc prototype low-pass is precomputed at construction and sampled
//! on an oversampled grid; the hot path reads it with linear interpolation to
//! obtain the exact fractional coefficient for every output position. Because
//! the prototype is evaluated at `cutoff * |pos - i|`, the same table serves
//! interpolation (ratio >= 1, cutoff at the input Nyquist) and decimation
//! (ratio < 1, cutoff lowered to the output Nyquist so images fold below the
//! stop band). Each output normalizes by the running coefficient sum, so the
//! DC gain is exactly unity at every fractional phase.
//!
//! The engine is parameterized by the one-sided tap count (`half`), the
//! prototype oversampling factor, and the minimum ratio it must support (which
//! bounds the anti-alias kernel's input-sample support and therefore the
//! preallocated history). [`crate::high_order_sinc`] reuses this exact engine
//! with a longer kernel and finer prototype, so the two grades share all
//! machinery.
//!
//! # Provenance
//!
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code, and no AI/ML.
//! Classic DSP only. The polyphase windowed-sinc resampler is a standard,
//! publicly documented construction (Smith, "Digital Audio Resampling";
//! Crochiere and Rabiner, multirate signal processing). All math routes through
//! [`bevy_math::ops`] for bit-reproducibility.
//!
//! # Relationship
//!
//! Downstream of [`prism_audio_core`] and [`crate::fractional_delay`] (whose
//! [`windowed_sinc`](crate::fractional_delay::windowed_sinc) builds the
//! prototype). Implements [`crate::resampler::Resampler`] at the
//! [`ResampleQuality::Sinc`] grade and is reused by
//! [`crate::high_order_sinc::HighOrderSincResampler`].

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use prism_audio_core::math::Sample;

use crate::fractional_delay::{EPS, sanitize, windowed_sinc};
use crate::resampler::{ResampleProgress, ResampleQuality, Resampler, clamp_ratio};

/// Returns the ceiling of `x` using only [`bevy_math::ops`] (`ceil(x) ==
/// -floor(-x)`), so no `f32` intrinsic method is used.
#[inline]
#[must_use]
fn ceil(x: Sample) -> Sample {
    -ops::floor(-x)
}

/// Polyphase windowed-sinc FIR resampler.
///
/// Construct with [`PolyphaseSincResampler::new`] (default grade) or
/// [`PolyphaseSincResampler::with_params`] (used by the high-order grade), set
/// the ratio with [`Resampler::set_ratio`], then stream blocks through
/// [`Resampler::process`].
#[derive(Clone, Debug)]
pub struct PolyphaseSincResampler {
    /// One-sided tap count at unity ratio; the kernel spans `2 * half` samples.
    half: usize,
    /// Prototype samples per unit tap (sub-sample resolution).
    oversample: usize,
    /// Minimum supported ratio; bounds the decimation kernel support.
    min_ratio: Sample,
    /// Single-sided prototype low-pass, `proto[q] = windowed_sinc(q / oversample)`.
    proto: Vec<Sample>,
    /// Quality grade this instance advertises.
    quality: ResampleQuality,
    /// Current ratio (`output_rate / input_rate`).
    ratio: Sample,
    /// Input advance per output sample (`1 / ratio`).
    step: Sample,
    /// Low-pass cutoff scale (`min(1, ratio)`).
    filt_scale: Sample,
    /// Preallocated history of input samples preceding the current block.
    history: Vec<Sample>,
    /// Scratch used to rebase `history` without allocating in the hot path.
    hist_scratch: Vec<Sample>,
    /// Fractional read position relative to the current block's `input[0]`.
    pos: Sample,
}

impl PolyphaseSincResampler {
    /// Builds the default-grade resampler (`half = 16`, oversample `256`,
    /// minimum ratio `1/16`).
    #[must_use]
    pub fn new() -> Self {
        Self::with_params(16, 256, 1.0 / 16.0, ResampleQuality::Sinc)
    }

    /// Builds a resampler with explicit kernel parameters.
    ///
    /// `half` is the one-sided tap count (clamped to at least `1`),
    /// `oversample` is the prototype sub-sample resolution (clamped to at least
    /// `1`), `min_ratio` is the lowest ratio the instance must support (clamped
    /// into the global `[MIN_RATIO, 1]` window), and `quality` is the grade the
    /// instance advertises. All tables and buffers are allocated here so
    /// [`Resampler::process`] never allocates.
    #[must_use]
    pub fn with_params(
        half: usize,
        oversample: usize,
        min_ratio: Sample,
        quality: ResampleQuality,
    ) -> Self {
        let half = half.max(1);
        let oversample = oversample.max(1);
        let min_ratio = clamp_ratio(min_ratio).min(1.0);
        // Build the single-sided prototype over x in [0, half].
        let proto_len = half * oversample + 2;
        let mut proto = Vec::with_capacity(proto_len);
        for q in 0..proto_len {
            let x = q as Sample / oversample as Sample;
            proto.push(windowed_sinc(x, half as Sample, 1.0));
        }
        // History must cover twice the worst-case (lowest ratio) support plus a
        // margin so the left kernel tail is always available after a rebase.
        let support_max = (half as Sample / min_ratio) as usize + 2;
        let hist_len = 2 * support_max + 4;
        Self {
            half,
            oversample,
            min_ratio,
            proto,
            quality,
            ratio: 1.0,
            step: 1.0,
            filt_scale: 1.0,
            history: vec![0.0; hist_len],
            hist_scratch: vec![0.0; hist_len],
            pos: 0.0,
        }
    }

    /// The one-sided tap count at unity ratio.
    #[inline]
    #[must_use]
    pub fn half(&self) -> usize {
        self.half
    }

    /// The prototype sub-sample resolution.
    #[inline]
    #[must_use]
    pub fn oversample(&self) -> usize {
        self.oversample
    }

    /// The lowest ratio this instance supports.
    #[inline]
    #[must_use]
    pub fn min_ratio(&self) -> Sample {
        self.min_ratio
    }

    /// Current input-sample support on each side of the read position.
    #[inline]
    #[must_use]
    fn support(&self) -> Sample {
        self.half as Sample / self.filt_scale
    }

    /// Looks up the prototype low-pass at non-negative argument `x` with linear
    /// interpolation between the two bracketing grid samples.
    #[inline]
    #[must_use]
    fn proto_at(&self, x: Sample) -> Sample {
        let g = x * self.oversample as Sample;
        let q0 = g as usize;
        if q0 + 1 >= self.proto.len() {
            return 0.0;
        }
        let frac = g - q0 as Sample;
        self.proto[q0] + (self.proto[q0 + 1] - self.proto[q0]) * frac
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

    /// Evaluates one output sample at fractional position `pos`.
    #[inline]
    #[must_use]
    fn eval(&self, input: &[Sample], pos: Sample) -> Sample {
        let support = self.support();
        let fs = self.filt_scale;
        let i_min = ceil(pos - support) as isize;
        let i_max = ops::floor(pos + support) as isize;
        let mut acc = 0.0;
        let mut norm = 0.0;
        let mut i = i_min;
        while i <= i_max {
            let x = fs * ops::abs(pos - i as Sample);
            let w = self.proto_at(x);
            acc += w * self.sample_at(input, i);
            norm += w;
            i += 1;
        }
        if ops::abs(norm) > EPS { acc / norm } else { 0.0 }
    }

    /// Rebases the history so that, for the next call, `input[0]` corresponds to
    /// the current index `consumed`. The last `history.len()` samples ending at
    /// index `consumed - 1` become the new history.
    fn rebase(&mut self, input: &[Sample], consumed: usize) {
        let h = self.history.len();
        let start = consumed as isize - h as isize;
        let mut k = 0;
        while k < h {
            let idx = start + k as isize;
            self.hist_scratch[k] = self.sample_at(input, idx);
            k += 1;
        }
        core::mem::swap(&mut self.history, &mut self.hist_scratch);
    }
}

impl Default for PolyphaseSincResampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Resampler for PolyphaseSincResampler {
    fn quality(&self) -> ResampleQuality {
        self.quality
    }

    fn ratio(&self) -> Sample {
        self.ratio
    }

    fn set_ratio(&mut self, ratio: Sample) {
        let ratio = clamp_ratio(ratio).max(self.min_ratio);
        self.ratio = ratio;
        self.step = 1.0 / ratio;
        self.filt_scale = if ratio < 1.0 { ratio } else { 1.0 };
    }

    fn reset(&mut self) {
        for v in &mut self.history {
            *v = 0.0;
        }
        self.pos = 0.0;
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) -> ResampleProgress {
        let n = input.len() as isize;
        let support = self.support();
        let mut produced = 0;
        while produced < output.len() {
            let right_needed = ops::floor(self.pos + support) as isize;
            if right_needed > n - 1 {
                break;
            }
            output[produced] = self.eval(input, self.pos);
            produced += 1;
            self.pos += self.step;
        }
        let drop = ops::floor(self.pos - support);
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
    use core::f32::consts::TAU;

    /// Drives a resampler to completion over `input`, collecting all output.
    fn run(res: &mut PolyphaseSincResampler, input: &[Sample]) -> Vec<Sample> {
        let mut out = Vec::new();
        let mut scratch = vec![0.0 as Sample; 512];
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
    fn unity_ratio_is_near_identity() {
        let mut res = PolyphaseSincResampler::new();
        res.set_ratio(1.0);
        let input = sine(440.0, 48_000.0, 4096);
        let out = run(&mut res, &input);
        // Output length tracks the ratio.
        assert!(out.len() >= input.len() - 64);
        // Compare steady-state RMS (skip filter warm-up transient).
        let a = rms(&input[128..input.len() - 128]);
        let b = rms(&out[128..out.len() - 128]);
        assert!(ops::abs(a - b) < 0.05);
    }

    #[test]
    fn upsample_preserves_passband_energy() {
        let mut res = PolyphaseSincResampler::new();
        res.set_ratio(2.0);
        let input = sine(1_000.0, 48_000.0, 4096);
        let out = run(&mut res, &input);
        // 2x the samples (within warm-up slack).
        assert!(out.len() > input.len() * 2 - 128);
        let a = rms(&input[256..input.len() - 256]);
        let b = rms(&out[512..out.len() - 512]);
        assert!(ops::abs(a - b) < 0.05);
    }

    #[test]
    fn downsample_rejects_images() {
        // A tone above the post-decimation Nyquist must be strongly attenuated.
        let mut res = PolyphaseSincResampler::new();
        res.set_ratio(0.5); // output Nyquist drops to 12 kHz.
        let input = sine(16_000.0, 48_000.0, 8192);
        let out = run(&mut res, &input);
        let residual = rms(&out[256..out.len() - 256]);
        // The alias image is pushed far below the input level.
        assert!(residual < 0.1, "residual = {residual}");
    }

    #[test]
    fn downsample_keeps_in_band_tone() {
        let mut res = PolyphaseSincResampler::new();
        res.set_ratio(0.5);
        let input = sine(2_000.0, 48_000.0, 8192);
        let out = run(&mut res, &input);
        let a = rms(&input[256..input.len() - 256]);
        let b = rms(&out[256..out.len() - 256]);
        assert!(ops::abs(a - b) < 0.08, "a={a} b={b}");
    }

    #[test]
    fn phase_continuous_across_ratio_sweep() {
        // Sweep the ratio block-by-block and confirm no sample-to-sample jump
        // large enough to be an audible click.
        let mut res = PolyphaseSincResampler::new();
        let input = sine(500.0, 48_000.0, 8192);
        let mut out = Vec::new();
        let mut scratch = vec![0.0 as Sample; 256];
        let mut offset = 0;
        let mut ratio = 0.8;
        while offset < input.len() {
            res.set_ratio(ratio);
            let prog = res.process(&input[offset..], &mut scratch);
            out.extend_from_slice(&scratch[..prog.produced]);
            offset += prog.consumed;
            ratio += 0.05;
            if ratio > 1.6 {
                ratio = 0.8;
            }
            if prog.produced == 0 && prog.consumed == 0 {
                break;
            }
        }
        // No discontinuity: consecutive output deltas stay bounded for a smooth
        // low-frequency tone.
        let mut max_delta = 0.0 as Sample;
        for w in out[64..out.len() - 64].windows(2) {
            let d = ops::abs(w[1] - w[0]);
            if d > max_delta {
                max_delta = d;
            }
        }
        assert!(max_delta < 0.25, "max_delta = {max_delta}");
    }

    #[test]
    fn determinism_two_instances_match() {
        let input = sine(777.0, 48_000.0, 4096);
        let mut a = PolyphaseSincResampler::new();
        let mut b = PolyphaseSincResampler::new();
        a.set_ratio(1.3);
        b.set_ratio(1.3);
        let oa = run(&mut a, &input);
        let ob = run(&mut b, &input);
        assert_eq!(oa.len(), ob.len());
        for (x, y) in oa.iter().zip(ob.iter()) {
            assert!(close(*x, *y));
        }
    }

    #[test]
    fn reset_reproduces_output() {
        let input = sine(321.0, 48_000.0, 4096);
        let mut r = PolyphaseSincResampler::new();
        r.set_ratio(0.75);
        let first = run(&mut r, &input);
        r.reset();
        r.set_ratio(0.75);
        let second = run(&mut r, &input);
        assert_eq!(first.len(), second.len());
        for (x, y) in first.iter().zip(second.iter()) {
            assert!(close(*x, *y));
        }
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut res = PolyphaseSincResampler::new();
        res.set_ratio(1.5);
        let mut input = sine(1000.0, 48_000.0, 2048);
        input[100] = Sample::NAN;
        input[200] = Sample::INFINITY;
        input[300] = Sample::NEG_INFINITY;
        let out = run(&mut res, &input);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn default_matches_new() {
        let a = PolyphaseSincResampler::default();
        let b = PolyphaseSincResampler::new();
        assert_eq!(a.half(), b.half());
        assert_eq!(a.oversample(), b.oversample());
    }
}
