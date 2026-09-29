//! Second-order (biquad) IIR filter node using the RBJ Audio EQ Cookbook.
//!
//! The coefficient formulas are the canonical Robert Bristow-Johnson cookbook
//! equations, which are standard public signal-processing knowledge. State is
//! held per channel as a Direct Form I history so the filter is exact and
//! stable for surround layouts.

use alloc::vec;
use alloc::vec::Vec;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use bevy_math::ops;

use crate::math::{Sample, flush_denormal};

/// The filter response computed from the cookbook coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BiquadKind {
    /// 12 dB/octave low-pass.
    LowPass,
    /// 12 dB/octave high-pass.
    HighPass,
    /// Constant-skirt-gain band-pass (peak gain = Q).
    BandPass,
    /// Band-reject / notch.
    Notch,
    /// Peaking (bell) EQ with `gain_db` boost/cut at the centre frequency.
    Peaking,
    /// Low shelf with `gain_db` boost/cut below the corner.
    LowShelf,
    /// High shelf with `gain_db` boost/cut above the corner.
    HighShelf,
}

/// Normalised transfer-function coefficients (a0 divided out).
#[derive(Debug, Clone, Copy, Default)]
struct Coeffs {
    b0: Sample,
    b1: Sample,
    b2: Sample,
    a1: Sample,
    a2: Sample,
}

/// A per-channel biquad filter node (input port 0 -> output port 0).
#[derive(Debug, Clone)]
pub struct BiquadNode {
    coeffs: Coeffs,
    /// Direct Form I state: `[x1, x2, y1, y2]` per channel.
    state: Vec<[Sample; 4]>,
}

impl BiquadNode {
    /// Builds a filter of `kind` at `freq_hz` with quality `q` (and `gain_db`
    /// for the peaking/shelving kinds), for a `channels`-wide signal running at
    /// `sample_rate` Hz.
    #[must_use]
    pub fn new(
        kind: BiquadKind,
        sample_rate: u32,
        freq_hz: Sample,
        q: Sample,
        gain_db: Sample,
        channels: usize,
    ) -> Self {
        Self {
            coeffs: design(kind, sample_rate, freq_hz, q, gain_db),
            state: vec![[0.0; 4]; channels.max(1)],
        }
    }

    /// Recomputes the coefficients for new filter settings.
    ///
    /// State is preserved so the filter keeps running without a click; call
    /// [`AudioNode::reset`] first if a hard restart is desired.
    #[inline]
    pub fn set_params(
        &mut self,
        kind: BiquadKind,
        sample_rate: u32,
        freq_hz: Sample,
        q: Sample,
        gain_db: Sample,
    ) {
        self.coeffs = design(kind, sample_rate, freq_hz, q, gain_db);
    }
}

impl AudioNode for BiquadNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(self.state.len());
        let c = self.coeffs;
        for ch in 0..channels {
            let st = &mut self.state[ch];
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            let (mut x1, mut x2, mut y1, mut y2) = (st[0], st[1], st[2], st[3]);
            for (d, &x0) in dst.iter_mut().zip(src) {
                let y0 = c.b0 * x0 + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
                let y0 = flush_denormal(y0);
                x2 = x1;
                x1 = x0;
                y2 = y1;
                y1 = y0;
                *d = y0;
            }
            *st = [x1, x2, y1, y2];
        }
    }

    fn reset(&mut self) {
        for s in &mut self.state {
            *s = [0.0; 4];
        }
    }
}

/// Computes normalised biquad coefficients from the RBJ cookbook.
fn design(kind: BiquadKind, sample_rate: u32, freq_hz: Sample, q: Sample, gain_db: Sample) -> Coeffs {
    let sr = sample_rate.max(1) as Sample;
    let f0 = freq_hz.clamp(1.0, sr * 0.499);
    let q = q.max(1.0e-4);
    let w0 = 2.0 * core::f32::consts::PI * f0 / sr;
    let (sin_w0, cos_w0) = sin_cos(w0);
    let alpha = sin_w0 / (2.0 * q);

    // Shared shelving amplitude.
    let a = pow10(gain_db / 40.0);

    let (b0, b1, b2, a0, a1, a2) = match kind {
        BiquadKind::LowPass => {
            let b1 = 1.0 - cos_w0;
            (b1 * 0.5, b1, b1 * 0.5, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
        }
        BiquadKind::HighPass => {
            let b1 = -(1.0 + cos_w0);
            ((1.0 + cos_w0) * 0.5, b1, (1.0 + cos_w0) * 0.5, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
        }
        BiquadKind::BandPass => {
            // Constant peak gain = Q (the "0 dB peak" variant uses alpha for b0).
            (alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
        }
        BiquadKind::Notch => {
            (1.0, -2.0 * cos_w0, 1.0, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
        }
        BiquadKind::Peaking => {
            (
                1.0 + alpha * a,
                -2.0 * cos_w0,
                1.0 - alpha * a,
                1.0 + alpha / a,
                -2.0 * cos_w0,
                1.0 - alpha / a,
            )
        }
        BiquadKind::LowShelf => {
            let sqrt_a = ops::sqrt(a.max(0.0));
            let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;
            (
                a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha),
                2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0),
                a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha),
                (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha,
                -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0),
                (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha,
            )
        }
        BiquadKind::HighShelf => {
            let sqrt_a = ops::sqrt(a.max(0.0));
            let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;
            (
                a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha),
                -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0),
                a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha),
                (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha,
                2.0 * ((a - 1.0) - (a + 1.0) * cos_w0),
                (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha,
            )
        }
    };

    let inv_a0 = 1.0 / a0;
    Coeffs {
        b0: b0 * inv_a0,
        b1: b1 * inv_a0,
        b2: b2 * inv_a0,
        a1: a1 * inv_a0,
        a2: a2 * inv_a0,
    }
}

#[inline]
fn sin_cos(x: Sample) -> (Sample, Sample) {
    ops::sin_cos(x)
}

#[inline]
fn pow10(x: Sample) -> Sample {
    ops::exp(x * core::f32::consts::LN_10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    fn ctx() -> RenderContext {
        RenderContext { sample_rate: 48_000, frames: 512, playhead: 0 }
    }

    fn rms(buf: &AudioBuffer) -> Sample {
        let ch = buf.channel(0);
        let sum: Sample = ch.iter().map(|s| s * s).sum();
        ops::sqrt(sum / ch.len() as Sample)
    }

    /// Drives a sine at `freq` through the filter and returns output RMS.
    fn response(kind: BiquadKind, cutoff: Sample, freq: Sample) -> Sample {
        let n = 512usize;
        let sr = 48_000u32;
        let mut node = BiquadNode::new(kind, sr, cutoff, 0.707, 0.0, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        {
            let ch = input.channel_mut(0);
            for (i, s) in ch.iter_mut().enumerate() {
                let t = i as Sample / sr as Sample;
                *s = ops::sin(2.0 * core::f32::consts::PI * freq * t);
            }
        }
        let output = AudioBuffer::new(ChannelLayout::Mono, n);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(), &mut io);
        rms(&outputs[0])
    }

    #[test]
    fn lowpass_attenuates_highs() {
        let low = response(BiquadKind::LowPass, 1_000.0, 200.0);
        let high = response(BiquadKind::LowPass, 1_000.0, 10_000.0);
        assert!(high < low * 0.5, "low={low} high={high}");
    }

    #[test]
    fn highpass_attenuates_lows() {
        let low = response(BiquadKind::HighPass, 1_000.0, 100.0);
        let high = response(BiquadKind::HighPass, 1_000.0, 8_000.0);
        assert!(low < high * 0.5, "low={low} high={high}");
    }

    #[test]
    fn stable_impulse_decays() {
        let mut node = BiquadNode::new(BiquadKind::LowPass, 48_000, 1_000.0, 0.707, 0.0, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 256);
        input.channel_mut(0)[0] = 1.0;
        let output = AudioBuffer::new(ChannelLayout::Mono, 256);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(), &mut io);
        let tail = &outputs[0].channel(0)[200..];
        assert!(tail.iter().all(|s| s.abs() < 1e-3), "filter did not settle");
    }
}
