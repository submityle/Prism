//! Stereo widener: a Mid-Side (M/S) processor that narrows or broadens the
//! perceived stereo image, with an optional bass-mono crossover.
//!
//! The stereo field is decomposed into a *mid* (sum) component that carries
//! the mono-compatible centre and a *side* (difference) component that carries
//! the stereo information:
//!
//! ```text
//! M = (L + R) / 2      S = (L - R) / 2
//! ```
//!
//! Scaling the side by a `width` factor and rebuilding the channels controls
//! how wide the image sounds:
//!
//! ```text
//! L = M + width * S    R = M - width * S
//! ```
//!
//! - `width = 0` collapses the image to mono (`L == R == M`).
//! - `width = 1` reproduces the input exactly.
//! - `width > 1` broadens the image by exaggerating the side signal.
//!
//! # Bass mono
//!
//! Wide low-frequency content translates poorly to mono playback and can smear
//! a mix's low end. When the bass-mono crossover is enabled, the low band of
//! the (widened) side signal is removed so everything below the crossover
//! folds back to the centre while the highs stay wide. The low band is
//! extracted with a shared second-order low-pass biquad
//! ([`BiquadCoeffs::design`]) and subtracted from the side path.
//!
//! Storage is fixed at construction, so
//! [`StereoWidthNode::process`](crate::graph::AudioNode::process) is real-time
//! safe (no allocation, locks, or panics on the hot path).
//!
//! # Provenance
//!
//! Mid-Side encoding is public, standard stereo technique dating back to Alan
//! Blumlein's 1931 sum-and-difference stereo patent and is described in every
//! audio-engineering text. The bass-mono crossover reuses the RBJ audio EQ
//! cookbook low-pass already implemented in
//! [`biquad`](crate::nodes::biquad). This module contains **no Unreal Engine,
//! Unity, Godot, Wwise, or FMOD source or derived code**; only the widely
//! documented M/S formulas and a standard biquad are used.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::nodes::biquad::{BiquadCoeffs, BiquadKind};
use crate::param::{Ramp, Smoothed};

/// Largest stereo-width factor the node accepts.
///
/// `width` is clamped to `[0, MAX_WIDTH]`; beyond roughly 2-3 the side signal
/// dominates and the image becomes hollow, so `4.0` is a generous ceiling.
pub const MAX_WIDTH: Sample = 4.0;

/// Q used for the bass-mono low-pass crossover (Butterworth, maximally flat).
const BASS_MONO_Q: Sample = core::f32::consts::FRAC_1_SQRT_2;

/// Construction parameters for a [`StereoWidthNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StereoWidthParams {
    /// Stereo-width factor: `0` = mono, `1` = unchanged, `> 1` = wider.
    ///
    /// Clamped to `[0, MAX_WIDTH]`.
    pub width: Sample,
    /// Bass-mono crossover frequency in Hz. Frequencies below this fold back
    /// to the centre. A value `<= 0` disables the crossover (fully wideband).
    pub bass_mono_hz: Sample,
}

impl Default for StereoWidthParams {
    fn default() -> Self {
        Self {
            width: 1.0,
            bass_mono_hz: 0.0,
        }
    }
}

/// A Mid-Side stereo widener (input port 0 -> output port 0).
///
/// The node operates on the first two channels as a stereo pair. Mono input
/// (fewer than two channels) is passed through untouched because it has no
/// side component to scale. Any channels beyond the stereo pair are copied
/// straight through.
///
/// The `width` factor is [`Smoothed`] for click-free automation. The optional
/// bass-mono low-pass keeps its Direct Form I state between blocks so the
/// crossover is continuous.
#[derive(Debug, Clone)]
pub struct StereoWidthNode {
    /// Sample rate in Hz (used when redesigning the crossover).
    sample_rate: u32,
    /// Smoothed stereo-width factor.
    width: Smoothed,
    /// Whether the bass-mono crossover is active.
    bass_mono: bool,
    /// Crossover frequency in Hz (retained for reset / redesign).
    bass_mono_hz: Sample,
    /// Low-pass coefficients used to extract the mono bass band of the side.
    bass_coeffs: BiquadCoeffs,
    /// Direct Form I state `[x1, x2, y1, y2]` for the side low-pass.
    bass_state: [Sample; 4],
}

impl StereoWidthNode {
    /// Builds a stereo widener at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, params: StereoWidthParams) -> Self {
        let bass_mono = params.bass_mono_hz > 0.0;
        let bass_coeffs = BiquadCoeffs::design(
            BiquadKind::LowPass,
            sample_rate,
            params.bass_mono_hz.max(1.0),
            BASS_MONO_Q,
            0.0,
        );
        Self {
            sample_rate,
            width: Smoothed::new(params.width.clamp(0.0, MAX_WIDTH)),
            bass_mono,
            bass_mono_hz: params.bass_mono_hz.max(0.0),
            bass_coeffs,
            bass_state: [0.0; 4],
        }
    }

    /// Sets the target stereo-width factor, clamped to `[0, MAX_WIDTH]`.
    #[inline]
    pub fn set_width(&mut self, width: Sample, ramp: Ramp) {
        self.width.set_target(width.clamp(0.0, MAX_WIDTH), ramp);
    }

    /// Returns the current (instantaneous) stereo-width factor.
    #[inline]
    #[must_use]
    pub fn width(&self) -> Sample {
        self.width.current()
    }

    /// Sets the bass-mono crossover frequency in Hz.
    ///
    /// A value `<= 0` disables the crossover. Changing the frequency preserves
    /// the low-pass filter state so the transition is click-free.
    #[inline]
    pub fn set_bass_mono_hz(&mut self, bass_mono_hz: Sample) {
        self.bass_mono = bass_mono_hz > 0.0;
        self.bass_mono_hz = bass_mono_hz.max(0.0);
        self.bass_coeffs = BiquadCoeffs::design(
            BiquadKind::LowPass,
            self.sample_rate,
            bass_mono_hz.max(1.0),
            BASS_MONO_Q,
            0.0,
        );
    }

    /// Returns whether the bass-mono crossover is currently active.
    #[inline]
    #[must_use]
    pub fn bass_mono_enabled(&self) -> bool {
        self.bass_mono
    }

    /// Advances the side low-pass by one sample (Direct Form I) and returns the
    /// filtered (low-band) output.
    #[inline]
    fn bass_lowpass(&mut self, x0: Sample) -> Sample {
        let c = self.bass_coeffs;
        let [x1, x2, y1, y2] = self.bass_state;
        let y0 = c.b0 * x0 + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
        let y0 = flush_denormal(y0);
        self.bass_state = [x0, x1, y0, y1];
        y0
    }
}

impl AudioNode for StereoWidthNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels());
        let frames = output.active_frames().min(input.active_frames());

        // Fewer than two channels: no side signal exists, pass through.
        if channels < 2 {
            for ch in 0..channels {
                let src = input.channel(ch);
                let dst = output.channel_mut(ch);
                dst[..frames].copy_from_slice(&src[..frames]);
            }
            return;
        }

        for f in 0..frames {
            let width = self.width.next_sample();
            let l = input.channel(0)[f];
            let r = input.channel(1)[f];

            let mid = 0.5 * (l + r);
            let mut side = width * (0.5 * (l - r));

            // Fold the low band of the side back to the centre.
            if self.bass_mono {
                let low = self.bass_lowpass(side);
                side -= low;
            }

            output.channel_mut(0)[f] = flush_denormal(mid + side);
            output.channel_mut(1)[f] = flush_denormal(mid - side);
        }

        // Copy any surplus channels (beyond the processed stereo pair) through.
        for ch in 2..channels {
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            dst[..frames].copy_from_slice(&src[..frames]);
        }
    }

    fn reset(&mut self) {
        self.width = Smoothed::new(self.width.target());
        self.bass_state = [0.0; 4];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use bevy_math::ops;
    use alloc::vec::Vec;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: 48_000,
            frames,
            playhead: 0,
        }
    }

    fn stereo(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Stereo, frames)
    }

    /// Fills a stereo buffer with independent left/right generators.
    fn fill<F: Fn(usize) -> (Sample, Sample)>(buf: &mut AudioBuffer, gen_fn: F) {
        let frames = buf.active_frames();
        for f in 0..frames {
            let (l, r) = gen_fn(f);
            buf.channel_mut(0)[f] = l;
            buf.channel_mut(1)[f] = r;
        }
    }

    fn run(node: &mut StereoWidthNode, input: AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let inputs = [input];
        let mut outputs = [stereo(frames)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [out] = outputs;
        out
    }

    #[test]
    fn unity_width_is_passthrough() {
        let mut node = StereoWidthNode::new(48_000, StereoWidthParams::default());
        let mut input = stereo(128);
        fill(&mut input, |f| {
            let x = f as Sample;
            (0.3 * ops::sin(0.05 * x), 0.7 * ops::sin(0.02 * x + 1.0))
        });
        let expected = input.clone();
        let out = run(&mut node, input);
        for f in 0..128 {
            assert!((out.channel(0)[f] - expected.channel(0)[f]).abs() < 1e-6);
            assert!((out.channel(1)[f] - expected.channel(1)[f]).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_width_collapses_to_mono() {
        let params = StereoWidthParams {
            width: 0.0,
            bass_mono_hz: 0.0,
        };
        let mut node = StereoWidthNode::new(48_000, params);
        let mut input = stereo(64);
        fill(&mut input, |f| {
            let x = f as Sample;
            (ops::sin(0.05 * x), -0.5 * ops::sin(0.03 * x))
        });
        let mono_ref: Vec<Sample> = (0..64)
            .map(|f| 0.5 * (input.channel(0)[f] + input.channel(1)[f]))
            .collect();
        let out = run(&mut node, input);
        for f in 0..64 {
            assert!((out.channel(0)[f] - out.channel(1)[f]).abs() < 1e-6);
            assert!((out.channel(0)[f] - mono_ref[f]).abs() < 1e-6);
        }
    }

    #[test]
    fn wide_increases_side_energy() {
        let make = |width: Sample| {
            let params = StereoWidthParams {
                width,
                bass_mono_hz: 0.0,
            };
            let mut node = StereoWidthNode::new(48_000, params);
            let mut input = stereo(256);
            fill(&mut input, |f| {
                let x = f as Sample;
                (ops::sin(0.05 * x), 0.4 * ops::sin(0.05 * x + 0.5))
            });
            let out = run(&mut node, input);
            let mut side_energy = 0.0;
            for f in 0..256 {
                let s = 0.5 * (out.channel(0)[f] - out.channel(1)[f]);
                side_energy += s * s;
            }
            side_energy
        };
        let narrow = make(1.0);
        let wide = make(2.0);
        assert!(wide > narrow * 3.0, "narrow={narrow} wide={wide}");
    }

    #[test]
    fn width_is_clamped_to_max() {
        let params = StereoWidthParams {
            width: 100.0,
            bass_mono_hz: 0.0,
        };
        let node = StereoWidthNode::new(48_000, params);
        assert!((node.width() - MAX_WIDTH).abs() < 1e-6);
    }

    #[test]
    fn mono_input_passes_through() {
        let mut node = StereoWidthNode::new(48_000, StereoWidthParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 32);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = (i as Sample) * 0.01;
        }
        let inputs = [input.clone()];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 32)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(32), &mut io);
        assert_eq!(outputs[0].channel(0), inputs[0].channel(0));
    }

    #[test]
    fn bass_mono_removes_low_side() {
        // Pure anti-phase low tone: L = sin, R = -sin -> mid = 0, side = sin.
        let params = StereoWidthParams {
            width: 1.0,
            bass_mono_hz: 500.0,
        };
        let mut node = StereoWidthNode::new(48_000, params);
        let mut input = stereo(4096);
        let w = 2.0 * core::f32::consts::PI * 40.0 / 48_000.0; // 40 Hz, well below 500
        fill(&mut input, |f| {
            let s = ops::sin(w * f as Sample);
            (s, -s)
        });
        let out = run(&mut node, input);
        // Steady-state tail should be strongly attenuated (bass folded to mono,
        // and mid is zero for anti-phase input).
        let mut tail = 0.0;
        for f in 3072..4096 {
            tail += out.channel(0)[f] * out.channel(0)[f];
        }
        let rms = ops::sqrt(tail / 1024.0);
        assert!(rms < 0.2, "low side not removed: rms={rms}");
    }

    #[test]
    fn bass_mono_keeps_high_side() {
        // High tone well above the crossover should pass through the side path.
        let params = StereoWidthParams {
            width: 1.0,
            bass_mono_hz: 200.0,
        };
        let mut node = StereoWidthNode::new(48_000, params);
        let mut input = stereo(4096);
        let w = 2.0 * core::f32::consts::PI * 6_000.0 / 48_000.0;
        fill(&mut input, |f| {
            let s = ops::sin(w * f as Sample);
            (s, -s)
        });
        let out = run(&mut node, input);
        let mut tail = 0.0;
        for f in 3072..4096 {
            tail += out.channel(0)[f] * out.channel(0)[f];
        }
        let rms = ops::sqrt(tail / 1024.0);
        assert!(rms > 0.6, "high side wrongly removed: rms={rms}");
    }

    #[test]
    fn extra_channels_pass_through() {
        let mut node = StereoWidthNode::new(48_000, StereoWidthParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Quad, 16);
        for ch in 0..4 {
            for f in 0..16 {
                input.channel_mut(ch)[f] = (ch as Sample) + (f as Sample) * 0.1;
            }
        }
        let inputs = [input.clone()];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Quad, 16)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(16), &mut io);
        // Channels 2 and 3 are copied verbatim.
        assert_eq!(outputs[0].channel(2), inputs[0].channel(2));
        assert_eq!(outputs[0].channel(3), inputs[0].channel(3));
    }

    #[test]
    fn output_is_finite_and_reset_clears_state() {
        let params = StereoWidthParams {
            width: 3.0,
            bass_mono_hz: 300.0,
        };
        let mut node = StereoWidthNode::new(48_000, params);
        let mut input = stereo(512);
        fill(&mut input, |f| {
            let x = f as Sample;
            (ops::sin(0.03 * x), ops::sin(0.09 * x))
        });
        let out = run(&mut node, input);
        for f in 0..512 {
            assert!(out.channel(0)[f].is_finite());
            assert!(out.channel(1)[f].is_finite());
        }
        node.reset();
        // After reset with silence in, output must be silent (no filter tail).
        let silence = stereo(128);
        let out2 = run(&mut node, silence);
        for f in 0..128 {
            assert!(out2.channel(0)[f].abs() < 1e-9);
            assert!(out2.channel(1)[f].abs() < 1e-9);
        }
    }
}
