//! Feedforward (FIR) comb filter with fractional delay and wet/dry mixing.
//!
//! A feedforward comb sums the input with a single delayed, scaled copy of
//! itself:
//!
//! ```text
//! combed[n] = x[n] + g * x[n - D]
//! y[n]      = (1 - mix) * x[n] + mix * combed[n]
//! ```
//!
//! Because the delayed term is a copy of the *input* (never of the output),
//! the filter is a finite-impulse-response structure: its impulse response is
//! just two taps and the loop can never ring or diverge. The transfer function
//! `H(z) = 1 + g * z^-D` places a regular series of peaks and notches across
//! the spectrum -- the "comb" shape. With a positive feedforward gain the
//! notches land at odd multiples of `sample_rate / (2 * D)` and the peaks at
//! the even multiples (`sample_rate / D` apart); a negative gain swaps the two.
//! At `|g| = 1` the notches are infinitely deep (perfect cancellation) and the
//! peaks reach `+6 dB`.
//!
//! Short delays (well under a millisecond up to a few milliseconds) make the
//! comb teeth dense enough to be heard as timbral coloration -- metallic
//! resonances, static flanging, phasiness, and the "doubling" sheen used to
//! thicken a source. Longer delays spread the teeth apart until the delayed
//! copy is heard as a discrete slap-back.
//!
//! # Relationship
//!
//! This is the feedforward counterpart to the recirculating effects in this
//! module. A [`CombResonatorNode`](crate::nodes::effects::CombResonatorNode) is
//! a *feedback* (IIR) comb whose delayed copy is taken from the output, so it
//! rings at a pitch and must clamp its gain for stability; this node adds a
//! single delayed copy of the input, so it is unconditionally stable for any
//! gain in `[-1, 1]` and colors the spectrum without sustaining. A
//! [`DelayNode`](crate::nodes::effects::DelayNode) targets audible echoes with
//! feedback, and a [`FlangerNode`](crate::nodes::effects::FlangerNode) sweeps a
//! short *feedback* delay with an LFO; the feedforward comb is the static,
//! non-recirculating primitive those effects build upon.
//!
//! # Real-time contract
//!
//! One ring buffer per channel is allocated in [`CombFilterNode::new`].
//! [`process`](crate::graph::AudioNode::process) performs no allocation, takes
//! no locks, and cannot panic: mismatched channel counts and zero-length
//! blocks degrade gracefully, the delayed input tap is denormal-flushed, and
//! the delay, gain, and mix are all [`Smoothed`] so automation stays
//! click-free even when the delay tap is swept.
//!
//! # Provenance
//!
//! The feedforward comb filter is textbook linear-systems theory (see, e.g.,
//! Julius Smith, "Physical Audio Signal Processing", and any standard DSP
//! treatment of FIR comb structures). This module reuses only this crate's own
//! [`Sample`] type, [`Smoothed`] parameter ramp, linear-interpolation [`lerp`],
//! and denormal-flushing primitive. It contains no code, data, or derivative
//! of Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance
//! Audio, the Web Audio API, or any other audio engine; only the shared
//! mathematical ideas are used. There is no AI or machine learning of any kind.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::buffer::ChannelLayout;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, lerp};
use crate::param::{Ramp, Smoothed};

/// Largest comb delay in milliseconds. Bounds the pre-allocated ring length.
pub const MAX_DELAY_MS: Sample = 50.0;

/// Largest feedforward gain magnitude. `|g| = 1` yields perfect notches and
/// `+6 dB` peaks; values are clamped to `[-1, 1]`.
pub const MAX_FEEDFORWARD: Sample = 1.0;

/// Default comb delay in milliseconds.
pub const DEFAULT_DELAY_MS: Sample = 5.0;

/// Default feedforward gain.
pub const DEFAULT_FEEDFORWARD: Sample = 0.7;

/// Default wet/dry blend (fully wet).
pub const DEFAULT_MIX: Sample = 1.0;

/// Returns `value` when finite, otherwise `fallback`. Guards the public setters
/// against `NaN`/infinity leaking into the delay-tap arithmetic.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Converts a delay in milliseconds to a (possibly fractional) frame count at
/// `sample_rate`, clamped to `[0, max_delay_frames]`.
#[inline]
fn ms_to_frames(ms: Sample, sample_rate: Sample, max_delay_frames: Sample) -> Sample {
    let frames = finite_or(ms, DEFAULT_DELAY_MS) * sample_rate / 1000.0;
    frames.clamp(0.0, max_delay_frames)
}

/// Construction parameters for a [`CombFilterNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CombFilterParams {
    /// Comb delay in milliseconds, clamped to `[0, MAX_DELAY_MS]`. The teeth of
    /// the comb are spaced `sample_rate / delay_frames` hertz apart.
    pub delay_ms: Sample,
    /// Feedforward gain in `[-1, 1]`. Positive values notch the odd harmonics
    /// of the comb frequency; negative values notch the even ones.
    pub feedforward: Sample,
    /// Wet/dry blend in `[0, 1]`: `0` is the untouched input, `1` is the fully
    /// combed signal.
    pub mix: Sample,
}

impl Default for CombFilterParams {
    fn default() -> Self {
        Self {
            delay_ms: DEFAULT_DELAY_MS,
            feedforward: DEFAULT_FEEDFORWARD,
            mix: DEFAULT_MIX,
        }
    }
}

/// A feedforward (FIR) comb filter (input port 0 -> output port 0).
///
/// Each channel owns an independent delay line, but all channels share the same
/// delay, gain, and mix so a stereo or surround signal is colored coherently.
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{CombFilterNode, CombFilterParams};
///
/// // A 100-frame comb at 48 kHz: delay_ms = 100 / 48.
/// let mut node = CombFilterNode::new(
///     48_000,
///     ChannelLayout::Mono,
///     CombFilterParams { delay_ms: 100.0 / 48.0, feedforward: 1.0, mix: 1.0 },
/// );
///
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 256);
/// input.set_active_frames(256);
/// input.channel_mut(0)[0] = 1.0;
///
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, 256);
/// output.set_active_frames(256);
///
/// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
/// let inputs = [input];
/// let mut outputs = [output];
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The impulse appears at 0 and its delayed copy (gain 1) at frame 100.
/// let out = &outputs[0];
/// assert!((out.channel(0)[0] - 1.0).abs() < 1e-5);
/// assert!((out.channel(0)[100] - 1.0).abs() < 1e-5);
/// ```
#[derive(Debug, Clone)]
pub struct CombFilterNode {
    /// Sample rate in Hz, used to convert delay times expressed in ms.
    sample_rate: u32,
    /// Ring length in frames (`max_delay + 2`), shared by every channel.
    ring_len: usize,
    /// One delay line per channel; each holds exactly `ring_len` input samples.
    rings: Vec<Vec<Sample>>,
    /// Shared write cursor into every channel's ring buffer.
    write_pos: usize,
    /// Maximum addressable delay in frames (`ring_len - 2`).
    max_delay: Sample,
    /// Smoothed loop delay in frames (possibly fractional).
    delay: Smoothed,
    /// Smoothed feedforward gain in `[-1, 1]`.
    feedforward: Smoothed,
    /// Smoothed wet/dry blend in `[0, 1]`.
    mix: Smoothed,
}

impl CombFilterNode {
    /// Builds a feedforward comb for `layout`'s channels running at
    /// `sample_rate` Hz.
    ///
    /// The ring buffer is sized so that a delay as long as [`MAX_DELAY_MS`]
    /// fits. All parameters are sanitized: non-finite values fall back to safe
    /// defaults, `delay_ms` is clamped to `[0, MAX_DELAY_MS]`, `feedforward` to
    /// `[-1, 1]`, and `mix` to `[0, 1]`.
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout, params: CombFilterParams) -> Self {
        let channels = layout.channel_count();
        let sr = sample_rate.max(1) as Sample;
        let max_delay_frames = ops::round(sr * MAX_DELAY_MS / 1000.0) as usize;
        let ring_len = max_delay_frames + 2;

        let mut rings = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut ring = Vec::with_capacity(ring_len);
            ring.resize(ring_len, 0.0);
            rings.push(ring);
        }

        let max_delay = max_delay_frames as Sample;
        let delay = ms_to_frames(params.delay_ms, sr, max_delay);
        let feedforward =
            finite_or(params.feedforward, DEFAULT_FEEDFORWARD).clamp(-MAX_FEEDFORWARD, MAX_FEEDFORWARD);
        let mix = finite_or(params.mix, DEFAULT_MIX).clamp(0.0, 1.0);

        Self {
            sample_rate: sample_rate.max(1),
            ring_len,
            rings,
            write_pos: 0,
            max_delay,
            delay: Smoothed::new(delay),
            feedforward: Smoothed::new(feedforward),
            mix: Smoothed::new(mix),
        }
    }

    /// Returns the number of channels this comb processes.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.rings.len()
    }

    /// Returns the target delay in frames the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn delay_frames(&self) -> Sample {
        self.delay.target()
    }

    /// Returns the target delay in milliseconds the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn delay_ms(&self) -> Sample {
        self.delay.target() * 1000.0 / self.sample_rate as Sample
    }

    /// Returns the target feedforward gain the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn feedforward(&self) -> Sample {
        self.feedforward.target()
    }

    /// Returns the target wet/dry blend the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn mix(&self) -> Sample {
        self.mix.target()
    }

    /// Sets a new target delay in milliseconds, gliding with `ramp`. The target
    /// is clamped to `[0, MAX_DELAY_MS]`.
    #[inline]
    pub fn set_delay_ms(&mut self, delay_ms: Sample, ramp: Ramp) {
        let frames = ms_to_frames(delay_ms, self.sample_rate as Sample, self.max_delay);
        self.delay.set_target(frames, ramp);
    }

    /// Sets a new target feedforward gain, gliding with `ramp`. The target is
    /// clamped to `[-1, 1]`.
    #[inline]
    pub fn set_feedforward(&mut self, feedforward: Sample, ramp: Ramp) {
        let g = finite_or(feedforward, self.feedforward.target())
            .clamp(-MAX_FEEDFORWARD, MAX_FEEDFORWARD);
        self.feedforward.set_target(g, ramp);
    }

    /// Sets a new target wet/dry blend, gliding with `ramp`. The target is
    /// clamped to `[0, 1]`.
    #[inline]
    pub fn set_mix(&mut self, mix: Sample, ramp: Ramp) {
        let m = finite_or(mix, self.mix.target()).clamp(0.0, 1.0);
        self.mix.set_target(m, ramp);
    }
}

impl AudioNode for CombFilterNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels()).min(self.rings.len());
        let frames = output.active_frames();
        if frames == 0 || channels == 0 {
            return;
        }

        let ring_len = self.ring_len;
        let len_i = ring_len as isize;

        for f in 0..frames {
            // Advance the shared smoothed controls exactly once per frame.
            let delay = self.delay.next_sample();
            let gain = self.feedforward.next_sample();
            let mix = self.mix.next_sample();
            let dry = 1.0 - mix;

            let w = self.write_pos;
            // Fractional read position `delay` frames behind the write head.
            // Writing happens first (below), so `delay == 0` reads the current
            // input and the comb degenerates to `(1 + g) * x`.
            let read_pos = w as Sample - delay;
            let base = ops::floor(read_pos);
            let frac = read_pos - base;
            let base_i = base as isize;
            let i0 = base_i.rem_euclid(len_i) as usize;
            let i1 = (base_i + 1).rem_euclid(len_i) as usize;

            for ch in 0..channels {
                let x = input.channel(ch)[f];
                // Store the current input before reading so a zero delay reads
                // it back and so the FIR tap only ever sees past inputs.
                self.rings[ch][w] = flush_denormal(x);
                let delayed = {
                    let ring = &self.rings[ch];
                    lerp(ring[i0], ring[i1], frac)
                };
                let combed = x + gain * delayed;
                output.channel_mut(ch)[f] = dry * x + mix * combed;
            }

            self.write_pos = if w + 1 == ring_len { 0 } else { w + 1 };
        }
    }

    fn reset(&mut self) {
        for ring in &mut self.rings {
            for s in ring.iter_mut() {
                *s = 0.0;
            }
        }
        self.write_pos = 0;
        self.delay = Smoothed::new(self.delay.target());
        self.feedforward = Smoothed::new(self.feedforward.target());
        self.mix = Smoothed::new(self.mix.target());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::AudioBuffer;

    const SR: u32 = 48_000;
    const TAU: Sample = core::f32::consts::TAU;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn signal(layout: ChannelLayout, frames: usize, per_channel: &[&[Sample]]) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        buf.set_active_frames(frames);
        for (ch, data) in per_channel.iter().enumerate() {
            let dst = buf.channel_mut(ch);
            for (d, &s) in dst.iter_mut().zip(data.iter()) {
                *d = s;
            }
        }
        buf
    }

    fn impulse(layout: ChannelLayout, frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        buf.set_active_frames(frames);
        for ch in 0..buf.channels() {
            buf.channel_mut(ch)[0] = 1.0;
        }
        buf
    }

    fn sine(layout: ChannelLayout, frames: usize, freq: Sample) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        buf.set_active_frames(frames);
        for ch in 0..buf.channels() {
            let dst = buf.channel_mut(ch);
            for (n, s) in dst.iter_mut().enumerate() {
                *s = ops::sin(TAU * freq * n as Sample / SR as Sample);
            }
        }
        buf
    }

    fn run(node: &mut CombFilterNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let mut out = AudioBuffer::new(input.layout(), input.capacity_frames());
        out.set_active_frames(frames);
        let inputs = [input.clone()];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [o] = outputs;
        o
    }

    fn node(delay_ms: Sample, feedforward: Sample, mix: Sample) -> CombFilterNode {
        CombFilterNode::new(
            SR,
            ChannelLayout::Mono,
            CombFilterParams { delay_ms, feedforward, mix },
        )
    }

    #[test]
    fn mix_zero_is_bypass() {
        let input = signal(ChannelLayout::Mono, 128, &[&[0.4; 128]]);
        let mut n = node(5.0, 0.9, 0.0);
        let out = run(&mut n, &input);
        for (o, i) in out.channel(0).iter().zip(input.channel(0)) {
            assert!((o - i).abs() < 1e-6, "not bypassed: {o} vs {i}");
        }
    }

    #[test]
    fn silence_in_silence_out() {
        let input = signal(ChannelLayout::Mono, 128, &[&[0.0; 128]]);
        let mut n = node(5.0, 0.9, 1.0);
        let out = run(&mut n, &input);
        for &s in out.channel(0) {
            assert!(s.abs() < 1e-9, "expected silence, got {s}");
        }
    }

    #[test]
    fn feedforward_adds_delayed_copy() {
        // D = 100 frames, g = 0.5 => impulse at 0, scaled copy at 100, nothing
        // else (the FIR tail is exactly two samples long).
        let input = impulse(ChannelLayout::Mono, 256);
        let mut n = node(100.0 / 48.0, 0.5, 1.0);
        let out = run(&mut n, &input);
        assert!((out.channel(0)[0] - 1.0).abs() < 1e-5);
        assert!((out.channel(0)[100] - 0.5).abs() < 1e-5);
        for (i, &s) in out.channel(0).iter().enumerate() {
            if i != 0 && i != 100 {
                assert!(s.abs() < 1e-5, "unexpected energy at {i}: {s}");
            }
        }
    }

    #[test]
    fn zero_delay_degenerates_to_gain() {
        // delay 0 => y = x + g*x = (1 + g) * x.
        let input = signal(ChannelLayout::Mono, 64, &[&[0.5; 64]]);
        let mut n = node(0.0, 0.5, 1.0);
        let out = run(&mut n, &input);
        for &s in out.channel(0) {
            assert!((s - 0.75).abs() < 1e-5, "got {s}");
        }
    }

    #[test]
    fn notch_cancels_at_half_comb_frequency() {
        // D = 100 frames => comb teeth 480 Hz apart; first notch at 240 Hz.
        // With g = 1 a 240 Hz sine cancels once the delay line is primed.
        let input = sine(ChannelLayout::Mono, 2048, 240.0);
        let mut n = node(100.0 / 48.0, 1.0, 1.0);
        let out = run(&mut n, &input);
        // Skip the first D samples (the delay line is still filling).
        let tail = &out.channel(0)[200..];
        let energy: Sample = tail.iter().map(|s| s * s).sum::<Sample>() / tail.len() as Sample;
        assert!(energy < 1e-3, "notch did not cancel: rms^2 {energy}");
    }

    #[test]
    fn peak_reinforces_at_comb_frequency() {
        // 480 Hz aligns with the comb peak; g = 1 => +6 dB (doubling).
        let input = sine(ChannelLayout::Mono, 2048, 480.0);
        let mut n = node(100.0 / 48.0, 1.0, 1.0);
        let out = run(&mut n, &input);
        let tail = &out.channel(0)[200..];
        let peak = tail.iter().fold(0.0_f32, |m, &s| m.max(s.abs()));
        assert!((peak - 2.0).abs() < 2e-2, "expected doubling, got peak {peak}");
    }

    #[test]
    fn negative_gain_swaps_notch_and_peak() {
        // With g = -1 the roles invert: 480 Hz now cancels.
        let input = sine(ChannelLayout::Mono, 2048, 480.0);
        let mut n = node(100.0 / 48.0, -1.0, 1.0);
        let out = run(&mut n, &input);
        let tail = &out.channel(0)[200..];
        let energy: Sample = tail.iter().map(|s| s * s).sum::<Sample>() / tail.len() as Sample;
        assert!(energy < 1e-3, "inverted notch did not cancel: rms^2 {energy}");
    }

    #[test]
    fn fractional_delay_interpolates() {
        // A non-integer delay must still place a fractional-weighted copy.
        let input = impulse(ChannelLayout::Mono, 256);
        let mut n = node(100.5 / 48.0, 1.0, 1.0);
        let out = run(&mut n, &input);
        // The impulse lands between frames 100 and 101, split by interpolation.
        let a = out.channel(0)[100];
        let b = out.channel(0)[101];
        assert!(a > 0.3 && a < 0.7, "frame 100 weight {a}");
        assert!(b > 0.3 && b < 0.7, "frame 101 weight {b}");
        assert!((a + b - 1.0).abs() < 1e-3, "weights should sum to ~1: {a}+{b}");
    }

    #[test]
    fn mix_blends_dry_and_wet() {
        let input = impulse(ChannelLayout::Mono, 256);
        let mut n = node(100.0 / 48.0, 1.0, 0.5);
        let out = run(&mut n, &input);
        // Wet copy at frame 100 is scaled by mix (0.5).
        assert!((out.channel(0)[100] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn feedforward_is_clamped() {
        let n = node(5.0, 4.0, 1.0);
        assert_eq!(n.feedforward(), MAX_FEEDFORWARD);
        let n2 = node(5.0, -4.0, 1.0);
        assert_eq!(n2.feedforward(), -MAX_FEEDFORWARD);
    }

    #[test]
    fn delay_is_clamped() {
        let n = node(10_000.0, 0.5, 1.0);
        assert!((n.delay_ms() - MAX_DELAY_MS).abs() < 1e-3, "delay {}", n.delay_ms());
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let n = node(Sample::NAN, Sample::INFINITY, Sample::NAN);
        assert_eq!(n.feedforward(), DEFAULT_FEEDFORWARD);
        assert_eq!(n.mix(), DEFAULT_MIX);
        assert!((n.delay_ms() - DEFAULT_DELAY_MS).abs() < 1e-3);
    }

    #[test]
    fn from_params_matches_direct_fields() {
        let p = CombFilterParams { delay_ms: 3.0, feedforward: 0.4, mix: 0.8 };
        let n = CombFilterNode::new(SR, ChannelLayout::Mono, p);
        assert!((n.delay_ms() - 3.0).abs() < 1e-3);
        assert_eq!(n.feedforward(), 0.4);
        assert_eq!(n.mix(), 0.8);
    }

    #[test]
    fn default_params_are_sane() {
        let n = CombFilterNode::new(SR, ChannelLayout::Mono, CombFilterParams::default());
        assert!((n.delay_ms() - DEFAULT_DELAY_MS).abs() < 1e-3);
        assert_eq!(n.feedforward(), DEFAULT_FEEDFORWARD);
        assert_eq!(n.mix(), DEFAULT_MIX);
    }

    #[test]
    fn stereo_channels_are_coherent() {
        let data: Vec<Sample> = (0..256)
            .map(|n| ops::sin(TAU * 220.0 * n as Sample / SR as Sample))
            .collect();
        let input = signal(ChannelLayout::Stereo, 256, &[&data, &data]);
        let mut n = CombFilterNode::new(SR, ChannelLayout::Stereo, CombFilterParams::default());
        let out = run(&mut n, &input);
        assert_eq!(out.channel(0), out.channel(1));
    }

    #[test]
    fn reset_clears_state() {
        let input = impulse(ChannelLayout::Mono, 256);
        let mut n = node(100.0 / 48.0, 1.0, 1.0);
        let first = run(&mut n, &input);
        n.reset();
        let second = run(&mut n, &input);
        for (a, b) in first.channel(0).iter().zip(second.channel(0)) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn delay_ramp_is_click_free() {
        let input = signal(ChannelLayout::Mono, 512, &[&[0.3; 512]]);
        let mut n = node(1.0, 0.8, 1.0);
        n.set_delay_ms(20.0, Ramp::Linear { samples: 512 });
        let out = run(&mut n, &input);
        for &s in out.channel(0) {
            assert!(s.is_finite());
            assert!(s.abs() <= 2.0 + 1e-4);
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let inputs = [AudioBuffer::new(ChannelLayout::Mono, 32)];
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 32);
        out.set_active_frames(0);
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        let mut n = node(5.0, 0.5, 1.0);
        n.process(&ctx(0), &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }

    #[test]
    fn channel_mismatch_is_graceful() {
        // Stereo node fed a mono input processes only the shared channel.
        let input = signal(ChannelLayout::Mono, 64, &[&[0.2; 64]]);
        let mut n = CombFilterNode::new(SR, ChannelLayout::Stereo, CombFilterParams::default());
        let frames = input.active_frames();
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
        out.set_active_frames(frames);
        let inputs = [input];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        n.process(&ctx(frames), &mut io);
        for &s in outputs[0].channel(0) {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn getters_report_targets() {
        let mut n = node(5.0, 0.5, 1.0);
        n.set_delay_ms(8.0, Ramp::Immediate);
        n.set_feedforward(-0.3, Ramp::Immediate);
        n.set_mix(0.25, Ramp::Immediate);
        assert!((n.delay_ms() - 8.0).abs() < 1e-3);
        assert_eq!(n.feedforward(), -0.3);
        assert_eq!(n.mix(), 0.25);
        assert_eq!(n.channels(), 1);
    }

    #[test]
    fn output_is_bounded_by_twice_input() {
        let input = sine(ChannelLayout::Mono, 1024, 330.0);
        let mut n = node(2.0, 1.0, 1.0);
        let out = run(&mut n, &input);
        for &s in out.channel(0) {
            assert!(s.is_finite());
            assert!(s.abs() <= 2.0 + 1e-4, "exceeded doubling: {s}");
        }
    }
}
