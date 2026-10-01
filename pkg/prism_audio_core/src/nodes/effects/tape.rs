//! Tape machine emulation: soft magnetic saturation, wow / flutter speed
//! modulation, and head / gap high-frequency loss, combined into one classic
//! analog-tape "warmth and movement" effect.
//!
//! Real analog tape colours a signal in three well-understood ways:
//!
//! 1. The magnetic medium saturates, gently compressing peaks and adding
//!    harmonics (an asymmetric transfer curve adds both even and odd orders).
//! 2. Small, slow speed variations of the transport (wow, a fraction of a Hz to
//!    a couple of Hz) and faster ones (flutter, several Hz) continuously shift
//!    the playback pitch by resampling through a time-varying delay.
//! 3. The record / playback head gap and the medium itself roll off the top
//!    octaves, softening the treble.
//!
//! This module reproduces all three with plain, memoryless-plus-one-pole DSP:
//! a drive / bias soft-saturator, a one-pole treble roll-off, and a fractional
//! delay line swept by two [`Lfo`] oscillators (wow and flutter). The ring and
//! all filter state are allocated at construction, so
//! [`TapeNode::process`](crate::graph::AudioNode::process) is real-time safe.
//!
//! # Signal model
//!
//! For each channel the per-sample chain is:
//!
//! ```text
//! saturated = saturate(x)                 // magnetic soft saturation
//! filtered  = one_pole_lowpass(saturated) // head / gap HF loss
//! wet       = delay_tap(filtered, d[n])   // wow + flutter resampling
//! y         = (1 - mix) * x + mix * wet
//! ```
//!
//! The instantaneous read delay in frames is
//!
//! ```text
//! d[n] = center + wow_depth * wow[n] + flutter_depth * flutter[n]
//! ```
//!
//! where `center = wow_depth + flutter_depth` keeps the read head behind the
//! write head, and `d[n]` is clamped to `[1, max_delay]` so the fractional read
//! always straddles two valid ring samples.
//!
//! The saturator is a drive / bias `tanh` curve normalised to unity
//! small-signal gain and corrected to pass through the origin:
//!
//! ```text
//! saturate(x) = (tanh(drive * x + bias) - tanh(bias))
//!             / (drive * (1 - tanh(bias)^2))
//! ```
//!
//! A non-zero `bias` makes the curve asymmetric, which is what produces the
//! even-harmonic component characteristic of tape.
//!
//! # Provenance
//!
//! Tape warmth, wow / flutter, and head-gap loss are long-standing, publicly
//! documented audio phenomena described in texts such as Udo Zolzer, "DAFX:
//! Digital Audio Effects". The saturating `tanh` transfer curve, the one-pole
//! treble filter, and the LFO-swept fractional delay below are standard DSP
//! primitives re-derived from that public literature and composed on top of
//! this crate's own [`Lfo`] oscillator. This file contains **no Unreal Engine,
//! Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or
//! derived code**; it is implemented purely from that publicly documented
//! mathematics.
//!
//! # Relationship
//!
//! `tape` composes an existing primitive rather than reimplementing it: both
//! speed-variation oscillators are [`Lfo`] instances (the same control-rate
//! oscillator used by [`vibrato`](super::vibrato) and
//! [`chorus`](super::chorus)). Where [`vibrato`](super::vibrato) sweeps a single
//! delay for pitch movement alone, `tape` layers two slow oscillators for
//! wow / flutter and additionally applies saturation and a treble roll-off, so
//! the result is the full tape colour rather than pitch modulation only. The
//! saturator is kept local (a drive / bias `tanh`) rather than reusing the
//! oversampled [`waveshaper`](super::waveshaper), because the following treble
//! roll-off already tames the gentle high products.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, lerp};
use crate::modulation::{Lfo, LfoWaveform};
use crate::param::{Ramp, Smoothed};

/// Largest drive fed into the saturator, so a very hot input cannot push the
/// normalisation into a numerically pathological region.
const MAX_DRIVE: Sample = 64.0;

/// Largest absolute bias allowed, keeping the normalising term
/// `1 - tanh(bias)^2` safely away from zero.
const MAX_BIAS: Sample = 3.0;

/// Largest wow / flutter delay excursion in milliseconds. Real transports
/// drift by only a few milliseconds; this generous cap bounds the ring-buffer
/// allocation so a pathological parameter cannot request an unbounded delay
/// line.
const MAX_DELAY_MS: Sample = 100.0;

/// Clamps a wow / flutter excursion to a finite, bounded millisecond range.
#[inline]
#[must_use]
fn clamp_depth_ms(ms: Sample) -> Sample {
    if ms.is_finite() {
        ms.clamp(0.0, MAX_DELAY_MS)
    } else {
        0.0
    }
}

/// Converts milliseconds to (fractional) frames at `sample_rate`.
#[inline]
#[must_use]
fn ms_to_frames(ms: Sample, sample_rate: u32) -> Sample {
    ms.max(0.0) * (sample_rate as Sample) * 0.001
}

/// Designs the one-pole low-pass coefficient for a cutoff of `fc` Hz.
///
/// Returns the smoothing factor `a` in `y += a * (x - y)`, with
/// `a = 1 - exp(-2 * pi * fc / sr)` clamped to `[0, 1]`.
#[inline]
#[must_use]
fn one_pole_coef(fc: Sample, sample_rate: u32) -> Sample {
    let sr = (sample_rate.max(1)) as Sample;
    let nyquist = sr * 0.5;
    let cutoff = fc.clamp(1.0, nyquist * 0.999);
    let a = 1.0 - ops::exp(-core::f32::consts::TAU * cutoff / sr);
    a.clamp(0.0, 1.0)
}

/// Construction parameters for a [`TapeNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TapeParams {
    /// Saturation drive; higher compresses peaks harder and adds more
    /// harmonics.
    pub drive: Sample,
    /// Transfer-curve asymmetry bias; non-zero adds even harmonics.
    pub bias: Sample,
    /// Wow rate in Hz (slow speed drift, typically well below 2 Hz).
    pub wow_hz: Sample,
    /// Peak wow delay excursion in milliseconds.
    pub wow_depth_ms: Sample,
    /// Flutter rate in Hz (faster speed jitter, typically several Hz).
    pub flutter_hz: Sample,
    /// Peak flutter delay excursion in milliseconds.
    pub flutter_depth_ms: Sample,
    /// Treble roll-off cutoff in Hz (head / gap high-frequency loss).
    pub hf_rolloff_hz: Sample,
    /// Wet/dry blend in `[0, 1]`: `0` is the untouched input, `1` is fully
    /// processed tape.
    pub mix: Sample,
}

impl Default for TapeParams {
    fn default() -> Self {
        Self {
            drive: 2.0,
            bias: 0.3,
            wow_hz: 0.7,
            wow_depth_ms: 2.0,
            flutter_hz: 7.0,
            flutter_depth_ms: 0.4,
            hf_rolloff_hz: 12_000.0,
            mix: 1.0,
        }
    }
}

/// Allocation-free tape DSP core: a drive / bias saturator, a one-pole treble
/// roll-off, and a wow / flutter modulated delay line.
///
/// All state is pre-allocated at construction, so [`Tape::advance`] and
/// [`Tape::voice`] are real-time safe (no allocation, no locking, no panic).
/// The owning [`TapeNode`] performs the dry blend.
#[derive(Debug, Clone)]
pub struct Tape {
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Ring length in frames, shared by every channel.
    ring_len: usize,
    /// One delay ring per channel.
    rings: Vec<Vec<Sample>>,
    /// One treble-filter state per channel.
    lp_state: Vec<Sample>,
    /// Shared write cursor into every channel's ring.
    write_pos: usize,
    /// Largest addressable delay in frames (`ring_len - 2`).
    max_delay: Sample,
    /// Center delay in frames (`wow_depth + flutter_depth`).
    center_delay: Sample,
    /// Wow depth in frames.
    wow_depth: Sample,
    /// Flutter depth in frames.
    flutter_depth: Sample,
    /// Slow wow oscillator.
    wow: Lfo,
    /// Faster flutter oscillator.
    flutter: Lfo,
    /// Saturation drive (clamped to a safe finite range).
    drive: Sample,
    /// Saturation bias (clamped to a safe finite range).
    bias: Sample,
    /// Precomputed `tanh(bias)` for the DC-correction term.
    tanh_bias: Sample,
    /// Precomputed normalising gain `drive * (1 - tanh(bias)^2)`.
    norm: Sample,
    /// One-pole treble-filter coefficient.
    lp_coef: Sample,
}

impl Tape {
    /// Builds a tape core for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: TapeParams, sample_rate: u32, channels: usize) -> Self {
        let sr = sample_rate.max(1);
        let channels = channels.max(1);

        let wow_depth = ms_to_frames(clamp_depth_ms(params.wow_depth_ms), sr);
        let flutter_depth = ms_to_frames(clamp_depth_ms(params.flutter_depth_ms), sr);
        let center_delay = wow_depth + flutter_depth;
        // The ring must hold the deepest excursion (2 * center) plus slack.
        let span = (2.0 * center_delay).max(1.0);
        let ring_len = (ops::round(span) as usize) + 2;
        let max_delay = (ring_len as Sample) - 2.0;

        let mut rings = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut ring = Vec::with_capacity(ring_len);
            ring.resize(ring_len, 0.0);
            rings.push(ring);
        }
        let mut lp_state = Vec::with_capacity(channels);
        lp_state.resize(channels, 0.0);

        let drive = clamp_drive(params.drive);
        let bias = clamp_bias(params.bias);
        let tanh_bias = ops::tanh(bias);

        Self {
            sample_rate: sr,
            ring_len,
            rings,
            lp_state,
            write_pos: 0,
            max_delay,
            center_delay,
            wow_depth,
            flutter_depth,
            wow: Lfo::new(sr, params.wow_hz, LfoWaveform::Sine),
            flutter: Lfo::new(sr, params.flutter_hz, LfoWaveform::Sine),
            drive,
            bias,
            tanh_bias,
            norm: normalising_gain(drive, tanh_bias),
            lp_coef: one_pole_coef(params.hf_rolloff_hz, sr),
        }
    }

    /// Number of channels the core tracks.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.rings.len()
    }

    /// Current saturation drive.
    #[inline]
    #[must_use]
    pub fn drive(&self) -> Sample {
        self.drive
    }

    /// Current saturation bias.
    #[inline]
    #[must_use]
    pub fn bias(&self) -> Sample {
        self.bias
    }

    /// Sets the saturation drive (clamped to a safe finite range).
    #[inline]
    pub fn set_drive(&mut self, drive: Sample) {
        self.drive = clamp_drive(drive);
        self.norm = normalising_gain(self.drive, self.tanh_bias);
    }

    /// Sets the saturation bias (clamped to a safe finite range).
    #[inline]
    pub fn set_bias(&mut self, bias: Sample) {
        self.bias = clamp_bias(bias);
        self.tanh_bias = ops::tanh(self.bias);
        self.norm = normalising_gain(self.drive, self.tanh_bias);
    }

    /// Sets the treble roll-off cutoff in Hz.
    #[inline]
    pub fn set_hf_rolloff_hz(&mut self, hf_rolloff_hz: Sample) {
        self.lp_coef = one_pole_coef(hf_rolloff_hz, self.sample_rate);
    }

    /// Sets the wow rate in Hz.
    #[inline]
    pub fn set_wow_hz(&mut self, wow_hz: Sample) {
        self.wow.set_frequency(self.sample_rate, wow_hz);
    }

    /// Sets the flutter rate in Hz.
    #[inline]
    pub fn set_flutter_hz(&mut self, flutter_hz: Sample) {
        self.flutter.set_frequency(self.sample_rate, flutter_hz);
    }

    /// Soft-saturates one sample through the drive / bias `tanh` curve,
    /// normalised to unity small-signal gain and corrected to pass through the
    /// origin.
    #[inline]
    #[must_use]
    pub fn saturate(&self, x: Sample) -> Sample {
        if self.norm <= 0.0 {
            return x;
        }
        let driven = self.drive * x + self.bias;
        (ops::tanh(driven) - self.tanh_bias) / self.norm
    }

    /// Advances the wow and flutter oscillators by one frame and returns the
    /// instantaneous read delay in frames, clamped to `[1, max_delay]`.
    #[inline]
    pub fn advance(&mut self) -> Sample {
        let wow = self.wow.next_sample();
        let flutter = self.flutter.next_sample();
        let delay = self.center_delay + self.wow_depth * wow + self.flutter_depth * flutter;
        delay.clamp(1.0, self.max_delay)
    }

    /// Processes one sample on channel `ch` at the given read `delay` and
    /// returns the fully processed (saturated, treble-rolled, delayed) wet
    /// sample. The dry blend is performed by the owning node.
    ///
    /// Non-finite input is treated as silence so the delay line and filter
    /// state cannot be poisoned into a NaN / infinity state.
    #[inline]
    pub fn voice(&mut self, ch: usize, x: Sample, delay: Sample) -> Sample {
        let input = if x.is_finite() { x } else { 0.0 };
        let saturated = self.saturate(input);
        let state = self.lp_state[ch] + self.lp_coef * (saturated - self.lp_state[ch]);
        let filtered = flush_denormal(state);
        self.lp_state[ch] = filtered;

        let wet = self.read_tap(ch, self.write_pos, delay);
        self.rings[ch][self.write_pos] = filtered;
        wet
    }

    /// Advances the shared write cursor by one frame. Call once per frame after
    /// every channel's [`Tape::voice`].
    #[inline]
    pub fn commit_frame(&mut self) {
        self.write_pos = if self.write_pos + 1 == self.ring_len {
            0
        } else {
            self.write_pos + 1
        };
    }

    /// Reads a linear-interpolated fractional tap `delay` frames behind
    /// `write_pos` on channel `ch`.
    #[inline]
    #[must_use]
    fn read_tap(&self, ch: usize, write_pos: usize, delay: Sample) -> Sample {
        let len_i = self.ring_len as isize;
        let read_pos = write_pos as Sample - delay;
        let base = ops::floor(read_pos);
        let frac = read_pos - base;
        let base_i = base as isize;
        let i0 = base_i.rem_euclid(len_i) as usize;
        let i1 = (base_i + 1).rem_euclid(len_i) as usize;
        let ring = &self.rings[ch];
        lerp(ring[i0], ring[i1], frac)
    }

    /// Clears all delay rings, filter state, and resets the oscillators.
    #[inline]
    pub fn reset(&mut self) {
        for ring in &mut self.rings {
            for s in ring.iter_mut() {
                *s = 0.0;
            }
        }
        for s in &mut self.lp_state {
            *s = 0.0;
        }
        self.write_pos = 0;
        self.wow.reset();
        self.flutter.reset();
    }
}

/// Clamps a drive request to a safe, finite, non-negative range.
#[inline]
#[must_use]
fn clamp_drive(drive: Sample) -> Sample {
    if drive.is_finite() {
        drive.clamp(0.0, MAX_DRIVE)
    } else {
        0.0
    }
}

/// Clamps a bias request to a safe, finite range.
#[inline]
#[must_use]
fn clamp_bias(bias: Sample) -> Sample {
    if bias.is_finite() {
        bias.clamp(-MAX_BIAS, MAX_BIAS)
    } else {
        0.0
    }
}

/// Computes the small-signal normalising gain `drive * (1 - tanh(bias)^2)`.
#[inline]
#[must_use]
fn normalising_gain(drive: Sample, tanh_bias: Sample) -> Sample {
    drive * (1.0 - tanh_bias * tanh_bias)
}

/// A tape-emulation node (input port 0 -> output port 0).
///
/// The dry signal is blended with the processed tape path by a [`Smoothed`]
/// `mix` so automation stays click-free. Every channel is processed
/// independently from shared coefficients and a shared pair of wow / flutter
/// oscillators, so a mono source fed to several channels stays phase-coherent.
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{TapeNode, TapeParams};
///
/// let mut node = TapeNode::new(TapeParams::default(), 48_000, 1);
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, 8);
/// input.set_active_frames(8);
/// output.set_active_frames(8);
/// for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
///     *s = if i % 2 == 0 { 0.5 } else { -0.5 };
/// }
/// let ctx = RenderContext { sample_rate: 48_000, frames: 8, playhead: 0 };
/// let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
/// node.process(&ctx, &mut io);
/// assert!(output.channel(0).iter().all(|s| s.is_finite()));
/// ```
#[derive(Debug, Clone)]
pub struct TapeNode {
    /// DSP core (saturation + treble roll-off + wow / flutter delay).
    tape: Tape,
    /// Smoothed wet/dry blend in `[0, 1]`.
    mix: Smoothed,
}

impl TapeNode {
    /// Builds a tape node for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: TapeParams, sample_rate: u32, channels: usize) -> Self {
        Self {
            mix: Smoothed::new(params.mix.clamp(0.0, 1.0)),
            tape: Tape::new(params, sample_rate, channels),
        }
    }

    /// Number of channels the node processes.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.tape.channels()
    }

    /// Sets the saturation drive.
    #[inline]
    pub fn set_drive(&mut self, drive: Sample) {
        self.tape.set_drive(drive);
    }

    /// Sets the saturation bias.
    #[inline]
    pub fn set_bias(&mut self, bias: Sample) {
        self.tape.set_bias(bias);
    }

    /// Sets the treble roll-off cutoff in Hz.
    #[inline]
    pub fn set_hf_rolloff_hz(&mut self, hf_rolloff_hz: Sample) {
        self.tape.set_hf_rolloff_hz(hf_rolloff_hz);
    }

    /// Sets the wet/dry blend with the given ramp.
    #[inline]
    pub fn set_mix(&mut self, mix: Sample, ramp: Ramp) {
        self.mix.set_target(mix.clamp(0.0, 1.0), ramp);
    }
}

impl AudioNode for TapeNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(self.tape.channels());
        let frames = output.active_frames();

        for f in 0..frames {
            let mix = self.mix.next_sample();
            let delay = self.tape.advance();
            for ch in 0..channels {
                let raw = input.channel(ch)[f];
                let x = if raw.is_finite() { raw } else { 0.0 };
                let wet = self.tape.voice(ch, x, delay);
                output.channel_mut(ch)[f] = (1.0 - mix) * x + mix * wet;
            }
            self.tape.commit_frame();
        }
    }

    fn reset(&mut self) {
        self.tape.reset();
        self.mix = Smoothed::new(self.mix.target());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    fn params() -> TapeParams {
        TapeParams::default()
    }

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn mono(len: usize) -> AudioBuffer {
        let mut b = AudioBuffer::new(ChannelLayout::Mono, len);
        b.set_active_frames(len);
        b
    }

    fn sine(buf: &mut AudioBuffer, freq_hz: Sample, amp: Sample) {
        let len = buf.active_frames();
        for i in 0..len {
            let phase = core::f32::consts::TAU * freq_hz * (i as Sample) / (SR as Sample);
            buf.channel_mut(0)[i] = amp * ops::sin(phase);
        }
    }

    #[test]
    fn saturation_is_bounded_and_finite() {
        let tape = Tape::new(params(), SR, 1);
        for &x in &[-1000.0, -1.0, 0.0, 1.0, 1000.0] {
            let y = tape.saturate(x);
            assert!(y.is_finite(), "saturate not finite at {x}: {y}");
        }
    }

    #[test]
    fn saturation_compresses_large_peaks() {
        // A hot input is compressed below its linear value, while a tiny input
        // keeps near-unity small-signal gain.
        let tape = Tape::new(params(), SR, 1);
        let big = tape.saturate(4.0);
        assert!(big.abs() < 4.0, "large peak was not compressed: {big}");
        let small = tape.saturate(0.001);
        assert!((small - 0.001).abs() < 1.0e-4, "small-signal gain drifted: {small}");
    }

    #[test]
    fn saturation_passes_through_origin() {
        let tape = Tape::new(params(), SR, 1);
        assert!(tape.saturate(0.0).abs() < 1.0e-6);
    }

    #[test]
    fn bias_breaks_symmetry() {
        // With a non-zero bias the curve is asymmetric, so +-x do not cancel.
        let tape = Tape::new(params(), SR, 1);
        let pos = tape.saturate(0.7);
        let neg = tape.saturate(-0.7);
        assert!((pos + neg).abs() > 1.0e-3, "biased curve should be asymmetric");
    }

    #[test]
    fn zero_bias_is_symmetric() {
        let mut p = params();
        p.bias = 0.0;
        let tape = Tape::new(p, SR, 1);
        let pos = tape.saturate(0.7);
        let neg = tape.saturate(-0.7);
        assert!((pos + neg).abs() < 1.0e-6, "unbiased curve should be symmetric");
    }

    #[test]
    fn zero_drive_is_safe_passthrough() {
        let mut p = params();
        p.drive = 0.0;
        let tape = Tape::new(p, SR, 1);
        // norm becomes zero, so saturate falls back to identity.
        assert!((tape.saturate(0.5) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn drive_is_clamped_to_safe_range() {
        let mut tape = Tape::new(params(), SR, 1);
        tape.set_drive(Sample::INFINITY);
        assert_eq!(tape.drive(), 0.0);
        tape.set_drive(1.0e9);
        assert!(tape.drive() <= MAX_DRIVE + 1.0e-3);
        tape.set_drive(-5.0);
        assert_eq!(tape.drive(), 0.0);
    }

    #[test]
    fn bias_is_clamped_to_safe_range() {
        let mut tape = Tape::new(params(), SR, 1);
        tape.set_bias(Sample::NAN);
        assert_eq!(tape.bias(), 0.0);
        tape.set_bias(100.0);
        assert!(tape.bias() <= MAX_BIAS + 1.0e-3);
        tape.set_bias(-100.0);
        assert!(tape.bias() >= -MAX_BIAS - 1.0e-3);
    }

    #[test]
    fn hf_rolloff_attenuates_highs_more_than_lows() {
        // Compare the steady-state output energy of a low tone and a high tone;
        // the treble roll-off should pass the low tone far more strongly.
        fn energy(freq_hz: Sample) -> Sample {
            let mut p = params();
            // Isolate the filter: no mix delay offset ambiguity by using full wet
            // but comparing relative energies at the same depth.
            p.hf_rolloff_hz = 3_000.0;
            let mut node = TapeNode::new(p, SR, 1);
            let mut input = mono(4096);
            sine(&mut input, freq_hz, 0.5);
            let inputs = [input];
            let mut outputs = [mono(4096)];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(4096), &mut io);
            outputs[0].channel(0)[2048..].iter().map(|&s| s * s).sum::<Sample>()
        }
        assert!(
            energy(500.0) > energy(10_000.0) * 2.0,
            "treble roll-off did not attenuate the high tone"
        );
    }

    #[test]
    fn wow_flutter_moves_the_signal() {
        // With wow and flutter active, the full-wet output departs from a static
        // delayed copy of the input (pitch movement).
        let mut node = TapeNode::new(params(), SR, 1);
        let mut input = mono(4096);
        sine(&mut input, 440.0, 0.5);
        let inputs = [input.clone()];
        let mut outputs = [mono(4096)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(4096), &mut io);
        let mut moved = false;
        for i in 2048..4096 {
            if (outputs[0].channel(0)[i] - input.channel(0)[i]).abs() > 1.0e-3 {
                moved = true;
                break;
            }
        }
        assert!(moved, "tape did not modulate the signal");
    }

    #[test]
    fn deeper_wow_widens_deviation() {
        // Measure the peak-to-peak read-delay excursion produced by the wow
        // oscillator directly from the core: deeper wow must sweep the read
        // head across a strictly wider range of delays. A fast wow rate keeps
        // the capture window short while still covering a full cycle.
        fn deviation(wow_depth_ms: Sample) -> Sample {
            let mut p = params();
            p.wow_depth_ms = wow_depth_ms;
            p.flutter_depth_ms = 0.0;
            p.wow_hz = 50.0;
            let mut tape = Tape::new(p, SR, 1);
            let mut min_delay = Sample::INFINITY;
            let mut max_delay = Sample::NEG_INFINITY;
            // 48 kHz / 50 Hz = 960 frames per cycle; 2048 covers two full cycles.
            for _ in 0..2048 {
                let delay = tape.advance();
                min_delay = min_delay.min(delay);
                max_delay = max_delay.max(delay);
            }
            max_delay - min_delay
        }
        assert!(deviation(4.0) > deviation(1.0));
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = TapeNode::new(params(), SR, 1);
        let inputs = [mono(2048)];
        let mut outputs = [mono(2048)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(2048), &mut io);
        for &y in outputs[0].channel(0) {
            assert!(y.abs() < 1.0e-6, "silence produced output: {y}");
        }
    }

    #[test]
    fn non_finite_input_is_safe() {
        let mut node = TapeNode::new(params(), SR, 1);
        let mut input = mono(256);
        input.channel_mut(0)[0] = Sample::NAN;
        input.channel_mut(0)[1] = Sample::INFINITY;
        let inputs = [input];
        let mut outputs = [mono(256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        for &y in outputs[0].channel(0) {
            assert!(y.is_finite(), "non-finite leaked: {y}");
        }
    }

    #[test]
    fn dry_mix_is_transparent() {
        let mut p = params();
        p.mix = 0.0;
        let mut node = TapeNode::new(p, SR, 1);
        let mut input = mono(256);
        sine(&mut input, 1_000.0, 0.5);
        let inputs = [input.clone()];
        let mut outputs = [mono(256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        for i in 0..256 {
            assert!(
                (outputs[0].channel(0)[i] - input.channel(0)[i]).abs() < 1.0e-6,
                "dry mix altered the signal at {i}"
            );
        }
    }

    #[test]
    fn reset_reproduces_output() {
        let mut node = TapeNode::new(params(), SR, 1);
        let mut input = mono(1024);
        sine(&mut input, 330.0, 0.5);
        let inputs = [input];

        let mut out_a = [mono(1024)];
        let mut io_a = ProcessIo::new(&inputs, &mut out_a);
        node.process(&ctx(1024), &mut io_a);

        node.reset();
        let mut out_b = [mono(1024)];
        let mut io_b = ProcessIo::new(&inputs, &mut out_b);
        node.process(&ctx(1024), &mut io_b);

        assert_eq!(out_a[0].channel(0), out_b[0].channel(0));
    }

    #[test]
    fn channels_are_independent() {
        let mut tape = Tape::new(params(), SR, 2);
        assert_eq!(tape.channels(), 2);
        for i in 0..512 {
            let phase = core::f32::consts::TAU * 440.0 * (i as Sample) / (SR as Sample);
            let d = tape.advance();
            let _ = tape.voice(0, 0.5 * ops::sin(phase), d);
            let _ = tape.voice(1, 0.0, d);
            tape.commit_frame();
        }
        let d = tape.advance();
        let a = tape.voice(0, 0.8, d);
        let b = tape.voice(1, 0.0, d);
        assert!((a - b).abs() > 1.0e-6, "channels should hold independent state");
    }

    #[test]
    fn stereo_is_phase_coherent_for_identical_input() {
        // Identical channel inputs through shared oscillators yield identical
        // outputs.
        let mut node = TapeNode::new(params(), SR, 2);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 1024);
        input.set_active_frames(1024);
        for ch in 0..2 {
            for i in 0..1024 {
                let phase = core::f32::consts::TAU * 220.0 * (i as Sample) / (SR as Sample);
                input.channel_mut(ch)[i] = 0.5 * ops::sin(phase);
            }
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Stereo, 1024)];
        outputs[0].set_active_frames(1024);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(1024), &mut io);
        assert_eq!(outputs[0].channel(0), outputs[0].channel(1));
    }

    #[test]
    fn node_reset_clears_tail() {
        let mut node = TapeNode::new(params(), SR, 1);
        let mut input = mono(512);
        sine(&mut input, 440.0, 0.8);
        let inputs = [input];
        {
            let mut outputs = [mono(512)];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(512), &mut io);
        }
        node.reset();
        let silent = [mono(512)];
        let mut after = [mono(512)];
        let mut io = ProcessIo::new(&silent, &mut after);
        node.process(&ctx(512), &mut io);
        for &y in after[0].channel(0) {
            assert!(y.abs() < 1.0e-6, "tail not cleared: {y}");
        }
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let mut p = params();
        p.drive = 1.0e9;
        p.bias = 1.0e9;
        p.wow_depth_ms = 1.0e9;
        p.flutter_depth_ms = 1.0e9;
        p.hf_rolloff_hz = 1.0e9;
        p.mix = 1.0e9;
        let mut node = TapeNode::new(p, SR, 1);
        let mut input = mono(64);
        sine(&mut input, 500.0, 1.0);
        let inputs = [input];
        let mut outputs = [mono(64)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(64), &mut io);
        for &y in outputs[0].channel(0) {
            assert!(y.is_finite(), "extreme params leaked non-finite: {y}");
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = TapeNode::new(params(), SR, 1);
        let inputs = [mono(8)];
        let mut outputs = [mono(8)];
        outputs[0].set_active_frames(0);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }

    #[test]
    fn clamp_depth_ms_bounds_the_delay() {
        // Finite values pass through within range; out-of-range and non-finite
        // excursions are bounded so the ring allocation stays finite.
        assert_eq!(clamp_depth_ms(2.0), 2.0);
        assert_eq!(clamp_depth_ms(-5.0), 0.0);
        assert_eq!(clamp_depth_ms(1.0e9), MAX_DELAY_MS);
        assert_eq!(clamp_depth_ms(Sample::NAN), 0.0);
        assert_eq!(clamp_depth_ms(Sample::INFINITY), 0.0);
    }

    #[test]
    fn one_pole_coef_is_bounded() {
        for &fc in &[-10.0, 0.0, 1_000.0, 24_000.0, 1.0e9] {
            let a = one_pole_coef(fc, SR);
            assert!((0.0..=1.0).contains(&a), "coef out of range for {fc}: {a}");
        }
    }

    #[test]
    fn collects_into_vec_without_panic() {
        // Smoke test that the public API can be exercised in a Vec-collecting
        // harness (and exercises the alloc import under no_std + alloc).
        let mut tape = Tape::new(params(), SR, 1);
        let mut out: Vec<Sample> = Vec::with_capacity(256);
        for i in 0..256 {
            let phase = core::f32::consts::TAU * 600.0 * (i as Sample) / (SR as Sample);
            let d = tape.advance();
            out.push(tape.voice(0, 0.5 * ops::sin(phase), d));
            tape.commit_frame();
        }
        assert!(out.iter().all(|s| s.is_finite()));
    }
}
