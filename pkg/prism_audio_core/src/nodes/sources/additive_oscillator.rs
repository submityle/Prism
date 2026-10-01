//! Real-time additive (Fourier-series) oscillator source node.
//!
//! [`AdditiveOscillatorNode`] synthesizes a tone by summing a bank of harmonic
//! sine partials whose amplitudes are controllable at audio rate. All partials
//! are phase-locked to a single normalized fundamental phase, so partial `k`
//! (one-based harmonic number) contributes `g_k * sin(2*pi*k*phase)`. This is a
//! direct, truncated Fourier series: any periodic timbre is approached by
//! choosing the partial-amplitude envelope, and because each `g_k` is a live
//! smoothed gain the spectrum can be morphed continuously while the note holds.
//!
//! # Model
//!
//! One phase accumulator advances at the fundamental rate `f0 / fs`. Harmonic
//! `k` is evaluated as `sin(2*pi*k*phase)`, which is numerically identical to a
//! partial running at `k*f0` but keeps every partial perfectly harmonically
//! locked with no per-partial phase drift. Any harmonic whose frequency
//! `k*f0` reaches or exceeds the Nyquist frequency `fs/2` is muted, so the
//! summed waveform never folds aliased energy back into the audible band as the
//! fundamental is swept upward.
//!
//! The partial gains are summed and the result is divided by the running sum of
//! their magnitudes whenever that sum exceeds one, which guarantees the mix
//! stays within `[-1, 1]` (times the master amplitude) without altering the
//! relative timbre of a normalized partial set. Per-partial gains and the
//! master amplitude are [`Smoothed`] so sweeping a harmonic in or out is
//! click-free.
//!
//! # Relationship
//!
//! Unlike [`super::oscillator::OscillatorNode`] (geometric `PolyBLEP` analog
//! waves) and [`super::wavetable_oscillator::WavetableOscillatorNode`] (which
//! *bakes* an additive spectrum into static mipmap tables once at build time),
//! this node sums its sines live every sample. That makes each harmonic
//! amplitude an audio-rate control: timbres can morph continuously in a way a
//! fixed table cannot without being rebuilt. It trades the wavetable's O(1)
//! lookup for an O(partials) inner loop in exchange for that expressiveness.
//!
//! # Real-time contract
//!
//! The partial bank is a fixed-size array sized at construction, so
//! [`AdditiveOscillatorNode::process`] performs no allocation, no locking, and
//! no panicking: it is a pure per-sample state machine.
//!
//! # Provenance
//!
//! Implemented from first principles against the classical additive-synthesis
//! and discrete-Fourier-series literature. It contains no code, data, or
//! derivative of Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google
//! Resonance Audio, the Web Audio API, or any other audio engine; only the
//! shared mathematical ideas are used.

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

/// Full turn in radians, the argument scale for the sine partials.
const TAU: Sample = core::f32::consts::TAU;

/// Maximum number of harmonic partials the bank can hold.
pub const ADDITIVE_MAX_PARTIALS: usize = 16;

/// Default fundamental frequency in hertz.
pub const DEFAULT_ADDITIVE_FREQUENCY_HZ: Sample = 110.0;

/// Default linear output amplitude.
pub const DEFAULT_ADDITIVE_AMPLITUDE: Sample = 1.0;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Construction parameters for an [`AdditiveOscillatorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AdditiveOscillatorParams {
    /// Fundamental frequency in hertz. Clamped non-negative.
    pub frequency_hz: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
    /// Per-harmonic linear amplitudes; index `k` is the `(k + 1)`-th harmonic.
    pub partials: [Sample; ADDITIVE_MAX_PARTIALS],
}

impl Default for AdditiveOscillatorParams {
    fn default() -> Self {
        let mut partials = [0.0; ADDITIVE_MAX_PARTIALS];
        // Default to a single pure fundamental so the node has a predictable,
        // unambiguous starting timbre.
        partials[0] = 1.0;
        Self {
            frequency_hz: DEFAULT_ADDITIVE_FREQUENCY_HZ,
            amplitude: DEFAULT_ADDITIVE_AMPLITUDE,
            partials,
        }
    }
}

/// A real-time additive (Fourier-series) oscillator source node (0 inputs,
/// 1 output).
///
/// Every output channel receives the same mono sum so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::AdditiveOscillatorNode;
/// use prism_audio_core::param::Ramp;
///
/// let mut node = AdditiveOscillatorNode::new(110.0, 1.0);
/// // Fade in the third harmonic to brighten the tone.
/// node.set_partial(2, 0.4, Ramp::Immediate);
/// assert_eq!(node.frequency(), 110.0);
/// ```
#[derive(Debug, Clone)]
pub struct AdditiveOscillatorNode {
    /// Fundamental frequency in hertz. Always non-negative. Stored as a plain
    /// scalar because the phase accumulator is continuous, so a frequency
    /// change is click-free without smoothing.
    frequency: Sample,
    /// Smoothed master linear amplitude.
    amplitude: Smoothed,
    /// Smoothed per-harmonic linear gains; index `k` drives harmonic `k + 1`.
    partials: [Smoothed; ADDITIVE_MAX_PARTIALS],
    /// Normalized fundamental phase accumulator in `[0, 1)`.
    phase: Sample,
}

impl AdditiveOscillatorNode {
    /// Creates an additive oscillator at `frequency_hz` with master
    /// `amplitude`, starting as a single pure fundamental.
    ///
    /// Non-finite inputs fall back to defaults; frequency is clamped
    /// non-negative.
    #[must_use]
    pub fn new(frequency_hz: Sample, amplitude: Sample) -> Self {
        let mut partials = core::array::from_fn(|_| Smoothed::new(0.0));
        partials[0] = Smoothed::new(1.0);
        Self {
            frequency: finite_or(frequency_hz, DEFAULT_ADDITIVE_FREQUENCY_HZ).max(0.0),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_ADDITIVE_AMPLITUDE)),
            partials,
            phase: 0.0,
        }
    }

    /// Builds an additive oscillator from an [`AdditiveOscillatorParams`]
    /// bundle.
    #[must_use]
    pub fn from_params(params: AdditiveOscillatorParams) -> Self {
        let partials = core::array::from_fn(|k| Smoothed::new(finite_or(params.partials[k], 0.0)));
        Self {
            frequency: finite_or(params.frequency_hz, DEFAULT_ADDITIVE_FREQUENCY_HZ).max(0.0),
            amplitude: Smoothed::new(finite_or(params.amplitude, DEFAULT_ADDITIVE_AMPLITUDE)),
            partials,
            phase: 0.0,
        }
    }

    /// Sets the fundamental frequency in hertz (clamped non-negative).
    ///
    /// Click-free without smoothing because the phase accumulator is
    /// continuous.
    #[inline]
    pub fn set_frequency(&mut self, hz: Sample) {
        self.frequency = finite_or(hz, self.frequency).max(0.0);
    }

    /// Sets a new target master amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude.set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Sets the target linear gain of harmonic `index + 1`, gliding with
    /// `ramp`. Out-of-range indices are ignored.
    #[inline]
    pub fn set_partial(&mut self, index: usize, gain: Sample, ramp: Ramp) {
        if let Some(partial) = self.partials.get_mut(index) {
            partial.set_target(finite_or(gain, partial.target()), ramp);
        }
    }

    /// Sets the target gains for the leading harmonics from `gains`, gliding
    /// with `ramp`. Harmonics beyond the supplied slice are set to zero; gains
    /// beyond [`ADDITIVE_MAX_PARTIALS`] are ignored.
    #[inline]
    pub fn set_partials(&mut self, gains: &[Sample], ramp: Ramp) {
        for (k, partial) in self.partials.iter_mut().enumerate() {
            let target = gains.get(k).copied().map_or(0.0, |g| finite_or(g, 0.0));
            partial.set_target(target, ramp);
        }
    }

    /// Returns the current fundamental frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency(&self) -> Sample {
        self.frequency
    }

    /// Returns the target master amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Returns the target gain of harmonic `index + 1`, or `0.0` if the index
    /// is out of range.
    #[inline]
    #[must_use]
    pub fn partial(&self, index: usize) -> Sample {
        self.partials.get(index).map_or(0.0, Smoothed::target)
    }

    /// Produces one output sample at `nyquist` (= `fs / 2`) and `phase_inc`
    /// (= `f0 / fs`), advancing the phase accumulator and every smoothed gain.
    #[inline]
    fn render_sample(&mut self, nyquist: Sample, phase_inc: Sample) -> Sample {
        let amp = self.amplitude.next_sample();
        let f0 = self.frequency;
        let phase = self.phase;

        let mut acc = 0.0;
        let mut gain_sum = 0.0;
        for (k, partial) in self.partials.iter_mut().enumerate() {
            let g = partial.next_sample();
            let harmonic = (k + 1) as Sample;
            // Mute any partial at or above Nyquist to avoid aliasing. Muted
            // partials are also excluded from the normalization sum so they do
            // not needlessly attenuate the audible ones.
            if harmonic * f0 < nyquist {
                gain_sum += g.abs();
                if g != 0.0 {
                    acc += g * ops::sin(TAU * harmonic * phase);
                }
            }
        }

        // Advance the shared fundamental phase and wrap into [0, 1).
        self.phase += phase_inc;
        if self.phase >= 1.0 {
            self.phase -= (self.phase as u32) as Sample;
        }

        // Divide by the gain sum only when it would otherwise clip, so a
        // normalized partial set keeps its intended level and timbre.
        let norm = if gain_sum > 1.0 { 1.0 / gain_sum } else { 1.0 };
        acc * norm * amp
    }
}

impl AudioNode for AdditiveOscillatorNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sample_rate = ctx.sample_rate.max(1) as Sample;
        let nyquist = sample_rate * 0.5;
        let phase_inc = self.frequency / sample_rate;

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample(nyquist, phase_inc);
            }
        }

        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.amplitude = Smoothed::new(self.amplitude.target());
        for partial in &mut self.partials {
            *partial = Smoothed::new(partial.target());
        }
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    fn render(node: &mut AdditiveOscillatorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    fn render_stereo(
        node: &mut AdditiveOscillatorNode,
        sample_rate: u32,
        frames: usize,
    ) -> AudioBuffer {
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    #[test]
    fn output_stays_bounded() {
        let mut node = AdditiveOscillatorNode::new(220.0, 1.0);
        node.set_partials(&[1.0, 0.8, 0.6, 0.4, 0.2], Ramp::Immediate);
        let out = render(&mut node, 48_000, 2_048);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-4, "s={s}");
        }
    }

    #[test]
    fn single_fundamental_is_pure_sine() {
        let freq = 100.0;
        let sr = 48_000;
        let mut node = AdditiveOscillatorNode::new(freq, 1.0);
        let out = render(&mut node, sr, 512);
        let inc = freq / sr as Sample;
        for (n, &s) in out.channel(0).iter().enumerate() {
            let expected = ops::sin(TAU * (n as Sample) * inc);
            assert!((s - expected).abs() < 1e-4, "n={n} got={s} want={expected}");
        }
    }

    #[test]
    fn adding_a_harmonic_changes_output() {
        let sr = 48_000;
        let frames = 512;
        let mut plain = AdditiveOscillatorNode::new(110.0, 1.0);
        let mut rich = AdditiveOscillatorNode::new(110.0, 1.0);
        rich.set_partial(2, 0.5, Ramp::Immediate);
        let a = render(&mut plain, sr, frames);
        let b = render(&mut rich, sr, frames);
        let diff: Sample = a
            .channel(0)
            .iter()
            .zip(b.channel(0))
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(diff > 1e-2, "adding a harmonic should change the signal, diff={diff}");
    }

    #[test]
    fn amplitude_scales_output() {
        let sr = 48_000;
        let frames = 512;
        let mut loud = AdditiveOscillatorNode::new(220.0, 1.0);
        let mut quiet = AdditiveOscillatorNode::new(220.0, 0.25);
        let a = render(&mut loud, sr, frames);
        let b = render(&mut quiet, sr, frames);
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            assert!((x * 0.25 - y).abs() < 1e-4, "a*0.25={} b={y}", x * 0.25);
        }
    }

    #[test]
    fn fundamental_is_periodic() {
        let sr = 48_000;
        let freq = sr as Sample / 128.0; // integer period of 128 samples
        let mut node = AdditiveOscillatorNode::new(freq, 1.0);
        node.set_partial(1, 0.5, Ramp::Immediate);
        let out = render(&mut node, sr, 384);
        let ch = out.channel(0);
        for n in 0..128 {
            assert!((ch[n] - ch[n + 128]).abs() < 1e-3, "n={n}");
        }
    }

    #[test]
    fn nyquist_partials_are_muted() {
        // At f0 = 12 kHz with fs = 48 kHz, Nyquist is 24 kHz: only the
        // fundamental (12 kHz) is below Nyquist; the 2nd harmonic (24 kHz) and
        // above must be muted, so output equals the pure fundamental sine.
        let sr = 48_000;
        let freq = 12_000.0;
        let mut node = AdditiveOscillatorNode::new(freq, 1.0);
        node.set_partials(&[1.0, 1.0, 1.0, 1.0], Ramp::Immediate);
        let out = render(&mut node, sr, 256);
        let inc = freq / sr as Sample;
        for (n, &s) in out.channel(0).iter().enumerate() {
            let expected = ops::sin(TAU * (n as Sample) * inc);
            assert!((s - expected).abs() < 1e-4, "n={n} got={s} want={expected}");
        }
    }

    #[test]
    fn normalization_keeps_many_partials_bounded() {
        let mut node = AdditiveOscillatorNode::new(55.0, 1.0);
        node.set_partials(&[1.0; ADDITIVE_MAX_PARTIALS], Ramp::Immediate);
        let out = render(&mut node, 48_000, 4_096);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-4, "s={s}");
        }
    }

    #[test]
    fn reset_makes_output_reproducible() {
        let mut node = AdditiveOscillatorNode::new(130.0, 1.0);
        node.set_partials(&[1.0, 0.5, 0.25], Ramp::Immediate);
        let a = render(&mut node, 48_000, 300);
        node.reset();
        let b = render(&mut node, 48_000, 300);
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn reset_restores_phase() {
        let mut node = AdditiveOscillatorNode::new(130.0, 1.0);
        let _ = render(&mut node, 48_000, 97);
        node.reset();
        assert_eq!(node.phase, 0.0);
    }

    #[test]
    fn negative_frequency_is_clamped() {
        let node = AdditiveOscillatorNode::new(-440.0, 1.0);
        assert_eq!(node.frequency(), 0.0);
    }

    #[test]
    fn out_of_range_partial_is_ignored() {
        let mut node = AdditiveOscillatorNode::new(110.0, 1.0);
        node.set_partial(ADDITIVE_MAX_PARTIALS + 4, 1.0, Ramp::Immediate);
        assert_eq!(node.partial(ADDITIVE_MAX_PARTIALS + 4), 0.0);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = AdditiveOscillatorNode::new(Sample::NAN, Sample::INFINITY);
        assert_eq!(node.frequency(), DEFAULT_ADDITIVE_FREQUENCY_HZ);
        assert_eq!(node.amplitude(), DEFAULT_ADDITIVE_AMPLITUDE);
    }

    #[test]
    fn from_params_matches_new() {
        let params = AdditiveOscillatorParams {
            frequency_hz: 123.0,
            amplitude: 0.7,
            ..AdditiveOscillatorParams::default()
        };
        let mut a = AdditiveOscillatorNode::from_params(params);
        let mut b = AdditiveOscillatorNode::new(123.0, 0.7);
        let ra = render(&mut a, 48_000, 256);
        let rb = render(&mut b, 48_000, 256);
        for (x, y) in ra.channel(0).iter().zip(rb.channel(0)) {
            assert!((x - y).abs() < 1e-6, "x={x} y={y}");
        }
    }

    #[test]
    fn default_params_are_a_pure_fundamental() {
        let params = AdditiveOscillatorParams::default();
        assert_eq!(params.frequency_hz, DEFAULT_ADDITIVE_FREQUENCY_HZ);
        assert_eq!(params.partials[0], 1.0);
        for &g in &params.partials[1..] {
            assert_eq!(g, 0.0);
        }
    }

    #[test]
    fn stereo_channels_are_identical() {
        let mut node = AdditiveOscillatorNode::new(110.0, 1.0);
        node.set_partials(&[1.0, 0.5, 0.3], Ramp::Immediate);
        let out = render_stereo(&mut node, 48_000, 256);
        let (left, right) = (out.channel(0), out.channel(1));
        assert_eq!(left, right);
    }

    #[test]
    fn super_nyquist_fundamental_does_not_panic() {
        let mut node = AdditiveOscillatorNode::new(40_000.0, 1.0);
        node.set_partials(&[1.0, 1.0, 1.0], Ramp::Immediate);
        let out = render(&mut node, 48_000, 128);
        for &s in out.channel(0) {
            assert!(s.is_finite(), "s={s}");
        }
    }

    #[test]
    fn partial_ramp_is_click_free() {
        let sr = 48_000;
        let frames: usize = 1_024;
        let mut node = AdditiveOscillatorNode::new(110.0, 1.0);
        node.set_partial(3, 0.8, Ramp::Linear { samples: frames as u32 });
        let out = render(&mut node, sr, frames);
        let ch = out.channel(0);
        for w in ch.windows(2) {
            assert!((w[1] - w[0]).abs() < 0.5, "step {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn zero_channel_output_is_noop() {
        let mut node = AdditiveOscillatorNode::new(110.0, 1.0);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(48_000, 0), &mut io);
        assert_eq!(node.phase, 0.0);
    }

    #[test]
    fn getters_report_targets() {
        let mut node = AdditiveOscillatorNode::new(110.0, 0.5);
        node.set_frequency(321.0);
        node.set_amplitude(0.9, Ramp::Immediate);
        node.set_partial(4, 0.42, Ramp::Immediate);
        assert_eq!(node.frequency(), 321.0);
        assert_eq!(node.amplitude(), 0.9);
        assert_eq!(node.partial(4), 0.42);
    }

    #[test]
    fn set_partials_fills_and_clears() {
        let mut node = AdditiveOscillatorNode::new(110.0, 1.0);
        node.set_partials(&[0.1, 0.2, 0.3], Ramp::Immediate);
        assert_eq!(node.partial(0), 0.1);
        assert_eq!(node.partial(1), 0.2);
        assert_eq!(node.partial(2), 0.3);
        // Harmonics past the supplied slice are cleared, including the default
        // fundamental if the slice did not cover it.
        assert_eq!(node.partial(3), 0.0);
    }

    #[test]
    fn not_silent_with_partials() {
        let mut node = AdditiveOscillatorNode::new(110.0, 1.0);
        node.set_partials(&[1.0, 0.5, 0.25], Ramp::Immediate);
        let out = render(&mut node, 48_000, 512);
        let energy: Sample = out.channel(0).iter().map(|s| s * s).sum();
        assert!(energy > 1e-3, "energy={energy}");
    }

    #[test]
    fn output_is_roughly_dc_free() {
        let mut node = AdditiveOscillatorNode::new(100.0, 1.0);
        node.set_partials(&[1.0, 0.5, 0.3, 0.2], Ramp::Immediate);
        let frames = 480; // whole number of 100 Hz cycles at 48 kHz
        let out = render(&mut node, 48_000, frames);
        let mean: Sample = out.channel(0).iter().sum::<Sample>() / frames as Sample;
        assert!(mean.abs() < 1e-3, "mean={mean}");
    }
}
