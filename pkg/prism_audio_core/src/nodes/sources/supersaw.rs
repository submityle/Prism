//! Detuned sawtooth-stack ("super saw") source node.
//!
//! [`SupersawNode`] stacks seven band-limited sawtooth oscillators tuned around
//! a common fundamental and spread apart by a detune control, reproducing the
//! lush, chorused "super saw" lead/pad timbre made famous by the Roland JP-8000
//! and now a staple of electronic dance music. The thick sound comes from the
//! slow beating between the slightly mistuned partials of each saw.
//!
//! # Model
//!
//! Seven voices share one fundamental `f0`. Voice `i` is retuned by
//!
//! ```text
//! f_i = f0 * 2^(offset_i * detune * MAX_DETUNE_CENTS / 1200)
//! ```
//!
//! where `offset_i` is a fixed symmetric spread
//! `[-1, -2/3, -1/3, 0, 1/3, 2/3, 1]` (so the center voice is never retuned)
//! and `detune` in `[0, 1]` scales the maximum interval. Equal-temperament
//! ratios (`2^(cents/1200)`) keep the detuning musically symmetric in pitch
//! rather than in linear hertz.
//!
//! Each voice is a `PolyBLEP` band-limited sawtooth: the naive bipolar ramp
//! `2t - 1` minus a polynomial band-limited step at the wrap discontinuity,
//! which suppresses the aliasing that a raw ramp would fold back into the
//! audible band. The shared [`poly_blep`](super::oscillator::poly_blep)
//! primitive is reused from [`super::oscillator`] rather than reimplemented.
//!
//! A `mix` control balances the center voice against the six detuned side
//! voices: at `mix = 0` only the center saw sounds (a plain band-limited saw);
//! as `mix` rises the side voices fade in to thicken the chorus. The summed
//! voices are normalized by the sum of their gains so the output stays within
//! `[-1, 1]` (times the overall amplitude) regardless of `mix`, and the voices
//! start at evenly spread phases so their saw ramps are decorrelated from the
//! first sample.
//!
//! # Relationship
//!
//! Unlike the single-waveform [`super::oscillator::OscillatorNode`] and the
//! table-based [`super::wavetable_oscillator::WavetableOscillatorNode`], this
//! node is specifically a *detuned unison stack*: its character comes from the
//! inter-voice beating, not from one oscillator's shape. It reuses the
//! oscillator module's `PolyBLEP` step; it does not embed a filter (route it
//! through a separate highpass/biquad node if DC/rumble removal is desired).
//!
//! # Real-time contract
//!
//! All voice state lives in fixed-size arrays sized at compile time, so
//! [`SupersawNode::process`] performs no allocation, no locking, and no
//! panicking. `mix` and `amplitude` are [`Smoothed`] to avoid zipper noise;
//! `frequency` and `detune` are plain scalars because they only change the
//! continuous phase slope and therefore never introduce a discontinuity.
//!
//! # Provenance
//!
//! Implemented from first principles from the public super-saw literature --
//! A. Szabo, "How to Emulate the Super Saw," KTH Royal Institute of Technology,
//! 2010 (for the qualitative structure of a detuned seven-saw unison), combined
//! with the standard `PolyBLEP` band-limiting technique. The exact detune/mix
//! curves here are derived from first principles (an equal-temperament spread
//! and a gain-summed normalization), not copied from any proprietary fit. It
//! contains no source code or derived code from Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Google Resonance Audio, or Web Audio; only the
//! shared mathematical ideas are referenced.

use bevy_math::ops;

use super::oscillator::poly_blep;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

/// Number of stacked sawtooth voices (odd so one voice is the undetuned
/// center).
pub const SUPERSAW_VOICES: usize = 7;

/// Index of the center (never-retuned) voice within the stack.
const CENTER_VOICE: usize = SUPERSAW_VOICES / 2;

/// Symmetric per-voice detune spread. The center entry is exactly `0.0` so the
/// fundamental is always present; the outer voices reach `+/-1.0`.
const VOICE_OFFSETS: [Sample; SUPERSAW_VOICES] = [
    -1.0,
    -2.0 / 3.0,
    -1.0 / 3.0,
    0.0,
    1.0 / 3.0,
    2.0 / 3.0,
    1.0,
];

/// Default fundamental frequency in hertz.
pub const DEFAULT_SUPERSAW_FREQUENCY_HZ: Sample = 110.0;

/// Default detune amount (0 = unison, 1 = widest spread).
pub const DEFAULT_SUPERSAW_DETUNE: Sample = 0.5;

/// Default center/side mix (0 = center only, 1 = full chorus).
pub const DEFAULT_SUPERSAW_MIX: Sample = 0.5;

/// Default linear output amplitude.
pub const DEFAULT_SUPERSAW_AMPLITUDE: Sample = 1.0;

/// Maximum detune interval of the outermost voices, in cents, reached at
/// `detune = 1.0`. One semitone of spread gives a wide, lush stack.
pub const MAX_SUPERSAW_DETUNE_CENTS: Sample = 100.0;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Band-limited bipolar sawtooth for normalized phase `t` with per-sample
/// increment `dt`, reusing the oscillator module's `PolyBLEP` step.
#[inline]
fn band_limited_saw(t: Sample, dt: Sample) -> Sample {
    (2.0 * t - 1.0) - poly_blep(t, dt)
}

/// The initial phase of voice `i`: evenly spread across `[0, 1)` so the seven
/// saw ramps are decorrelated from the very first sample.
#[inline]
fn initial_phase(voice: usize) -> Sample {
    voice as Sample / SUPERSAW_VOICES as Sample
}

/// Construction parameters for a [`SupersawNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SupersawParams {
    /// Fundamental frequency in hertz. Clamped non-negative.
    pub frequency_hz: Sample,
    /// Detune amount in `[0, 1]`: 0 is unison, 1 is the widest spread.
    pub detune: Sample,
    /// Center/side mix in `[0, 1]`: 0 is center voice only, 1 is full chorus.
    pub mix: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for SupersawParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_SUPERSAW_FREQUENCY_HZ,
            detune: DEFAULT_SUPERSAW_DETUNE,
            mix: DEFAULT_SUPERSAW_MIX,
            amplitude: DEFAULT_SUPERSAW_AMPLITUDE,
        }
    }
}

/// A detuned seven-sawtooth "super saw" source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono stack so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::SupersawNode;
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = SupersawNode::new(110.0, 0.6, 0.7, 1.0);
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 512)];
/// outputs[0].set_active_frames(512);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 512, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // A detuned stack is not silent and stays bounded.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak <= 1.1);
/// ```
#[derive(Debug, Clone)]
pub struct SupersawNode {
    /// Fundamental frequency in hertz. Always non-negative.
    frequency: Sample,
    /// Detune amount in `[0, 1]`.
    detune: Sample,
    /// Smoothed center/side mix in `[0, 1]`.
    mix: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Per-voice normalized phase accumulators in `[0, 1)`.
    phases: [Sample; SUPERSAW_VOICES],
}

impl SupersawNode {
    /// Creates a super saw at `frequency_hz` with the given `detune` and `mix`
    /// (both in `[0, 1]`) and linear `amplitude`.
    ///
    /// All parameters are sanitized: non-finite values fall back to their
    /// defaults, frequency is clamped non-negative, and `detune`/`mix` are
    /// clamped to `[0, 1]`.
    #[must_use]
    pub fn new(frequency_hz: Sample, detune: Sample, mix: Sample, amplitude: Sample) -> Self {
        let mut phases = [0.0; SUPERSAW_VOICES];
        for (v, p) in phases.iter_mut().enumerate() {
            *p = initial_phase(v);
        }
        Self {
            frequency: finite_or(frequency_hz, DEFAULT_SUPERSAW_FREQUENCY_HZ).max(0.0),
            detune: finite_or(detune, DEFAULT_SUPERSAW_DETUNE).clamp(0.0, 1.0),
            mix: Smoothed::new(finite_or(mix, DEFAULT_SUPERSAW_MIX).clamp(0.0, 1.0)),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_SUPERSAW_AMPLITUDE)),
            phases,
        }
    }

    /// Builds a super saw from a [`SupersawParams`] bundle.
    #[must_use]
    pub fn from_params(params: SupersawParams) -> Self {
        Self::new(params.frequency_hz, params.detune, params.mix, params.amplitude)
    }

    /// Sets the fundamental frequency in hertz (clamped non-negative).
    ///
    /// Click-free without smoothing because the phase accumulators are
    /// continuous.
    #[inline]
    pub fn set_frequency(&mut self, hz: Sample) {
        self.frequency = finite_or(hz, self.frequency).max(0.0);
    }

    /// Sets the detune amount in `[0, 1]` (click-free, phase-continuous).
    #[inline]
    pub fn set_detune(&mut self, detune: Sample) {
        self.detune = finite_or(detune, self.detune).clamp(0.0, 1.0);
    }

    /// Sets a new target center/side mix in `[0, 1]`, gliding with `ramp`.
    #[inline]
    pub fn set_mix(&mut self, mix: Sample, ramp: Ramp) {
        let target = finite_or(mix, self.mix.target()).clamp(0.0, 1.0);
        self.mix.set_target(target, ramp);
    }

    /// Sets a new target amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude.set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the current fundamental frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency(&self) -> Sample {
        self.frequency
    }

    /// Returns the current detune amount in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn detune(&self) -> Sample {
        self.detune
    }

    /// Returns the center/side mix the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn mix(&self) -> Sample {
        self.mix.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Computes the seven per-voice phase increments for `sample_rate`, applying
    /// the equal-temperament detune spread to the fundamental.
    #[inline]
    fn voice_increments(&self, sample_rate: Sample) -> [Sample; SUPERSAW_VOICES] {
        let base_dt = self.frequency / sample_rate;
        let cents_scale = self.detune * MAX_SUPERSAW_DETUNE_CENTS;
        let mut dts = [0.0; SUPERSAW_VOICES];
        for (v, dt) in dts.iter_mut().enumerate() {
            let cents = VOICE_OFFSETS[v] * cents_scale;
            // Equal-temperament ratio 2^(cents / 1200); the center voice has
            // cents == 0 so its ratio is exactly 1.
            let ratio = ops::exp(cents / 1200.0 * core::f32::consts::LN_2);
            *dt = base_dt * ratio;
        }
        dts
    }

    /// Produces one output sample from the current voice phases and `dts`,
    /// advancing every phase accumulator.
    #[inline]
    fn render_sample(&mut self, dts: &[Sample; SUPERSAW_VOICES]) -> Sample {
        let mix = self.mix.next_sample();
        let amp = self.amplitude.next_sample();

        // Center voice at full level; side voices fade in with mix. Normalizing
        // by the sum of absolute gains guarantees |output| <= 1 before amp.
        let center_gain = 1.0 - 0.5 * mix;
        let side_gain = 0.5 * mix;
        let gain_sum = center_gain + (SUPERSAW_VOICES as Sample - 1.0) * side_gain;
        let norm = if gain_sum > 0.0 { 1.0 / gain_sum } else { 0.0 };

        let mut acc = 0.0;
        for (v, (phase, dt)) in self.phases.iter_mut().zip(dts.iter()).enumerate() {
            let g = if v == CENTER_VOICE { center_gain } else { side_gain };
            acc += g * band_limited_saw(*phase, *dt);

            *phase += *dt;
            if *phase >= 1.0 {
                *phase -= (*phase as u32) as Sample;
            }
        }

        acc * norm * amp
    }
}

impl AudioNode for SupersawNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sample_rate = ctx.sample_rate.max(1) as Sample;
        let dts = self.voice_increments(sample_rate);

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample(&dts);
            }
        }

        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        for (v, p) in self.phases.iter_mut().enumerate() {
            *p = initial_phase(v);
        }
        self.mix = Smoothed::new(self.mix.target());
        self.amplitude = Smoothed::new(self.amplitude.target());
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

    fn render(node: &mut SupersawNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        let [buf] = outputs;
        buf
    }

    fn peak(buf: &AudioBuffer) -> Sample {
        buf.channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()))
    }

    #[test]
    fn output_stays_bounded() {
        let sample_rate = 48_000;
        let frames = 4_096;
        let mut node = SupersawNode::new(110.0, 1.0, 1.0, 1.0);
        let out = render(&mut node, sample_rate, frames);
        // Sum-normalized, so pre-amp peak <= 1 plus a little PolyBLEP overshoot.
        assert!(peak(&out) <= 1.1, "peak={}", peak(&out));
    }

    #[test]
    fn mix_zero_is_single_center_saw() {
        let sample_rate = 48_000;
        let frames = 300;
        let freq = 220.0;
        let mut node = SupersawNode::new(freq, 0.7, 0.0, 1.0);
        let out = render(&mut node, sample_rate, frames);

        // At mix == 0 only the center voice sounds (gain 1, norm 1). Reproduce
        // it: center voice has offset 0 so dt == base_dt, phase starts at 3/7.
        let dt = freq / sample_rate as Sample;
        let mut phase = initial_phase(CENTER_VOICE);
        for &produced in out.channel(0) {
            let expected = band_limited_saw(phase, dt);
            assert!(
                (produced - expected).abs() < 1e-5,
                "produced={produced} expected={expected}"
            );
            phase += dt;
            if phase >= 1.0 {
                phase -= (phase as u32) as Sample;
            }
        }
    }

    #[test]
    fn detune_changes_output() {
        let sample_rate = 48_000;
        let frames = 2_048;
        let mut narrow = SupersawNode::new(110.0, 0.05, 1.0, 1.0);
        let mut wide = SupersawNode::new(110.0, 0.9, 1.0, 1.0);
        let a = render(&mut narrow, sample_rate, frames);
        let b = render(&mut wide, sample_rate, frames);
        let mut max_diff = 0.0f32;
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 0.05, "detune had no effect (max_diff={max_diff})");
    }

    #[test]
    fn zero_detune_is_periodic_at_fundamental() {
        // With detune 0 every voice runs at f0; the stack is periodic at f0.
        let sample_rate = 48_000;
        let freq = 480.0; // integer period in samples: 100
        let period = (sample_rate as Sample / freq) as usize; // 100
        let frames = period * 4;
        let mut node = SupersawNode::new(freq, 0.0, 1.0, 1.0);
        let out = render(&mut node, sample_rate, frames);
        let ch = out.channel(0);
        // Compare the 2nd period against the 3rd period (steady state).
        for i in 0..period {
            let a = ch[period + i];
            let b = ch[2 * period + i];
            assert!((a - b).abs() < 1e-4, "i={i} a={a} b={b}");
        }
    }

    #[test]
    fn mix_increases_side_contribution() {
        let sample_rate = 48_000;
        let frames = 2_048;
        let mut only_center = SupersawNode::new(110.0, 0.6, 0.0, 1.0);
        let mut full = SupersawNode::new(110.0, 0.6, 1.0, 1.0);
        let a = render(&mut only_center, sample_rate, frames);
        let b = render(&mut full, sample_rate, frames);
        let mut max_diff = 0.0f32;
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 0.05, "mix had no effect (max_diff={max_diff})");
    }

    #[test]
    fn output_scales_with_amplitude() {
        let sample_rate = 48_000;
        let frames = 512;
        let mut quiet = SupersawNode::new(110.0, 0.5, 0.5, 0.25);
        let mut loud = SupersawNode::new(110.0, 0.5, 0.5, 1.0);
        let q = render(&mut quiet, sample_rate, frames);
        let l = render(&mut loud, sample_rate, frames);
        for (a, b) in q.channel(0).iter().zip(l.channel(0)) {
            assert!((a - 0.25 * b).abs() < 1e-6, "a={a} b={b}");
        }
    }

    #[test]
    fn voices_start_decorrelated() {
        let node = SupersawNode::new(110.0, 0.5, 0.5, 1.0);
        for i in 0..SUPERSAW_VOICES {
            for j in (i + 1)..SUPERSAW_VOICES {
                assert!(
                    (node.phases[i] - node.phases[j]).abs() > 1e-6,
                    "voices {i} and {j} share a phase"
                );
            }
        }
    }

    #[test]
    fn reset_makes_output_reproducible() {
        let sample_rate = 48_000;
        let frames = 512;
        let mut node = SupersawNode::new(130.0, 0.6, 0.7, 0.9);
        let first = render(&mut node, sample_rate, frames);
        node.reset();
        let second = render(&mut node, sample_rate, frames);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn reset_restores_initial_phases() {
        let sample_rate = 48_000;
        let frames = 128;
        let mut node = SupersawNode::new(110.0, 0.5, 0.5, 1.0);
        let _ = render(&mut node, sample_rate, frames);
        node.reset();
        for v in 0..SUPERSAW_VOICES {
            assert_eq!(node.phases[v], initial_phase(v));
        }
    }

    #[test]
    fn negative_frequency_is_clamped() {
        let mut node = SupersawNode::new(-100.0, 0.5, 0.5, 1.0);
        assert_eq!(node.frequency(), 0.0);
        node.set_frequency(-5.0);
        assert_eq!(node.frequency(), 0.0);
    }

    #[test]
    fn detune_is_clamped() {
        let node = SupersawNode::new(110.0, 5.0, 0.5, 1.0);
        assert_eq!(node.detune(), 1.0);
        let node = SupersawNode::new(110.0, -5.0, 0.5, 1.0);
        assert_eq!(node.detune(), 0.0);
    }

    #[test]
    fn mix_is_clamped() {
        let node = SupersawNode::new(110.0, 0.5, 5.0, 1.0);
        assert_eq!(node.mix(), 1.0);
        let node = SupersawNode::new(110.0, 0.5, -5.0, 1.0);
        assert_eq!(node.mix(), 0.0);
    }

    #[test]
    fn non_finite_params_fall_back_to_defaults() {
        let node = SupersawNode::new(
            Sample::NAN,
            Sample::INFINITY,
            Sample::NEG_INFINITY,
            Sample::NAN,
        );
        assert_eq!(node.frequency(), DEFAULT_SUPERSAW_FREQUENCY_HZ);
        assert_eq!(node.detune(), DEFAULT_SUPERSAW_DETUNE);
        assert_eq!(node.mix(), DEFAULT_SUPERSAW_MIX);
        assert_eq!(node.amplitude(), DEFAULT_SUPERSAW_AMPLITUDE);
    }

    #[test]
    fn from_params_matches_new() {
        let params = SupersawParams {
            frequency_hz: 220.0,
            detune: 0.4,
            mix: 0.8,
            amplitude: 0.7,
        };
        let mut a = SupersawNode::from_params(params);
        let mut b = SupersawNode::new(220.0, 0.4, 0.8, 0.7);
        let oa = render(&mut a, 48_000, 256);
        let ob = render(&mut b, 48_000, 256);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn default_params_are_sane() {
        let p = SupersawParams::default();
        assert_eq!(p.frequency_hz, DEFAULT_SUPERSAW_FREQUENCY_HZ);
        assert_eq!(p.detune, DEFAULT_SUPERSAW_DETUNE);
        assert_eq!(p.mix, DEFAULT_SUPERSAW_MIX);
        assert_eq!(p.amplitude, DEFAULT_SUPERSAW_AMPLITUDE);
    }

    #[test]
    fn stereo_channels_are_identical() {
        let sample_rate = 48_000;
        let frames = 256;
        let mut node = SupersawNode::new(110.0, 0.6, 0.7, 1.0);
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        let [buf] = outputs;
        assert_eq!(buf.channel(0), buf.channel(1));
    }

    #[test]
    fn high_frequency_does_not_panic() {
        let sample_rate = 8_000;
        let frames = 256;
        let mut node = SupersawNode::new(20_000.0, 1.0, 1.0, 1.0);
        let out = render(&mut node, sample_rate, frames);
        for &s in out.channel(0) {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn mix_ramp_is_click_free() {
        let sample_rate = 48_000;
        let frames: usize = 1_024;
        let mut node = SupersawNode::new(110.0, 0.7, 0.0, 1.0);
        node.set_mix(1.0, Ramp::Linear { samples: frames as u32 });
        let out = render(&mut node, sample_rate, frames);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.1, "s={s}");
        }
    }

    #[test]
    fn zero_channel_output_is_noop() {
        let mut node = SupersawNode::new(110.0, 0.5, 0.5, 1.0);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 1);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(48_000, 0), &mut io);
        assert_eq!(node.phases[CENTER_VOICE], initial_phase(CENTER_VOICE));
    }

    #[test]
    fn output_is_roughly_dc_free() {
        let sample_rate = 48_000;
        let frames = 8_000;
        let mut node = SupersawNode::new(110.0, 0.6, 0.8, 1.0);
        let out = render(&mut node, sample_rate, frames);
        let mean: Sample = out.channel(0).iter().sum::<Sample>() / frames as Sample;
        assert!(mean.abs() < 0.05, "mean={mean}");
    }

    #[test]
    fn getters_report_targets() {
        let mut node = SupersawNode::new(110.0, 0.3, 0.4, 0.9);
        node.set_frequency(220.0);
        node.set_detune(0.8);
        node.set_mix(0.6, Ramp::Immediate);
        node.set_amplitude(0.5, Ramp::Immediate);
        assert_eq!(node.frequency(), 220.0);
        assert_eq!(node.detune(), 0.8);
        assert_eq!(node.mix(), 0.6);
        assert_eq!(node.amplitude(), 0.5);
    }

    #[test]
    fn not_silent_with_detune() {
        let sample_rate = 48_000;
        let frames = 1_024;
        let mut node = SupersawNode::new(110.0, 0.6, 0.8, 1.0);
        let out = render(&mut node, sample_rate, frames);
        assert!(peak(&out) > 0.05, "output too quiet: {}", peak(&out));
    }
}
