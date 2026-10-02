//! Analog-style synthesized tom-tom voice source node.
//!
//! [`AnalogTomNode`] synthesizes the classic analog tom voice: a pair of tuned
//! sine partials -- a fundamental and a membrane overtone -- whose pitch sweeps
//! gently down from the attack toward a sustained musical fundamental, shaped
//! by an exponential amplitude decay and pushed through a `tanh` saturator. The
//! short upward pitch sweep gives the characteristic "boing" of a struck drum
//! head while the overtone supplies the hollow membrane colour. Unlike the
//! modal physical drums ([`membrane_drum`](super::membrane_drum),
//! [`struck_bar`](super::struck_bar)) which excite a bank of resonant modes,
//! this voice is a direct subtractive two-partial oscillator, exactly as
//! realized by the twin-sine topology of a classic analog drum machine. It is a
//! *source* (zero inputs, one output): it supplies its own excitation through
//! [`AnalogTomNode::trigger`] and never reads its inputs.
//!
//! # Model
//!
//! Two phase accumulators run sine partials. The fundamental tracks an
//! instantaneous frequency that starts `pitch_env_hz` above `tune_hz` and
//! relaxes exponentially toward `tune_hz`, and the overtone tracks that same
//! swept frequency scaled by `overtone_ratio` (a membrane-like inharmonic
//! factor). Their weighted sum is normalized so it never leaves `[-1, 1]`:
//!
//! ```text
//!   f(t)    = tune_hz + pitch_env_hz * exp(-t / pitch_tau)
//!   body(t) = (sin(2*pi*f(t)*t) + overtone_level * sin(2*pi*ratio*f(t)*t))
//!             / (1 + overtone_level)
//! ```
//!
//! The body is scaled by an exponential VCA envelope with a `-60 dB` time of
//! `amp_decay_s`:
//!
//! ```text
//!   voice(t) = body(t) * exp(-t / amp_tau)
//! ```
//!
//! Finally the voice is driven through a `tanh` saturator so that increasing
//! `drive` fattens the harmonics without ever exceeding full scale:
//!
//! ```text
//!   shaped = tanh(drive * voice) / tanh(drive)
//!   out    = amplitude * shaped
//! ```
//!
//! Because the saturator sits *before* the final `amplitude` gain, the output
//! level is a strict square law in `amplitude` while the `tanh` guarantees the
//! voice can never clip past `|amplitude|`.
//!
//! # Real-time contract
//!
//! Construction, [`AnalogTomNode::trigger`], and every scalar setter
//! pre-compute all per-voice coefficients (the amplitude and pitch decay
//! coefficients and the overtone normalization), so [`AnalogTomNode::process`]
//! performs no allocation, no locking, and cannot panic: it is a pure
//! per-sample state machine. The `drive` and `amplitude` controls are driven
//! through [`Smoothed`] values so that automation never introduces zipper
//! noise, and a trigger is latched to a block boundary by the caller.
//!
//! # Provenance
//!
//! Independent clean-room implementation of textbook subtractive drum
//! synthesis. It borrows only public-domain *ideas*: the twin-sine
//! pitch-swept drum-voice topology popularised by analog drum machines, the
//! classic `-60 dB`-time exponential envelope, and the `tanh` soft-clip. No
//! source code or derived code from any audio engine or toolkit (UE, Unity,
//! Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, Web Audio, or STK)
//! was consulted or reused.
//!
//! # Relationship
//!
//! Shares the Prism source-node conventions (`Sample`, [`Smoothed`], [`Ramp`],
//! the mono-core render loop) with the other analog drum voices. Unlike
//! [`AnalogKickNode`](super::analog_kick::AnalogKickNode) -- a single sine with
//! a deep pitch sweep and a transient click that never settles on a musical
//! pitch -- this voice adds a tuned membrane overtone and uses a gentler sweep
//! that resolves to a sustained fundamental, and it carries no click. Unlike
//! [`AnalogSnareNode`](super::analog_snare::AnalogSnareNode) it has no noise
//! layer at all and is therefore a purely tonal, fully deterministic voice.
//! Unlike the modal physical drums it is a direct two-partial subtractive
//! voice rather than a bank of excited resonant modes.

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Lowest tunable fundamental in hertz (deep floor tom).
pub const MIN_TUNE_HZ: Sample = 50.0;

/// Highest tunable fundamental in hertz (high rack tom).
pub const MAX_TUNE_HZ: Sample = 400.0;

/// Default fundamental in hertz.
pub const DEFAULT_TUNE_HZ: Sample = 160.0;

/// Smallest initial pitch-sweep depth above the fundamental, in hertz.
pub const MIN_PITCH_ENV_HZ: Sample = 0.0;

/// Largest initial pitch-sweep depth above the fundamental, in hertz.
pub const MAX_PITCH_ENV_HZ: Sample = 2_000.0;

/// Default initial pitch-sweep depth above the fundamental, in hertz.
pub const DEFAULT_PITCH_ENV_HZ: Sample = 140.0;

/// Shortest pitch-sweep time constant in seconds.
pub const MIN_PITCH_DECAY_S: Sample = 0.002;

/// Longest pitch-sweep time constant in seconds.
pub const MAX_PITCH_DECAY_S: Sample = 0.5;

/// Default pitch-sweep time constant in seconds.
pub const DEFAULT_PITCH_DECAY_S: Sample = 0.05;

/// Shortest `-60 dB` amplitude decay time in seconds.
pub const MIN_AMP_DECAY_S: Sample = 0.05;

/// Longest `-60 dB` amplitude decay time in seconds.
pub const MAX_AMP_DECAY_S: Sample = 4.0;

/// Default `-60 dB` amplitude decay time in seconds.
pub const DEFAULT_AMP_DECAY_S: Sample = 0.5;

/// Smallest membrane overtone ratio relative to the fundamental.
pub const MIN_OVERTONE_RATIO: Sample = 1.1;

/// Largest membrane overtone ratio relative to the fundamental.
pub const MAX_OVERTONE_RATIO: Sample = 3.0;

/// Default membrane overtone ratio (near the second circular-membrane mode).
pub const DEFAULT_OVERTONE_RATIO: Sample = 1.5;

/// Smallest overtone level (fundamental only).
pub const MIN_OVERTONE_LEVEL: Sample = 0.0;

/// Largest overtone level (overtone as loud as the fundamental).
pub const MAX_OVERTONE_LEVEL: Sample = 1.0;

/// Default overtone level.
pub const DEFAULT_OVERTONE_LEVEL: Sample = 0.5;

/// Smallest saturator drive (`1.0` is nearly linear).
pub const MIN_DRIVE: Sample = 1.0;

/// Largest saturator drive (heavy harmonic fattening).
pub const MAX_DRIVE: Sample = 12.0;

/// Default saturator drive.
pub const DEFAULT_DRIVE: Sample = 1.5;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.85;

/// Default strike velocity used by [`AnalogTomNode::trigger`].
pub const DEFAULT_VELOCITY: Sample = 1.0;

/// `ln(1000) == 3 * ln(10)`, used by the `-60 dB`-time-to-decay mapping.
const LN_1000: Sample = 6.907_755;

/// Fraction of the sample rate above which an oscillator is muted (anti-alias).
const NYQUIST_GUARD: Sample = 0.49;

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Clamps a fundamental to the tunable range and the Nyquist guard.
#[inline]
fn clamp_tune(freq_hz: Sample, sample_rate: u32) -> Sample {
    let nyquist = sample_rate.max(1) as Sample * NYQUIST_GUARD;
    let upper = MAX_TUNE_HZ.min(nyquist).max(MIN_TUNE_HZ);
    freq_hz.clamp(MIN_TUNE_HZ, upper)
}

/// Construction parameters for an [`AnalogTomNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AnalogTomParams {
    /// Sustained body fundamental frequency in hertz.
    pub tune_hz: Sample,
    /// Initial pitch-sweep depth above the fundamental, in hertz.
    pub pitch_env_hz: Sample,
    /// Pitch-sweep time constant in seconds.
    pub pitch_decay_s: Sample,
    /// Amplitude `-60 dB` decay time in seconds.
    pub amp_decay_s: Sample,
    /// Membrane overtone ratio relative to the fundamental.
    pub overtone_ratio: Sample,
    /// Overtone level relative to the fundamental, in `[0, 1]`.
    pub overtone_level: Sample,
    /// Saturator drive (`1.0` nearly linear, higher adds harmonics).
    pub drive: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for AnalogTomParams {
    fn default() -> Self {
        Self {
            tune_hz: DEFAULT_TUNE_HZ,
            pitch_env_hz: DEFAULT_PITCH_ENV_HZ,
            pitch_decay_s: DEFAULT_PITCH_DECAY_S,
            amp_decay_s: DEFAULT_AMP_DECAY_S,
            overtone_ratio: DEFAULT_OVERTONE_RATIO,
            overtone_level: DEFAULT_OVERTONE_LEVEL,
            drive: DEFAULT_DRIVE,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl AnalogTomParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range. The fundamental is clamped against the Nyquist limit too.
    #[must_use]
    pub fn sanitised(self, sample_rate: u32) -> Self {
        let d = Self::default();
        let tune_hz = clamp_tune(finite_or(self.tune_hz, d.tune_hz), sample_rate);
        let pitch_env_hz = finite_or(self.pitch_env_hz, d.pitch_env_hz)
            .clamp(MIN_PITCH_ENV_HZ, MAX_PITCH_ENV_HZ);
        let pitch_decay_s = finite_or(self.pitch_decay_s, d.pitch_decay_s)
            .clamp(MIN_PITCH_DECAY_S, MAX_PITCH_DECAY_S);
        let amp_decay_s =
            finite_or(self.amp_decay_s, d.amp_decay_s).clamp(MIN_AMP_DECAY_S, MAX_AMP_DECAY_S);
        let overtone_ratio = finite_or(self.overtone_ratio, d.overtone_ratio)
            .clamp(MIN_OVERTONE_RATIO, MAX_OVERTONE_RATIO);
        let overtone_level = finite_or(self.overtone_level, d.overtone_level)
            .clamp(MIN_OVERTONE_LEVEL, MAX_OVERTONE_LEVEL);
        let drive = finite_or(self.drive, d.drive).clamp(MIN_DRIVE, MAX_DRIVE);
        let amplitude = finite_or(self.amplitude, d.amplitude);
        Self {
            tune_hz,
            pitch_env_hz,
            pitch_decay_s,
            amp_decay_s,
            overtone_ratio,
            overtone_level,
            drive,
            amplitude,
        }
    }
}

/// Analog-style synthesized tom-tom voice source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::{AnalogTomNode, AnalogTomParams};
///
/// let mut node = AnalogTomNode::new(48_000, AnalogTomParams::default());
/// node.trigger(1.0);
/// // The trigger injects a fresh enveloped voice, so output is non-silent.
/// ```
#[derive(Clone, Debug)]
pub struct AnalogTomNode {
    sample_rate: u32,
    tune_hz: Sample,
    pitch_env_depth_hz: Sample,
    pitch_decay_s: Sample,
    amp_decay_s: Sample,
    overtone_ratio: Sample,
    overtone_level: Sample,
    drive: Smoothed,
    amplitude: Smoothed,
    fund_phase: Sample,
    over_phase: Sample,
    amp_env: Sample,
    pitch_env: Sample,
    amp_decay_coeff: Sample,
    pitch_decay_coeff: Sample,
    over_norm: Sample,
    velocity: Sample,
}

impl AnalogTomNode {
    /// Builds an analog tom voice for `sample_rate` Hz from `params`. The voice
    /// is triggered once at [`DEFAULT_VELOCITY`] so a freshly built node renders
    /// audible output immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: AnalogTomParams) -> Self {
        let p = params.sanitised(sample_rate);
        let mut node = Self {
            sample_rate: sample_rate.max(1),
            tune_hz: p.tune_hz,
            pitch_env_depth_hz: p.pitch_env_hz,
            pitch_decay_s: p.pitch_decay_s,
            amp_decay_s: p.amp_decay_s,
            overtone_ratio: p.overtone_ratio,
            overtone_level: p.overtone_level,
            drive: Smoothed::new(p.drive),
            amplitude: Smoothed::new(p.amplitude),
            fund_phase: 0.0,
            over_phase: 0.0,
            amp_env: 0.0,
            pitch_env: 0.0,
            amp_decay_coeff: 0.0,
            pitch_decay_coeff: 0.0,
            over_norm: 1.0,
            velocity: DEFAULT_VELOCITY,
        };
        node.recompute();
        node.trigger(DEFAULT_VELOCITY);
        node
    }

    /// Builds a node from [`AnalogTomParams`] (an alias of [`AnalogTomNode::new`]).
    #[must_use]
    pub fn from_params(sample_rate: u32, params: AnalogTomParams) -> Self {
        Self::new(sample_rate, params)
    }

    /// Returns the sustained fundamental in hertz.
    #[must_use]
    pub fn tune_hz(&self) -> Sample {
        self.tune_hz
    }

    /// Returns the initial pitch-sweep depth above the fundamental, in hertz.
    #[must_use]
    pub fn pitch_env_hz(&self) -> Sample {
        self.pitch_env_depth_hz
    }

    /// Returns the pitch-sweep time constant in seconds.
    #[must_use]
    pub fn pitch_decay_s(&self) -> Sample {
        self.pitch_decay_s
    }

    /// Returns the amplitude `-60 dB` decay time in seconds.
    #[must_use]
    pub fn amp_decay_s(&self) -> Sample {
        self.amp_decay_s
    }

    /// Returns the membrane overtone ratio.
    #[must_use]
    pub fn overtone_ratio(&self) -> Sample {
        self.overtone_ratio
    }

    /// Returns the overtone level.
    #[must_use]
    pub fn overtone_level(&self) -> Sample {
        self.overtone_level
    }

    /// Returns the saturator drive.
    #[must_use]
    pub fn drive(&self) -> Sample {
        self.drive.target()
    }

    /// Returns the target output amplitude.
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Sets the sustained fundamental, clamped to the tunable range. Takes
    /// effect from the next sample; it only shifts the sweep target and is
    /// click-free.
    pub fn set_tune_hz(&mut self, tune_hz: Sample) {
        self.tune_hz = clamp_tune(finite_or(tune_hz, self.tune_hz), self.sample_rate);
    }

    /// Sets the initial pitch-sweep depth, clamped to its range.
    pub fn set_pitch_env_hz(&mut self, pitch_env_hz: Sample) {
        self.pitch_env_depth_hz = finite_or(pitch_env_hz, self.pitch_env_depth_hz)
            .clamp(MIN_PITCH_ENV_HZ, MAX_PITCH_ENV_HZ);
    }

    /// Sets the pitch-sweep time constant, clamped to its range.
    pub fn set_pitch_decay_s(&mut self, pitch_decay_s: Sample) {
        self.pitch_decay_s = finite_or(pitch_decay_s, self.pitch_decay_s)
            .clamp(MIN_PITCH_DECAY_S, MAX_PITCH_DECAY_S);
        self.recompute();
    }

    /// Sets the amplitude `-60 dB` decay time, clamped to its range.
    pub fn set_amp_decay_s(&mut self, amp_decay_s: Sample) {
        self.amp_decay_s =
            finite_or(amp_decay_s, self.amp_decay_s).clamp(MIN_AMP_DECAY_S, MAX_AMP_DECAY_S);
        self.recompute();
    }

    /// Sets the membrane overtone ratio, clamped to its range. Takes effect
    /// from the next sample and is click-free.
    pub fn set_overtone_ratio(&mut self, overtone_ratio: Sample) {
        self.overtone_ratio = finite_or(overtone_ratio, self.overtone_ratio)
            .clamp(MIN_OVERTONE_RATIO, MAX_OVERTONE_RATIO);
    }

    /// Sets the overtone level, clamped to `[0, 1]`.
    pub fn set_overtone_level(&mut self, overtone_level: Sample) {
        self.overtone_level = finite_or(overtone_level, self.overtone_level)
            .clamp(MIN_OVERTONE_LEVEL, MAX_OVERTONE_LEVEL);
        self.recompute();
    }

    /// Sets the saturator drive, gliding over `ramp`.
    pub fn set_drive(&mut self, drive: Sample, ramp: Ramp) {
        let target = finite_or(drive, self.drive.target()).clamp(MIN_DRIVE, MAX_DRIVE);
        self.drive.set_target(target, ramp);
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Retriggers the voice with the given `velocity` (clamped to `[0, 1]`),
    /// restarting the pitch and amplitude envelopes and both oscillator phases.
    pub fn trigger(&mut self, velocity: Sample) {
        self.velocity = finite_or(velocity, DEFAULT_VELOCITY).clamp(0.0, 1.0);
        self.fund_phase = 0.0;
        self.over_phase = 0.0;
        self.amp_env = self.velocity;
        self.pitch_env = self.pitch_env_depth_hz;
    }

    /// Recomputes the per-sample decay coefficients and the overtone
    /// normalization from the current scalar parameters. Never runs on the
    /// audio hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        self.amp_decay_coeff = ops::exp(-LN_1000 / (self.amp_decay_s * sr));
        self.pitch_decay_coeff = ops::exp(-1.0 / (self.pitch_decay_s * sr));
        self.over_norm = 1.0 / (1.0 + self.overtone_level);
    }

    /// Renders one mono output sample, advancing the two oscillators and both
    /// envelopes by one step.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let sr = self.sample_rate.max(1) as Sample;
        let nyquist = sr * NYQUIST_GUARD;

        // Swept instantaneous fundamental and its membrane overtone.
        let inst = (self.tune_hz + self.pitch_env).clamp(0.0, nyquist);
        let fund = ops::sin(self.fund_phase);
        self.fund_phase += TAU * inst / sr;
        if self.fund_phase >= TAU {
            self.fund_phase -= TAU;
        }

        let over_freq = (inst * self.overtone_ratio).min(nyquist);
        let over = ops::sin(self.over_phase);
        self.over_phase += TAU * over_freq / sr;
        if self.over_phase >= TAU {
            self.over_phase -= TAU;
        }

        // Normalized two-partial body stays within [-1, 1] by construction.
        let body = (fund + self.overtone_level * over) * self.over_norm;
        let excitation = body * self.amp_env;

        // Advance the envelopes for the next sample.
        self.amp_env = flush_denormal(self.amp_env * self.amp_decay_coeff);
        self.pitch_env = flush_denormal(self.pitch_env * self.pitch_decay_coeff);

        // Saturate before the output gain so level is a strict square law in
        // `amplitude` and the voice can never exceed full scale.
        let drive = self.drive.next_sample();
        let shaped = ops::tanh(drive * excitation) / ops::tanh(drive);
        let amp = self.amplitude.next_sample();
        flush_denormal(amp * shaped)
    }
}

impl AudioNode for AnalogTomNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }
        let frames = io.output(0).active_frames();
        if frames == 0 {
            return;
        }

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample();
            }
        }
        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.drive = Smoothed::new(self.drive.target());
        self.amplitude = Smoothed::new(self.amplitude.target());
        self.recompute();
        self.trigger(DEFAULT_VELOCITY);
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use alloc::vec::Vec;

    const SR: u32 = 48_000;

    fn node_default() -> AnalogTomNode {
        AnalogTomNode::new(SR, AnalogTomParams::default())
    }

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut AnalogTomNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(
        node: &mut AnalogTomNode,
        frames: usize,
        layout: ChannelLayout,
    ) -> Vec<Vec<Sample>> {
        let inputs: [AudioBuffer; 0] = [];
        let mut out = AudioBuffer::new(layout, frames.max(1));
        out.set_active_frames(frames);
        let mut outputs = [out];
        let ctx = RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        let channels = outputs[0].channels();
        (0..channels)
            .map(|ch| outputs[0].channel(ch).to_vec())
            .collect()
    }

    fn peak(block: &[Sample]) -> Sample {
        block.iter().fold(0.0, |m, &s| m.max(s.abs()))
    }

    fn energy(block: &[Sample]) -> f64 {
        block.iter().map(|&s| (s as f64) * (s as f64)).sum()
    }

    /// High-frequency energy proxy: sum of squared first differences.
    fn hf_energy(block: &[Sample]) -> f64 {
        block
            .windows(2)
            .map(|w| {
                let d = (w[1] - w[0]) as f64;
                d * d
            })
            .sum()
    }

    #[test]
    fn renders_bounded_finite() {
        for &tune in &[50.0, 120.0, 250.0, 400.0] {
            for &depth in &[0.0, 140.0, 2_000.0] {
                for &drive in &[1.0, 1.5, 12.0] {
                    let params = AnalogTomParams {
                        tune_hz: tune,
                        pitch_env_hz: depth,
                        drive,
                        ..AnalogTomParams::default()
                    };
                    let mut node = AnalogTomNode::new(SR, params);
                    let block = render(&mut node, 4_096);
                    assert!(block.iter().all(|s| s.is_finite()));
                    assert!(peak(&block) <= 1.0 + 1e-4);
                }
            }
        }
    }

    #[test]
    fn output_is_bounded_under_extreme_settings() {
        let params = AnalogTomParams {
            tune_hz: 400.0,
            pitch_env_hz: 2_000.0,
            pitch_decay_s: 0.5,
            amp_decay_s: 4.0,
            overtone_ratio: 3.0,
            overtone_level: 1.0,
            drive: 12.0,
            amplitude: 1.0,
        };
        let mut node = AnalogTomNode::new(SR, params);
        let block = render(&mut node, 96_000);
        assert!(block.iter().all(|s| s.is_finite()));
        assert!(peak(&block) <= 1.0 + 1e-4);
    }

    #[test]
    fn default_params_in_domain() {
        let p = AnalogTomParams::default();
        assert!((MIN_TUNE_HZ..=MAX_TUNE_HZ).contains(&p.tune_hz));
        assert!((MIN_PITCH_ENV_HZ..=MAX_PITCH_ENV_HZ).contains(&p.pitch_env_hz));
        assert!((MIN_PITCH_DECAY_S..=MAX_PITCH_DECAY_S).contains(&p.pitch_decay_s));
        assert!((MIN_AMP_DECAY_S..=MAX_AMP_DECAY_S).contains(&p.amp_decay_s));
        assert!((MIN_OVERTONE_RATIO..=MAX_OVERTONE_RATIO).contains(&p.overtone_ratio));
        assert!((MIN_OVERTONE_LEVEL..=MAX_OVERTONE_LEVEL).contains(&p.overtone_level));
        assert!((MIN_DRIVE..=MAX_DRIVE).contains(&p.drive));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let params = AnalogTomParams {
            tune_hz: 5.0,
            pitch_env_hz: 9_999.0,
            pitch_decay_s: 0.0,
            amp_decay_s: 100.0,
            overtone_ratio: 99.0,
            overtone_level: 9.0,
            drive: 999.0,
            amplitude: 0.5,
        };
        let node = AnalogTomNode::new(SR, params);
        assert!((MIN_TUNE_HZ..=MAX_TUNE_HZ).contains(&node.tune_hz()));
        assert_eq!(node.pitch_env_hz(), MAX_PITCH_ENV_HZ);
        assert_eq!(node.pitch_decay_s(), MIN_PITCH_DECAY_S);
        assert_eq!(node.amp_decay_s(), MAX_AMP_DECAY_S);
        assert_eq!(node.overtone_ratio(), MAX_OVERTONE_RATIO);
        assert_eq!(node.overtone_level(), MAX_OVERTONE_LEVEL);
        assert_eq!(node.drive(), MAX_DRIVE);
    }

    #[test]
    fn getters_report_state() {
        let node = node_default();
        assert_eq!(node.tune_hz(), DEFAULT_TUNE_HZ);
        assert_eq!(node.pitch_env_hz(), DEFAULT_PITCH_ENV_HZ);
        assert_eq!(node.pitch_decay_s(), DEFAULT_PITCH_DECAY_S);
        assert_eq!(node.amp_decay_s(), DEFAULT_AMP_DECAY_S);
        assert_eq!(node.overtone_ratio(), DEFAULT_OVERTONE_RATIO);
        assert_eq!(node.overtone_level(), DEFAULT_OVERTONE_LEVEL);
        assert_eq!(node.drive(), DEFAULT_DRIVE);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn from_params_matches_new() {
        let params = AnalogTomParams {
            tune_hz: 110.0,
            overtone_level: 0.7,
            ..AnalogTomParams::default()
        };
        let mut a = AnalogTomNode::new(SR, params);
        let mut b = AnalogTomNode::from_params(SR, params);
        let ba = render(&mut a, 2_048);
        let bb = render(&mut b, 2_048);
        assert_eq!(ba, bb);
    }

    #[test]
    fn different_tune_alters_output() {
        let mut low = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                tune_hz: 80.0,
                ..AnalogTomParams::default()
            },
        );
        let mut high = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                tune_hz: 320.0,
                ..AnalogTomParams::default()
            },
        );
        let bl = render(&mut low, 2_048);
        let bh = render(&mut high, 2_048);
        assert!(hf_energy(&bh) > hf_energy(&bl));
    }

    #[test]
    fn latency_is_zero() {
        assert_eq!(node_default().latency_frames(), 0);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let params = AnalogTomParams {
            tune_hz: Sample::NAN,
            pitch_env_hz: Sample::INFINITY,
            pitch_decay_s: Sample::NAN,
            amp_decay_s: Sample::NEG_INFINITY,
            overtone_ratio: Sample::NAN,
            overtone_level: Sample::INFINITY,
            drive: Sample::NAN,
            amplitude: Sample::NAN,
        };
        let node = AnalogTomNode::new(SR, params);
        assert_eq!(node.tune_hz(), DEFAULT_TUNE_HZ);
        assert_eq!(node.pitch_env_hz(), DEFAULT_PITCH_ENV_HZ);
        assert_eq!(node.pitch_decay_s(), DEFAULT_PITCH_DECAY_S);
        assert_eq!(node.amp_decay_s(), DEFAULT_AMP_DECAY_S);
        assert_eq!(node.overtone_ratio(), DEFAULT_OVERTONE_RATIO);
        assert_eq!(node.overtone_level(), DEFAULT_OVERTONE_LEVEL);
        assert_eq!(node.drive(), DEFAULT_DRIVE);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn not_silent_with_default_params() {
        let mut node = node_default();
        let block = render(&mut node, 2_048);
        assert!(energy(&block) > 1.0);
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                amplitude: 0.25,
                ..AnalogTomParams::default()
            },
        );
        let mut loud = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                amplitude: 0.5,
                ..AnalogTomParams::default()
            },
        );
        let eq = energy(&render(&mut quiet, 8_192));
        let el = energy(&render(&mut loud, 8_192));
        assert!((el / eq - 4.0).abs() < 1e-2, "ratio {}", el / eq);
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono_node = node_default();
        let mono = render(&mut mono_node, 1_024);
        let mut stereo_node = node_default();
        let stereo = render_layout(&mut stereo_node, 1_024, ChannelLayout::Stereo);
        let mut quad_node = node_default();
        let quad = render_layout(&mut quad_node, 1_024, ChannelLayout::Quad);
        for ch in &stereo {
            assert_eq!(*ch, mono);
        }
        for ch in &quad {
            assert_eq!(*ch, mono);
        }
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = node_default();
        let mut b = node_default();
        assert_eq!(render(&mut a, 4_096), render(&mut b, 4_096));
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let p = AnalogTomParams::default();
        assert_eq!(p.sanitised(SR), p);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = node_default();
        node.set_tune_hz(Sample::NAN);
        assert_eq!(node.tune_hz(), DEFAULT_TUNE_HZ);
        node.set_tune_hz(9_999.0);
        assert!(node.tune_hz() <= MAX_TUNE_HZ);
        node.set_pitch_env_hz(Sample::INFINITY);
        assert_eq!(node.pitch_env_hz(), DEFAULT_PITCH_ENV_HZ);
        node.set_pitch_decay_s(-1.0);
        assert_eq!(node.pitch_decay_s(), MIN_PITCH_DECAY_S);
        node.set_amp_decay_s(100.0);
        assert_eq!(node.amp_decay_s(), MAX_AMP_DECAY_S);
        node.set_overtone_ratio(99.0);
        assert_eq!(node.overtone_ratio(), MAX_OVERTONE_RATIO);
        node.set_overtone_level(9.0);
        assert_eq!(node.overtone_level(), MAX_OVERTONE_LEVEL);
        node.set_drive(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.drive(), DEFAULT_DRIVE);
        node.set_drive(999.0, Ramp::Immediate);
        assert_eq!(node.drive(), MAX_DRIVE);
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                amplitude: 0.0,
                ..AnalogTomParams::default()
            },
        );
        let block = render(&mut node, 2_048);
        assert!(peak(&block) < 1e-6);
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = node_default();
        let before = node.clone();
        let _ = render(&mut node, 0);
        let mut after_a = node;
        let mut after_b = before;
        assert_eq!(render(&mut after_a, 512), render(&mut after_b, 512));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = node_default();
        let first = render(&mut node, 4_096);
        node.reset();
        let second = render(&mut node, 4_096);
        assert_eq!(first, second);
    }

    #[test]
    fn trigger_retriggers_envelope() {
        let mut node = node_default();
        // Ring the voice down to near silence (default amp decay 0.5 s).
        let _ = render(&mut node, 96_000);
        let tail = render(&mut node, 512);
        assert!(peak(&tail) < 1e-2, "tail peak {}", peak(&tail));
        node.trigger(1.0);
        let fresh = render(&mut node, 512);
        assert!(peak(&fresh) > peak(&tail));
    }

    #[test]
    fn longer_decay_rings_longer() {
        let mut short = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                amp_decay_s: 0.1,
                ..AnalogTomParams::default()
            },
        );
        let mut long = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                amp_decay_s: 2.0,
                ..AnalogTomParams::default()
            },
        );
        // Skip the shared attack, then measure the sustained tail energy.
        let _ = render(&mut short, 8_000);
        let _ = render(&mut long, 8_000);
        let es = energy(&render(&mut short, 8_000));
        let el = energy(&render(&mut long, 8_000));
        assert!(el > es * 2.0, "short {es} long {el}");
    }

    #[test]
    fn higher_overtone_level_adds_high_frequency_energy() {
        let mut none = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                overtone_level: 0.0,
                ..AnalogTomParams::default()
            },
        );
        let mut much = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                overtone_level: 1.0,
                ..AnalogTomParams::default()
            },
        );
        let bn = render(&mut none, 4_096);
        let bm = render(&mut much, 4_096);
        let ratio_none = hf_energy(&bn) / energy(&bn);
        let ratio_much = hf_energy(&bm) / energy(&bm);
        assert!(
            ratio_much > ratio_none,
            "none {ratio_none} much {ratio_much}"
        );
    }

    #[test]
    fn pitch_sweep_adds_initial_high_frequency() {
        let mut flat = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                pitch_env_hz: 0.0,
                ..AnalogTomParams::default()
            },
        );
        let mut swept = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                pitch_env_hz: 1_500.0,
                ..AnalogTomParams::default()
            },
        );
        // The sweep lives in the first few milliseconds of the attack.
        let bf = render(&mut flat, 1_024);
        let bs = render(&mut swept, 1_024);
        assert!(hf_energy(&bs) > hf_energy(&bf));
    }

    #[test]
    fn overtone_ratio_affects_spectrum() {
        let mut low = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                overtone_ratio: 1.2,
                overtone_level: 1.0,
                ..AnalogTomParams::default()
            },
        );
        let mut high = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                overtone_ratio: 3.0,
                overtone_level: 1.0,
                ..AnalogTomParams::default()
            },
        );
        let bl = render(&mut low, 4_096);
        let bh = render(&mut high, 4_096);
        assert!(hf_energy(&bh) > hf_energy(&bl));
    }

    #[test]
    fn is_pitched_tonal_voice() {
        // With no noise layer the voice is purely deterministic and periodic:
        // two independent instances must agree sample for sample.
        let mut a = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                pitch_env_hz: 0.0,
                ..AnalogTomParams::default()
            },
        );
        let mut b = AnalogTomNode::new(
            SR,
            AnalogTomParams {
                pitch_env_hz: 0.0,
                ..AnalogTomParams::default()
            },
        );
        assert_eq!(render(&mut a, 8_192), render(&mut b, 8_192));
    }
}
