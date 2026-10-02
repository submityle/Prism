//! Analog-style synthesized bass-drum (kick) voice source node.
//!
//! [`AnalogKickNode`] synthesizes the classic analog bass-drum voice: a single
//! sine "body" oscillator whose pitch sweeps rapidly downward from a bright
//! attack toward a low fundamental, shaped by an exponential amplitude decay,
//! topped with a short transient "click" for attack definition and pushed
//! through a saturating waveshaper for weight and harmonic body. Unlike the
//! modal physical drums ([`membrane_drum`](super::membrane_drum),
//! [`struck_bar`](super::struck_bar), [`struck_plate`](super::struck_plate))
//! which excite a bank of resonant modes, this voice is a direct subtractive
//! synthesis of a single enveloped tone, exactly as realized by the bridged-T
//! oscillator of a drum machine. It is a *source* (zero inputs, one output):
//! it supplies its own excitation through [`AnalogKickNode::trigger`] and never
//! reads its inputs.
//!
//! # Model
//!
//! A single phase accumulator runs a sine whose instantaneous frequency is
//!
//! ```text
//!   f(t) = base_freq_hz + pitch_env_hz(t)
//!   pitch_env_hz(t) = pitch_env_depth_hz * exp(-t / pitch_decay_s)
//! ```
//!
//! so the body starts at `base_freq_hz + pitch_env_depth_hz` and relaxes toward
//! `base_freq_hz` with time constant `pitch_decay_s`. This fast downward chirp
//! is what gives a kick its characteristic "thump then tone". The body is
//! multiplied by an exponential amplitude envelope with a `-60 dB` time of
//! `amp_decay_s`, summed with a short Hann-windowed high-frequency click burst
//! (scaled by `click_amount`), and finally driven through a `tanh` saturator so
//! that increasing `drive` fattens the harmonics without ever exceeding full
//! scale:
//!
//! ```text
//!   excitation = sin(body_phase) * amp_env + click
//!   shaped     = tanh(drive * excitation) / tanh(drive)
//!   out        = amplitude * shaped
//! ```
//!
//! Because the saturator sits *before* the final `amplitude` gain, the output
//! level is a strict square law in `amplitude` while the `tanh` guarantees the
//! voice can never clip past `|amplitude|`.
//!
//! # Real-time contract
//!
//! Construction and [`AnalogKickNode::trigger`] pre-compute all per-voice
//! coefficients, so [`AnalogKickNode::process`] performs no allocation, no
//! locking, and cannot panic: it is a pure per-sample state machine. The
//! `drive` and `amplitude` controls are driven through [`Smoothed`] values so
//! that automation never introduces zipper noise, and a trigger is latched to
//! the block boundary so retriggering is click-free. [`AnalogKickNode::reset`]
//! restarts the exact same voice, so two nodes built identically and triggered
//! identically emit bit-identical streams on every platform.
//!
//! # Provenance
//!
//! The synthesis technique here -- a pitch-swept, exponentially decaying sine
//! with a short transient click and a saturating waveshaper -- is classic
//! public-domain analog drum-voice DSP, exemplified by the Roland `TR-808`
//! bridged-T bass-drum circuit (1980) and documented widely since. Only the
//! general technique is reproduced from first principles; no source code or
//! derivative from any audio engine or toolbox (Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Google Resonance Audio, Web Audio, STK) is used.
//!
//! # Relationship
//!
//! Unlike [`membrane_drum`](super::membrane_drum),
//! [`struck_bar`](super::struck_bar), and [`struck_plate`](super::struck_plate)
//! -- which model a struck object as a bank of resonant modal filters -- this
//! node is a direct, single-oscillator subtractive voice with explicit pitch
//! and amplitude envelopes, so its tone is a clean pitched thump rather than a
//! ringing modal spectrum. Unlike [`oscillator`](super::oscillator), which
//! emits a steady band-limited waveform, this is an enveloped one-shot with a
//! downward pitch sweep. Unlike [`noise`](super::noise), it is fully tonal and
//! deterministic.

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Lowest tunable body fundamental in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Highest tunable body fundamental in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 400.0;

/// Default body fundamental in hertz (a typical deep kick).
pub const DEFAULT_FREQUENCY_HZ: Sample = 55.0;

/// Smallest initial pitch-sweep depth above the fundamental, in hertz.
pub const MIN_PITCH_ENV_HZ: Sample = 0.0;

/// Largest initial pitch-sweep depth above the fundamental, in hertz.
pub const MAX_PITCH_ENV_HZ: Sample = 3_000.0;

/// Default initial pitch-sweep depth above the fundamental, in hertz.
pub const DEFAULT_PITCH_ENV_HZ: Sample = 240.0;

/// Shortest pitch-sweep time constant in seconds.
pub const MIN_PITCH_DECAY_S: Sample = 0.002;

/// Longest pitch-sweep time constant in seconds.
pub const MAX_PITCH_DECAY_S: Sample = 0.5;

/// Default pitch-sweep time constant in seconds.
pub const DEFAULT_PITCH_DECAY_S: Sample = 0.03;

/// Shortest `-60 dB` amplitude decay time in seconds.
pub const MIN_AMP_DECAY_S: Sample = 0.02;

/// Longest `-60 dB` amplitude decay time in seconds.
pub const MAX_AMP_DECAY_S: Sample = 4.0;

/// Default `-60 dB` amplitude decay time in seconds.
pub const DEFAULT_AMP_DECAY_S: Sample = 0.35;

/// Smallest saturator drive (`1.0` is nearly linear).
pub const MIN_DRIVE: Sample = 1.0;

/// Largest saturator drive (heavy harmonic fattening).
pub const MAX_DRIVE: Sample = 12.0;

/// Default saturator drive.
pub const DEFAULT_DRIVE: Sample = 1.5;

/// Default transient click amount in `[0, 1]`.
pub const DEFAULT_CLICK: Sample = 0.3;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.9;

/// Default strike velocity used by [`AnalogKickNode::trigger`].
pub const DEFAULT_VELOCITY: Sample = 1.0;

/// Nominal transient-click frequency in hertz (bounded by the Nyquist guard).
const CLICK_FREQUENCY_HZ: Sample = 1_200.0;

/// Transient-click duration in milliseconds.
const CLICK_MS: Sample = 3.0;

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

/// Clamps a body fundamental to the tunable range and the Nyquist guard.
#[inline]
fn clamp_frequency(freq_hz: Sample, sample_rate: u32) -> Sample {
    let nyquist = sample_rate.max(1) as Sample * NYQUIST_GUARD;
    let upper = MAX_FREQUENCY_HZ.min(nyquist).max(MIN_FREQUENCY_HZ);
    freq_hz.clamp(MIN_FREQUENCY_HZ, upper)
}

/// Construction parameters for an [`AnalogKickNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AnalogKickParams {
    /// Body fundamental frequency in hertz.
    pub frequency_hz: Sample,
    /// Initial pitch-sweep depth above the fundamental, in hertz.
    pub pitch_env_hz: Sample,
    /// Pitch-sweep time constant in seconds.
    pub pitch_decay_s: Sample,
    /// Amplitude `-60 dB` decay time in seconds.
    pub amp_decay_s: Sample,
    /// Saturator drive (`1.0` nearly linear, higher adds harmonics).
    pub drive: Sample,
    /// Transient-click amount in `[0, 1]`.
    pub click_amount: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for AnalogKickParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            pitch_env_hz: DEFAULT_PITCH_ENV_HZ,
            pitch_decay_s: DEFAULT_PITCH_DECAY_S,
            amp_decay_s: DEFAULT_AMP_DECAY_S,
            drive: DEFAULT_DRIVE,
            click_amount: DEFAULT_CLICK,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl AnalogKickParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range. Frequency is clamped against the Nyquist limit too.
    #[must_use]
    pub fn sanitised(self, sample_rate: u32) -> Self {
        let d = Self::default();
        let frequency_hz =
            clamp_frequency(finite_or(self.frequency_hz, d.frequency_hz), sample_rate);
        let pitch_env_hz = finite_or(self.pitch_env_hz, d.pitch_env_hz)
            .clamp(MIN_PITCH_ENV_HZ, MAX_PITCH_ENV_HZ);
        let pitch_decay_s = finite_or(self.pitch_decay_s, d.pitch_decay_s)
            .clamp(MIN_PITCH_DECAY_S, MAX_PITCH_DECAY_S);
        let amp_decay_s =
            finite_or(self.amp_decay_s, d.amp_decay_s).clamp(MIN_AMP_DECAY_S, MAX_AMP_DECAY_S);
        let drive = finite_or(self.drive, d.drive).clamp(MIN_DRIVE, MAX_DRIVE);
        let click_amount = finite_or(self.click_amount, d.click_amount).clamp(0.0, 1.0);
        let amplitude = finite_or(self.amplitude, d.amplitude);
        Self {
            frequency_hz,
            pitch_env_hz,
            pitch_decay_s,
            amp_decay_s,
            drive,
            click_amount,
            amplitude,
        }
    }
}

/// Analog-style synthesized bass-drum (kick) voice source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::{AnalogKickNode, AnalogKickParams};
///
/// let mut node = AnalogKickNode::new(48_000, AnalogKickParams::default());
/// node.trigger(1.0);
/// // The trigger injects a fresh enveloped voice, so output is non-silent.
/// ```
#[derive(Clone, Debug)]
pub struct AnalogKickNode {
    sample_rate: u32,
    frequency_hz: Sample,
    pitch_env_depth_hz: Sample,
    pitch_decay_s: Sample,
    amp_decay_s: Sample,
    click_amount: Sample,
    drive: Smoothed,
    amplitude: Smoothed,
    body_phase: Sample,
    click_phase: Sample,
    amp_env: Sample,
    pitch_env: Sample,
    amp_decay_coeff: Sample,
    pitch_decay_coeff: Sample,
    click_len: u32,
    click_pos: u32,
    velocity: Sample,
}

impl AnalogKickNode {
    /// Builds an analog kick voice for `sample_rate` Hz from `params`. The
    /// voice is triggered once at [`DEFAULT_VELOCITY`] so a freshly built node
    /// renders audible output immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: AnalogKickParams) -> Self {
        let p = params.sanitised(sample_rate);
        let mut node = Self {
            sample_rate: sample_rate.max(1),
            frequency_hz: p.frequency_hz,
            pitch_env_depth_hz: p.pitch_env_hz,
            pitch_decay_s: p.pitch_decay_s,
            amp_decay_s: p.amp_decay_s,
            click_amount: p.click_amount,
            drive: Smoothed::new(p.drive),
            amplitude: Smoothed::new(p.amplitude),
            body_phase: 0.0,
            click_phase: 0.0,
            amp_env: 0.0,
            pitch_env: 0.0,
            amp_decay_coeff: 0.0,
            pitch_decay_coeff: 0.0,
            click_len: 0,
            click_pos: u32::MAX,
            velocity: DEFAULT_VELOCITY,
        };
        node.recompute();
        node.trigger(DEFAULT_VELOCITY);
        node
    }

    /// Returns the body fundamental in hertz.
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
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

    /// Returns the saturator drive.
    #[must_use]
    pub fn drive(&self) -> Sample {
        self.drive.target()
    }

    /// Returns the transient-click amount in `[0, 1]`.
    #[must_use]
    pub fn click_amount(&self) -> Sample {
        self.click_amount
    }

    /// Returns the target output amplitude.
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Sets the body fundamental, clamped to the tunable range. Takes effect
    /// from the next sample; it only shifts the sweep target and is click-free.
    pub fn set_frequency(&mut self, frequency_hz: Sample) {
        self.frequency_hz =
            clamp_frequency(finite_or(frequency_hz, self.frequency_hz), self.sample_rate);
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

    /// Sets the saturator drive, gliding over `ramp`.
    pub fn set_drive(&mut self, drive: Sample, ramp: Ramp) {
        let target = finite_or(drive, self.drive.target()).clamp(MIN_DRIVE, MAX_DRIVE);
        self.drive.set_target(target, ramp);
    }

    /// Sets the transient-click amount, clamped to `[0, 1]`.
    pub fn set_click_amount(&mut self, click_amount: Sample) {
        self.click_amount = finite_or(click_amount, self.click_amount).clamp(0.0, 1.0);
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Retriggers the voice with the given `velocity` (clamped to `[0, 1]`),
    /// restarting the pitch and amplitude envelopes and the transient click.
    pub fn trigger(&mut self, velocity: Sample) {
        self.velocity = finite_or(velocity, DEFAULT_VELOCITY).clamp(0.0, 1.0);
        self.body_phase = 0.0;
        self.click_phase = 0.0;
        self.amp_env = self.velocity;
        self.pitch_env = self.pitch_env_depth_hz;
        self.click_pos = 0;
    }

    /// Recomputes the per-sample decay coefficients and the click length from
    /// the current scalar parameters. Never runs on the audio hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        self.amp_decay_coeff = ops::exp(-LN_1000 / (self.amp_decay_s * sr));
        self.pitch_decay_coeff = ops::exp(-1.0 / (self.pitch_decay_s * sr));
        let len = ops::round(CLICK_MS * sr / 1000.0) as i32;
        self.click_len = len.max(1) as u32;
    }

    /// Renders one mono output sample, advancing the oscillator, envelopes, and
    /// the transient click by one step.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let sr = self.sample_rate.max(1) as Sample;
        let nyquist = sr * NYQUIST_GUARD;

        // Body oscillator with the swept instantaneous frequency.
        let inst_freq = (self.frequency_hz + self.pitch_env).clamp(0.0, nyquist);
        let body = ops::sin(self.body_phase);
        self.body_phase += TAU * inst_freq / sr;
        if self.body_phase >= TAU {
            self.body_phase -= TAU;
        }

        // Transient click: a short Hann-windowed high-frequency burst.
        let click = if self.click_pos < self.click_len {
            let n = self.click_pos as Sample;
            let window = 0.5 - 0.5 * ops::cos(TAU * (n + 1.0) / (self.click_len as Sample + 1.0));
            let click_freq = CLICK_FREQUENCY_HZ.min(nyquist);
            let tone = ops::sin(self.click_phase);
            self.click_phase += TAU * click_freq / sr;
            if self.click_phase >= TAU {
                self.click_phase -= TAU;
            }
            self.click_pos += 1;
            self.click_amount * self.velocity * window * tone
        } else {
            0.0
        };

        let excitation = body * self.amp_env + click;

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

impl AudioNode for AnalogKickNode {
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

    fn node_default() -> AnalogKickNode {
        AnalogKickNode::new(SR, AnalogKickParams::default())
    }

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut AnalogKickNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(
        node: &mut AnalogKickNode,
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
        for &freq in &[20.0, 55.0, 200.0, 400.0] {
            for &depth in &[0.0, 240.0, 3_000.0] {
                for &drive in &[1.0, 1.5, 12.0] {
                    let params = AnalogKickParams {
                        frequency_hz: freq,
                        pitch_env_hz: depth,
                        drive,
                        ..AnalogKickParams::default()
                    };
                    let mut node = AnalogKickNode::new(SR, params);
                    let block = render(&mut node, 4_096);
                    assert!(block.iter().all(|s| s.is_finite()));
                    assert!(peak(&block) <= 1.0 + 1e-4);
                }
            }
        }
    }

    #[test]
    fn default_trigger_produces_sound() {
        let mut node = node_default();
        let block = render(&mut node, 2_048);
        assert!(energy(&block) > 1.0);
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let params = AnalogKickParams {
            amplitude: 0.0,
            ..AnalogKickParams::default()
        };
        let mut node = AnalogKickNode::new(SR, params);
        let block = render(&mut node, 2_048);
        assert!(block.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = node_default();
        let mut b = node_default();
        let ba = render(&mut a, 2_048);
        let bb = render(&mut b, 2_048);
        assert_eq!(ba, bb);
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = node_default();
        let first = render(&mut node, 2_048);
        node.reset();
        let second = render(&mut node, 2_048);
        assert_eq!(first, second);
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono_node = node_default();
        let mono = render(&mut mono_node, 1_024);

        let mut stereo_node = node_default();
        let stereo = render_layout(&mut stereo_node, 1_024, ChannelLayout::Stereo);
        for ch in &stereo {
            assert_eq!(ch, &mono);
        }

        let mut quad_node = node_default();
        let quad = render_layout(&mut quad_node, 1_024, ChannelLayout::Quad);
        for ch in &quad {
            assert_eq!(ch, &mono);
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = node_default();
        let before = node.body_phase;
        let out = render(&mut node, 0);
        assert!(out.is_empty());
        assert_eq!(node.body_phase, before);
    }

    #[test]
    fn latency_is_zero() {
        assert_eq!(node_default().latency_frames(), 0);
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let quiet_params = AnalogKickParams {
            amplitude: 0.25,
            ..AnalogKickParams::default()
        };
        let loud_params = AnalogKickParams {
            amplitude: 0.5,
            ..AnalogKickParams::default()
        };
        let mut quiet = AnalogKickNode::new(SR, quiet_params);
        let mut loud = AnalogKickNode::new(SR, loud_params);
        let eq = energy(&render(&mut quiet, 4_096));
        let el = energy(&render(&mut loud, 4_096));
        assert!(eq > 0.0);
        let ratio = el / eq;
        assert!((ratio - 4.0).abs() < 1e-2, "ratio = {ratio}");
    }

    #[test]
    fn trigger_retriggers_envelope() {
        let mut node = node_default();
        // Let the first hit decay well into its tail.
        let _ = render(&mut node, 32_000);
        let tail = energy(&render(&mut node, 1_024));
        node.trigger(1.0);
        let fresh = energy(&render(&mut node, 1_024));
        assert!(fresh > tail * 4.0, "fresh = {fresh}, tail = {tail}");
    }

    #[test]
    fn amplitude_envelope_decays() {
        let mut node = node_default();
        let block = render(&mut node, 24_000);
        let early = energy(&block[0..2_048]);
        let late = energy(&block[block.len() - 2_048..]);
        assert!(early > late * 4.0, "early = {early}, late = {late}");
    }

    #[test]
    fn pitch_sweeps_downward() {
        // A long sweep makes the attack much brighter than the sustained body.
        let params = AnalogKickParams {
            pitch_env_hz: 2_000.0,
            pitch_decay_s: 0.05,
            amp_decay_s: 2.0,
            click_amount: 0.0,
            ..AnalogKickParams::default()
        };
        let mut node = AnalogKickNode::new(SR, params);
        let block = render(&mut node, 12_000);
        // Normalise HF content by local energy so the amplitude decay does not
        // dominate the comparison: compare spectral brightness, not loudness.
        let w = 2_048;
        let early_hf = hf_energy(&block[0..w]) / energy(&block[0..w]).max(1e-12);
        let late_hf =
            hf_energy(&block[block.len() - w..]) / energy(&block[block.len() - w..]).max(1e-12);
        assert!(early_hf > late_hf * 2.0, "early = {early_hf}, late = {late_hf}");
    }

    #[test]
    fn higher_base_freq_raises_pitch() {
        fn zero_crossings(block: &[Sample]) -> usize {
            block
                .windows(2)
                .filter(|w| (w[0] <= 0.0) != (w[1] <= 0.0))
                .count()
        }
        // Compare the sustained body after the pitch sweep has settled.
        let low_params = AnalogKickParams {
            frequency_hz: 50.0,
            pitch_env_hz: 0.0,
            amp_decay_s: 4.0,
            ..AnalogKickParams::default()
        };
        let high_params = AnalogKickParams {
            frequency_hz: 200.0,
            pitch_env_hz: 0.0,
            amp_decay_s: 4.0,
            ..AnalogKickParams::default()
        };
        let mut low = AnalogKickNode::new(SR, low_params);
        let mut high = AnalogKickNode::new(SR, high_params);
        let lb = render(&mut low, 8_000);
        let hb = render(&mut high, 8_000);
        assert!(zero_crossings(&hb) > zero_crossings(&lb) * 2);
    }

    #[test]
    fn pitch_env_adds_attack_brightness() {
        let flat_params = AnalogKickParams {
            pitch_env_hz: 0.0,
            click_amount: 0.0,
            ..AnalogKickParams::default()
        };
        let swept_params = AnalogKickParams {
            pitch_env_hz: 2_000.0,
            click_amount: 0.0,
            ..AnalogKickParams::default()
        };
        let mut flat = AnalogKickNode::new(SR, flat_params);
        let mut swept = AnalogKickNode::new(SR, swept_params);
        let fb = render(&mut flat, 2_048);
        let sb = render(&mut swept, 2_048);
        assert!(hf_energy(&sb) > hf_energy(&fb) * 2.0);
    }

    #[test]
    fn drive_increases_harmonic_content() {
        let clean_params = AnalogKickParams {
            drive: 1.0,
            click_amount: 0.0,
            ..AnalogKickParams::default()
        };
        let dirty_params = AnalogKickParams {
            drive: 12.0,
            click_amount: 0.0,
            ..AnalogKickParams::default()
        };
        let mut clean = AnalogKickNode::new(SR, clean_params);
        let mut dirty = AnalogKickNode::new(SR, dirty_params);
        // Early window, where the body is near full scale and saturation bites.
        let cb = render(&mut clean, 1_024);
        let db = render(&mut dirty, 1_024);
        assert!(hf_energy(&db) > hf_energy(&cb) * 1.5);
    }

    #[test]
    fn click_adds_attack_energy() {
        let no_click_params = AnalogKickParams {
            click_amount: 0.0,
            pitch_env_hz: 0.0,
            ..AnalogKickParams::default()
        };
        let click_params = AnalogKickParams {
            click_amount: 1.0,
            pitch_env_hz: 0.0,
            ..AnalogKickParams::default()
        };
        let mut no_click = AnalogKickNode::new(SR, no_click_params);
        let mut click = AnalogKickNode::new(SR, click_params);
        let nb = render(&mut no_click, 512);
        let cb = render(&mut click, 512);
        assert!(hf_energy(&cb) > hf_energy(&nb) * 2.0);
    }

    #[test]
    fn getters_report_state() {
        let node = node_default();
        assert!((node.frequency_hz() - DEFAULT_FREQUENCY_HZ).abs() < 1e-6);
        assert!((node.pitch_env_hz() - DEFAULT_PITCH_ENV_HZ).abs() < 1e-6);
        assert!((node.pitch_decay_s() - DEFAULT_PITCH_DECAY_S).abs() < 1e-6);
        assert!((node.amp_decay_s() - DEFAULT_AMP_DECAY_S).abs() < 1e-6);
        assert!((node.drive() - DEFAULT_DRIVE).abs() < 1e-6);
        assert!((node.click_amount() - DEFAULT_CLICK).abs() < 1e-6);
        assert!((node.amplitude() - DEFAULT_AMPLITUDE).abs() < 1e-6);
    }

    #[test]
    fn from_params_matches_new() {
        let params = AnalogKickParams {
            frequency_hz: 70.0,
            drive: 4.0,
            ..AnalogKickParams::default()
        };
        let mut a = AnalogKickNode::new(SR, params);
        let mut b = AnalogKickNode::new(SR, params);
        assert_eq!(render(&mut a, 1_024), render(&mut b, 1_024));
    }

    #[test]
    fn default_params_in_domain() {
        let d = AnalogKickParams::default();
        assert!((MIN_FREQUENCY_HZ..=MAX_FREQUENCY_HZ).contains(&d.frequency_hz));
        assert!((MIN_PITCH_ENV_HZ..=MAX_PITCH_ENV_HZ).contains(&d.pitch_env_hz));
        assert!((MIN_PITCH_DECAY_S..=MAX_PITCH_DECAY_S).contains(&d.pitch_decay_s));
        assert!((MIN_AMP_DECAY_S..=MAX_AMP_DECAY_S).contains(&d.amp_decay_s));
        assert!((MIN_DRIVE..=MAX_DRIVE).contains(&d.drive));
        assert!((0.0..=1.0).contains(&d.click_amount));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let params = AnalogKickParams {
            frequency_hz: 1.0e9,
            pitch_env_hz: -5.0,
            pitch_decay_s: 100.0,
            amp_decay_s: 0.0,
            drive: 1.0e9,
            click_amount: 5.0,
            amplitude: 0.5,
        };
        let node = AnalogKickNode::new(SR, params);
        assert!((MIN_FREQUENCY_HZ..=MAX_FREQUENCY_HZ).contains(&node.frequency_hz()));
        assert!((MIN_PITCH_ENV_HZ..=MAX_PITCH_ENV_HZ).contains(&node.pitch_env_hz()));
        assert!((MIN_PITCH_DECAY_S..=MAX_PITCH_DECAY_S).contains(&node.pitch_decay_s()));
        assert!((MIN_AMP_DECAY_S..=MAX_AMP_DECAY_S).contains(&node.amp_decay_s()));
        assert!((MIN_DRIVE..=MAX_DRIVE).contains(&node.drive()));
        assert!((0.0..=1.0).contains(&node.click_amount()));
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let params = AnalogKickParams {
            frequency_hz: Sample::NAN,
            pitch_env_hz: Sample::INFINITY,
            pitch_decay_s: Sample::NAN,
            amp_decay_s: Sample::NEG_INFINITY,
            drive: Sample::NAN,
            click_amount: Sample::INFINITY,
            amplitude: Sample::NAN,
        };
        let node = AnalogKickNode::new(SR, params);
        assert!((node.frequency_hz() - DEFAULT_FREQUENCY_HZ).abs() < 1e-6);
        assert!((node.pitch_env_hz() - DEFAULT_PITCH_ENV_HZ).abs() < 1e-6);
        assert!((node.pitch_decay_s() - DEFAULT_PITCH_DECAY_S).abs() < 1e-6);
        assert!((node.amp_decay_s() - DEFAULT_AMP_DECAY_S).abs() < 1e-6);
        assert!((node.drive() - DEFAULT_DRIVE).abs() < 1e-6);
        assert!((node.click_amount() - DEFAULT_CLICK).abs() < 1e-6);
        assert!((node.amplitude() - DEFAULT_AMPLITUDE).abs() < 1e-6);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = node_default();
        node.set_frequency(Sample::NAN);
        assert!((node.frequency_hz() - DEFAULT_FREQUENCY_HZ).abs() < 1e-6);
        node.set_frequency(1.0e9);
        assert!(node.frequency_hz() <= MAX_FREQUENCY_HZ + 1e-3);
        node.set_drive(Sample::INFINITY, Ramp::Immediate);
        assert!((node.drive() - DEFAULT_DRIVE).abs() < 1e-6);
        node.set_drive(1.0e9, Ramp::Immediate);
        assert!((node.drive() - MAX_DRIVE).abs() < 1e-6);
        node.set_click_amount(5.0);
        assert!((node.click_amount() - 1.0).abs() < 1e-6);
        node.set_pitch_env_hz(-1.0);
        assert!((node.pitch_env_hz() - MIN_PITCH_ENV_HZ).abs() < 1e-6);
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let params = AnalogKickParams::default();
        assert_eq!(params.sanitised(SR), params);
    }

    #[test]
    fn output_is_bounded_under_extreme_settings() {
        let params = AnalogKickParams {
            frequency_hz: MAX_FREQUENCY_HZ,
            pitch_env_hz: MAX_PITCH_ENV_HZ,
            pitch_decay_s: MIN_PITCH_DECAY_S,
            amp_decay_s: MAX_AMP_DECAY_S,
            drive: MAX_DRIVE,
            click_amount: 1.0,
            amplitude: 1.0,
        };
        let mut node = AnalogKickNode::new(SR, params);
        for _ in 0..40 {
            node.trigger(1.0);
            let block = render(&mut node, 5_000);
            assert!(block.iter().all(|s| s.is_finite()));
            assert!(peak(&block) <= 1.0 + 1e-4);
        }
    }
}
