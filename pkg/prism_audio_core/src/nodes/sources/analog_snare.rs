//! Analog-style synthesized snare-drum voice source node.
//!
//! [`AnalogSnareNode`] synthesizes the classic analog snare-drum voice: a pair
//! of tuned sine "shell" partials that give the drum its body pitch, summed
//! with a burst of band-passed noise that models the metallic snare wires
//! rattling against the resonant head. Each layer has its own exponential
//! amplitude envelope, a `tone_noise_mix` control balances the two, and the sum
//! is pushed through a `tanh` saturator for weight before the final amplitude
//! gain. Unlike the modal physical drums ([`membrane_drum`](super::membrane_drum),
//! [`struck_bar`](super::struck_bar), [`struck_plate`](super::struck_plate))
//! which excite a bank of resonant modes, this voice is a direct subtractive
//! synthesis of two enveloped layers, exactly as realized by the two-oscillator
//! plus filtered-noise topology of a classic drum machine. It is a *source*
//! (zero inputs, one output): it supplies its own excitation through
//! [`AnalogSnareNode::trigger`] and never reads its inputs.
//!
//! # Model
//!
//! Two phase accumulators run fixed-pitch sine partials at `tone_freq_hz` and
//! `tone_freq_hz * TONE_PARTIAL_RATIO` (a tuned inharmonic pair, as on the
//! Roland `TR-808`). Their mean is scaled by an exponential tone envelope with
//! a `-60 dB` time of `tone_decay_s`:
//!
//! ```text
//!   tone(t) = 0.5 * (sin(w0 t) + sin(w1 t)) * exp(-t / tone_tau)
//! ```
//!
//! A deterministic white-noise stream (a seeded `xorshift32` generator, reset
//! on every trigger so the voice is reproducible) is passed through a two-pole
//! band-pass biquad centred at `noise_cutoff_hz` with resonance `noise_q`, then
//! scaled by its own exponential envelope with a `-60 dB` time of
//! `noise_decay_s`:
//!
//! ```text
//!   noise(t) = bandpass(white(t)) * exp(-t / noise_tau)
//! ```
//!
//! The two layers are crossfaded by `tone_noise_mix` (`0` = all tone, `1` = all
//! noise), summed, and driven through a `tanh` saturator so that increasing
//! `drive` fattens the harmonics without ever exceeding full scale:
//!
//! ```text
//!   excitation = (1 - mix) * tone + mix * noise
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
//! Construction, [`AnalogSnareNode::trigger`], and every scalar setter
//! pre-compute all per-voice coefficients (envelope decays and the band-pass
//! biquad), so [`AnalogSnareNode::process`] performs no allocation, no locking,
//! and cannot panic: it is a pure per-sample state machine. The `drive` and
//! `amplitude` controls are driven through [`Smoothed`] values so that
//! automation never introduces zipper noise, and a trigger is latched to the
//! block boundary so retriggering is click-free. The noise generator is a
//! deterministic integer recurrence reseeded to a fixed constant on every
//! trigger, so [`AnalogSnareNode::reset`] restarts the exact same voice and two
//! nodes built identically and triggered identically emit bit-identical streams
//! on every platform.
//!
//! # Provenance
//!
//! The synthesis technique here -- two tuned decaying sine partials mixed with
//! an enveloped band-passed noise burst and a saturating waveshaper -- is
//! classic public-domain analog drum-voice DSP, exemplified by the Roland
//! `TR-808` snare circuit (1980). The band-pass is a textbook public-domain
//! `RBJ` biquad (the Audio EQ Cookbook formulae, placed in the public domain by
//! Robert Bristow-Johnson), and the noise source is a public-domain `xorshift`
//! integer generator. Only the general techniques are reproduced from first
//! principles; no source code or derivative from any audio engine or toolbox
//! (Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance
//! Audio, Web Audio, STK) is used.
//!
//! # Relationship
//!
//! Unlike [`membrane_drum`](super::membrane_drum),
//! [`struck_bar`](super::struck_bar), and [`struck_plate`](super::struck_plate)
//! -- which model a struck object as a bank of resonant modal filters -- this
//! node is a direct, two-layer subtractive voice with explicit tone and noise
//! envelopes, so its character is a crisp pitched crack plus a filtered rattle
//! rather than a ringing modal spectrum. Unlike
//! [`analog_kick`](super::analog_kick), which is a single pitch-swept sine with
//! a transient click, this voice has no pitch sweep but adds a second tuned
//! partial and a dominant band-passed noise layer. Unlike
//! [`noise`](super::noise), whose stream is a raw stochastic generator, the
//! noise here is a *deterministic* seeded recurrence, band-limited by a
//! resonant filter and shaped by an envelope, so the whole voice reproduces
//! bit-for-bit.

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Lowest tunable shell fundamental in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 80.0;

/// Highest tunable shell fundamental in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 500.0;

/// Default shell fundamental in hertz (a typical analog snare body).
pub const DEFAULT_FREQUENCY_HZ: Sample = 180.0;

/// Shortest `-60 dB` tone-layer decay time in seconds.
pub const MIN_TONE_DECAY_S: Sample = 0.01;

/// Longest `-60 dB` tone-layer decay time in seconds.
pub const MAX_TONE_DECAY_S: Sample = 1.0;

/// Default `-60 dB` tone-layer decay time in seconds.
pub const DEFAULT_TONE_DECAY_S: Sample = 0.12;

/// Shortest `-60 dB` noise-layer decay time in seconds.
pub const MIN_NOISE_DECAY_S: Sample = 0.01;

/// Longest `-60 dB` noise-layer decay time in seconds.
pub const MAX_NOISE_DECAY_S: Sample = 2.0;

/// Default `-60 dB` noise-layer decay time in seconds.
pub const DEFAULT_NOISE_DECAY_S: Sample = 0.2;

/// Lowest band-pass centre for the noise layer, in hertz.
pub const MIN_NOISE_CUTOFF_HZ: Sample = 300.0;

/// Highest band-pass centre for the noise layer, in hertz.
pub const MAX_NOISE_CUTOFF_HZ: Sample = 12_000.0;

/// Default band-pass centre for the noise layer, in hertz.
pub const DEFAULT_NOISE_CUTOFF_HZ: Sample = 3_000.0;

/// Smallest band-pass resonance (broad, airy noise).
pub const MIN_NOISE_Q: Sample = 0.3;

/// Largest band-pass resonance (narrow, whistling noise).
pub const MAX_NOISE_Q: Sample = 8.0;

/// Default band-pass resonance for the noise layer.
pub const DEFAULT_NOISE_Q: Sample = 0.7;

/// Default tone/noise crossfade (`0` tone, `1` noise).
pub const DEFAULT_TONE_NOISE_MIX: Sample = 0.6;

/// Smallest saturator drive (`1.0` is nearly linear).
pub const MIN_DRIVE: Sample = 1.0;

/// Largest saturator drive (heavy harmonic fattening).
pub const MAX_DRIVE: Sample = 12.0;

/// Default saturator drive.
pub const DEFAULT_DRIVE: Sample = 1.5;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Default strike velocity used by [`AnalogSnareNode::trigger`].
pub const DEFAULT_VELOCITY: Sample = 1.0;

/// Frequency ratio of the second tuned shell partial to the fundamental; the
/// inharmonic value gives the body its characteristic non-musical "tom" pitch.
const TONE_PARTIAL_RATIO: Sample = 1.78;

/// `ln(1000) == 3 * ln(10)`, used by the `-60 dB`-time-to-decay mapping.
const LN_1000: Sample = 6.907_755;

/// Fraction of the sample rate above which a partial is muted (anti-alias).
const NYQUIST_GUARD: Sample = 0.49;

/// Fixed non-zero seed for the `xorshift32` noise generator (reset per trigger).
const NOISE_SEED: u32 = 0x9E37_79B9;

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Clamps a shell fundamental to the tunable range and the Nyquist guard.
#[inline]
fn clamp_frequency(freq_hz: Sample, sample_rate: u32) -> Sample {
    let nyquist = sample_rate.max(1) as Sample * NYQUIST_GUARD;
    let upper = MAX_FREQUENCY_HZ.min(nyquist).max(MIN_FREQUENCY_HZ);
    freq_hz.clamp(MIN_FREQUENCY_HZ, upper)
}

/// Clamps the band-pass centre to its range and the Nyquist guard.
#[inline]
fn clamp_cutoff(cutoff_hz: Sample, sample_rate: u32) -> Sample {
    let nyquist = sample_rate.max(1) as Sample * NYQUIST_GUARD;
    let upper = MAX_NOISE_CUTOFF_HZ.min(nyquist).max(MIN_NOISE_CUTOFF_HZ);
    cutoff_hz.clamp(MIN_NOISE_CUTOFF_HZ, upper)
}

/// Construction parameters for an [`AnalogSnareNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AnalogSnareParams {
    /// Shell fundamental frequency in hertz.
    pub frequency_hz: Sample,
    /// Tone-layer `-60 dB` decay time in seconds.
    pub tone_decay_s: Sample,
    /// Noise-layer `-60 dB` decay time in seconds.
    pub noise_decay_s: Sample,
    /// Band-pass centre for the noise layer, in hertz.
    pub noise_cutoff_hz: Sample,
    /// Band-pass resonance for the noise layer.
    pub noise_q: Sample,
    /// Tone/noise crossfade in `[0, 1]` (`0` tone, `1` noise).
    pub tone_noise_mix: Sample,
    /// Saturator drive (`1.0` nearly linear, higher adds harmonics).
    pub drive: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for AnalogSnareParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            tone_decay_s: DEFAULT_TONE_DECAY_S,
            noise_decay_s: DEFAULT_NOISE_DECAY_S,
            noise_cutoff_hz: DEFAULT_NOISE_CUTOFF_HZ,
            noise_q: DEFAULT_NOISE_Q,
            tone_noise_mix: DEFAULT_TONE_NOISE_MIX,
            drive: DEFAULT_DRIVE,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl AnalogSnareParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range. Frequency and the band-pass centre are clamped against the
    /// Nyquist limit too.
    #[must_use]
    pub fn sanitised(self, sample_rate: u32) -> Self {
        let d = Self::default();
        let frequency_hz =
            clamp_frequency(finite_or(self.frequency_hz, d.frequency_hz), sample_rate);
        let tone_decay_s =
            finite_or(self.tone_decay_s, d.tone_decay_s).clamp(MIN_TONE_DECAY_S, MAX_TONE_DECAY_S);
        let noise_decay_s = finite_or(self.noise_decay_s, d.noise_decay_s)
            .clamp(MIN_NOISE_DECAY_S, MAX_NOISE_DECAY_S);
        let noise_cutoff_hz =
            clamp_cutoff(finite_or(self.noise_cutoff_hz, d.noise_cutoff_hz), sample_rate);
        let noise_q = finite_or(self.noise_q, d.noise_q).clamp(MIN_NOISE_Q, MAX_NOISE_Q);
        let tone_noise_mix = finite_or(self.tone_noise_mix, d.tone_noise_mix).clamp(0.0, 1.0);
        let drive = finite_or(self.drive, d.drive).clamp(MIN_DRIVE, MAX_DRIVE);
        let amplitude = finite_or(self.amplitude, d.amplitude);
        Self {
            frequency_hz,
            tone_decay_s,
            noise_decay_s,
            noise_cutoff_hz,
            noise_q,
            tone_noise_mix,
            drive,
            amplitude,
        }
    }
}

/// Analog-style synthesized snare-drum voice source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::{AnalogSnareNode, AnalogSnareParams};
///
/// let mut node = AnalogSnareNode::new(48_000, AnalogSnareParams::default());
/// node.trigger(1.0);
/// // The trigger injects a fresh enveloped voice, so output is non-silent.
/// ```
#[derive(Clone, Debug)]
pub struct AnalogSnareNode {
    sample_rate: u32,
    frequency_hz: Sample,
    tone_decay_s: Sample,
    noise_decay_s: Sample,
    noise_cutoff_hz: Sample,
    noise_q: Sample,
    tone_noise_mix: Sample,
    drive: Smoothed,
    amplitude: Smoothed,
    phase0: Sample,
    phase1: Sample,
    tone_env: Sample,
    noise_env: Sample,
    tone_decay_coeff: Sample,
    noise_decay_coeff: Sample,
    // Band-pass biquad coefficients (`b1` is identically zero and omitted).
    bp_b0: Sample,
    bp_b2: Sample,
    bp_a1: Sample,
    bp_a2: Sample,
    // Band-pass biquad state (direct form I).
    bp_x1: Sample,
    bp_x2: Sample,
    bp_y1: Sample,
    bp_y2: Sample,
    rng: u32,
    velocity: Sample,
}

impl AnalogSnareNode {
    /// Builds an analog snare voice for `sample_rate` Hz from `params`. The
    /// voice is triggered once at [`DEFAULT_VELOCITY`] so a freshly built node
    /// renders audible output immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: AnalogSnareParams) -> Self {
        let p = params.sanitised(sample_rate);
        let mut node = Self {
            sample_rate: sample_rate.max(1),
            frequency_hz: p.frequency_hz,
            tone_decay_s: p.tone_decay_s,
            noise_decay_s: p.noise_decay_s,
            noise_cutoff_hz: p.noise_cutoff_hz,
            noise_q: p.noise_q,
            tone_noise_mix: p.tone_noise_mix,
            drive: Smoothed::new(p.drive),
            amplitude: Smoothed::new(p.amplitude),
            phase0: 0.0,
            phase1: 0.0,
            tone_env: 0.0,
            noise_env: 0.0,
            tone_decay_coeff: 0.0,
            noise_decay_coeff: 0.0,
            bp_b0: 0.0,
            bp_b2: 0.0,
            bp_a1: 0.0,
            bp_a2: 0.0,
            bp_x1: 0.0,
            bp_x2: 0.0,
            bp_y1: 0.0,
            bp_y2: 0.0,
            rng: NOISE_SEED,
            velocity: DEFAULT_VELOCITY,
        };
        node.recompute();
        node.trigger(DEFAULT_VELOCITY);
        node
    }

    /// Builds a snare voice directly from a [`AnalogSnareParams`] value.
    #[must_use]
    pub fn from_params(sample_rate: u32, params: AnalogSnareParams) -> Self {
        Self::new(sample_rate, params)
    }

    /// Returns the shell fundamental in hertz.
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the tone-layer `-60 dB` decay time in seconds.
    #[must_use]
    pub fn tone_decay_s(&self) -> Sample {
        self.tone_decay_s
    }

    /// Returns the noise-layer `-60 dB` decay time in seconds.
    #[must_use]
    pub fn noise_decay_s(&self) -> Sample {
        self.noise_decay_s
    }

    /// Returns the band-pass centre for the noise layer, in hertz.
    #[must_use]
    pub fn noise_cutoff_hz(&self) -> Sample {
        self.noise_cutoff_hz
    }

    /// Returns the band-pass resonance for the noise layer.
    #[must_use]
    pub fn noise_q(&self) -> Sample {
        self.noise_q
    }

    /// Returns the tone/noise crossfade in `[0, 1]`.
    #[must_use]
    pub fn tone_noise_mix(&self) -> Sample {
        self.tone_noise_mix
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

    /// Sets the shell fundamental, clamped to the tunable range. Takes effect
    /// from the next sample and is click-free (it only shifts the partial
    /// frequencies).
    pub fn set_frequency(&mut self, frequency_hz: Sample) {
        self.frequency_hz =
            clamp_frequency(finite_or(frequency_hz, self.frequency_hz), self.sample_rate);
    }

    /// Sets the tone-layer `-60 dB` decay time, clamped to its range.
    pub fn set_tone_decay_s(&mut self, tone_decay_s: Sample) {
        self.tone_decay_s =
            finite_or(tone_decay_s, self.tone_decay_s).clamp(MIN_TONE_DECAY_S, MAX_TONE_DECAY_S);
        self.recompute();
    }

    /// Sets the noise-layer `-60 dB` decay time, clamped to its range.
    pub fn set_noise_decay_s(&mut self, noise_decay_s: Sample) {
        self.noise_decay_s = finite_or(noise_decay_s, self.noise_decay_s)
            .clamp(MIN_NOISE_DECAY_S, MAX_NOISE_DECAY_S);
        self.recompute();
    }

    /// Sets the band-pass centre for the noise layer, clamped to its range.
    pub fn set_noise_cutoff_hz(&mut self, noise_cutoff_hz: Sample) {
        self.noise_cutoff_hz =
            clamp_cutoff(finite_or(noise_cutoff_hz, self.noise_cutoff_hz), self.sample_rate);
        self.recompute();
    }

    /// Sets the band-pass resonance for the noise layer, clamped to its range.
    pub fn set_noise_q(&mut self, noise_q: Sample) {
        self.noise_q = finite_or(noise_q, self.noise_q).clamp(MIN_NOISE_Q, MAX_NOISE_Q);
        self.recompute();
    }

    /// Sets the tone/noise crossfade, clamped to `[0, 1]`.
    pub fn set_tone_noise_mix(&mut self, tone_noise_mix: Sample) {
        self.tone_noise_mix = finite_or(tone_noise_mix, self.tone_noise_mix).clamp(0.0, 1.0);
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
    /// restarting both envelopes, both oscillator phases, the band-pass state,
    /// and the deterministic noise generator.
    pub fn trigger(&mut self, velocity: Sample) {
        self.velocity = finite_or(velocity, DEFAULT_VELOCITY).clamp(0.0, 1.0);
        self.phase0 = 0.0;
        self.phase1 = 0.0;
        self.tone_env = self.velocity;
        self.noise_env = self.velocity;
        self.bp_x1 = 0.0;
        self.bp_x2 = 0.0;
        self.bp_y1 = 0.0;
        self.bp_y2 = 0.0;
        self.rng = NOISE_SEED;
    }

    /// Recomputes the per-sample envelope decay coefficients and the band-pass
    /// biquad coefficients from the current scalar parameters. Never runs on
    /// the audio hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        self.tone_decay_coeff = ops::exp(-LN_1000 / (self.tone_decay_s * sr));
        self.noise_decay_coeff = ops::exp(-LN_1000 / (self.noise_decay_s * sr));

        // RBJ constant-0 dB-peak-gain band-pass biquad, normalised by a0.
        let w0 = TAU * self.noise_cutoff_hz / sr;
        let cos_w0 = ops::cos(w0);
        let alpha = ops::sin(w0) / (2.0 * self.noise_q);
        let a0 = 1.0 + alpha;
        let inv_a0 = 1.0 / a0;
        self.bp_b0 = alpha * inv_a0;
        self.bp_b2 = -alpha * inv_a0;
        self.bp_a1 = (-2.0 * cos_w0) * inv_a0;
        self.bp_a2 = (1.0 - alpha) * inv_a0;
    }

    /// Draws one deterministic white-noise sample in `[-1, 1)` from the
    /// `xorshift32` generator.
    #[inline]
    fn next_white(&mut self) -> Sample {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x as Sample / u32::MAX as Sample) * 2.0 - 1.0
    }

    /// Renders one mono output sample, advancing the oscillators, envelopes,
    /// band-pass filter, and noise generator by one step.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let sr = self.sample_rate.max(1) as Sample;
        let nyquist = sr * NYQUIST_GUARD;

        // Two tuned shell partials.
        let f0 = self.frequency_hz.min(nyquist);
        let f1 = (self.frequency_hz * TONE_PARTIAL_RATIO).min(nyquist);
        let s0 = ops::sin(self.phase0);
        let s1 = ops::sin(self.phase1);
        self.phase0 += TAU * f0 / sr;
        if self.phase0 >= TAU {
            self.phase0 -= TAU;
        }
        self.phase1 += TAU * f1 / sr;
        if self.phase1 >= TAU {
            self.phase1 -= TAU;
        }
        let tone = 0.5 * (s0 + s1) * self.tone_env;

        // Band-passed deterministic noise (direct form I biquad).
        let x0 = self.next_white();
        let y0 = self.bp_b0 * x0 + self.bp_b2 * self.bp_x2
            - self.bp_a1 * self.bp_y1
            - self.bp_a2 * self.bp_y2;
        self.bp_x2 = self.bp_x1;
        self.bp_x1 = x0;
        self.bp_y2 = self.bp_y1;
        self.bp_y1 = flush_denormal(y0);
        let noise = y0 * self.noise_env;

        let mix = self.tone_noise_mix;
        let excitation = (1.0 - mix) * tone + mix * noise;

        // Advance both envelopes for the next sample.
        self.tone_env = flush_denormal(self.tone_env * self.tone_decay_coeff);
        self.noise_env = flush_denormal(self.noise_env * self.noise_decay_coeff);

        // Saturate before the output gain so level is a strict square law in
        // `amplitude` and the voice can never exceed full scale.
        let drive = self.drive.next_sample();
        let shaped = ops::tanh(drive * excitation) / ops::tanh(drive);
        let amp = self.amplitude.next_sample();
        flush_denormal(amp * shaped)
    }
}

impl AudioNode for AnalogSnareNode {
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

    const SR: u32 = 48_000;

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    fn render_layout(
        node: &mut AnalogSnareNode,
        sample_rate: u32,
        frames: usize,
        layout: ChannelLayout,
    ) -> AudioBuffer {
        let mut out = AudioBuffer::new(layout, frames.max(1));
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    fn render(node: &mut AnalogSnareNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn peak(buf: &AudioBuffer) -> Sample {
        buf.channel(0).iter().fold(0.0, |m, s| m.max(s.abs()))
    }

    fn energy(buf: &AudioBuffer) -> Sample {
        buf.channel(0).iter().map(|s| s * s).sum()
    }

    /// Sum of squared first differences: a monotone proxy for high-frequency
    /// energy.
    fn hf_energy(buf: &AudioBuffer) -> Sample {
        buf.channel(0)
            .windows(2)
            .map(|w| (w[1] - w[0]) * (w[1] - w[0]))
            .sum()
    }

    #[test]
    fn default_params_in_domain() {
        let d = AnalogSnareParams::default();
        assert!(d.frequency_hz >= MIN_FREQUENCY_HZ && d.frequency_hz <= MAX_FREQUENCY_HZ);
        assert!(d.tone_decay_s >= MIN_TONE_DECAY_S && d.tone_decay_s <= MAX_TONE_DECAY_S);
        assert!(d.noise_decay_s >= MIN_NOISE_DECAY_S && d.noise_decay_s <= MAX_NOISE_DECAY_S);
        assert!(d.noise_cutoff_hz >= MIN_NOISE_CUTOFF_HZ && d.noise_cutoff_hz <= MAX_NOISE_CUTOFF_HZ);
        assert!(d.noise_q >= MIN_NOISE_Q && d.noise_q <= MAX_NOISE_Q);
        assert!(d.tone_noise_mix >= 0.0 && d.tone_noise_mix <= 1.0);
        assert!(d.drive >= MIN_DRIVE && d.drive <= MAX_DRIVE);
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let params = AnalogSnareParams {
            frequency_hz: 5.0,
            tone_decay_s: 100.0,
            noise_decay_s: 0.0,
            noise_cutoff_hz: 1.0e9,
            noise_q: 1000.0,
            tone_noise_mix: 5.0,
            drive: 0.0,
            amplitude: 0.5,
        };
        let node = AnalogSnareNode::new(SR, params);
        assert_eq!(node.frequency_hz(), MIN_FREQUENCY_HZ);
        assert_eq!(node.tone_decay_s(), MAX_TONE_DECAY_S);
        assert_eq!(node.noise_decay_s(), MIN_NOISE_DECAY_S);
        assert_eq!(node.noise_q(), MAX_NOISE_Q);
        assert_eq!(node.tone_noise_mix(), 1.0);
        assert_eq!(node.drive(), MIN_DRIVE);
        assert!(node.noise_cutoff_hz() <= SR as Sample * NYQUIST_GUARD);
    }

    #[test]
    fn from_params_matches_new() {
        let params = AnalogSnareParams::default();
        let mut a = AnalogSnareNode::new(SR, params);
        let mut b = AnalogSnareNode::from_params(SR, params);
        let oa = render(&mut a, SR, 1_024);
        let ob = render(&mut b, SR, 1_024);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn getters_report_state() {
        let node = AnalogSnareNode::new(SR, AnalogSnareParams::default());
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.tone_decay_s(), DEFAULT_TONE_DECAY_S);
        assert_eq!(node.noise_decay_s(), DEFAULT_NOISE_DECAY_S);
        assert_eq!(node.noise_cutoff_hz(), DEFAULT_NOISE_CUTOFF_HZ);
        assert_eq!(node.noise_q(), DEFAULT_NOISE_Q);
        assert_eq!(node.tone_noise_mix(), DEFAULT_TONE_NOISE_MIX);
        assert_eq!(node.drive(), DEFAULT_DRIVE);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn latency_is_zero() {
        let node = AnalogSnareNode::new(SR, AnalogSnareParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn renders_bounded_finite() {
        for &freq in &[80.0, 180.0, 500.0] {
            for &q in &[0.3, 0.7, 8.0] {
                for &mix in &[0.0, 0.5, 1.0] {
                    let params = AnalogSnareParams {
                        frequency_hz: freq,
                        noise_q: q,
                        tone_noise_mix: mix,
                        ..AnalogSnareParams::default()
                    };
                    let mut node = AnalogSnareNode::new(SR, params);
                    let out = render(&mut node, SR, 8_192);
                    for &s in out.channel(0) {
                        assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
                    }
                }
            }
        }
    }

    #[test]
    fn not_silent_with_default_params() {
        let mut node = AnalogSnareNode::new(SR, AnalogSnareParams::default());
        let out = render(&mut node, SR, 4_096);
        assert!(peak(&out) > 1e-2, "peak={}", peak(&out));
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let params = AnalogSnareParams {
            amplitude: 0.0,
            ..AnalogSnareParams::default()
        };
        let mut node = AnalogSnareNode::new(SR, params);
        let out = render(&mut node, SR, 2_048);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = AnalogSnareNode::new(SR, AnalogSnareParams::default());
        let mut b = AnalogSnareNode::new(SR, AnalogSnareParams::default());
        let oa = render(&mut a, SR, 4_096);
        let ob = render(&mut b, SR, 4_096);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = AnalogSnareNode::new(SR, AnalogSnareParams::default());
        let first = render(&mut node, SR, 4_096).channel(0).to_vec();
        node.reset();
        let second = render(&mut node, SR, 4_096).channel(0).to_vec();
        assert_eq!(first, second);
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono = AnalogSnareNode::new(SR, AnalogSnareParams::default());
        let mono_out = render(&mut mono, SR, 2_048);
        let mut stereo = AnalogSnareNode::new(SR, AnalogSnareParams::default());
        let stereo_out = render_layout(&mut stereo, SR, 2_048, ChannelLayout::Stereo);
        let mut quad = AnalogSnareNode::new(SR, AnalogSnareParams::default());
        let quad_out = render_layout(&mut quad, SR, 2_048, ChannelLayout::Quad);
        for ch in 0..stereo_out.channels() {
            assert_eq!(stereo_out.channel(ch), mono_out.channel(0));
        }
        for ch in 0..quad_out.channels() {
            assert_eq!(quad_out.channel(ch), mono_out.channel(0));
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = AnalogSnareNode::new(SR, AnalogSnareParams::default());
        let before = node.clone();
        let _ = render(&mut node, SR, 0);
        let a = render(&mut node, SR, 512).channel(0).to_vec();
        let mut fresh = before;
        let b = render(&mut fresh, SR, 512).channel(0).to_vec();
        assert_eq!(a, b);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let params = AnalogSnareParams {
            frequency_hz: Sample::NAN,
            tone_decay_s: Sample::INFINITY,
            noise_decay_s: Sample::NAN,
            noise_cutoff_hz: Sample::INFINITY,
            noise_q: Sample::NAN,
            tone_noise_mix: Sample::NAN,
            drive: Sample::INFINITY,
            amplitude: Sample::NAN,
        };
        let node = AnalogSnareNode::new(SR, params);
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.tone_decay_s(), DEFAULT_TONE_DECAY_S);
        assert_eq!(node.noise_decay_s(), DEFAULT_NOISE_DECAY_S);
        assert_eq!(node.noise_cutoff_hz(), DEFAULT_NOISE_CUTOFF_HZ);
        assert_eq!(node.noise_q(), DEFAULT_NOISE_Q);
        assert_eq!(node.tone_noise_mix(), DEFAULT_TONE_NOISE_MIX);
        assert_eq!(node.drive(), DEFAULT_DRIVE);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = AnalogSnareNode::new(SR, AnalogSnareParams::default());

        node.set_frequency(Sample::NAN);
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        node.set_frequency(1.0e9);
        assert!(node.frequency_hz() <= MAX_FREQUENCY_HZ);

        node.set_noise_q(Sample::INFINITY);
        assert_eq!(node.noise_q(), DEFAULT_NOISE_Q);
        node.set_noise_q(1000.0);
        assert_eq!(node.noise_q(), MAX_NOISE_Q);

        node.set_tone_noise_mix(5.0);
        assert_eq!(node.tone_noise_mix(), 1.0);
        node.set_tone_noise_mix(Sample::NAN);
        assert_eq!(node.tone_noise_mix(), 1.0);

        node.set_drive(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.drive(), DEFAULT_DRIVE);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn trigger_retriggers_envelope() {
        let mut node = AnalogSnareNode::new(SR, AnalogSnareParams::default());
        // Let the voice decay away.
        let _ = render(&mut node, SR, 48_000);
        let quiet = render(&mut node, SR, 256);
        assert!(peak(&quiet) < 1e-2, "quiet peak={}", peak(&quiet));
        node.trigger(1.0);
        let loud = render(&mut node, SR, 256);
        assert!(peak(&loud) > peak(&quiet) * 10.0, "loud peak={}", peak(&loud));
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let quiet_params = AnalogSnareParams {
            amplitude: 0.25,
            ..AnalogSnareParams::default()
        };
        let loud_params = AnalogSnareParams {
            amplitude: 0.5,
            ..AnalogSnareParams::default()
        };
        let mut quiet = AnalogSnareNode::new(SR, quiet_params);
        let mut loud = AnalogSnareNode::new(SR, loud_params);
        let eq = energy(&render(&mut quiet, SR, 8_192));
        let el = energy(&render(&mut loud, SR, 8_192));
        let ratio = el / eq;
        assert!((ratio - 4.0).abs() < 1e-2, "ratio={ratio}");
    }

    #[test]
    fn mix_changes_alter_output() {
        let tone_params = AnalogSnareParams {
            tone_noise_mix: 0.0,
            ..AnalogSnareParams::default()
        };
        let noise_params = AnalogSnareParams {
            tone_noise_mix: 1.0,
            ..AnalogSnareParams::default()
        };
        let mut tone = AnalogSnareNode::new(SR, tone_params);
        let mut noise = AnalogSnareNode::new(SR, noise_params);
        let a = render(&mut tone, SR, 4_096);
        let b = render(&mut noise, SR, 4_096);
        let differing = a
            .channel(0)
            .iter()
            .zip(b.channel(0))
            .filter(|(x, y)| (**x - **y).abs() > 1e-3)
            .count();
        assert!(differing * 10 > a.channel(0).len(), "differing={differing}");
    }

    #[test]
    fn noise_layer_is_brighter_than_tone_layer() {
        let tone_params = AnalogSnareParams {
            tone_noise_mix: 0.0,
            ..AnalogSnareParams::default()
        };
        let noise_params = AnalogSnareParams {
            tone_noise_mix: 1.0,
            ..AnalogSnareParams::default()
        };
        let mut tone = AnalogSnareNode::new(SR, tone_params);
        let mut noise = AnalogSnareNode::new(SR, noise_params);
        let ht = hf_energy(&render(&mut tone, SR, 8_192));
        let hn = hf_energy(&render(&mut noise, SR, 8_192));
        assert!(hn > 1.5 * ht, "noise={hn} tone={ht}");
    }

    #[test]
    fn higher_noise_cutoff_increases_high_frequency_energy() {
        let low_params = AnalogSnareParams {
            tone_noise_mix: 1.0,
            noise_cutoff_hz: 800.0,
            ..AnalogSnareParams::default()
        };
        let high_params = AnalogSnareParams {
            tone_noise_mix: 1.0,
            noise_cutoff_hz: 8_000.0,
            ..AnalogSnareParams::default()
        };
        let mut low = AnalogSnareNode::new(SR, low_params);
        let mut high = AnalogSnareNode::new(SR, high_params);
        let hl = hf_energy(&render(&mut low, SR, 8_192));
        let hh = hf_energy(&render(&mut high, SR, 8_192));
        assert!(hh > 1.5 * hl, "high={hh} low={hl}");
    }

    #[test]
    fn noise_layer_is_deterministic() {
        let params = AnalogSnareParams {
            tone_noise_mix: 1.0,
            ..AnalogSnareParams::default()
        };
        let mut a = AnalogSnareNode::new(SR, params);
        let mut b = AnalogSnareNode::new(SR, params);
        let oa = render(&mut a, SR, 4_096);
        let ob = render(&mut b, SR, 4_096);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn shorter_tone_decay_decays_faster() {
        let short_params = AnalogSnareParams {
            tone_noise_mix: 0.0,
            tone_decay_s: 0.02,
            ..AnalogSnareParams::default()
        };
        let long_params = AnalogSnareParams {
            tone_noise_mix: 0.0,
            tone_decay_s: 0.8,
            ..AnalogSnareParams::default()
        };
        let mut short = AnalogSnareNode::new(SR, short_params);
        let mut long = AnalogSnareNode::new(SR, long_params);
        let os = render(&mut short, SR, 24_000);
        let ol = render(&mut long, SR, 24_000);
        let tail_short: Sample = os.channel(0)[12_000..].iter().map(|s| s * s).sum();
        let tail_long: Sample = ol.channel(0)[12_000..].iter().map(|s| s * s).sum();
        assert!(tail_long > tail_short * 10.0, "short={tail_short} long={tail_long}");
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let params = AnalogSnareParams {
            frequency_hz: 200.0,
            tone_decay_s: 0.1,
            noise_decay_s: 0.3,
            noise_cutoff_hz: 4_000.0,
            noise_q: 1.2,
            tone_noise_mix: 0.5,
            drive: 2.0,
            amplitude: 0.7,
        };
        let s = params.sanitised(SR);
        assert_eq!(s, params);
    }

    #[test]
    fn output_is_bounded_under_extreme_settings() {
        let params = AnalogSnareParams {
            frequency_hz: MAX_FREQUENCY_HZ,
            tone_decay_s: MAX_TONE_DECAY_S,
            noise_decay_s: MAX_NOISE_DECAY_S,
            noise_cutoff_hz: MAX_NOISE_CUTOFF_HZ,
            noise_q: MAX_NOISE_Q,
            tone_noise_mix: 1.0,
            drive: MAX_DRIVE,
            amplitude: 1.0,
        };
        let mut node = AnalogSnareNode::new(SR, params);
        let out = render(&mut node, SR, 200_000);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
        }
    }
}
