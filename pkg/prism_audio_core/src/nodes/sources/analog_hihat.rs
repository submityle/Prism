//! Analog-style synthesized hi-hat / cymbal voice source node.
//!
//! [`AnalogHiHatNode`] synthesizes the classic analog hi-hat voice: a bank of
//! six band-limited square oscillators tuned to a mutually inharmonic set of
//! frequencies, summed into a clangorous "metal" tone, then shaped by a
//! band-pass and a high-pass filter into the characteristic bright, sizzling
//! transient. A single exponential VCA envelope sets the decay, so a short
//! decay yields a *closed* hi-hat and a long decay yields an *open* one, and
//! the sum is pushed through a `tanh` saturator before the final amplitude
//! gain. Unlike the modal physical metals ([`bell`](super::bell),
//! [`struck_plate`](super::struck_plate)) which excite a bank of resonant
//! modes, this voice is a direct subtractive synthesis of an inharmonic
//! oscillator cluster, exactly as realized by the six-square-oscillator plus
//! band-pass/high-pass topology of a classic drum machine. It is a *source*
//! (zero inputs, one output): it supplies its own excitation through
//! [`AnalogHiHatNode::trigger`] and never reads its inputs.
//!
//! # Model
//!
//! Six phase accumulators run band-limited square partials at
//! `base_freq_hz * OSC_RATIOS[i]`, where the ratios are the square roots of the
//! first small integers (`1`, `sqrt(2)`, `sqrt(3)`, `sqrt(5)`, `sqrt(7)`, `3`),
//! a principled irrational spread that maximizes inharmonicity and gives the
//! voice its metallic, non-pitched clang. Their mean is a bipolar square
//! cluster:
//!
//! ```text
//!   metal(t) = mean_i square(base * ratio_i * t)
//! ```
//!
//! The cluster is passed through a two-pole band-pass biquad centred at
//! `bandpass_hz` (resonance `bandpass_q`) that carves the body of the metal,
//! then a two-pole high-pass biquad at `highpass_hz` that removes the low end
//! and leaves the bright sizzle, and the result is scaled by an exponential VCA
//! envelope with a `-60 dB` time of `decay_s`:
//!
//! ```text
//!   voice(t) = highpass(bandpass(metal(t))) * exp(-t / tau)
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
//! Construction, [`AnalogHiHatNode::trigger`], and every scalar setter
//! pre-compute all per-voice coefficients (the envelope decay and both
//! biquads), so [`AnalogHiHatNode::process`] performs no allocation, no
//! locking, and cannot panic: it is a pure per-sample state machine. The
//! `drive` and `amplitude` controls are driven through [`Smoothed`] values so
//! that automation never introduces zipper noise, and a trigger is latched to
//! the block boundary so retriggering is click-free. The oscillator bank is a
//! deterministic set of phase accumulators reset on every trigger, so
//! [`AnalogHiHatNode::reset`] restarts the exact same voice and two nodes built
//! identically and triggered identically emit bit-identical streams on every
//! platform.
//!
//! # Provenance
//!
//! The synthesis technique here -- a bank of inharmonic square oscillators
//! summed, band-passed, high-passed, and shaped by a decay envelope -- is
//! classic public-domain analog drum-voice DSP, exemplified by the Roland
//! `TR-808` hi-hat/cymbal circuit (1980). The band-limited square uses a
//! `PolyBLEP` edge correction derived from first principles, the band-pass and
//! high-pass are textbook public-domain `RBJ` biquads (the Audio EQ Cookbook
//! formulae, placed in the public domain by Robert Bristow-Johnson), and the
//! inharmonic ratios are a fresh irrational set derived here. Only the general
//! techniques are reproduced from first principles; no source code or
//! derivative from any audio engine or toolbox (Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Google Resonance Audio, Web Audio, STK) is used.
//!
//! # Relationship
//!
//! Unlike [`bell`](super::bell) and [`struck_plate`](super::struck_plate) --
//! which model a struck metal object as a bank of resonant modal filters -- this
//! node is a direct subtractive voice built from an inharmonic oscillator
//! cluster plus fixed shaping filters, so its character is a bright metallic
//! sizzle rather than a ringing modal spectrum. Unlike
//! [`analog_snare`](super::analog_snare), whose noise layer is a stochastic
//! band-passed burst, this voice is *fully deterministic* and tonal: its sizzle
//! comes from the beating of six inharmonic squares, not from filtered noise.
//! Unlike [`analog_kick`](super::analog_kick), which is a single pitch-swept
//! sine, this voice has no pitch sweep and sums six fixed inharmonic partials.
//! It shares the `PolyBLEP` edge-correction primitive with
//! [`super::oscillator::OscillatorNode`] and [`super::pwm_oscillator`].

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::nodes::sources::oscillator::poly_blep;
use crate::param::{Ramp, Smoothed};

/// Lowest tunable oscillator-bank fundamental in hertz.
pub const MIN_BASE_FREQ_HZ: Sample = 200.0;

/// Highest tunable oscillator-bank fundamental in hertz.
pub const MAX_BASE_FREQ_HZ: Sample = 1_200.0;

/// Default oscillator-bank fundamental in hertz (a typical analog hi-hat).
pub const DEFAULT_BASE_FREQ_HZ: Sample = 325.0;

/// Shortest `-60 dB` decay time in seconds (a tightly closed hi-hat).
pub const MIN_DECAY_S: Sample = 0.01;

/// Longest `-60 dB` decay time in seconds (a wide open hi-hat / cymbal).
pub const MAX_DECAY_S: Sample = 3.0;

/// Default `-60 dB` decay time in seconds.
pub const DEFAULT_DECAY_S: Sample = 0.3;

/// Lowest band-pass centre for the metal body, in hertz.
pub const MIN_BANDPASS_HZ: Sample = 1_000.0;

/// Highest band-pass centre for the metal body, in hertz.
pub const MAX_BANDPASS_HZ: Sample = 16_000.0;

/// Default band-pass centre for the metal body, in hertz.
pub const DEFAULT_BANDPASS_HZ: Sample = 10_000.0;

/// Smallest band-pass resonance (broad metal body).
pub const MIN_BANDPASS_Q: Sample = 0.3;

/// Largest band-pass resonance (narrow, whistling metal body).
pub const MAX_BANDPASS_Q: Sample = 8.0;

/// Default band-pass resonance.
pub const DEFAULT_BANDPASS_Q: Sample = 1.5;

/// Lowest high-pass cutoff that removes the low end, in hertz.
pub const MIN_HIGHPASS_HZ: Sample = 1_000.0;

/// Highest high-pass cutoff, in hertz.
pub const MAX_HIGHPASS_HZ: Sample = 12_000.0;

/// Default high-pass cutoff, in hertz.
pub const DEFAULT_HIGHPASS_HZ: Sample = 7_000.0;

/// Smallest saturator drive (`1.0` is nearly linear).
pub const MIN_DRIVE: Sample = 1.0;

/// Largest saturator drive (heavy harmonic fattening).
pub const MAX_DRIVE: Sample = 12.0;

/// Default saturator drive.
pub const DEFAULT_DRIVE: Sample = 1.5;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.7;

/// Default strike velocity used by [`AnalogHiHatNode::trigger`].
pub const DEFAULT_VELOCITY: Sample = 1.0;

/// Number of square oscillators in the inharmonic metal bank.
pub const NUM_OSC: usize = 6;

/// Inharmonic frequency ratios of the oscillator bank relative to the
/// fundamental: `1`, `sqrt(2)`, `sqrt(3)`, `sqrt(5)`, `sqrt(7)`, `3`. The
/// irrational spread maximizes inharmonicity, giving the bank its metallic
/// (non-pitched) clang.
const OSC_RATIOS: [Sample; NUM_OSC] = [
    1.0,
    core::f32::consts::SQRT_2,
    1.732_050_8,
    2.236_068,
    2.645_751_3,
    3.0,
];

/// Reciprocal of [`NUM_OSC`], used to average the bank without a per-sample
/// division.
const INV_NUM_OSC: Sample = 1.0 / NUM_OSC as Sample;

/// Fixed Butterworth resonance for the high-pass stage.
const HIGHPASS_Q: Sample = 0.707_106_77;

/// `ln(1000) == 3 * ln(10)`, used by the `-60 dB`-time-to-decay mapping.
const LN_1000: Sample = 6.907_755;

/// Fraction of the sample rate above which a partial is muted (anti-alias).
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

/// Clamps the oscillator-bank fundamental to the tunable range.
#[inline]
fn clamp_base_freq(freq_hz: Sample) -> Sample {
    freq_hz.clamp(MIN_BASE_FREQ_HZ, MAX_BASE_FREQ_HZ)
}

/// Clamps a filter cutoff to `[min, max]` and the Nyquist guard.
#[inline]
fn clamp_cutoff(cutoff_hz: Sample, min: Sample, max: Sample, sample_rate: u32) -> Sample {
    let nyquist = sample_rate.max(1) as Sample * NYQUIST_GUARD;
    let upper = max.min(nyquist).max(min);
    cutoff_hz.clamp(min, upper)
}

/// Band-limited bipolar square of unit period for phase `t` with per-sample
/// phase increment `dt`.
///
/// The naive square is `+1` over `[0, 0.5)` and `-1` over `[0.5, 1)`; both the
/// rising edge at `0` and the falling edge at `0.5` are rounded with a
/// `PolyBLEP` correction so the band-limited result does not alias.
#[inline]
fn band_limited_square(t: Sample, dt: Sample) -> Sample {
    let mut value = if t < 0.5 { 1.0 } else { -1.0 };
    value += poly_blep(t, dt);
    let mut t_fall = t + 0.5;
    if t_fall >= 1.0 {
        t_fall -= 1.0;
    }
    value -= poly_blep(t_fall, dt);
    value
}

/// Construction parameters for an [`AnalogHiHatNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AnalogHiHatParams {
    /// Oscillator-bank fundamental frequency in hertz.
    pub base_freq_hz: Sample,
    /// `-60 dB` decay time in seconds (short = closed, long = open).
    pub decay_s: Sample,
    /// Band-pass centre for the metal body, in hertz.
    pub bandpass_hz: Sample,
    /// Band-pass resonance for the metal body.
    pub bandpass_q: Sample,
    /// High-pass cutoff that removes the low end, in hertz.
    pub highpass_hz: Sample,
    /// Saturator drive (`1.0` nearly linear, higher adds harmonics).
    pub drive: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for AnalogHiHatParams {
    fn default() -> Self {
        Self {
            base_freq_hz: DEFAULT_BASE_FREQ_HZ,
            decay_s: DEFAULT_DECAY_S,
            bandpass_hz: DEFAULT_BANDPASS_HZ,
            bandpass_q: DEFAULT_BANDPASS_Q,
            highpass_hz: DEFAULT_HIGHPASS_HZ,
            drive: DEFAULT_DRIVE,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl AnalogHiHatParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range. The filter cutoffs are clamped against the Nyquist limit
    /// too.
    #[must_use]
    pub fn sanitised(self, sample_rate: u32) -> Self {
        let d = Self::default();
        let base_freq_hz = clamp_base_freq(finite_or(self.base_freq_hz, d.base_freq_hz));
        let decay_s = finite_or(self.decay_s, d.decay_s).clamp(MIN_DECAY_S, MAX_DECAY_S);
        let bandpass_hz = clamp_cutoff(
            finite_or(self.bandpass_hz, d.bandpass_hz),
            MIN_BANDPASS_HZ,
            MAX_BANDPASS_HZ,
            sample_rate,
        );
        let bandpass_q =
            finite_or(self.bandpass_q, d.bandpass_q).clamp(MIN_BANDPASS_Q, MAX_BANDPASS_Q);
        let highpass_hz = clamp_cutoff(
            finite_or(self.highpass_hz, d.highpass_hz),
            MIN_HIGHPASS_HZ,
            MAX_HIGHPASS_HZ,
            sample_rate,
        );
        let drive = finite_or(self.drive, d.drive).clamp(MIN_DRIVE, MAX_DRIVE);
        let amplitude = finite_or(self.amplitude, d.amplitude);
        Self {
            base_freq_hz,
            decay_s,
            bandpass_hz,
            bandpass_q,
            highpass_hz,
            drive,
            amplitude,
        }
    }
}

/// Analog-style synthesized hi-hat / cymbal voice source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::{AnalogHiHatNode, AnalogHiHatParams};
///
/// let mut node = AnalogHiHatNode::new(48_000, AnalogHiHatParams::default());
/// node.trigger(1.0);
/// // The trigger injects a fresh enveloped voice, so output is non-silent.
/// ```
#[derive(Clone, Debug)]
pub struct AnalogHiHatNode {
    sample_rate: u32,
    base_freq_hz: Sample,
    decay_s: Sample,
    bandpass_hz: Sample,
    bandpass_q: Sample,
    highpass_hz: Sample,
    drive: Smoothed,
    amplitude: Smoothed,
    phase: [Sample; NUM_OSC],
    env: Sample,
    decay_coeff: Sample,
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
    // High-pass biquad coefficients.
    hp_b0: Sample,
    hp_b1: Sample,
    hp_b2: Sample,
    hp_a1: Sample,
    hp_a2: Sample,
    // High-pass biquad state (direct form I).
    hp_x1: Sample,
    hp_x2: Sample,
    hp_y1: Sample,
    hp_y2: Sample,
    velocity: Sample,
}

impl AnalogHiHatNode {
    /// Builds an analog hi-hat voice for `sample_rate` Hz from `params`. The
    /// voice is triggered once at [`DEFAULT_VELOCITY`] so a freshly built node
    /// renders audible output immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: AnalogHiHatParams) -> Self {
        let p = params.sanitised(sample_rate);
        let mut node = Self {
            sample_rate: sample_rate.max(1),
            base_freq_hz: p.base_freq_hz,
            decay_s: p.decay_s,
            bandpass_hz: p.bandpass_hz,
            bandpass_q: p.bandpass_q,
            highpass_hz: p.highpass_hz,
            drive: Smoothed::new(p.drive),
            amplitude: Smoothed::new(p.amplitude),
            phase: [0.0; NUM_OSC],
            env: 0.0,
            decay_coeff: 0.0,
            bp_b0: 0.0,
            bp_b2: 0.0,
            bp_a1: 0.0,
            bp_a2: 0.0,
            bp_x1: 0.0,
            bp_x2: 0.0,
            bp_y1: 0.0,
            bp_y2: 0.0,
            hp_b0: 0.0,
            hp_b1: 0.0,
            hp_b2: 0.0,
            hp_a1: 0.0,
            hp_a2: 0.0,
            hp_x1: 0.0,
            hp_x2: 0.0,
            hp_y1: 0.0,
            hp_y2: 0.0,
            velocity: DEFAULT_VELOCITY,
        };
        node.recompute();
        node.trigger(DEFAULT_VELOCITY);
        node
    }

    /// Builds a hi-hat voice directly from an [`AnalogHiHatParams`] value.
    #[must_use]
    pub fn from_params(sample_rate: u32, params: AnalogHiHatParams) -> Self {
        Self::new(sample_rate, params)
    }

    /// Returns the oscillator-bank fundamental in hertz.
    #[must_use]
    pub fn base_freq_hz(&self) -> Sample {
        self.base_freq_hz
    }

    /// Returns the `-60 dB` decay time in seconds.
    #[must_use]
    pub fn decay_s(&self) -> Sample {
        self.decay_s
    }

    /// Returns the band-pass centre for the metal body, in hertz.
    #[must_use]
    pub fn bandpass_hz(&self) -> Sample {
        self.bandpass_hz
    }

    /// Returns the band-pass resonance for the metal body.
    #[must_use]
    pub fn bandpass_q(&self) -> Sample {
        self.bandpass_q
    }

    /// Returns the high-pass cutoff, in hertz.
    #[must_use]
    pub fn highpass_hz(&self) -> Sample {
        self.highpass_hz
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

    /// Sets the oscillator-bank fundamental, clamped to the tunable range.
    /// Takes effect from the next sample and is click-free (it only shifts the
    /// partial frequencies).
    pub fn set_base_freq_hz(&mut self, base_freq_hz: Sample) {
        self.base_freq_hz = clamp_base_freq(finite_or(base_freq_hz, self.base_freq_hz));
    }

    /// Sets the `-60 dB` decay time, clamped to its range.
    pub fn set_decay_s(&mut self, decay_s: Sample) {
        self.decay_s = finite_or(decay_s, self.decay_s).clamp(MIN_DECAY_S, MAX_DECAY_S);
        self.recompute();
    }

    /// Sets the band-pass centre for the metal body, clamped to its range.
    pub fn set_bandpass_hz(&mut self, bandpass_hz: Sample) {
        self.bandpass_hz = clamp_cutoff(
            finite_or(bandpass_hz, self.bandpass_hz),
            MIN_BANDPASS_HZ,
            MAX_BANDPASS_HZ,
            self.sample_rate,
        );
        self.recompute();
    }

    /// Sets the band-pass resonance for the metal body, clamped to its range.
    pub fn set_bandpass_q(&mut self, bandpass_q: Sample) {
        self.bandpass_q =
            finite_or(bandpass_q, self.bandpass_q).clamp(MIN_BANDPASS_Q, MAX_BANDPASS_Q);
        self.recompute();
    }

    /// Sets the high-pass cutoff, clamped to its range.
    pub fn set_highpass_hz(&mut self, highpass_hz: Sample) {
        self.highpass_hz = clamp_cutoff(
            finite_or(highpass_hz, self.highpass_hz),
            MIN_HIGHPASS_HZ,
            MAX_HIGHPASS_HZ,
            self.sample_rate,
        );
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
    /// restarting the envelope, every oscillator phase, and both biquad states.
    pub fn trigger(&mut self, velocity: Sample) {
        self.velocity = finite_or(velocity, DEFAULT_VELOCITY).clamp(0.0, 1.0);
        self.phase = [0.0; NUM_OSC];
        self.env = self.velocity;
        self.bp_x1 = 0.0;
        self.bp_x2 = 0.0;
        self.bp_y1 = 0.0;
        self.bp_y2 = 0.0;
        self.hp_x1 = 0.0;
        self.hp_x2 = 0.0;
        self.hp_y1 = 0.0;
        self.hp_y2 = 0.0;
    }

    /// Recomputes the per-sample envelope decay coefficient and both biquad
    /// coefficients from the current scalar parameters. Never runs on the audio
    /// hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        self.decay_coeff = ops::exp(-LN_1000 / (self.decay_s * sr));

        // RBJ constant-0 dB-peak-gain band-pass biquad, normalised by a0.
        let w0 = TAU * self.bandpass_hz / sr;
        let cos_w0 = ops::cos(w0);
        let alpha = ops::sin(w0) / (2.0 * self.bandpass_q);
        let a0 = 1.0 + alpha;
        let inv_a0 = 1.0 / a0;
        self.bp_b0 = alpha * inv_a0;
        self.bp_b2 = -alpha * inv_a0;
        self.bp_a1 = (-2.0 * cos_w0) * inv_a0;
        self.bp_a2 = (1.0 - alpha) * inv_a0;

        // RBJ high-pass biquad, normalised by a0.
        let hw0 = TAU * self.highpass_hz / sr;
        let hcos = ops::cos(hw0);
        let halpha = ops::sin(hw0) / (2.0 * HIGHPASS_Q);
        let ha0 = 1.0 + halpha;
        let hinv = 1.0 / ha0;
        self.hp_b0 = ((1.0 + hcos) * 0.5) * hinv;
        self.hp_b1 = -(1.0 + hcos) * hinv;
        self.hp_b2 = ((1.0 + hcos) * 0.5) * hinv;
        self.hp_a1 = (-2.0 * hcos) * hinv;
        self.hp_a2 = (1.0 - halpha) * hinv;
    }

    /// Renders one mono output sample, advancing the oscillator bank, envelope,
    /// and both filters by one step.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let sr = self.sample_rate.max(1) as Sample;
        let nyquist = sr * NYQUIST_GUARD;

        // Inharmonic band-limited square cluster.
        let mut metal = 0.0;
        for (i, &ratio) in OSC_RATIOS.iter().enumerate() {
            let f = (self.base_freq_hz * ratio).min(nyquist);
            let dt = f / sr;
            metal += band_limited_square(self.phase[i], dt);
            let mut p = self.phase[i] + dt;
            if p >= 1.0 {
                p -= 1.0;
            }
            self.phase[i] = p;
        }
        metal *= INV_NUM_OSC;

        // Band-pass the metal body (direct form I biquad).
        let bp = self.bp_b0 * metal + self.bp_b2 * self.bp_x2
            - self.bp_a1 * self.bp_y1
            - self.bp_a2 * self.bp_y2;
        self.bp_x2 = self.bp_x1;
        self.bp_x1 = metal;
        self.bp_y2 = self.bp_y1;
        self.bp_y1 = flush_denormal(bp);

        // High-pass to leave the bright sizzle (direct form I biquad).
        let hp = self.hp_b0 * bp + self.hp_b1 * self.hp_x1 + self.hp_b2 * self.hp_x2
            - self.hp_a1 * self.hp_y1
            - self.hp_a2 * self.hp_y2;
        self.hp_x2 = self.hp_x1;
        self.hp_x1 = bp;
        self.hp_y2 = self.hp_y1;
        self.hp_y1 = flush_denormal(hp);

        let voice = hp * self.env;

        // Advance the envelope for the next sample.
        self.env = flush_denormal(self.env * self.decay_coeff);

        // Saturate before the output gain so level is a strict square law in
        // `amplitude` and the voice can never exceed full scale.
        let drive = self.drive.next_sample();
        let shaped = ops::tanh(drive * voice) / ops::tanh(drive);
        let amp = self.amplitude.next_sample();
        flush_denormal(amp * shaped)
    }
}

impl AudioNode for AnalogHiHatNode {
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
        node: &mut AnalogHiHatNode,
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

    fn render(node: &mut AnalogHiHatNode, sample_rate: u32, frames: usize) -> AudioBuffer {
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
        let d = AnalogHiHatParams::default();
        assert!(d.base_freq_hz >= MIN_BASE_FREQ_HZ && d.base_freq_hz <= MAX_BASE_FREQ_HZ);
        assert!(d.decay_s >= MIN_DECAY_S && d.decay_s <= MAX_DECAY_S);
        assert!(d.bandpass_hz >= MIN_BANDPASS_HZ && d.bandpass_hz <= MAX_BANDPASS_HZ);
        assert!(d.bandpass_q >= MIN_BANDPASS_Q && d.bandpass_q <= MAX_BANDPASS_Q);
        assert!(d.highpass_hz >= MIN_HIGHPASS_HZ && d.highpass_hz <= MAX_HIGHPASS_HZ);
        assert!(d.drive >= MIN_DRIVE && d.drive <= MAX_DRIVE);
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let params = AnalogHiHatParams {
            base_freq_hz: 5.0,
            decay_s: 100.0,
            bandpass_hz: 1.0e9,
            bandpass_q: 1000.0,
            highpass_hz: 1.0e9,
            drive: 0.0,
            amplitude: 0.5,
        };
        let node = AnalogHiHatNode::new(SR, params);
        assert_eq!(node.base_freq_hz(), MIN_BASE_FREQ_HZ);
        assert_eq!(node.decay_s(), MAX_DECAY_S);
        assert_eq!(node.bandpass_q(), MAX_BANDPASS_Q);
        assert_eq!(node.drive(), MIN_DRIVE);
        assert!(node.bandpass_hz() <= SR as Sample * NYQUIST_GUARD);
        assert!(node.highpass_hz() <= SR as Sample * NYQUIST_GUARD);
    }

    #[test]
    fn from_params_matches_new() {
        let params = AnalogHiHatParams::default();
        let mut a = AnalogHiHatNode::new(SR, params);
        let mut b = AnalogHiHatNode::from_params(SR, params);
        let oa = render(&mut a, SR, 1_024);
        let ob = render(&mut b, SR, 1_024);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn getters_report_state() {
        let node = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
        assert_eq!(node.base_freq_hz(), DEFAULT_BASE_FREQ_HZ);
        assert_eq!(node.decay_s(), DEFAULT_DECAY_S);
        assert_eq!(node.bandpass_hz(), DEFAULT_BANDPASS_HZ);
        assert_eq!(node.bandpass_q(), DEFAULT_BANDPASS_Q);
        assert_eq!(node.highpass_hz(), DEFAULT_HIGHPASS_HZ);
        assert_eq!(node.drive(), DEFAULT_DRIVE);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn latency_is_zero() {
        let node = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn renders_bounded_finite() {
        for &base in &[200.0, 325.0, 1_200.0] {
            for &q in &[0.3, 1.5, 8.0] {
                for &hp in &[1_000.0, 7_000.0, 12_000.0] {
                    let params = AnalogHiHatParams {
                        base_freq_hz: base,
                        bandpass_q: q,
                        highpass_hz: hp,
                        ..AnalogHiHatParams::default()
                    };
                    let mut node = AnalogHiHatNode::new(SR, params);
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
        let mut node = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
        let out = render(&mut node, SR, 4_096);
        assert!(peak(&out) > 1e-2, "peak={}", peak(&out));
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let params = AnalogHiHatParams {
            amplitude: 0.0,
            ..AnalogHiHatParams::default()
        };
        let mut node = AnalogHiHatNode::new(SR, params);
        let out = render(&mut node, SR, 2_048);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
        let mut b = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
        let oa = render(&mut a, SR, 4_096);
        let ob = render(&mut b, SR, 4_096);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
        let first = render(&mut node, SR, 4_096).channel(0).to_vec();
        node.reset();
        let second = render(&mut node, SR, 4_096).channel(0).to_vec();
        assert_eq!(first, second);
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
        let mono_out = render(&mut mono, SR, 2_048);
        let mut stereo = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
        let stereo_out = render_layout(&mut stereo, SR, 2_048, ChannelLayout::Stereo);
        let mut quad = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
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
        let mut node = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
        let before = node.clone();
        let _ = render(&mut node, SR, 0);
        let a = render(&mut node, SR, 512).channel(0).to_vec();
        let mut fresh = before;
        let b = render(&mut fresh, SR, 512).channel(0).to_vec();
        assert_eq!(a, b);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let params = AnalogHiHatParams {
            base_freq_hz: Sample::NAN,
            decay_s: Sample::INFINITY,
            bandpass_hz: Sample::NAN,
            bandpass_q: Sample::INFINITY,
            highpass_hz: Sample::NAN,
            drive: Sample::INFINITY,
            amplitude: Sample::NAN,
        };
        let node = AnalogHiHatNode::new(SR, params);
        assert_eq!(node.base_freq_hz(), DEFAULT_BASE_FREQ_HZ);
        assert_eq!(node.decay_s(), DEFAULT_DECAY_S);
        assert_eq!(node.bandpass_hz(), DEFAULT_BANDPASS_HZ);
        assert_eq!(node.bandpass_q(), DEFAULT_BANDPASS_Q);
        assert_eq!(node.highpass_hz(), DEFAULT_HIGHPASS_HZ);
        assert_eq!(node.drive(), DEFAULT_DRIVE);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());

        node.set_base_freq_hz(Sample::NAN);
        assert_eq!(node.base_freq_hz(), DEFAULT_BASE_FREQ_HZ);
        node.set_base_freq_hz(1.0e9);
        assert!(node.base_freq_hz() <= MAX_BASE_FREQ_HZ);

        node.set_bandpass_q(Sample::INFINITY);
        assert_eq!(node.bandpass_q(), DEFAULT_BANDPASS_Q);
        node.set_bandpass_q(1000.0);
        assert_eq!(node.bandpass_q(), MAX_BANDPASS_Q);

        node.set_highpass_hz(Sample::NAN);
        assert_eq!(node.highpass_hz(), DEFAULT_HIGHPASS_HZ);

        node.set_drive(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.drive(), DEFAULT_DRIVE);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn trigger_retriggers_envelope() {
        let mut node = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
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
        let quiet_params = AnalogHiHatParams {
            amplitude: 0.25,
            ..AnalogHiHatParams::default()
        };
        let loud_params = AnalogHiHatParams {
            amplitude: 0.5,
            ..AnalogHiHatParams::default()
        };
        let mut quiet = AnalogHiHatNode::new(SR, quiet_params);
        let mut loud = AnalogHiHatNode::new(SR, loud_params);
        let eq = energy(&render(&mut quiet, SR, 8_192));
        let el = energy(&render(&mut loud, SR, 8_192));
        let ratio = el / eq;
        assert!((ratio - 4.0).abs() < 1e-2, "ratio={ratio}");
    }

    #[test]
    fn open_decays_slower_than_closed() {
        let closed_params = AnalogHiHatParams {
            decay_s: 0.05,
            ..AnalogHiHatParams::default()
        };
        let open_params = AnalogHiHatParams {
            decay_s: 1.0,
            ..AnalogHiHatParams::default()
        };
        let mut closed = AnalogHiHatNode::new(SR, closed_params);
        let mut open = AnalogHiHatNode::new(SR, open_params);
        let oc = render(&mut closed, SR, 48_000);
        let oo = render(&mut open, SR, 48_000);
        let tail_closed: Sample = oc.channel(0)[24_000..].iter().map(|s| s * s).sum();
        let tail_open: Sample = oo.channel(0)[24_000..].iter().map(|s| s * s).sum();
        assert!(
            tail_open > tail_closed * 10.0,
            "closed={tail_closed} open={tail_open}"
        );
    }

    #[test]
    fn higher_highpass_reduces_low_frequency_energy() {
        let low_params = AnalogHiHatParams {
            highpass_hz: 1_000.0,
            ..AnalogHiHatParams::default()
        };
        let high_params = AnalogHiHatParams {
            highpass_hz: 12_000.0,
            ..AnalogHiHatParams::default()
        };
        let mut low = AnalogHiHatNode::new(SR, low_params);
        let mut high = AnalogHiHatNode::new(SR, high_params);
        // A stronger high-pass keeps a larger fraction of energy in the fast
        // (high-frequency) first differences, so the hf-to-total energy ratio
        // rises.
        let lo = render(&mut low, SR, 8_192);
        let hi = render(&mut high, SR, 8_192);
        let lo_ratio = hf_energy(&lo) / (energy(&lo) + 1e-12);
        let hi_ratio = hf_energy(&hi) / (energy(&hi) + 1e-12);
        assert!(hi_ratio > lo_ratio, "lo_ratio={lo_ratio} hi_ratio={hi_ratio}");
    }

    #[test]
    fn different_base_freq_alters_output() {
        let low_params = AnalogHiHatParams {
            base_freq_hz: 250.0,
            ..AnalogHiHatParams::default()
        };
        let high_params = AnalogHiHatParams {
            base_freq_hz: 900.0,
            ..AnalogHiHatParams::default()
        };
        let mut low = AnalogHiHatNode::new(SR, low_params);
        let mut high = AnalogHiHatNode::new(SR, high_params);
        let a = render(&mut low, SR, 4_096);
        let b = render(&mut high, SR, 4_096);
        let differing = a
            .channel(0)
            .iter()
            .zip(b.channel(0))
            .filter(|(x, y)| (**x - **y).abs() > 1e-3)
            .count();
        assert!(differing * 10 > a.channel(0).len(), "differing={differing}");
    }

    #[test]
    fn is_deterministic_tonal_voice() {
        // No stochastic source: two fresh voices are bit-identical.
        let mut a = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
        let mut b = AnalogHiHatNode::new(SR, AnalogHiHatParams::default());
        let oa = render(&mut a, SR, 16_384);
        let ob = render(&mut b, SR, 16_384);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let params = AnalogHiHatParams {
            base_freq_hz: 400.0,
            decay_s: 0.2,
            bandpass_hz: 9_000.0,
            bandpass_q: 2.0,
            highpass_hz: 6_000.0,
            drive: 2.0,
            amplitude: 0.6,
        };
        let s = params.sanitised(SR);
        assert_eq!(s, params);
    }

    #[test]
    fn output_is_bounded_under_extreme_settings() {
        let params = AnalogHiHatParams {
            base_freq_hz: MAX_BASE_FREQ_HZ,
            decay_s: MAX_DECAY_S,
            bandpass_hz: MAX_BANDPASS_HZ,
            bandpass_q: MAX_BANDPASS_Q,
            highpass_hz: MAX_HIGHPASS_HZ,
            drive: MAX_DRIVE,
            amplitude: 1.0,
        };
        let mut node = AnalogHiHatNode::new(SR, params);
        let out = render(&mut node, SR, 200_000);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
        }
    }

    #[test]
    fn ratios_are_distinct_and_ascending() {
        for i in 1..NUM_OSC {
            assert!(OSC_RATIOS[i] > OSC_RATIOS[i - 1], "ratio {i} not ascending");
        }
    }
}
