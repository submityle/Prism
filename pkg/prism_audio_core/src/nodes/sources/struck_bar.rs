//! Struck-bar modal percussion source (marimba / vibraphone / glockenspiel /
//! tubular-bell family).
//!
//! [`StruckBarNode`] is a *source* (zero inputs, one output) that synthesizes a
//! mallet-struck stiff bar by *modal synthesis*: a short mallet contact force
//! excites a parallel bank of [`NUM_MODES`] independently decaying two-pole
//! resonators, each tuned to one transverse bending partial of the bar. The
//! summed output is the characteristic metallic / wooden "ping" that rings out
//! and dies away after each strike.
//!
//! # Model
//!
//! A stiff free-free bar does not vibrate at a harmonic series: its transverse
//! bending modes sit at the inharmonic ratios of the Euler-Bernoulli beam,
//! `1 : 2.756 : 5.404 : 8.933 : ...` (the squared ratios of the beam's
//! eigenvalues). An idealised metallic bar (glockenspiel, tubular bell) rings
//! at those ratios; a tuned mallet instrument (marimba, vibraphone) is undercut
//! so its first few overtones are pulled toward octave-like ratios
//! (`1 : 4 : 10 : ...`). The `inharmonicity` control linearly blends the mode
//! ratios between the tuned set and the ideal free-free set, so one node spans
//! the wooden-to-metallic continuum.
//!
//! Each mode `m` is a two-pole resonator
//! `y[n] = b0 * x[n] + a1 * y[n-1] + a2 * y[n-2]` whose complex pole pair sits
//! at radius `R = exp(-ln(1000) / (t60 * sample_rate))` and angle
//! `theta = 2*pi*f_m / sample_rate`, giving `a1 = 2*R*cos(theta)` and
//! `a2 = -R*R`. Its impulse response is a sinusoid at `f_m` decaying by
//! `-60 dB` over `t60` seconds. The feed gain `b0 = gain_m * sin(theta)`
//! normalises the ringing peak to `gain_m` independently of the decay radius.
//! Higher partials are given shorter decay (`t60_m = decay / ratio_m^0.7`) and
//! lower gain (`gain_m = ratio_m^-exp`, with `exp` set by `brightness`), the
//! usual spectral envelope of a struck bar.
//!
//! The excitation `x[n]` is a single raised-cosine (Hann) contact-force pulse,
//! normalised to unit area so each strike imparts a fixed momentum regardless
//! of its width. A hard mallet is modelled by a short pulse (bright, lots of
//! high-mode energy); a soft mallet by a long pulse (dull, high modes barely
//! driven). `brightness` sets both the pulse width and the mode-gain rolloff.
//! The node is struck once at construction so it sounds immediately;
//! [`StruckBarNode::strike`] retriggers it, adding a fresh pulse while the
//! existing modes keep ringing.
//!
//! # Determinism
//!
//! The excitation is a closed-form deterministic pulse, not noise, so the node
//! holds no random state: two [`StruckBarNode`]s built with the same sample
//! rate and parameters produce bit-identical output, and [`StruckBarNode::reset`]
//! clears the resonators and re-strikes to replay the identical attack.
//!
//! # Real-time contract
//!
//! All per-mode coefficient and history storage is a fixed-size array sized for
//! [`NUM_MODES`]; [`StruckBarNode::process`] performs no allocation, locking, or
//! panic on the hot path. Non-finite parameters are sanitised on the way in and
//! outputs are flushed of denormals, so the generator cannot stall the audio
//! thread. Latency is zero.
//!
//! # Relationship
//!
//! Unlike the sibling [`modal_resonator`](crate::nodes::effects::modal_resonator)
//! *effect*, which filters an *external* input signal through a modal bank, this
//! *source* supplies its own mallet-strike excitation and needs no input. It
//! differs from the one-dimensional waveguide voices
//! [`karplus_strong`](crate::nodes::sources::karplus_strong),
//! [`bowed_string`](crate::nodes::sources::bowed_string), and
//! [`reed_woodwind`](crate::nodes::sources::reed_woodwind), which model strings
//! and air columns and therefore sound a (near-)harmonic series: this node
//! models the *inharmonic* bending modes of a stiff bar. It also differs from
//! [`additive_oscillator`](crate::nodes::sources::additive_oscillator), a steady
//! harmonic sum with no strike or decay, and from
//! [`impulse_train`](crate::nodes::sources::impulse_train), a raw pulse source
//! with no resonant body.
//!
//! # Provenance
//!
//! Modal synthesis (an object modelled as a parallel bank of independently
//! decaying resonators) is the classic technique described by J.-M. Adrien,
//! "The Missing Link: Modal Synthesis" (in *Representations of Musical Signals*,
//! MIT Press, 1991). The inharmonic free-free bending-mode ratios are the
//! standard Euler-Bernoulli beam partials tabulated in acoustics texts (for
//! example N. H. Fletcher and T. D. Rossing, *The Physics of Musical
//! Instruments*). The two-pole resonator, the `t60`-to-pole-radius mapping, the
//! Hann (raised-cosine) contact pulse, and the mallet-hardness spectral
//! envelope are standard, publicly documented DSP. This is pure classic DSP
//! with no AI or ML. This module contains **no Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Google Resonance Audio, Web Audio, or STK source or
//! derived code**; only the widely documented beam-mode ratios, resonator, and
//! window formulas are used.


use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Number of parallel bending modes the bar is modelled with.
pub const NUM_MODES: usize = 6;

/// Lowest tunable fundamental (strike pitch) in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Highest tunable fundamental in hertz (further bounded by the Nyquist limit).
pub const MAX_FREQUENCY_HZ: Sample = 12_000.0;

/// Default fundamental (strike pitch) frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 440.0;

/// Shortest `-60 dB` decay time, in seconds, the fundamental may request.
pub const MIN_DECAY_S: Sample = 0.02;

/// Longest `-60 dB` decay time, in seconds, the fundamental may request.
pub const MAX_DECAY_S: Sample = 20.0;

/// Default fundamental `-60 dB` decay time in seconds.
pub const DEFAULT_DECAY_S: Sample = 2.0;

/// Default mallet hardness / brightness in `[0, 1]`.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Default inharmonicity in `[0, 1]` (`0` tuned, `1` ideal free-free bar).
pub const DEFAULT_INHARMONICITY: Sample = 1.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Default strike velocity used by [`StruckBarNode::strike`].
pub const DEFAULT_STRIKE_VELOCITY: Sample = 1.0;

/// Ideal free-free (Euler-Bernoulli) bar bending-mode ratios.
const IDEAL_BAR_RATIOS: [Sample; NUM_MODES] =
    [1.0, 2.756, 5.404, 8.933, 13.344, 18.638];

/// Octave-tuned (marimba / vibraphone undercut) bending-mode ratios.
const TUNED_BAR_RATIOS: [Sample; NUM_MODES] = [1.0, 4.0, 10.0, 20.0, 33.0, 50.0];

/// `ln(1000) == 3 * ln(10)`, used by the `t60`-to-pole-radius mapping.
const LN_1000: Sample = 6.907_755;

/// Fraction of the sample rate above which a mode is muted (anti-alias guard).
const NYQUIST_GUARD: Sample = 0.49;

/// Exponent controlling how much faster high modes decay than the fundamental.
const DECAY_RATIO_EXP: Sample = 0.7;

/// Shortest mallet-contact pulse (hardest mallet), in milliseconds.
const PULSE_MS_MIN: Sample = 0.2;

/// Longest mallet-contact pulse (softest mallet), in milliseconds.
const PULSE_MS_MAX: Sample = 4.0;

/// Softest mode-gain rolloff exponent (brightest mallet).
const GAIN_EXP_MIN: Sample = 0.5;

/// Steepest mode-gain rolloff exponent (dullest mallet).
const GAIN_EXP_MAX: Sample = 2.0;

/// Overall output scale, keeping the summed modal peak below full scale.
const OUTPUT_GAIN: Sample = 0.8;

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Clamps a fundamental to the tunable range, also honouring the Nyquist guard.
#[inline]
fn clamp_frequency(freq_hz: Sample, sample_rate: u32) -> Sample {
    let nyquist = sample_rate.max(1) as Sample * NYQUIST_GUARD;
    let upper = MAX_FREQUENCY_HZ.min(nyquist).max(MIN_FREQUENCY_HZ);
    freq_hz.clamp(MIN_FREQUENCY_HZ, upper)
}

/// Construction parameters for a [`StruckBarNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StruckBarParams {
    /// Fundamental (strike pitch) frequency in hertz.
    pub frequency_hz: Sample,
    /// Fundamental `-60 dB` decay time in seconds (longer rings longer).
    pub decay_s: Sample,
    /// Mallet hardness / brightness in `[0, 1]` (`1` is a hard, bright mallet).
    pub brightness: Sample,
    /// Inharmonicity in `[0, 1]` (`0` tuned bar, `1` ideal free-free bar).
    pub inharmonicity: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for StruckBarParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            decay_s: DEFAULT_DECAY_S,
            brightness: DEFAULT_BRIGHTNESS,
            inharmonicity: DEFAULT_INHARMONICITY,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl StruckBarParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range. Frequency is clamped against the Nyquist limit too.
    #[must_use]
    pub fn sanitised(self, sample_rate: u32) -> Self {
        let d = Self::default();
        let frequency_hz =
            clamp_frequency(finite_or(self.frequency_hz, d.frequency_hz), sample_rate);
        let decay_s = finite_or(self.decay_s, d.decay_s).clamp(MIN_DECAY_S, MAX_DECAY_S);
        let brightness = finite_or(self.brightness, d.brightness).clamp(0.0, 1.0);
        let inharmonicity = finite_or(self.inharmonicity, d.inharmonicity).clamp(0.0, 1.0);
        let amplitude = finite_or(self.amplitude, d.amplitude);
        Self {
            frequency_hz,
            decay_s,
            brightness,
            inharmonicity,
            amplitude,
        }
    }
}

/// Mallet-struck inharmonic modal bar synthesis source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{StruckBarNode, StruckBarParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = StruckBarNode::new(48_000, StruckBarParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The mallet strike excites the modal bar, which rings and decays.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
#[derive(Clone, Debug)]
pub struct StruckBarNode {
    sample_rate: u32,
    frequency_hz: Sample,
    decay_s: Sample,
    brightness: Sample,
    inharmonicity: Sample,
    amplitude: Smoothed,
    // Per-mode resonator coefficients and ringing history.
    a1: [Sample; NUM_MODES],
    a2: [Sample; NUM_MODES],
    b0: [Sample; NUM_MODES],
    enabled: [bool; NUM_MODES],
    y1: [Sample; NUM_MODES],
    y2: [Sample; NUM_MODES],
    // Mallet-contact pulse state.
    pulse_len: u32,
    pulse_pos: u32,
    pulse_scale: Sample,
    velocity: Sample,
}

impl StruckBarNode {
    /// Builds a struck bar at `sample_rate` from (sanitised) `params`, struck
    /// once so it sounds immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: StruckBarParams) -> Self {
        let sr = sample_rate.max(1);
        let p = params.sanitised(sr);
        let mut node = Self {
            sample_rate: sr,
            frequency_hz: p.frequency_hz,
            decay_s: p.decay_s,
            brightness: p.brightness,
            inharmonicity: p.inharmonicity,
            amplitude: Smoothed::new(p.amplitude),
            a1: [0.0; NUM_MODES],
            a2: [0.0; NUM_MODES],
            b0: [0.0; NUM_MODES],
            enabled: [false; NUM_MODES],
            y1: [0.0; NUM_MODES],
            y2: [0.0; NUM_MODES],
            pulse_len: 1,
            pulse_pos: 0,
            pulse_scale: 1.0,
            velocity: DEFAULT_STRIKE_VELOCITY,
        };
        node.recompute();
        node.strike(DEFAULT_STRIKE_VELOCITY);
        node
    }

    /// Returns the current fundamental (strike pitch) in hertz.
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the fundamental `-60 dB` decay time in seconds.
    #[must_use]
    pub fn decay_s(&self) -> Sample {
        self.decay_s
    }

    /// Returns the mallet hardness / brightness in `[0, 1]`.
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
    }

    /// Returns the inharmonicity in `[0, 1]`.
    #[must_use]
    pub fn inharmonicity(&self) -> Sample {
        self.inharmonicity
    }

    /// Returns the target linear output amplitude.
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Sets the fundamental (strike pitch), clamped to the tunable range.
    pub fn set_frequency(&mut self, frequency_hz: Sample) {
        self.frequency_hz =
            clamp_frequency(finite_or(frequency_hz, self.frequency_hz), self.sample_rate);
        self.recompute();
    }

    /// Sets the fundamental `-60 dB` decay time, clamped to the valid range.
    pub fn set_decay(&mut self, decay_s: Sample) {
        self.decay_s = finite_or(decay_s, self.decay_s).clamp(MIN_DECAY_S, MAX_DECAY_S);
        self.recompute();
    }

    /// Sets the mallet hardness / brightness, clamped to `[0, 1]`.
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the inharmonicity, clamped to `[0, 1]`.
    pub fn set_inharmonicity(&mut self, inharmonicity: Sample) {
        self.inharmonicity = finite_or(inharmonicity, self.inharmonicity).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Retriggers the bar with the given strike `velocity` (clamped to
    /// `[0, 1]`), injecting a fresh mallet pulse while existing modes keep
    /// ringing.
    pub fn strike(&mut self, velocity: Sample) {
        self.velocity = finite_or(velocity, DEFAULT_STRIKE_VELOCITY).clamp(0.0, 1.0);
        self.pulse_pos = 0;
    }

    /// Recomputes every mode coefficient and the mallet-pulse geometry from the
    /// current scalar parameters. Never runs on the audio hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        let inh = self.inharmonicity;
        let gain_exp = GAIN_EXP_MIN + (1.0 - self.brightness) * (GAIN_EXP_MAX - GAIN_EXP_MIN);
        let nyquist = sr * NYQUIST_GUARD;

        for m in 0..NUM_MODES {
            let ratio = (1.0 - inh) * TUNED_BAR_RATIOS[m] + inh * IDEAL_BAR_RATIOS[m];
            let f_m = self.frequency_hz * ratio;
            if f_m <= 0.0 || f_m >= nyquist {
                self.a1[m] = 0.0;
                self.a2[m] = 0.0;
                self.b0[m] = 0.0;
                self.enabled[m] = false;
                continue;
            }
            let t60 = (self.decay_s / ops::powf(ratio, DECAY_RATIO_EXP))
                .clamp(MIN_DECAY_S, MAX_DECAY_S);
            let radius = ops::exp(-LN_1000 / (t60 * sr));
            let theta = TAU * f_m / sr;
            let (sin_t, cos_t) = (ops::sin(theta), ops::cos(theta));
            let gain = ops::powf(ratio, -gain_exp);
            self.a1[m] = 2.0 * radius * cos_t;
            self.a2[m] = -(radius * radius);
            self.b0[m] = gain * sin_t;
            self.enabled[m] = true;
        }

        let pulse_ms = PULSE_MS_MAX - self.brightness * (PULSE_MS_MAX - PULSE_MS_MIN);
        let len = ops::round(pulse_ms * sr / 1000.0) as i32;
        self.pulse_len = len.max(1) as u32;
        // Unit-area Hann pulse: sum_{n=0}^{L-1} w[n] == (L + 1) / 2.
        self.pulse_scale = 2.0 / (self.pulse_len as Sample + 1.0);
    }

    /// Renders one mono output sample, advancing every resonator and the mallet
    /// pulse by one step.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let drive = if self.pulse_pos < self.pulse_len {
            let n = self.pulse_pos as Sample;
            self.pulse_pos += 1;
            let window = 0.5 - 0.5 * ops::cos(TAU * (n + 1.0) / (self.pulse_len as Sample + 1.0));
            window * self.pulse_scale * self.velocity
        } else {
            0.0
        };

        let mut acc = 0.0;
        for m in 0..NUM_MODES {
            if !self.enabled[m] {
                continue;
            }
            let y = flush_denormal(self.b0[m] * drive + self.a1[m] * self.y1[m] + self.a2[m] * self.y2[m]);
            self.y2[m] = self.y1[m];
            self.y1[m] = y;
            acc += y;
        }

        let amp = self.amplitude.next_sample();
        flush_denormal(acc * OUTPUT_GAIN * amp)
    }
}

impl AudioNode for StruckBarNode {
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
        self.y1 = [0.0; NUM_MODES];
        self.y2 = [0.0; NUM_MODES];
        self.amplitude = Smoothed::new(self.amplitude.target());
        self.recompute();
        self.strike(DEFAULT_STRIKE_VELOCITY);
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut StruckBarNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(
        node: &mut StruckBarNode,
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

    /// Single-frequency magnitude via the Goertzel sum (test-only analysis).
    fn goertzel(block: &[Sample], freq: Sample) -> f64 {
        let w = TAU as f64 * (freq as f64) / (SR as f64);
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for (n, &s) in block.iter().enumerate() {
            re += (s as f64) * (w * n as f64).cos();
            im -= (s as f64) * (w * n as f64).sin();
        }
        (re * re + im * im).sqrt()
    }

    #[test]
    fn default_strike_produces_sound() {
        let mut node = StruckBarNode::new(SR, StruckBarParams::default());
        let out = render(&mut node, SR as usize / 10);
        let p = peak(&out);
        assert!(p > 1.0e-3, "struck bar should ring, peak = {p}");
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn tone_decays_toward_silence() {
        let mut node = StruckBarNode::new(SR, StruckBarParams::default());
        let out = render(&mut node, 4 * SR as usize);
        let head = energy(&out[..SR as usize / 10]);
        let tail = energy(&out[out.len() - SR as usize / 10..]);
        assert!(head > 0.0);
        assert!(
            tail < head * 1.0e-2,
            "ring should decay: head = {head}, tail = {tail}"
        );
    }

    #[test]
    fn long_run_stays_bounded_and_finite() {
        let mut node = StruckBarNode::new(SR, StruckBarParams::default());
        let out = render(&mut node, 4 * SR as usize);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(peak(&out) < 1.0, "peak = {}", peak(&out));
    }

    #[test]
    fn deterministic_across_instances() {
        let params = StruckBarParams::default();
        let mut a = StruckBarNode::new(SR, params);
        let mut b = StruckBarNode::new(SR, params);
        let oa = render(&mut a, SR as usize / 2);
        let ob = render(&mut b, SR as usize / 2);
        assert_eq!(oa, ob);
    }

    #[test]
    fn reset_restarts_identical_attack() {
        let mut node = StruckBarNode::new(SR, StruckBarParams::default());
        let first = render(&mut node, SR as usize / 2);
        node.reset();
        let second = render(&mut node, SR as usize / 2);
        assert_eq!(first, second);
    }

    #[test]
    fn amplitude_scales_output_energy() {
        let mut quiet = StruckBarNode::new(
            SR,
            StruckBarParams {
                amplitude: 0.25,
                ..StruckBarParams::default()
            },
        );
        let mut loud = StruckBarNode::new(
            SR,
            StruckBarParams {
                amplitude: 0.5,
                ..StruckBarParams::default()
            },
        );
        let eq = energy(&render(&mut quiet, SR as usize / 2));
        let el = energy(&render(&mut loud, SR as usize / 2));
        // Twice the amplitude is four times the energy.
        assert!((el / eq - 4.0).abs() < 0.05, "ratio = {}", el / eq);
    }

    #[test]
    fn inharmonic_partials_present() {
        // With full inharmonicity the second partial sits at ~2.756 * f0, well
        // away from the harmonic 2 * f0 of a pitched source.
        let f0 = 440.0;
        let mut node = StruckBarNode::new(
            SR,
            StruckBarParams {
                frequency_hz: f0,
                inharmonicity: 1.0,
                decay_s: 3.0,
                ..StruckBarParams::default()
            },
        );
        let out = render(&mut node, SR as usize / 2);
        let at_inharmonic = goertzel(&out, f0 * IDEAL_BAR_RATIOS[1]);
        let at_harmonic = goertzel(&out, f0 * 2.0);
        assert!(
            at_inharmonic > at_harmonic * 4.0,
            "inharmonic = {at_inharmonic}, harmonic = {at_harmonic}"
        );
    }

    #[test]
    fn fundamental_is_strongest_low_partial() {
        let f0 = 330.0;
        let mut node = StruckBarNode::new(
            SR,
            StruckBarParams {
                frequency_hz: f0,
                ..StruckBarParams::default()
            },
        );
        let out = render(&mut node, SR as usize / 2);
        let at_f0 = goertzel(&out, f0);
        let below = goertzel(&out, f0 * 0.5);
        assert!(at_f0 > below * 4.0, "f0 = {at_f0}, below = {below}");
    }

    #[test]
    fn softer_mallet_is_duller() {
        let f0 = 440.0;
        let bright = {
            let mut n = StruckBarNode::new(
                SR,
                StruckBarParams {
                    frequency_hz: f0,
                    brightness: 1.0,
                    ..StruckBarParams::default()
                },
            );
            render(&mut n, SR as usize / 2)
        };
        let dull = {
            let mut n = StruckBarNode::new(
                SR,
                StruckBarParams {
                    frequency_hz: f0,
                    brightness: 0.0,
                    ..StruckBarParams::default()
                },
            );
            render(&mut n, SR as usize / 2)
        };
        // Ratio of a high partial to the fundamental is larger for a hard mallet.
        let hi = f0 * IDEAL_BAR_RATIOS[3];
        let bright_ratio = goertzel(&bright, hi) / goertzel(&bright, f0).max(1.0e-9);
        let dull_ratio = goertzel(&dull, hi) / goertzel(&dull, f0).max(1.0e-9);
        assert!(
            bright_ratio > dull_ratio,
            "bright = {bright_ratio}, dull = {dull_ratio}"
        );
    }

    #[test]
    fn strike_retriggers_envelope() {
        let mut node = StruckBarNode::new(
            SR,
            StruckBarParams {
                decay_s: 0.1,
                ..StruckBarParams::default()
            },
        );
        // Let the first strike decay almost away.
        let _ = render(&mut node, SR as usize);
        let before = peak(&render(&mut node, 64));
        node.strike(1.0);
        let after = peak(&render(&mut node, SR as usize / 20));
        assert!(after > before * 10.0, "before = {before}, after = {after}");
    }

    #[test]
    fn mono_core_replicated_to_all_channels() {
        let mut node = StruckBarNode::new(SR, StruckBarParams::default());
        let chans = render_layout(&mut node, 512, ChannelLayout::Quad);
        assert_eq!(chans.len(), 4);
        for ch in 1..chans.len() {
            assert_eq!(chans[0], chans[ch]);
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = StruckBarNode::new(SR, StruckBarParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = StruckBarParams {
            frequency_hz: 523.0,
            decay_s: 1.5,
            brightness: 0.3,
            inharmonicity: 0.7,
            amplitude: 0.4,
        };
        let node = StruckBarNode::new(SR, params);
        assert!((node.frequency_hz() - 523.0).abs() < 1.0e-3);
        assert!((node.decay_s() - 1.5).abs() < 1.0e-3);
        assert!((node.brightness() - 0.3).abs() < 1.0e-3);
        assert!((node.inharmonicity() - 0.7).abs() < 1.0e-3);
        assert!((node.amplitude() - 0.4).abs() < 1.0e-3);
    }

    #[test]
    fn set_frequency_clamps_to_range() {
        let mut node = StruckBarNode::new(SR, StruckBarParams::default());
        node.set_frequency(1.0e9);
        assert!(node.frequency_hz() <= MAX_FREQUENCY_HZ);
        node.set_frequency(0.0);
        assert!((node.frequency_hz() - MIN_FREQUENCY_HZ).abs() < 1.0e-3);
    }

    #[test]
    fn set_decay_clamps_to_range() {
        let mut node = StruckBarNode::new(SR, StruckBarParams::default());
        node.set_decay(1.0e6);
        assert!((node.decay_s() - MAX_DECAY_S).abs() < 1.0e-3);
        node.set_decay(-1.0);
        assert!((node.decay_s() - MIN_DECAY_S).abs() < 1.0e-3);
    }

    #[test]
    fn set_brightness_and_inharmonicity_clamp() {
        let mut node = StruckBarNode::new(SR, StruckBarParams::default());
        node.set_brightness(5.0);
        assert!((node.brightness() - 1.0).abs() < 1.0e-6);
        node.set_brightness(-5.0);
        assert!(node.brightness().abs() < 1.0e-6);
        node.set_inharmonicity(5.0);
        assert!((node.inharmonicity() - 1.0).abs() < 1.0e-6);
        node.set_inharmonicity(-5.0);
        assert!(node.inharmonicity().abs() < 1.0e-6);
    }

    #[test]
    fn setters_reject_non_finite() {
        let mut node = StruckBarNode::new(SR, StruckBarParams::default());
        let (f, d, b, i) = (
            node.frequency_hz(),
            node.decay_s(),
            node.brightness(),
            node.inharmonicity(),
        );
        node.set_frequency(Sample::NAN);
        node.set_decay(Sample::INFINITY);
        node.set_brightness(Sample::NAN);
        node.set_inharmonicity(Sample::NEG_INFINITY);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert!((node.frequency_hz() - f).abs() < 1.0e-6);
        assert!((node.decay_s() - d).abs() < 1.0e-6);
        assert!((node.brightness() - b).abs() < 1.0e-6);
        assert!((node.inharmonicity() - i).abs() < 1.0e-6);
    }

    #[test]
    fn constructor_sanitizes_non_finite_params() {
        let node = StruckBarNode::new(
            SR,
            StruckBarParams {
                frequency_hz: Sample::NAN,
                decay_s: Sample::INFINITY,
                brightness: Sample::NAN,
                inharmonicity: Sample::NEG_INFINITY,
                amplitude: Sample::NAN,
            },
        );
        assert!(node.frequency_hz().is_finite());
        assert!(node.decay_s().is_finite());
        assert!(node.brightness().is_finite());
        assert!(node.inharmonicity().is_finite());
        assert!(node.amplitude().is_finite());
    }

    #[test]
    fn frequency_changes_output() {
        let mut low = StruckBarNode::new(
            SR,
            StruckBarParams {
                frequency_hz: 220.0,
                ..StruckBarParams::default()
            },
        );
        let mut high = StruckBarNode::new(
            SR,
            StruckBarParams {
                frequency_hz: 660.0,
                ..StruckBarParams::default()
            },
        );
        let ol = render(&mut low, SR as usize / 4);
        let oh = render(&mut high, SR as usize / 4);
        assert!(goertzel(&ol, 220.0) > goertzel(&ol, 660.0));
        assert!(goertzel(&oh, 660.0) > goertzel(&oh, 220.0));
    }

    #[test]
    fn inharmonicity_changes_output() {
        let mut tuned = StruckBarNode::new(
            SR,
            StruckBarParams {
                inharmonicity: 0.0,
                ..StruckBarParams::default()
            },
        );
        let mut ideal = StruckBarNode::new(
            SR,
            StruckBarParams {
                inharmonicity: 1.0,
                ..StruckBarParams::default()
            },
        );
        let ot = render(&mut tuned, SR as usize / 4);
        let oi = render(&mut ideal, SR as usize / 4);
        assert_ne!(ot, oi);
    }

    #[test]
    fn brightness_changes_output() {
        let mut dull = StruckBarNode::new(
            SR,
            StruckBarParams {
                brightness: 0.1,
                ..StruckBarParams::default()
            },
        );
        let mut bright = StruckBarNode::new(
            SR,
            StruckBarParams {
                brightness: 0.9,
                ..StruckBarParams::default()
            },
        );
        assert_ne!(render(&mut dull, 2048), render(&mut bright, 2048));
    }

    #[test]
    fn high_and_low_frequencies_both_sound() {
        for f0 in [40.0, 110.0, 440.0, 2000.0, 6000.0] {
            let mut node = StruckBarNode::new(
                SR,
                StruckBarParams {
                    frequency_hz: f0,
                    ..StruckBarParams::default()
                },
            );
            let out = render(&mut node, SR as usize / 10);
            let p = peak(&out);
            assert!(p > 1.0e-4 && p < 1.0, "f0 = {f0}, peak = {p}");
            assert!(out.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn amplitude_target_tracks_setter() {
        let mut node = StruckBarNode::new(SR, StruckBarParams::default());
        node.set_amplitude(0.33, Ramp::Immediate);
        assert!((node.amplitude() - 0.33).abs() < 1.0e-6);
    }

    #[test]
    fn latency_is_zero() {
        let node = StruckBarNode::new(SR, StruckBarParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
