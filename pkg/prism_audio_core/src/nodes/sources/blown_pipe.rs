//! Noise-excited resonant-pipe wind source node.
//!
//! [`BlownPipeNode`] is a *source* (zero inputs, one output) that synthesizes a
//! sustained, breathy pitched wind tone (pan-pipe / flue-organ-pipe / whistle /
//! ocarina family) by passing turbulent breath noise through a parallel bank of
//! [`NUM_HARMONICS`] resonant band-pass filters tuned to the harmonic series of
//! an air column. Unlike its self-oscillating wind siblings it is *linear*: the
//! pipe is a passive resonator driven by broadband noise, not a feedback loop
//! closed through a nonlinear jet or reed valve.
//!
//! # The pipe as a resonant filter bank
//!
//! A short air column has sharp acoustic resonances at the harmonics of its
//! fundamental. Each resonance is modelled as a two-pole resonator
//!
//! ```text
//! y[n] = b0*x[n] + a1*y[n-1] + a2*y[n-2]
//! a1 = 2*R*cos(theta),  a2 = -R*R,  theta = 2*pi*f_h / sr
//! ```
//!
//! whose pole radius `R` in `(0, 1)` sets the sharpness (the closer to `1`, the
//! narrower and purer the resonance). The gain `b0` uses the classic
//! constant-peak-gain *reson* normalization
//!
//! ```text
//! b0 = gain * (1 - R) * sqrt(1 - 2*R*cos(2*theta) + R*R)
//! ```
//!
//! so the resonator's frequency-response peak equals `gain` regardless of the
//! pole radius; `gain = ratio^(-tilt)` tilts the harmonic spectrum. Feeding
//! white noise into such a bank yields a pitched but airy tone: the resonances
//! colour the breath into a whistling pitch while residual turbulence remains
//! audible, exactly the character of a flue pipe or pan pipe.
//!
//! # Open versus stopped pipes
//!
//! A pipe open at both ends (or a flute-like flue) resonates on its *complete*
//! harmonic series (`f`, `2f`, `3f`, ...). A pipe stopped (closed) at one end
//! resonates only on its *odd* harmonics (`f`, `3f`, `5f`, ...), the hollow
//! timbre of a stopped organ rank or a bottle. The `stopped` flag selects
//! which harmonic set the resonator bank is tuned to.
//!
//! # Breath and air
//!
//! `breath_pressure` scales the turbulent excitation feeding the pipe and gates
//! the voice: at zero breath the pipe is silent. `breath_noise` mixes a
//! band-unshaped portion of that turbulence straight into the output for the
//! characteristic breathy hiss. The excitation is a self-contained,
//! deterministic `xorshift64` stream, so a given `(sample_rate, params, seed)`
//! reproduces bit-identical audio on every platform.
//!
//! # Relationship
//!
//! Reuses this crate's [`Sample`], [`Smoothed`], and denormal-flush primitives
//! and mirrors the two-pole modal resonator structure of
//! [`super::bell::BellNode`] and [`super::struck_plate::StruckPlateNode`], but
//! is driven by *continuous noise* instead of a one-shot contact pulse, so it
//! sustains rather than decays. It differs fundamentally from the four
//! self-oscillating waveguide winds -- [`super::air_jet_flute::AirJetFluteNode`],
//! [`super::reed_woodwind::ReedWoodwindNode`],
//! [`super::conical_reed::ConicalReedNode`], and
//! [`super::brass_lip_reed::BrassLipReedNode`] -- which close a nonlinear
//! feedback loop around a bidirectional delay line to self-oscillate; this node
//! has no feedback nonlinearity and never self-oscillates. It also differs from
//! [`super::helmholtz_resonator::HelmholtzResonatorNode`], a single
//! lumped self-oscillating mode, and from
//! [`super::karplus_strong::KarplusStrongNode`], whose feedback delay line is
//! excited by a one-shot burst and decays.
//!
//! # Provenance
//!
//! Classic public-domain DSP only; no third-party engine, library, or toolkit
//! source or derivative was consulted or copied. Source-filter (subtractive)
//! synthesis of wind tones -- a noise excitation shaped by resonant formants --
//! is textbook signal processing. The two-pole resonator and its
//! constant-peak-gain *reson* normalization are standard filter design
//! (Steiglitz, "A Digital Signal Processing Primer", 1996; Smith,
//! "Introduction to Digital Filters", CCRMA). The open-open versus stopped-pipe
//! harmonic acoustics are from Fletcher & Rossing, "The Physics of Musical
//! Instruments". The `xorshift64` generator seeded via `SplitMix64` is
//! public-domain (Marsaglia 2003; Vigna). Only these ideas are used.

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Number of harmonic resonators in the pipe bank.
pub const NUM_HARMONICS: usize = 16;

/// Lowest tunable fundamental in hertz.
const MIN_FREQUENCY_HZ: Sample = 20.0;
/// Highest tunable fundamental in hertz.
const MAX_FREQUENCY_HZ: Sample = 12_000.0;
/// Default fundamental in hertz (concert A).
const DEFAULT_FREQUENCY_HZ: Sample = 440.0;
/// Default spectral brightness.
const DEFAULT_BRIGHTNESS: Sample = 0.5;
/// Default resonance (tone purity).
const DEFAULT_RESONANCE: Sample = 0.65;
/// Default breath pressure (drive / gate).
const DEFAULT_BREATH_PRESSURE: Sample = 0.8;
/// Default breathy-air mix.
const DEFAULT_BREATH_NOISE: Sample = 0.2;
/// Default linear output amplitude.
const DEFAULT_AMPLITUDE: Sample = 0.5;
/// Default pipe termination (open-open resonates on every harmonic).
const DEFAULT_STOPPED: bool = false;
/// Default PRNG seed (arbitrary fixed, non-zero).
const DEFAULT_SEED: u64 = 0x5069_7065_0053_6565;
/// Pole radius at `resonance == 0` (widest, breathiest resonance).
const R_MIN: Sample = 0.980;
/// Pole radius at `resonance == 1` (narrowest, purest resonance).
const R_MAX: Sample = 0.9990;
/// Spectral tilt exponent at `brightness == 1` (bright, little roll-off).
const TILT_MIN: Sample = 0.3;
/// Spectral tilt exponent at `brightness == 0` (dull, steep roll-off).
const TILT_MAX: Sample = 2.5;
/// Scale of the raw turbulence mixed into the output by `breath_noise`.
const BREATH_LEVEL: Sample = 0.3;
/// Fraction of Nyquist above which a harmonic resonator is muted.
const NYQUIST_GUARD: Sample = 0.49;
/// Overall output scale keeping the summed bank peak below full scale.
///
/// Calibrated so the deterministic module parameter-grid test peaks near
/// `0.59` with `amplitude == 1`, leaving comfortable headroom to full scale.
const OUTPUT_GAIN: Sample = 0.35;

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Clamps a fundamental to the tunable range, honouring the Nyquist guard.
#[inline]
fn clamp_frequency(freq_hz: Sample, sample_rate: u32) -> Sample {
    let nyquist = sample_rate.max(1) as Sample * NYQUIST_GUARD;
    let upper = MAX_FREQUENCY_HZ.min(nyquist).max(MIN_FREQUENCY_HZ);
    freq_hz.clamp(MIN_FREQUENCY_HZ, upper)
}

/// Self-contained deterministic PRNG (Marsaglia `xorshift64`) seeded via
/// `SplitMix64`.
///
/// # Provenance
///
/// Public-domain generators (Marsaglia 2003; Vigna's `SplitMix64`); reproduced
/// as a tiny self-contained primitive so this node needs no shared RNG state.
#[derive(Debug, Clone, Copy)]
struct Xorshift64 {
    /// Current 64-bit generator state; kept non-zero by the seeding routine.
    state: u64,
}

impl Xorshift64 {
    /// Builds a generator whose state is diffused from `seed` via `SplitMix64`.
    #[inline]
    fn new(seed: u64) -> Self {
        Self {
            state: seed_to_state(seed),
        }
    }

    /// Returns the next white sample uniformly distributed in `[-1.0, 1.0)`.
    #[inline]
    fn next_bipolar(&mut self) -> Sample {
        // Marsaglia's xorshift64 (shift triple 13/7/17), full period 2^64 - 1.
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // The high 32 bits carry the best statistical quality for xorshift.
        let bits = (x >> 32) as u32;
        let unit = (bits >> 8) as Sample * (1.0 / 16_777_216.0);
        unit * 2.0 - 1.0
    }
}

/// Diffuses a user seed into a non-zero `xorshift64` state via `SplitMix64`.
#[inline]
fn seed_to_state(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        z
    }
}

/// Construction parameters for a [`BlownPipeNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BlownPipeParams {
    /// Fundamental frequency in hertz.
    pub frequency_hz: Sample,
    /// Spectral brightness in `[0, 1]` (`1` keeps more upper harmonics).
    pub brightness: Sample,
    /// Resonance / tone purity in `[0, 1]` (`1` is a pure whistle).
    pub resonance: Sample,
    /// Breath pressure (drive / gate) in `[0, 1]` (`0` is silent).
    pub breath_pressure: Sample,
    /// Breathy-air mix in `[0, 1]` (`1` is maximally airy).
    pub breath_noise: Sample,
    /// Pipe termination: `false` is open-open (all harmonics), `true` is stopped
    /// (odd harmonics only).
    pub stopped: bool,
    /// Linear output amplitude.
    pub amplitude: Sample,
    /// PRNG seed for the turbulent excitation.
    pub seed: u64,
}

impl Default for BlownPipeParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            brightness: DEFAULT_BRIGHTNESS,
            resonance: DEFAULT_RESONANCE,
            breath_pressure: DEFAULT_BREATH_PRESSURE,
            breath_noise: DEFAULT_BREATH_NOISE,
            stopped: DEFAULT_STOPPED,
            amplitude: DEFAULT_AMPLITUDE,
            seed: DEFAULT_SEED,
        }
    }
}

impl BlownPipeParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range. Frequency is clamped against the Nyquist limit too.
    #[must_use]
    pub fn sanitised(self, sample_rate: u32) -> Self {
        let d = Self::default();
        let frequency_hz =
            clamp_frequency(finite_or(self.frequency_hz, d.frequency_hz), sample_rate);
        let brightness = finite_or(self.brightness, d.brightness).clamp(0.0, 1.0);
        let resonance = finite_or(self.resonance, d.resonance).clamp(0.0, 1.0);
        let breath_pressure = finite_or(self.breath_pressure, d.breath_pressure).clamp(0.0, 1.0);
        let breath_noise = finite_or(self.breath_noise, d.breath_noise).clamp(0.0, 1.0);
        let amplitude = finite_or(self.amplitude, d.amplitude);
        Self {
            frequency_hz,
            brightness,
            resonance,
            breath_pressure,
            breath_noise,
            stopped: self.stopped,
            amplitude,
            seed: self.seed,
        }
    }
}

/// A noise-excited resonant-pipe wind source.
///
/// See the [module documentation](self) for the model, determinism guarantee,
/// and real-time contract.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{BlownPipeNode, BlownPipeParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = BlownPipeNode::new(48_000, BlownPipeParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // Breath drives the pipe resonances into a sustained, finite tone.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
pub struct BlownPipeNode {
    sample_rate: u32,
    frequency_hz: Sample,
    brightness: Sample,
    resonance: Sample,
    breath_noise: Sample,
    stopped: bool,
    breath: Smoothed,
    amplitude: Smoothed,
    a1: [Sample; NUM_HARMONICS],
    a2: [Sample; NUM_HARMONICS],
    b0: [Sample; NUM_HARMONICS],
    enabled: [bool; NUM_HARMONICS],
    y1: [Sample; NUM_HARMONICS],
    y2: [Sample; NUM_HARMONICS],
    rng: Xorshift64,
    seed: u64,
}

impl BlownPipeNode {
    /// Builds a pipe voice for `sample_rate` from `params`, sanitising every
    /// field. The default breath pressure makes it sound immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: BlownPipeParams) -> Self {
        let p = params.sanitised(sample_rate);
        let mut node = Self {
            sample_rate,
            frequency_hz: p.frequency_hz,
            brightness: p.brightness,
            resonance: p.resonance,
            breath_noise: p.breath_noise,
            stopped: p.stopped,
            breath: Smoothed::new(p.breath_pressure),
            amplitude: Smoothed::new(p.amplitude),
            a1: [0.0; NUM_HARMONICS],
            a2: [0.0; NUM_HARMONICS],
            b0: [0.0; NUM_HARMONICS],
            enabled: [false; NUM_HARMONICS],
            y1: [0.0; NUM_HARMONICS],
            y2: [0.0; NUM_HARMONICS],
            rng: Xorshift64::new(p.seed),
            seed: p.seed,
        };
        node.recompute();
        node
    }

    /// Returns the current fundamental in hertz.
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the spectral brightness in `[0, 1]`.
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
    }

    /// Returns the resonance / tone purity in `[0, 1]`.
    #[must_use]
    pub fn resonance(&self) -> Sample {
        self.resonance
    }

    /// Returns the breathy-air mix in `[0, 1]`.
    #[must_use]
    pub fn breath_noise(&self) -> Sample {
        self.breath_noise
    }

    /// Returns whether the pipe is stopped (odd harmonics only).
    #[must_use]
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Returns the target breath pressure in `[0, 1]`.
    #[must_use]
    pub fn breath_pressure(&self) -> Sample {
        self.breath.target()
    }

    /// Returns the target linear output amplitude.
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Reports whether a given harmonic resonator is currently audible (below
    /// Nyquist).
    #[must_use]
    pub fn mode_enabled(&self, harmonic: usize) -> bool {
        self.enabled.get(harmonic).copied().unwrap_or(false)
    }

    /// Sets the fundamental, clamped to the tunable range.
    pub fn set_frequency(&mut self, frequency_hz: Sample) {
        self.frequency_hz =
            clamp_frequency(finite_or(frequency_hz, self.frequency_hz), self.sample_rate);
        self.recompute();
    }

    /// Sets the spectral brightness, clamped to `[0, 1]`.
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the resonance / tone purity, clamped to `[0, 1]`.
    pub fn set_resonance(&mut self, resonance: Sample) {
        self.resonance = finite_or(resonance, self.resonance).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the breathy-air mix, clamped to `[0, 1]`.
    pub fn set_breath_noise(&mut self, breath_noise: Sample) {
        self.breath_noise = finite_or(breath_noise, self.breath_noise).clamp(0.0, 1.0);
    }

    /// Selects the pipe termination: `false` open-open, `true` stopped.
    pub fn set_stopped(&mut self, stopped: bool) {
        self.stopped = stopped;
        self.recompute();
    }

    /// Sets the target breath pressure in `[0, 1]`, gliding over `ramp`.
    pub fn set_breath(&mut self, breath_pressure: Sample, ramp: Ramp) {
        let target = finite_or(breath_pressure, self.breath.target()).clamp(0.0, 1.0);
        self.breath.set_target(target, ramp);
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Recomputes every harmonic resonator coefficient from the current scalar
    /// parameters. Never runs on the audio hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        let radius = R_MIN + self.resonance * (R_MAX - R_MIN);
        let tilt = TILT_MAX - self.brightness * (TILT_MAX - TILT_MIN);
        let nyquist = sr * NYQUIST_GUARD;

        for h in 0..NUM_HARMONICS {
            let ratio = if self.stopped {
                (2 * h + 1) as Sample
            } else {
                (h + 1) as Sample
            };
            let f_h = self.frequency_hz * ratio;
            if f_h <= 0.0 || f_h >= nyquist {
                self.a1[h] = 0.0;
                self.a2[h] = 0.0;
                self.b0[h] = 0.0;
                self.enabled[h] = false;
                continue;
            }
            let theta = TAU * f_h / sr;
            let cos_t = ops::cos(theta);
            let gain = ops::powf(ratio, -tilt);
            // Constant-peak-gain reson normalization: the resonator's
            // frequency-response peak equals `gain` for any pole radius.
            let norm =
                (1.0 - radius) * ops::sqrt(1.0 - 2.0 * radius * ops::cos(2.0 * theta) + radius * radius);
            self.a1[h] = 2.0 * radius * cos_t;
            self.a2[h] = -(radius * radius);
            self.b0[h] = gain * norm;
            self.enabled[h] = true;
        }
    }

    /// Renders one mono output sample, advancing every resonator by one step.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let white = self.rng.next_bipolar();
        let breath = self.breath.next_sample();
        let drive = breath * white;

        let mut acc = 0.0;
        for h in 0..NUM_HARMONICS {
            if !self.enabled[h] {
                continue;
            }
            let y = flush_denormal(
                self.b0[h] * drive + self.a1[h] * self.y1[h] + self.a2[h] * self.y2[h],
            );
            self.y2[h] = self.y1[h];
            self.y1[h] = y;
            acc += y;
        }

        let air = self.breath_noise * BREATH_LEVEL * drive;
        let amp = self.amplitude.next_sample();
        flush_denormal((acc + air) * OUTPUT_GAIN * amp)
    }
}

impl AudioNode for BlownPipeNode {
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
        self.y1 = [0.0; NUM_HARMONICS];
        self.y2 = [0.0; NUM_HARMONICS];
        self.breath = Smoothed::new(self.breath.target());
        self.amplitude = Smoothed::new(self.amplitude.target());
        self.rng = Xorshift64::new(self.seed);
        self.recompute();
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

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut BlownPipeNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(
        node: &mut BlownPipeNode,
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

    fn rms(block: &[Sample]) -> f64 {
        if block.is_empty() {
            return 0.0;
        }
        (energy(block) / block.len() as f64).sqrt()
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

    /// Fraction of the block energy concentrated at `freq`, a tonal-purity
    /// proxy that is amplitude-invariant.
    fn purity(block: &[Sample], freq: Sample) -> f64 {
        let g = goertzel(block, freq);
        g * g / energy(block).max(1e-12)
    }

    /// First-difference energy, a proxy for high-frequency content.
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
    fn default_breath_produces_sound() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        let out = render(&mut node, SR as usize);
        let p = peak(&out);
        assert!(p > 0.0 && p.is_finite(), "default pipe should sound: {p}");
    }

    #[test]
    fn silent_without_breath() {
        let params = BlownPipeParams {
            breath_pressure: 0.0,
            ..BlownPipeParams::default()
        };
        let mut node = BlownPipeNode::new(SR, params);
        let out = render(&mut node, SR as usize / 2);
        assert_eq!(peak(&out), 0.0, "zero breath pressure must gate to silence");
    }

    #[test]
    fn tone_sustains() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        let out = render(&mut node, 2 * SR as usize);
        let half = out.len() / 2;
        let early = rms(&out[..half]);
        let late = rms(&out[half..]);
        assert!(
            late > 0.5 * early && late.is_finite(),
            "pipe should sustain: early={early} late={late}"
        );
    }

    #[test]
    fn long_run_stays_bounded_and_finite() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        let out = render(&mut node, 10 * SR as usize);
        let p = peak(&out);
        assert!(p < 1.0 && p.is_finite(), "ten-second run peak {p} must stay bounded");
    }

    #[test]
    fn full_parameter_grid_stays_below_full_scale() {
        let freqs = [110.0_f32, 440.0, 1760.0];
        let resonances = [0.0_f32, 0.5, 1.0];
        let brightnesses = [0.0_f32, 1.0];
        let breath_noises = [0.0_f32, 1.0];
        let mut worst = 0.0_f32;
        for &frequency_hz in &freqs {
            for &resonance in &resonances {
                for &brightness in &brightnesses {
                    for &breath_noise in &breath_noises {
                        for &stopped in &[false, true] {
                            let params = BlownPipeParams {
                                frequency_hz,
                                brightness,
                                resonance,
                                breath_pressure: 1.0,
                                breath_noise,
                                stopped,
                                amplitude: 1.0,
                                seed: DEFAULT_SEED,
                            };
                            let mut node = BlownPipeNode::new(SR, params);
                            let out = render(&mut node, SR as usize / 2);
                            worst = worst.max(peak(&out));
                        }
                    }
                }
            }
        }
        // Pin the calibrated worst-case headroom so OUTPUT_GAIN drift is caught.
        assert!(
            (0.55..0.65).contains(&worst),
            "grid worst-case peak should stay in the calibrated band: {worst}"
        );
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = BlownPipeNode::new(SR, BlownPipeParams::default());
        let mut b = BlownPipeNode::new(SR, BlownPipeParams::default());
        let out_a = render(&mut a, SR as usize);
        let out_b = render(&mut b, SR as usize);
        assert_eq!(out_a, out_b, "equal params must produce bit-identical audio");
    }

    #[test]
    fn reset_replays_identical_output() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        let first = render(&mut node, SR as usize);
        node.reset();
        let second = render(&mut node, SR as usize);
        assert_eq!(first, second, "reset must replay identical audio");
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let quiet_params = BlownPipeParams {
            amplitude: 0.25,
            ..BlownPipeParams::default()
        };
        let loud_params = BlownPipeParams {
            amplitude: 0.5,
            ..BlownPipeParams::default()
        };
        let mut quiet = BlownPipeNode::new(SR, quiet_params);
        let mut loud = BlownPipeNode::new(SR, loud_params);
        let eq = energy(&render(&mut quiet, SR as usize));
        let el = energy(&render(&mut loud, SR as usize));
        let ratio = el / eq.max(1e-12);
        assert!(
            (3.9..4.1).contains(&ratio),
            "doubling amplitude should quadruple energy: {ratio}"
        );
    }

    #[test]
    fn fundamental_present() {
        let params = BlownPipeParams {
            resonance: 0.95,
            breath_noise: 0.0,
            ..BlownPipeParams::default()
        };
        let mut node = BlownPipeNode::new(SR, params);
        let out = render(&mut node, SR as usize);
        let f0 = node.frequency_hz();
        let at_f0 = goertzel(&out, f0);
        let off = goertzel(&out, f0 * 1.5);
        assert!(
            at_f0 > off * 4.0,
            "fundamental should dominate an inter-harmonic bin: {at_f0} vs {off}"
        );
    }

    #[test]
    fn resonance_increases_tonal_purity() {
        let f0 = 330.0_f32;
        let dull_params = BlownPipeParams {
            frequency_hz: f0,
            resonance: 0.0,
            breath_noise: 0.0,
            ..BlownPipeParams::default()
        };
        let pure_params = BlownPipeParams {
            resonance: 1.0,
            ..dull_params
        };
        let mut dull = BlownPipeNode::new(SR, dull_params);
        let mut pure = BlownPipeNode::new(SR, pure_params);
        let p_dull = purity(&render(&mut dull, SR as usize), f0);
        let p_pure = purity(&render(&mut pure, SR as usize), f0);
        assert!(
            p_pure > p_dull * 3.0,
            "higher resonance should concentrate energy at f0: pure={p_pure} dull={p_dull}"
        );
    }

    #[test]
    fn stopped_suppresses_even_harmonics() {
        let f0 = 300.0_f32;
        let open_params = BlownPipeParams {
            frequency_hz: f0,
            resonance: 0.95,
            breath_noise: 0.0,
            stopped: false,
            ..BlownPipeParams::default()
        };
        let stopped_params = BlownPipeParams {
            stopped: true,
            ..open_params
        };
        let mut open = BlownPipeNode::new(SR, open_params);
        let mut stopped = BlownPipeNode::new(SR, stopped_params);
        let open_out = render(&mut open, SR as usize);
        let stopped_out = render(&mut stopped, SR as usize);
        let open_ratio = goertzel(&open_out, 2.0 * f0) / goertzel(&open_out, f0).max(1e-12);
        let stopped_ratio =
            goertzel(&stopped_out, 2.0 * f0) / goertzel(&stopped_out, f0).max(1e-12);
        assert!(
            stopped_ratio < open_ratio * 0.1,
            "stopped pipe should suppress the second harmonic: stopped={stopped_ratio} open={open_ratio}"
        );
    }

    #[test]
    fn brighter_has_more_high_frequency() {
        let dark_params = BlownPipeParams {
            brightness: 0.1,
            breath_noise: 0.0,
            ..BlownPipeParams::default()
        };
        let bright_params = BlownPipeParams {
            brightness: 0.9,
            ..dark_params
        };
        let mut dark = BlownPipeNode::new(SR, dark_params);
        let mut bright = BlownPipeNode::new(SR, bright_params);
        let hf_dark = hf_energy(&render(&mut dark, SR as usize));
        let hf_bright = hf_energy(&render(&mut bright, SR as usize));
        assert!(
            hf_bright > hf_dark * 2.0,
            "brighter tone should hold more high-frequency energy: bright={hf_bright} dark={hf_dark}"
        );
    }

    #[test]
    fn breath_noise_adds_broadband_air() {
        let dry_params = BlownPipeParams {
            resonance: 0.9,
            breath_noise: 0.0,
            ..BlownPipeParams::default()
        };
        let airy_params = BlownPipeParams {
            breath_noise: 1.0,
            ..dry_params
        };
        let mut dry = BlownPipeNode::new(SR, dry_params);
        let mut airy = BlownPipeNode::new(SR, airy_params);
        let hf_dry = hf_energy(&render(&mut dry, SR as usize));
        let hf_airy = hf_energy(&render(&mut airy, SR as usize));
        assert!(
            hf_airy > hf_dry * 5.0,
            "breath noise should add broadband air: airy={hf_airy} dry={hf_dry}"
        );
    }

    #[test]
    fn frequency_changes_output() {
        let mut low = BlownPipeNode::new(SR, BlownPipeParams::default());
        let mut high = BlownPipeNode::new(SR, BlownPipeParams::default());
        high.set_frequency(1500.0);
        let low_out = render(&mut low, SR as usize / 4);
        let high_out = render(&mut high, SR as usize / 4);
        assert_ne!(low_out, high_out, "retuning the pipe must change its output");
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty(), "zero active frames should write nothing");
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = BlownPipeParams {
            frequency_hz: 523.25,
            brightness: 0.3,
            resonance: 0.7,
            breath_pressure: 0.6,
            breath_noise: 0.4,
            stopped: true,
            amplitude: 0.25,
            seed: 0x1234_5678_9ABC_DEF0,
        };
        let node = BlownPipeNode::new(SR, params);
        assert_eq!(node.frequency_hz(), 523.25);
        assert_eq!(node.brightness(), 0.3);
        assert_eq!(node.resonance(), 0.7);
        assert_eq!(node.breath_noise(), 0.4);
        assert!(node.stopped());
        assert_eq!(node.breath_pressure(), 0.6);
        assert_eq!(node.amplitude(), 0.25);
    }

    #[test]
    fn frequency_is_clamped() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        node.set_frequency(1.0);
        assert_eq!(node.frequency_hz(), MIN_FREQUENCY_HZ);
        node.set_frequency(1.0e9);
        assert!(node.frequency_hz() <= MAX_FREQUENCY_HZ);
    }

    #[test]
    fn resonance_is_clamped() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        node.set_resonance(-1.0);
        assert_eq!(node.resonance(), 0.0);
        node.set_resonance(5.0);
        assert_eq!(node.resonance(), 1.0);
    }

    #[test]
    fn brightness_is_clamped() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        node.set_brightness(-2.0);
        assert_eq!(node.brightness(), 0.0);
        node.set_brightness(2.0);
        assert_eq!(node.brightness(), 1.0);
    }

    #[test]
    fn breath_noise_is_clamped() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        node.set_breath_noise(-1.0);
        assert_eq!(node.breath_noise(), 0.0);
        node.set_breath_noise(3.0);
        assert_eq!(node.breath_noise(), 1.0);
    }

    #[test]
    fn breath_pressure_is_clamped() {
        let high_params = BlownPipeParams {
            breath_pressure: 5.0,
            ..BlownPipeParams::default()
        };
        let low_params = BlownPipeParams {
            breath_pressure: -3.0,
            ..BlownPipeParams::default()
        };
        assert_eq!(BlownPipeNode::new(SR, high_params).breath_pressure(), 1.0);
        assert_eq!(BlownPipeNode::new(SR, low_params).breath_pressure(), 0.0);
    }

    #[test]
    fn setters_reject_non_finite_and_keep_previous() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        let f = node.frequency_hz();
        let b = node.brightness();
        let r = node.resonance();
        let bn = node.breath_noise();
        node.set_frequency(Sample::NAN);
        node.set_brightness(Sample::INFINITY);
        node.set_resonance(Sample::NEG_INFINITY);
        node.set_breath_noise(Sample::NAN);
        assert_eq!(node.frequency_hz(), f);
        assert_eq!(node.brightness(), b);
        assert_eq!(node.resonance(), r);
        assert_eq!(node.breath_noise(), bn);
    }

    #[test]
    fn constructor_sanitises_non_finite() {
        let params = BlownPipeParams {
            frequency_hz: Sample::NAN,
            brightness: Sample::INFINITY,
            resonance: Sample::NAN,
            breath_pressure: Sample::NAN,
            breath_noise: Sample::NEG_INFINITY,
            amplitude: Sample::NAN,
            ..BlownPipeParams::default()
        };
        let mut node = BlownPipeNode::new(SR, params);
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.brightness(), DEFAULT_BRIGHTNESS);
        let out = render(&mut node, SR as usize / 10);
        assert!(peak(&out).is_finite(), "sanitised node must stay finite");
    }

    #[test]
    fn mono_core_copies_to_all_channels() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        let chans = render_layout(&mut node, SR as usize / 10, ChannelLayout::Quad);
        assert_eq!(chans.len(), 4);
        for ch in 1..chans.len() {
            assert_eq!(chans[0], chans[ch], "channel {ch} should mirror channel 0");
        }
    }

    #[test]
    fn high_pitch_mutes_supersonic_modes() {
        let params = BlownPipeParams {
            frequency_hz: MAX_FREQUENCY_HZ,
            ..BlownPipeParams::default()
        };
        let node = BlownPipeNode::new(SR, params);
        assert!(node.mode_enabled(0), "the fundamental must stay audible");
        assert!(
            !node.mode_enabled(1),
            "the second harmonic exceeds Nyquist and must be muted"
        );
    }

    #[test]
    fn latency_is_zero() {
        let node = BlownPipeNode::new(SR, BlownPipeParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn set_stopped_changes_output() {
        let mut open = BlownPipeNode::new(SR, BlownPipeParams::default());
        let mut stopped = BlownPipeNode::new(SR, BlownPipeParams::default());
        stopped.set_stopped(true);
        assert!(stopped.stopped());
        assert!(!open.stopped());
        let open_out = render(&mut open, SR as usize / 4);
        let stopped_out = render(&mut stopped, SR as usize / 4);
        assert_ne!(open_out, stopped_out, "stopping the pipe must change its spectrum");
    }

    #[test]
    fn set_breath_and_amplitude_track_target() {
        let mut node = BlownPipeNode::new(SR, BlownPipeParams::default());
        node.set_breath(0.4, Ramp::Immediate);
        node.set_amplitude(0.9, Ramp::Immediate);
        assert_eq!(node.breath_pressure(), 0.4);
        assert_eq!(node.amplitude(), 0.9);
    }
}
