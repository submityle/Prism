//! Air-jet flute digital-waveguide physical-modeling source node.
//!
//! [`AirJetFluteNode`] is a *source* (zero inputs, one output) that synthesizes
//! a sustained, self-oscillating open-flute tone (concert-flute / recorder
//! family) by modeling a cylindrical bore excited by an air jet striking a
//! labium (the edge tone). Like its wind siblings
//! [`super::bowed_string::BowedStringNode`] and
//! [`super::reed_woodwind::ReedWoodwindNode`] it is *continuously driven*: a
//! steady breath feeds a jet whose nonlinear deflection pumps energy into the
//! bore every sample, so the tone sustains for as long as the player blows.
//!
//! # The bore and the two travelling-wave rails
//!
//! The open-open bore is modelled as a bidirectional digital waveguide: two
//! delay lines carry the pressure waves travelling in opposite directions
//! between the embouchure and the open far end, plus a third line for the
//! convective jet transit:
//!
//! ```text
//! at_emb = bore_line.read(bore_delay)   (wave arriving at the embouchure)
//! at_end = end_line.read(bore_delay)    (wave arriving at the open far end)
//! jet    = jet_line.read(jet_delay)     (flow travelling along the jet)
//! ```
//!
//! The bore is an *open-open* cylinder: both the embouchure and the far end
//! radiate, so each one-way trip reflects the pressure wave with an inversion.
//! Over a full round trip the *two* inversions cancel, giving *no* net
//! inversion, so the tube resonates on its *full* harmonic series (every
//! integer harmonic), the acoustic reason a flute sounds an octave above a
//! clarinet of the same length. A one-zero low-pass loss filter with loop gain
//! just below unity models the frequency-dependent radiation and wall losses at
//! the open far end.
//!
//! # The jet (edge-tone nonlinearity)
//!
//! The player's breath forms a thin jet that crosses the embouchure hole and
//! strikes the labium. The jet is deflected above or below the edge by the
//! acoustic pressure in the bore, and that deflection drives the tube. The
//! deflection saturates as the jet is pushed fully to one side, which the
//! classic model captures with a cubic characteristic:
//!
//! ```text
//! drop       = (breath + turbulence) - at_emb        (pressure offset at the lip)
//! x          = clamp(jet_sensed, -1, 1)               (convectively delayed offset)
//! jet        = JET_DRIVE * (x - x^3)                  (cubic edge-tone nonlinearity)
//! end_refl   = -loss_filter(at_end)                   (open far end: inverting, lossy)
//! emb_refl   = -at_emb + jet                          (open embouchure: inverting + jet)
//! end_line.write(emb_refl)                            (cross-couple the two rails)
//! bore_line.write(end_refl)
//! jet_line.write(drop)
//! ```
//!
//! The jet line delays `drop` by the jet transit time
//! `jet_delay = jet_ratio * bore_delay`, so the edge tone and the bore
//! resonance phase-lock into a stable oscillation. Because the cubic argument is
//! clamped to `[-1, 1]` the jet can never inject unbounded energy, so the
//! nonlinearity is self-limiting and (with the sub-unity loop gain) the loop
//! stays stable without any hard clamp. A small amount of deterministic breath
//! turbulence (scaled by `breath_noise` and by the breath itself) breaks the
//! initial symmetry so the oscillation starts, exactly as real breath noise
//! initiates a flute tone.
//!
//! # Pitch and timbre
//!
//! Because each one-way bore delay is a half wavelength (the open-open round
//! trip is two of them with two cancelling inversions), the sounding
//! fundamental is `f0 = sample_rate / (2 * bore_delay)` (minus half the loss
//! filter's small group delay, folded in for tuning accuracy). `jet_ratio` sets
//! the embouchure/jet geometry and selects the oscillation regime (the "octave"
//! the jet locks onto); `brightness` sets the loss-filter coefficient
//! `S = 0.5 * (1 - brightness)` exactly as in the Karplus-Strong loop; `breath`
//! sets loudness and, through the jet nonlinearity, the richness of the
//! harmonics. The radiated signal is the embouchure wave scaled by a fixed
//! `OUTPUT_GAIN` normalization and the user amplitude.
//!
//! # Determinism
//!
//! The breath turbulence is a self-contained seeded `xorshift64` stream, so two
//! [`AirJetFluteNode`]s built with the same sample rate, parameters, and seed
//! emit bit-identical streams on every platform via [`bevy_math::ops`], and
//! [`AudioNode::reset`] reseeds the generator and clears the lines to restart
//! the identical attack.
//!
//! # Relationship
//!
//! This is the *air-jet-driven* sibling of the *bow-driven*
//! [`super::bowed_string::BowedStringNode`] and the *reed-driven*
//! [`super::reed_woodwind::ReedWoodwindNode`]: all three recirculate energy in a
//! tuned, loss-filtered delay loop with fractional-delay tuning, but the flute
//! is sustained by a jet/edge-tone nonlinearity rather than bow friction or a
//! reed valve, and its non-inverting open-open loop gives the *full* harmonic
//! series rather than the clarinet's odd-only partials. It reuses this crate's
//! own [`Sample`] type, [`Smoothed`] parameter smoother, denormal-flushing
//! primitive, and the public-domain `xorshift64`/`SplitMix64` PRNG shared with
//! [`super::noise::NoiseNode`].
//!
//! # Real-time contract
//!
//! Both delay lines are pre-allocated in [`AirJetFluteNode::new`] for the lowest
//! supported pitch, so [`process`](crate::graph::AudioNode::process) performs no
//! allocation, takes no locks, and cannot panic: every recirculated sample is
//! denormal-flushed, the loop gain is below unity, the jet argument is clamped
//! to `[-1, 1]`, and non-finite parameters are rejected at the setters. The
//! breath pressure and output amplitude are driven through [`Smoothed`] values
//! so performance gestures never zipper.
//!
//! # Provenance
//!
//! The jet drive and bore model follow the standard public digital-waveguide
//! treatment of air-jet instruments in J. O. Smith's *Physical Audio Signal
//! Processing* (public online text) and the meta-wind-instrument formulation of
//! P. R. Cook ("A meta-wind-instrument physical model", ICMC 1992), built on the
//! jet physics of `McIntyre`, Schumacher and Woodhouse ("On the oscillations of
//! musical instruments", *Journal of the Acoustical Society of America*, 1983)
//! and the air-jet studies of Verge, Fabre and Hirschberg. The one-pole loss
//! filter and `brightness` mapping are shared with the Jaffe-Smith
//! Karplus-Strong extensions; the PRNG is the public-domain `xorshift64` seeded
//! by `SplitMix64` (S. Vigna). This module contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, or Web Audio source
//! or derived code**, and nothing from the STK or any other audio toolkit's
//! implementation; it is written purely from that publicly documented theory.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Lowest tunable fundamental in hertz. Bounds the pre-allocated delay lines.
pub const MIN_FREQUENCY_HZ: Sample = 40.0;

/// Default fundamental (pitch) frequency in hertz (concert-flute A4).
pub const DEFAULT_FREQUENCY_HZ: Sample = 440.0;

/// Default normalized breath pressure in `[0, 1]`.
pub const DEFAULT_BREATH_PRESSURE: Sample = 0.5;

/// Default jet/embouchure ratio in `[MIN_JET_RATIO, MAX_JET_RATIO]`.
pub const DEFAULT_JET_RATIO: Sample = 0.2;

/// Lowest jet ratio (jet transit as a fraction of the bore delay).
pub const MIN_JET_RATIO: Sample = 0.08;

/// Highest jet ratio.
pub const MAX_JET_RATIO: Sample = 0.9;

/// Default brightness in `[0, 1]`.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Default normalized breath turbulence in `[0, 1]`.
pub const DEFAULT_BREATH_NOISE: Sample = 0.04;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Default PRNG seed for the breath-turbulence stream.
pub const DEFAULT_SEED: u64 = 0x5EED_F10E_1234_ABCD;

/// Internal scale mapping normalized breath pressure `[0, 1]` to bore units.
const MAX_BREATH_PRESSURE: Sample = 1.0;

/// Magnitude of the inverting pressure reflection at the (open) embouchure end.
/// Unity keeps the embouchure reflection lossless so the jet alone controls the
/// loop gain; the far-end loss filter provides the only passive damping.
const EMB_REFLECTION: Sample = 1.0;

/// Loss-filter loop gain at the open far end, just below unity so the passive
/// loop decays while the jet supplies the sustaining energy.
const LOSS_GAIN: Sample = 0.995;

/// Jet drive gain: scales the cubic edge-tone flow injected into the bore. Set
/// just high enough that the small-signal round-trip gain exceeds unity, so the
/// tone self-starts and then grows until the cubic saturates into a stable
/// limit cycle.
const JET_DRIVE: Sample = 1.0;

/// Maps the convectively delayed embouchure pressure drop onto the cubic jet
/// characteristic's argument (the normalized jet-to-labium offset).
const JET_OFFSET: Sample = 1.0;

/// Normalizes the internal limit-cycle pressure to roughly unity peak before the
/// user amplitude so a default voice stays well within `[-1, 1]`.
const OUTPUT_GAIN: Sample = 0.3;

/// Peak turbulence amplitude relative to the breath drive. The breath noise
/// breaks the initial symmetry so the oscillation starts.
const TURBULENCE_SCALE: Sample = 0.5;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Clamps `frequency_hz` to `[MIN_FREQUENCY_HZ, sample_rate / 2]`, falling back
/// to [`MIN_FREQUENCY_HZ`] for non-finite input.
fn sanitize_frequency(frequency_hz: Sample, sample_rate: Sample) -> Sample {
    let nyquist = (sample_rate * 0.5).max(MIN_FREQUENCY_HZ);
    finite_or(frequency_hz, MIN_FREQUENCY_HZ).clamp(MIN_FREQUENCY_HZ, nyquist)
}

/// Self-contained deterministic PRNG (Marsaglia xorshift64) seeded via
/// `SplitMix64`, matching the generator in [`super::noise`].
#[derive(Debug, Clone, Copy)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: seed_to_state(seed) }
    }

    #[inline]
    fn next_bipolar(&mut self) -> Sample {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
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
    if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z }
}

/// A linear-interpolating waveguide delay line (a travelling bore/jet segment).
#[derive(Debug, Clone)]
struct WaveguideDelay {
    buf: Vec<Sample>,
    write: usize,
}

impl WaveguideDelay {
    fn new(capacity: usize) -> Self {
        Self { buf: vec![0.0; capacity.max(2)], write: 0 }
    }

    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "delay is clamped to [1, len-1]; the integer part fits a usize exactly"
    )]
    fn read(&self, delay: Sample) -> Sample {
        let len = self.buf.len();
        let d = delay.clamp(1.0, (len - 1) as Sample);
        let di = d as usize;
        let frac = d - di as Sample;
        let i0 = (self.write + len - di) % len;
        let i1 = (self.write + len - di - 1) % len;
        self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac
    }

    #[inline]
    fn write_sample(&mut self, x: Sample) {
        self.buf[self.write] = flush_denormal(x);
        self.write += 1;
        if self.write == self.buf.len() {
            self.write = 0;
        }
    }

    fn clear(&mut self) {
        for v in &mut self.buf {
            *v = 0.0;
        }
        self.write = 0;
    }
}

/// Construction parameters for an [`AirJetFluteNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AirJetFluteParams {
    /// Fundamental (pitch) frequency in hertz.
    pub frequency_hz: Sample,
    /// Normalized breath pressure in `[0, 1]` (loudness / drive).
    pub breath_pressure: Sample,
    /// Jet/embouchure ratio in `[MIN_JET_RATIO, MAX_JET_RATIO]` (regime).
    pub jet_ratio: Sample,
    /// Brightness in `[0, 1]` (loss-filter high-frequency damping).
    pub brightness: Sample,
    /// Normalized breath turbulence in `[0, 1]` (attack noise / breathiness).
    pub breath_noise: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
    /// Seed for the deterministic breath-turbulence stream.
    pub seed: u64,
}

impl Default for AirJetFluteParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            breath_pressure: DEFAULT_BREATH_PRESSURE,
            jet_ratio: DEFAULT_JET_RATIO,
            brightness: DEFAULT_BRIGHTNESS,
            breath_noise: DEFAULT_BREATH_NOISE,
            amplitude: DEFAULT_AMPLITUDE,
            seed: DEFAULT_SEED,
        }
    }
}

/// An air-jet flute digital-waveguide voice source node (0 inputs, 1 output).
///
/// The mono bore signal is replicated into every output channel.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{AirJetFluteNode, AirJetFluteParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = AirJetFluteNode::new(48_000, AirJetFluteParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The jet drives the bore into sustained self-oscillation.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct AirJetFluteNode {
    /// Sample rate the delay lines were sized for.
    sample_rate: Sample,
    /// Fundamental frequency in hertz.
    frequency_hz: Sample,
    /// Jet/embouchure ratio in `[MIN_JET_RATIO, MAX_JET_RATIO]`.
    jet_ratio: Sample,
    /// Brightness in `[0, 1]`.
    brightness: Sample,
    /// Normalized breath turbulence in `[0, 1]`.
    breath_noise: Sample,
    /// Smoothed normalized breath pressure in `[0, 1]`.
    breath_pressure: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,

    /// Embouchure-side travelling-wave bore delay line.
    bore_line: WaveguideDelay,
    /// Open-far-end-side travelling-wave bore delay line.
    end_line: WaveguideDelay,
    /// Jet convective-flow delay line (embouchure edge-tone transit).
    jet_line: WaveguideDelay,
    /// Previous loss-filter output (`z^-1`) for the one-pole low-pass.
    loss_filter_z1: Sample,

    /// Deterministic breath-turbulence generator.
    rng: Xorshift64,
    /// Seed the turbulence generator was constructed/reseeded with.
    seed: u64,

    /// Latched bore delay in samples.
    bore_delay: Sample,
    /// Latched jet delay in samples.
    jet_delay: Sample,
    /// Latched one-zero loss-filter coefficient `S`.
    damping_s: Sample,
}

impl AirJetFluteNode {
    /// Builds an air-jet flute for `sample_rate` Hz from a parameter bundle.
    ///
    /// The delay lines are sized so a fundamental as low as [`MIN_FREQUENCY_HZ`]
    /// fits. All parameters are sanitized: non-finite values fall back to safe
    /// defaults, `frequency_hz` is clamped to `[MIN_FREQUENCY_HZ, sample_rate /
    /// 2]`, `jet_ratio` to `[MIN_JET_RATIO, MAX_JET_RATIO]`, and
    /// `breath_pressure`/`brightness`/`breath_noise` to `[0, 1]`.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "the buffer length is tiny relative to f32's integer precision"
    )]
    pub fn new(sample_rate: u32, params: AirJetFluteParams) -> Self {
        let sr = (sample_rate.max(1)) as Sample;
        // Longest bore delay (frames at the lowest pitch): bore_delay = sr / f0.
        let max_delay_frames = ops::round(sr / MIN_FREQUENCY_HZ) as usize;
        let capacity = max_delay_frames + 4;

        let frequency_hz = sanitize_frequency(params.frequency_hz, sr);
        let jet_ratio = finite_or(params.jet_ratio, DEFAULT_JET_RATIO).clamp(MIN_JET_RATIO, MAX_JET_RATIO);
        let brightness = finite_or(params.brightness, DEFAULT_BRIGHTNESS).clamp(0.0, 1.0);
        let breath_noise = finite_or(params.breath_noise, DEFAULT_BREATH_NOISE).clamp(0.0, 1.0);
        let breath_pressure = finite_or(params.breath_pressure, DEFAULT_BREATH_PRESSURE).clamp(0.0, 1.0);
        let amplitude = finite_or(params.amplitude, DEFAULT_AMPLITUDE);

        let mut node = Self {
            sample_rate: sr,
            frequency_hz,
            jet_ratio,
            brightness,
            breath_noise,
            breath_pressure: Smoothed::new(breath_pressure),
            amplitude: Smoothed::new(amplitude),
            bore_line: WaveguideDelay::new(capacity),
            end_line: WaveguideDelay::new(capacity),
            jet_line: WaveguideDelay::new(capacity),
            loss_filter_z1: 0.0,
            rng: Xorshift64::new(params.seed),
            seed: params.seed,
            bore_delay: 1.0,
            jet_delay: 1.0,
            damping_s: 0.25,
        };
        node.recompute();
        node
    }

    /// Retunes the bore to `frequency_hz` (clamped to
    /// `[MIN_FREQUENCY_HZ, sample_rate / 2]`).
    #[inline]
    pub fn set_frequency_hz(&mut self, frequency_hz: Sample) {
        let requested = finite_or(frequency_hz, self.frequency_hz);
        self.frequency_hz = sanitize_frequency(requested, self.sample_rate);
        self.recompute();
    }

    /// Sets the jet/embouchure ratio in `[MIN_JET_RATIO, MAX_JET_RATIO]`.
    #[inline]
    pub fn set_jet_ratio(&mut self, jet_ratio: Sample) {
        self.jet_ratio = finite_or(jet_ratio, self.jet_ratio).clamp(MIN_JET_RATIO, MAX_JET_RATIO);
        self.recompute();
    }

    /// Sets the brightness in `[0, 1]`.
    #[inline]
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the normalized breath turbulence in `[0, 1]`.
    #[inline]
    pub fn set_breath_noise(&mut self, breath_noise: Sample) {
        self.breath_noise = finite_or(breath_noise, self.breath_noise).clamp(0.0, 1.0);
    }

    /// Sets the normalized breath pressure in `[0, 1]`, smoothing over `ramp`.
    #[inline]
    pub fn set_breath_pressure(&mut self, breath_pressure: Sample, ramp: Ramp) {
        let v = finite_or(breath_pressure, self.breath_pressure.target()).clamp(0.0, 1.0);
        self.breath_pressure.set_target(v, ramp);
    }

    /// Sets a new target output amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Reseeds the breath-turbulence generator, restarting the stream.
    #[inline]
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = Xorshift64::new(seed);
    }

    /// Returns the fundamental frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the jet/embouchure ratio.
    #[inline]
    #[must_use]
    pub fn jet_ratio(&self) -> Sample {
        self.jet_ratio
    }

    /// Returns the brightness in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
    }

    /// Returns the normalized breath turbulence in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn breath_noise(&self) -> Sample {
        self.breath_noise
    }

    /// Returns the target normalized breath pressure in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn breath_pressure(&self) -> Sample {
        self.breath_pressure.target()
    }

    /// Returns the target output amplitude (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Returns the current turbulence seed.
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Recomputes the latched loop coefficients from the user-facing parameters.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the buffer length is tiny relative to f32's integer precision"
    )]
    fn recompute(&mut self) {
        let sr = self.sample_rate;
        // One-pole loss-filter coefficient S in [0, 0.5]: brightness 1 -> S 0.
        let s = 0.5 * (1.0 - self.brightness);
        self.damping_s = s;

        // Open-open bore: the round trip (2 * bore_delay) carries two inverting
        // reflections, so there is no net inversion and the tube resonates on
        // its full harmonic series at f0 = sr / (2 * bore_delay). The one-zero
        // loss filter adds a low-frequency group delay of about s samples per
        // round trip; fold half of it into each one-way bore delay so the
        // sounding pitch stays accurate.
        let max_delay = (self.bore_line.buf.len() - 2) as Sample;
        let bore_delay = (sr / (2.0 * self.frequency_hz) - 0.5 * s).clamp(1.0, max_delay);
        self.bore_delay = bore_delay;
        // Jet convective transit as a fraction of the one-way bore delay.
        self.jet_delay = (bore_delay * self.jet_ratio).clamp(1.0, max_delay);
    }

    /// Applies the one-zero low-pass loss reflection filter at the open far end
    /// and advances its memory. Storing the input (not the output) makes it a
    /// stable one-zero FIR whose group delay is about `s` samples.
    #[inline]
    fn loss_filter(&mut self, x: Sample) -> Sample {
        let s = self.damping_s;
        let y = LOSS_GAIN * ((1.0 - s) * x + s * self.loss_filter_z1);
        self.loss_filter_z1 = x;
        y
    }

    /// Advances the waveguide by one sample and returns the radiated output.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let breath = self.breath_pressure.next_sample() * MAX_BREATH_PRESSURE;
        let turbulence = self.rng.next_bipolar() * breath * self.breath_noise * TURBULENCE_SCALE;

        // Travelling pressure waves arriving at each end of the open-open bore.
        let at_emb = self.bore_line.read(self.bore_delay);
        let at_end = self.end_line.read(self.bore_delay);

        // Open far end: inverting, lossy low-pass reflection launched back
        // toward the embouchure.
        let end_refl = -self.loss_filter(at_end);

        // Jet drive: the breath forms a jet whose offset across the labium is
        // set by the pressure drop (breath minus the bore wave) delayed by the
        // convective jet transit. A saturating cubic turns that offset into the
        // volume flow injected at the embouchure.
        let drop = (breath + turbulence) - at_emb;
        let jet_sensed = self.jet_line.read(self.jet_delay);
        self.jet_line.write_sample(drop);
        let x = (jet_sensed * JET_OFFSET).clamp(-1.0, 1.0);
        let jet = JET_DRIVE * (x - x * x * x);

        // Open embouchure end: inverting reflection plus the jet injection,
        // launched back toward the far end.
        let emb_refl = -EMB_REFLECTION * at_emb + jet;

        // Cross-couple the two travelling-wave rails.
        self.end_line.write_sample(emb_refl);
        self.bore_line.write_sample(end_refl);

        flush_denormal(at_emb * OUTPUT_GAIN * self.amplitude.next_sample())
    }
}

impl AudioNode for AirJetFluteNode {
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
        self.bore_line.clear();
        self.end_line.clear();
        self.jet_line.clear();
        self.loss_filter_z1 = 0.0;
        self.rng = Xorshift64::new(self.seed);
        self.breath_pressure = Smoothed::new(self.breath_pressure.target());
        self.amplitude = Smoothed::new(self.amplitude.target());
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

    const SR: u32 = 48_000;

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut AirJetFluteNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout` and returns per-channel
    /// sample vectors.
    fn render_layout(
        node: &mut AirJetFluteNode,
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
        block.iter().map(|&s| f64::from(s) * f64::from(s)).sum()
    }

    /// Magnitude of the Goertzel estimate at frequency `f` over `block`.
    fn goertzel(block: &[Sample], f: Sample, sr: Sample) -> f64 {
        let w = core::f32::consts::TAU * f / sr;
        let c = 2.0 * ops::cos(w);
        let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
        for &x in block {
            let s0 = f64::from(x) + f64::from(c) * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - f64::from(c) * s1 * s2).sqrt()
    }

    #[test]
    fn default_self_oscillates() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        let block = render(&mut node, SR as usize);
        // Steady breath pumps the jet into a sustained tone, not a decaying pluck.
        assert!(peak(&block) > 0.0);
        assert!(peak(&block[SR as usize / 2..]) > 1e-2);
        assert!(block.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn long_run_stays_bounded() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        let block = render(&mut node, 4 * SR as usize);
        assert!(block.iter().all(|s| s.is_finite()));
        // The clamped cubic jet plus sub-unity far-end loss bound the loop; it
        // must never run away.
        assert!(peak(&block) < 1.0);
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        let mut b = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        assert_eq!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn reset_restarts_identical_attack() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        let first = render(&mut node, 8192);
        node.reset();
        let second = render(&mut node, 8192);
        assert_eq!(first, second);
    }

    #[test]
    fn zero_breath_pressure_is_silent() {
        let params = AirJetFluteParams {
            breath_pressure: 0.0,
            ..AirJetFluteParams::default()
        };
        let mut node = AirJetFluteNode::new(SR, params);
        // With no breath the turbulence vanishes and `drop = -at_emb`; starting
        // from silence the loop can never leave zero, so the bore stays silent.
        let block = render(&mut node, SR as usize / 2);
        assert!(block.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn amplitude_scales_output_energy() {
        // Disable breath noise so both voices share the same deterministic drive
        // and only the pure output gain differs.
        let loud = AirJetFluteParams {
            amplitude: 1.0,
            breath_noise: 0.0,
            ..AirJetFluteParams::default()
        };
        let soft = AirJetFluteParams {
            amplitude: 0.5,
            breath_noise: 0.0,
            ..AirJetFluteParams::default()
        };
        let mut a = AirJetFluteNode::new(SR, loud);
        let mut b = AirJetFluteNode::new(SR, soft);
        let ea = energy(&render(&mut a, 8192));
        let eb = energy(&render(&mut b, 8192));
        // Amplitude only scales the radiated signal; halving it quarters energy.
        assert!(eb > 0.0);
        assert!((ea / eb - 4.0).abs() < 1e-3);
    }

    #[test]
    fn full_harmonic_series_present() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        let block = render(&mut node, 2 * SR as usize);
        // Analyze the settled second half so the attack transient is excluded.
        let tail = &block[SR as usize..];
        let f0 = DEFAULT_FREQUENCY_HZ;
        let h1 = goertzel(tail, f0, SR as Sample);
        let h2 = goertzel(tail, 2.0 * f0, SR as Sample);
        let h3 = goertzel(tail, 3.0 * f0, SR as Sample);
        // The non-inverting open-open loop supports every integer harmonic, so
        // the even partials survive (unlike the clarinet's odd-only series).
        assert!(h1 > 1.0, "weak fundamental: {h1}");
        assert!(h2 > 1.0, "missing second harmonic: {h2}");
        assert!(h3 > 1.0, "missing third harmonic: {h3}");
        // The fundamental dominates its own partials for a flute-like timbre.
        assert!(h1 > h3, "fundamental should lead the third: {h1} vs {h3}");
    }

    #[test]
    fn pitch_matches_fundamental() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        let block = render(&mut node, 2 * SR as usize);
        let tail = &block[SR as usize..];
        let f0 = DEFAULT_FREQUENCY_HZ;
        // Energy must concentrate at f0 rather than a half-octave neighbour.
        let at_f0 = goertzel(tail, f0, SR as Sample);
        let at_half = goertzel(tail, 0.5 * f0, SR as Sample);
        let at_third = goertzel(tail, 1.5 * f0, SR as Sample);
        assert!(at_f0 > at_half, "subharmonic leaked: {at_f0} vs {at_half}");
        assert!(at_f0 > at_third, "off-pitch regime: {at_f0} vs {at_third}");
    }

    #[test]
    fn mono_core_replicated_to_all_channels() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        let channels = render_layout(&mut node, 4096, ChannelLayout::Stereo);
        assert_eq!(channels.len(), 2);
        assert_eq!(channels[0], channels[1]);
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        let block = render(&mut node, 0);
        assert!(block.is_empty());
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = AirJetFluteParams {
            frequency_hz: 523.25,
            breath_pressure: 0.7,
            jet_ratio: 0.3,
            brightness: 0.6,
            breath_noise: 0.1,
            amplitude: 0.4,
            seed: 0x1234_5678_9ABC_DEF0,
        };
        let node = AirJetFluteNode::new(SR, params);
        assert!((node.frequency_hz() - 523.25).abs() < 1e-3);
        assert!((node.breath_pressure() - 0.7).abs() < 1e-6);
        assert!((node.jet_ratio() - 0.3).abs() < 1e-6);
        assert!((node.brightness() - 0.6).abs() < 1e-6);
        assert!((node.breath_noise() - 0.1).abs() < 1e-6);
        assert!((node.amplitude() - 0.4).abs() < 1e-6);
        assert_eq!(node.seed(), 0x1234_5678_9ABC_DEF0);
    }

    #[test]
    fn set_frequency_clamps_to_range() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        node.set_frequency_hz(1.0);
        assert!((node.frequency_hz() - MIN_FREQUENCY_HZ).abs() < 1e-6);
        node.set_frequency_hz(1.0e9);
        assert!((node.frequency_hz() - SR as Sample * 0.5).abs() < 1e-3);
    }

    #[test]
    fn set_jet_ratio_clamps_to_range() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        node.set_jet_ratio(5.0);
        assert!((node.jet_ratio() - MAX_JET_RATIO).abs() < 1e-6);
        node.set_jet_ratio(-5.0);
        assert!((node.jet_ratio() - MIN_JET_RATIO).abs() < 1e-6);
    }

    #[test]
    fn set_brightness_and_breath_noise_clamp() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        node.set_brightness(9.0);
        assert!((node.brightness() - 1.0).abs() < 1e-6);
        node.set_brightness(-9.0);
        assert!((node.brightness() - 0.0).abs() < 1e-6);
        node.set_breath_noise(9.0);
        assert!((node.breath_noise() - 1.0).abs() < 1e-6);
        node.set_breath_noise(-9.0);
        assert!((node.breath_noise() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn set_breath_pressure_clamps_to_range() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        node.set_breath_pressure(5.0, Ramp::Immediate);
        assert!((node.breath_pressure() - 1.0).abs() < 1e-6);
        node.set_breath_pressure(-5.0, Ramp::Immediate);
        assert!((node.breath_pressure() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn setters_reject_non_finite() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        let f = node.frequency_hz();
        let jr = node.jet_ratio();
        let b = node.brightness();
        let bn = node.breath_noise();
        let bp = node.breath_pressure();
        let a = node.amplitude();
        node.set_frequency_hz(Sample::NAN);
        node.set_jet_ratio(Sample::INFINITY);
        node.set_brightness(Sample::NEG_INFINITY);
        node.set_breath_noise(Sample::NAN);
        node.set_breath_pressure(Sample::NAN, Ramp::Immediate);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.frequency_hz(), f);
        assert_eq!(node.jet_ratio(), jr);
        assert_eq!(node.brightness(), b);
        assert_eq!(node.breath_noise(), bn);
        assert_eq!(node.breath_pressure(), bp);
        assert_eq!(node.amplitude(), a);
    }

    #[test]
    fn constructor_sanitizes_non_finite_params() {
        let params = AirJetFluteParams {
            frequency_hz: Sample::NAN,
            breath_pressure: Sample::INFINITY,
            jet_ratio: Sample::NAN,
            brightness: Sample::NEG_INFINITY,
            breath_noise: Sample::NAN,
            amplitude: Sample::NAN,
            seed: DEFAULT_SEED,
        };
        let node = AirJetFluteNode::new(SR, params);
        assert!(node.frequency_hz().is_finite());
        assert!(node.breath_pressure().is_finite());
        assert!(node.jet_ratio().is_finite());
        assert!(node.brightness().is_finite());
        assert!(node.breath_noise().is_finite());
        assert!(node.amplitude().is_finite());
    }

    #[test]
    fn frequency_changes_output() {
        let low = AirJetFluteParams {
            frequency_hz: 220.0,
            ..AirJetFluteParams::default()
        };
        let high = AirJetFluteParams {
            frequency_hz: 880.0,
            ..AirJetFluteParams::default()
        };
        let mut a = AirJetFluteNode::new(SR, low);
        let mut b = AirJetFluteNode::new(SR, high);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn jet_ratio_changes_timbre() {
        let narrow = AirJetFluteParams {
            jet_ratio: 0.12,
            ..AirJetFluteParams::default()
        };
        let wide = AirJetFluteParams {
            jet_ratio: 0.5,
            ..AirJetFluteParams::default()
        };
        let mut a = AirJetFluteNode::new(SR, narrow);
        let mut b = AirJetFluteNode::new(SR, wide);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn brightness_changes_output() {
        let dark = AirJetFluteParams {
            brightness: 0.05,
            ..AirJetFluteParams::default()
        };
        let bright = AirJetFluteParams {
            brightness: 0.95,
            ..AirJetFluteParams::default()
        };
        let mut a = AirJetFluteNode::new(SR, dark);
        let mut b = AirJetFluteNode::new(SR, bright);
        // The far-end loss coefficient S tracks brightness, so the radiated
        // waveform must differ.
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn seed_changes_turbulence_stream() {
        let a_params = AirJetFluteParams {
            seed: 0x1111_2222_3333_4444,
            ..AirJetFluteParams::default()
        };
        let b_params = AirJetFluteParams {
            seed: 0xAAAA_BBBB_CCCC_DDDD,
            ..AirJetFluteParams::default()
        };
        let mut a = AirJetFluteNode::new(SR, a_params);
        let mut b = AirJetFluteNode::new(SR, b_params);
        // Different seeds drive different turbulence, so the attacks diverge even
        // though each remains fully deterministic.
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn high_and_low_frequencies_both_oscillate() {
        for &f in &[MIN_FREQUENCY_HZ, 1_500.0] {
            let params = AirJetFluteParams {
                frequency_hz: f,
                ..AirJetFluteParams::default()
            };
            let mut node = AirJetFluteNode::new(SR, params);
            let block = render(&mut node, SR as usize);
            assert!(peak(&block) > 0.0, "silent at {f} Hz");
            assert!(block.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn breath_pressure_target_tracks_setter() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        node.set_breath_pressure(0.8, Ramp::linear_seconds(0.01, SR));
        assert!((node.breath_pressure() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn amplitude_target_tracks_setter() {
        let mut node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        node.set_amplitude(0.25, Ramp::linear_seconds(0.01, SR));
        assert!((node.amplitude() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn latency_is_zero() {
        let node = AirJetFluteNode::new(SR, AirJetFluteParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
