//! Brass lip-reed digital-waveguide physical-modeling source node.
//!
//! [`BrassLipReedNode`] is a *source* (zero inputs, one output) that synthesizes
//! a sustained, self-oscillating brass tone (trumpet / trombone / horn family)
//! by modeling a flaring bore excited by the player's buzzing lips. Like its
//! wind siblings [`super::reed_woodwind::ReedWoodwindNode`] and
//! [`super::air_jet_flute::AirJetFluteNode`] it is *continuously driven*: a
//! steady breath feeds a mechanically resonant lip valve whose nonlinear flow
//! pumps energy into the bore every sample, so the tone sustains as long as the
//! player blows.
//!
//! # The bore and the lips
//!
//! The bore is modelled as a bidirectional digital waveguide: two
//! linear-interpolating delay lines carry the pressure waves travelling in
//! opposite directions between the mouthpiece and the open (flaring) bell:
//!
//! ```text
//! at_mp   = mp_line.read(bore_delay)    (wave arriving at the mouthpiece)
//! at_bell = bell_line.read(bore_delay)  (wave arriving at the flaring bell)
//! ```
//!
//! A real brass bore is nearly cylindrical but flares into a bell whose horn
//! correction shifts the air-column resonances onto an (almost) complete
//! harmonic series: a bugle plays `do-mi-sol-do`, every integer harmonic of the
//! pedal tone. The model captures that with an *open-open* loop (two inverting
//! end reflections cancel over a round trip, leaving no net inversion), so the
//! bore resonates on its full harmonic series and `f0 = sample_rate /
//! (2 * bore_delay)`, an octave above a cylindrical closed-open tube of the same
//! length. A one-zero low-pass loss filter with loop gain just below unity
//! models the frequency-dependent radiation and wall losses at the bell.
//!
//! # The lip valve (outward-striking reed)
//!
//! Unlike the clarinet's effectively static, *inward*-striking cane reed, the
//! brass player's lips are a mass-spring mechanical oscillator: an
//! *outward*-striking pressure valve with its own mechanical resonance. That
//! lip resonance is what the player tunes (by lip tension and embouchure) to
//! select which bore partial sounds -- the physical mechanism behind bugle
//! calls and trombone lip slurs. The lip is modelled as a damped second-order
//! resonator driven by the *oscillating* part of the pressure drop across the
//! lips:
//!
//! ```text
//! dp       = breath + turbulence - at_mp       (pressure across the lips)
//! drive_in = dp - dp_z1                        (high-pass: zero at DC)
//! lip      = a1 * lip_z1 + a2 * lip_z2 + LIP_DRIVE * drive_in   (resonant lip motion)
//! opening  = clamp(LIP_REST + lip, 0, 1)       (lip aperture; closes at 0)
//! flow     = opening * clamp(dp, -FLOW_CLIP, FLOW_CLIP)   (clamped-linear flow)
//! mp_refl  = -at_mp + FLOW_GAIN * flow         (mouthpiece: inverting + lip flow)
//! bell_refl = -loss_filter(at_bell)            (open bell: inverting, lossy)
//! bell_line.write(mp_refl)                     (cross-couple the two rails)
//! mp_line.write(bell_refl)
//! ```
//!
//! The resonator poles sit at radius `LIP_POLE_RADIUS` and angle `2*pi*f_lip /
//! sample_rate`, where the lip resonance `f_lip = frequency_hz * lip_ratio` is
//! set by the `lip_tension` control. The resonator is driven by the *differenced*
//! pressure drop (`dp - dp_z1`), a one-zero high-pass whose zero at DC stops a
//! steady breath from railing the aperture; the lips then respond only to the
//! acoustic pressure and the mechanical resonance narrows the loop gain around
//! `f_lip`, so the oscillation locks onto whichever bore partial lies nearest.
//! Raising `lip_tension` walks `f_lip` up the harmonic series, climbing partial
//! by partial exactly like a bugle call or a trombone lip slur. The volume flow
//! is linear in the pressure drop for small signals -- the negative resistance
//! that starts the tone -- but `clamp(dp, -FLOW_CLIP, FLOW_CLIP)` makes it
//! constant once the lips are driven hard; that saturation, together with the
//! sub-unity bell loop gain, bounds the limit cycle and sets its amplitude, and
//! the lip aperture is independently clamped to `[0, 1]` so the lips can close
//! but never inject unbounded flow. The onset of the steady breath (and a small
//! amount of deterministic breath turbulence) breaks the initial symmetry so the
//! oscillation starts, exactly as real breath initiates a buzz.
//!
//! # Pitch and timbre
//!
//! The sounding fundamental is `f0 = sample_rate / (2 * bore_delay)` (minus half
//! the loss filter's small group delay, folded in for tuning accuracy).
//! `lip_tension` sets the lip resonance ratio and selects the oscillation
//! regime (which harmonic the lips lock onto, and bends the pitch as a player
//! lips up or down); `brightness` sets the loss-filter coefficient `S = 0.5 *
//! (1 - brightness)` exactly as in the Karplus-Strong loop; `breath` sets
//! loudness and, through the clamping lip valve, the brassy growth of the upper
//! harmonics. The radiated signal is the bell wave scaled by a fixed
//! `OUTPUT_GAIN` normalization and the user amplitude.
//!
//! # Determinism
//!
//! The breath turbulence is a self-contained seeded `xorshift64` stream, so two
//! [`BrassLipReedNode`]s built with the same sample rate, parameters, and seed
//! emit bit-identical streams on every platform via [`bevy_math::ops`], and
//! [`AudioNode::reset`] reseeds the generator, clears the lines, and zeroes the
//! lip resonator to restart the identical attack.
//!
//! # Relationship
//!
//! This is the *lip-driven* sibling of the *reed-driven*
//! [`super::reed_woodwind::ReedWoodwindNode`], the *jet-driven*
//! [`super::air_jet_flute::AirJetFluteNode`], and the *bow-driven*
//! [`super::bowed_string::BowedStringNode`]: all recirculate energy in a tuned,
//! loss-filtered delay loop with fractional-delay tuning, but the brass is
//! sustained by a *resonant* outward-striking lip valve rather than a static
//! reed, a massless jet, or bow friction. The lip's own mechanical resonance
//! selects the bore partial, which no other wind sibling does. It reuses this
//! crate's own [`Sample`] type, [`Smoothed`] parameter smoother,
//! denormal-flushing primitive, and the public-domain `xorshift64`/`SplitMix64`
//! PRNG shared with [`super::noise::NoiseNode`].
//!
//! # Real-time contract
//!
//! Both delay lines are pre-allocated in [`BrassLipReedNode::new`] for the
//! lowest supported pitch, so [`process`](crate::graph::AudioNode::process)
//! performs no allocation, takes no locks, and cannot panic: every recirculated
//! sample is denormal-flushed, the bore loop gain is below unity, the lip
//! aperture is clamped to `[0, 1]`, and non-finite parameters are rejected at
//! the setters. The breath pressure and output amplitude are driven through
//! [`Smoothed`] values so performance gestures never zipper.
//!
//! # Provenance
//!
//! The lip-reed and bore model follow the standard public digital-waveguide
//! treatment of brass instruments in J. O. Smith's *Physical Audio Signal
//! Processing* (public online text) and the meta-wind-instrument formulation of
//! P. R. Cook ("A meta-wind-instrument physical model", ICMC 1992), built on the
//! outward-striking lip-valve physics of S. J. Elliott and J. M. Bowsher ("Regeneration
//! in brass wind instruments", *Journal of Sound and Vibration*, 1982) and the
//! reed/valve nonlinearity of `McIntyre`, Schumacher and Woodhouse ("On the
//! oscillations of musical instruments", *Journal of the Acoustical Society of
//! America*, 1983). The one-pole loss filter and `brightness` mapping are shared
//! with the Jaffe-Smith Karplus-Strong extensions; the PRNG is the
//! public-domain `xorshift64` seeded by `SplitMix64` (S. Vigna). This module
//! contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google
//! Resonance Audio, or Web Audio source or derived code**, and nothing from the
//! STK or any other audio toolkit's implementation; it is written purely from
//! that publicly documented theory.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Lowest tunable fundamental in hertz. Bounds the pre-allocated delay lines.
pub const MIN_FREQUENCY_HZ: Sample = 40.0;

/// Default fundamental (pitch) frequency in hertz (concert B-flat-ish A4).
pub const DEFAULT_FREQUENCY_HZ: Sample = 440.0;

/// Default normalized breath pressure in `[0, 1]`.
pub const DEFAULT_BREATH_PRESSURE: Sample = 0.5;

/// Default normalized lip tension in `[0, 1]` (selects the playing regime).
pub const DEFAULT_LIP_TENSION: Sample = 0.5;

/// Default brightness in `[0, 1]`.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Default normalized breath turbulence in `[0, 1]`.
pub const DEFAULT_BREATH_NOISE: Sample = 0.04;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Default PRNG seed for the breath-turbulence stream.
pub const DEFAULT_SEED: u64 = 0x8A55_11D0_C0DE_FACE;

/// Lowest lip-resonance ratio (lip resonance / sounding fundamental).
const MIN_LIP_RATIO: Sample = 0.9;

/// Highest lip-resonance ratio.
const MAX_LIP_RATIO: Sample = 3.2;

/// Internal scale mapping normalized breath pressure `[0, 1]` to bore units.
const MAX_BREATH_PRESSURE: Sample = 1.0;

/// Loss-filter loop gain at the open bell, just below unity so the passive loop
/// decays while the lip supplies the sustaining energy.
const LOSS_GAIN: Sample = 0.99;

/// Pole radius of the second-order lip resonator (just inside the unit circle so
/// the lips have a sharp but finite mechanical Q).
const LIP_POLE_RADIUS: Sample = 0.99;

/// Drive gain feeding the pressure drop into the lip resonator.
const LIP_DRIVE: Sample = 1.5;

/// Static (rest) lip aperture, offset so the lips idle partly open.
const LIP_REST: Sample = 0.5;

/// Flow gain: scales the lip volume flow injected into the bore. Set just high
/// enough that the small-signal round-trip gain exceeds unity, so the tone
/// self-starts and then grows until the flow clamp ([`FLOW_CLIP`]) saturates it.
const FLOW_GAIN: Sample = 1.0;

/// Saturation limit on the pressure drop seen by the lip volume flow. Clamping
/// it makes the active injection constant once the lips are driven hard, which
/// both bounds the limit cycle and sets its amplitude together with the bell
/// loss.
const FLOW_CLIP: Sample = 0.02;

/// Normalizes the internal limit-cycle pressure to roughly unity peak before the
/// user amplitude so a default voice stays well within `[-1, 1]`.
const OUTPUT_GAIN: Sample = 0.4;

/// Peak turbulence amplitude relative to the breath drive.
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

/// A linear-interpolating waveguide delay line (a travelling bore segment).
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

/// Construction parameters for a [`BrassLipReedNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BrassLipReedParams {
    /// Fundamental (pitch) frequency in hertz.
    pub frequency_hz: Sample,
    /// Normalized breath pressure in `[0, 1]` (loudness / drive).
    pub breath_pressure: Sample,
    /// Normalized lip tension in `[0, 1]` (regime / partial selection).
    pub lip_tension: Sample,
    /// Brightness in `[0, 1]` (loss-filter high-frequency damping).
    pub brightness: Sample,
    /// Normalized breath turbulence in `[0, 1]` (attack noise / breathiness).
    pub breath_noise: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
    /// Seed for the deterministic breath-turbulence stream.
    pub seed: u64,
}

impl Default for BrassLipReedParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            breath_pressure: DEFAULT_BREATH_PRESSURE,
            lip_tension: DEFAULT_LIP_TENSION,
            brightness: DEFAULT_BRIGHTNESS,
            breath_noise: DEFAULT_BREATH_NOISE,
            amplitude: DEFAULT_AMPLITUDE,
            seed: DEFAULT_SEED,
        }
    }
}

/// A brass lip-reed digital-waveguide voice source node (0 inputs, 1 output).
///
/// The mono bell signal is replicated into every output channel.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{BrassLipReedNode, BrassLipReedParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = BrassLipReedNode::new(48_000, BrassLipReedParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The buzzing lips drive the bore into sustained self-oscillation.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct BrassLipReedNode {
    /// Sample rate the delay lines were sized for.
    sample_rate: Sample,
    /// Fundamental frequency in hertz.
    frequency_hz: Sample,
    /// Normalized lip tension in `[0, 1]`.
    lip_tension: Sample,
    /// Brightness in `[0, 1]`.
    brightness: Sample,
    /// Normalized breath turbulence in `[0, 1]`.
    breath_noise: Sample,
    /// Smoothed normalized breath pressure in `[0, 1]`.
    breath_pressure: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,

    /// Mouthpiece-side travelling-wave bore delay line.
    mp_line: WaveguideDelay,
    /// Bell-side travelling-wave bore delay line.
    bell_line: WaveguideDelay,
    /// Previous loss-filter input (`z^-1`) for the one-zero low-pass.
    loss_filter_z1: Sample,

    /// Lip-resonator state `z^-1`.
    lip_z1: Sample,
    /// Lip-resonator state `z^-2`.
    lip_z2: Sample,
    /// Previous pressure drop, used to high-pass the resonator drive so the lips
    /// respond only to the oscillating (not static) part of the breath.
    dp_z1: Sample,

    /// Deterministic breath-turbulence generator.
    rng: Xorshift64,
    /// Seed the turbulence generator was constructed/reseeded with.
    seed: u64,

    /// Latched bore delay in samples.
    bore_delay: Sample,
    /// Latched lip-resonator feedback coefficient `a1 = 2 r cos(w_lip)`.
    lip_a1: Sample,
    /// Latched lip-resonator feedback coefficient `a2 = -r^2`.
    lip_a2: Sample,
    /// Latched one-zero loss-filter coefficient `S`.
    damping_s: Sample,
}

impl BrassLipReedNode {
    /// Builds a brass lip-reed voice for `sample_rate` Hz from a parameter
    /// bundle.
    ///
    /// The delay lines are sized so a fundamental as low as [`MIN_FREQUENCY_HZ`]
    /// fits. All parameters are sanitized: non-finite values fall back to safe
    /// defaults, `frequency_hz` is clamped to `[MIN_FREQUENCY_HZ, sample_rate /
    /// 2]`, and `breath_pressure`/`lip_tension`/`brightness`/`breath_noise` to
    /// `[0, 1]`.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "the buffer length is tiny relative to f32's integer precision"
    )]
    pub fn new(sample_rate: u32, params: BrassLipReedParams) -> Self {
        let sr = (sample_rate.max(1)) as Sample;
        // Longest bore delay (frames at the lowest pitch): bore_delay = sr /
        // (2 * f0).
        let max_delay_frames = ops::round(sr / (2.0 * MIN_FREQUENCY_HZ)) as usize;
        let capacity = max_delay_frames + 4;

        let frequency_hz = sanitize_frequency(params.frequency_hz, sr);
        let lip_tension = finite_or(params.lip_tension, DEFAULT_LIP_TENSION).clamp(0.0, 1.0);
        let brightness = finite_or(params.brightness, DEFAULT_BRIGHTNESS).clamp(0.0, 1.0);
        let breath_noise = finite_or(params.breath_noise, DEFAULT_BREATH_NOISE).clamp(0.0, 1.0);
        let breath_pressure = finite_or(params.breath_pressure, DEFAULT_BREATH_PRESSURE).clamp(0.0, 1.0);
        let amplitude = finite_or(params.amplitude, DEFAULT_AMPLITUDE);

        let mut node = Self {
            sample_rate: sr,
            frequency_hz,
            lip_tension,
            brightness,
            breath_noise,
            breath_pressure: Smoothed::new(breath_pressure),
            amplitude: Smoothed::new(amplitude),
            mp_line: WaveguideDelay::new(capacity),
            bell_line: WaveguideDelay::new(capacity),
            loss_filter_z1: 0.0,
            lip_z1: 0.0,
            lip_z2: 0.0,
            dp_z1: 0.0,
            rng: Xorshift64::new(params.seed),
            seed: params.seed,
            bore_delay: 1.0,
            lip_a1: 0.0,
            lip_a2: 0.0,
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

    /// Sets the normalized lip tension in `[0, 1]`.
    #[inline]
    pub fn set_lip_tension(&mut self, lip_tension: Sample) {
        self.lip_tension = finite_or(lip_tension, self.lip_tension).clamp(0.0, 1.0);
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

    /// Returns the normalized lip tension in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn lip_tension(&self) -> Sample {
        self.lip_tension
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
        // One-zero loss-filter coefficient S in [0, 0.5]: brightness 1 -> S 0.
        let s = 0.5 * (1.0 - self.brightness);
        self.damping_s = s;

        // Open-open bore: the round trip (2 * bore_delay) carries two inverting
        // reflections, so there is no net inversion and the tube resonates on
        // its full harmonic series at f0 = sr / (2 * bore_delay). Fold half the
        // loss filter's group delay into each one-way bore delay for tuning.
        let max_delay = (self.mp_line.buf.len() - 2) as Sample;
        let bore_delay = (sr / (2.0 * self.frequency_hz) - 0.5 * s).clamp(1.0, max_delay);
        self.bore_delay = bore_delay;

        // Lip mechanical resonance: f_lip = f0 * lip_ratio, with lip_ratio
        // selected by lip_tension across [MIN_LIP_RATIO, MAX_LIP_RATIO]. The
        // outward-striking regime locks the bore just below the lip resonance.
        let lip_ratio = MIN_LIP_RATIO + self.lip_tension * (MAX_LIP_RATIO - MIN_LIP_RATIO);
        let f_lip = (self.frequency_hz * lip_ratio).clamp(MIN_FREQUENCY_HZ, sr * 0.49);
        let w = core::f32::consts::TAU * f_lip / sr;
        let r = LIP_POLE_RADIUS;
        self.lip_a1 = 2.0 * r * ops::cos(w);
        self.lip_a2 = -(r * r);
    }

    /// Applies the one-zero low-pass loss reflection filter at the open bell and
    /// advances its memory. Storing the input (not the output) makes it a stable
    /// one-zero FIR whose group delay is about `s` samples.
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
        let at_mp = self.mp_line.read(self.bore_delay);
        let at_bell = self.bell_line.read(self.bore_delay);

        // Open bell: inverting, lossy low-pass reflection launched back toward
        // the mouthpiece.
        let bell_refl = -self.loss_filter(at_bell);

        // Lip valve: a resonant mass-spring oscillator driven by the pressure
        // drop across the lips.
        let dp = (breath + turbulence) - at_mp;
        // Drive the resonator with the high-passed pressure drop (dp - dp_z1):
        // the differencing zero at DC stops a steady breath from railing the
        // aperture, so the lips respond only to the oscillating pressure and the
        // mechanical resonance can select a bore partial.
        let drive_in = dp - self.dp_z1;
        self.dp_z1 = dp;
        let lip = self.lip_a1 * self.lip_z1 + self.lip_a2 * self.lip_z2 + LIP_DRIVE * drive_in;
        self.lip_z2 = self.lip_z1;
        self.lip_z1 = lip;

        // Outward-striking lip aperture: idles partly open and rises with the
        // lip displacement, clamped shut-to-open in [0, 1] so the lips can slap
        // closed but never inject unbounded flow.
        let opening = (LIP_REST + lip).clamp(0.0, 1.0);
        // Lip volume flow: clamped-linear in the pressure drop. Near zero drop
        // it is linear, giving the small-signal negative resistance that starts
        // the tone; once the lips are driven hard the clamp makes the injection
        // constant, which bounds the limit cycle and (with the bell loss) sets
        // its amplitude.
        let drive = dp.clamp(-FLOW_CLIP, FLOW_CLIP);
        let flow = opening * drive;

        // Mouthpiece: inverting passive reflection -- paired with the inverting
        // open bell this leaves no net inversion over a round trip, so the bore
        // sounds the full harmonic series -- plus the lip-flow injection launched
        // back toward the bell.
        let mp_refl = -at_mp + FLOW_GAIN * flow;

        // Cross-couple the two travelling-wave rails.
        self.bell_line.write_sample(mp_refl);
        self.mp_line.write_sample(bell_refl);

        flush_denormal(at_bell * OUTPUT_GAIN * self.amplitude.next_sample())
    }
}

impl AudioNode for BrassLipReedNode {
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
        self.mp_line.clear();
        self.bell_line.clear();
        self.loss_filter_z1 = 0.0;
        self.lip_z1 = 0.0;
        self.lip_z2 = 0.0;
        self.dp_z1 = 0.0;
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

    /// Builds a node at the test sample rate.
    fn node(params: BrassLipReedParams) -> BrassLipReedNode {
        BrassLipReedNode::new(SR, params)
    }

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut BrassLipReedNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout` and returns per-channel
    /// sample vectors.
    fn render_layout(
        node: &mut BrassLipReedNode,
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
    fn default_voice_self_oscillates() {
        let mut node = node(BrassLipReedParams::default());
        let block = render(&mut node, SR as usize);
        // A steady breath pumps the lip valve into a sustained tone that is
        // still ringing in the second half of the block, not a decaying pluck.
        assert!(peak(&block[SR as usize / 2..]) > 1e-2);
        assert!(block.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn long_run_stays_finite_and_bounded() {
        let mut node = node(BrassLipReedParams {
            amplitude: 1.0,
            ..BrassLipReedParams::default()
        });
        let block = render(&mut node, 192_000);
        assert!(block.iter().all(|s| s.is_finite()));
        // The clamped-linear flow plus the sub-unity bell loss cap the limit
        // cycle well inside the normalized output range.
        assert!(peak(&block) < 1.0);
    }

    #[test]
    fn same_params_and_seed_are_bit_identical() {
        let mut a = node(BrassLipReedParams::default());
        let mut b = node(BrassLipReedParams::default());
        assert_eq!(render(&mut a, 4096), render(&mut b, 4096));
    }

    #[test]
    fn reset_restarts_the_identical_attack() {
        let mut node = node(BrassLipReedParams::default());
        let first = render(&mut node, 4096);
        node.reset();
        let second = render(&mut node, 4096);
        assert_eq!(first, second);
    }

    #[test]
    fn zero_breath_is_silent() {
        let mut node = node(BrassLipReedParams {
            breath_pressure: 0.0,
            breath_noise: 0.0,
            ..BrassLipReedParams::default()
        });
        let block = render(&mut node, 4096);
        // With no breath the turbulence (which scales with breath) vanishes and
        // nothing excites the bore, so the output is exactly silent.
        assert!(block.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn amplitude_scales_the_output_linearly() {
        // breath_noise = 0 makes the internal limit cycle fully deterministic
        // and independent of amplitude, which only scales the final output.
        let quiet = render(
            &mut node(BrassLipReedParams {
                breath_noise: 0.0,
                amplitude: 0.25,
                ..BrassLipReedParams::default()
            }),
            SR as usize,
        );
        let loud = render(
            &mut node(BrassLipReedParams {
                breath_noise: 0.0,
                amplitude: 0.5,
                ..BrassLipReedParams::default()
            }),
            SR as usize,
        );
        let ratio = energy(&loud) / energy(&quiet).max(1e-30);
        // Doubling the amplitude quadruples the energy.
        assert!((ratio - 4.0).abs() < 0.1, "ratio = {ratio}");
    }

    #[test]
    fn lip_tension_selects_the_bore_partial() {
        let f0 = DEFAULT_FREQUENCY_HZ;
        let low = render(
            &mut node(BrassLipReedParams {
                lip_tension: 0.0,
                breath_noise: 0.0,
                ..BrassLipReedParams::default()
            }),
            SR as usize,
        );
        let high = render(
            &mut node(BrassLipReedParams {
                lip_tension: 1.0,
                breath_noise: 0.0,
                ..BrassLipReedParams::default()
            }),
            SR as usize,
        );
        // Slack lips lock the lowest partial; tight lips climb to a higher one.
        assert!(goertzel(&low, f0, SR as Sample) > goertzel(&low, 3.0 * f0, SR as Sample));
        assert!(goertzel(&high, 3.0 * f0, SR as Sample) > goertzel(&high, f0, SR as Sample));
    }

    #[test]
    fn mono_core_is_replicated_to_every_channel() {
        let mut node = node(BrassLipReedParams::default());
        let channels = render_layout(&mut node, 4096, ChannelLayout::Quad);
        assert_eq!(channels.len(), 4);
        for ch in 1..channels.len() {
            assert_eq!(channels[0], channels[ch]);
        }
    }

    #[test]
    fn zero_frames_is_a_no_op() {
        let mut node = node(BrassLipReedParams::default());
        let block = render(&mut node, 0);
        assert!(block.is_empty());
    }

    #[test]
    fn latency_is_zero() {
        let node = node(BrassLipReedParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = BrassLipReedParams {
            frequency_hz: 220.0,
            breath_pressure: 0.6,
            lip_tension: 0.3,
            brightness: 0.7,
            breath_noise: 0.2,
            amplitude: 0.8,
            seed: 0x1234_5678_9ABC_DEF0,
        };
        let node = node(params);
        assert!((node.frequency_hz() - 220.0).abs() < 1e-3);
        assert!((node.breath_pressure() - 0.6).abs() < 1e-6);
        assert!((node.lip_tension() - 0.3).abs() < 1e-6);
        assert!((node.brightness() - 0.7).abs() < 1e-6);
        assert!((node.breath_noise() - 0.2).abs() < 1e-6);
        assert!((node.amplitude() - 0.8).abs() < 1e-6);
        assert_eq!(node.seed(), 0x1234_5678_9ABC_DEF0);
    }

    #[test]
    fn frequency_is_clamped_to_the_audible_bore_range() {
        let low = node(BrassLipReedParams {
            frequency_hz: 1.0,
            ..BrassLipReedParams::default()
        });
        assert!(low.frequency_hz() >= MIN_FREQUENCY_HZ);
        let high = node(BrassLipReedParams {
            frequency_hz: 1_000_000.0,
            ..BrassLipReedParams::default()
        });
        assert!(high.frequency_hz() <= SR as Sample * 0.5);
    }

    #[test]
    fn normalized_controls_are_clamped() {
        let node = node(BrassLipReedParams {
            lip_tension: 5.0,
            brightness: -2.0,
            breath_noise: 9.0,
            breath_pressure: 3.0,
            ..BrassLipReedParams::default()
        });
        assert!((node.lip_tension() - 1.0).abs() < 1e-6);
        assert!((node.brightness() - 0.0).abs() < 1e-6);
        assert!((node.breath_noise() - 1.0).abs() < 1e-6);
        assert!((node.breath_pressure() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn constructor_sanitizes_non_finite_parameters() {
        let node = node(BrassLipReedParams {
            frequency_hz: Sample::NAN,
            lip_tension: Sample::INFINITY,
            brightness: Sample::NAN,
            breath_noise: Sample::NEG_INFINITY,
            breath_pressure: Sample::NAN,
            amplitude: Sample::NAN,
            ..BrassLipReedParams::default()
        });
        assert!(node.frequency_hz().is_finite());
        assert!(node.lip_tension().is_finite());
        assert!(node.brightness().is_finite());
        assert!(node.breath_noise().is_finite());
        assert!(node.breath_pressure().is_finite());
        assert!(node.amplitude().is_finite());
    }

    #[test]
    fn setters_reject_non_finite_and_keep_the_previous_value() {
        let mut node = node(BrassLipReedParams::default());

        let f = node.frequency_hz();
        node.set_frequency_hz(Sample::NAN);
        assert!((node.frequency_hz() - f).abs() < 1e-6);

        let lt = node.lip_tension();
        node.set_lip_tension(Sample::INFINITY);
        assert!((node.lip_tension() - lt).abs() < 1e-6);

        let b = node.brightness();
        node.set_brightness(Sample::NAN);
        assert!((node.brightness() - b).abs() < 1e-6);

        let bn = node.breath_noise();
        node.set_breath_noise(Sample::NEG_INFINITY);
        assert!((node.breath_noise() - bn).abs() < 1e-6);

        let bp = node.breath_pressure();
        node.set_breath_pressure(Sample::NAN, Ramp::Immediate);
        assert!((node.breath_pressure() - bp).abs() < 1e-6);

        let a = node.amplitude();
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert!((node.amplitude() - a).abs() < 1e-6);
    }

    #[test]
    fn retuning_the_frequency_changes_the_output() {
        let low = render(
            &mut node(BrassLipReedParams {
                frequency_hz: 220.0,
                breath_noise: 0.0,
                ..BrassLipReedParams::default()
            }),
            SR as usize,
        );
        let high = render(
            &mut node(BrassLipReedParams {
                frequency_hz: 440.0,
                breath_noise: 0.0,
                ..BrassLipReedParams::default()
            }),
            SR as usize,
        );
        // The lower pitch concentrates more energy at 220 Hz than the higher.
        assert!(goertzel(&low, 220.0, SR as Sample) > goertzel(&high, 220.0, SR as Sample));
    }

    #[test]
    fn brightness_changes_the_output() {
        let dark = render(
            &mut node(BrassLipReedParams {
                brightness: 0.1,
                breath_noise: 0.0,
                ..BrassLipReedParams::default()
            }),
            4096,
        );
        let bright = render(
            &mut node(BrassLipReedParams {
                brightness: 0.9,
                breath_noise: 0.0,
                ..BrassLipReedParams::default()
            }),
            4096,
        );
        assert_ne!(dark, bright);
    }

    #[test]
    fn reseeding_changes_the_turbulence() {
        let mut a = node(BrassLipReedParams {
            breath_noise: 0.5,
            ..BrassLipReedParams::default()
        });
        let mut b = node(BrassLipReedParams {
            breath_noise: 0.5,
            ..BrassLipReedParams::default()
        });
        b.set_seed(0xDEAD_BEEF_1234_5678);
        assert_ne!(render(&mut a, 4096), render(&mut b, 4096));
    }

    #[test]
    fn oscillates_across_the_pitch_range() {
        for &f in &[40.0, 110.0, 440.0, 1760.0] {
            let mut node = node(BrassLipReedParams {
                frequency_hz: f,
                ..BrassLipReedParams::default()
            });
            let block = render(&mut node, SR as usize);
            assert!(block.iter().all(|s| s.is_finite()), "f = {f}");
            assert!(peak(&block[SR as usize / 2..]) > 1e-3, "f = {f}");
        }
    }

    #[test]
    fn breath_pressure_target_tracks_the_setter() {
        let mut node = node(BrassLipReedParams::default());
        node.set_breath_pressure(0.9, Ramp::Immediate);
        assert!((node.breath_pressure() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn amplitude_target_tracks_the_setter() {
        let mut node = node(BrassLipReedParams::default());
        node.set_amplitude(0.75, Ramp::Immediate);
        assert!((node.amplitude() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn breath_noise_adds_breathiness_to_the_attack() {
        let clean = render(
            &mut node(BrassLipReedParams {
                breath_noise: 0.0,
                ..BrassLipReedParams::default()
            }),
            4096,
        );
        let breathy = render(
            &mut node(BrassLipReedParams {
                breath_noise: 0.8,
                ..BrassLipReedParams::default()
            }),
            4096,
        );
        assert_ne!(clean, breathy);
    }

    #[test]
    fn higher_breath_pressure_drives_more_energy() {
        let soft = render(
            &mut node(BrassLipReedParams {
                breath_pressure: 0.3,
                breath_noise: 0.0,
                ..BrassLipReedParams::default()
            }),
            SR as usize,
        );
        let hard = render(
            &mut node(BrassLipReedParams {
                breath_pressure: 0.9,
                breath_noise: 0.0,
                ..BrassLipReedParams::default()
            }),
            SR as usize,
        );
        let tail = SR as usize / 2;
        assert!(energy(&hard[tail..]) > energy(&soft[tail..]));
    }

    #[test]
    fn stereo_layout_fills_both_channels_identically() {
        let mut node = node(BrassLipReedParams::default());
        let channels = render_layout(&mut node, 4096, ChannelLayout::Stereo);
        assert_eq!(channels.len(), 2);
        assert_eq!(channels[0], channels[1]);
    }

    #[test]
    fn seed_getter_reports_the_reseeded_value() {
        let mut node = node(BrassLipReedParams::default());
        node.set_seed(0x0BAD_F00D_DEAD_C0DE);
        assert_eq!(node.seed(), 0x0BAD_F00D_DEAD_C0DE);
    }
}
