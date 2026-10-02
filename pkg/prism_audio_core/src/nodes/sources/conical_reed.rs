//! Conical double-reed digital-waveguide physical-modeling source node.
//!
//! [`ConicalReedNode`] is a *source* (zero inputs, one output) that synthesizes
//! a sustained, self-oscillating conical double-reed woodwind tone (the oboe /
//! bassoon / saxophone family) by modeling a *conical* bore driven by a
//! nonlinear reed valve at its truncated apex. Like
//! [`super::reed_woodwind::ReedWoodwindNode`] it is *continuously driven*:
//! steady breath pressure feeds energy into the bore every sample through the
//! reed nonlinearity, so the tone sustains for as long as the player blows.
//!
//! # Why a cone, not a cylinder
//!
//! The defining acoustic fact of this voice is its *bore shape*. A cylindrical
//! reed bore (the clarinet modeled by [`super::reed_woodwind::ReedWoodwindNode`])
//! is closed at the mouthpiece and open at the bell, so it behaves as a
//! quarter-wave resonator and sounds only the *odd* harmonics an octave below a
//! comparable open tube. A *conical* reed bore -- narrow truncated apex at the
//! reed, flaring to the open bell -- instead resonates the *complete* harmonic
//! series an octave higher, exactly like an open-open tube. This is why an oboe
//! of a given length sounds an octave above a clarinet of the same length and
//! overblows at the octave rather than the twelfth.
//!
//! # The bore (two inverting terminations)
//!
//! The air column is a pair of digital-waveguide delay lines carrying the
//! pressure waves travelling toward the bell and back toward the apex:
//!
//! ```text
//! at_apex = apex_line.read(delay)   (wave arriving at the reed / cone apex)
//! at_bell = bell_line.read(delay)   (wave arriving at the open bell)
//! bell_refl = -g * ((1-S)*at_bell + S*z^-1)   (bell: loss, low-pass, invert)
//! ```
//!
//! The open bell reflects with inversion and a one-zero low-pass loss `(1-S) +
//! S z^-1` scaled by a loop gain `g`, modeling frequency-dependent radiation
//! loss, exactly as the clarinet's bell does. The crucial difference is the
//! *apex*: where the clarinet's near-closed mouthpiece reflects *without*
//! inversion (one inversion per round trip -> odd harmonics), the truncated cone
//! presents an *inverting* conical-cap reflectance -- a first-order *high-pass*
//! (a zero at DC, gain approaching unity at high frequency) that models the
//! `1/r` spherical-wave spreading of a cone. Its corner tracks the pitch at
//! `f0 / APEX_CORNER_RATIO`, so the fundamental always passes while the
//! low-frequency relaxation mode -- which would otherwise capture the net
//! non-inverting loop and drag the pitch far below the bore resonance -- is
//! reflected away. With both ends inverting, the two inversions cancel per round
//! trip, so the loop is net non-inverting and resonates the *complete* harmonic
//! series:
//!
//! ```text
//! apex_src = at_apex + h*(breath - at_apex)      (reed-shaped junction wave)
//! apex_hp  = apex_src - apex_src_z^-1 + rho*apex_hp_z^-1   (conical high-pass)
//! cap      = apex_hp - apex_hp_z^-1 + Rdc*cap_z^-1         (loop DC blocker)
//! lim      = DRIVE_LIMIT * tanh(cap / DRIVE_LIMIT)         (nonlinear loss)
//! to_bell  = -ga * lim                           (cone apex: invert + loss)
//! ```
//!
//! # The reed (pressure-controlled valve)
//!
//! The reed is driven by the pressure drop `delta = breath - at_apex` between
//! the player's steady mouth pressure `breath` and the bore wave returning to
//! the apex. A clipped-linear reed table returns a reflection coefficient that
//! falls as the drop grows (the reed closes and finally beats shut, clamped at
//! `1`):
//!
//! ```text
//! h = clamp(REED_REST - slope*delta, -1, 1)
//! ```
//!
//! `slope` is set from `reed_stiffness`: the hard, narrow double reed of an oboe
//! closes over a tighter pressure range than a clarinet's single reed, which
//! sharpens the beating and gives the bright, nasal, upper-partial-rich oboe
//! timbre.
//!
//! # Nonlinear bore loss (the limit cycle)
//!
//! Unlike the clarinet's net-*inverting* loop -- where the reed table alone
//! bounds the oscillation -- the cone's net *non-inverting* loop, driven by the
//! reed's active region, would grow without bound under a purely linear return
//! path. A real bore caps its own amplitude through nonlinear acoustic losses
//! (turbulent and radiation damping that rise with level); the model reproduces
//! this with one gentle `tanh` soft-saturation in the recirculating path. Small
//! signals pass essentially linearly -- so tuning and the low-order spectrum are
//! unchanged -- while large excursions are compressed, which fixes a bounded
//! limit cycle rather than a hard clip (no aliasing-prone discontinuity). A
//! first-order loop DC blocker additionally removes any slow drift the net
//! non-inverting loop would otherwise integrate out of the breath's DC term.
//!
//! # Pitch and timbre
//!
//! Because the two inversions cancel, the sounding fundamental is
//! `f0 = sample_rate / (2 * delay)` where `delay` is each line's length -- an
//! octave above the clarinet's `sample_rate / (4 * delay)` for the same delay.
//! The lines read at a fractional length by linear interpolation, so the
//! instrument tunes continuously. `brightness` sets the bell low-pass
//! coefficient `S = 0.5 * (1 - brightness)`; `breath` sets loudness and,
//! through the reed, the richness of the beating regime.
//!
//! # Determinism
//!
//! The excitation is the deterministic breath pressure, not noise, so the node
//! holds no random state: two [`ConicalReedNode`]s built with the same sample
//! rate and parameters emit bit-identical streams on every platform via
//! [`bevy_math::ops`], and [`AudioNode::reset`] restarts the identical attack.
//!
//! # Relationship
//!
//! This is the *conical* sibling of the *cylindrical*
//! [`super::reed_woodwind::ReedWoodwindNode`]: both are reed-driven
//! digital-waveguide bores sharing the clipped-linear
//! `McIntyre`-Schumacher-Woodhouse reed table and the one-zero bell loss filter,
//! but the clarinet's single per-round-trip inversion gives it the *odd*-only
//! series an octave low, while this cone's second (apex) inversion restores the
//! *complete* harmonic series an octave high. It differs from the jet-driven
//! open-open cylinder [`super::air_jet_flute::AirJetFluteNode`] (which also
//! sounds the full series but is pumped by a cubic air-jet edge tone, not a
//! reed) and from the lip-valve brass [`super::brass_lip_reed::BrassLipReedNode`]
//! (whose tunable mechanical lip resonance, not a fixed bore length, selects the
//! sounding partial). It reuses only this crate's own [`Sample`] type,
//! [`bevy_math::ops`], and the shared smoothing primitives.
//!
//! # Real-time contract
//!
//! Both delay lines are pre-allocated in [`ConicalReedNode::new`] for the lowest
//! supported pitch, so [`process`](crate::graph::AudioNode::process) performs no
//! allocation, takes no locks, and cannot panic: every recirculated sample is
//! denormal-flushed, the two termination gains multiply to below unity, the reed
//! coefficient is clamped to `[-1, 1]`, and non-finite parameters are rejected
//! at the setters. The breath pressure and output amplitude are driven through
//! [`Smoothed`] values so performance gestures never zipper.
//!
//! # Provenance
//!
//! The conical digital waveguide follows the standard public treatment of
//! conical acoustic tubes in J. O. Smith's *Physical Audio Signal Processing*
//! (public online text) -- a cylindrical waveguide pair terminated at the
//! truncated apex by a first-order high-pass reflectance (a zero at DC) that
//! reproduces the complete-harmonic, octave-up resonances of a cone. The reed
//! valve is the clipped-linear reed table of `McIntyre`, Schumacher and
//! Woodhouse ("On the oscillations of musical instruments", *Journal of the
//! Acoustical Society of America*, 1983); the one-zero bell loss and
//! `brightness` mapping are shared with the Jaffe-Smith Karplus-Strong
//! extensions; and the `tanh` soft-saturation that bounds the limit cycle and
//! the first-order DC blocker are textbook public DSP building blocks. This
//! module contains
//! **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance
//! Audio, or Web Audio source or derived code**, and nothing from any audio
//! toolkit's implementation (including STK); it is written purely from that
//! publicly documented theory.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Lowest tunable fundamental in hertz. Bounds the pre-allocated delay lines.
pub const MIN_FREQUENCY_HZ: Sample = 40.0;

/// Default fundamental (pitch) frequency in hertz (roughly an oboe A4).
pub const DEFAULT_FREQUENCY_HZ: Sample = 440.0;

/// Default normalized breath pressure in `[0, 1]`.
pub const DEFAULT_BREATH_PRESSURE: Sample = 0.5;

/// Default normalized reed stiffness in `[0, 1]`.
pub const DEFAULT_REED_STIFFNESS: Sample = 0.5;

/// Default brightness in `[0, 1]`.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Internal scale mapping normalized breath pressure `[0, 1]` to bore units.
const MAX_BREATH_PRESSURE: Sample = 0.55;

/// Reed-table value at zero pressure drop (reed slightly open at rest).
const REED_REST: Sample = 0.7;

/// Reed-table slope at the softest reed.
const REED_SLOPE_MIN: Sample = 2.0;

/// Reed-table slope at the stiffest reed (a hard, narrow double reed closes
/// over a tighter pressure range than a single reed, so the range runs higher).
const REED_SLOPE_MAX: Sample = 8.0;

/// Open-bell reflection loop gain, just below unity so the passive loop is
/// stable while the breath supplies the sustaining energy.
const BELL_REFLECTION_GAIN: Sample = 0.97;

/// Conical-cap (apex) reflection loop gain. The product with
/// [`BELL_REFLECTION_GAIN`] stays below unity so the two-inversion loop is
/// passive-stable.
const APEX_REFLECTION_GAIN: Sample = 0.98;

/// Conical-cap reflectance corner ratio. The truncated apex reflects the
/// returning wave through a first-order high-pass whose zero sits at DC (no
/// steady reflection) and whose gain approaches unity at high frequency -- the
/// digital-waveguide image of the spherical-wave spreading at a cone's
/// truncated apex. Its corner tracks the pitch at `f0 / APEX_CORNER_RATIO` so
/// the fundamental always passes while the sub-fundamental relaxation mode is
/// suppressed, locking the two-inversion (half-wave) loop onto the cone's
/// complete harmonic series across the playable range. The reflectance pole is
/// `1 - 2 pi (f0 / APEX_CORNER_RATIO) / sr`.
const APEX_CORNER_RATIO: Sample = 2.87;

/// Loop DC-blocker pole. A real truncated cone cannot sustain a steady pressure
/// (the net non-inverting loop would otherwise integrate the breath's DC into a
/// runaway), so the apex reflectance has a zero at DC. The cutoff sits well
/// below [`MIN_FREQUENCY_HZ`] so no audible harmonic is affected.
const DC_BLOCK_POLE: Sample = 0.9995;

/// Loop phase compensation in samples. The conical high-pass and the bell loss
/// filter advance the loop phase slightly; this empirically calibrated term
/// (measured against a swept pitch reference) keeps the sounding pitch locked
/// to `frequency_hz` across the playable range.
const APEX_PHASE_COMP: Sample = 0.6;

/// Soft-saturation limit for the recirculating wave. The net non-inverting
/// (full-harmonic) loop needs the reed's active region to self-oscillate, which
/// would otherwise grow without bound; a gentle `tanh` nonlinear loss leaves
/// small signals essentially linear (so tuning and the low-order spectrum are
/// unaffected) and bounds the limit cycle, the acoustic nonlinear damping that
/// caps a real bore's amplitude.
const DRIVE_LIMIT: Sample = 0.8;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Clamps `frequency_hz` to `[MIN_FREQUENCY_HZ, sample_rate / 2]`, falling back
/// to [`MIN_FREQUENCY_HZ`] for non-finite input.
fn sanitize_frequency(frequency_hz: Sample, sample_rate: Sample) -> Sample {
    let nyquist = (sample_rate * 0.5).max(MIN_FREQUENCY_HZ);
    finite_or(frequency_hz, MIN_FREQUENCY_HZ).clamp(MIN_FREQUENCY_HZ, nyquist)
}

/// The nonlinear reed reflection characteristic: a clipped-linear coefficient in
/// `[-1, 1]` that falls as the pressure drop closes the reed.
#[inline]
fn reed_reflection(pressure_drop: Sample, slope: Sample) -> Sample {
    (REED_REST - slope * pressure_drop).clamp(-1.0, 1.0)
}

/// A linear-interpolating waveguide delay line (a travelling bore segment).
#[derive(Debug, Clone)]
struct WaveguideDelay {
    buf: Vec<Sample>,
    write: usize,
}

impl WaveguideDelay {
    fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity.max(2)],
            write: 0,
        }
    }

    /// Reads the wave delayed by `delay` samples with linear interpolation.
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

    /// Writes the next sample and advances the write head.
    #[inline]
    fn write_sample(&mut self, x: Sample) {
        self.buf[self.write] = flush_denormal(x);
        self.write += 1;
        if self.write == self.buf.len() {
            self.write = 0;
        }
    }

    /// Zeroes the line and resets the write head.
    fn clear(&mut self) {
        for v in &mut self.buf {
            *v = 0.0;
        }
        self.write = 0;
    }
}

/// Construction parameters for a [`ConicalReedNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConicalReedParams {
    /// Fundamental (pitch) frequency in hertz.
    pub frequency_hz: Sample,
    /// Normalized breath pressure in `[0, 1]` (loudness / drive).
    pub breath_pressure: Sample,
    /// Normalized reed stiffness in `[0, 1]` (timbre / beating sharpness).
    pub reed_stiffness: Sample,
    /// Brightness in `[0, 1]` (bell filter high-frequency loss).
    pub brightness: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for ConicalReedParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            breath_pressure: DEFAULT_BREATH_PRESSURE,
            reed_stiffness: DEFAULT_REED_STIFFNESS,
            brightness: DEFAULT_BRIGHTNESS,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

/// A conical double-reed digital-waveguide voice source node (0 inputs, 1 out).
///
/// The mono bore signal is replicated into every output channel.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{ConicalReedNode, ConicalReedParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = ConicalReedNode::new(48_000, ConicalReedParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The breath drives the conical bore into sustained self-oscillation.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct ConicalReedNode {
    /// Sample rate the delay lines were sized for.
    sample_rate: Sample,
    /// Fundamental frequency in hertz.
    frequency_hz: Sample,
    /// Reed stiffness in `[0, 1]`.
    reed_stiffness: Sample,
    /// Brightness in `[0, 1]`.
    brightness: Sample,
    /// Smoothed normalized breath pressure in `[0, 1]`.
    breath_pressure: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,

    /// Apex-bound (bell to apex) waveguide segment.
    apex_line: WaveguideDelay,
    /// Bell-bound (apex to bell) waveguide segment.
    bell_line: WaveguideDelay,
    /// Previous bell-filter input (`z^-1`) for the one-zero loss filter.
    bell_filter_z1: Sample,
    /// Conical-cap high-pass output state (`z^-1`) at the apex.
    apex_filter_z1: Sample,
    /// Conical-cap high-pass input state (`z^-1`).
    apex_filter_in_z1: Sample,
    /// Loop DC-blocker input memory (`z^-1`).
    dc_block_in_z1: Sample,
    /// Loop DC-blocker output memory (`z^-1`).
    dc_block_out_z1: Sample,

    /// Latched per-line fractional delay in samples.
    delay: Sample,
    /// Latched one-zero bell-filter coefficient `S`.
    damping_s: Sample,
    /// Latched reed-table slope.
    slope: Sample,
    /// Latched conical-cap high-pass pole (pitch-tracking).
    apex_pole: Sample,
}

impl ConicalReedNode {
    /// Builds a conical reed for `sample_rate` Hz from a parameter bundle.
    ///
    /// The delay lines are sized so a fundamental as low as [`MIN_FREQUENCY_HZ`]
    /// fits. All parameters are sanitized: non-finite values fall back to safe
    /// defaults, `frequency_hz` is clamped to `[MIN_FREQUENCY_HZ, sample_rate /
    /// 2]`, and `breath_pressure`/`reed_stiffness`/`brightness` to `[0, 1]`.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "the buffer length is tiny relative to f32's integer precision"
    )]
    pub fn new(sample_rate: u32, params: ConicalReedParams) -> Self {
        let sr = (sample_rate.max(1)) as Sample;
        // Longest per-line delay (frames at the lowest pitch): delay = sr /
        // (2 f0), with headroom for the filters and interpolation taps.
        let max_delay_frames = ops::round(sr / (2.0 * MIN_FREQUENCY_HZ)) as usize;
        let capacity = max_delay_frames + 4;

        let frequency_hz = sanitize_frequency(params.frequency_hz, sr);
        let reed_stiffness =
            finite_or(params.reed_stiffness, DEFAULT_REED_STIFFNESS).clamp(0.0, 1.0);
        let brightness = finite_or(params.brightness, DEFAULT_BRIGHTNESS).clamp(0.0, 1.0);
        let breath_pressure =
            finite_or(params.breath_pressure, DEFAULT_BREATH_PRESSURE).clamp(0.0, 1.0);
        let amplitude = finite_or(params.amplitude, DEFAULT_AMPLITUDE);

        let mut node = Self {
            sample_rate: sr,
            frequency_hz,
            reed_stiffness,
            brightness,
            breath_pressure: Smoothed::new(breath_pressure),
            amplitude: Smoothed::new(amplitude),
            apex_line: WaveguideDelay::new(capacity),
            bell_line: WaveguideDelay::new(capacity),
            bell_filter_z1: 0.0,
            apex_filter_z1: 0.0,
            apex_filter_in_z1: 0.0,
            dc_block_in_z1: 0.0,
            dc_block_out_z1: 0.0,
            delay: 1.0,
            damping_s: 0.25,
            slope: REED_SLOPE_MIN,
            apex_pole: 0.98,
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

    /// Sets the reed stiffness in `[0, 1]`.
    #[inline]
    pub fn set_reed_stiffness(&mut self, reed_stiffness: Sample) {
        self.reed_stiffness = finite_or(reed_stiffness, self.reed_stiffness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the brightness in `[0, 1]`.
    #[inline]
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
        self.recompute();
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

    /// Returns the fundamental frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the reed stiffness in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn reed_stiffness(&self) -> Sample {
        self.reed_stiffness
    }

    /// Returns the brightness in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
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

    /// Recomputes the latched loop coefficients from the user-facing parameters.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the buffer length is tiny relative to f32's integer precision"
    )]
    fn recompute(&mut self) {
        let sr = self.sample_rate;
        // One-zero bell-filter coefficient S in [0, 0.5]: brightness 1 -> S 0.
        let s = 0.5 * (1.0 - self.brightness);
        self.damping_s = s;
        self.slope = REED_SLOPE_MIN + self.reed_stiffness * (REED_SLOPE_MAX - REED_SLOPE_MIN);

        // Full-harmonic (half-wave) loop: the bell inversion and the conical
        // apex inversion cancel, so f0 = sr / (2 D). The bell one-zero filter
        // adds S/2 samples of phase delay per one-way segment and the conical
        // high-pass contributes a small, nearly frequency-independent phase
        // lead; both are folded into the tuning so A440 lands on pitch.
        let bell_phase = 0.5 * s;
        let max_delay = (self.apex_line.buf.len() - 2) as Sample;
        let d =
            (sr / (2.0 * self.frequency_hz) - bell_phase + APEX_PHASE_COMP).clamp(1.0, max_delay);
        self.delay = d;
        // Conical reflectance corner tracks the pitch so the fundamental passes
        // while the sub-fundamental relaxation mode stays suppressed.
        let corner = self.frequency_hz / APEX_CORNER_RATIO;
        self.apex_pole = (1.0 - core::f32::consts::TAU * corner / sr).clamp(0.0, 0.9995);
    }

    /// Applies the one-zero bell loss filter and advances its memory.
    #[inline]
    fn bell_filter(&mut self, x: Sample) -> Sample {
        let s = self.damping_s;
        let y = BELL_REFLECTION_GAIN * ((1.0 - s) * x + s * self.bell_filter_z1);
        self.bell_filter_z1 = x;
        y
    }

    /// Applies the conical-cap leaky low-pass and advances its memory.
    #[inline]
    fn apex_filter(&mut self, x: Sample) -> Sample {
        // Conical-cap reflectance: first-order high-pass (zero at DC). The
        // truncated-apex spherical spreading reflects nothing at DC and ~-1 at
        // high frequency, which suppresses the sub-fundamental relaxation and
        // locks the half-wave cone onto its full-harmonic series.
        let y = x - self.apex_filter_in_z1 + self.apex_pole * self.apex_filter_z1;
        self.apex_filter_in_z1 = x;
        self.apex_filter_z1 = y;
        y
    }

    /// Applies the loop DC blocker (first-order high-pass) and advances memory.
    #[inline]
    fn dc_block(&mut self, x: Sample) -> Sample {
        let y = x - self.dc_block_in_z1 + DC_BLOCK_POLE * self.dc_block_out_z1;
        self.dc_block_in_z1 = x;
        self.dc_block_out_z1 = y;
        y
    }

    /// Advances the waveguide by one sample and returns the radiated output.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let breath = self.breath_pressure.next_sample() * MAX_BREATH_PRESSURE;

        let at_apex = self.apex_line.read(self.delay);
        let at_bell = self.bell_line.read(self.delay);

        // Open bell: lossy low-pass reflection with inversion.
        let bell_refl = -self.bell_filter(at_bell);

        // Reed (truncated apex): nonlinear pressure-controlled valve shapes the
        // junction wave, then the conical cap reflects it with inversion and a
        // leaky low-pass -- the second inversion that makes the cone sound the
        // complete harmonic series.
        let delta = breath - at_apex;
        let h = reed_reflection(delta, self.slope);
        let apex_src = at_apex + h * delta;
        // Conical cap: a passive (|H|<=1), DC-blocked leaky low-pass shapes the
        // reed-launched wave. Its low-frequency phase lag relocates the bore's
        // resonances onto the complete harmonic series an octave above an
        // equal-length cylinder, without adding loop gain, so stability matches
        // the clarinet while the timbre is the cone's full-harmonic one.
        let lp = self.apex_filter(apex_src);
        let cap = self.dc_block(lp);
        // Soft nonlinear loss bounds the full-harmonic limit cycle.
        let limited = DRIVE_LIMIT * ops::tanh(cap / DRIVE_LIMIT);
        let to_bell = -APEX_REFLECTION_GAIN * limited;

        self.bell_line.write_sample(to_bell);
        self.apex_line.write_sample(bell_refl);

        flush_denormal(at_bell * self.amplitude.next_sample())
    }
}

impl AudioNode for ConicalReedNode {
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
        self.apex_line.clear();
        self.bell_line.clear();
        self.bell_filter_z1 = 0.0;
        self.apex_filter_z1 = 0.0;
        self.apex_filter_in_z1 = 0.0;
        self.dc_block_in_z1 = 0.0;
        self.dc_block_out_z1 = 0.0;
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
    fn render(node: &mut ConicalReedNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout` and returns per-channel
    /// sample vectors.
    fn render_layout(
        node: &mut ConicalReedNode,
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
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        let block = render(&mut node, SR as usize);
        // Steady breath drives sustained self-oscillation, not a decaying pluck.
        assert!(peak(&block) > 0.0);
        assert!(peak(&block[SR as usize / 2..]) > 1e-2);
        assert!(block.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn long_run_stays_bounded() {
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        let block = render(&mut node, 4 * SR as usize);
        assert!(block.iter().all(|s| s.is_finite()));
        // The net non-inverting loop is held to a bounded limit cycle by the
        // tanh nonlinear loss plus the sub-unity termination gains; it must
        // never run away.
        assert!(peak(&block) < 1.0);
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = ConicalReedNode::new(SR, ConicalReedParams::default());
        let mut b = ConicalReedNode::new(SR, ConicalReedParams::default());
        assert_eq!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn reset_restarts_identical_attack() {
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        let first = render(&mut node, 8192);
        node.reset();
        let second = render(&mut node, 8192);
        assert_eq!(first, second);
    }

    #[test]
    fn zero_breath_pressure_is_silent() {
        let params = ConicalReedParams {
            breath_pressure: 0.0,
            ..ConicalReedParams::default()
        };
        let mut node = ConicalReedNode::new(SR, params);
        // With no breath and no stored energy the reed never opens: delta = 0,
        // apex_src = 0, every filter stays at rest, so the bore is silent.
        let block = render(&mut node, SR as usize / 2);
        assert!(block.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn amplitude_scales_output_energy() {
        let loud = ConicalReedParams {
            amplitude: 1.0,
            ..ConicalReedParams::default()
        };
        let soft = ConicalReedParams {
            amplitude: 0.5,
            ..ConicalReedParams::default()
        };
        let mut a = ConicalReedNode::new(SR, loud);
        let mut b = ConicalReedNode::new(SR, soft);
        let ea = energy(&render(&mut a, 8192));
        let eb = energy(&render(&mut b, 8192));
        // Amplitude is a pure output scale (never fed back into the loop), so
        // halving it quarters the radiated energy exactly.
        assert!(eb > 0.0);
        assert!((ea / eb - 4.0).abs() < 1e-3);
    }

    #[test]
    fn full_harmonic_series_present() {
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        let block = render(&mut node, 2 * SR as usize);
        // Analyze the settled second half so the attack transient is excluded.
        let tail = &block[SR as usize..];
        let f0 = DEFAULT_FREQUENCY_HZ;
        let h1 = goertzel(tail, f0, SR as Sample);
        let h2 = goertzel(tail, 2.0 * f0, SR as Sample);
        // The two-inversion (half-wave) cone sounds the complete harmonic
        // series, so the even second harmonic carries substantial energy --
        // unlike the clarinet, whose quarter-wave bore suppresses it. This is
        // the defining spectral signature versus `ReedWoodwindNode`.
        assert!(h1 > 0.0);
        assert!(h2 > 0.1 * h1, "even harmonic too weak: h1={h1}, h2={h2}");
    }

    #[test]
    fn pitch_tracks_frequency_within_two_percent() {
        for &target in &[110.0_f32, 220.0, 440.0] {
            let params = ConicalReedParams {
                frequency_hz: target,
                ..ConicalReedParams::default()
            };
            let mut node = ConicalReedNode::new(SR, params);
            let block = render(&mut node, 2 * SR as usize);
            let tail = &block[SR as usize..];
            // Scan a wide window and confirm the spectral peak lands on the
            // requested pitch (the conical reflectance corner tracks f0, which
            // keeps the sounding mode on the fundamental across the range).
            let mut best = (0.0_f64, 0.0_f32);
            let mut f = (target * 0.5).max(MIN_FREQUENCY_HZ);
            let hi = target * 1.6;
            while f < hi {
                let m = goertzel(tail, f, SR as Sample);
                if m > best.0 {
                    best = (m, f);
                }
                f += 0.5;
            }
            let err = (best.1 - target).abs() / target;
            assert!(err < 0.02, "pitch off at {target} Hz: measured {}", best.1);
        }
    }

    #[test]
    fn mono_core_replicated_to_all_channels() {
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        let chans = render_layout(&mut node, 4096, ChannelLayout::Stereo);
        assert_eq!(chans.len(), 2);
        // The mono bore core is copied verbatim into every output channel.
        assert_eq!(chans[0], chans[1]);
        assert!(peak(&chans[0]) > 0.0);
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        let block = render(&mut node, 0);
        assert!(block.is_empty());
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = ConicalReedParams {
            frequency_hz: 330.0,
            breath_pressure: 0.6,
            reed_stiffness: 0.4,
            brightness: 0.7,
            amplitude: 0.25,
        };
        let node = ConicalReedNode::new(SR, params);
        assert!((node.frequency_hz() - 330.0).abs() < 1e-4);
        assert!((node.breath_pressure() - 0.6).abs() < 1e-6);
        assert!((node.reed_stiffness() - 0.4).abs() < 1e-6);
        assert!((node.brightness() - 0.7).abs() < 1e-6);
        assert!((node.amplitude() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn set_frequency_clamps_to_range() {
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        node.set_frequency_hz(10.0);
        assert!((node.frequency_hz() - MIN_FREQUENCY_HZ).abs() < 1e-4);
        node.set_frequency_hz(1.0e9);
        assert!((node.frequency_hz() - SR as Sample / 2.0).abs() < 1.0);
    }

    #[test]
    fn set_reed_stiffness_and_brightness_clamp() {
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        node.set_reed_stiffness(5.0);
        assert!((node.reed_stiffness() - 1.0).abs() < 1e-6);
        node.set_reed_stiffness(-2.0);
        assert!((node.reed_stiffness() - 0.0).abs() < 1e-6);
        node.set_brightness(9.0);
        assert!((node.brightness() - 1.0).abs() < 1e-6);
        node.set_brightness(-9.0);
        assert!((node.brightness() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn set_breath_pressure_clamps_to_range() {
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        node.set_breath_pressure(5.0, Ramp::Immediate);
        assert!((node.breath_pressure() - 1.0).abs() < 1e-6);
        node.set_breath_pressure(-5.0, Ramp::Immediate);
        assert!((node.breath_pressure() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn setters_reject_non_finite() {
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        let f = node.frequency_hz();
        let st = node.reed_stiffness();
        let b = node.brightness();
        let bp = node.breath_pressure();
        let a = node.amplitude();
        node.set_frequency_hz(Sample::NAN);
        node.set_reed_stiffness(Sample::INFINITY);
        node.set_brightness(Sample::NEG_INFINITY);
        node.set_breath_pressure(Sample::NAN, Ramp::Immediate);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.frequency_hz(), f);
        assert_eq!(node.reed_stiffness(), st);
        assert_eq!(node.brightness(), b);
        assert_eq!(node.breath_pressure(), bp);
        assert_eq!(node.amplitude(), a);
    }

    #[test]
    fn constructor_sanitizes_non_finite_params() {
        let params = ConicalReedParams {
            frequency_hz: Sample::NAN,
            breath_pressure: Sample::INFINITY,
            reed_stiffness: Sample::NAN,
            brightness: Sample::NEG_INFINITY,
            amplitude: Sample::NAN,
        };
        let node = ConicalReedNode::new(SR, params);
        assert!(node.frequency_hz().is_finite());
        assert!(node.breath_pressure().is_finite());
        assert!(node.reed_stiffness().is_finite());
        assert!(node.brightness().is_finite());
        assert!(node.amplitude().is_finite());
    }

    #[test]
    fn frequency_changes_output() {
        let low = ConicalReedParams {
            frequency_hz: 110.0,
            ..ConicalReedParams::default()
        };
        let high = ConicalReedParams {
            frequency_hz: 440.0,
            ..ConicalReedParams::default()
        };
        let mut a = ConicalReedNode::new(SR, low);
        let mut b = ConicalReedNode::new(SR, high);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn reed_stiffness_changes_timbre() {
        let soft = ConicalReedParams {
            reed_stiffness: 0.1,
            ..ConicalReedParams::default()
        };
        let stiff = ConicalReedParams {
            reed_stiffness: 0.9,
            ..ConicalReedParams::default()
        };
        let mut a = ConicalReedNode::new(SR, soft);
        let mut b = ConicalReedNode::new(SR, stiff);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn brightness_changes_output() {
        let dark = ConicalReedParams {
            brightness: 0.05,
            ..ConicalReedParams::default()
        };
        let bright = ConicalReedParams {
            brightness: 0.95,
            ..ConicalReedParams::default()
        };
        let mut a = ConicalReedNode::new(SR, dark);
        let mut b = ConicalReedNode::new(SR, bright);
        // The bell loss coefficient S tracks brightness, so the radiated
        // waveform must differ.
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn high_and_low_frequencies_both_oscillate() {
        for &f in &[MIN_FREQUENCY_HZ, 1_500.0] {
            let params = ConicalReedParams {
                frequency_hz: f,
                ..ConicalReedParams::default()
            };
            let mut node = ConicalReedNode::new(SR, params);
            let block = render(&mut node, SR as usize);
            assert!(peak(&block) > 0.0, "silent at {f} Hz");
            assert!(block.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn frequency_sweep_stays_finite_and_bounded() {
        for &f in &[40.0_f32, 60.0, 1_000.0, 5_000.0, 20_000.0] {
            let params = ConicalReedParams {
                frequency_hz: f,
                ..ConicalReedParams::default()
            };
            let mut node = ConicalReedNode::new(SR, params);
            let block = render(&mut node, SR as usize);
            assert!(block.iter().all(|s| s.is_finite()), "non-finite at {f} Hz");
            assert!(peak(&block) < 1.0, "runaway at {f} Hz");
        }
    }

    #[test]
    fn breath_pressure_target_tracks_setter() {
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        node.set_breath_pressure(0.8, Ramp::linear_seconds(0.01, SR));
        assert!((node.breath_pressure() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn amplitude_target_tracks_setter() {
        let mut node = ConicalReedNode::new(SR, ConicalReedParams::default());
        node.set_amplitude(0.25, Ramp::linear_seconds(0.01, SR));
        assert!((node.amplitude() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn latency_is_zero() {
        let node = ConicalReedNode::new(SR, ConicalReedParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
