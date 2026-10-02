//! Tine electric-piano source (Rhodes / Wurlitzer "electric piano" family).
//!
//! [`TineElectricPianoNode`] is a *source* (zero inputs, one output) that
//! synthesizes a struck metal *tine* sensed by an *electromagnetic pickup*. A
//! felt hammer strikes a stiff steel tine clamped to a resonating tonebar; the
//! free end of the tine vibrates past a magnetic pickup whose nonlinear coupling
//! turns that motion into the characteristic bell-like attack and velocity-
//! dependent "bark" of the electric piano. On its own it is a complete struck
//! voice that rings out and decays after each strike.
//!
//! # Model
//!
//! The tine is a stiff *clamped-free* (cantilever) bar: one end welded to the
//! tonebar, the other free to vibrate. Its transverse bending modes sit at the
//! inharmonic ratios of the clamped-free Euler-Bernoulli beam,
//! `1 : 6.267 : 17.55 : ...` (the squared ratios of the beam eigenvalues
//! `beta_n = 1.875, 4.694, 7.855, ...`). This is a *different* boundary
//! condition from the free-free bar of [`super::struck_bar`] (whose ratios are
//! `1 : 2.756 : 5.404 : ...`), and it is why a tine's bright partials sit much
//! higher than a marimba's. The tine is modelled as a small bank of
//! [`NUM_TINE_MODES`] decaying two-pole resonators:
//!
//! * mode 0 -- the tine fundamental (long decay, the sustained body tone);
//! * mode 1 -- a slightly detuned partner at `1 + BEAT_DETUNE` times the
//!   fundamental, representing the coupling between the tine and its tonebar.
//!   Beating against mode 0 it produces the slow amplitude shimmer heard on a
//!   sustained electric-piano note;
//! * modes 2.. -- the bright inharmonic cantilever partials that form the
//!   metallic attack "bell"; their level is set by `tine_level` and they decay
//!   faster than the fundamental, so the bell fades to leave the body tone.
//!
//! Each mode `m` is a two-pole resonator
//! `y[n] = b0 * x[n] + a1 * y[n-1] + a2 * y[n-2]` whose complex pole pair sits
//! at radius `R = exp(-ln(1000) / (t60 * sample_rate))` and angle
//! `theta = 2*pi*f_m / sample_rate`, giving `a1 = 2*R*cos(theta)`,
//! `a2 = -R*R`, and feed gain `b0 = gain_m * sin(theta)` (which normalises the
//! ringing peak to `gain_m` independent of decay). Higher partials get shorter
//! decay (`t60_m = decay / ratio_m^DECAY_RATIO_EXP`). The excitation is one
//! unit-area raised-cosine (Hann) hammer-contact pulse whose width is set by
//! `hardness`: a hard hammer is a short pulse (bright, drives the high modes), a
//! soft hammer a long pulse (mellow).
//!
//! The summed mode output is the tine *displacement* `x[n]` at the pickup. The
//! pickup does not read it linearly: the magnetic flux linked by the coil is a
//! nonlinear function of the tine's position, so the induced voltage follows a
//! static saturating, asymmetric transfer
//!
//! ```text
//!   V(x) = (x + asymmetry * x^2) / (1 + (x / saturation)^2)
//! ```
//!
//! * the `asymmetry * x^2` term injects *even* harmonics -- the pickup sits to
//!   one side of the tine, so the coupling is not symmetric; this is the growl;
//! * the `1 / (1 + (x/saturation)^2)` factor is the geometric saturation of the
//!   flux as the tine swings far from the magnet; it compresses loud strikes and
//!   adds odd harmonics (the "bark").
//!
//! Because `strike` velocity scales the displacement, a harder strike swings the
//! tine further into the nonlinearity, so both the growl and the bark grow with
//! velocity -- the signature dynamic timbre of the instrument. The transfer is
//! bounded for every `x` (as `|x| -> inf`, `V -> asymmetry * saturation^2`), so
//! the node can never diverge. The small transient `x^2` even-harmonic term
//! carries a slight DC offset while a note rings; route through a
//! [`super::super::effects::DcBlockerNode`] if a strictly DC-free output is
//! required.
//!
//! # Determinism
//!
//! The excitation is a closed-form deterministic pulse (not noise) and the
//! pickup is a closed-form rational map, so two nodes built with the same sample
//! rate and parameters produce bit-identical output, and [`TineElectricPianoNode::reset`]
//! clears the resonators and re-strikes to replay the identical attack.
//!
//! # Real-time contract
//!
//! All per-mode coefficient and history storage is a fixed-size array sized for
//! [`NUM_TINE_MODES`]; `process` performs no allocation, locking, or panic on
//! the hot path. Non-finite parameters are sanitised on the way in and outputs
//! are flushed of denormals. Latency is zero.
//!
//! # Relationship
//!
//! Reuses this crate's [`Sample`], [`Smoothed`], denormal flush, and the same
//! two-pole-resonator / `t60`-to-radius / unit-area Hann-pulse primitives that
//! [`super::struck_bar`] uses, but is a distinct instrument: it models the
//! *clamped-free* tine (not the free-free bar), adds the detuned tonebar
//! partner that beats against the fundamental, and -- the defining difference --
//! runs the tine displacement through the nonlinear electromagnetic *pickup*
//! transfer, giving the even-harmonic growl and velocity-dependent bark that no
//! purely modal percussion source produces. It is unlike the
//! [`super::super::effects::modal_resonator`] *effect*, which filters an
//! external input, because it supplies its own hammer excitation; unlike the
//! near-harmonic waveguide voices [`super::karplus_strong`] and
//! [`super::bowed_string`]; and unlike the steady, strike-free
//! [`super::additive_oscillator`].
//!
//! # Provenance
//!
//! Classic public-domain DSP only; no third-party engine, library, or toolkit
//! source or derivative was consulted or copied. Modal synthesis (an object as a
//! parallel bank of decaying resonators) is the classic technique of J.-M.
//! Adrien, "The Missing Link: Modal Synthesis" (MIT Press, 1991). The
//! clamped-free (cantilever) bending-mode ratios are the standard
//! Euler-Bernoulli beam partials tabulated in acoustics texts (e.g. N. H.
//! Fletcher and T. D. Rossing, *The Physics of Musical Instruments*). The
//! two-pole resonator, the `t60`-to-radius mapping, and the Hann contact pulse
//! are standard published DSP. The electromagnetic pickup nonlinearity follows
//! from first-principles magnetics (Faraday's law; flux linkage that varies
//! nonlinearly with the magnet-to-tine gap), modelled here as a bounded rational
//! transfer -- a long-published public technique. No code from Unreal Engine,
//! Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, Web Audio, or
//! STK was referenced.

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Number of tine bending modes synthesized (fundamental, tonebar partner, and
/// two bright cantilever partials).
pub const NUM_TINE_MODES: usize = 4;

/// Lowest tunable fundamental (strike pitch) in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;
/// Highest tunable fundamental in hertz (further bounded by the Nyquist guard).
pub const MAX_FREQUENCY_HZ: Sample = 4_000.0;
/// Default fundamental (strike pitch) in hertz (A3).
pub const DEFAULT_FREQUENCY_HZ: Sample = 220.0;

/// Shortest `-60 dB` fundamental decay time in seconds.
pub const MIN_DECAY_S: Sample = 0.1;
/// Longest `-60 dB` fundamental decay time in seconds.
pub const MAX_DECAY_S: Sample = 20.0;
/// Default fundamental `-60 dB` decay time in seconds.
pub const DEFAULT_DECAY_S: Sample = 4.0;

/// Default hammer hardness / brightness in `[0, 1]`.
pub const DEFAULT_HARDNESS: Sample = 0.5;

/// Default bright-partial (tine bell) level in `[0, 1]`.
pub const DEFAULT_TINE_LEVEL: Sample = 0.6;

/// Smallest pickup asymmetry (symmetric pickup, no growl).
pub const MIN_PICKUP_ASYMMETRY: Sample = 0.0;
/// Largest pickup asymmetry (strongest even-harmonic growl).
pub const MAX_PICKUP_ASYMMETRY: Sample = 1.0;
/// Default pickup asymmetry.
pub const DEFAULT_PICKUP_ASYMMETRY: Sample = 0.3;

/// Smallest pickup saturation knee (earliest bark onset, most compression).
pub const MIN_PICKUP_SATURATION: Sample = 0.1;
/// Largest pickup saturation knee (nearly linear pickup).
pub const MAX_PICKUP_SATURATION: Sample = 4.0;
/// Default pickup saturation knee.
pub const DEFAULT_PICKUP_SATURATION: Sample = 0.8;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Default strike velocity used at construction and by [`TineElectricPianoNode::strike`].
pub const DEFAULT_STRIKE_VELOCITY: Sample = 1.0;

/// Fraction of Nyquist above which a mode is muted (anti-aliasing guard).
pub const NYQUIST_GUARD: Sample = 0.49;

/// Overall output scale keeping the pickup output below full scale.
///
/// Calibrated so the deterministic parameter-grid test (every shape across the
/// pitch range at strike velocity 1 and `amplitude == 1`) peaks near `0.88`.
pub const OUTPUT_GAIN: Sample = 0.137;

/// Scales the modal displacement into the pickup's nonlinear range so the growl
/// and bark engage at musical strike velocities.
const EXCITATION_GAIN: Sample = 0.9;

/// Fractional detuning of the tonebar partner above the fundamental (the slow
/// sustained beat).
const BEAT_DETUNE: Sample = 0.004;

/// Clamped-free (cantilever) bending-mode ratios, `(beta_n / beta_0)^2` with
/// `beta = 1.875, 4.694, 7.855, ...`. Index 1 is overwritten by the detuned
/// tonebar partner in [`TineElectricPianoNode::recompute`].
const TINE_MODE_RATIOS: [Sample; NUM_TINE_MODES] = [1.0, 1.0, 6.267, 17.55];

/// Base linear gain of each tine mode before `tine_level` scaling.
const TINE_MODE_GAINS: [Sample; NUM_TINE_MODES] = [1.0, 0.7, 1.0, 0.45];

/// Decay-vs-ratio exponent: higher partials ring shorter.
const DECAY_RATIO_EXP: Sample = 0.7;

/// Shortest hammer-contact pulse (hard hammer), in milliseconds.
const PULSE_MS_MIN: Sample = 0.3;
/// Longest hammer-contact pulse (soft hammer), in milliseconds.
const PULSE_MS_MAX: Sample = 3.0;

/// `ln(1000) == 3 * ln(10)`, used by the `t60`-to-pole-radius mapping.
const LN_1000: Sample = 6.907_755;

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Clamps a fundamental to the tunable range.
#[inline]
fn clamp_frequency(freq_hz: Sample) -> Sample {
    freq_hz.clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ)
}

/// Construction parameters for a [`TineElectricPianoNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TineElectricPianoParams {
    /// Fundamental (strike pitch) in hertz.
    pub frequency_hz: Sample,
    /// Fundamental `-60 dB` decay time in seconds.
    pub decay_s: Sample,
    /// Hammer hardness / brightness in `[0, 1]` (shorter contact when harder).
    pub hardness: Sample,
    /// Bright-partial (tine bell) level in `[0, 1]`.
    pub tine_level: Sample,
    /// Pickup asymmetry in `[MIN_PICKUP_ASYMMETRY, MAX_PICKUP_ASYMMETRY]`
    /// (even-harmonic growl).
    pub pickup_asymmetry: Sample,
    /// Pickup saturation knee in `[MIN_PICKUP_SATURATION, MAX_PICKUP_SATURATION]`
    /// (bark onset; smaller is dirtier).
    pub pickup_saturation: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for TineElectricPianoParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            decay_s: DEFAULT_DECAY_S,
            hardness: DEFAULT_HARDNESS,
            tine_level: DEFAULT_TINE_LEVEL,
            pickup_asymmetry: DEFAULT_PICKUP_ASYMMETRY,
            pickup_saturation: DEFAULT_PICKUP_SATURATION,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl TineElectricPianoParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        Self {
            frequency_hz: clamp_frequency(finite_or(self.frequency_hz, d.frequency_hz)),
            decay_s: finite_or(self.decay_s, d.decay_s).clamp(MIN_DECAY_S, MAX_DECAY_S),
            hardness: finite_or(self.hardness, d.hardness).clamp(0.0, 1.0),
            tine_level: finite_or(self.tine_level, d.tine_level).clamp(0.0, 1.0),
            pickup_asymmetry: finite_or(self.pickup_asymmetry, d.pickup_asymmetry)
                .clamp(MIN_PICKUP_ASYMMETRY, MAX_PICKUP_ASYMMETRY),
            pickup_saturation: finite_or(self.pickup_saturation, d.pickup_saturation)
                .clamp(MIN_PICKUP_SATURATION, MAX_PICKUP_SATURATION),
            amplitude: finite_or(self.amplitude, d.amplitude),
        }
    }
}

/// A struck tine electric-piano voice with a nonlinear electromagnetic pickup.
///
/// See the [module documentation](self) for the cantilever mode model, the
/// pickup nonlinearity, the determinism guarantee, and the real-time contract.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{TineElectricPianoNode, TineElectricPianoParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = TineElectricPianoNode::new(48_000, TineElectricPianoParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The struck tine is audible, bounded, and finite.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak < 1.0 && peak.is_finite());
/// ```
pub struct TineElectricPianoNode {
    sample_rate: u32,
    frequency_hz: Sample,
    decay_s: Sample,
    hardness: Sample,
    tine_level: Sample,
    pickup_asymmetry: Sample,
    pickup_saturation: Sample,
    amplitude: Smoothed,
    // Per-mode resonator coefficients and state.
    a1: [Sample; NUM_TINE_MODES],
    a2: [Sample; NUM_TINE_MODES],
    b0: [Sample; NUM_TINE_MODES],
    enabled: [bool; NUM_TINE_MODES],
    y1: [Sample; NUM_TINE_MODES],
    y2: [Sample; NUM_TINE_MODES],
    // Hammer-pulse geometry and strike state.
    pulse_len: u32,
    pulse_scale: Sample,
    pulse_pos: u32,
    velocity: Sample,
}

impl TineElectricPianoNode {
    /// Builds a tine electric-piano voice for `sample_rate` from `params`,
    /// sanitising every field, and strikes it once so it sounds immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: TineElectricPianoParams) -> Self {
        let p = params.sanitised();
        let mut node = Self {
            sample_rate,
            frequency_hz: p.frequency_hz,
            decay_s: p.decay_s,
            hardness: p.hardness,
            tine_level: p.tine_level,
            pickup_asymmetry: p.pickup_asymmetry,
            pickup_saturation: p.pickup_saturation,
            amplitude: Smoothed::new(p.amplitude),
            a1: [0.0; NUM_TINE_MODES],
            a2: [0.0; NUM_TINE_MODES],
            b0: [0.0; NUM_TINE_MODES],
            enabled: [false; NUM_TINE_MODES],
            y1: [0.0; NUM_TINE_MODES],
            y2: [0.0; NUM_TINE_MODES],
            pulse_len: 1,
            pulse_scale: 1.0,
            pulse_pos: 0,
            velocity: DEFAULT_STRIKE_VELOCITY,
        };
        node.recompute();
        node.strike(DEFAULT_STRIKE_VELOCITY);
        node
    }

    /// Returns the fundamental (strike pitch) in hertz.
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the fundamental `-60 dB` decay time in seconds.
    #[must_use]
    pub fn decay_s(&self) -> Sample {
        self.decay_s
    }

    /// Returns the hammer hardness.
    #[must_use]
    pub fn hardness(&self) -> Sample {
        self.hardness
    }

    /// Returns the bright-partial (tine bell) level.
    #[must_use]
    pub fn tine_level(&self) -> Sample {
        self.tine_level
    }

    /// Returns the pickup asymmetry.
    #[must_use]
    pub fn pickup_asymmetry(&self) -> Sample {
        self.pickup_asymmetry
    }

    /// Returns the pickup saturation knee.
    #[must_use]
    pub fn pickup_saturation(&self) -> Sample {
        self.pickup_saturation
    }

    /// Returns the target linear output amplitude.
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Sets the fundamental, clamped to the tunable range, and recomputes modes.
    pub fn set_frequency(&mut self, frequency_hz: Sample) {
        self.frequency_hz = clamp_frequency(finite_or(frequency_hz, self.frequency_hz));
        self.recompute();
    }

    /// Sets the fundamental decay time, clamped, and recomputes modes.
    pub fn set_decay(&mut self, decay_s: Sample) {
        self.decay_s = finite_or(decay_s, self.decay_s).clamp(MIN_DECAY_S, MAX_DECAY_S);
        self.recompute();
    }

    /// Sets the hammer hardness in `[0, 1]` and recomputes the pulse geometry.
    pub fn set_hardness(&mut self, hardness: Sample) {
        self.hardness = finite_or(hardness, self.hardness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the bright-partial (tine bell) level in `[0, 1]` and recomputes.
    pub fn set_tine_level(&mut self, tine_level: Sample) {
        self.tine_level = finite_or(tine_level, self.tine_level).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the pickup asymmetry (even-harmonic growl).
    pub fn set_pickup_asymmetry(&mut self, asymmetry: Sample) {
        self.pickup_asymmetry = finite_or(asymmetry, self.pickup_asymmetry)
            .clamp(MIN_PICKUP_ASYMMETRY, MAX_PICKUP_ASYMMETRY);
    }

    /// Sets the pickup saturation knee (bark onset).
    pub fn set_pickup_saturation(&mut self, saturation: Sample) {
        self.pickup_saturation = finite_or(saturation, self.pickup_saturation)
            .clamp(MIN_PICKUP_SATURATION, MAX_PICKUP_SATURATION);
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Retriggers the tine with the given strike `velocity` (clamped to
    /// `[0, 1]`), injecting a fresh hammer pulse while existing modes keep
    /// ringing. A harder strike drives the pickup nonlinearity harder.
    pub fn strike(&mut self, velocity: Sample) {
        self.velocity = finite_or(velocity, DEFAULT_STRIKE_VELOCITY).clamp(0.0, 1.0);
        self.pulse_pos = 0;
    }

    /// Recomputes every mode coefficient and the hammer-pulse geometry from the
    /// current scalar parameters. Never runs on the audio hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        let nyquist = sr * NYQUIST_GUARD;

        for m in 0..NUM_TINE_MODES {
            // Mode 1 is the detuned tonebar partner at the fundamental.
            let ratio = if m == 1 {
                1.0 + BEAT_DETUNE
            } else {
                TINE_MODE_RATIOS[m]
            };
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
            // Bright cantilever partials (index >= 2) are scaled by tine_level.
            let gain = if m >= 2 {
                TINE_MODE_GAINS[m] * self.tine_level
            } else {
                TINE_MODE_GAINS[m]
            };
            self.a1[m] = 2.0 * radius * cos_t;
            self.a2[m] = -(radius * radius);
            self.b0[m] = gain * sin_t;
            self.enabled[m] = true;
        }

        let pulse_ms = PULSE_MS_MAX - self.hardness * (PULSE_MS_MAX - PULSE_MS_MIN);
        let len = ops::round(pulse_ms * sr / 1000.0) as i32;
        self.pulse_len = len.max(1) as u32;
        // Unit-area Hann pulse: sum_{n=0}^{L-1} w[n] == (L + 1) / 2.
        self.pulse_scale = 2.0 / (self.pulse_len as Sample + 1.0);
    }

    /// Nonlinear electromagnetic pickup transfer applied to tine displacement.
    #[inline]
    fn pickup(&self, x: Sample) -> Sample {
        let xn = x / self.pickup_saturation;
        (x + self.pickup_asymmetry * x * x) / (1.0 + xn * xn)
    }

    /// Renders one mono output sample, advancing every resonator and the hammer
    /// pulse by one step and reading the tine through the pickup.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let drive = if self.pulse_pos < self.pulse_len {
            let n = self.pulse_pos as Sample;
            self.pulse_pos += 1;
            let window =
                0.5 - 0.5 * ops::cos(TAU * (n + 1.0) / (self.pulse_len as Sample + 1.0));
            window * self.pulse_scale * self.velocity
        } else {
            0.0
        };

        let mut acc = 0.0;
        for m in 0..NUM_TINE_MODES {
            if !self.enabled[m] {
                continue;
            }
            let y = flush_denormal(
                self.b0[m] * drive + self.a1[m] * self.y1[m] + self.a2[m] * self.y2[m],
            );
            self.y2[m] = self.y1[m];
            self.y1[m] = y;
            acc += y;
        }

        let voltage = self.pickup(acc * EXCITATION_GAIN);
        let amp = self.amplitude.next_sample();
        flush_denormal(voltage * OUTPUT_GAIN * amp)
    }
}

impl AudioNode for TineElectricPianoNode {
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
        self.y1 = [0.0; NUM_TINE_MODES];
        self.y2 = [0.0; NUM_TINE_MODES];
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
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use alloc::vec::Vec;

    const SR: u32 = 48_000;

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut TineElectricPianoNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(
        node: &mut TineElectricPianoNode,
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
    fn renders_bounded_finite() {
        let mut node = TineElectricPianoNode::new(SR, TineElectricPianoParams::default());
        let out = render(&mut node, SR as usize);
        let p = peak(&out);
        assert!(p > 0.0 && p < 1.0, "default peak = {p}");
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn peak_grid_stays_below_full_scale() {
        let freqs = [55.0, 110.0, 220.0, 440.0, 880.0, 1760.0];
        let hardnesses = [0.0, 0.5, 1.0];
        let tine_levels = [0.0, 0.6, 1.0];
        let asyms = [0.0, 0.3, 1.0];
        let sats = [MIN_PICKUP_SATURATION, 0.8, MAX_PICKUP_SATURATION];
        let mut worst = 0.0_f32;
        for &f in &freqs {
            for &h in &hardnesses {
                for &t in &tine_levels {
                    for &a in &asyms {
                        for &s in &sats {
                            let mut node = TineElectricPianoNode::new(
                                SR,
                                TineElectricPianoParams {
                                    frequency_hz: f,
                                    decay_s: DEFAULT_DECAY_S,
                                    hardness: h,
                                    tine_level: t,
                                    pickup_asymmetry: a,
                                    pickup_saturation: s,
                                    amplitude: 1.0,
                                },
                            );
                            node.strike(1.0);
                            let out = render(&mut node, SR as usize / 2);
                            worst = worst.max(peak(&out));
                            assert!(
                                out.iter().all(|v| v.is_finite()),
                                "non-finite at f={f} h={h} t={t} a={a} s={s}"
                            );
                        }
                    }
                }
            }
        }
        assert!(worst < 1.0, "grid worst peak = {worst}");
        assert!(worst > 0.3, "grid worst peak too quiet = {worst}");
    }

    #[test]
    fn struck_sounds_immediately() {
        let mut node = TineElectricPianoNode::new(SR, TineElectricPianoParams::default());
        let out = render(&mut node, 256);
        assert!(peak(&out) > 1.0e-4, "attack should be audible");
    }

    #[test]
    fn strike_retriggers() {
        let mut node = TineElectricPianoNode::new(SR, TineElectricPianoParams::default());
        let _ = render(&mut node, 3 * SR as usize);
        let decayed = energy(&render(&mut node, SR as usize / 20));
        node.strike(1.0);
        let restruck = energy(&render(&mut node, SR as usize / 20));
        assert!(
            restruck > decayed * 4.0,
            "re-strike should revive energy: decayed = {decayed}, restruck = {restruck}"
        );
    }

    #[test]
    fn harder_strike_is_brighter() {
        let params = TineElectricPianoParams {
            pickup_asymmetry: 0.6,
            pickup_saturation: 0.6,
            ..Default::default()
        };
        let f0 = params.frequency_hz;

        let mut soft = TineElectricPianoNode::new(SR, params);
        soft.strike(0.2);
        let soft_out = render(&mut soft, SR as usize / 4);

        let mut hard = TineElectricPianoNode::new(SR, params);
        hard.strike(1.0);
        let hard_out = render(&mut hard, SR as usize / 4);

        let soft_ratio = goertzel(&soft_out, 2.0 * f0) / goertzel(&soft_out, f0).max(1.0e-9);
        let hard_ratio = goertzel(&hard_out, 2.0 * f0) / goertzel(&hard_out, f0).max(1.0e-9);
        assert!(
            hard_ratio > soft_ratio,
            "harder strike should add harmonics: soft = {soft_ratio}, hard = {hard_ratio}"
        );
    }

    #[test]
    fn pickup_asymmetry_adds_even_harmonics() {
        let base = TineElectricPianoParams {
            pickup_asymmetry: 0.0,
            ..Default::default()
        };
        let f0 = base.frequency_hz;
        let mut sym = TineElectricPianoNode::new(SR, base);
        let sym_out = render(&mut sym, SR as usize / 2);

        let mut asym = TineElectricPianoNode::new(
            SR,
            TineElectricPianoParams {
                pickup_asymmetry: 0.9,
                ..base
            },
        );
        let asym_out = render(&mut asym, SR as usize / 2);

        let sym_even = goertzel(&sym_out, 2.0 * f0);
        let asym_even = goertzel(&asym_out, 2.0 * f0);
        assert!(
            asym_even > sym_even * 1.5,
            "asymmetry should raise the 2nd harmonic: sym = {sym_even}, asym = {asym_even}"
        );
    }

    #[test]
    fn tine_level_controls_brightness() {
        let f0 = DEFAULT_FREQUENCY_HZ;
        let bright_f = f0 * TINE_MODE_RATIOS[2];

        let mut dark = TineElectricPianoNode::new(
            SR,
            TineElectricPianoParams {
                tine_level: 0.0,
                ..Default::default()
            },
        );
        let dark_out = render(&mut dark, SR as usize / 4);

        let mut bright = TineElectricPianoNode::new(
            SR,
            TineElectricPianoParams {
                tine_level: 1.0,
                ..Default::default()
            },
        );
        let bright_out = render(&mut bright, SR as usize / 4);

        assert!(
            goertzel(&bright_out, bright_f) > goertzel(&dark_out, bright_f) * 1.5,
            "higher tine_level should raise the bright cantilever partial"
        );
    }

    #[test]
    fn fundamental_present() {
        let mut node = TineElectricPianoNode::new(SR, TineElectricPianoParams::default());
        let out = render(&mut node, SR as usize / 2);
        let f0 = DEFAULT_FREQUENCY_HZ;
        let fund = goertzel(&out, f0);
        let off = goertzel(&out, f0 * 1.37);
        assert!(fund > off * 2.0, "fundamental should dominate: {fund} vs {off}");
    }

    #[test]
    fn frequency_change_moves_spectrum() {
        let mut node = TineElectricPianoNode::new(SR, TineElectricPianoParams::default());
        node.set_frequency(330.0);
        node.strike(1.0);
        let out = render(&mut node, SR as usize / 2);
        assert!(
            goertzel(&out, 330.0) > goertzel(&out, DEFAULT_FREQUENCY_HZ) * 1.5,
            "spectrum should follow the new fundamental"
        );
    }

    #[test]
    fn shorter_decay_dies_faster() {
        let mut quick = TineElectricPianoNode::new(
            SR,
            TineElectricPianoParams {
                decay_s: 0.3,
                ..Default::default()
            },
        );
        let quick_out = render(&mut quick, 2 * SR as usize);
        let quick_head = energy(&quick_out[..SR as usize / 10]);
        let quick_tail = energy(&quick_out[quick_out.len() - SR as usize / 10..]);

        let mut slow = TineElectricPianoNode::new(
            SR,
            TineElectricPianoParams {
                decay_s: 10.0,
                ..Default::default()
            },
        );
        let slow_out = render(&mut slow, 2 * SR as usize);
        let slow_head = energy(&slow_out[..SR as usize / 10]);
        let slow_tail = energy(&slow_out[slow_out.len() - SR as usize / 10..]);

        assert!(
            quick_tail / quick_head < slow_tail / slow_head,
            "shorter decay should fade faster"
        );
    }

    #[test]
    fn pickup_bounded_under_hard_strike() {
        let mut node = TineElectricPianoNode::new(
            SR,
            TineElectricPianoParams {
                pickup_saturation: MIN_PICKUP_SATURATION,
                pickup_asymmetry: 1.0,
                amplitude: 1.0,
                ..Default::default()
            },
        );
        node.strike(1.0);
        let out = render(&mut node, SR as usize / 2);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(peak(&out) < 1.0, "saturation must bound the output");
    }

    #[test]
    fn deterministic_across_instances() {
        let params = TineElectricPianoParams::default();
        let mut a = TineElectricPianoNode::new(SR, params);
        let mut b = TineElectricPianoNode::new(SR, params);
        let oa = render(&mut a, SR as usize / 2);
        let ob = render(&mut b, SR as usize / 2);
        assert_eq!(oa, ob);
    }

    #[test]
    fn reset_replays_identical_attack() {
        let mut node = TineElectricPianoNode::new(SR, TineElectricPianoParams::default());
        let first = render(&mut node, SR as usize / 2);
        node.reset();
        let second = render(&mut node, SR as usize / 2);
        assert_eq!(first, second);
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = TineElectricPianoNode::new(
            SR,
            TineElectricPianoParams {
                amplitude: 0.0,
                ..Default::default()
            },
        );
        let out = render(&mut node, SR as usize / 2);
        assert!(peak(&out) == 0.0, "zero amplitude must be silent");
    }

    #[test]
    fn amplitude_squared_scales_energy() {
        let mut quiet = TineElectricPianoNode::new(
            SR,
            TineElectricPianoParams {
                amplitude: 0.25,
                ..Default::default()
            },
        );
        let quiet_e = energy(&render(&mut quiet, SR as usize / 2));

        let mut loud = TineElectricPianoNode::new(
            SR,
            TineElectricPianoParams {
                amplitude: 0.5,
                ..Default::default()
            },
        );
        let loud_e = energy(&render(&mut loud, SR as usize / 2));

        let ratio = loud_e / quiet_e;
        assert!(
            (ratio - 4.0).abs() < 0.1,
            "doubling amplitude should quadruple energy, ratio = {ratio}"
        );
    }

    #[test]
    fn mono_core_copied_to_channels() {
        let mut node = TineElectricPianoNode::new(SR, TineElectricPianoParams::default());
        let chans = render_layout(&mut node, SR as usize / 4, ChannelLayout::Quad);
        assert_eq!(chans.len(), 4);
        for ch in 1..chans.len() {
            assert_eq!(chans[0], chans[ch], "channel {ch} should mirror the core");
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = TineElectricPianoNode::new(SR, TineElectricPianoParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn latency_is_zero() {
        let node = TineElectricPianoNode::new(SR, TineElectricPianoParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_sanitised_state() {
        let node = TineElectricPianoNode::new(
            SR,
            TineElectricPianoParams {
                frequency_hz: 262.0,
                decay_s: 3.0,
                hardness: 0.7,
                tine_level: 0.4,
                pickup_asymmetry: 0.5,
                pickup_saturation: 1.2,
                amplitude: 0.3,
            },
        );
        assert_eq!(node.frequency_hz(), 262.0);
        assert_eq!(node.decay_s(), 3.0);
        assert_eq!(node.hardness(), 0.7);
        assert_eq!(node.tine_level(), 0.4);
        assert_eq!(node.pickup_asymmetry(), 0.5);
        assert_eq!(node.pickup_saturation(), 1.2);
        assert_eq!(node.amplitude(), 0.3);
    }

    #[test]
    fn default_params_in_domain() {
        let d = TineElectricPianoParams::default();
        assert_eq!(d, d.sanitised());
        assert_eq!(d.sanitised(), d.sanitised().sanitised());
    }

    #[test]
    fn sanitise_clamps_and_repairs() {
        let p = TineElectricPianoParams {
            frequency_hz: f32::NAN,
            decay_s: 1.0e9,
            hardness: 5.0,
            tine_level: -2.0,
            pickup_asymmetry: 9.0,
            pickup_saturation: 0.0,
            amplitude: f32::INFINITY,
        }
        .sanitised();
        assert_eq!(p.frequency_hz, DEFAULT_FREQUENCY_HZ);
        assert_eq!(p.decay_s, MAX_DECAY_S);
        assert_eq!(p.hardness, 1.0);
        assert_eq!(p.tine_level, 0.0);
        assert_eq!(p.pickup_asymmetry, MAX_PICKUP_ASYMMETRY);
        assert_eq!(p.pickup_saturation, MIN_PICKUP_SATURATION);
        assert_eq!(p.amplitude, DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = TineElectricPianoNode::new(SR, TineElectricPianoParams::default());

        node.set_frequency(1.0e9);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);
        node.set_frequency(f32::NAN);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);

        node.set_decay(0.0);
        assert_eq!(node.decay_s(), MIN_DECAY_S);
        node.set_decay(f32::NAN);
        assert_eq!(node.decay_s(), MIN_DECAY_S);

        node.set_hardness(9.0);
        assert_eq!(node.hardness(), 1.0);
        node.set_tine_level(-1.0);
        assert_eq!(node.tine_level(), 0.0);

        node.set_pickup_asymmetry(9.0);
        assert_eq!(node.pickup_asymmetry(), MAX_PICKUP_ASYMMETRY);
        node.set_pickup_saturation(0.0);
        assert_eq!(node.pickup_saturation(), MIN_PICKUP_SATURATION);

        node.set_amplitude(f32::NAN, Ramp::Immediate);
        assert!(node.amplitude().is_finite());
    }
}
