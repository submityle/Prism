//! Struck church / cast-bell modal percussion source (bell / carillon /
//! bell-plate family).
//!
//! [`BellNode`] is a *source* (zero inputs, one output) that synthesizes a
//! clapper-struck cast bell by *modal synthesis*: a short contact-force pulse
//! excites a parallel bank of [`NUM_MODES`] independently decaying two-pole
//! resonators tuned to the named partials of a Western church bell. The summed
//! output is the inharmonic, slowly beating peal of a bronze bell: a bright
//! strike that resolves into a long, humming after-ring.
//!
//! # Model
//!
//! A cast bell is a thick axisymmetric shell, not a flat plate or a thin bar.
//! Its lowest audible partials are a small, well-documented, strongly
//! inharmonic set that bell founders have tuned for centuries, named relative
//! to the perceived strike note (the *prime*):
//!
//! | partial     | ratio to prime | role                         |
//! |-------------|----------------|------------------------------|
//! | hum         | `0.5`          | octave-below drone, longest  |
//! | prime       | `1.0`          | perceived strike note        |
//! | tierce      | `1.2`          | **minor** third, bell colour |
//! | quint       | `1.5`          | fifth                        |
//! | nominal     | `2.0`          | octave, carries the strike   |
//! | deciem      | `2.5`          | major third above nominal    |
//! | undeciem    | `2.667`        | fourth above nominal         |
//! | duodecime   | `3.0`          | twelfth                      |
//! | upper octave| `4.0`          | double octave                |
//! | ...         | `5.333..8.0`   | fast-fading upper shell modes|
//!
//! The characteristic *minor-third* tierce is what gives a church bell its
//! brooding voice (a "major-third bell" moves it to `~1.26`); this node fixes
//! the classic minor third. Each partial is realised as a two-pole resonator
//! `y[n] = b0 * x[n] + a1 * y[n-1] + a2 * y[n-2]` whose complex pole pair sits
//! at radius `R = exp(-ln(1000) / (t60 * sample_rate))` and angle
//! `theta = 2*pi*f_m / sample_rate`, giving `a1 = 2*R*cos(theta)` and
//! `a2 = -R*R`. Its impulse response is a sinusoid at `f_m` decaying by
//! `-60 dB` over `t60` seconds. The feed gain `b0 = gain_m * sin(theta)`
//! normalises the ringing peak to `gain_m` independently of the decay radius.
//! The low partials ring far longer than the high ones (`t60_m = decay /
//! ratio_m^0.7`), so the hum and prime drone on while the shimmering upper
//! shell modes fade within a second -- the signature bell decay profile.
//!
//! A real casting is never perfectly axisymmetric, so each partial actually
//! appears as a close *doublet*: two near-degenerate modes a fraction of a
//! percent apart that beat slowly against one another, producing the gentle
//! warble of a struck bell. Each named partial is therefore voiced as two
//! resonators split by `+/- warble * WARBLE_MAX_DETUNE`; `warble == 0` collapses
//! the doublet to a pure tone, higher values widen the beat.
//!
//! The excitation `x[n]` is a single raised-cosine (Hann) contact-force pulse,
//! normalised to unit area so each strike imparts a fixed momentum regardless
//! of its width. A hard clapper is modelled by a short pulse (bright, lots of
//! high-partial energy) and a soft one by a long pulse; `brightness` also tilts
//! the per-partial gain, cutting the upper shell modes for a mellow strike. The
//! node is struck once at construction so it sounds immediately;
//! [`BellNode::strike`] retriggers it, adding a fresh pulse while the existing
//! partials keep ringing.
//!
//! # Determinism
//!
//! The excitation is a closed-form deterministic pulse, not noise, so the node
//! holds no random state: two [`BellNode`]s built with the same sample rate and
//! parameters produce bit-identical output, and [`BellNode::reset`] clears the
//! resonators and re-strikes to replay the identical attack.
//!
//! # Real-time contract
//!
//! All per-mode coefficient and history storage is a fixed-size array sized for
//! [`NUM_MODES`]; [`BellNode::process`] performs no allocation, locking, or
//! panic on the hot path. Partial ratios and strengths are const tables, so
//! [`BellNode::recompute`] only evaluates closed-form coefficients off the hot
//! path. Non-finite parameters are sanitised on the way in and outputs are
//! flushed of denormals, so the generator cannot stall the audio thread.
//! Latency is zero.
//!
//! # Relationship
//!
//! Unlike the sibling [`modal_resonator`](crate::nodes::effects::modal_resonator)
//! *effect*, which filters an *external* input signal through a modal bank, this
//! *source* supplies its own strike excitation and needs no input. It shares the
//! two-pole modal-resonator engine of its percussion siblings but a distinct
//! partial set: [`struck_bar`](crate::nodes::sources::struck_bar) uses the
//! sparse one-dimensional free-free beam series (`1 : 2.76 : 5.40 : ...`) of a
//! glockenspiel or chime, and [`struck_plate`](crate::nodes::sources::struck_plate)
//! the dense two-dimensional Kirchhoff grid of a gong; a cast bell's thick
//! three-dimensional shell instead rings at the named hum / prime / tierce /
//! quint / nominal set tabulated above, with its defining sub-octave hum and
//! minor-third tierce. It differs again from
//! [`membrane_drum`](crate::nodes::sources::membrane_drum), whose tension-restored
//! circular membrane follows the Bessel-zero ratios and decays quickly.
//!
//! # Provenance
//!
//! Modal synthesis of a struck resonator is a textbook technique; the named
//! church-bell partials (hum, prime, tierce, quint, nominal, and the upper
//! shell modes) are tabulated in acoustics texts (for example N. H. Fletcher
//! and T. D. Rossing, *The Physics of Musical Instruments*, and T. D. Rossing,
//! *Science of Percussion Instruments*). The two-pole resonator, the
//! `t60`-to-pole-radius mapping, the Hann (raised-cosine) contact pulse, the
//! near-degenerate doublet warble, and the clapper-hardness spectral envelope
//! are standard, publicly documented DSP. This is pure classic DSP with no AI
//! or ML. This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, Google Resonance Audio, Web Audio, or STK source or derived
//! code**; only the widely documented bell-partial ratios, resonator, and
//! window formulas are used.

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Number of named church-bell partials voiced by the node.
pub const NUM_PARTIALS: usize = 12;

/// Number of parallel two-pole resonators (one near-degenerate doublet per
/// partial: `NUM_PARTIALS * 2`).
pub const NUM_MODES: usize = NUM_PARTIALS * 2;

/// Named church-bell partial ratios, relative to the prime (strike note).
///
/// Hum, prime, (minor) tierce, quint, nominal, deciem, undeciem, duodecime,
/// upper octave, and three fast-fading upper shell modes.
const PARTIAL_RATIOS: [Sample; NUM_PARTIALS] = [
    0.5, 1.0, 1.2, 1.5, 2.0, 2.5, 2.667, 3.0, 4.0, 5.333, 6.667, 8.0,
];

/// Relative linear strike strength of each partial, shaping the bell timbre so
/// the prime, nominal, hum, and tierce dominate the strike while the high shell
/// modes stay faint. Modulated further by the `brightness` tilt.
const PARTIAL_GAINS: [Sample; NUM_PARTIALS] = [
    0.8, 1.0, 0.6, 0.4, 0.9, 0.3, 0.25, 0.4, 0.3, 0.15, 0.1, 0.1,
];

/// Lowest tunable prime (strike pitch) in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Highest tunable prime in hertz (further bounded by the Nyquist limit).
pub const MAX_FREQUENCY_HZ: Sample = 12_000.0;

/// Default prime (strike pitch) frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 262.0;

/// Shortest `-60 dB` decay time, in seconds, the prime may request.
pub const MIN_DECAY_S: Sample = 0.05;

/// Longest `-60 dB` decay time, in seconds, the prime may request (bells ring).
pub const MAX_DECAY_S: Sample = 40.0;

/// Default prime `-60 dB` decay time in seconds.
pub const DEFAULT_DECAY_S: Sample = 6.0;

/// Default clapper hardness / brightness in `[0, 1]`.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Lowest warble (doublet beating) amount (`0` is a pure, non-beating tone).
pub const MIN_WARBLE: Sample = 0.0;

/// Highest warble (doublet beating) amount.
pub const MAX_WARBLE: Sample = 1.0;

/// Default warble amount (a gentle, lifelike beat).
pub const DEFAULT_WARBLE: Sample = 0.3;

/// Maximum fractional detune of a doublet member at `warble == 1`.
const WARBLE_MAX_DETUNE: Sample = 0.008;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Default strike velocity used by [`BellNode::strike`].
pub const DEFAULT_STRIKE_VELOCITY: Sample = 1.0;

/// `ln(1000) == 3 * ln(10)`, used by the `t60`-to-pole-radius mapping.
const LN_1000: Sample = 6.907_755;

/// Fraction of the sample rate above which a mode is muted (anti-alias guard).
const NYQUIST_GUARD: Sample = 0.49;

/// Exponent controlling how much faster high partials decay than the prime.
const DECAY_RATIO_EXP: Sample = 0.7;

/// Softest per-partial gain tilt (brightest clapper, keeps the high modes).
const TILT_MIN: Sample = 0.0;

/// Steepest per-partial gain tilt (dullest clapper, cuts the high modes).
const TILT_MAX: Sample = 1.2;

/// Shortest clapper-contact pulse (hardest clapper), in milliseconds.
const PULSE_MS_MIN: Sample = 0.15;

/// Longest clapper-contact pulse (softest clapper), in milliseconds.
const PULSE_MS_MAX: Sample = 3.5;

/// Overall output scale, keeping the summed modal peak below full scale.
///
/// Calibrated so the worst-case parameter grid peaks at `0.86` with
/// `amplitude == 1`; see the module tests.
const OUTPUT_GAIN: Sample = 0.25;

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Clamps a prime frequency to the tunable range, honouring the Nyquist guard.
#[inline]
fn clamp_frequency(freq_hz: Sample, sample_rate: u32) -> Sample {
    let nyquist = sample_rate.max(1) as Sample * NYQUIST_GUARD;
    let upper = MAX_FREQUENCY_HZ.min(nyquist).max(MIN_FREQUENCY_HZ);
    freq_hz.clamp(MIN_FREQUENCY_HZ, upper)
}

/// Construction parameters for a [`BellNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BellParams {
    /// Prime (strike pitch) frequency in hertz.
    pub frequency_hz: Sample,
    /// Prime `-60 dB` decay time in seconds (longer rings longer).
    pub decay_s: Sample,
    /// Clapper hardness / brightness in `[0, 1]` (`1` is a hard, bright clapper).
    pub brightness: Sample,
    /// Warble (doublet beating) amount in `[0, 1]` (`0` is a pure tone).
    pub warble: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for BellParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            decay_s: DEFAULT_DECAY_S,
            brightness: DEFAULT_BRIGHTNESS,
            warble: DEFAULT_WARBLE,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl BellParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range. Frequency is clamped against the Nyquist limit too.
    #[must_use]
    pub fn sanitised(self, sample_rate: u32) -> Self {
        let d = Self::default();
        let frequency_hz =
            clamp_frequency(finite_or(self.frequency_hz, d.frequency_hz), sample_rate);
        let decay_s = finite_or(self.decay_s, d.decay_s).clamp(MIN_DECAY_S, MAX_DECAY_S);
        let brightness = finite_or(self.brightness, d.brightness).clamp(0.0, 1.0);
        let warble = finite_or(self.warble, d.warble).clamp(MIN_WARBLE, MAX_WARBLE);
        let amplitude = finite_or(self.amplitude, d.amplitude);
        Self {
            frequency_hz,
            decay_s,
            brightness,
            warble,
            amplitude,
        }
    }
}

/// A clapper-struck church / cast-bell modal percussion source.
///
/// See the [module documentation](self) for the model, determinism guarantee,
/// and real-time contract.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{BellNode, BellParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = BellNode::new(48_000, BellParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The strike excites the inharmonic bell partials, which ring and decay.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
pub struct BellNode {
    sample_rate: u32,
    frequency_hz: Sample,
    decay_s: Sample,
    brightness: Sample,
    warble: Sample,
    amplitude: Smoothed,
    a1: [Sample; NUM_MODES],
    a2: [Sample; NUM_MODES],
    b0: [Sample; NUM_MODES],
    enabled: [bool; NUM_MODES],
    y1: [Sample; NUM_MODES],
    y2: [Sample; NUM_MODES],
    pulse_len: u32,
    pulse_pos: u32,
    pulse_scale: Sample,
    velocity: Sample,
}

impl BellNode {
    /// Builds a bell voice for `sample_rate` from `params`, sanitising every
    /// field, then strikes it once so it sounds immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: BellParams) -> Self {
        let p = params.sanitised(sample_rate);
        let mut node = Self {
            sample_rate,
            frequency_hz: p.frequency_hz,
            decay_s: p.decay_s,
            brightness: p.brightness,
            warble: p.warble,
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

    /// Returns the current prime (strike pitch) in hertz.
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the prime `-60 dB` decay time in seconds.
    #[must_use]
    pub fn decay_s(&self) -> Sample {
        self.decay_s
    }

    /// Returns the clapper hardness / brightness in `[0, 1]`.
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
    }

    /// Returns the warble (doublet beating) amount in `[0, 1]`.
    #[must_use]
    pub fn warble(&self) -> Sample {
        self.warble
    }

    /// Returns the target linear output amplitude.
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Reports whether a given mode index is currently audible (below Nyquist).
    #[must_use]
    pub fn mode_enabled(&self, mode: usize) -> bool {
        self.enabled.get(mode).copied().unwrap_or(false)
    }

    /// Sets the prime (strike pitch), clamped to the tunable range.
    pub fn set_frequency(&mut self, frequency_hz: Sample) {
        self.frequency_hz =
            clamp_frequency(finite_or(frequency_hz, self.frequency_hz), self.sample_rate);
        self.recompute();
    }

    /// Sets the prime `-60 dB` decay time, clamped to the valid range.
    pub fn set_decay(&mut self, decay_s: Sample) {
        self.decay_s = finite_or(decay_s, self.decay_s).clamp(MIN_DECAY_S, MAX_DECAY_S);
        self.recompute();
    }

    /// Sets the clapper hardness / brightness, clamped to `[0, 1]`.
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the warble (doublet beating) amount, clamped to `[0, 1]`.
    pub fn set_warble(&mut self, warble: Sample) {
        self.warble = finite_or(warble, self.warble).clamp(MIN_WARBLE, MAX_WARBLE);
        self.recompute();
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Retriggers the bell with the given strike `velocity` (clamped to
    /// `[0, 1]`), injecting a fresh clapper pulse while existing partials keep
    /// ringing.
    pub fn strike(&mut self, velocity: Sample) {
        self.velocity = finite_or(velocity, DEFAULT_STRIKE_VELOCITY).clamp(0.0, 1.0);
        self.pulse_pos = 0;
    }

    /// Recomputes every mode coefficient and the clapper-pulse geometry from the
    /// current scalar parameters. Never runs on the audio hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        let tilt = TILT_MAX - self.brightness * (TILT_MAX - TILT_MIN);
        let detune = self.warble * WARBLE_MAX_DETUNE;
        let nyquist = sr * NYQUIST_GUARD;

        for (p, &ratio) in PARTIAL_RATIOS.iter().enumerate() {
            let base_gain = PARTIAL_GAINS[p];
            let t60 =
                (self.decay_s / ops::powf(ratio, DECAY_RATIO_EXP)).clamp(MIN_DECAY_S, MAX_DECAY_S);
            let radius = ops::exp(-LN_1000 / (t60 * sr));
            // Split each partial into a near-degenerate beating doublet.
            for d in 0..2 {
                let m = p * 2 + d;
                let sign = if d == 0 { -1.0 } else { 1.0 };
                let f_m = self.frequency_hz * ratio * (1.0 + sign * detune);
                if f_m <= 0.0 || f_m >= nyquist {
                    self.a1[m] = 0.0;
                    self.a2[m] = 0.0;
                    self.b0[m] = 0.0;
                    self.enabled[m] = false;
                    continue;
                }
                let theta = TAU * f_m / sr;
                let (sin_t, cos_t) = (ops::sin(theta), ops::cos(theta));
                // Half strength per doublet member keeps the summed partial gain
                // independent of the warble split.
                let gain = 0.5 * base_gain * ops::powf(ratio, -tilt);
                self.a1[m] = 2.0 * radius * cos_t;
                self.a2[m] = -(radius * radius);
                self.b0[m] = gain * sin_t;
                self.enabled[m] = true;
            }
        }

        let pulse_ms = PULSE_MS_MAX - self.brightness * (PULSE_MS_MAX - PULSE_MS_MIN);
        let len = ops::round(pulse_ms * sr / 1000.0) as i32;
        self.pulse_len = len.max(1) as u32;
        // Unit-area Hann pulse: sum_{n=0}^{L-1} w[n] == (L + 1) / 2.
        self.pulse_scale = 2.0 / (self.pulse_len as Sample + 1.0);
    }

    /// Renders one mono output sample, advancing every resonator and the clapper
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
            let y = flush_denormal(
                self.b0[m] * drive + self.a1[m] * self.y1[m] + self.a2[m] * self.y2[m],
            );
            self.y2[m] = self.y1[m];
            self.y1[m] = y;
            acc += y;
        }

        let amp = self.amplitude.next_sample();
        flush_denormal(acc * OUTPUT_GAIN * amp)
    }
}

impl AudioNode for BellNode {
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
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use alloc::vec::Vec;

    const SR: u32 = 48_000;

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut BellNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(node: &mut BellNode, frames: usize, layout: ChannelLayout) -> Vec<Vec<Sample>> {
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

    /// Coefficient of variation of the block-wise RMS envelope, a proxy for the
    /// slow amplitude modulation (beating) a warbling doublet imposes.
    fn envelope_cv(block: &[Sample], block_len: usize) -> f64 {
        let mut rms = Vec::new();
        let mut i = 0;
        while i + block_len <= block.len() {
            let e: f64 = block[i..i + block_len]
                .iter()
                .map(|&s| (s as f64) * (s as f64))
                .sum();
            rms.push((e / block_len as f64).sqrt());
            i += block_len;
        }
        let mean = rms.iter().sum::<f64>() / rms.len() as f64;
        let var = rms.iter().map(|&r| (r - mean) * (r - mean)).sum::<f64>() / rms.len() as f64;
        var.sqrt() / (mean + 1.0e-12)
    }

    #[test]
    fn default_strike_produces_sound() {
        let mut node = BellNode::new(SR, BellParams::default());
        let out = render(&mut node, SR as usize / 10);
        let p = peak(&out);
        assert!(p > 1.0e-3, "bell should ring, peak = {p}");
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn tone_decays_toward_silence() {
        let mut node = BellNode::new(
            SR,
            BellParams {
                decay_s: 1.0,
                ..BellParams::default()
            },
        );
        let out = render(&mut node, 6 * SR as usize);
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
        let mut node = BellNode::new(SR, BellParams::default());
        let out = render(&mut node, 4 * SR as usize);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(peak(&out) < 1.0, "peak = {}", peak(&out));
    }

    #[test]
    fn full_parameter_grid_stays_below_full_scale() {
        let mut worst = 0.0_f32;
        for &f0 in &[40.0, 131.0, 262.0, 880.0, 2000.0] {
            for &decay in &[0.1, 2.0, 12.0, 40.0] {
                for &bright in &[0.0, 0.5, 1.0] {
                    for &warble in &[0.0, 0.3, 1.0] {
                        let params = BellParams {
                            frequency_hz: f0,
                            decay_s: decay,
                            brightness: bright,
                            warble,
                            amplitude: 1.0,
                        };
                        let mut node = BellNode::new(SR, params);
                        let out = render(&mut node, SR as usize / 4);
                        let p = peak(&out);
                        worst = worst.max(p);
                        assert!(
                            p < 1.0 && out.iter().all(|s| s.is_finite()),
                            "f0={f0} decay={decay} bright={bright} warble={warble} peak={p}"
                        );
                    }
                }
            }
        }
        // Pin the calibrated worst-case headroom so OUTPUT_GAIN drift is caught.
        assert!(
            (0.80..0.95).contains(&worst),
            "grid worst-case peak should stay in the calibrated band: {worst}"
        );
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = BellNode::new(SR, BellParams::default());
        let mut b = BellNode::new(SR, BellParams::default());
        let out_a = render(&mut a, SR as usize);
        let out_b = render(&mut b, SR as usize);
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn reset_replays_identical_attack() {
        let mut node = BellNode::new(SR, BellParams::default());
        let first = render(&mut node, SR as usize / 2);
        node.reset();
        let after = render(&mut node, SR as usize / 2);
        assert_eq!(first, after);
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut loud = BellNode::new(
            SR,
            BellParams {
                amplitude: 1.0,
                ..BellParams::default()
            },
        );
        let mut soft = BellNode::new(
            SR,
            BellParams {
                amplitude: 0.5,
                ..BellParams::default()
            },
        );
        let el = energy(&render(&mut loud, SR as usize / 2));
        let es = energy(&render(&mut soft, SR as usize / 2));
        let ratio = el / es;
        assert!((ratio - 4.0).abs() < 1.0e-2, "energy ratio = {ratio}");
    }

    #[test]
    fn prime_partial_present() {
        let f0 = 262.0;
        let mut node = BellNode::new(
            SR,
            BellParams {
                frequency_hz: f0,
                ..BellParams::default()
            },
        );
        let out = render(&mut node, SR as usize / 2);
        assert!(
            goertzel(&out, f0) > 10.0,
            "prime should ring: {}",
            goertzel(&out, f0)
        );
    }

    #[test]
    fn tierce_is_a_minor_third() {
        let f0 = 262.0;
        let mut node = BellNode::new(
            SR,
            BellParams {
                frequency_hz: f0,
                warble: 0.0,
                ..BellParams::default()
            },
        );
        let out = render(&mut node, SR as usize / 2);
        let minor = goertzel(&out, f0 * 1.2);
        let major = goertzel(&out, f0 * 1.26);
        assert!(
            minor > major * 2.0,
            "tierce should sit at the minor third: minor = {minor}, major = {major}"
        );
    }

    #[test]
    fn hum_outlasts_higher_partials() {
        let f0 = 262.0;
        let mut node = BellNode::new(
            SR,
            BellParams {
                frequency_hz: f0,
                decay_s: 8.0,
                warble: 0.0,
                ..BellParams::default()
            },
        );
        let out = render(&mut node, 4 * SR as usize);
        let late = &out[3 * SR as usize..];
        let hum = goertzel(late, f0 * 0.5);
        let nominal = goertzel(late, f0 * 2.0);
        assert!(
            hum > nominal * 4.0,
            "hum should outlast the nominal: hum = {hum}, nominal = {nominal}"
        );
    }

    #[test]
    fn warble_increases_beating() {
        let f0 = 262.0;
        let mut steady = BellNode::new(
            SR,
            BellParams {
                frequency_hz: f0,
                decay_s: 8.0,
                warble: 0.0,
                ..BellParams::default()
            },
        );
        let mut beating = BellNode::new(
            SR,
            BellParams {
                frequency_hz: f0,
                decay_s: 8.0,
                warble: 1.0,
                ..BellParams::default()
            },
        );
        let start = (SR as f32 * 0.3) as usize;
        let block = (SR as f32 * 0.02) as usize;
        let s = render(&mut steady, SR as usize);
        let b = render(&mut beating, SR as usize);
        let cv_steady = envelope_cv(&s[start..], block);
        let cv_beating = envelope_cv(&b[start..], block);
        assert!(
            cv_beating > cv_steady * 1.3,
            "warble should add beating: steady = {cv_steady}, beating = {cv_beating}"
        );
    }

    #[test]
    fn harder_clapper_is_brighter() {
        let f0 = 262.0;
        let mut bright = BellNode::new(
            SR,
            BellParams {
                frequency_hz: f0,
                brightness: 1.0,
                ..BellParams::default()
            },
        );
        let mut dull = BellNode::new(
            SR,
            BellParams {
                frequency_hz: f0,
                brightness: 0.0,
                ..BellParams::default()
            },
        );
        let hb = hf_energy(&render(&mut bright, SR as usize / 3));
        let hd = hf_energy(&render(&mut dull, SR as usize / 3));
        assert!(hb > hd * 1.5, "hard clapper should be brighter: {hb} vs {hd}");
    }

    #[test]
    fn frequency_changes_output() {
        let mut low = BellNode::new(
            SR,
            BellParams {
                frequency_hz: 150.0,
                ..BellParams::default()
            },
        );
        let mut high = BellNode::new(
            SR,
            BellParams {
                frequency_hz: 400.0,
                ..BellParams::default()
            },
        );
        let out_low = render(&mut low, SR as usize / 4);
        let out_high = render(&mut high, SR as usize / 4);
        assert!(out_low != out_high, "different primes should differ");
        assert!(goertzel(&out_low, 150.0) > goertzel(&out_low, 400.0));
        assert!(goertzel(&out_high, 400.0) > goertzel(&out_high, 150.0));
    }

    #[test]
    fn retriggers_while_ringing() {
        let mut node = BellNode::new(SR, BellParams::default());
        let _ = render(&mut node, SR as usize);
        let decayed = render(&mut node, SR as usize / 100);
        node.strike(1.0);
        let restruck = render(&mut node, SR as usize / 100);
        assert!(
            peak(&restruck) > peak(&decayed) * 2.0,
            "re-strike should re-energise: decayed={} restruck={}",
            peak(&decayed),
            peak(&restruck)
        );
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = BellNode::new(SR, BellParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = BellParams {
            frequency_hz: 330.0,
            decay_s: 5.0,
            brightness: 0.3,
            warble: 0.7,
            amplitude: 0.7,
        };
        let node = BellNode::new(SR, params);
        assert!((node.frequency_hz() - 330.0).abs() < 1.0e-3);
        assert!((node.decay_s() - 5.0).abs() < 1.0e-3);
        assert!((node.brightness() - 0.3).abs() < 1.0e-3);
        assert!((node.warble() - 0.7).abs() < 1.0e-3);
        assert!((node.amplitude() - 0.7).abs() < 1.0e-3);
    }

    #[test]
    fn frequency_is_clamped() {
        let mut node = BellNode::new(SR, BellParams::default());
        node.set_frequency(-100.0);
        assert!(node.frequency_hz() >= MIN_FREQUENCY_HZ);
        node.set_frequency(1.0e9);
        assert!(node.frequency_hz() <= MAX_FREQUENCY_HZ);
    }

    #[test]
    fn decay_is_clamped() {
        let mut node = BellNode::new(SR, BellParams::default());
        node.set_decay(-1.0);
        assert!((node.decay_s() - MIN_DECAY_S).abs() < 1.0e-6);
        node.set_decay(1.0e6);
        assert!((node.decay_s() - MAX_DECAY_S).abs() < 1.0e-6);
    }

    #[test]
    fn brightness_is_clamped() {
        let mut node = BellNode::new(SR, BellParams::default());
        node.set_brightness(-1.0);
        assert!((node.brightness() - 0.0).abs() < 1.0e-6);
        node.set_brightness(5.0);
        assert!((node.brightness() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn warble_is_clamped() {
        let mut node = BellNode::new(SR, BellParams::default());
        node.set_warble(-1.0);
        assert!((node.warble() - MIN_WARBLE).abs() < 1.0e-6);
        node.set_warble(5.0);
        assert!((node.warble() - MAX_WARBLE).abs() < 1.0e-6);
    }

    #[test]
    fn setters_reject_non_finite_and_keep_previous() {
        let mut node = BellNode::new(SR, BellParams::default());
        let (f, d, b, w) = (
            node.frequency_hz(),
            node.decay_s(),
            node.brightness(),
            node.warble(),
        );
        node.set_frequency(Sample::NAN);
        node.set_decay(Sample::INFINITY);
        node.set_brightness(Sample::NAN);
        node.set_warble(Sample::NEG_INFINITY);
        assert!((node.frequency_hz() - f).abs() < 1.0e-6);
        assert!((node.decay_s() - d).abs() < 1.0e-6);
        assert!((node.brightness() - b).abs() < 1.0e-6);
        assert!((node.warble() - w).abs() < 1.0e-6);
    }

    #[test]
    fn constructor_sanitises_non_finite() {
        let params = BellParams {
            frequency_hz: Sample::NAN,
            decay_s: Sample::INFINITY,
            brightness: Sample::NAN,
            warble: Sample::NEG_INFINITY,
            amplitude: Sample::NAN,
        };
        let mut node = BellNode::new(SR, params);
        let out = render(&mut node, SR as usize / 10);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!((node.frequency_hz() - DEFAULT_FREQUENCY_HZ).abs() < 1.0e-3);
        assert!((node.warble() - DEFAULT_WARBLE).abs() < 1.0e-3);
    }

    #[test]
    fn mono_core_copies_to_all_channels() {
        let mut node = BellNode::new(SR, BellParams::default());
        let chans = render_layout(&mut node, SR as usize / 10, ChannelLayout::Quad);
        assert_eq!(chans.len(), 4);
        for ch in 1..chans.len() {
            assert_eq!(chans[0], chans[ch]);
        }
    }

    #[test]
    fn high_pitch_mutes_supersonic_modes() {
        let node = BellNode::new(
            SR,
            BellParams {
                frequency_hz: MAX_FREQUENCY_HZ,
                ..BellParams::default()
            },
        );
        // The hum (half the prime) stays audible, but the top shell modes land
        // above the Nyquist guard and must be muted.
        assert!(node.mode_enabled(0));
        assert!(!node.mode_enabled(NUM_MODES - 1));
    }

    #[test]
    fn latency_is_zero() {
        let node = BellNode::new(SR, BellParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
