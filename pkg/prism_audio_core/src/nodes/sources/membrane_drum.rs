//! Struck circular-membrane modal percussion source (timpani / tom / tabla /
//! frame-drum family).
//!
//! [`MembraneDrumNode`] is a *source* (zero inputs, one output) that synthesizes
//! a mallet- or stick-struck drumhead by *modal synthesis*: a short contact
//! force excites a parallel bank of [`NUM_MODES`] independently decaying two-pole
//! resonators, each tuned to one vibration mode of an ideal circular membrane.
//! The summed output is the characteristic drum "thump" or pitched timpani note
//! that rings out and dies away after each strike.
//!
//! # Model
//!
//! An ideal circular membrane under uniform tension does not vibrate at a
//! harmonic series: its modes sit at the ratios of the zeros of the Bessel
//! functions. A mode is indexed by an azimuthal order `m` (number of nodal
//! diameters) and a radial order `n` (number of nodal circles); its frequency
//! is proportional to the `n`-th positive zero of the order-`m` Bessel function
//! `J_m`. Relative to the fundamental `(0, 1)` mode the first several ratios are
//! `1 : 1.593 : 2.136 : 2.296 : 2.653 : ...`, a strongly inharmonic set that
//! gives an untuned tom or frame drum its noisy, pitchless "thud". A real
//! kettledrum (timpani) is loaded by the enclosed air and radiation, which pulls
//! the principal modes toward a near-harmonic set `1 : 1.5 : 2 : 2.5 : ...` so
//! the instrument sounds a definite pitch. The `inharmonicity` control linearly
//! blends the mode ratios between that tuned kettledrum set and the ideal
//! membrane set, so one node spans the pitched-to-noisy drum continuum.
//!
//! Each mode `m` is a two-pole resonator
//! `y[n] = b0 * x[n] + a1 * y[n-1] + a2 * y[n-2]` whose complex pole pair sits
//! at radius `R = exp(-ln(1000) / (t60 * sample_rate))` and angle
//! `theta = 2*pi*f_m / sample_rate`, giving `a1 = 2*R*cos(theta)` and
//! `a2 = -R*R`. Its impulse response is a sinusoid at `f_m` decaying by
//! `-60 dB` over `t60` seconds. The feed gain `b0 = gain_m * sin(theta)`
//! normalises the ringing peak to `gain_m` independently of the decay radius.
//! Higher modes are given shorter decay (`t60_m = decay / ratio_m^0.7`) and
//! lower gain (`gain_m = ratio_m^-exp`, with `exp` set by `brightness`), the
//! usual spectral envelope of a struck head.
//!
//! # Strike position
//!
//! Where the head is struck decides which modes are driven. A mode with
//! azimuthal order `m` and radial zero `alpha` has, at a fractional radius `r`
//! from the centre, a transverse displacement proportional to `J_m(alpha * r)`.
//! At the exact centre (`r = 0`) only the axisymmetric `m = 0` modes have any
//! amplitude (`J_0(0) = 1`, `J_{m>0}(0) = 0`), so a dead-centre strike excites
//! only the deep concentric modes and produces a dark, hollow "boom"; striking
//! nearer the rim brings the higher nodal-diameter modes to life for a brighter
//! "slap". The `strike_position` control sets `r`, and each mode's excitation
//! gain is scaled by `|J_m(alpha * r)|`, so the node reproduces this physical
//! centre-to-rim timbral sweep.
//!
//! The excitation `x[n]` is a single raised-cosine (Hann) contact-force pulse,
//! normalised to unit area so each strike imparts a fixed momentum regardless of
//! its width. A hard stick is modelled by a short pulse (bright, lots of
//! high-mode energy); a soft mallet by a long pulse (dull, high modes barely
//! driven). `brightness` sets both the pulse width and the mode-gain rolloff.
//! The node is struck once at construction so it sounds immediately;
//! [`MembraneDrumNode::strike`] retriggers it, adding a fresh pulse while the
//! existing modes keep ringing.
//!
//! # Determinism
//!
//! The excitation is a closed-form deterministic pulse, not noise, so the node
//! holds no random state: two [`MembraneDrumNode`]s built with the same sample
//! rate and parameters produce bit-identical output, and
//! [`MembraneDrumNode::reset`] clears the resonators and re-strikes to replay the
//! identical attack.
//!
//! # Real-time contract
//!
//! All per-mode coefficient and history storage is a fixed-size array sized for
//! [`NUM_MODES`]; [`MembraneDrumNode::process`] performs no allocation, locking,
//! or panic on the hot path. The Bessel mode-shape weights are evaluated only in
//! the cold `recompute` path, never per sample. Non-finite parameters are
//! sanitised on the way in and outputs are flushed of denormals, so the
//! generator cannot stall the audio thread. Latency is zero.
//!
//! # Relationship
//!
//! Unlike the sibling [`modal_resonator`](crate::nodes::effects::modal_resonator)
//! *effect*, which filters an *external* input signal through a modal bank, this
//! *source* supplies its own strike excitation and needs no input. It is the
//! two-dimensional counterpart of the one-dimensional
//! [`struck_bar`](crate::nodes::sources::struck_bar): where that node uses the
//! Euler-Bernoulli bending-mode ratios of a stiff bar, this node uses the
//! Bessel-zero ratios of a circular membrane and adds a strike-position control
//! that no one-dimensional voice has. It differs from the (near-)harmonic
//! waveguide voices [`karplus_strong`](crate::nodes::sources::karplus_strong),
//! [`bowed_string`](crate::nodes::sources::bowed_string), and
//! [`reed_woodwind`](crate::nodes::sources::reed_woodwind), which model strings
//! and air columns, and from [`noise`](crate::nodes::sources::noise), whose
//! drum-like transients carry no resonant modal pitch at all.
//!
//! # Provenance
//!
//! Modal synthesis (an object modelled as a parallel bank of independently
//! decaying resonators) is the classic technique described by J.-M. Adrien,
//! "The Missing Link: Modal Synthesis" (in *Representations of Musical Signals*,
//! MIT Press, 1991). The circular-membrane mode frequencies as ratios of Bessel
//! function zeros, and the kettledrum's air-loaded near-harmonic tuning, are the
//! standard results tabulated in acoustics texts (for example N. H. Fletcher and
//! T. D. Rossing, *The Physics of Musical Instruments*). The `J_m(alpha * r)`
//! mode shape, the ascending-series Bessel evaluation, the two-pole resonator,
//! the `t60`-to-pole-radius mapping, and the Hann (raised-cosine) contact pulse
//! are standard, publicly documented mathematics and DSP. This is pure classic
//! DSP with no AI or ML. This module contains **no Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Google Resonance Audio, Web Audio, or STK source or
//! derived code**; only the widely documented membrane-mode ratios, Bessel
//! series, resonator, and window formulas are used.


use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Number of parallel membrane modes the drumhead is modelled with.
pub const NUM_MODES: usize = 8;

/// Lowest tunable fundamental (strike pitch) in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Highest tunable fundamental in hertz (further bounded by the Nyquist limit).
pub const MAX_FREQUENCY_HZ: Sample = 4_000.0;

/// Default fundamental (strike pitch) frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 150.0;

/// Shortest `-60 dB` decay time, in seconds, the fundamental may request.
pub const MIN_DECAY_S: Sample = 0.02;

/// Longest `-60 dB` decay time, in seconds, the fundamental may request.
pub const MAX_DECAY_S: Sample = 20.0;

/// Default fundamental `-60 dB` decay time in seconds.
pub const DEFAULT_DECAY_S: Sample = 0.8;

/// Default stick / mallet hardness (brightness) in `[0, 1]`.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Default inharmonicity in `[0, 1]` (`0` tuned kettledrum, `1` ideal membrane).
pub const DEFAULT_INHARMONICITY: Sample = 1.0;

/// Smallest strike position (dead centre) in `[0, 1]` fractional radius.
pub const MIN_STRIKE_POSITION: Sample = 0.0;

/// Largest strike position (near the rim) in `[0, 1]` fractional radius.
pub const MAX_STRIKE_POSITION: Sample = 0.95;

/// Default strike position (an off-centre playing spot) in fractional radius.
pub const DEFAULT_STRIKE_POSITION: Sample = 0.35;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Default strike velocity used by [`MembraneDrumNode::strike`].
pub const DEFAULT_STRIKE_VELOCITY: Sample = 1.0;

/// Azimuthal order `m` (number of nodal diameters) of each modelled mode.
const MODE_AZIMUTHAL: [usize; NUM_MODES] = [0, 1, 2, 0, 3, 1, 4, 2];

/// Positive Bessel-function zeros `alpha_{m,n}` for each modelled mode, ordered
/// by ascending frequency: `(0,1) (1,1) (2,1) (0,2) (3,1) (1,2) (4,1) (2,2)`.
const BESSEL_ZEROS: [Sample; NUM_MODES] = [
    2.404_826, 3.831_706, 5.135_622, 5.520_078, 6.380_162, 7.015_587, 7.588_342,
    8.417_244,
];

/// Air-loaded (near-harmonic) kettledrum mode ratios for the same modes.
const TUNED_DRUM_RATIOS: [Sample; NUM_MODES] =
    [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];

/// `ln(1000) == 3 * ln(10)`, used by the `t60`-to-pole-radius mapping.
const LN_1000: Sample = 6.907_755;

/// Fraction of the sample rate above which a mode is muted (anti-alias guard).
const NYQUIST_GUARD: Sample = 0.49;

/// Exponent controlling how much faster high modes decay than the fundamental.
const DECAY_RATIO_EXP: Sample = 0.7;

/// Shortest contact pulse (hardest stick), in milliseconds.
const PULSE_MS_MIN: Sample = 0.2;

/// Longest contact pulse (softest mallet), in milliseconds.
const PULSE_MS_MAX: Sample = 4.0;

/// Softest mode-gain rolloff exponent (brightest stick).
const GAIN_EXP_MIN: Sample = 0.5;

/// Steepest mode-gain rolloff exponent (dullest mallet).
const GAIN_EXP_MAX: Sample = 2.0;

/// Overall output scale, keeping the summed modal peak below full scale.
const OUTPUT_GAIN: Sample = 0.5;

/// Maximum number of ascending-series terms evaluated for a Bessel value.
const BESSEL_SERIES_TERMS: usize = 40;

/// Early-exit magnitude below which further Bessel series terms are negligible.
const BESSEL_SERIES_EPS: Sample = 1.0e-9;

/// Evaluates the order-`order` Bessel function of the first kind `J_order(x)`
/// by its ascending power series. Used only in the cold `recompute` path to
/// weight each mode by its strike-position mode shape, never on the hot path.
///
/// The series is `sum_{k>=0} (-1)^k / (k! (order + k)!) * (x/2)^(2k + order)`,
/// evaluated by the stable term recurrence
/// `t_{k+1} = t_k * (-(x/2)^2) / ((k+1)(order+k+1))` so no factorials or large
/// powers are formed directly.
#[inline]
fn bessel_j(order: usize, x: Sample) -> Sample {
    let half = x * 0.5;
    // term_0 = (x/2)^order / order! == product_{k=1}^{order} (half / k).
    let mut term = 1.0;
    for k in 1..=order {
        term *= half / k as Sample;
    }
    let mut sum = term;
    let neg = -(half * half);
    for k in 0..BESSEL_SERIES_TERMS {
        term *= neg / (((k + 1) * (order + k + 1)) as Sample);
        sum += term;
        if ops::abs(term) < BESSEL_SERIES_EPS {
            break;
        }
    }
    sum
}

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

/// Construction parameters for a [`MembraneDrumNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MembraneDrumParams {
    /// Fundamental (strike pitch) frequency in hertz.
    pub frequency_hz: Sample,
    /// Fundamental `-60 dB` decay time in seconds (longer rings longer).
    pub decay_s: Sample,
    /// Stick / mallet hardness (brightness) in `[0, 1]` (`1` is a hard stick).
    pub brightness: Sample,
    /// Inharmonicity in `[0, 1]` (`0` tuned kettledrum, `1` ideal membrane).
    pub inharmonicity: Sample,
    /// Strike position in `[0, 1]` fractional radius (`0` centre, `1` rim).
    pub strike_position: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for MembraneDrumParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            decay_s: DEFAULT_DECAY_S,
            brightness: DEFAULT_BRIGHTNESS,
            inharmonicity: DEFAULT_INHARMONICITY,
            strike_position: DEFAULT_STRIKE_POSITION,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl MembraneDrumParams {
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
        let strike_position = finite_or(self.strike_position, d.strike_position)
            .clamp(MIN_STRIKE_POSITION, MAX_STRIKE_POSITION);
        let amplitude = finite_or(self.amplitude, d.amplitude);
        Self {
            frequency_hz,
            decay_s,
            brightness,
            inharmonicity,
            strike_position,
            amplitude,
        }
    }
}

/// Struck circular-membrane inharmonic modal synthesis source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{MembraneDrumNode, MembraneDrumParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = MembraneDrumNode::new(48_000, MembraneDrumParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The strike excites the membrane modes, which ring and decay.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
#[derive(Clone, Debug)]
pub struct MembraneDrumNode {
    sample_rate: u32,
    frequency_hz: Sample,
    decay_s: Sample,
    brightness: Sample,
    inharmonicity: Sample,
    strike_position: Sample,
    amplitude: Smoothed,
    // Per-mode resonator coefficients and ringing history.
    a1: [Sample; NUM_MODES],
    a2: [Sample; NUM_MODES],
    b0: [Sample; NUM_MODES],
    enabled: [bool; NUM_MODES],
    y1: [Sample; NUM_MODES],
    y2: [Sample; NUM_MODES],
    // Contact-pulse state.
    pulse_len: u32,
    pulse_pos: u32,
    pulse_scale: Sample,
    velocity: Sample,
}

impl MembraneDrumNode {
    /// Builds a drumhead at `sample_rate` from (sanitised) `params`, struck once
    /// so it sounds immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: MembraneDrumParams) -> Self {
        let sr = sample_rate.max(1);
        let p = params.sanitised(sr);
        let mut node = Self {
            sample_rate: sr,
            frequency_hz: p.frequency_hz,
            decay_s: p.decay_s,
            brightness: p.brightness,
            inharmonicity: p.inharmonicity,
            strike_position: p.strike_position,
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

    /// Returns the stick / mallet hardness (brightness) in `[0, 1]`.
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
    }

    /// Returns the inharmonicity in `[0, 1]`.
    #[must_use]
    pub fn inharmonicity(&self) -> Sample {
        self.inharmonicity
    }

    /// Returns the strike position in `[0, 1]` fractional radius.
    #[must_use]
    pub fn strike_position(&self) -> Sample {
        self.strike_position
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

    /// Sets the stick / mallet hardness (brightness), clamped to `[0, 1]`.
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the inharmonicity, clamped to `[0, 1]`.
    pub fn set_inharmonicity(&mut self, inharmonicity: Sample) {
        self.inharmonicity = finite_or(inharmonicity, self.inharmonicity).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the strike position, clamped to `[MIN_STRIKE_POSITION,
    /// MAX_STRIKE_POSITION]` fractional radius.
    pub fn set_strike_position(&mut self, strike_position: Sample) {
        self.strike_position = finite_or(strike_position, self.strike_position)
            .clamp(MIN_STRIKE_POSITION, MAX_STRIKE_POSITION);
        self.recompute();
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Retriggers the head with the given strike `velocity` (clamped to
    /// `[0, 1]`), injecting a fresh contact pulse while existing modes keep
    /// ringing.
    pub fn strike(&mut self, velocity: Sample) {
        self.velocity = finite_or(velocity, DEFAULT_STRIKE_VELOCITY).clamp(0.0, 1.0);
        self.pulse_pos = 0;
    }

    /// Recomputes every mode coefficient and the contact-pulse geometry from the
    /// current scalar parameters. Never runs on the audio hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        let inh = self.inharmonicity;
        let gain_exp = GAIN_EXP_MIN + (1.0 - self.brightness) * (GAIN_EXP_MAX - GAIN_EXP_MIN);
        let nyquist = sr * NYQUIST_GUARD;
        let r = self.strike_position;
        let fundamental_zero = BESSEL_ZEROS[0];

        for m in 0..NUM_MODES {
            let ideal_ratio = BESSEL_ZEROS[m] / fundamental_zero;
            let ratio = (1.0 - inh) * TUNED_DRUM_RATIOS[m] + inh * ideal_ratio;
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
            // Strike-position mode shape: displacement of mode (m, n) at radius r.
            let shape = ops::abs(bessel_j(MODE_AZIMUTHAL[m], BESSEL_ZEROS[m] * r));
            let gain = ops::powf(ratio, -gain_exp) * shape;
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

    /// Renders one mono output sample, advancing every resonator and the contact
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

impl AudioNode for MembraneDrumNode {
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
    fn render(node: &mut MembraneDrumNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(
        node: &mut MembraneDrumNode,
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
    fn bessel_j_matches_known_values() {
        assert!((bessel_j(0, 0.0) - 1.0).abs() < 1.0e-6);
        assert!(bessel_j(1, 0.0).abs() < 1.0e-6);
        assert!(bessel_j(2, 0.0).abs() < 1.0e-6);
        // J_0 and J_1 vanish at their first zeros.
        assert!(bessel_j(0, 2.404_826).abs() < 1.0e-3);
        assert!(bessel_j(1, 3.831_706).abs() < 1.0e-3);
        assert!(bessel_j(2, 5.135_622).abs() < 1.0e-3);
    }

    #[test]
    fn default_strike_produces_sound() {
        let mut node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        let out = render(&mut node, SR as usize / 10);
        let p = peak(&out);
        assert!(p > 1.0e-3, "membrane should ring, peak = {p}");
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn tone_decays_toward_silence() {
        let mut node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        let out = render(&mut node, 4 * SR as usize);
        let head = energy(&out[..SR as usize / 10]);
        let tail = energy(&out[out.len() - SR as usize / 10..]);
        assert!(head > 0.0);
        assert!(
            tail < head * 1.0e-2,
            "struck head should decay: head = {head}, tail = {tail}"
        );
    }

    #[test]
    fn long_run_stays_bounded_and_finite() {
        let params = MembraneDrumParams {
            decay_s: 6.0,
            amplitude: 1.0,
            strike_position: 0.3,
            ..MembraneDrumParams::default()
        };
        let mut node = MembraneDrumNode::new(SR, params);
        let out = render(&mut node, SR as usize * 4);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(peak(&out) < 1.0, "peak must stay below full scale");
    }

    #[test]
    fn deterministic_across_instances() {
        let p = MembraneDrumParams::default();
        let mut a = MembraneDrumNode::new(SR, p);
        let mut b = MembraneDrumNode::new(SR, p);
        let oa = render(&mut a, 4_096);
        let ob = render(&mut b, 4_096);
        assert_eq!(oa, ob);
    }

    #[test]
    fn reset_restarts_identical_attack() {
        let mut node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        let first = render(&mut node, 4_096);
        node.reset();
        let second = render(&mut node, 4_096);
        assert_eq!(first, second);
    }

    #[test]
    fn amplitude_scales_output_energy() {
        let mut quiet = MembraneDrumNode::new(
            SR,
            MembraneDrumParams {
                amplitude: 0.25,
                ..MembraneDrumParams::default()
            },
        );
        let mut loud = MembraneDrumNode::new(
            SR,
            MembraneDrumParams {
                amplitude: 0.5,
                ..MembraneDrumParams::default()
            },
        );
        let eq = energy(&render(&mut quiet, SR as usize / 4));
        let el = energy(&render(&mut loud, SR as usize / 4));
        // Doubling amplitude quadruples energy.
        assert!((el / eq - 4.0).abs() < 0.2, "ratio = {}", el / eq);
    }

    #[test]
    fn inharmonic_partials_present() {
        // The ideal membrane has a strong partial at ratio 1.593 that is absent
        // from any harmonic series; it must dominate the non-mode point at 2.0.
        let f0 = 150.0;
        let mut node = MembraneDrumNode::new(
            SR,
            MembraneDrumParams {
                frequency_hz: f0,
                inharmonicity: 1.0,
                strike_position: 0.5,
                ..MembraneDrumParams::default()
            },
        );
        let out = render(&mut node, SR as usize / 2);
        let membrane = goertzel(&out, f0 * 1.593);
        let between = goertzel(&out, f0 * 2.0);
        assert!(
            membrane > between * 4.0,
            "inharmonic membrane partial should dominate: 1.593 = {membrane}, 2.0 = {between}"
        );
    }

    #[test]
    fn fundamental_is_strongest_low_partial() {
        let f0 = 150.0;
        let mut node = MembraneDrumNode::new(
            SR,
            MembraneDrumParams {
                frequency_hz: f0,
                strike_position: 0.5,
                ..MembraneDrumParams::default()
            },
        );
        let out = render(&mut node, SR as usize / 2);
        assert!(goertzel(&out, f0) > goertzel(&out, f0 * 1.593));
    }

    #[test]
    fn centre_strike_excites_only_axisymmetric_modes() {
        // A dead-centre strike silences the m > 0 modes (J_{m>0}(0) == 0) while
        // the m == 0 fundamental still rings.
        let f0 = 150.0;
        let mut node = MembraneDrumNode::new(
            SR,
            MembraneDrumParams {
                frequency_hz: f0,
                inharmonicity: 1.0,
                strike_position: 0.0,
                ..MembraneDrumParams::default()
            },
        );
        let out = render(&mut node, SR as usize / 2);
        let fundamental = goertzel(&out, f0);
        let diameter_mode = goertzel(&out, f0 * 1.593);
        assert!(fundamental > 1.0, "centre strike should still boom");
        assert!(
            diameter_mode < fundamental * 1.0e-2,
            "centre strike must mute the (1,1) mode: fund = {fundamental}, mode = {diameter_mode}"
        );
    }

    #[test]
    fn edge_strike_is_brighter_than_centre() {
        // The (2,1) mode is silent at the centre but excited near the rim.
        let f0 = 150.0;
        let mk = |pos: Sample| {
            MembraneDrumNode::new(
                SR,
                MembraneDrumParams {
                    frequency_hz: f0,
                    inharmonicity: 1.0,
                    strike_position: pos,
                    ..MembraneDrumParams::default()
                },
            )
        };
        let mut centre = mk(0.0);
        let mut edge = mk(0.85);
        // `hc` is leakage from the loud (0,2) mode into this bin, since the (2,1)
        // mode itself is silent at the centre; the edge strike adds genuine (2,1)
        // energy, so it clears that leakage floor by a comfortable margin.
        let hc = goertzel(&render(&mut centre, SR as usize / 2), f0 * 2.136);
        let he = goertzel(&render(&mut edge, SR as usize / 2), f0 * 2.136);
        assert!(he > hc * 2.0, "edge strike should be brighter: centre = {hc}, edge = {he}");
    }

    #[test]
    fn strike_position_changes_output() {
        let base = MembraneDrumParams {
            strike_position: 0.2,
            ..MembraneDrumParams::default()
        };
        let mut a = MembraneDrumNode::new(SR, base);
        let mut b = MembraneDrumNode::new(
            SR,
            MembraneDrumParams {
                strike_position: 0.8,
                ..base
            },
        );
        assert_ne!(render(&mut a, 4_096), render(&mut b, 4_096));
    }

    #[test]
    fn softer_mallet_is_duller() {
        let f0 = 150.0;
        let mk = |brightness: Sample| {
            MembraneDrumNode::new(
                SR,
                MembraneDrumParams {
                    frequency_hz: f0,
                    brightness,
                    strike_position: 0.6,
                    ..MembraneDrumParams::default()
                },
            )
        };
        let mut hard = mk(1.0);
        let mut soft = mk(0.0);
        let hard_out = render(&mut hard, SR as usize / 2);
        let soft_out = render(&mut soft, SR as usize / 2);
        let ratio_hard = goertzel(&hard_out, f0 * 2.136) / goertzel(&hard_out, f0);
        let ratio_soft = goertzel(&soft_out, f0 * 2.136) / goertzel(&soft_out, f0);
        assert!(
            ratio_hard > ratio_soft,
            "hard stick should be brighter: hard = {ratio_hard}, soft = {ratio_soft}"
        );
    }

    #[test]
    fn strike_retriggers_envelope() {
        let mut node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        let _ = render(&mut node, SR as usize);
        let quiet = render(&mut node, 2_048);
        node.strike(1.0);
        let loud = render(&mut node, 2_048);
        assert!(
            energy(&loud) > energy(&quiet) * 4.0,
            "a fresh strike should re-energise the head"
        );
    }

    #[test]
    fn mono_core_replicated_to_all_channels() {
        let mut node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        let chans = render_layout(&mut node, 2_048, ChannelLayout::Quad);
        assert_eq!(chans.len(), 4);
        for ch in 1..chans.len() {
            assert_eq!(chans[0], chans[ch]);
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = MembraneDrumParams {
            frequency_hz: 180.0,
            decay_s: 1.5,
            brightness: 0.3,
            inharmonicity: 0.4,
            strike_position: 0.6,
            amplitude: 0.7,
        };
        let node = MembraneDrumNode::new(SR, params);
        assert_eq!(node.frequency_hz(), 180.0);
        assert_eq!(node.decay_s(), 1.5);
        assert_eq!(node.brightness(), 0.3);
        assert_eq!(node.inharmonicity(), 0.4);
        assert_eq!(node.strike_position(), 0.6);
        assert_eq!(node.amplitude(), 0.7);
    }

    #[test]
    fn set_frequency_clamps_to_range() {
        let mut node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        node.set_frequency(1.0e9);
        assert!(node.frequency_hz() <= MAX_FREQUENCY_HZ);
        node.set_frequency(-5.0);
        assert_eq!(node.frequency_hz(), MIN_FREQUENCY_HZ);
    }

    #[test]
    fn set_decay_clamps_to_range() {
        let mut node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        node.set_decay(1.0e6);
        assert_eq!(node.decay_s(), MAX_DECAY_S);
        node.set_decay(0.0);
        assert_eq!(node.decay_s(), MIN_DECAY_S);
    }

    #[test]
    fn set_brightness_inharmonicity_and_position_clamp() {
        let mut node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        node.set_brightness(5.0);
        assert_eq!(node.brightness(), 1.0);
        node.set_inharmonicity(-5.0);
        assert_eq!(node.inharmonicity(), 0.0);
        node.set_strike_position(5.0);
        assert_eq!(node.strike_position(), MAX_STRIKE_POSITION);
        node.set_strike_position(-5.0);
        assert_eq!(node.strike_position(), MIN_STRIKE_POSITION);
    }

    #[test]
    fn setters_reject_non_finite() {
        let mut node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        let f = node.frequency_hz();
        node.set_frequency(Sample::NAN);
        assert_eq!(node.frequency_hz(), f);
        let d = node.decay_s();
        node.set_decay(Sample::INFINITY);
        assert_eq!(node.decay_s(), d);
        let p = node.strike_position();
        node.set_strike_position(Sample::NAN);
        assert_eq!(node.strike_position(), p);
    }

    #[test]
    fn constructor_sanitizes_non_finite_params() {
        let params = MembraneDrumParams {
            frequency_hz: Sample::NAN,
            decay_s: Sample::INFINITY,
            brightness: Sample::NAN,
            inharmonicity: Sample::NAN,
            strike_position: Sample::NAN,
            amplitude: Sample::NAN,
        };
        let node = MembraneDrumNode::new(SR, params);
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.decay_s(), DEFAULT_DECAY_S);
        assert_eq!(node.brightness(), DEFAULT_BRIGHTNESS);
        assert_eq!(node.inharmonicity(), DEFAULT_INHARMONICITY);
        assert_eq!(node.strike_position(), DEFAULT_STRIKE_POSITION);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
        let mut node = node;
        assert!(render(&mut node, 512).iter().all(|s| s.is_finite()));
    }

    #[test]
    fn frequency_changes_output() {
        let mut low = MembraneDrumNode::new(
            SR,
            MembraneDrumParams {
                frequency_hz: 100.0,
                ..MembraneDrumParams::default()
            },
        );
        let mut high = MembraneDrumNode::new(
            SR,
            MembraneDrumParams {
                frequency_hz: 300.0,
                ..MembraneDrumParams::default()
            },
        );
        assert_ne!(render(&mut low, 4_096), render(&mut high, 4_096));
    }

    #[test]
    fn inharmonicity_changes_output() {
        let mut tuned = MembraneDrumNode::new(
            SR,
            MembraneDrumParams {
                inharmonicity: 0.0,
                strike_position: 0.5,
                ..MembraneDrumParams::default()
            },
        );
        let mut ideal = MembraneDrumNode::new(
            SR,
            MembraneDrumParams {
                inharmonicity: 1.0,
                strike_position: 0.5,
                ..MembraneDrumParams::default()
            },
        );
        assert_ne!(render(&mut tuned, 4_096), render(&mut ideal, 4_096));
    }

    #[test]
    fn high_and_low_frequencies_both_sound() {
        for f in [40.0, 2_000.0] {
            let mut node = MembraneDrumNode::new(
                SR,
                MembraneDrumParams {
                    frequency_hz: f,
                    ..MembraneDrumParams::default()
                },
            );
            let out = render(&mut node, SR as usize / 10);
            assert!(peak(&out) > 1.0e-3, "frequency {f} should sound");
            assert!(out.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn amplitude_target_tracks_setter() {
        let mut node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        node.set_amplitude(0.9, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.9);
    }

    #[test]
    fn latency_is_zero() {
        let node = MembraneDrumNode::new(SR, MembraneDrumParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
