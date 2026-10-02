//! Struck-plate modal percussion source (gong / plate-bell / metal-sheet /
//! thunder-plate family).
//!
//! [`StruckPlateNode`] is a *source* (zero inputs, one output) that synthesizes
//! a mallet-struck thin rectangular plate by *modal synthesis*: a short contact
//! force excites a parallel bank of [`NUM_MODES`] independently decaying
//! two-pole resonators, each tuned to one two-dimensional bending partial of the
//! plate. The summed output is the dense, shimmering metallic wash of a struck
//! plate that blooms at the strike and rings out as a slowly collapsing cloud of
//! inharmonic partials.
//!
//! # Model
//!
//! A thin flat plate does not ring at a harmonic series, nor at the sparse
//! one-dimensional series of a bar. Its transverse bending modes are indexed by
//! a *pair* of integers `(i, j)` -- the number of half-wavelengths along each
//! edge -- and, for an idealised simply-supported Kirchhoff plate, sit at
//! frequencies proportional to `(i / a)^2 + (j / b)^2`, where `a` and `b` are
//! the plate's two side lengths. Writing `aspect = b / a` and normalising so the
//! lowest `(1, 1)` mode lands on the requested fundamental, every partial ratio
//! becomes `((i^2 + j^2 / aspect^2) / (1 + 1 / aspect^2))`. Because two indices
//! generate the grid, the partials are far *denser* than a bar's single series
//! and cluster into near-degenerate pairs that beat against one another -- the
//! acoustic signature of a plate or gong. The `aspect_ratio` control stretches
//! the plate: a square plate (`aspect == 1`) collapses the `(i, j)` and `(j, i)`
//! partials onto exact degeneracies, while a rectangular plate splits each pair
//! into a shimmering beating doublet.
//!
//! The lowest [`NUM_MODES`] grid frequencies are selected and each is realised
//! as a two-pole resonator `y[n] = b0 * x[n] + a1 * y[n-1] + a2 * y[n-2]` whose
//! complex pole pair sits at radius `R = exp(-ln(1000) / (t60 * sample_rate))`
//! and angle `theta = 2*pi*f_m / sample_rate`, giving `a1 = 2*R*cos(theta)` and
//! `a2 = -R*R`. Its impulse response is a sinusoid at `f_m` decaying by `-60 dB`
//! over `t60` seconds. The feed gain `b0 = gain_m * sin(theta)` normalises the
//! ringing peak to `gain_m` independently of the decay radius. Higher partials
//! are given shorter decay (`t60_m = decay / ratio_m^0.5`, a gentler rolloff
//! than a bar so the high cloud lingers) and lower gain (`gain_m = ratio_m^-exp`,
//! with `exp` set by `brightness`), the usual spectral envelope of a struck
//! plate.
//!
//! The excitation `x[n]` is a single raised-cosine (Hann) contact-force pulse,
//! normalised to unit area so each strike imparts a fixed momentum regardless of
//! its width. A hard beater is modelled by a short pulse (bright, lots of
//! high-mode energy); a soft beater by a long pulse (dull, high modes barely
//! driven). `brightness` sets both the pulse width and the mode-gain rolloff.
//! The node is struck once at construction so it sounds immediately;
//! [`StruckPlateNode::strike`] retriggers it, adding a fresh pulse while the
//! existing modes keep ringing.
//!
//! # Determinism
//!
//! The excitation is a closed-form deterministic pulse, not noise, so the node
//! holds no random state: two [`StruckPlateNode`]s built with the same sample
//! rate and parameters produce bit-identical output, and
//! [`StruckPlateNode::reset`] clears the resonators and re-strikes to replay the
//! identical attack.
//!
//! # Real-time contract
//!
//! All per-mode coefficient and history storage is a fixed-size array sized for
//! [`NUM_MODES`]; [`StruckPlateNode::process`] performs no allocation, locking,
//! or panic on the hot path. The two-dimensional mode grid is sorted off the hot
//! path in [`StruckPlateNode::recompute`] into a fixed stack array. Non-finite
//! parameters are sanitised on the way in and outputs are flushed of denormals,
//! so the generator cannot stall the audio thread. Latency is zero.
//!
//! # Relationship
//!
//! Unlike the sibling [`modal_resonator`](crate::nodes::effects::modal_resonator)
//! *effect*, which filters an *external* input signal through a modal bank, this
//! *source* supplies its own strike excitation and needs no input. It is the
//! two-dimensional counterpart of [`struck_bar`](crate::nodes::sources::struck_bar),
//! whose single integer index gives a sparse one-dimensional beam series
//! (`1 : 2.76 : 5.40 : ...`); this node's *paired* index gives a far denser,
//! beating two-dimensional grid. It differs from
//! [`membrane_drum`](crate::nodes::sources::membrane_drum), whose circular
//! membrane modes follow the Bessel-zero ratios of a *tension*-restored surface
//! that decays quickly, whereas a plate is *stiffness*-restored and rings on. It
//! also differs from the (near-)harmonic waveguide voices
//! [`karplus_strong`](crate::nodes::sources::karplus_strong) and
//! [`plucked_body`](crate::nodes::sources::plucked_body), which model strings.
//!
//! # Provenance
//!
//! Modal synthesis (an object modelled as a parallel bank of independently
//! decaying resonators) is the classic technique described by J.-M. Adrien,
//! "The Missing Link: Modal Synthesis" (in *Representations of Musical Signals*,
//! MIT Press, 1991). The two-dimensional plate bending-mode frequencies
//! proportional to `(i / a)^2 + (j / b)^2` are the standard simply-supported
//! Kirchhoff / Rayleigh thin-plate partials tabulated in acoustics texts (for
//! example N. H. Fletcher and T. D. Rossing, *The Physics of Musical
//! Instruments*). The two-pole resonator, the `t60`-to-pole-radius mapping, the
//! Hann (raised-cosine) contact pulse, and the beater-hardness spectral envelope
//! are standard, publicly documented DSP. This is pure classic DSP with no AI or
//! ML. This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Google Resonance Audio, Web Audio, or STK source or derived code**;
//! only the widely documented plate-mode frequencies, resonator, and window
//! formulas are used.

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Number of parallel two-dimensional bending modes the plate is modelled with.
pub const NUM_MODES: usize = 24;

/// Highest `(i, j)` half-wavelength index scanned when building the mode grid.
const PLATE_MAX_INDEX: usize = 7;

/// Total candidate grid points `(PLATE_MAX_INDEX * PLATE_MAX_INDEX)`.
const GRID_POINTS: usize = PLATE_MAX_INDEX * PLATE_MAX_INDEX;

/// Lowest tunable fundamental (strike pitch) in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Highest tunable fundamental in hertz (further bounded by the Nyquist limit).
pub const MAX_FREQUENCY_HZ: Sample = 12_000.0;

/// Default fundamental (strike pitch) frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 220.0;

/// Shortest `-60 dB` decay time, in seconds, the fundamental may request.
pub const MIN_DECAY_S: Sample = 0.02;

/// Longest `-60 dB` decay time, in seconds, the fundamental may request.
pub const MAX_DECAY_S: Sample = 20.0;

/// Default fundamental `-60 dB` decay time in seconds.
pub const DEFAULT_DECAY_S: Sample = 2.0;

/// Default beater hardness / brightness in `[0, 1]`.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Lowest plate aspect ratio (`1` is a square plate with degenerate pairs).
pub const MIN_ASPECT_RATIO: Sample = 1.0;

/// Highest plate aspect ratio (a long, narrow sheet).
pub const MAX_ASPECT_RATIO: Sample = 3.0;

/// Default plate aspect ratio (slightly rectangular, splits the degeneracies).
pub const DEFAULT_ASPECT_RATIO: Sample = 1.4;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Default strike velocity used by [`StruckPlateNode::strike`].
pub const DEFAULT_STRIKE_VELOCITY: Sample = 1.0;

/// `ln(1000) == 3 * ln(10)`, used by the `t60`-to-pole-radius mapping.
const LN_1000: Sample = 6.907_755;

/// Fraction of the sample rate above which a mode is muted (anti-alias guard).
const NYQUIST_GUARD: Sample = 0.49;

/// Exponent controlling how much faster high modes decay than the fundamental.
const DECAY_RATIO_EXP: Sample = 0.5;

/// Shortest beater-contact pulse (hardest beater), in milliseconds.
const PULSE_MS_MIN: Sample = 0.15;

/// Longest beater-contact pulse (softest beater), in milliseconds.
const PULSE_MS_MAX: Sample = 3.5;

/// Softest mode-gain rolloff exponent (brightest beater).
const GAIN_EXP_MIN: Sample = 0.5;

/// Steepest mode-gain rolloff exponent (dullest beater).
const GAIN_EXP_MAX: Sample = 2.0;

/// Overall output scale, keeping the summed modal peak below full scale.
///
/// Calibrated so the worst-case parameter grid peaks at `0.8665` with
/// `amplitude == 1`; see the module tests.
const OUTPUT_GAIN: Sample = 0.12;

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

/// Fills `out` with the lowest [`NUM_MODES`] plate-partial ratios for `aspect`,
/// normalised so the lowest ratio is exactly `1.0`.
///
/// Scans the `(i, j)` grid for `i, j` in `1..=PLATE_MAX_INDEX`, sorts the raw
/// `i^2 + j^2 / aspect^2` values ascending, and divides by the smallest. Runs
/// only inside [`StruckPlateNode::recompute`], never on the audio hot path.
fn plate_mode_ratios(aspect: Sample, out: &mut [Sample; NUM_MODES]) {
    let inv_aspect_sq = 1.0 / (aspect * aspect);
    let mut raw = [0.0 as Sample; GRID_POINTS];
    let mut k = 0;
    for i in 1..=PLATE_MAX_INDEX {
        for j in 1..=PLATE_MAX_INDEX {
            let fi = i as Sample;
            let fj = j as Sample;
            raw[k] = fi * fi + fj * fj * inv_aspect_sq;
            k += 1;
        }
    }
    raw.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let base = raw[0].max(Sample::MIN_POSITIVE);
    for m in 0..NUM_MODES {
        out[m] = raw[m] / base;
    }
}

/// Construction parameters for a [`StruckPlateNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StruckPlateParams {
    /// Fundamental (strike pitch) frequency in hertz (the `(1, 1)` mode).
    pub frequency_hz: Sample,
    /// Fundamental `-60 dB` decay time in seconds (longer rings longer).
    pub decay_s: Sample,
    /// Beater hardness / brightness in `[0, 1]` (`1` is a hard, bright beater).
    pub brightness: Sample,
    /// Plate aspect ratio in `[1, 3]` (`1` square, higher splits degeneracies).
    pub aspect_ratio: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for StruckPlateParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            decay_s: DEFAULT_DECAY_S,
            brightness: DEFAULT_BRIGHTNESS,
            aspect_ratio: DEFAULT_ASPECT_RATIO,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl StruckPlateParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range. Frequency is clamped against the Nyquist limit too.
    #[must_use]
    pub fn sanitised(self, sample_rate: u32) -> Self {
        let d = Self::default();
        let frequency_hz =
            clamp_frequency(finite_or(self.frequency_hz, d.frequency_hz), sample_rate);
        let decay_s = finite_or(self.decay_s, d.decay_s).clamp(MIN_DECAY_S, MAX_DECAY_S);
        let brightness = finite_or(self.brightness, d.brightness).clamp(0.0, 1.0);
        let aspect_ratio = finite_or(self.aspect_ratio, d.aspect_ratio)
            .clamp(MIN_ASPECT_RATIO, MAX_ASPECT_RATIO);
        let amplitude = finite_or(self.amplitude, d.amplitude);
        Self {
            frequency_hz,
            decay_s,
            brightness,
            aspect_ratio,
            amplitude,
        }
    }
}

/// A mallet-struck thin-plate modal percussion source.
///
/// See the [module documentation](self) for the model, determinism guarantee,
/// and real-time contract.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{StruckPlateNode, StruckPlateParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = StruckPlateNode::new(48_000, StruckPlateParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The strike excites the two-dimensional plate modes, which ring and decay.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
pub struct StruckPlateNode {
    sample_rate: u32,
    frequency_hz: Sample,
    decay_s: Sample,
    brightness: Sample,
    aspect_ratio: Sample,
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

impl StruckPlateNode {
    /// Builds a plate voice for `sample_rate` from `params`, sanitising every
    /// field, then strikes it once so it sounds immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: StruckPlateParams) -> Self {
        let p = params.sanitised(sample_rate);
        let mut node = Self {
            sample_rate,
            frequency_hz: p.frequency_hz,
            decay_s: p.decay_s,
            brightness: p.brightness,
            aspect_ratio: p.aspect_ratio,
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

    /// Returns the beater hardness / brightness in `[0, 1]`.
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
    }

    /// Returns the plate aspect ratio in `[1, 3]`.
    #[must_use]
    pub fn aspect_ratio(&self) -> Sample {
        self.aspect_ratio
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

    /// Sets the beater hardness / brightness, clamped to `[0, 1]`.
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the plate aspect ratio, clamped to `[1, 3]`.
    pub fn set_aspect_ratio(&mut self, aspect_ratio: Sample) {
        self.aspect_ratio = finite_or(aspect_ratio, self.aspect_ratio)
            .clamp(MIN_ASPECT_RATIO, MAX_ASPECT_RATIO);
        self.recompute();
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Retriggers the plate with the given strike `velocity` (clamped to
    /// `[0, 1]`), injecting a fresh beater pulse while existing modes keep
    /// ringing.
    pub fn strike(&mut self, velocity: Sample) {
        self.velocity = finite_or(velocity, DEFAULT_STRIKE_VELOCITY).clamp(0.0, 1.0);
        self.pulse_pos = 0;
    }

    /// Recomputes every mode coefficient and the beater-pulse geometry from the
    /// current scalar parameters. Never runs on the audio hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        let gain_exp = GAIN_EXP_MIN + (1.0 - self.brightness) * (GAIN_EXP_MAX - GAIN_EXP_MIN);
        let nyquist = sr * NYQUIST_GUARD;

        let mut ratios = [0.0 as Sample; NUM_MODES];
        plate_mode_ratios(self.aspect_ratio, &mut ratios);

        for (m, &ratio) in ratios.iter().enumerate() {
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

    /// Renders one mono output sample, advancing every resonator and the beater
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

impl AudioNode for StruckPlateNode {
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
    fn render(node: &mut StruckPlateNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(
        node: &mut StruckPlateNode,
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
    fn default_strike_produces_sound() {
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
        let out = render(&mut node, SR as usize / 10);
        let p = peak(&out);
        assert!(p > 1.0e-3, "plate should ring, peak = {p}");
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn tone_decays_toward_silence() {
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
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
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
        let out = render(&mut node, 4 * SR as usize);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(peak(&out) < 1.0, "peak = {}", peak(&out));
    }

    #[test]
    fn full_parameter_grid_stays_below_full_scale() {
        let mut worst = 0.0_f32;
        for &f0 in &[40.0, 110.0, 220.0, 880.0, 2000.0] {
            for &decay in &[0.1, 2.0, 12.0, 20.0] {
                for &bright in &[0.0, 0.5, 1.0] {
                    for &aspect in &[1.0, 1.4, 2.5, 3.0] {
                        let params = StruckPlateParams {
                            frequency_hz: f0,
                            decay_s: decay,
                            brightness: bright,
                            aspect_ratio: aspect,
                            amplitude: 1.0,
                        };
                        let mut node = StruckPlateNode::new(SR, params);
                        let out = render(&mut node, SR as usize / 4);
                        let p = peak(&out);
                        worst = worst.max(p);
                        assert!(
                            p < 1.0 && out.iter().all(|s| s.is_finite()),
                            "f0={f0} decay={decay} bright={bright} aspect={aspect} peak={p}"
                        );
                    }
                }
            }
        }
        assert!(worst > 0.1, "grid should actually produce sound: {worst}");
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = StruckPlateNode::new(SR, StruckPlateParams::default());
        let mut b = StruckPlateNode::new(SR, StruckPlateParams::default());
        let out_a = render(&mut a, SR as usize);
        let out_b = render(&mut b, SR as usize);
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn reset_replays_identical_attack() {
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
        let first = render(&mut node, SR as usize / 2);
        node.reset();
        let after = render(&mut node, SR as usize / 2);
        assert_eq!(first, after);
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut loud = StruckPlateNode::new(
            SR,
            StruckPlateParams {
                amplitude: 1.0,
                ..StruckPlateParams::default()
            },
        );
        let mut soft = StruckPlateNode::new(
            SR,
            StruckPlateParams {
                amplitude: 0.5,
                ..StruckPlateParams::default()
            },
        );
        let el = energy(&render(&mut loud, SR as usize / 2));
        let es = energy(&render(&mut soft, SR as usize / 2));
        let ratio = el / es;
        assert!((ratio - 4.0).abs() < 1.0e-2, "energy ratio = {ratio}");
    }

    #[test]
    fn fundamental_mode_present() {
        let f0 = 220.0;
        let mut node = StruckPlateNode::new(
            SR,
            StruckPlateParams {
                frequency_hz: f0,
                ..StruckPlateParams::default()
            },
        );
        let out = render(&mut node, SR as usize / 2);
        assert!(
            goertzel(&out, f0) > 10.0,
            "fundamental should ring: {}",
            goertzel(&out, f0)
        );
    }

    #[test]
    fn frequency_changes_output() {
        let mut low = StruckPlateNode::new(
            SR,
            StruckPlateParams {
                frequency_hz: 150.0,
                ..StruckPlateParams::default()
            },
        );
        let mut high = StruckPlateNode::new(
            SR,
            StruckPlateParams {
                frequency_hz: 450.0,
                ..StruckPlateParams::default()
            },
        );
        let out_low = render(&mut low, SR as usize / 2);
        let out_high = render(&mut high, SR as usize / 2);
        assert!(goertzel(&out_low, 150.0) > goertzel(&out_low, 450.0));
        assert!(goertzel(&out_high, 450.0) > goertzel(&out_high, 150.0));
    }

    #[test]
    fn brightness_changes_timbre() {
        let mut bright = StruckPlateNode::new(
            SR,
            StruckPlateParams {
                brightness: 1.0,
                ..StruckPlateParams::default()
            },
        );
        let mut dull = StruckPlateNode::new(
            SR,
            StruckPlateParams {
                brightness: 0.0,
                ..StruckPlateParams::default()
            },
        );
        let hb = hf_energy(&render(&mut bright, SR as usize / 5));
        let hd = hf_energy(&render(&mut dull, SR as usize / 5));
        assert!(hb > hd * 3.0, "bright HF {hb} should exceed dull HF {hd}");
    }

    #[test]
    fn aspect_ratio_reshapes_spectrum() {
        let f0 = 220.0;
        // A square plate (aspect 1) rings at ratios 1, 2.5, 2.5, 4, ...; it has
        // no partial at ratio 1.6. A 2:1 plate does (its second mode).
        let probe = f0 * 1.6;
        let mut square = StruckPlateNode::new(
            SR,
            StruckPlateParams {
                frequency_hz: f0,
                aspect_ratio: 1.0,
                ..StruckPlateParams::default()
            },
        );
        let mut oblong = StruckPlateNode::new(
            SR,
            StruckPlateParams {
                frequency_hz: f0,
                aspect_ratio: 2.0,
                ..StruckPlateParams::default()
            },
        );
        let gs = goertzel(&render(&mut square, SR as usize / 2), probe);
        let go = goertzel(&render(&mut oblong, SR as usize / 2), probe);
        assert!(
            go > gs * 20.0,
            "oblong plate should ring at {probe} Hz: square={gs} oblong={go}"
        );
    }

    #[test]
    fn strike_retriggers_while_ringing() {
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
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
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = StruckPlateParams {
            frequency_hz: 330.0,
            decay_s: 5.0,
            brightness: 0.3,
            aspect_ratio: 2.2,
            amplitude: 0.7,
        };
        let node = StruckPlateNode::new(SR, params);
        assert!((node.frequency_hz() - 330.0).abs() < 1.0e-3);
        assert!((node.decay_s() - 5.0).abs() < 1.0e-3);
        assert!((node.brightness() - 0.3).abs() < 1.0e-3);
        assert!((node.aspect_ratio() - 2.2).abs() < 1.0e-3);
        assert!((node.amplitude() - 0.7).abs() < 1.0e-3);
    }

    #[test]
    fn frequency_is_clamped() {
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
        node.set_frequency(-100.0);
        assert!(node.frequency_hz() >= MIN_FREQUENCY_HZ);
        node.set_frequency(1.0e9);
        assert!(node.frequency_hz() <= MAX_FREQUENCY_HZ);
    }

    #[test]
    fn decay_is_clamped() {
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
        node.set_decay(-1.0);
        assert!((node.decay_s() - MIN_DECAY_S).abs() < 1.0e-6);
        node.set_decay(1.0e6);
        assert!((node.decay_s() - MAX_DECAY_S).abs() < 1.0e-6);
    }

    #[test]
    fn brightness_is_clamped() {
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
        node.set_brightness(-1.0);
        assert!((node.brightness() - 0.0).abs() < 1.0e-6);
        node.set_brightness(5.0);
        assert!((node.brightness() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn aspect_ratio_is_clamped() {
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
        node.set_aspect_ratio(0.1);
        assert!((node.aspect_ratio() - MIN_ASPECT_RATIO).abs() < 1.0e-6);
        node.set_aspect_ratio(100.0);
        assert!((node.aspect_ratio() - MAX_ASPECT_RATIO).abs() < 1.0e-6);
    }

    #[test]
    fn setters_reject_non_finite_and_keep_previous() {
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
        let (f, d, b, a) = (
            node.frequency_hz(),
            node.decay_s(),
            node.brightness(),
            node.aspect_ratio(),
        );
        node.set_frequency(Sample::NAN);
        node.set_decay(Sample::INFINITY);
        node.set_brightness(Sample::NAN);
        node.set_aspect_ratio(Sample::NEG_INFINITY);
        assert!((node.frequency_hz() - f).abs() < 1.0e-6);
        assert!((node.decay_s() - d).abs() < 1.0e-6);
        assert!((node.brightness() - b).abs() < 1.0e-6);
        assert!((node.aspect_ratio() - a).abs() < 1.0e-6);
    }

    #[test]
    fn constructor_sanitises_non_finite() {
        let params = StruckPlateParams {
            frequency_hz: Sample::NAN,
            decay_s: Sample::INFINITY,
            brightness: Sample::NAN,
            aspect_ratio: Sample::NEG_INFINITY,
            amplitude: Sample::NAN,
        };
        let mut node = StruckPlateNode::new(SR, params);
        let out = render(&mut node, SR as usize / 10);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!((node.frequency_hz() - DEFAULT_FREQUENCY_HZ).abs() < 1.0e-3);
        assert!((node.aspect_ratio() - DEFAULT_ASPECT_RATIO).abs() < 1.0e-3);
    }

    #[test]
    fn mono_core_copies_to_all_channels() {
        let mut node = StruckPlateNode::new(SR, StruckPlateParams::default());
        let chans = render_layout(&mut node, SR as usize / 10, ChannelLayout::Quad);
        assert_eq!(chans.len(), 4);
        for ch in 1..chans.len() {
            assert_eq!(chans[0], chans[ch]);
        }
    }

    #[test]
    fn high_pitch_mutes_supersonic_modes() {
        let node = StruckPlateNode::new(
            SR,
            StruckPlateParams {
                frequency_hz: MAX_FREQUENCY_HZ,
                ..StruckPlateParams::default()
            },
        );
        // The fundamental is audible, but the top of the dense grid lands above
        // the Nyquist guard and must be muted.
        assert!(node.mode_enabled(0));
        assert!(!node.mode_enabled(NUM_MODES - 1));
    }

    #[test]
    fn latency_is_zero() {
        let node = StruckPlateNode::new(SR, StruckPlateParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
