//! Diode-clipper distortion: an antiparallel diode pair shunt-clamping the
//! signal to ground across a series load resistor, solved implicitly each
//! sample, with an asymmetry bias and a coupling `DC` block.
//!
//! A diode clipper is the classic analog "overdrive" building block (the
//! clamp at the heart of countless guitar-pedal distortion circuits). A pair
//! of diodes wired antiparallel across the signal path conducts once the
//! voltage across them exceeds the diode threshold, shunting the excess to
//! ground. Below threshold the diodes are effectively open and the stage is
//! unity gain; above it they clamp hard but *smoothly*, because the diode
//! current grows exponentially rather than switching abruptly.
//!
//! The governing relation is the Shockley diode law combined with the series
//! resistor's Ohmic drop. For the antiparallel pair the two exponentials
//! combine into a hyperbolic sine, so the output voltage `y` across the diodes
//! for an input drive voltage `u` satisfies the implicit equation
//!
//! ```text
//! u = y + ca * sinh(y / vt)
//! ```
//!
//! where `vt` is the diode thermal-voltage "knee" (how soft the onset is) and
//! `ca` folds the series resistance and saturation current into one scale
//! (fixed internally at `1`). There is no closed form, so each sample is
//! solved with a fixed, bounded **Newton-Raphson** iteration:
//!
//! ```text
//! F(y)  = y + ca * sinh(y / vt) - u
//! F'(y) = 1 + (ca / vt) * cosh(y / vt)
//! y    <- y - F(y) / F'(y)
//! ```
//!
//! The iteration is seeded with `y0 = sign(u) * min(yLin, ySat)`, where
//! `yLin = |u| / (1 + ca / vt)` is the exact small-signal (linear-region)
//! answer and `ySat = vt * asinh(|u| / ca)` is the exact large-signal
//! (saturated-region) answer. Each bound is accurate in its own regime, so the
//! smaller of the two is an excellent global seed and a fixed four-step
//! Newton loop converges to machine precision across the entire audio range --
//! making the solve real-time deterministic, allocation-free, and panic-free.
//!
//! The raw solution is normalised to unity small-signal gain by multiplying by
//! `(vt + ca) / vt`, which cancels the `vt / (vt + ca)` slope the clamp has at
//! the origin:
//!
//! ```text
//! diode_shape(v) = solve(v) * (vt + ca) / vt
//! shaped         = diode_shape(drive * x + bias) / drive
//! ```
//!
//! Dividing the shaped result by `drive` keeps the overall small-signal gain
//! at unity regardless of how hard the clamp is pushed. The `sinh`, `cosh`,
//! and `asinh` are expanded from `exp`, `ln`, and `sqrt` (all through
//! [`bevy_math::ops`]) so the solver is bit-for-bit deterministic.
//!
//! **Asymmetry** is produced by biasing the operating point: adding a small
//! `bias` to the drive before the (perfectly odd) clamp shifts the signal onto
//! an asymmetric portion of the curve, which injects even harmonics the way an
//! unmatched real diode pair (silicon one side, germanium the other) would.
//! Because the clamp itself is `C`-infinity smooth there is no origin kink; the
//! even harmonics come purely from the bias. A first-order coupling `DC` block
//! then removes the static offset the bias leaves behind:
//!
//! ```text
//! hp = shaped - shaped_prev + r * hp_prev
//! ```
//!
//! # Provenance
//!
//! The Shockley diode equation, the antiparallel-pair hyperbolic-sine clamp,
//! and implicit Newton solving of a memoryless non-linearity are long-standing
//! public results in circuit theory and virtual-analog `DSP` (for example Udo
//! Zolzer, "DAFX: Digital Audio Effects", and the standard literature on
//! diode-clipper and wave-digital circuit modelling). The hyperbolic
//! identities, the Newton-Raphson method, and the first-order `DC` blocker are
//! standard public mathematics, re-derived here from first principles. This
//! file contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, or Web Audio source or derived code**, and uses
//! **no machine-learning or AI techniques** -- it is purely classic signal
//! processing.
//!
//! # Relationship
//!
//! `diode_clipper` is the only node whose non-linearity is defined
//! *implicitly* by the Shockley `V`-`I` law `u = y + ca * sinh(y / vt)` and
//! solved per sample with Newton-Raphson (the `asinh` / `log` family). That
//! implicit, physically-grounded clamp is its defining contrast with the other
//! shapers:
//!
//! - [`tube`](super::tube) models a valve with an explicit dual-slope `tanh`
//!   grid clamp wrapped in first-order `RC` shelving / Miller roll-off; it has
//!   no implicit solve.
//! - [`saturation`](super::saturation) is a memoryless *fixed-curve* shaper
//!   (`tanh` / `arctan` / cubic / ...), evaluated directly with no iteration.
//! - [`waveshaper`](super::waveshaper) applies a fixed symmetric `tanh` clip.
//! - [`wavefolder`](super::wavefolder) *reflects* the signal past a threshold
//!   rather than clamping it.
//! - [`exciter`](super::exciter) synthesises added high harmonics instead of
//!   clamping the broadband signal.
//!
//! No other node solves a Shockley diode law; the exponential-onset clamp with
//! bias-driven even harmonics and a coupling `DC` block is unique to this
//! module.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Largest clipping drive, so a very hot input cannot push the implicit solve
/// into a numerically pathological region.
pub const MAX_DRIVE: Sample = 64.0;

/// Smallest clipping drive. The shaper divides by `drive` to normalise its
/// small-signal gain, so the drive is clamped away from zero.
pub const MIN_DRIVE: Sample = 0.1;

/// Largest asymmetry bias. Kept modest so the bias stays a small operating-
/// point shift rather than a gross `DC` offset.
pub const MAX_ASYMMETRY: Sample = 1.0;

/// Smallest diode knee (thermal-voltage scale). Smaller knees clamp harder;
/// clamped away from zero so the reciprocal `1 / vt` stays finite.
pub const MIN_KNEE: Sample = 0.02;

/// Largest diode knee. Larger knees are softer and closer to linear.
pub const MAX_KNEE: Sample = 1.0;

/// Largest absolute output trim in decibels.
pub const MAX_TRIM_DB: Sample = 24.0;

/// Lowest allowed coupling `DC`-block corner frequency in Hz.
const MIN_DC_BLOCK_HZ: Sample = 1.0;
/// Highest allowed coupling `DC`-block corner frequency in Hz.
const MAX_DC_BLOCK_HZ: Sample = 60.0;

/// Fixed diode current / series-resistance scale `ca` in the implicit law
/// `u = y + ca * sinh(y / vt)`. Folding both constants into unity keeps the
/// single remaining shape control the knee `vt`.
const DIODE_CA: Sample = 1.0;

/// Number of Newton-Raphson steps per sample. Four steps converge the seeded
/// iteration to machine precision across the entire audio range, and the fixed
/// count keeps the solve real-time bounded.
const NEWTON_ITERS: usize = 4;

/// Hyperbolic sine via `exp`, so the solver stays on the deterministic
/// [`bevy_math::ops`] math path.
#[inline]
#[must_use]
fn sinh_ops(x: Sample) -> Sample {
    let e = ops::exp(x);
    0.5 * (e - 1.0 / e)
}

/// Hyperbolic cosine via `exp`, so the solver stays on the deterministic
/// [`bevy_math::ops`] math path.
#[inline]
#[must_use]
fn cosh_ops(x: Sample) -> Sample {
    let e = ops::exp(x);
    0.5 * (e + 1.0 / e)
}

/// Inverse hyperbolic sine via `ln` and `sqrt` for non-negative `x`, so the
/// seed stays on the deterministic [`bevy_math::ops`] math path.
#[inline]
#[must_use]
fn asinh_ops(x: Sample) -> Sample {
    ops::ln(x + ops::sqrt(x * x + 1.0))
}

/// Clamps a drive request to the safe, finite, strictly positive range.
#[inline]
#[must_use]
fn clamp_drive(drive: Sample) -> Sample {
    if drive.is_finite() {
        drive.clamp(MIN_DRIVE, MAX_DRIVE)
    } else {
        1.0
    }
}

/// Clamps a diode knee request to the finite `[MIN_KNEE, MAX_KNEE]` range.
#[inline]
#[must_use]
fn clamp_knee(knee: Sample) -> Sample {
    if knee.is_finite() {
        knee.clamp(MIN_KNEE, MAX_KNEE)
    } else {
        0.1
    }
}

/// Clamps an asymmetry bias request to `[0, MAX_ASYMMETRY]`.
#[inline]
#[must_use]
fn clamp_asymmetry(asym: Sample) -> Sample {
    if asym.is_finite() {
        asym.clamp(0.0, MAX_ASYMMETRY)
    } else {
        0.0
    }
}

/// Clamps an output trim (decibels) to a safe, finite range.
#[inline]
#[must_use]
fn clamp_trim_db(db: Sample) -> Sample {
    if db.is_finite() {
        db.clamp(-MAX_TRIM_DB, MAX_TRIM_DB)
    } else {
        0.0
    }
}

/// Clamps a corner frequency to the finite `[lo, hi]` range.
#[inline]
#[must_use]
fn clamp_freq(freq: Sample, lo: Sample, hi: Sample) -> Sample {
    if freq.is_finite() {
        freq.clamp(lo, hi)
    } else {
        lo
    }
}

/// First-order `DC`-blocker pole for a corner at `fc` Hz, kept strictly below
/// unity so the recursion stays stable.
#[inline]
#[must_use]
fn dc_block_pole(fc: Sample, sr: Sample) -> Sample {
    ops::exp(-core::f32::consts::TAU * fc / sr).clamp(0.0, 0.999_999)
}

/// Parameters controlling the diode-clipper colour.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DiodeClipperParams {
    /// Clipping drive (how hard the signal is pushed into the diode clamp).
    pub drive: Sample,
    /// Diode knee (thermal-voltage scale) in `[MIN_KNEE, MAX_KNEE]`: smaller
    /// knees clamp harder and sooner, larger knees are softer / more linear.
    pub knee: Sample,
    /// Asymmetry bias in `[0, MAX_ASYMMETRY]`: `0` is a symmetric clamp (odd
    /// harmonics only), larger values shift the operating point to inject even
    /// harmonics (the static offset is removed by the `DC` block).
    pub asymmetry: Sample,
    /// Corner frequency in Hz of the coupling `DC`-blocking high pass.
    pub dc_block_hz: Sample,
    /// Output trim in decibels applied to the wet path.
    pub output_trim_db: Sample,
    /// Wet / dry blend in `[0, 1]`: `0` is the untouched input, `1` is the
    /// fully processed clipper.
    pub mix: Sample,
}

impl Default for DiodeClipperParams {
    fn default() -> Self {
        Self {
            drive: 4.0,
            knee: 0.1,
            asymmetry: 0.3,
            dc_block_hz: 10.0,
            output_trim_db: 0.0,
            mix: 1.0,
        }
    }
}

/// Allocation-free diode-clipper `DSP` core: an implicit antiparallel-diode
/// clamp solved per sample with a fixed Newton iteration, an asymmetry bias,
/// and a first-order coupling `DC` block.
///
/// All state is pre-allocated at construction, so [`DiodeClipper::voice`] is
/// real-time safe (no allocation, no locking, no panic). The owning
/// [`DiodeClipperNode`] performs the dry blend and output trim.
#[derive(Debug, Clone)]
pub struct DiodeClipper {
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Clipping drive (clamped to `[MIN_DRIVE, MAX_DRIVE]`).
    drive: Sample,
    /// Reciprocal of `drive`, precomputed for the small-signal normalisation.
    inv_drive: Sample,
    /// Diode knee `vt` (clamped to `[MIN_KNEE, MAX_KNEE]`).
    knee: Sample,
    /// Reciprocal of `knee`, precomputed for the Newton iteration.
    inv_knee: Sample,
    /// Unity small-signal normalisation factor `(knee + ca) / knee`.
    norm: Sample,
    /// Asymmetry bias added to the drive before the clamp.
    bias: Sample,
    /// Steady-state shaped output for a zero input (the clamp's response to
    /// the bias alone), subtracted so the operating-point `DC` offset is
    /// removed analytically and silence maps to silence.
    bias_offset: Sample,
    /// Coupling `DC`-blocker pole.
    dc_coef: Sample,
    /// Coupling `DC`-block corner frequency in Hz (clamped).
    dc_block_hz: Sample,
    /// Per-channel coupling `DC`-blocker previous input (shaped sample).
    dc_x1: Vec<Sample>,
    /// Per-channel coupling `DC`-blocker previous output.
    dc_y1: Vec<Sample>,
}

impl DiodeClipper {
    /// Builds a diode-clipper core for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: DiodeClipperParams, sample_rate: u32, channels: usize) -> Self {
        let sr = sample_rate.max(1);
        let srf = sr as Sample;
        let channels = channels.max(1);

        let drive = clamp_drive(params.drive);
        let knee = clamp_knee(params.knee);
        let dc_block_hz = clamp_freq(params.dc_block_hz, MIN_DC_BLOCK_HZ, MAX_DC_BLOCK_HZ);

        let mut core = Self {
            sample_rate: sr,
            drive,
            inv_drive: 1.0 / drive,
            knee,
            inv_knee: 1.0 / knee,
            norm: (knee + DIODE_CA) / knee,
            bias: clamp_asymmetry(params.asymmetry),
            bias_offset: 0.0,
            dc_coef: dc_block_pole(dc_block_hz, srf),
            dc_block_hz,
            dc_x1: vec![0.0; channels],
            dc_y1: vec![0.0; channels],
        };
        core.recompute_bias_offset();
        core
    }

    /// Recomputes the steady-state operating-point offset for the current
    /// drive, knee, and bias, so a zero input produces a zero shaped output.
    #[inline]
    fn recompute_bias_offset(&mut self) {
        self.bias_offset = self.diode_shape(self.bias) * self.inv_drive;
    }

    /// Number of channels the core tracks.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.dc_x1.len()
    }

    /// Current clipping drive.
    #[inline]
    #[must_use]
    pub fn drive(&self) -> Sample {
        self.drive
    }

    /// Current diode knee.
    #[inline]
    #[must_use]
    pub fn knee(&self) -> Sample {
        self.knee
    }

    /// Current asymmetry bias.
    #[inline]
    #[must_use]
    pub fn asymmetry(&self) -> Sample {
        self.bias
    }

    /// Solves the implicit diode law `u = y + ca * sinh(y / vt)` for `y` with a
    /// seeded, fixed Newton iteration. The seed is the smaller magnitude of the
    /// exact linear-region and saturated-region answers, so the fixed loop
    /// converges across the whole audio range.
    #[inline]
    #[must_use]
    fn solve(&self, u: Sample) -> Sample {
        let au = u.abs();
        let y_lin = au / (1.0 + DIODE_CA * self.inv_knee);
        let y_sat = self.knee * asinh_ops(au / DIODE_CA);
        let sign = if u >= 0.0 { 1.0 } else { -1.0 };
        let mut y = sign * y_lin.min(y_sat);
        for _ in 0..NEWTON_ITERS {
            let yv = y * self.inv_knee;
            let f = y + DIODE_CA * sinh_ops(yv) - u;
            let fp = 1.0 + DIODE_CA * self.inv_knee * cosh_ops(yv);
            y -= f / fp;
        }
        y
    }

    /// The implicit diode-clamp transfer curve, normalised to unity
    /// small-signal gain. `v` is the pre-scaled, biased input.
    #[inline]
    #[must_use]
    fn diode_shape(&self, v: Sample) -> Sample {
        self.solve(v) * self.norm
    }

    /// Sets the clipping drive (clamped to `[MIN_DRIVE, MAX_DRIVE]`).
    #[inline]
    pub fn set_drive(&mut self, drive: Sample) {
        self.drive = clamp_drive(drive);
        self.inv_drive = 1.0 / self.drive;
        self.recompute_bias_offset();
    }

    /// Sets the diode knee (clamped to `[MIN_KNEE, MAX_KNEE]`).
    #[inline]
    pub fn set_knee(&mut self, knee: Sample) {
        self.knee = clamp_knee(knee);
        self.inv_knee = 1.0 / self.knee;
        self.norm = (self.knee + DIODE_CA) / self.knee;
        self.recompute_bias_offset();
    }

    /// Sets the asymmetry bias (clamped to `[0, MAX_ASYMMETRY]`).
    #[inline]
    pub fn set_asymmetry(&mut self, asymmetry: Sample) {
        self.bias = clamp_asymmetry(asymmetry);
        self.recompute_bias_offset();
    }

    /// Sets the coupling `DC`-block corner frequency in Hz.
    #[inline]
    pub fn set_dc_block_hz(&mut self, dc_block_hz: Sample) {
        self.dc_block_hz = clamp_freq(dc_block_hz, MIN_DC_BLOCK_HZ, MAX_DC_BLOCK_HZ);
        self.dc_coef = dc_block_pole(self.dc_block_hz, self.sample_rate as Sample);
    }

    /// Processes one input sample for channel `ch` through the clamp and the
    /// coupling `DC` block and returns the wet output (before any dry blend or
    /// output trim).
    #[inline]
    #[must_use]
    pub fn voice(&mut self, ch: usize, x: Sample) -> Sample {
        // Implicit antiparallel-diode clamp with asymmetry bias, normalised to
        // unity small-signal gain by dividing out the drive.
        let shaped = self.diode_shape(self.drive * x + self.bias) * self.inv_drive - self.bias_offset;

        // Coupling-capacitor first-order high pass (DC block) removes the
        // static offset the asymmetry bias introduces.
        let hp = shaped - self.dc_x1[ch] + self.dc_coef * self.dc_y1[ch];
        self.dc_x1[ch] = flush_denormal(shaped);
        self.dc_y1[ch] = flush_denormal(hp);
        hp
    }

    /// Clears all filter state.
    #[inline]
    pub fn reset(&mut self) {
        for s in &mut self.dc_x1 {
            *s = 0.0;
        }
        for s in &mut self.dc_y1 {
            *s = 0.0;
        }
    }
}

/// A diode-clipper distortion node (input port 0 -> output port 0).
///
/// The dry signal is blended with the processed clipper path by a [`Smoothed`]
/// `mix` so automation stays click-free, and the wet path is scaled by a
/// linear output trim. Every channel is processed independently from shared
/// coefficients, so a mono source fed to several channels stays phase
/// coherent.
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{DiodeClipperNode, DiodeClipperParams};
///
/// let mut node = DiodeClipperNode::new(DiodeClipperParams::default(), 48_000, 1);
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, 8);
/// input.set_active_frames(8);
/// output.set_active_frames(8);
/// for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
///     *s = 0.7 * (i as f32 * 0.3).sin();
/// }
/// let ctx = RenderContext { sample_rate: 48_000, frames: 8, playhead: 0 };
/// let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
/// node.process(&ctx, &mut io);
/// assert!(output.channel(0).iter().all(|s| s.is_finite()));
/// ```
#[derive(Debug, Clone)]
pub struct DiodeClipperNode {
    /// `DSP` core.
    core: DiodeClipper,
    /// Smoothed wet/dry blend in `[0, 1]`.
    mix: Smoothed,
    /// Linear output trim applied to the wet path.
    trim: Sample,
}

impl DiodeClipperNode {
    /// Builds a diode-clipper node for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: DiodeClipperParams, sample_rate: u32, channels: usize) -> Self {
        Self {
            mix: Smoothed::new(params.mix.clamp(0.0, 1.0)),
            trim: db_to_linear(clamp_trim_db(params.output_trim_db)),
            core: DiodeClipper::new(params, sample_rate, channels),
        }
    }

    /// Number of channels the node processes.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.core.channels()
    }

    /// Sets the clipping drive.
    #[inline]
    pub fn set_drive(&mut self, drive: Sample) {
        self.core.set_drive(drive);
    }

    /// Sets the diode knee.
    #[inline]
    pub fn set_knee(&mut self, knee: Sample) {
        self.core.set_knee(knee);
    }

    /// Sets the asymmetry bias.
    #[inline]
    pub fn set_asymmetry(&mut self, asymmetry: Sample) {
        self.core.set_asymmetry(asymmetry);
    }

    /// Sets the coupling `DC`-block corner frequency in Hz.
    #[inline]
    pub fn set_dc_block_hz(&mut self, dc_block_hz: Sample) {
        self.core.set_dc_block_hz(dc_block_hz);
    }

    /// Sets the linear output trim from a decibel value.
    #[inline]
    pub fn set_output_trim_db(&mut self, output_trim_db: Sample) {
        self.trim = db_to_linear(clamp_trim_db(output_trim_db));
    }

    /// Sets the wet/dry blend with the given ramp.
    #[inline]
    pub fn set_mix(&mut self, mix: Sample, ramp: Ramp) {
        self.mix.set_target(mix.clamp(0.0, 1.0), ramp);
    }
}

impl AudioNode for DiodeClipperNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(self.core.channels());
        let frames = output.active_frames();

        for f in 0..frames {
            let mix = self.mix.next_sample();
            for ch in 0..channels {
                let raw = input.channel(ch)[f];
                let x = if raw.is_finite() { raw } else { 0.0 };
                let wet = self.core.voice(ch, x);
                let wet = if wet.is_finite() { wet } else { 0.0 };
                output.channel_mut(ch)[f] = (1.0 - mix) * x + mix * (self.trim * wet);
            }
        }
    }

    fn reset(&mut self) {
        self.core.reset();
        self.mix = Smoothed::new(self.mix.target());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use alloc::vec::Vec;

    const SR: u32 = 48_000;

    fn params() -> DiodeClipperParams {
        DiodeClipperParams::default()
    }

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn mono(frames: usize) -> AudioBuffer {
        let mut b = AudioBuffer::new(ChannelLayout::Mono, frames);
        b.set_active_frames(frames);
        b
    }

    fn stereo(frames: usize) -> AudioBuffer {
        let mut b = AudioBuffer::new(ChannelLayout::Stereo, frames);
        b.set_active_frames(frames);
        b
    }

    fn sine(buf: &mut AudioBuffer, freq_hz: Sample, amp: Sample) {
        let frames = buf.active_frames();
        for ch in 0..buf.channels() {
            for i in 0..frames {
                let phase = core::f32::consts::TAU * freq_hz * (i as Sample) / (SR as Sample);
                buf.channel_mut(ch)[i] = amp * ops::sin(phase);
            }
        }
    }

    /// Single-bin Goertzel magnitude (normalised by length).
    fn goertzel(signal: &[Sample], freq: Sample) -> Sample {
        let n = signal.len();
        if n == 0 {
            return 0.0;
        }
        let w = core::f32::consts::TAU * freq / SR as Sample;
        let (sin_w, cos_w) = ops::sin_cos(w);
        let coeff = 2.0 * cos_w;
        let mut s_prev = 0.0_f32;
        let mut s_prev2 = 0.0_f32;
        for &x in signal {
            let s = x + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
        }
        let real = s_prev - s_prev2 * cos_w;
        let imag = s_prev2 * sin_w;
        ops::sqrt(real * real + imag * imag) / n as Sample
    }

    /// Runs a single mono tone through a node and returns the output tail
    /// (second half, so filter transients have settled).
    fn run_tone(mut node: DiodeClipperNode, freq_hz: Sample, amp: Sample, frames: usize) -> Vec<Sample> {
        let mut input = mono(frames);
        sine(&mut input, freq_hz, amp);
        let inputs = [input];
        let mut outputs = [mono(frames)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        outputs[0].channel(0)[frames / 2..].to_vec()
    }

    #[test]
    fn latency_is_zero() {
        let node = DiodeClipperNode::new(params(), SR, 1);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = DiodeClipperNode::new(params(), SR, 1);
        let inputs = [mono(512)];
        let mut outputs = [mono(512)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(512), &mut io);
        for &y in outputs[0].channel(0) {
            assert!(y.abs() < 1.0e-6, "silence leaked output: {y}");
        }
    }

    #[test]
    fn output_is_finite_for_a_tone() {
        let out = run_tone(DiodeClipperNode::new(params(), SR, 1), 440.0, 0.8, 2048);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut node = DiodeClipperNode::new(params(), SR, 1);
        let mut input = mono(64);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = if i % 3 == 0 { Sample::NAN } else { Sample::INFINITY };
        }
        let inputs = [input];
        let mut outputs = [mono(64)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(64), &mut io);
        for &y in outputs[0].channel(0) {
            assert!(y.is_finite(), "non-finite input leaked: {y}");
        }
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = DiodeClipperNode::new(params(), SR, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
        input.set_active_frames(0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 8);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }

    #[test]
    fn mix_zero_passes_input_through() {
        let mut p = params();
        p.mix = 0.0;
        let mut node = DiodeClipperNode::new(p, SR, 1);
        let mut input = mono(256);
        sine(&mut input, 440.0, 0.9);
        let reference = input.channel(0).to_vec();
        let inputs = [input];
        let mut outputs = [mono(256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        for (y, x) in outputs[0].channel(0).iter().zip(reference.iter()) {
            assert!((y - x).abs() < 1.0e-6, "dry path altered input: {y} vs {x}");
        }
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let mut p = params();
        p.drive = 1.0e9;
        p.knee = -5.0;
        p.asymmetry = 50.0;
        p.dc_block_hz = 1.0e9;
        p.output_trim_db = 1.0e9;
        p.mix = 7.0;
        let mut node = DiodeClipperNode::new(p, SR, 2);
        let mut input = stereo(512);
        sine(&mut input, 220.0, 4.0);
        let inputs = [input];
        let mut outputs = [stereo(512)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(512), &mut io);
        for ch in 0..2 {
            for &y in outputs[0].channel(ch) {
                assert!(y.is_finite(), "extreme params leaked non-finite: {y}");
            }
        }
    }

    #[test]
    fn non_finite_params_fall_back_to_safe_defaults() {
        let mut p = params();
        p.drive = Sample::NAN;
        p.knee = Sample::INFINITY;
        p.asymmetry = Sample::NAN;
        p.dc_block_hz = Sample::NAN;
        let core = DiodeClipper::new(p, SR, 1);
        assert!(core.drive().is_finite());
        assert!(core.knee().is_finite());
        assert!(core.asymmetry().is_finite());
        assert!((MIN_DRIVE..=MAX_DRIVE).contains(&core.drive()));
        assert!((MIN_KNEE..=MAX_KNEE).contains(&core.knee()));
    }

    #[test]
    fn setters_clamp_and_reject_non_finite() {
        let mut core = DiodeClipper::new(params(), SR, 1);
        core.set_drive(Sample::NAN);
        assert!(core.drive().is_finite());
        core.set_drive(1.0e9);
        assert_eq!(core.drive(), MAX_DRIVE);
        core.set_knee(-1.0);
        assert_eq!(core.knee(), MIN_KNEE);
        core.set_knee(Sample::INFINITY);
        assert!(core.knee().is_finite());
        core.set_asymmetry(1.0e9);
        assert_eq!(core.asymmetry(), MAX_ASYMMETRY);
        core.set_asymmetry(Sample::NAN);
        assert!(core.asymmetry().is_finite());
    }

    #[test]
    fn dc_input_is_blocked() {
        // A constant (DC) input should settle toward zero at the output once
        // the coupling high pass has run for long enough.
        let mut node = DiodeClipperNode::new(params(), SR, 1);
        let mut input = mono(SR as usize);
        for s in input.channel_mut(0).iter_mut() {
            *s = 0.5;
        }
        let inputs = [input];
        let mut outputs = [mono(SR as usize)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(SR as usize), &mut io);
        let tail = &outputs[0].channel(0)[(SR as usize) * 3 / 4..];
        let mean = tail.iter().sum::<Sample>() / tail.len() as Sample;
        assert!(mean.abs() < 1.0e-3, "DC was not blocked: settled mean={mean}");
    }

    #[test]
    fn asymmetry_generates_even_harmonics() {
        let f = 500.0;
        let frames = 8192;
        let amp = 0.6;

        let mut sym = params();
        sym.asymmetry = 0.0;
        sym.drive = 8.0;
        let out_sym = run_tone(DiodeClipperNode::new(sym, SR, 1), f, amp, frames);
        let h2_sym = goertzel(&out_sym, 2.0 * f);

        let mut asym = params();
        asym.asymmetry = 0.5;
        asym.drive = 8.0;
        let out_asym = run_tone(DiodeClipperNode::new(asym, SR, 1), f, amp, frames);
        let h2_asym = goertzel(&out_asym, 2.0 * f);

        assert!(
            h2_asym > 3.0 * h2_sym.max(1.0e-9),
            "asymmetry did not raise the second harmonic: sym={h2_sym}, asym={h2_asym}"
        );
    }

    #[test]
    fn small_signal_gain_is_near_unity() {
        // At drive = 1 and a tiny amplitude the clamp is in its linear region,
        // so the normalised shaper should pass the tone almost unchanged.
        let f = 500.0;
        let frames = 8192;
        let amp = 1.0e-3;

        let mut p = params();
        p.drive = 1.0;
        p.asymmetry = 0.0;
        let out = run_tone(DiodeClipperNode::new(p, SR, 1), f, amp, frames);
        let mag = goertzel(&out, f);
        // Compare directly against the input tone's own Goertzel magnitude.
        let mut reference = mono(frames);
        sine(&mut reference, f, amp);
        let ref_mag = goertzel(&reference.channel(0)[frames / 2..], f);
        let ratio = mag / ref_mag.max(1.0e-12);
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "small-signal gain was not near unity: ratio={ratio}"
        );
    }

    #[test]
    fn large_drive_output_is_bounded() {
        let f = 220.0;
        let frames = 4096;
        let mut p = params();
        p.drive = 48.0;
        p.knee = 0.1;
        let out = run_tone(DiodeClipperNode::new(p, SR, 1), f, 1.0, frames);
        let peak = out.iter().fold(0.0_f32, |m, &y| m.max(y.abs()));
        assert!(peak < 1.0, "large drive output was not bounded: peak={peak}");
    }

    #[test]
    fn harder_knee_adds_more_harmonics() {
        // A smaller knee clamps harder, so a loud tone should carry more
        // third-harmonic energy than a softer (larger) knee.
        let f = 500.0;
        let frames = 8192;
        let amp = 0.9;

        let mut soft = params();
        soft.knee = 0.8;
        soft.asymmetry = 0.0;
        let out_soft = run_tone(DiodeClipperNode::new(soft, SR, 1), f, amp, frames);
        let h3_soft = goertzel(&out_soft, 3.0 * f);

        let mut hard = params();
        hard.knee = 0.05;
        hard.asymmetry = 0.0;
        let out_hard = run_tone(DiodeClipperNode::new(hard, SR, 1), f, amp, frames);
        let h3_hard = goertzel(&out_hard, 3.0 * f);

        assert!(
            h3_hard > 1.5 * h3_soft.max(1.0e-9),
            "harder knee did not add harmonics: soft={h3_soft}, hard={h3_hard}"
        );
    }

    #[test]
    fn stereo_is_phase_coherent() {
        let mut node = DiodeClipperNode::new(params(), SR, 2);
        let frames = 1024;
        let mut input = stereo(frames);
        sine(&mut input, 500.0, 0.6);
        let inputs = [input];
        let mut outputs = [stereo(frames)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let left = outputs[0].channel(0).to_vec();
        let right = outputs[0].channel(1);
        for (l, r) in left.iter().zip(right.iter()) {
            assert!((l - r).abs() < 1.0e-6, "channels diverged: {l} vs {r}");
        }
    }

    #[test]
    fn reset_clears_tail() {
        let mut node = DiodeClipperNode::new(params(), SR, 1);
        let mut input = mono(512);
        sine(&mut input, 440.0, 0.9);
        let inputs = [input];
        let mut outputs = [mono(512)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(512), &mut io);

        node.reset();
        let silent = [mono(512)];
        let mut after = [mono(512)];
        let mut io2 = ProcessIo::new(&silent, &mut after);
        node.process(&ctx(512), &mut io2);
        for &y in after[0].channel(0) {
            assert!(y.abs() < 1.0e-6, "reset did not clear tail: {y}");
        }
    }

    #[test]
    fn output_trim_scales_level() {
        // A +6 dB trim should roughly double the output level of a near-linear
        // tone compared to a 0 dB trim.
        let f = 500.0;
        let frames = 8192;
        let amp = 1.0e-3;

        let mut unity = params();
        unity.drive = 1.0;
        let out_unity = run_tone(DiodeClipperNode::new(unity, SR, 1), f, amp, frames);
        let mag_unity = goertzel(&out_unity, f);

        let mut hot = params();
        hot.drive = 1.0;
        hot.output_trim_db = 6.0;
        let out_hot = run_tone(DiodeClipperNode::new(hot, SR, 1), f, amp, frames);
        let mag_hot = goertzel(&out_hot, f);

        let ratio = mag_hot / mag_unity.max(1.0e-9);
        assert!(
            (ratio - 1.995).abs() < 0.2,
            "+6 dB trim did not double the level: ratio={ratio}"
        );
    }

    #[test]
    fn vec_collect_does_not_panic() {
        let out = run_tone(DiodeClipperNode::new(params(), SR, 1), 100.0, 0.5, 1024);
        assert_eq!(out.len(), 512);
    }
}
