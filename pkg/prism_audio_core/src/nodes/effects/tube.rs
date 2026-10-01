//! Vacuum-tube (valve) preamp emulation: grid-conduction asymmetric soft
//! clipping, a cathode-bypass low shelf, a Miller / anode high-frequency
//! roll-off, and a coupling-capacitor `DC` block.
//!
//! A triode gain stage colours audio through a handful of interlocking,
//! well-understood electrical mechanisms:
//!
//! 1. **Grid conduction.** As the grid voltage swings positive it begins to
//!    draw current, which clamps the positive half of the waveform *sooner*
//!    than the negative half. The transfer curve is therefore asymmetric,
//!    producing even harmonics (and a small `DC` shift) on top of the odd
//!    harmonics of a symmetric soft clipper.
//! 2. **Cathode-bypass shelving.** A cathode resistor bypassed by a capacitor
//!    raises the stage gain for low frequencies while leaving the highs at the
//!    unbypassed gain -- a gentle first-order bass lift.
//! 3. **Miller / anode roll-off.** The Miller capacitance together with the
//!    anode load forms a first-order low pass that softens the extreme top.
//! 4. **Coupling capacitor.** The capacitor that couples one stage to the next
//!    cannot pass a static (`DC`) voltage, so the stage is intrinsically high
//!    pass at the very bottom; it also removes the `DC` shift left by the
//!    asymmetric clipper.
//!
//! This module reproduces all four with plain, classic `DSP`. Every filter is
//! a first-order section whose coefficient has a direct resistor-capacitor
//! (`RC`) interpretation, and the only non-linearity is a memoryless
//! asymmetric soft clip. The per-channel chain is:
//!
//! ```text
//! lp      = cathode_lp + cathode_coef * (x - cathode_lp)   // one-pole low pass of x
//! shelved = x + (cathode_gain - 1) * lp                    // first-order bass shelf
//! shaped  = tube_shape(drive * shelved)                    // grid-conduction clip
//! ml      = miller_lp + miller_coef * (shaped - miller_lp) // Miller HF roll-off
//! hp      = coupling_coef * (coupling_y1 + ml - coupling_x1) // coupling DC block
//! out     = hp
//! y       = (1 - mix) * x + mix * (trim * out)
//! ```
//!
//! The soft clip is a dual-slope `tanh` normalised to unity small-signal gain:
//!
//! ```text
//! tube_shape(d) = (if d >= 0 { h_pos * tanh(d / h_pos) }
//!                  else      { h_neg * tanh(d / h_neg) }) / drive
//! ```
//!
//! with `h_pos = 1 / (1 + asymmetry)` and `h_neg = 1 * (1 + asymmetry)`. A
//! zero `asymmetry` gives `h_pos == h_neg`, a symmetric curve with odd
//! harmonics only; a positive `asymmetry` makes the positive side saturate
//! earlier (grid conduction), adding even harmonics. Each half is `h *
//! tanh(d / h)`, whose value, first derivative, and second derivative all
//! agree at the origin (slope 1, curvature 0), so the two halves join with
//! continuous curvature; only the third and higher derivatives differ, which
//! is precisely the asymmetry that generates the even-harmonic colour.
//!
//! # Provenance
//!
//! Triode grid conduction, cathode-bypass shelving, Miller roll-off, and
//! coupling-capacitor high passing are long-standing, publicly documented
//! vacuum-tube amplifier phenomena described in standard electronics and audio
//! `DSP` literature (for example Udo Zolzer, "DAFX: Digital Audio Effects",
//! and classic valve-amplifier texts). The first-order `RC` filters and the
//! saturating `tanh` transfer curve are standard public mathematics,
//! re-derived here from first principles. This file contains **no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, or
//! Web Audio source or derived code**, and uses **no machine-learning or AI
//! techniques** -- it is purely classic signal processing.
//!
//! # Relationship
//!
//! `tube` deliberately models the valve stage with **first-order (one-pole)
//! `RC` filters only**, because each stage element (cathode bypass, Miller
//! capacitance, coupling capacitor) is physically a single `RC` network. It
//! keeps a handful of scalar per-channel states and never allocates a biquad.
//! This is its implementation-level contrast with [`transformer`](super::transformer),
//! which models an *iron core* with second-order shelving / peaking biquads
//! (Direct Form I state) and a frequency-weighted "bass saturates first"
//! character plus a winding resonance. The tube has no winding resonance and
//! no frequency-weighted saturation ordering.
//!
//! It is likewise distinct from the other saturating nodes in this family:
//!
//! - [`saturation`](super::saturation) is a *memoryless* fixed-curve shaper
//!   with no shelving, roll-off, or `DC` block around it.
//! - [`tape`](super::tape) models magnetic *tape*: a symmetric `tanh` plus
//!   wow / flutter and a treble roll-off; it has no grid-conduction asymmetry,
//!   cathode shelf, or coupling high pass.
//! - [`exciter`](super::exciter) synthesises high harmonics above a split
//!   frequency, the opposite spectral emphasis to a bass-lifting tube stage.
//! - [`wavefolder`](super::wavefolder) reflects the signal past a threshold;
//!   it is not a saturating valve model at all.
//!
//! The tube's defining features -- grid-conduction asymmetry (even harmonics),
//! a cathode-bypass bass shelf, a Miller high-frequency roll-off, and a
//! coupling-capacitor `DC` block, all built from first-order `RC` filters --
//! are not provided together by any of those nodes.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Largest saturation drive, so a very hot input cannot push the clipper into
/// a numerically pathological region.
pub const MAX_DRIVE: Sample = 64.0;

/// Smallest saturation drive. The soft clip divides by `drive` to normalise
/// its small-signal gain, so the drive is clamped away from zero.
pub const MIN_DRIVE: Sample = 0.1;

/// Largest grid-conduction asymmetry. Kept below unity so neither half of the
/// dual-slope `tanh` collapses to zero width.
pub const MAX_ASYMMETRY: Sample = 0.95;

/// Largest absolute cathode-bypass shelf gain in decibels.
pub const MAX_CATHODE_DB: Sample = 24.0;

/// Largest absolute output trim in decibels.
pub const MAX_TRIM_DB: Sample = 24.0;

/// Lowest allowed cathode-bypass corner frequency in Hz.
const MIN_CATHODE_HZ: Sample = 20.0;
/// Highest allowed cathode-bypass corner frequency in Hz.
const MAX_CATHODE_HZ: Sample = 2_000.0;
/// Lowest allowed Miller roll-off corner frequency in Hz.
const MIN_MILLER_HZ: Sample = 1_000.0;
/// Highest allowed Miller roll-off corner frequency in Hz.
const MAX_MILLER_HZ: Sample = 20_000.0;
/// Lowest allowed coupling-capacitor corner frequency in Hz.
const MIN_COUPLING_HZ: Sample = 1.0;
/// Highest allowed coupling-capacitor corner frequency in Hz.
const MAX_COUPLING_HZ: Sample = 60.0;

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

/// Clamps a grid-conduction asymmetry request to `[0, MAX_ASYMMETRY]`.
#[inline]
#[must_use]
fn clamp_asymmetry(asym: Sample) -> Sample {
    if asym.is_finite() {
        asym.clamp(0.0, MAX_ASYMMETRY)
    } else {
        0.0
    }
}

/// Clamps a cathode-bypass shelf gain (decibels) to a safe, finite range.
#[inline]
#[must_use]
fn clamp_cathode_db(db: Sample) -> Sample {
    if db.is_finite() {
        db.clamp(-MAX_CATHODE_DB, MAX_CATHODE_DB)
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

/// One-pole low-pass smoothing coefficient for a corner at `fc` Hz, so that a
/// step relaxes toward its target with the `RC` time constant `1 / (2 pi fc)`.
#[inline]
#[must_use]
fn one_pole_lp_coef(fc: Sample, sr: Sample) -> Sample {
    (1.0 - ops::exp(-core::f32::consts::TAU * fc / sr)).clamp(0.0, 1.0)
}

/// First-order `DC`-blocker pole for a corner at `fc` Hz, kept strictly below
/// unity so the recursion stays stable.
#[inline]
#[must_use]
fn dc_block_pole(fc: Sample, sr: Sample) -> Sample {
    ops::exp(-core::f32::consts::TAU * fc / sr).clamp(0.0, 0.999_999)
}

/// Parameters controlling the vacuum-tube colour.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TubeParams {
    /// Clipping drive (how hard the signal is pushed into the soft clip).
    pub drive: Sample,
    /// Grid-conduction asymmetry in `[0, 1)`: `0` is symmetric (odd harmonics
    /// only), larger values make the positive half clip earlier and add even
    /// harmonics.
    pub asymmetry: Sample,
    /// Corner frequency in Hz of the cathode-bypass bass shelf.
    pub cathode_bypass_hz: Sample,
    /// Cathode-bypass shelf gain in decibels (low-frequency lift).
    pub cathode_bypass_db: Sample,
    /// Corner frequency in Hz of the Miller / anode high-frequency roll-off.
    pub miller_hz: Sample,
    /// Corner frequency in Hz of the coupling-capacitor `DC`-blocking high
    /// pass.
    pub coupling_hz: Sample,
    /// Output trim in decibels applied to the wet path.
    pub output_trim_db: Sample,
    /// Wet / dry blend in `[0, 1]`: `0` is the untouched input, `1` is the
    /// fully processed tube.
    pub mix: Sample,
}

impl Default for TubeParams {
    fn default() -> Self {
        Self {
            drive: 2.0,
            asymmetry: 0.3,
            cathode_bypass_hz: 120.0,
            cathode_bypass_db: 3.0,
            miller_hz: 10_000.0,
            coupling_hz: 20.0,
            output_trim_db: 0.0,
            mix: 1.0,
        }
    }
}

/// Allocation-free vacuum-tube `DSP` core: a cathode-bypass bass shelf, a
/// grid-conduction asymmetric soft clip, a Miller high-frequency roll-off, and
/// a coupling-capacitor `DC` block, all first-order.
///
/// All state is pre-allocated at construction, so [`Tube::voice`] is real-time
/// safe (no allocation, no locking, no panic). The owning [`TubeNode`]
/// performs the dry blend and output trim.
#[derive(Debug, Clone)]
pub struct Tube {
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Cathode-bypass one-pole low-pass coefficient.
    cathode_coef: Sample,
    /// Cathode-bypass shelf gain (linear).
    cathode_gain: Sample,
    /// Miller / anode one-pole low-pass coefficient.
    miller_coef: Sample,
    /// Coupling-capacitor `DC`-blocker pole.
    coupling_coef: Sample,
    /// Per-channel cathode low-pass state.
    cathode_lp: Vec<Sample>,
    /// Per-channel Miller low-pass state.
    miller_lp: Vec<Sample>,
    /// Per-channel coupling `DC`-blocker previous input.
    coupling_x1: Vec<Sample>,
    /// Per-channel coupling `DC`-blocker previous output.
    coupling_y1: Vec<Sample>,
    /// Clipping drive (clamped to `[MIN_DRIVE, MAX_DRIVE]`).
    drive: Sample,
    /// Reciprocal of `drive`, precomputed for the clip normalisation.
    inv_drive: Sample,
    /// Grid-conduction asymmetry (clamped to `[0, MAX_ASYMMETRY]`).
    asymmetry: Sample,
    /// Positive-half soft-clip width `1 / (1 + asymmetry)`.
    h_pos: Sample,
    /// Negative-half soft-clip width `1 * (1 + asymmetry)`.
    h_neg: Sample,
    /// Cathode-bypass corner frequency in Hz (clamped).
    cathode_hz: Sample,
    /// Cathode-bypass shelf gain in decibels (clamped).
    cathode_db: Sample,
    /// Miller roll-off corner frequency in Hz (clamped).
    miller_hz: Sample,
    /// Coupling corner frequency in Hz (clamped).
    coupling_hz: Sample,
}

impl Tube {
    /// Builds a tube core for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: TubeParams, sample_rate: u32, channels: usize) -> Self {
        let sr = sample_rate.max(1);
        let srf = sr as Sample;
        let channels = channels.max(1);

        let drive = clamp_drive(params.drive);
        let asymmetry = clamp_asymmetry(params.asymmetry);
        let cathode_hz = clamp_freq(params.cathode_bypass_hz, MIN_CATHODE_HZ, MAX_CATHODE_HZ);
        let cathode_db = clamp_cathode_db(params.cathode_bypass_db);
        let miller_hz = clamp_freq(params.miller_hz, MIN_MILLER_HZ, MAX_MILLER_HZ);
        let coupling_hz = clamp_freq(params.coupling_hz, MIN_COUPLING_HZ, MAX_COUPLING_HZ);

        Self {
            sample_rate: sr,
            cathode_coef: one_pole_lp_coef(cathode_hz, srf),
            cathode_gain: db_to_linear(cathode_db),
            miller_coef: one_pole_lp_coef(miller_hz, srf),
            coupling_coef: dc_block_pole(coupling_hz, srf),
            cathode_lp: vec![0.0; channels],
            miller_lp: vec![0.0; channels],
            coupling_x1: vec![0.0; channels],
            coupling_y1: vec![0.0; channels],
            drive,
            inv_drive: 1.0 / drive,
            asymmetry,
            h_pos: 1.0 / (1.0 + asymmetry),
            h_neg: 1.0 + asymmetry,
            cathode_hz,
            cathode_db,
            miller_hz,
            coupling_hz,
        }
    }

    /// Number of channels the core tracks.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.cathode_lp.len()
    }

    /// Current clipping drive.
    #[inline]
    #[must_use]
    pub fn drive(&self) -> Sample {
        self.drive
    }

    /// Current grid-conduction asymmetry.
    #[inline]
    #[must_use]
    pub fn asymmetry(&self) -> Sample {
        self.asymmetry
    }

    /// Current cathode-bypass shelf gain in decibels.
    #[inline]
    #[must_use]
    pub fn cathode_db(&self) -> Sample {
        self.cathode_db
    }

    /// The dual-slope grid-conduction transfer curve, normalised to unity
    /// small-signal gain. `driven` is the pre-scaled input `drive * shelved`.
    #[inline]
    #[must_use]
    fn tube_shape(&self, driven: Sample) -> Sample {
        let s = if driven >= 0.0 {
            self.h_pos * ops::tanh(driven / self.h_pos)
        } else {
            self.h_neg * ops::tanh(driven / self.h_neg)
        };
        s * self.inv_drive
    }

    /// Sets the clipping drive (clamped to `[MIN_DRIVE, MAX_DRIVE]`).
    #[inline]
    pub fn set_drive(&mut self, drive: Sample) {
        self.drive = clamp_drive(drive);
        self.inv_drive = 1.0 / self.drive;
    }

    /// Sets the grid-conduction asymmetry (clamped to `[0, MAX_ASYMMETRY]`).
    #[inline]
    pub fn set_asymmetry(&mut self, asymmetry: Sample) {
        self.asymmetry = clamp_asymmetry(asymmetry);
        self.h_pos = 1.0 / (1.0 + self.asymmetry);
        self.h_neg = 1.0 + self.asymmetry;
    }

    /// Sets the cathode-bypass corner frequency in Hz.
    #[inline]
    pub fn set_cathode_hz(&mut self, cathode_hz: Sample) {
        self.cathode_hz = clamp_freq(cathode_hz, MIN_CATHODE_HZ, MAX_CATHODE_HZ);
        self.cathode_coef = one_pole_lp_coef(self.cathode_hz, self.sample_rate as Sample);
    }

    /// Sets the cathode-bypass shelf gain in decibels.
    #[inline]
    pub fn set_cathode_db(&mut self, cathode_db: Sample) {
        self.cathode_db = clamp_cathode_db(cathode_db);
        self.cathode_gain = db_to_linear(self.cathode_db);
    }

    /// Sets the Miller roll-off corner frequency in Hz.
    #[inline]
    pub fn set_miller_hz(&mut self, miller_hz: Sample) {
        self.miller_hz = clamp_freq(miller_hz, MIN_MILLER_HZ, MAX_MILLER_HZ);
        self.miller_coef = one_pole_lp_coef(self.miller_hz, self.sample_rate as Sample);
    }

    /// Sets the coupling-capacitor corner frequency in Hz.
    #[inline]
    pub fn set_coupling_hz(&mut self, coupling_hz: Sample) {
        self.coupling_hz = clamp_freq(coupling_hz, MIN_COUPLING_HZ, MAX_COUPLING_HZ);
        self.coupling_coef = dc_block_pole(self.coupling_hz, self.sample_rate as Sample);
    }

    /// Processes one input sample for channel `ch` through the full chain and
    /// returns the wet output (before any dry blend or output trim).
    #[inline]
    #[must_use]
    pub fn voice(&mut self, ch: usize, x: Sample) -> Sample {
        // Cathode-bypass first-order bass shelf: low-pass x, then lift the
        // low-frequency content by (cathode_gain - 1).
        let lp = self.cathode_lp[ch] + self.cathode_coef * (x - self.cathode_lp[ch]);
        self.cathode_lp[ch] = flush_denormal(lp);
        let shelved = x + (self.cathode_gain - 1.0) * lp;

        // Grid-conduction asymmetric soft clip.
        let shaped = self.tube_shape(self.drive * shelved);

        // Miller / anode first-order high-frequency roll-off.
        let ml = self.miller_lp[ch] + self.miller_coef * (shaped - self.miller_lp[ch]);
        self.miller_lp[ch] = flush_denormal(ml);

        // Coupling-capacitor first-order high pass (DC block).
        let hp = self.coupling_coef * (self.coupling_y1[ch] + ml - self.coupling_x1[ch]);
        self.coupling_x1[ch] = ml;
        self.coupling_y1[ch] = flush_denormal(hp);
        hp
    }

    /// Clears all filter state.
    #[inline]
    pub fn reset(&mut self) {
        for s in &mut self.cathode_lp {
            *s = 0.0;
        }
        for s in &mut self.miller_lp {
            *s = 0.0;
        }
        for s in &mut self.coupling_x1 {
            *s = 0.0;
        }
        for s in &mut self.coupling_y1 {
            *s = 0.0;
        }
    }
}

/// A vacuum-tube-emulation node (input port 0 -> output port 0).
///
/// The dry signal is blended with the processed tube path by a [`Smoothed`]
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
/// use prism_audio_core::nodes::effects::{TubeNode, TubeParams};
///
/// let mut node = TubeNode::new(TubeParams::default(), 48_000, 1);
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, 8);
/// input.set_active_frames(8);
/// output.set_active_frames(8);
/// for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
///     *s = 0.5 * (i as f32 * 0.3).sin();
/// }
/// let ctx = RenderContext { sample_rate: 48_000, frames: 8, playhead: 0 };
/// let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
/// node.process(&ctx, &mut io);
/// assert!(output.channel(0).iter().all(|s| s.is_finite()));
/// ```
#[derive(Debug, Clone)]
pub struct TubeNode {
    /// `DSP` core.
    core: Tube,
    /// Smoothed wet/dry blend in `[0, 1]`.
    mix: Smoothed,
    /// Linear output trim applied to the wet path.
    trim: Sample,
}

impl TubeNode {
    /// Builds a tube node for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: TubeParams, sample_rate: u32, channels: usize) -> Self {
        Self {
            mix: Smoothed::new(params.mix.clamp(0.0, 1.0)),
            trim: db_to_linear(clamp_trim_db(params.output_trim_db)),
            core: Tube::new(params, sample_rate, channels),
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

    /// Sets the grid-conduction asymmetry.
    #[inline]
    pub fn set_asymmetry(&mut self, asymmetry: Sample) {
        self.core.set_asymmetry(asymmetry);
    }

    /// Sets the cathode-bypass corner frequency in Hz.
    #[inline]
    pub fn set_cathode_hz(&mut self, cathode_hz: Sample) {
        self.core.set_cathode_hz(cathode_hz);
    }

    /// Sets the cathode-bypass shelf gain in decibels.
    #[inline]
    pub fn set_cathode_db(&mut self, cathode_db: Sample) {
        self.core.set_cathode_db(cathode_db);
    }

    /// Sets the Miller roll-off corner frequency in Hz.
    #[inline]
    pub fn set_miller_hz(&mut self, miller_hz: Sample) {
        self.core.set_miller_hz(miller_hz);
    }

    /// Sets the coupling-capacitor corner frequency in Hz.
    #[inline]
    pub fn set_coupling_hz(&mut self, coupling_hz: Sample) {
        self.core.set_coupling_hz(coupling_hz);
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

impl AudioNode for TubeNode {
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

    fn params() -> TubeParams {
        TubeParams::default()
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
    fn run_tone(mut node: TubeNode, freq_hz: Sample, amp: Sample, frames: usize) -> Vec<Sample> {
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
        let node = TubeNode::new(params(), SR, 1);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = TubeNode::new(params(), SR, 1);
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
        let out = run_tone(TubeNode::new(params(), SR, 1), 440.0, 0.8, 2048);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut node = TubeNode::new(params(), SR, 1);
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
    fn zero_frames_is_noop() {
        let mut node = TubeNode::new(params(), SR, 1);
        let inputs = [mono(8)];
        let mut outputs = [mono(8)];
        outputs[0].set_active_frames(0);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }

    #[test]
    fn dry_mix_is_bit_exact_passthrough() {
        let mut p = params();
        p.mix = 0.0;
        let mut node = TubeNode::new(p, SR, 1);
        let mut input = mono(256);
        sine(&mut input, 330.0, 0.7);
        let expected = input.channel(0).to_vec();
        let inputs = [input];
        let mut outputs = [mono(256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        assert_eq!(outputs[0].channel(0), expected.as_slice());
    }

    #[test]
    fn extreme_params_do_not_panic_or_leak() {
        let mut p = params();
        p.drive = 1.0e9;
        p.asymmetry = 1.0e9;
        p.cathode_bypass_hz = 1.0e9;
        p.cathode_bypass_db = 1.0e9;
        p.miller_hz = -1.0e9;
        p.coupling_hz = 1.0e9;
        p.output_trim_db = 1.0e9;
        let out = run_tone(TubeNode::new(p, SR, 1), 220.0, 2.0, 1024);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn non_finite_params_fall_back_to_defaults() {
        let mut p = params();
        p.drive = Sample::NAN;
        p.asymmetry = Sample::INFINITY;
        p.cathode_bypass_hz = Sample::NAN;
        p.cathode_bypass_db = Sample::NAN;
        p.miller_hz = Sample::NAN;
        p.coupling_hz = Sample::NAN;
        let core = Tube::new(p, SR, 1);
        assert!(core.drive().is_finite());
        assert!(core.asymmetry().is_finite());
        assert!(core.cathode_db().is_finite());
    }

    #[test]
    fn setters_clamp_and_reject_non_finite() {
        let mut core = Tube::new(params(), SR, 1);
        core.set_drive(Sample::NAN);
        core.set_asymmetry(Sample::INFINITY);
        core.set_cathode_db(Sample::NAN);
        assert!(core.drive().is_finite());
        assert!(core.drive() >= MIN_DRIVE);
        assert!(core.asymmetry() >= 0.0 && core.asymmetry() <= MAX_ASYMMETRY);
        assert!(core.cathode_db().is_finite());

        core.set_drive(1.0e9);
        assert!((core.drive() - MAX_DRIVE).abs() < 1.0e-3);
        core.set_asymmetry(5.0);
        assert!((core.asymmetry() - MAX_ASYMMETRY).abs() < 1.0e-3);
    }

    #[test]
    fn dc_input_is_blocked() {
        // A constant input should settle toward zero at the output because the
        // coupling capacitor passes no DC.
        let mut node = TubeNode::new(params(), SR, 1);
        let frames = 8192;
        let mut input = mono(frames);
        for s in input.channel_mut(0).iter_mut() {
            *s = 0.5;
        }
        let inputs = [input];
        let mut outputs = [mono(frames)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let tail = &outputs[0].channel(0)[frames - 512..];
        let mean = tail.iter().copied().sum::<Sample>() / tail.len() as Sample;
        assert!(mean.abs() < 1.0e-3, "DC leaked through coupling block: {mean}");
    }

    #[test]
    fn asymmetry_generates_even_harmonics() {
        // A positive asymmetry should add a second harmonic that a symmetric
        // (zero-asymmetry) curve does not.
        let f0 = 1_000.0;
        let frames = 8192;

        let mut sym = params();
        sym.asymmetry = 0.0;
        let out_sym = run_tone(TubeNode::new(sym, SR, 1), f0, 0.8, frames);
        let h2_sym = goertzel(&out_sym, 2.0 * f0);

        let mut asym = params();
        asym.asymmetry = 0.7;
        let out_asym = run_tone(TubeNode::new(asym, SR, 1), f0, 0.8, frames);
        let h2_asym = goertzel(&out_asym, 2.0 * f0);

        assert!(
            h2_asym > 4.0 * h2_sym.max(1.0e-9),
            "asymmetry did not raise the second harmonic: sym={h2_sym}, asym={h2_asym}"
        );
    }

    #[test]
    fn miller_rolloff_attenuates_highs() {
        // A low Miller corner should attenuate a high tone more than a high
        // Miller corner does. Use a small amplitude to stay near-linear.
        let f = 12_000.0;
        let frames = 8192;
        let amp = 0.05;

        let mut low = params();
        low.miller_hz = 2_000.0;
        let out_low = run_tone(TubeNode::new(low, SR, 1), f, amp, frames);
        let mag_low = goertzel(&out_low, f);

        let mut high = params();
        high.miller_hz = 18_000.0;
        let out_high = run_tone(TubeNode::new(high, SR, 1), f, amp, frames);
        let mag_high = goertzel(&out_high, f);

        assert!(
            mag_high > 1.5 * mag_low,
            "Miller roll-off did not attenuate highs: low_corner={mag_low}, high_corner={mag_high}"
        );
    }

    #[test]
    fn cathode_bypass_shelf_lifts_lows() {
        // A large cathode-bypass boost should raise a sub-corner tone relative
        // to a flat (0 dB) shelf. Small amplitude keeps the stage near-linear.
        let f = 60.0;
        let frames = 16384;
        let amp = 0.05;

        let mut flat = params();
        flat.cathode_bypass_db = 0.0;
        let out_flat = run_tone(TubeNode::new(flat, SR, 1), f, amp, frames);
        let mag_flat = goertzel(&out_flat, f);

        let mut boost = params();
        boost.cathode_bypass_db = 12.0;
        boost.cathode_bypass_hz = 120.0;
        let out_boost = run_tone(TubeNode::new(boost, SR, 1), f, amp, frames);
        let mag_boost = goertzel(&out_boost, f);

        assert!(
            mag_boost > 1.5 * mag_flat,
            "cathode shelf did not lift lows: flat={mag_flat}, boost={mag_boost}"
        );
    }

    #[test]
    fn stereo_is_phase_coherent() {
        let mut node = TubeNode::new(params(), SR, 2);
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
        let mut node = TubeNode::new(params(), SR, 1);
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
        let amp = 0.05;

        let out_unity = run_tone(TubeNode::new(params(), SR, 1), f, amp, frames);
        let mag_unity = goertzel(&out_unity, f);

        let mut hot = params();
        hot.output_trim_db = 6.0;
        let out_hot = run_tone(TubeNode::new(hot, SR, 1), f, amp, frames);
        let mag_hot = goertzel(&out_hot, f);

        let ratio = mag_hot / mag_unity.max(1.0e-9);
        assert!(
            (ratio - 1.995).abs() < 0.2,
            "+6 dB trim did not double the level: ratio={ratio}"
        );
    }

    #[test]
    fn vec_collect_does_not_panic() {
        let out = run_tone(TubeNode::new(params(), SR, 1), 100.0, 0.5, 1024);
        assert_eq!(out.len(), 512);
    }
}
