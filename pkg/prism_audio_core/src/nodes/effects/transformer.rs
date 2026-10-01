//! Audio transformer (iron-core / output-transformer) emulation: frequency
//! dependent magnetic-core saturation, asymmetric even-harmonic colour,
//! winding / leakage high-frequency resonance, and a `DC`-blocking output.
//!
//! A real audio transformer couples its primary and secondary windings through
//! a magnetic core, and that core colours the signal in several interlocking,
//! well-understood ways:
//!
//! 1. The core material saturates. Because the flux produced for a given signal
//!    voltage is inversely proportional to frequency, **low frequencies drive
//!    the core into saturation sooner than high frequencies** -- a transformer
//!    distorts bass before treble. The saturation curve is slightly
//!    asymmetric, so it generates both even and odd harmonics.
//! 2. The windings and the leakage inductance together with stray capacitance
//!    form a lightly damped resonance in the top octave, adding a small
//!    high-frequency peak ("winding ring").
//! 3. A transformer cannot pass a static (`DC`) voltage across its windings, so
//!    the transferred signal is intrinsically high-pass at the very bottom.
//!
//! This module reproduces all three with plain, classic `DSP`. The chain for
//! each channel is a frequency-weighted saturator built from four
//! second-order sections plus one memoryless non-linearity:
//!
//! ```text
//! pre   = low_shelf(+emphasis_db) applied to x   // lift lows into the core
//! core  = asym_tanh(drive * pre + bias)          // frequency-weighted saturation
//! post  = low_shelf(-emphasis_db) applied to core // restore the tonal balance
//! wind  = peaking(+winding_db @ winding_hz)       // winding / leakage ring
//! out   = high_pass(dc_block_hz) applied to wind  // transformer passes no DC
//! y     = (1 - mix) * x + mix * (trim * out)
//! ```
//!
//! Lifting the lows with a low shelf *before* the saturator and cutting them
//! back with the inverse shelf *after* it is what makes the saturation
//! frequency dependent: a low tone reaches the non-linearity with more level
//! than a high tone of the same amplitude, so it clips harder and produces more
//! harmonics, exactly as an iron core does. An optional small hysteresis term
//! feeds a fraction of the previous core output back into its input, giving the
//! transfer curve a short memory (magnetic hysteresis) without destabilising
//! the bounded `tanh`.
//!
//! The asymmetric saturator is a drive / bias `tanh` normalised to unity
//! small-signal gain and corrected to pass through the origin:
//!
//! ```text
//! asym_tanh(pre) = (tanh(drive * pre + bias) - tanh(bias))
//!                / (drive * (1 - tanh(bias)^2))
//! ```
//!
//! A non-zero `bias` makes the curve asymmetric, which is what produces the
//! even-harmonic component characteristic of transformer iron.
//!
//! # Provenance
//!
//! Transformer core saturation, its frequency dependence, winding resonance,
//! and the `DC`-blocking nature of a transformer are long-standing, publicly
//! documented phenomena described in standard audio-electronics and `DSP`
//! literature (for example Udo Zolzer, "DAFX: Digital Audio Effects", and
//! classic magnetics texts). The shelving / peaking / high-pass sections are
//! the canonical Robert Bristow-Johnson (`RBJ`) cookbook biquads, and the
//! saturating `tanh` transfer curve is standard public `DSP`. Everything here
//! is re-derived from that public mathematics. This file contains **no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, or
//! Web Audio source or derived code**, and uses **no machine-learning or AI
//! techniques** -- it is purely classic signal processing.
//!
//! # Relationship
//!
//! `transformer` *reuses* this crate's own coefficient designer
//! [`BiquadCoeffs::design`](crate::nodes::biquad::BiquadCoeffs::design) to build
//! its two low shelves, its peaking winding resonance, and its `DC`-blocking
//! high pass rather than restating the cookbook formulas. It cannot, however,
//! reuse [`Biquad::process_inplace`](crate::nodes::biquad::Biquad::process_inplace):
//! that path filters a whole block, whereas a transformer must interleave the
//! shelves with the non-linearity *per sample* (shelf, then saturate, then the
//! remaining sections). It therefore keeps private Direct Form I (`DF1`)
//! per-channel state and steps one sample at a time through the shared
//! coefficients.
//!
//! It is distinct from the other saturating nodes in this family:
//!
//! - [`saturation`](super::saturation) is a *memoryless* fixed-curve shaper
//!   (symmetric or statically biased) with no frequency weighting.
//! - [`tape`](super::tape) models magnetic *tape*: a memoryless `tanh` plus
//!   wow / flutter and a one-pole treble roll-off; it has no frequency
//!   dependent core saturation, winding resonance, or `DC` block.
//! - [`exciter`](super::exciter) adds synthetic high harmonics above a split
//!   frequency, the opposite spectral emphasis to a transformer.
//! - [`wavefolder`](super::wavefolder) reflects the signal past a fold
//!   threshold; it is not a saturating magnetic model at all.
//!
//! The transformer's defining features -- frequency-weighted (bass-first)
//! saturation, asymmetric even-harmonic colour, a winding resonance peak, and
//! a `DC` block -- are not provided together by any of those nodes.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};
use crate::nodes::biquad::{BiquadCoeffs, BiquadKind};
use crate::param::{Ramp, Smoothed};

/// Largest saturation drive, so a very hot input cannot push the
/// normalisation into a numerically pathological region.
pub const MAX_DRIVE: Sample = 64.0;

/// Largest absolute bias allowed, keeping the normalising term
/// `1 - tanh(bias)^2` safely away from zero.
pub const MAX_BIAS: Sample = 3.0;

/// Largest low-shelf emphasis magnitude in decibels. Bounds how aggressively
/// the lows can be lifted into the core and restored afterwards.
pub const MAX_EMPHASIS_DB: Sample = 36.0;

/// Largest winding-resonance peak gain in decibels.
pub const MAX_WINDING_DB: Sample = 24.0;

/// Largest winding-resonance quality factor.
pub const MAX_WINDING_Q: Sample = 16.0;

/// Largest hysteresis feedback depth. Kept below unity so the bounded `tanh`
/// core stays stable with its one-sample memory.
pub const MAX_HYSTERESIS: Sample = 0.9;

/// Largest absolute output trim in decibels.
pub const MAX_TRIM_DB: Sample = 24.0;

/// Lowest allowed `DC`-block corner in Hz.
pub const MIN_DC_BLOCK_HZ: Sample = 1.0;

/// Highest allowed `DC`-block corner in Hz. A transformer's low rolloff sits a
/// few Hz up, well below the audible band of interest.
pub const MAX_DC_BLOCK_HZ: Sample = 60.0;

/// Clamps a drive request to a safe, finite, non-negative range.
#[inline]
#[must_use]
fn clamp_drive(drive: Sample) -> Sample {
    if drive.is_finite() {
        drive.clamp(0.0, MAX_DRIVE)
    } else {
        0.0
    }
}

/// Clamps a bias request to a safe, finite range.
#[inline]
#[must_use]
fn clamp_bias(bias: Sample) -> Sample {
    if bias.is_finite() {
        bias.clamp(-MAX_BIAS, MAX_BIAS)
    } else {
        0.0
    }
}

/// Clamps a low-shelf emphasis request (decibels) to a safe, finite range.
#[inline]
#[must_use]
fn clamp_emphasis_db(db: Sample) -> Sample {
    if db.is_finite() {
        db.clamp(0.0, MAX_EMPHASIS_DB)
    } else {
        0.0
    }
}

/// Clamps a shelf / resonance corner frequency to an audible, finite range.
#[inline]
#[must_use]
fn clamp_freq(freq: Sample, lo: Sample, hi: Sample) -> Sample {
    if freq.is_finite() {
        freq.clamp(lo, hi)
    } else {
        lo
    }
}

/// Clamps the winding-resonance gain (decibels) to a safe, finite range.
#[inline]
#[must_use]
fn clamp_winding_db(db: Sample) -> Sample {
    if db.is_finite() {
        db.clamp(0.0, MAX_WINDING_DB)
    } else {
        0.0
    }
}

/// Clamps the winding-resonance `Q` to a safe, finite range.
#[inline]
#[must_use]
fn clamp_winding_q(q: Sample) -> Sample {
    if q.is_finite() {
        q.clamp(0.1, MAX_WINDING_Q)
    } else {
        0.707
    }
}

/// Clamps the hysteresis depth to the stable, finite `[0, MAX_HYSTERESIS]`.
#[inline]
#[must_use]
fn clamp_hysteresis(h: Sample) -> Sample {
    if h.is_finite() {
        h.clamp(0.0, MAX_HYSTERESIS)
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

/// Computes the small-signal normalising gain `drive * (1 - tanh(bias)^2)`,
/// guarded so a near-zero drive cannot divide the saturator by zero.
#[inline]
#[must_use]
fn normalising_gain(drive: Sample, tanh_bias: Sample) -> Sample {
    let g = drive * (1.0 - tanh_bias * tanh_bias);
    if g.abs() < 1.0e-6 { 1.0 } else { g }
}

/// A single Direct Form I (`DF1`) biquad state cell: `[x1, x2, y1, y2]`.
///
/// The transformer keeps one of these per channel for each of its four
/// sections and steps them a sample at a time so the non-linearity can be
/// interleaved between the shelves.
#[derive(Debug, Clone, Copy, Default)]
struct Df1 {
    x1: Sample,
    x2: Sample,
    y1: Sample,
    y2: Sample,
}

impl Df1 {
    /// Advances the section by one sample through `coeffs` and returns the
    /// (denormal-flushed) output.
    #[inline]
    fn step(&mut self, coeffs: &BiquadCoeffs, x0: Sample) -> Sample {
        let y0 = coeffs.b0 * x0 + coeffs.b1 * self.x1 + coeffs.b2 * self.x2
            - coeffs.a1 * self.y1
            - coeffs.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = flush_denormal(y0);
        self.y1
    }

    /// Clears the stored history.
    #[inline]
    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Parameters controlling the transformer colour.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TransformerParams {
    /// Core saturation drive (how hard the signal is pushed into the `tanh`).
    pub drive: Sample,
    /// Asymmetry bias; non-zero values add the even harmonics characteristic
    /// of a transformer core.
    pub bias: Sample,
    /// Low-shelf emphasis in decibels applied before the core (and inverted
    /// after it). Larger values make the saturation more strongly bass-first.
    pub emphasis_db: Sample,
    /// Corner frequency in Hz of the low-shelf emphasis / de-emphasis pair.
    pub emphasis_hz: Sample,
    /// Hysteresis feedback depth in `[0, 1]`: a fraction of the previous core
    /// output fed back into its input to model magnetic memory.
    pub hysteresis: Sample,
    /// Centre frequency in Hz of the winding / leakage resonance peak.
    pub winding_hz: Sample,
    /// Quality factor of the winding resonance.
    pub winding_q: Sample,
    /// Gain in decibels of the winding resonance peak.
    pub winding_db: Sample,
    /// Corner frequency in Hz of the `DC`-blocking output high pass.
    pub dc_block_hz: Sample,
    /// Output trim in decibels applied to the wet path.
    pub output_trim_db: Sample,
    /// Wet / dry blend in `[0, 1]`: `0` is the untouched input, `1` is the
    /// fully processed transformer.
    pub mix: Sample,
}

impl Default for TransformerParams {
    fn default() -> Self {
        Self {
            drive: 2.0,
            bias: 0.2,
            emphasis_db: 9.0,
            emphasis_hz: 150.0,
            hysteresis: 0.2,
            winding_hz: 8_000.0,
            winding_q: 2.0,
            winding_db: 4.0,
            dc_block_hz: 10.0,
            output_trim_db: 0.0,
            mix: 1.0,
        }
    }
}

/// Allocation-free transformer `DSP` core: a frequency-weighted asymmetric
/// saturator (low-shelf lift, biased `tanh`, inverse low shelf), a winding
/// resonance peak, and a `DC`-blocking high pass.
///
/// All state is pre-allocated at construction, so [`Transformer::voice`] is
/// real-time safe (no allocation, no locking, no panic). The owning
/// [`TransformerNode`] performs the dry blend and output trim.
#[derive(Debug, Clone)]
pub struct Transformer {
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Low-shelf (+emphasis) coefficients applied before the core.
    emphasis_boost: BiquadCoeffs,
    /// Low-shelf (-emphasis) coefficients applied after the core.
    emphasis_cut: BiquadCoeffs,
    /// Peaking coefficients for the winding resonance.
    winding: BiquadCoeffs,
    /// High-pass coefficients for the `DC` block.
    dc_block: BiquadCoeffs,
    /// Per-channel emphasis-boost section state.
    boost_state: Vec<Df1>,
    /// Per-channel emphasis-cut section state.
    cut_state: Vec<Df1>,
    /// Per-channel winding-resonance section state.
    winding_state: Vec<Df1>,
    /// Per-channel `DC`-block section state.
    dc_state: Vec<Df1>,
    /// Per-channel one-sample hysteresis memory (previous core output).
    hysteresis_state: Vec<Sample>,
    /// Saturation drive (clamped to a safe finite range).
    drive: Sample,
    /// Saturation bias (clamped to a safe finite range).
    bias: Sample,
    /// Precomputed `tanh(bias)` for the origin-correction term.
    tanh_bias: Sample,
    /// Precomputed normalising gain `drive * (1 - tanh(bias)^2)` (guarded).
    norm: Sample,
    /// Hysteresis feedback depth (clamped to `[0, MAX_HYSTERESIS]`).
    hysteresis: Sample,
    /// Low-shelf emphasis in decibels (clamped).
    emphasis_db: Sample,
    /// Low-shelf corner frequency in Hz (clamped).
    emphasis_hz: Sample,
    /// Winding-resonance centre frequency in Hz (clamped).
    winding_hz: Sample,
    /// Winding-resonance quality factor (clamped).
    winding_q: Sample,
    /// Winding-resonance gain in decibels (clamped).
    winding_db: Sample,
    /// `DC`-block corner frequency in Hz (clamped).
    dc_block_hz: Sample,
}

impl Transformer {
    /// Builds a transformer core for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: TransformerParams, sample_rate: u32, channels: usize) -> Self {
        let sr = sample_rate.max(1);
        let channels = channels.max(1);

        let drive = clamp_drive(params.drive);
        let bias = clamp_bias(params.bias);
        let tanh_bias = ops::tanh(bias);
        let emphasis_db = clamp_emphasis_db(params.emphasis_db);
        let emphasis_hz = clamp_freq(params.emphasis_hz, 20.0, 2_000.0);
        let winding_hz = clamp_freq(params.winding_hz, 1_000.0, 20_000.0);
        let winding_q = clamp_winding_q(params.winding_q);
        let winding_db = clamp_winding_db(params.winding_db);
        let dc_block_hz = clamp_freq(params.dc_block_hz, MIN_DC_BLOCK_HZ, MAX_DC_BLOCK_HZ);

        let emphasis_boost =
            BiquadCoeffs::design(BiquadKind::LowShelf, sr, emphasis_hz, 0.707, emphasis_db);
        let emphasis_cut =
            BiquadCoeffs::design(BiquadKind::LowShelf, sr, emphasis_hz, 0.707, -emphasis_db);
        let winding =
            BiquadCoeffs::design(BiquadKind::Peaking, sr, winding_hz, winding_q, winding_db);
        let dc_block = BiquadCoeffs::design(BiquadKind::HighPass, sr, dc_block_hz, 0.707, 0.0);

        Self {
            sample_rate: sr,
            emphasis_boost,
            emphasis_cut,
            winding,
            dc_block,
            boost_state: vec![Df1::default(); channels],
            cut_state: vec![Df1::default(); channels],
            winding_state: vec![Df1::default(); channels],
            dc_state: vec![Df1::default(); channels],
            hysteresis_state: vec![0.0; channels],
            drive,
            bias,
            tanh_bias,
            norm: normalising_gain(drive, tanh_bias),
            hysteresis: clamp_hysteresis(params.hysteresis),
            emphasis_db,
            emphasis_hz,
            winding_hz,
            winding_q,
            winding_db,
            dc_block_hz,
        }
    }

    /// Number of channels the core tracks.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.boost_state.len()
    }

    /// Current saturation drive.
    #[inline]
    #[must_use]
    pub fn drive(&self) -> Sample {
        self.drive
    }

    /// Current saturation bias.
    #[inline]
    #[must_use]
    pub fn bias(&self) -> Sample {
        self.bias
    }

    /// Current hysteresis depth.
    #[inline]
    #[must_use]
    pub fn hysteresis(&self) -> Sample {
        self.hysteresis
    }

    /// Current low-shelf emphasis in decibels.
    #[inline]
    #[must_use]
    pub fn emphasis_db(&self) -> Sample {
        self.emphasis_db
    }

    /// The applied asymmetric transfer curve for a (shelf-weighted) input.
    #[inline]
    #[must_use]
    fn asym_tanh(&self, pre: Sample) -> Sample {
        (ops::tanh(self.drive * pre + self.bias) - self.tanh_bias) / self.norm
    }

    /// Rebuilds the emphasis-boost / emphasis-cut shelf pair.
    #[inline]
    fn redesign_emphasis(&mut self) {
        self.emphasis_boost = BiquadCoeffs::design(
            BiquadKind::LowShelf,
            self.sample_rate,
            self.emphasis_hz,
            0.707,
            self.emphasis_db,
        );
        self.emphasis_cut = BiquadCoeffs::design(
            BiquadKind::LowShelf,
            self.sample_rate,
            self.emphasis_hz,
            0.707,
            -self.emphasis_db,
        );
    }

    /// Rebuilds the winding-resonance peaking section.
    #[inline]
    fn redesign_winding(&mut self) {
        self.winding = BiquadCoeffs::design(
            BiquadKind::Peaking,
            self.sample_rate,
            self.winding_hz,
            self.winding_q,
            self.winding_db,
        );
    }

    /// Rebuilds the `DC`-block high pass.
    #[inline]
    fn redesign_dc_block(&mut self) {
        self.dc_block = BiquadCoeffs::design(
            BiquadKind::HighPass,
            self.sample_rate,
            self.dc_block_hz,
            0.707,
            0.0,
        );
    }

    /// Sets the saturation drive (clamped to a safe finite range).
    #[inline]
    pub fn set_drive(&mut self, drive: Sample) {
        self.drive = clamp_drive(drive);
        self.norm = normalising_gain(self.drive, self.tanh_bias);
    }

    /// Sets the saturation bias (clamped to a safe finite range).
    #[inline]
    pub fn set_bias(&mut self, bias: Sample) {
        self.bias = clamp_bias(bias);
        self.tanh_bias = ops::tanh(self.bias);
        self.norm = normalising_gain(self.drive, self.tanh_bias);
    }

    /// Sets the hysteresis feedback depth (clamped to `[0, MAX_HYSTERESIS]`).
    #[inline]
    pub fn set_hysteresis(&mut self, hysteresis: Sample) {
        self.hysteresis = clamp_hysteresis(hysteresis);
    }

    /// Sets the low-shelf emphasis in decibels and rebuilds the shelf pair.
    #[inline]
    pub fn set_emphasis_db(&mut self, emphasis_db: Sample) {
        self.emphasis_db = clamp_emphasis_db(emphasis_db);
        self.redesign_emphasis();
    }

    /// Sets the low-shelf corner frequency in Hz and rebuilds the shelf pair.
    #[inline]
    pub fn set_emphasis_hz(&mut self, emphasis_hz: Sample) {
        self.emphasis_hz = clamp_freq(emphasis_hz, 20.0, 2_000.0);
        self.redesign_emphasis();
    }

    /// Sets the winding-resonance centre frequency in Hz.
    #[inline]
    pub fn set_winding_hz(&mut self, winding_hz: Sample) {
        self.winding_hz = clamp_freq(winding_hz, 1_000.0, 20_000.0);
        self.redesign_winding();
    }

    /// Sets the winding-resonance quality factor.
    #[inline]
    pub fn set_winding_q(&mut self, winding_q: Sample) {
        self.winding_q = clamp_winding_q(winding_q);
        self.redesign_winding();
    }

    /// Sets the winding-resonance gain in decibels.
    #[inline]
    pub fn set_winding_db(&mut self, winding_db: Sample) {
        self.winding_db = clamp_winding_db(winding_db);
        self.redesign_winding();
    }

    /// Sets the `DC`-block corner frequency in Hz.
    #[inline]
    pub fn set_dc_block_hz(&mut self, dc_block_hz: Sample) {
        self.dc_block_hz = clamp_freq(dc_block_hz, MIN_DC_BLOCK_HZ, MAX_DC_BLOCK_HZ);
        self.redesign_dc_block();
    }

    /// Processes one input sample for channel `ch` through the full chain and
    /// returns the wet output (before any dry blend or output trim).
    #[inline]
    #[must_use]
    pub fn voice(&mut self, ch: usize, x: Sample) -> Sample {
        let fed = x + self.hysteresis * self.hysteresis_state[ch];
        let pre = self.boost_state[ch].step(&self.emphasis_boost, fed);
        let core = self.asym_tanh(pre);
        self.hysteresis_state[ch] = flush_denormal(core);
        let post = self.cut_state[ch].step(&self.emphasis_cut, core);
        let wind = self.winding_state[ch].step(&self.winding, post);
        self.dc_state[ch].step(&self.dc_block, wind)
    }

    /// Clears all filter state and hysteresis memory.
    #[inline]
    pub fn reset(&mut self) {
        for s in &mut self.boost_state {
            s.reset();
        }
        for s in &mut self.cut_state {
            s.reset();
        }
        for s in &mut self.winding_state {
            s.reset();
        }
        for s in &mut self.dc_state {
            s.reset();
        }
        for s in &mut self.hysteresis_state {
            *s = 0.0;
        }
    }
}

/// A transformer-emulation node (input port 0 -> output port 0).
///
/// The dry signal is blended with the processed transformer path by a
/// [`Smoothed`] `mix` so automation stays click-free, and the wet path is
/// scaled by a linear output trim. Every channel is processed independently
/// from shared coefficients, so a mono source fed to several channels stays
/// phase-coherent.
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{TransformerNode, TransformerParams};
///
/// let mut node = TransformerNode::new(TransformerParams::default(), 48_000, 1);
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, 8);
/// input.set_active_frames(8);
/// output.set_active_frames(8);
/// for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
///     *s = if i % 2 == 0 { 0.5 } else { -0.5 };
/// }
/// let ctx = RenderContext { sample_rate: 48_000, frames: 8, playhead: 0 };
/// let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
/// node.process(&ctx, &mut io);
/// assert!(output.channel(0).iter().all(|s| s.is_finite()));
/// ```
#[derive(Debug, Clone)]
pub struct TransformerNode {
    /// `DSP` core.
    core: Transformer,
    /// Smoothed wet/dry blend in `[0, 1]`.
    mix: Smoothed,
    /// Linear output trim applied to the wet path.
    trim: Sample,
}

impl TransformerNode {
    /// Builds a transformer node for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: TransformerParams, sample_rate: u32, channels: usize) -> Self {
        Self {
            mix: Smoothed::new(params.mix.clamp(0.0, 1.0)),
            trim: db_to_linear(clamp_trim_db(params.output_trim_db)),
            core: Transformer::new(params, sample_rate, channels),
        }
    }

    /// Number of channels the node processes.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.core.channels()
    }

    /// Sets the saturation drive.
    #[inline]
    pub fn set_drive(&mut self, drive: Sample) {
        self.core.set_drive(drive);
    }

    /// Sets the saturation bias.
    #[inline]
    pub fn set_bias(&mut self, bias: Sample) {
        self.core.set_bias(bias);
    }

    /// Sets the hysteresis feedback depth.
    #[inline]
    pub fn set_hysteresis(&mut self, hysteresis: Sample) {
        self.core.set_hysteresis(hysteresis);
    }

    /// Sets the low-shelf emphasis in decibels.
    #[inline]
    pub fn set_emphasis_db(&mut self, emphasis_db: Sample) {
        self.core.set_emphasis_db(emphasis_db);
    }

    /// Sets the low-shelf corner frequency in Hz.
    #[inline]
    pub fn set_emphasis_hz(&mut self, emphasis_hz: Sample) {
        self.core.set_emphasis_hz(emphasis_hz);
    }

    /// Sets the winding-resonance centre frequency in Hz.
    #[inline]
    pub fn set_winding_hz(&mut self, winding_hz: Sample) {
        self.core.set_winding_hz(winding_hz);
    }

    /// Sets the winding-resonance quality factor.
    #[inline]
    pub fn set_winding_q(&mut self, winding_q: Sample) {
        self.core.set_winding_q(winding_q);
    }

    /// Sets the winding-resonance gain in decibels.
    #[inline]
    pub fn set_winding_db(&mut self, winding_db: Sample) {
        self.core.set_winding_db(winding_db);
    }

    /// Sets the `DC`-block corner frequency in Hz.
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

impl AudioNode for TransformerNode {
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
    use alloc::vec::Vec;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    fn params() -> TransformerParams {
        TransformerParams::default()
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
    fn run_tone(mut node: TransformerNode, freq_hz: Sample, amp: Sample, frames: usize) -> Vec<Sample> {
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
        let node = TransformerNode::new(params(), SR, 1);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = TransformerNode::new(params(), SR, 1);
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
        let out = run_tone(TransformerNode::new(params(), SR, 1), 440.0, 0.8, 2048);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut node = TransformerNode::new(params(), SR, 1);
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
        let mut node = TransformerNode::new(params(), SR, 1);
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
        let mut node = TransformerNode::new(p, SR, 1);
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
        p.bias = 1.0e9;
        p.emphasis_db = 1.0e9;
        p.emphasis_hz = 1.0e9;
        p.hysteresis = 1.0e9;
        p.winding_hz = 1.0e9;
        p.winding_q = 1.0e9;
        p.winding_db = 1.0e9;
        p.dc_block_hz = 1.0e9;
        p.output_trim_db = 1.0e9;
        p.mix = 1.0e9;
        let mut node = TransformerNode::new(p, SR, 1);
        let mut input = mono(256);
        sine(&mut input, 500.0, 1.0);
        let inputs = [input];
        let mut outputs = [mono(256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        for &y in outputs[0].channel(0) {
            assert!(y.is_finite(), "extreme params leaked non-finite: {y}");
        }
    }

    #[test]
    fn non_finite_params_fall_back_to_safe_defaults() {
        let mut p = params();
        p.drive = Sample::NAN;
        p.bias = Sample::INFINITY;
        p.emphasis_db = Sample::NAN;
        p.hysteresis = Sample::NAN;
        let core = Transformer::new(p, SR, 1);
        assert_eq!(core.drive(), 0.0);
        assert_eq!(core.bias(), 0.0);
        assert_eq!(core.emphasis_db(), 0.0);
        assert_eq!(core.hysteresis(), 0.0);
    }

    #[test]
    fn setters_clamp_and_reject_non_finite() {
        let mut core = Transformer::new(params(), SR, 1);
        core.set_drive(1.0e9);
        assert_eq!(core.drive(), MAX_DRIVE);
        core.set_drive(Sample::NAN);
        assert_eq!(core.drive(), 0.0);
        core.set_bias(1.0e9);
        assert_eq!(core.bias(), MAX_BIAS);
        core.set_hysteresis(1.0e9);
        assert_eq!(core.hysteresis(), MAX_HYSTERESIS);
        core.set_hysteresis(Sample::NAN);
        assert_eq!(core.hysteresis(), 0.0);
        core.set_emphasis_db(1.0e9);
        assert_eq!(core.emphasis_db(), MAX_EMPHASIS_DB);
    }

    #[test]
    fn dc_input_is_blocked() {
        // A constant input must not survive the DC-blocking output: the mean of
        // the settled tail trends to zero.
        let mut node = TransformerNode::new(params(), SR, 1);
        let frames = 8192;
        let mut input = mono(frames);
        for s in input.channel_mut(0).iter_mut() {
            *s = 0.5;
        }
        let inputs = [input];
        let mut outputs = [mono(frames)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let tail = &outputs[0].channel(0)[frames / 2..];
        let mean = tail.iter().sum::<Sample>() / tail.len() as Sample;
        assert!(mean.abs() < 1.0e-3, "DC component not blocked: mean = {mean}");
    }

    #[test]
    fn low_frequencies_saturate_before_high_frequencies() {
        // Equal-amplitude tones: the low tone is pushed harder into the core by
        // the pre-emphasis shelf, so its total harmonic energy relative to its
        // own fundamental exceeds the high tone's.
        let frames = 16_384;
        let amp = 0.9;

        let low = 80.0;
        let low_tail = run_tone(TransformerNode::new(params(), SR, 1), low, amp, frames);
        let low_fund = goertzel(&low_tail, low);
        let low_harm = goertzel(&low_tail, 2.0 * low) + goertzel(&low_tail, 3.0 * low);
        let low_ratio = low_harm / low_fund.max(1.0e-9);

        let high = 2_000.0;
        let high_tail = run_tone(TransformerNode::new(params(), SR, 1), high, amp, frames);
        let high_fund = goertzel(&high_tail, high);
        let high_harm = goertzel(&high_tail, 2.0 * high) + goertzel(&high_tail, 3.0 * high);
        let high_ratio = high_harm / high_fund.max(1.0e-9);

        assert!(
            low_ratio > high_ratio * 2.0,
            "low tone should saturate harder: low {low_ratio} vs high {high_ratio}"
        );
    }

    #[test]
    fn bias_generates_even_harmonics() {
        // A symmetric (zero-bias) core produces mostly odd harmonics; adding
        // bias injects a measurable second harmonic.
        let frames = 16_384;
        let amp = 0.9;
        let freq = 300.0;

        let mut sym = params();
        sym.bias = 0.0;
        let sym_tail = run_tone(TransformerNode::new(sym, SR, 1), freq, amp, frames);
        let sym_second = goertzel(&sym_tail, 2.0 * freq);

        let mut asym = params();
        asym.bias = 1.0;
        let asym_tail = run_tone(TransformerNode::new(asym, SR, 1), freq, amp, frames);
        let asym_second = goertzel(&asym_tail, 2.0 * freq);

        assert!(
            asym_second > sym_second * 4.0,
            "bias should add even harmonics: biased {asym_second} vs symmetric {sym_second}"
        );
    }

    #[test]
    fn winding_resonance_lifts_its_band() {
        // Raising the winding peak gain increases the output level of a tone at
        // the resonance centre.
        let frames = 8192;
        let amp = 0.2;
        let freq = 8_000.0;

        let mut flat = params();
        flat.winding_db = 0.0;
        let flat_tail = run_tone(TransformerNode::new(flat, SR, 1), freq, amp, frames);
        let flat_level = goertzel(&flat_tail, freq);

        let mut peaked = params();
        peaked.winding_db = 12.0;
        let peaked_tail = run_tone(TransformerNode::new(peaked, SR, 1), freq, amp, frames);
        let peaked_level = goertzel(&peaked_tail, freq);

        assert!(
            peaked_level > flat_level * 1.5,
            "winding resonance should lift its band: peaked {peaked_level} vs flat {flat_level}"
        );
    }

    #[test]
    fn stereo_channels_are_phase_coherent_for_identical_input() {
        let mut node = TransformerNode::new(params(), SR, 2);
        let mut input = stereo(1024);
        sine(&mut input, 220.0, 0.6);
        let inputs = [input];
        let mut outputs = [stereo(1024)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(1024), &mut io);
        assert_eq!(outputs[0].channel(0), outputs[0].channel(1));
    }

    #[test]
    fn node_reset_clears_tail() {
        let mut node = TransformerNode::new(params(), SR, 1);
        let mut input = mono(512);
        sine(&mut input, 440.0, 0.8);
        let inputs = [input];
        {
            let mut outputs = [mono(512)];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(512), &mut io);
        }
        node.reset();
        let silent = [mono(512)];
        let mut after = [mono(512)];
        let mut io = ProcessIo::new(&silent, &mut after);
        node.process(&ctx(512), &mut io);
        for &y in after[0].channel(0) {
            assert!(y.abs() < 1.0e-6, "tail not cleared: {y}");
        }
    }

    #[test]
    fn output_trim_scales_the_wet_path() {
        let frames = 4096;
        let amp = 0.1;
        let freq = 500.0;

        let unity_tail = run_tone(TransformerNode::new(params(), SR, 1), freq, amp, frames);
        let unity_level = goertzel(&unity_tail, freq);

        let mut p = params();
        p.output_trim_db = 6.0;
        let hot_tail = run_tone(TransformerNode::new(p, SR, 1), freq, amp, frames);
        let hot_level = goertzel(&hot_tail, freq);

        let ratio = hot_level / unity_level.max(1.0e-9);
        assert!(
            (ratio - 2.0).abs() < 0.2,
            "+6 dB trim should roughly double the level: ratio = {ratio}"
        );
    }

    #[test]
    fn collects_into_vec_without_panic() {
        let mut core = Transformer::new(params(), SR, 1);
        let mut out: Vec<Sample> = Vec::with_capacity(256);
        for i in 0..256 {
            let phase = core::f32::consts::TAU * 600.0 * (i as Sample) / (SR as Sample);
            out.push(core.voice(0, 0.5 * ops::sin(phase)));
        }
        assert!(out.iter().all(|s| s.is_finite()));
    }
}
