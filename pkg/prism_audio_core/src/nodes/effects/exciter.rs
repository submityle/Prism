//! Harmonic exciter / aural enhancer: synthesises high-frequency harmonics
//! from a band-limited copy of the signal and blends them back to add
//! "air", presence, and perceived detail without a static shelf boost.
//!
//! An exciter does not simply boost the treble. It isolates the upper band of
//! the signal, drives it through a gentle nonlinearity so new harmonics appear
//! *above* the original content, keeps only those new high harmonics, and mixes
//! that sparkle back onto the dry signal. Because the added energy is harmonic
//! (musically related to what is already there) rather than a flat gain, the
//! result reads as extra clarity and brightness that survives on small speakers.
//!
//! The band split uses two topology-preserving high-pass
//! [`Svf`](super::super::svf::Svf) stages (one before and one after the
//! waveshaper), so the crossover stays phase-coherent and well behaved while
//! the cutoff is automated. The waveshaper itself is a smooth `tanh` family with
//! a selectable even / odd / mixed harmonic character.
//!
//! # Signal model
//!
//! For each channel the per-sample chain is:
//!
//! ```text
//! band      = highpass(x, fc)         // isolate the upper band
//! shaped    = shape(drive * band)     // generate harmonics
//! harmonics = highpass(shaped, fc)    // keep only the new highs
//! y         = x + amount * harmonics  // blend sparkle onto the dry signal
//! ```
//!
//! The nonlinearity is chosen by [`HarmonicMode`]:
//!
//! - [`HarmonicMode::Odd`] uses the symmetric `tanh`, which produces odd
//!   harmonics (a smoother, valve-like brightness).
//! - [`HarmonicMode::Even`] uses a DC-corrected biased `tanh`, which produces
//!   predominantly even harmonics (a sweeter, octave-flavoured sheen).
//! - [`HarmonicMode::Mix`] averages the two.
//!
//! # Provenance
//!
//! The aural-exciter idea (band-split, generate high harmonics, blend back) is
//! long-standing, publicly documented audio-effect knowledge associated with
//! the Aphex Aural Exciter and described in texts such as Udo Zolzer, "DAFX:
//! Digital Audio Effects". The `tanh` harmonic shaper and the biased, DC-removed
//! even-harmonic variant below are standard waveshaping mathematics re-derived
//! from that public literature and composed on top of this crate's own
//! [`Svf`](super::super::svf::Svf) crossover. This file contains **no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio
//! source or derived code**; it is implemented purely from that publicly
//! documented mathematics.
//!
//! # Relationship
//!
//! `exciter` composes an existing primitive rather than reimplementing it: both
//! band-split stages are [`Svf`](super::super::svf::Svf) high-pass filters (the
//! same TPT core wrapped by [`SvfNode`](super::super::svf::SvfNode)). It is the
//! generative counterpart to the subtractive [`parametric_eq`](super::parametric_eq):
//! where the EQ reshapes existing spectrum, the exciter adds new harmonic
//! content. The waveshaping is kept local (a smooth, bounded `tanh` family)
//! rather than reusing the oversampled [`waveshaper`](super::waveshaper), because
//! the post high-pass already discards the aliasing-prone low products.

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::nodes::svf::{Svf, SvfCoeffs, SvfKind};
use crate::param::{Ramp, Smoothed};

/// DC bias (in shaper input units) used by the even-harmonic nonlinearity.
///
/// A small asymmetry breaks the odd symmetry of `tanh` so even harmonics
/// appear; the shaper subtracts `tanh(drive * bias)` to keep the output free of
/// a static DC offset.
const EVEN_BIAS: Sample = 0.5;

/// Largest drive fed into the shaper, so very hot bands cannot push the
/// nonlinearity into a numerically pathological region.
const MAX_DRIVE: Sample = 64.0;

/// The harmonic character synthesised by the exciter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum HarmonicMode {
    /// Symmetric `tanh`: odd harmonics, a smooth valve-like brightness.
    Odd,
    /// DC-corrected biased `tanh`: predominantly even harmonics, a sweeter
    /// octave-flavoured sheen.
    Even,
    /// An even average of the odd and even shapers.
    Mix,
}

impl HarmonicMode {
    /// Applies the mode's waveshaper to a (pre-driven) sample.
    ///
    /// The result is always finite and bounded in `[-2, 2]` (each `tanh` term
    /// is bounded in `[-1, 1]`).
    #[inline]
    #[must_use]
    fn shape(self, driven: Sample) -> Sample {
        match self {
            HarmonicMode::Odd => ops::tanh(driven),
            HarmonicMode::Even => biased_tanh(driven),
            HarmonicMode::Mix => 0.5 * (ops::tanh(driven) + biased_tanh(driven)),
        }
    }
}

/// DC-corrected biased `tanh` used for even-harmonic generation.
#[inline]
#[must_use]
fn biased_tanh(driven: Sample) -> Sample {
    // `EVEN_BIAS` is a fixed input-domain offset, so the whole expression is
    // already pre-driven consistently with the symmetric branch.
    ops::tanh(driven + EVEN_BIAS) - ops::tanh(EVEN_BIAS)
}

/// Construction parameters for an [`ExciterNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ExciterParams {
    /// Crossover frequency in Hz: only content above this is excited.
    pub frequency_hz: Sample,
    /// Harmonic character to synthesise.
    pub mode: HarmonicMode,
    /// Pre-shaper drive; higher generates more (and higher-order) harmonics.
    pub drive: Sample,
    /// Blend of synthesised harmonics added onto the dry signal.
    pub amount: Sample,
}

impl Default for ExciterParams {
    fn default() -> Self {
        Self {
            frequency_hz: 3_000.0,
            mode: HarmonicMode::Even,
            drive: 2.0,
            amount: 0.25,
        }
    }
}

/// Allocation-free harmonic-exciter DSP core: two high-pass [`Svf`] band-split
/// stages around a `tanh` waveshaper.
///
/// All state is pre-allocated at construction, so [`Exciter::harmonics`] is
/// real-time safe (no allocation, no locking, no panic). The core returns only
/// the synthesised harmonic sample; the owning [`ExciterNode`] performs the dry
/// blend.
#[derive(Debug, Clone)]
pub struct Exciter {
    /// Pre-shaper band-isolation high-pass (per-channel state lives here).
    pre: Svf,
    /// Post-shaper high-pass that keeps only the new high harmonics.
    post: Svf,
    /// Sample rate used to design the crossover coefficients.
    sample_rate: u32,
    /// Crossover frequency in Hz (clamped positive).
    frequency_hz: Sample,
    /// Harmonic character.
    mode: HarmonicMode,
    /// Pre-shaper drive (clamped to a safe finite range).
    drive: Sample,
}

impl Exciter {
    /// Builds an exciter core for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: ExciterParams, sample_rate: u32, channels: usize) -> Self {
        let sr = sample_rate.max(1);
        let freq = params.frequency_hz.max(1.0);
        // A gentle Butterworth-ish high-pass (Q = 1/sqrt(2)) for both stages.
        let coeffs = SvfCoeffs::design(SvfKind::HighPass, sr, freq, core::f32::consts::FRAC_1_SQRT_2, 0.0);
        let chans = channels.max(1);
        Self {
            pre: Svf::new(coeffs, chans),
            post: Svf::new(coeffs, chans),
            sample_rate: sr,
            frequency_hz: freq,
            mode: params.mode,
            drive: clamp_drive(params.drive),
        }
    }

    /// Number of channels the internal filters track.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.pre.channels()
    }

    /// Current crossover frequency in Hz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Current pre-shaper drive.
    #[inline]
    #[must_use]
    pub fn drive(&self) -> Sample {
        self.drive
    }

    /// Current harmonic mode.
    #[inline]
    #[must_use]
    pub fn mode(&self) -> HarmonicMode {
        self.mode
    }

    /// Sets the crossover frequency (Hz, clamped positive) and redesigns both
    /// band-split stages.
    #[inline]
    pub fn set_frequency_hz(&mut self, frequency_hz: Sample) {
        self.frequency_hz = frequency_hz.max(1.0);
        let coeffs = SvfCoeffs::design(
            SvfKind::HighPass,
            self.sample_rate,
            self.frequency_hz,
            core::f32::consts::FRAC_1_SQRT_2,
            0.0,
        );
        self.pre.set_coeffs(coeffs);
        self.post.set_coeffs(coeffs);
    }

    /// Sets the pre-shaper drive (clamped to a safe finite range).
    #[inline]
    pub fn set_drive(&mut self, drive: Sample) {
        self.drive = clamp_drive(drive);
    }

    /// Sets the harmonic character.
    #[inline]
    pub fn set_mode(&mut self, mode: HarmonicMode) {
        self.mode = mode;
    }

    /// Processes one sample on channel `ch` and returns only the synthesised
    /// high-harmonic content (the dry signal is not included).
    ///
    /// Non-finite input is treated as silence so the band-split integrators
    /// cannot be poisoned into a NaN / infinity state.
    #[inline]
    pub fn harmonics(&mut self, ch: usize, x: Sample) -> Sample {
        let input = if x.is_finite() { x } else { 0.0 };
        let band = self.pre.tick(ch, input);
        let shaped = self.mode.shape(self.drive * band);
        let harmonics = self.post.tick(ch, shaped);
        flush_denormal(harmonics)
    }

    /// Clears both band-split filter integrators.
    #[inline]
    pub fn reset(&mut self) {
        self.pre.reset();
        self.post.reset();
    }
}

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

/// A harmonic-exciter node (input port 0 -> output port 0).
///
/// The dry signal is passed through unchanged and the synthesised harmonics are
/// added on top, scaled by a [`Smoothed`] `amount` so automation stays
/// click-free. Every channel is excited independently from shared coefficients.
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{ExciterNode, ExciterParams};
///
/// let mut node = ExciterNode::new(ExciterParams::default(), 48_000, 1);
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
/// let mut output = AudioBuffer::new(ChannelLayout::Mono, 4);
/// input.set_active_frames(4);
/// output.set_active_frames(4);
/// for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
///     *s = if i % 2 == 0 { 0.5 } else { -0.5 };
/// }
/// let ctx = RenderContext { sample_rate: 48_000, frames: 4, playhead: 0 };
/// let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
/// node.process(&ctx, &mut io);
/// assert!(output.channel(0).iter().all(|s| s.is_finite()));
/// ```
#[derive(Debug, Clone)]
pub struct ExciterNode {
    /// DSP core (band-split + waveshaper).
    exciter: Exciter,
    /// Smoothed harmonic blend amount.
    amount: Smoothed,
}

impl ExciterNode {
    /// Builds an exciter node for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(params: ExciterParams, sample_rate: u32, channels: usize) -> Self {
        Self {
            amount: Smoothed::new(params.amount),
            exciter: Exciter::new(params, sample_rate, channels),
        }
    }

    /// Sets the crossover frequency in Hz.
    #[inline]
    pub fn set_frequency_hz(&mut self, frequency_hz: Sample) {
        self.exciter.set_frequency_hz(frequency_hz);
    }

    /// Sets the pre-shaper drive.
    #[inline]
    pub fn set_drive(&mut self, drive: Sample) {
        self.exciter.set_drive(drive);
    }

    /// Sets the harmonic character.
    #[inline]
    pub fn set_mode(&mut self, mode: HarmonicMode) {
        self.exciter.set_mode(mode);
    }

    /// Sets the harmonic blend amount with the given ramp.
    #[inline]
    pub fn set_amount(&mut self, amount: Sample, ramp: Ramp) {
        self.amount.set_target(amount, ramp);
    }

    /// Current crossover frequency in Hz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.exciter.frequency_hz()
    }
}

impl AudioNode for ExciterNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(self.exciter.channels());
        let frames = output.active_frames();

        for f in 0..frames {
            let amount = self.amount.next_sample();
            for ch in 0..channels {
                let x = input.channel(ch)[f];
                let harmonics = self.exciter.harmonics(ch, x);
                output.channel_mut(ch)[f] = x + amount * harmonics;
            }
        }
    }

    fn reset(&mut self) {
        self.exciter.reset();
        self.amount = Smoothed::new(self.amount.target());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    fn params() -> ExciterParams {
        ExciterParams::default()
    }

    fn settle(exciter: &mut Exciter, freq_hz: Sample, samples: usize) -> Vec<Sample> {
        let step = core::f32::consts::TAU * freq_hz / (SR as Sample);
        let mut out = Vec::with_capacity(samples);
        for i in 0..samples {
            let phase = step * (i as Sample);
            let x = ops::sin(phase);
            out.push(exciter.harmonics(0, x));
        }
        out
    }

    #[test]
    fn low_tone_below_crossover_is_mostly_rejected() {
        // A tone well below the 3 kHz crossover should yield little harmonic
        // energy because the band-split high-pass removes it before shaping.
        let mut exciter = Exciter::new(params(), SR, 1);
        let out = settle(&mut exciter, 100.0, 4_096);
        let tail = &out[2_048..];
        let peak = tail.iter().fold(0.0_f32, |m, &s| m.max(s.abs()));
        assert!(peak < 0.05, "low tone leaked too much: {peak}");
    }

    #[test]
    fn high_tone_above_crossover_generates_harmonics() {
        // A tone above the crossover passes the band-split and is shaped, so
        // measurable harmonic content appears.
        let mut exciter = Exciter::new(params(), SR, 1);
        let out = settle(&mut exciter, 8_000.0, 4_096);
        let tail = &out[2_048..];
        let peak = tail.iter().fold(0.0_f32, |m, &s| m.max(s.abs()));
        assert!(peak > 0.01, "high tone produced no harmonics: {peak}");
    }

    #[test]
    fn more_drive_generates_more_harmonics() {
        let mut low = Exciter::new(params(), SR, 1);
        low.set_drive(0.5);
        let mut high = Exciter::new(params(), SR, 1);
        high.set_drive(8.0);
        let low_out = settle(&mut low, 6_000.0, 4_096);
        let high_out = settle(&mut high, 6_000.0, 4_096);
        let energy = |v: &[Sample]| v[2_048..].iter().map(|&s| s * s).sum::<Sample>();
        assert!(
            energy(&high_out) > energy(&low_out),
            "more drive should add more harmonic energy"
        );
    }

    #[test]
    fn silence_in_silence_out() {
        let mut exciter = Exciter::new(params(), SR, 1);
        for _ in 0..512 {
            let y = exciter.harmonics(0, 0.0);
            assert_eq!(y, 0.0);
        }
    }

    #[test]
    fn non_finite_input_is_safe() {
        let mut exciter = Exciter::new(params(), SR, 1);
        let a = exciter.harmonics(0, Sample::NAN);
        let b = exciter.harmonics(0, Sample::INFINITY);
        let c = exciter.harmonics(0, 0.5);
        assert!(a.is_finite());
        assert!(b.is_finite());
        assert!(c.is_finite());
    }

    #[test]
    fn odd_mode_is_symmetric() {
        // The symmetric tanh shaper maps +-x to +-shape, so the shaper itself
        // is odd about zero.
        let pos = HarmonicMode::Odd.shape(0.7);
        let neg = HarmonicMode::Odd.shape(-0.7);
        assert!((pos + neg).abs() < 1e-6, "odd shaper is not symmetric");
    }

    #[test]
    fn even_mode_breaks_symmetry() {
        // The even shaper is asymmetric, so +-x do not cancel.
        let pos = HarmonicMode::Even.shape(0.7);
        let neg = HarmonicMode::Even.shape(-0.7);
        assert!((pos + neg).abs() > 1e-3, "even shaper should be asymmetric");
    }

    #[test]
    fn even_shaper_has_no_dc_at_zero() {
        // The DC correction makes the even shaper pass through the origin.
        assert!(HarmonicMode::Even.shape(0.0).abs() < 1e-6);
    }

    #[test]
    fn shaper_is_bounded() {
        for mode in [HarmonicMode::Odd, HarmonicMode::Even, HarmonicMode::Mix] {
            for &x in &[-1000.0, -1.0, 0.0, 1.0, 1000.0] {
                let y = mode.shape(x);
                assert!(y.is_finite());
                assert!(y.abs() <= 2.0 + 1e-6, "mode {mode:?} unbounded: {y}");
            }
        }
    }

    #[test]
    fn drive_is_clamped_to_safe_range() {
        let mut exciter = Exciter::new(params(), SR, 1);
        exciter.set_drive(Sample::INFINITY);
        assert_eq!(exciter.drive(), 0.0);
        exciter.set_drive(1.0e9);
        assert!(exciter.drive() <= MAX_DRIVE + 1e-3);
        exciter.set_drive(-5.0);
        assert_eq!(exciter.drive(), 0.0);
    }

    #[test]
    fn set_frequency_is_clamped_positive() {
        let mut exciter = Exciter::new(params(), SR, 1);
        exciter.set_frequency_hz(-100.0);
        assert!(exciter.frequency_hz() >= 1.0);
    }

    #[test]
    fn reset_restores_reproducible_output() {
        let mut exciter = Exciter::new(params(), SR, 1);
        let first = settle(&mut exciter, 7_000.0, 1_024);
        exciter.reset();
        let second = settle(&mut exciter, 7_000.0, 1_024);
        assert_eq!(first, second);
    }

    #[test]
    fn channels_are_independent() {
        let mut exciter = Exciter::new(params(), SR, 2);
        assert_eq!(exciter.channels(), 2);
        for i in 0..256 {
            let phase = core::f32::consts::TAU * 6_000.0 * (i as Sample) / (SR as Sample);
            let _ = exciter.harmonics(0, ops::sin(phase));
            let _ = exciter.harmonics(1, 0.0);
        }
        let a = exciter.harmonics(0, 0.8);
        let b = exciter.harmonics(1, 0.0);
        assert!((a - b).abs() > 1e-6, "channels should hold independent state");
    }

    #[test]
    fn node_passes_dry_signal_through() {
        // With amount = 0 the node is a transparent pass-through.
        let mut p = params();
        p.amount = 0.0;
        let mut node = ExciterNode::new(p, SR, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 32);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 32);
        input.set_active_frames(32);
        output.set_active_frames(32);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            let phase = core::f32::consts::TAU * 6_000.0 * (i as Sample) / (SR as Sample);
            *s = ops::sin(phase);
        }
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 32,
            playhead: 0,
        };
        let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
        node.process(&ctx, &mut io);
        for f in 0..32 {
            assert!((output.channel(0)[f] - input.channel(0)[f]).abs() < 1e-6);
        }
    }

    #[test]
    fn node_adds_brightness_when_excited() {
        // A positive amount should change the output relative to the dry input.
        let mut node = ExciterNode::new(params(), SR, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 256);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 256);
        input.set_active_frames(256);
        output.set_active_frames(256);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            let phase = core::f32::consts::TAU * 7_000.0 * (i as Sample) / (SR as Sample);
            *s = 0.8 * ops::sin(phase);
        }
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 256,
            playhead: 0,
        };
        let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
        node.process(&ctx, &mut io);
        let mut changed = false;
        for f in 128..256 {
            if (output.channel(0)[f] - input.channel(0)[f]).abs() > 1e-4 {
                changed = true;
                break;
            }
        }
        assert!(changed, "exciter did not modify the signal");
        assert!(output.channel(0).iter().all(|s| s.is_finite()));
    }

    #[test]
    fn node_reset_clears_tail() {
        let mut node = ExciterNode::new(params(), SR, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 64);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 64);
        input.set_active_frames(64);
        output.set_active_frames(64);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            let phase = core::f32::consts::TAU * 8_000.0 * (i as Sample) / (SR as Sample);
            *s = ops::sin(phase);
        }
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 64,
            playhead: 0,
        };
        {
            let mut io =
                ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
            node.process(&ctx, &mut io);
        }
        node.reset();
        let silent = AudioBuffer::new(ChannelLayout::Mono, 64);
        let mut after = AudioBuffer::new(ChannelLayout::Mono, 64);
        after.set_active_frames(64);
        let mut io = ProcessIo::new(core::slice::from_ref(&silent), core::slice::from_mut(&mut after));
        node.process(&ctx, &mut io);
        for f in 0..64 {
            assert!(after.channel(0)[f].abs() < 1e-6, "tail not cleared at {f}");
        }
    }

    #[test]
    fn mix_mode_runs_and_is_finite() {
        let mut p = params();
        p.mode = HarmonicMode::Mix;
        let mut exciter = Exciter::new(p, SR, 1);
        let out = settle(&mut exciter, 6_000.0, 1_024);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let mut p = params();
        p.frequency_hz = 1.0e9;
        p.drive = 1.0e9;
        p.amount = 1.0e9;
        let mut node = ExciterNode::new(p, SR, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 16);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 16);
        input.set_active_frames(16);
        output.set_active_frames(16);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 16,
            playhead: 0,
        };
        let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
        node.process(&ctx, &mut io);
        assert!(output.channel(0).iter().all(|s| s.is_finite()));
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = ExciterNode::new(params(), SR, 1);
        let input = AudioBuffer::new(ChannelLayout::Mono, 8);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 8);
        output.set_active_frames(0);
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 0,
            playhead: 0,
        };
        let mut io = ProcessIo::new(core::slice::from_ref(&input), core::slice::from_mut(&mut output));
        node.process(&ctx, &mut io);
        assert_eq!(output.active_frames(), 0);
    }
}
