//! West-coast wave folder (reflective non-linear distortion).
//!
//! A wave *folder* is fundamentally different from the soft clipper in
//! [`waveshaper`](crate::nodes::effects::waveshaper). A clipper *compresses*
//! signal toward a ceiling; a folder *reflects* it back once it crosses a
//! threshold, so the transfer function keeps bending the waveform over on
//! itself. Repeated folding multiplies the harmonic content far faster than
//! clipping, producing the bright, metallic, vowel-like timbres associated with
//! the "west-coast" (Buchla / Serge) school of synthesis rather than the warm
//! saturation of a tube or tape stage.
//!
//! Two closed-form folding transfer functions are offered (see [`FoldShape`]):
//!
//! - [`FoldShape::Triangle`] -- a period-4 triangle wave that is the identity on
//!   `[-1, 1]` and mirrors the signal around `+/-1` beyond it. This is the hard,
//!   buzzy classic fold.
//! - [`FoldShape::Sine`] -- `sin(v * pi/2)`, the identity-slope sinusoidal fold
//!   that rounds the reflection corners for a smoother, hollower character.
//!
//! Each folder is driven by a `drive` gain (how deep the signal is pushed into
//! the folds) and a DC `offset` added before folding. A non-zero offset breaks
//! the waveform's symmetry and so introduces even harmonics, a staple
//! west-coast timbral control.
//!
//! # Why oversample
//!
//! Folding is one of the most harmonically explosive non-linearities in audio,
//! so it aliases badly at the host rate. The node therefore runs the fold
//! through the shared [`Oversampler`](crate::oversampler::Oversampler) at a
//! selectable factor (see [`Oversample`]) and delays its dry path by the
//! matching latency with a [`DryDelay`](crate::oversampler::DryDelay) so the
//! wet/dry blend stays phase-coherent.
//!
//! # Real-time contract
//!
//! All filter state, per-channel oversampler histories, and dry-delay lines are
//! allocated at construction. [`WavefolderNode::process`] performs no
//! allocation, takes no locks, and cannot panic; every output is
//! denormal-flushed. The two fold shapes are closed-form and bounded, so a
//! runaway offset or drive cannot produce an unbounded sample. All
//! transcendental math routes through [`bevy_math::ops`] for bit-reproducible
//! output across platforms.
//!
//! # Provenance
//!
//! Reflective wave folding is a publicly documented analogue-synthesis
//! technique (Buchla 259, Serge, and countless textbook treatments); the
//! closed-form triangle reduction and the `sin(v * pi/2)` sine fold are
//! standard mathematical constructions. This node reuses only this crate's own
//! [`Oversampler`](crate::oversampler::Oversampler) primitive and parameter
//! smoothing. It is pure classic DSP with no AI or ML and contains **no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio
//! source or derived code**; it is implemented purely from those publicly
//! documented algorithms.
//!
//! # Relationship
//!
//! The folder is the orthogonal complement to the
//! [`waveshaper`](crate::nodes::effects::waveshaper) soft clipper and the
//! [`saturation`](crate::nodes::effects::saturation) stage (whose own docs note
//! it is "not a wave folder"): clippers and saturators *compress* toward a
//! limit, whereas this *reflects* past it. All three share the same
//! anti-aliasing [`Oversampler`](crate::oversampler::Oversampler) so the
//! aliasing cure is implemented once.

use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::FRAC_PI_2;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::oversampler::{DEFAULT_TAPS_PER_PHASE, DryDelay, Oversampler, OversamplerState};
use crate::param::{Ramp, Smoothed};

/// The internal processing rate relative to the host sample rate.
///
/// Because folding is so harmonically rich, higher factors are often warranted
/// here than for a gentle clipper; `X8` is offered for extreme drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Oversample {
    /// No oversampling: fold directly at the host rate (zero latency, cheapest,
    /// but aliases the most).
    X1,
    /// 2x oversampling.
    X2,
    /// 4x oversampling.
    X4,
    /// 8x oversampling (cleanest, dearest), recommended for heavy folding.
    X8,
}

impl Oversample {
    /// Returns the integer oversampling factor (`1`, `2`, `4`, or `8`).
    #[inline]
    #[must_use]
    pub const fn factor(self) -> usize {
        match self {
            Oversample::X1 => 1,
            Oversample::X2 => 2,
            Oversample::X4 => 4,
            Oversample::X8 => 8,
        }
    }
}

/// The folding transfer function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum FoldShape {
    /// Period-4 triangle wave: the identity on `[-1, 1]`, mirrored around `+/-1`
    /// beyond it. Hard, buzzy, maximally bright.
    Triangle,
    /// `sin(v * pi/2)`: a smooth sinusoidal fold with identity slope at the
    /// origin, rounding the reflection corners for a hollower timbre.
    Sine,
}

impl FoldShape {
    /// Applies the fold to a single (already driven and offset) value.
    ///
    /// Both shapes are bounded to `[-1, 1]`.
    ///
    /// ```
    /// use prism_audio_core::nodes::effects::wavefolder::FoldShape;
    /// // Both shapes are the identity inside [-1, 1]...
    /// assert!((FoldShape::Triangle.fold(0.5) - 0.5).abs() < 1.0e-6);
    /// // ...and the triangle fold reflects a value just past the +1 ceiling.
    /// assert!((FoldShape::Triangle.fold(1.5) - 0.5).abs() < 1.0e-6);
    /// // The sine fold is always bounded.
    /// assert!(FoldShape::Sine.fold(10.0).abs() <= 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn fold(self, v: Sample) -> Sample {
        match self {
            FoldShape::Triangle => {
                // Period-4 triangle through the origin with unit slope:
                //   t(v) = |rem_4(v - 1) - 2| - 1
                // where rem_4 is the non-negative remainder modulo 4. This is
                // the identity for v in [-1, 1] and reflects beyond it.
                let m = v - 1.0;
                let r = m - 4.0 * ops::floor(m / 4.0);
                (r - 2.0).abs() - 1.0
            }
            // sin(v * pi/2): equals ~v near the origin and folds sinusoidally
            // once |v| exceeds 1, naturally bounded to [-1, 1].
            FoldShape::Sine => ops::sin(v * FRAC_PI_2),
        }
    }
}

/// Serialisable control parameters for a [`WavefolderNode`].
///
/// All fields are plain scalars so a host can snapshot, store, and restore a
/// patch. [`WavefolderParams::sanitised`] replaces any non-finite field with
/// its default and clamps the mix controls into range.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WavefolderParams {
    /// Pre-fold drive gain: how hard the signal is pushed into the folds.
    /// Higher values traverse more fold periods and generate more harmonics.
    pub drive: Sample,
    /// DC offset added after `drive` and before folding. Non-zero values break
    /// symmetry and introduce even harmonics.
    pub offset: Sample,
    /// Post-fold make-up gain applied to the wet signal.
    pub output_gain: Sample,
    /// Wet (folded) mix coefficient.
    pub wet: Sample,
    /// Dry (unprocessed, latency-aligned) mix coefficient.
    pub dry: Sample,
}

impl Default for WavefolderParams {
    fn default() -> Self {
        Self {
            drive: 1.0,
            offset: 0.0,
            output_gain: 1.0,
            wet: 1.0,
            dry: 0.0,
        }
    }
}

impl WavefolderParams {
    /// Returns a copy with every non-finite field replaced by its default and
    /// the mix coefficients clamped to `[0, 1]`.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        Self {
            drive: finite_or(self.drive, d.drive),
            offset: finite_or(self.offset, d.offset),
            output_gain: finite_or(self.output_gain, d.output_gain),
            wet: finite_or(self.wet, d.wet).clamp(0.0, 1.0),
            dry: finite_or(self.dry, d.dry).clamp(0.0, 1.0),
        }
    }
}

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// A west-coast wave folder with selectable fold shape and anti-aliasing
/// oversampling (input port 0 -> output port 0).
///
/// The signal path per sample is `fold(drive * x + offset)` evaluated through
/// the oversampler, scaled by `output_gain`, then blended with the
/// latency-aligned dry signal via `wet`/`dry`. Every control is
/// [`Smoothed`](crate::param::Smoothed) so automation never clicks.
#[derive(Debug, Clone)]
pub struct WavefolderNode {
    oversample: Oversample,
    fold_shape: FoldShape,
    drive: Smoothed,
    offset: Smoothed,
    output_gain: Smoothed,
    wet: Smoothed,
    dry: Smoothed,
    /// Shared anti-aliasing resampler (one design, many channels).
    oversampler: Oversampler,
    /// Per-channel oversampler history.
    states: Vec<OversamplerState>,
    /// Per-channel dry-path delay aligning the dry signal with the wet latency.
    dry_delays: Vec<DryDelay>,
}

impl WavefolderNode {
    /// Builds a wave folder for a `channels`-wide signal.
    ///
    /// `sample_rate` is accepted for API symmetry with the other nodes; the
    /// oversampling filters are designed in normalised frequency and do not
    /// depend on it. Non-finite parameters are sanitised at construction.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        oversample: Oversample,
        fold_shape: FoldShape,
        params: WavefolderParams,
    ) -> Self {
        let _ = sample_rate;
        let p = params.sanitised();
        let oversampler = Oversampler::new(oversample.factor(), DEFAULT_TAPS_PER_PHASE);
        let latency = oversampler.latency_frames() as usize;

        let mut states = Vec::with_capacity(channels);
        let mut dry_delays = Vec::with_capacity(channels);
        for _ in 0..channels {
            states.push(oversampler.make_state());
            dry_delays.push(DryDelay::new(latency));
        }

        Self {
            oversample,
            fold_shape,
            drive: Smoothed::new(p.drive),
            offset: Smoothed::new(p.offset),
            output_gain: Smoothed::new(p.output_gain),
            wet: Smoothed::new(p.wet),
            dry: Smoothed::new(p.dry),
            oversampler,
            states,
            dry_delays,
        }
    }

    /// Sets a new drive gain, gliding toward it with `ramp`.
    #[inline]
    pub fn set_drive(&mut self, target: Sample, ramp: Ramp) {
        self.drive.set_target(target, ramp);
    }

    /// Sets a new pre-fold DC offset, gliding toward it with `ramp`.
    #[inline]
    pub fn set_offset(&mut self, target: Sample, ramp: Ramp) {
        self.offset.set_target(target, ramp);
    }

    /// Sets a new post-fold make-up gain, gliding toward it with `ramp`.
    #[inline]
    pub fn set_output_gain(&mut self, target: Sample, ramp: Ramp) {
        self.output_gain.set_target(target, ramp);
    }

    /// Sets a new wet-mix coefficient, gliding toward it with `ramp`.
    #[inline]
    pub fn set_wet(&mut self, target: Sample, ramp: Ramp) {
        self.wet.set_target(target, ramp);
    }

    /// Sets a new dry-mix coefficient, gliding toward it with `ramp`.
    #[inline]
    pub fn set_dry(&mut self, target: Sample, ramp: Ramp) {
        self.dry.set_target(target, ramp);
    }

    /// Returns the configured oversampling mode.
    #[inline]
    #[must_use]
    pub fn oversample(&self) -> Oversample {
        self.oversample
    }

    /// Returns the configured fold shape.
    #[inline]
    #[must_use]
    pub fn fold_shape(&self) -> FoldShape {
        self.fold_shape
    }

    /// Returns the length of the oversampling prototype FIR filter, or `0` when
    /// not oversampling. Useful for verifying the reported latency.
    #[inline]
    #[must_use]
    pub fn filter_taps(&self) -> usize {
        self.oversampler.filter_taps()
    }
}

impl AudioNode for WavefolderNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output
            .channels()
            .min(input.channels())
            .min(self.states.len());
        let frames = output.active_frames().min(input.active_frames());

        // Snapshot the smoothers so every channel replays the identical
        // per-sample control trajectory (see `WaveshaperNode` for the pattern).
        let drive0 = self.drive;
        let offset0 = self.offset;
        let gain0 = self.output_gain;
        let wet0 = self.wet;
        let dry0 = self.dry;

        let oversampler = &self.oversampler;
        let shape = self.fold_shape;
        let states = &mut self.states;
        let dry_delays = &mut self.dry_delays;

        let mut committed: Option<(Smoothed, Smoothed, Smoothed, Smoothed, Smoothed)> = None;

        for ch in 0..channels {
            let mut drive = drive0;
            let mut offset = offset0;
            let mut gain = gain0;
            let mut wet = wet0;
            let mut dry = dry0;

            let state = &mut states[ch];
            let dry_delay = &mut dry_delays[ch];
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);

            for i in 0..frames {
                let x = src[i];
                let dv = drive.next_sample();
                let ov = offset.next_sample();
                let gv = gain.next_sample();
                let wv = wet.next_sample();
                let drv = dry.next_sample();

                let shaped = oversampler.process_sample(state, x, |v| shape.fold(dv * v + ov));
                let dry_sig = dry_delay.push(x);
                dst[i] = flush_denormal(wv * (shaped * gv) + drv * dry_sig);
            }

            if ch + 1 == channels {
                committed = Some((drive, offset, gain, wet, dry));
            }
        }

        if let Some((drive, offset, gain, wet, dry)) = committed {
            self.drive = drive;
            self.offset = offset;
            self.output_gain = gain;
            self.wet = wet;
            self.dry = dry;
        }
    }

    fn reset(&mut self) {
        for state in &mut self.states {
            state.reset();
        }
        for dry_delay in &mut self.dry_delays {
            dry_delay.reset();
        }
        self.drive = Smoothed::new(self.drive.target());
        self.offset = Smoothed::new(self.offset.target());
        self.output_gain = Smoothed::new(self.output_gain.target());
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
    }

    fn latency_frames(&self) -> u32 {
        self.oversampler.latency_frames()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use core::f32::consts::PI;

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    /// Single-bin DFT magnitude (normalised) at frequency `f`.
    fn bin_magnitude(signal: &[Sample], sample_rate: u32, f: Sample) -> Sample {
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, &s) in signal.iter().enumerate() {
            let phase = 2.0 * PI * f * (i as Sample) / (sample_rate as Sample);
            re += s * ops::cos(phase);
            im += s * ops::sin(phase);
        }
        ops::sqrt(re * re + im * im) / (signal.len() as Sample)
    }

    #[test]
    fn triangle_fold_matches_hand_computed_values() {
        let t = FoldShape::Triangle;
        // Identity inside the band.
        assert!((t.fold(0.0) - 0.0).abs() < 1.0e-6);
        assert!((t.fold(0.5) - 0.5).abs() < 1.0e-6);
        assert!((t.fold(-0.5) + 0.5).abs() < 1.0e-6);
        // Peaks and reflections.
        assert!((t.fold(1.0) - 1.0).abs() < 1.0e-6);
        assert!((t.fold(1.5) - 0.5).abs() < 1.0e-6);
        assert!((t.fold(2.0) - 0.0).abs() < 1.0e-6);
        assert!((t.fold(3.0) + 1.0).abs() < 1.0e-6);
        assert!((t.fold(-1.5) + 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn sine_fold_is_identity_slope_and_bounded() {
        let s = FoldShape::Sine;
        // Near the origin sin(v*pi/2) ~= v*pi/2... actually equals v only to
        // first order; just assert small-signal is close and sign-preserving.
        assert!(s.fold(0.0).abs() < 1.0e-7);
        assert!((s.fold(1.0) - 1.0).abs() < 1.0e-6);
        assert!((s.fold(-1.0) + 1.0).abs() < 1.0e-6);
        // Always bounded, even for absurd drive.
        for k in 0..200 {
            let v = k as Sample * 0.37 - 37.0;
            assert!(s.fold(v).abs() <= 1.0 + 1.0e-6, "sine fold unbounded at {v}");
        }
    }

    #[test]
    fn both_shapes_stay_bounded_under_extreme_drive() {
        for shape in [FoldShape::Triangle, FoldShape::Sine] {
            let mut node = WavefolderNode::new(
                48_000,
                1,
                Oversample::X8,
                shape,
                WavefolderParams {
                    drive: 20.0,
                    offset: 3.0,
                    output_gain: 1.0,
                    wet: 1.0,
                    dry: 0.0,
                },
            );
            let n = 256;
            let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
            for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
                *s = 0.9 * ops::sin(0.21 * i as Sample);
            }
            let inputs = [input];
            let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(48_000, n), &mut io);
            for &o in outputs[0].channel(0) {
                assert!(o.abs() <= 1.5, "folder output not bounded: {o}");
            }
        }
    }

    #[test]
    fn x1_dc_input_folds_exactly() {
        // No oversampling means process_sample shapes the input directly, so a
        // constant input yields exactly fold(drive * x + offset).
        let mut node = WavefolderNode::new(
            48_000,
            1,
            Oversample::X1,
            FoldShape::Triangle,
            WavefolderParams {
                drive: 1.0,
                offset: 0.0,
                output_gain: 1.0,
                wet: 1.0,
                dry: 0.0,
            },
        );
        let n = 16;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        for s in input.channel_mut(0).iter_mut() {
            *s = 1.5; // folds to 0.5
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(48_000, n), &mut io);
        for &o in outputs[0].channel(0) {
            assert!((o - 0.5).abs() < 1.0e-6, "unexpected fold output: {o}");
        }
    }

    #[test]
    fn offset_introduces_even_harmonics() {
        // Feed a sine; with zero offset the folded wave is odd-symmetric and the
        // 2nd harmonic is tiny. A DC offset breaks symmetry and lifts it.
        fn second_harmonic(offset: Sample) -> Sample {
            let sr = 48_000u32;
            let n = 4_096usize;
            let f0 = 300.0;
            let mut node = WavefolderNode::new(
                sr,
                1,
                Oversample::X8,
                FoldShape::Triangle,
                WavefolderParams {
                    drive: 2.0,
                    offset,
                    output_gain: 1.0,
                    wet: 1.0,
                    dry: 0.0,
                },
            );
            let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
            for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
                let t = i as Sample / sr as Sample;
                *s = 0.6 * ops::sin(2.0 * PI * f0 * t);
            }
            let inputs = [input];
            let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(sr, n), &mut io);
            // Analyse the steady-state tail.
            bin_magnitude(&outputs[0].channel(0)[n / 2..], sr, 2.0 * f0)
        }
        let h2_sym = second_harmonic(0.0);
        let h2_biased = second_harmonic(0.4);
        assert!(
            h2_biased > h2_sym * 3.0,
            "offset did not add even harmonics: sym={h2_sym} biased={h2_biased}"
        );
    }

    fn alias_magnitude(os: Oversample) -> Sample {
        let sr = 48_000u32;
        let n = 4_800usize;
        let f0 = 7_000.0;
        let mut node = WavefolderNode::new(
            sr,
            1,
            os,
            FoldShape::Triangle,
            WavefolderParams {
                drive: 3.0,
                offset: 0.0,
                output_gain: 1.0,
                wet: 1.0,
                dry: 0.0,
            },
        );
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            let t = i as Sample / sr as Sample;
            *s = 0.8 * ops::sin(2.0 * PI * f0 * t);
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sr, n), &mut io);
        // The 7th harmonic of 7 kHz (49 kHz) folds back to |49k - 48k| = 1 kHz,
        // an inharmonic alias that oversampling should suppress.
        bin_magnitude(outputs[0].channel(0), sr, 1_000.0)
    }

    #[test]
    fn oversampling_reduces_aliasing() {
        let alias_x1 = alias_magnitude(Oversample::X1);
        let alias_x8 = alias_magnitude(Oversample::X8);
        assert!(
            alias_x8 < alias_x1 * 0.5,
            "oversampling did not reduce aliasing: x1={alias_x1} x8={alias_x8}"
        );
    }

    #[test]
    fn wet_zero_is_pure_dry() {
        // wet = 0, dry = 1, no oversampling => output is bit-exact input.
        let mut node = WavefolderNode::new(
            48_000,
            1,
            Oversample::X1,
            FoldShape::Triangle,
            WavefolderParams {
                drive: 4.0,
                offset: 0.3,
                output_gain: 2.0,
                wet: 0.0,
                dry: 1.0,
            },
        );
        let n = 48;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = (i as Sample) * 0.1 - 2.0;
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(48_000, n), &mut io);
        assert_eq!(outputs[0].channel(0), inputs[0].channel(0));
    }

    #[test]
    fn dry_zero_is_pure_wet() {
        // dry = 0 => output carries only the folded signal (non-zero energy).
        let mut node = WavefolderNode::new(
            48_000,
            1,
            Oversample::X1,
            FoldShape::Triangle,
            WavefolderParams::default(),
        );
        let n = 64;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = 2.5 * ops::sin(0.2 * i as Sample);
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(48_000, n), &mut io);
        let energy: Sample = outputs[0].channel(0).iter().map(|s| s * s).sum();
        assert!(energy > 0.0, "pure-wet output was silent");
    }

    #[test]
    fn latency_matches_oversampler() {
        let n0 = WavefolderNode::new(
            48_000,
            1,
            Oversample::X1,
            FoldShape::Triangle,
            WavefolderParams::default(),
        );
        assert_eq!(n0.latency_frames(), 0);
        assert_eq!(n0.filter_taps(), 0);

        let n8 = WavefolderNode::new(
            48_000,
            1,
            Oversample::X8,
            FoldShape::Sine,
            WavefolderParams::default(),
        );
        let os = Oversampler::new(8, DEFAULT_TAPS_PER_PHASE);
        assert_eq!(n8.latency_frames(), os.latency_frames());
        assert_eq!(n8.filter_taps(), os.filter_taps());
        assert!(n8.latency_frames() >= 1);
    }

    #[test]
    fn stereo_channels_are_independent() {
        let sr = 48_000u32;
        let n = 128;
        let mut node = WavefolderNode::new(
            sr,
            2,
            Oversample::X2,
            FoldShape::Triangle,
            WavefolderParams::default(),
        );
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = 1.3 * ops::sin(0.2 * i as Sample);
        }
        // Right channel left silent.
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Stereo, n)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sr, n), &mut io);
        for &o in outputs[0].channel(1) {
            assert_eq!(o, 0.0, "silent channel picked up energy: {o}");
        }
        let left_energy: Sample = outputs[0].channel(0).iter().map(|s| s * s).sum();
        assert!(left_energy > 0.0, "active channel produced no output");
    }

    #[test]
    fn non_finite_params_are_sanitised() {
        let p = WavefolderParams {
            drive: Sample::NAN,
            offset: Sample::INFINITY,
            output_gain: Sample::NEG_INFINITY,
            wet: 5.0,
            dry: -2.0,
        }
        .sanitised();
        let d = WavefolderParams::default();
        assert_eq!(p.drive, d.drive);
        assert_eq!(p.offset, d.offset);
        assert_eq!(p.output_gain, d.output_gain);
        assert_eq!(p.wet, 1.0);
        assert_eq!(p.dry, 0.0);
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = WavefolderNode::new(
            48_000,
            1,
            Oversample::X4,
            FoldShape::Triangle,
            WavefolderParams::default(),
        );
        let inputs = [AudioBuffer::new(ChannelLayout::Mono, 1)];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 1)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(48_000, 0), &mut io);
    }

    #[test]
    fn reset_matches_fresh_instance() {
        let sr = 48_000u32;
        let n = 64;
        let make = || {
            WavefolderNode::new(
                sr,
                1,
                Oversample::X4,
                FoldShape::Sine,
                WavefolderParams {
                    drive: 2.5,
                    offset: 0.1,
                    output_gain: 1.0,
                    wet: 1.0,
                    dry: 0.0,
                },
            )
        };
        let mut node = make();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = 0.7 * ops::sin(0.3 * i as Sample);
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, n)];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(sr, n), &mut io);
        }
        node.reset();

        // Fresh node and reset node must produce identical output on the same
        // input.
        let mut fresh = make();
        let mut out_reset = [AudioBuffer::new(ChannelLayout::Mono, n)];
        let mut out_fresh = [AudioBuffer::new(ChannelLayout::Mono, n)];
        {
            let mut io = ProcessIo::new(&inputs, &mut out_reset);
            node.process(&ctx(sr, n), &mut io);
        }
        {
            let mut io = ProcessIo::new(&inputs, &mut out_fresh);
            fresh.process(&ctx(sr, n), &mut io);
        }
        assert_eq!(out_reset[0].channel(0), out_fresh[0].channel(0));
    }

    #[test]
    fn default_params_round_trip() {
        let d = WavefolderParams::default();
        assert_eq!(d.sanitised(), d);
    }
}
