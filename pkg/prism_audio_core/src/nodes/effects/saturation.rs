//! Multi-curve analogue-style saturation with asymmetric bias and DC blocking.
//!
//! Where [`waveshaper::WaveshaperNode`](crate::nodes::effects::waveshaper) is a
//! single fixed `tanh` soft clipper, this node offers a palette of saturation
//! transfer functions ([`SaturationCurve`]) plus an adjustable `bias` that
//! shifts the operating point off centre. A biased non-linearity is no longer
//! odd-symmetric, so it generates *even* harmonics in addition to the odd ones,
//! which is the characteristic warmth of single-ended tube and transformer
//! stages. The static offset that an asymmetric curve would otherwise leave on
//! the signal is removed in two independent ways:
//!
//! 1. The transfer function is evaluated as `f(drive * x + bias) - f(bias)`, so
//!    a silent input maps to exactly zero regardless of `bias`.
//! 2. A one-pole DC blocker on the shaped output rejects the slow DC component
//!    that an asymmetric curve pumps out of a dynamic signal.
//!
//! # Curves
//!
//! Every curve is normalised to saturate toward `+/-1`:
//!
//! - [`SaturationCurve::Tanh`] -- hyperbolic tangent, the smoothest knee.
//! - [`SaturationCurve::Arctan`] -- `arctan` scaled by `2/pi`, a slightly
//!   brighter knee than `tanh`.
//! - [`SaturationCurve::Cubic`] -- the classic cubic soft clipper
//!   `1.5 c - 0.5 c^3` on the clamped input `c`, reaching a hard limit of
//!   exactly `+/-1` at `|x| >= 1`.
//! - [`SaturationCurve::Reciprocal`] -- the algebraic clipper `x / (1 + |x|)`.
//! - [`SaturationCurve::Sine`] -- `sin(c * pi/2)` on the clamped input, a soft
//!   sinusoidal saturator (not a wave folder).
//!
//! # Anti-aliasing
//!
//! Any non-linearity creates harmonics that fold back below Nyquist as
//! aliasing. Like the waveshaper, this node can run the shaper at an integer
//! multiple of the host rate ([`Oversample`]): the input is upsampled through a
//! polyphase windowed-sinc interpolator, shaped at the higher rate, then
//! decimated through the matching low-pass so the fresh out-of-band harmonics
//! are removed before they can alias. The linear-phase FIR pair introduces a
//! fixed group delay reported by [`AudioNode::latency_frames`]; the dry path is
//! delayed by the same amount so the mix stays phase-coherent.
//!
//! All state (interpolation/decimation histories, dry delay, DC-blocker memory)
//! is pre-allocated at construction, so [`SaturationNode::process`] performs no
//! allocation, takes no locks, and cannot panic -- it is real-time safe.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The saturation
//! curves are standard closed-form non-linearities and the anti-aliasing uses a
//! textbook windowed-sinc polyphase resampler. It is pure classic DSP with no
//! AI/ML of any kind.
//!
//! # Relationship
//!
//! This node reuses the [`Oversample`] configuration enum defined by the
//! sibling [`waveshaper`](crate::nodes::effects::waveshaper) module and applies
//! the same standard polyphase anti-aliasing technique, but it is a distinct
//! processor: the waveshaper is a fixed symmetric `tanh` clipper, whereas this
//! node adds a selectable curve set, an asymmetric `bias` that introduces even
//! harmonics, and a DC blocker. It shares the crate-wide [`Sample`] type and the
//! [`Smoothed`]/[`Ramp`] automation primitives with every other node.

use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::{FRAC_2_PI, FRAC_PI_2, PI};

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::nodes::effects::waveshaper::Oversample;
use crate::param::{Ramp, Smoothed};

/// Number of filter taps allocated to each polyphase branch of the
/// oversampler. The full prototype length is `TAPS_PER_PHASE * factor`.
const TAPS_PER_PHASE: usize = 16;

/// Default pole of the one-pole DC blocker (`y = x - x[-1] + R * y[-1]`).
///
/// `0.9995` places the high-pass corner near 4 Hz at 48 kHz, well below the
/// audio band, so it removes the DC produced by asymmetric saturation without
/// audibly thinning the low end.
pub const DEFAULT_DC_BLOCK_COEFF: Sample = 0.9995;

/// The saturation transfer function applied by a [`SaturationNode`].
///
/// Each variant is normalised to saturate toward `+/-1` for a drive-scaled,
/// biased input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SaturationCurve {
    /// Hyperbolic tangent `tanh(x)` -- the smoothest, most neutral knee.
    Tanh,
    /// `arctan(x) * 2/pi` -- a slightly brighter knee than `tanh`.
    Arctan,
    /// Cubic soft clipper `1.5 c - 0.5 c^3` on the clamped input, hard-limiting
    /// to exactly `+/-1` for `|x| >= 1`.
    Cubic,
    /// Algebraic clipper `x / (1 + |x|)`.
    Reciprocal,
    /// `sin(c * pi/2)` on the clamped input -- a soft sinusoidal saturator.
    Sine,
}

impl SaturationCurve {
    /// Evaluates the normalised, odd-symmetric transfer function at `x`.
    #[inline]
    #[must_use]
    pub fn shape(self, x: Sample) -> Sample {
        match self {
            SaturationCurve::Tanh => ops::tanh(x),
            SaturationCurve::Arctan => ops::atan(x) * FRAC_2_PI,
            SaturationCurve::Cubic => {
                let c = x.clamp(-1.0, 1.0);
                1.5 * c - 0.5 * c * c * c
            }
            SaturationCurve::Reciprocal => x / (1.0 + ops::abs(x)),
            SaturationCurve::Sine => {
                let c = x.clamp(-1.0, 1.0);
                ops::sin(c * FRAC_PI_2)
            }
        }
    }
}

/// Evaluates the biased transfer function `f(x + bias) - f(bias)`.
///
/// Subtracting `f(bias)` re-centres the curve so a silent input always maps to
/// exactly zero, leaving the DC blocker to deal only with signal-dependent DC.
#[inline]
fn shaped_biased(curve: SaturationCurve, x: Sample, bias: Sample) -> Sample {
    curve.shape(x + bias) - curve.shape(bias)
}

/// Construction-time settings for a [`SaturationNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SaturationParams {
    /// The transfer function.
    pub curve: SaturationCurve,
    /// Internal oversampling factor for anti-aliasing.
    pub oversample: Oversample,
    /// Pre-shaping input gain. Higher values saturate harder.
    pub drive: Sample,
    /// Asymmetry bias added to the driven input before shaping. Non-zero values
    /// introduce even harmonics for a single-ended tube/transformer character.
    pub bias: Sample,
    /// Post-shaping make-up gain applied to the wet signal.
    pub output_gain: Sample,
    /// Wet/dry blend in `[0, 1]`: `0` is fully dry, `1` is fully wet.
    pub mix: Sample,
    /// Whether the output DC blocker is enabled.
    pub dc_block: bool,
}

impl Default for SaturationParams {
    fn default() -> Self {
        Self {
            curve: SaturationCurve::Tanh,
            oversample: Oversample::X1,
            drive: 1.0,
            bias: 0.0,
            output_gain: 1.0,
            mix: 1.0,
            dc_block: true,
        }
    }
}

/// Per-channel filter memory for the oversampler, dry delay, and DC blocker.
#[derive(Debug, Clone)]
struct ChannelState {
    /// Interpolation input history (length [`TAPS_PER_PHASE`], empty at `X1`).
    up_hist: Vec<Sample>,
    /// Write cursor into [`ChannelState::up_hist`].
    up_pos: usize,
    /// Decimation input history (length `TAPS_PER_PHASE * factor`, empty at
    /// `X1`).
    down_hist: Vec<Sample>,
    /// Write cursor into [`ChannelState::down_hist`].
    down_pos: usize,
    /// Dry-path delay aligning the dry signal to the wet-path latency.
    dry_delay: Vec<Sample>,
    /// Write cursor into [`ChannelState::dry_delay`].
    dry_pos: usize,
    /// Previous DC-blocker input `x[-1]`.
    dc_x1: Sample,
    /// Previous DC-blocker output `y[-1]`.
    dc_y1: Sample,
}

impl ChannelState {
    /// Allocates zeroed histories of the given lengths.
    fn new(up_len: usize, down_len: usize, dry_len: usize) -> Self {
        Self {
            up_hist: zeroed(up_len),
            up_pos: 0,
            down_hist: zeroed(down_len),
            down_pos: 0,
            dry_delay: zeroed(dry_len),
            dry_pos: 0,
            dc_x1: 0.0,
            dc_y1: 0.0,
        }
    }

    /// Clears every history and the DC-blocker memory back to silence.
    fn reset(&mut self) {
        for s in &mut self.up_hist {
            *s = 0.0;
        }
        for s in &mut self.down_hist {
            *s = 0.0;
        }
        for s in &mut self.dry_delay {
            *s = 0.0;
        }
        self.up_pos = 0;
        self.down_pos = 0;
        self.dry_pos = 0;
        self.dc_x1 = 0.0;
        self.dc_y1 = 0.0;
    }

    /// Shapes one raw input sample through the oversampled, anti-aliased path.
    ///
    /// `drive` and `bias` are applied at the oversampled rate; `phases` are the
    /// polyphase interpolation branches and `proto` the shared decimation
    /// prototype.
    fn shape_oversampled(
        &mut self,
        x: Sample,
        drive: Sample,
        bias: Sample,
        curve: SaturationCurve,
        phases: &[Vec<Sample>],
        proto: &[Sample],
    ) -> Sample {
        let up_len = self.up_hist.len();
        self.up_pos = (self.up_pos + 1) % up_len;
        self.up_hist[self.up_pos] = x;

        let down_len = self.down_hist.len();
        for phase in phases {
            let mut acc = 0.0;
            for (j, &coeff) in phase.iter().enumerate() {
                let idx = (self.up_pos + up_len - j) % up_len;
                acc += coeff * self.up_hist[idx];
            }
            let shaped = shaped_biased(curve, drive * acc, bias);
            self.down_pos = (self.down_pos + 1) % down_len;
            self.down_hist[self.down_pos] = shaped;
        }

        let mut out = 0.0;
        for (j, &coeff) in proto.iter().enumerate() {
            let idx = (self.down_pos + down_len - j) % down_len;
            out += coeff * self.down_hist[idx];
        }
        flush_denormal(out)
    }

    /// Returns `x` delayed by the dry-path delay length.
    fn dry_delayed(&mut self, x: Sample) -> Sample {
        let len = self.dry_delay.len();
        if len == 0 {
            return x;
        }
        let out = self.dry_delay[self.dry_pos];
        self.dry_delay[self.dry_pos] = x;
        self.dry_pos = (self.dry_pos + 1) % len;
        out
    }

    /// Applies the one-pole DC blocker `y = x - x[-1] + coeff * y[-1]`.
    fn dc_block(&mut self, x: Sample, coeff: Sample) -> Sample {
        let y = x - self.dc_x1 + coeff * self.dc_y1;
        self.dc_x1 = x;
        self.dc_y1 = flush_denormal(y);
        self.dc_y1
    }
}

/// A multi-curve saturation processor with asymmetric bias, optional
/// oversampling, and a DC blocker (input port 0 -> output port 0).
///
/// # Examples
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{SaturationCurve, SaturationNode, SaturationParams};
///
/// let params = SaturationParams {
///     curve: SaturationCurve::Tanh,
///     drive: 4.0,
///     mix: 1.0,
///     dc_block: false,
///     ..Default::default()
/// };
/// let mut node = SaturationNode::new(48_000, 1, params);
/// assert_eq!(node.oversample().factor(), 1);
///
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 1);
/// input.set_active_frames(1);
/// input.channel_mut(0)[0] = 1.0;
/// let output = AudioBuffer::new(ChannelLayout::Mono, 1);
///
/// let inputs = [input];
/// let mut outputs = [output];
/// let ctx = RenderContext { sample_rate: 48_000, frames: 1, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // A driven input is soft-clipped below unity (tanh(4.0) ~= 0.999).
/// let [out] = outputs;
/// assert!(out.channel(0)[0] < 1.0 && out.channel(0)[0] > 0.9);
/// ```
#[derive(Debug, Clone)]
pub struct SaturationNode {
    /// Selected transfer function.
    curve: SaturationCurve,
    /// Selected internal processing rate.
    oversample: Oversample,
    /// Whether the DC blocker is active.
    dc_block: bool,
    /// DC-blocker pole.
    dc_coeff: Sample,
    /// Pre-shaping drive gain.
    drive: Smoothed,
    /// Asymmetry bias.
    bias: Smoothed,
    /// Post-shaping make-up gain.
    output_gain: Smoothed,
    /// Wet/dry blend in `[0, 1]`.
    mix: Smoothed,
    /// Decimation prototype filter (unity DC gain). Empty at `X1`.
    proto: Vec<Sample>,
    /// Polyphase interpolation branches. Empty at `X1`.
    phases: Vec<Vec<Sample>>,
    /// Per-channel filter state.
    channels: Vec<ChannelState>,
    /// Reported processing latency in host-rate frames.
    latency: u32,
    /// Length of the prototype FIR filter (`0` at `X1`).
    filter_taps: usize,
}

impl SaturationNode {
    /// Builds a saturation node for a `channels`-wide signal.
    ///
    /// `sample_rate` is accepted for API symmetry; the oversampling filters are
    /// designed in normalised frequency and do not depend on it.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: SaturationParams) -> Self {
        let _ = sample_rate;
        let factor = params.oversample.factor();

        let (proto, phases, filter_taps) = if factor > 1 {
            let taps = TAPS_PER_PHASE * factor;
            let proto = design_lowpass(taps, factor);
            let mut phases = Vec::with_capacity(factor);
            for p in 0..factor {
                let mut phase = Vec::with_capacity(TAPS_PER_PHASE);
                let mut k = 0usize;
                while p + k * factor < taps {
                    phase.push(factor as Sample * proto[p + k * factor]);
                    k += 1;
                }
                phases.push(phase);
            }
            (proto, phases, taps)
        } else {
            (Vec::new(), Vec::new(), 0)
        };

        let latency = if factor > 1 {
            ops::round((filter_taps as Sample - 1.0) / factor as Sample) as u32
        } else {
            0
        };

        let up_len = if factor > 1 { TAPS_PER_PHASE } else { 0 };
        let mut states = Vec::with_capacity(channels);
        for _ in 0..channels {
            states.push(ChannelState::new(up_len, filter_taps, latency as usize));
        }

        Self {
            curve: params.curve,
            oversample: params.oversample,
            dc_block: params.dc_block,
            dc_coeff: DEFAULT_DC_BLOCK_COEFF,
            drive: Smoothed::new(params.drive),
            bias: Smoothed::new(params.bias),
            output_gain: Smoothed::new(params.output_gain),
            mix: Smoothed::new(params.mix),
            proto,
            phases,
            channels: states,
            latency,
            filter_taps,
        }
    }

    /// Sets a new drive gain, gliding toward it with `ramp`.
    #[inline]
    pub fn set_drive(&mut self, target: Sample, ramp: Ramp) {
        self.drive.set_target(target, ramp);
    }

    /// Sets a new asymmetry bias, gliding toward it with `ramp`.
    #[inline]
    pub fn set_bias(&mut self, target: Sample, ramp: Ramp) {
        self.bias.set_target(target, ramp);
    }

    /// Sets a new make-up gain, gliding toward it with `ramp`.
    #[inline]
    pub fn set_output_gain(&mut self, target: Sample, ramp: Ramp) {
        self.output_gain.set_target(target, ramp);
    }

    /// Sets a new wet/dry blend in `[0, 1]`, gliding toward it with `ramp`.
    #[inline]
    pub fn set_mix(&mut self, target: Sample, ramp: Ramp) {
        self.mix.set_target(target, ramp);
    }

    /// Switches to a different transfer function. The change takes effect on the
    /// next processed sample.
    #[inline]
    pub fn set_curve(&mut self, curve: SaturationCurve) {
        self.curve = curve;
    }

    /// Enables or disables the output DC blocker.
    #[inline]
    pub fn set_dc_block(&mut self, enabled: bool) {
        self.dc_block = enabled;
    }

    /// Returns the active transfer function.
    #[inline]
    #[must_use]
    pub fn curve(&self) -> SaturationCurve {
        self.curve
    }

    /// Returns the configured oversampling mode.
    #[inline]
    #[must_use]
    pub fn oversample(&self) -> Oversample {
        self.oversample
    }

    /// Returns whether the DC blocker is active.
    #[inline]
    #[must_use]
    pub fn dc_block(&self) -> bool {
        self.dc_block
    }

    /// Returns the length of the oversampling prototype FIR filter, or `0` at
    /// `X1`.
    #[inline]
    #[must_use]
    pub fn filter_taps(&self) -> usize {
        self.filter_taps
    }
}

impl AudioNode for SaturationNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels()).min(self.channels.len());
        let frames = output.active_frames().min(input.active_frames());
        let factor = self.oversample.factor();
        let curve = self.curve;
        let dc_on = self.dc_block;
        let dc_coeff = self.dc_coeff;

        // Snapshot the smoothers so every channel replays the identical
        // per-sample control trajectory.
        let drive0 = self.drive;
        let bias0 = self.bias;
        let gain0 = self.output_gain;
        let mix0 = self.mix;

        let proto = &self.proto;
        let phases = &self.phases;
        let states = &mut self.channels;

        let mut committed: Option<(Smoothed, Smoothed, Smoothed, Smoothed)> = None;

        for (ch, state) in states.iter_mut().enumerate().take(channels) {
            let mut drive = drive0;
            let mut bias = bias0;
            let mut gain = gain0;
            let mut mix = mix0;

            let src = input.channel(ch);
            let dst = output.channel_mut(ch);

            for i in 0..frames {
                // Treat non-finite inputs as silence so a stray NaN/inf never
                // propagates through the non-linearity.
                let x = if src[i].is_finite() { src[i] } else { 0.0 };
                let dv = drive.next_sample();
                let bv = bias.next_sample();
                let gv = gain.next_sample();
                let mv = mix.next_sample().clamp(0.0, 1.0);

                let shaped_raw = if factor > 1 {
                    state.shape_oversampled(x, dv, bv, curve, phases, proto)
                } else {
                    flush_denormal(shaped_biased(curve, dv * x, bv))
                };
                let shaped = if dc_on {
                    state.dc_block(shaped_raw, dc_coeff)
                } else {
                    shaped_raw
                };
                let dry_sig = state.dry_delayed(x);
                dst[i] = flush_denormal(mv * (shaped * gv) + (1.0 - mv) * dry_sig);
            }

            if ch + 1 == channels {
                committed = Some((drive, bias, gain, mix));
            }
        }

        if let Some((drive, bias, gain, mix)) = committed {
            self.drive = drive;
            self.bias = bias;
            self.output_gain = gain;
            self.mix = mix;
        }
    }

    fn reset(&mut self) {
        for state in &mut self.channels {
            state.reset();
        }
        self.drive = Smoothed::new(self.drive.target());
        self.bias = Smoothed::new(self.bias.target());
        self.output_gain = Smoothed::new(self.output_gain.target());
        self.mix = Smoothed::new(self.mix.target());
    }

    fn latency_frames(&self) -> u32 {
        self.latency
    }
}

/// Allocates a zero-filled sample vector of length `n`.
fn zeroed(n: usize) -> Vec<Sample> {
    let mut v = Vec::with_capacity(n);
    v.resize(n, 0.0);
    v
}

/// Designs a `taps`-long, unity-DC-gain, Hann-windowed sinc low-pass prototype.
///
/// The cutoff is the host-rate Nyquist, i.e. a normalised cutoff of
/// `1/(2*factor)` cycles/sample at the oversampled rate, so the filter both
/// reconstructs interpolated samples and rejects the images/aliases created by
/// the non-linearity.
fn design_lowpass(taps: usize, factor: usize) -> Vec<Sample> {
    let center = (taps - 1) as Sample / 2.0;
    let cutoff = 1.0 / factor as Sample;
    let denom = (taps - 1) as Sample;

    let mut h = Vec::with_capacity(taps);
    let mut sum = 0.0;
    for n in 0..taps {
        let t = n as Sample - center;
        let ideal = if t == 0.0 {
            cutoff
        } else {
            let arg = PI * cutoff * t;
            cutoff * (ops::sin(arg) / arg)
        };
        let window = 0.5 - 0.5 * ops::cos(2.0 * PI * n as Sample / denom);
        let coeff = ideal * window;
        h.push(coeff);
        sum += coeff;
    }

    let inv = if sum != 0.0 { 1.0 / sum } else { 1.0 };
    for c in &mut h {
        *c *= inv;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    /// Runs `node` over a single mono buffer and returns the output samples.
    fn run_mono(node: &mut SaturationNode, data: &[Sample]) -> Vec<Sample> {
        let frames = data.len();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames.max(1));
        input.set_active_frames(frames);
        input.channel_mut(0)[..frames].copy_from_slice(data);
        let output = AudioBuffer::new(ChannelLayout::Mono, frames.max(1));

        let inputs = [input];
        let mut outputs = [output];
        let c = ctx(48_000, frames);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);
        let [out] = outputs;
        out.channel(0)[..frames].to_vec()
    }

    fn node(curve: SaturationCurve, drive: Sample, bias: Sample, mix: Sample, dc: bool) -> SaturationNode {
        SaturationNode::new(
            48_000,
            1,
            SaturationParams {
                curve,
                oversample: Oversample::X1,
                drive,
                bias,
                output_gain: 1.0,
                mix,
                dc_block: dc,
            },
        )
    }

    #[test]
    fn default_params_are_neutral_tanh() {
        let p = SaturationParams::default();
        assert_eq!(p.curve, SaturationCurve::Tanh);
        assert_eq!(p.oversample, Oversample::X1);
        assert_eq!(p.drive, 1.0);
        assert_eq!(p.bias, 0.0);
        assert_eq!(p.output_gain, 1.0);
        assert_eq!(p.mix, 1.0);
        assert!(p.dc_block);
    }

    #[test]
    fn tanh_curve_matches_reference() {
        let mut n = node(SaturationCurve::Tanh, 1.0, 0.0, 1.0, false);
        let data = [0.1, -0.3, 0.7, -0.9];
        let out = run_mono(&mut n, &data);
        for (i, &x) in data.iter().enumerate() {
            assert!((out[i] - ops::tanh(x)).abs() < 1e-6, "tanh mismatch at {i}");
        }
    }

    #[test]
    fn reciprocal_curve_matches_formula() {
        let mut n = node(SaturationCurve::Reciprocal, 1.0, 0.0, 1.0, false);
        let data = [0.5, -2.0, 4.0];
        let out = run_mono(&mut n, &data);
        for (i, &x) in data.iter().enumerate() {
            let want = x / (1.0 + x.abs());
            assert!((out[i] - want).abs() < 1e-6, "reciprocal mismatch at {i}");
        }
    }

    #[test]
    fn cubic_hard_limits_to_unity() {
        let mut n = node(SaturationCurve::Cubic, 1.0, 0.0, 1.0, false);
        let out = run_mono(&mut n, &[5.0, -5.0]);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn arctan_curve_approaches_unity() {
        let mut n = node(SaturationCurve::Arctan, 1.0, 0.0, 1.0, false);
        let out = run_mono(&mut n, &[1000.0]);
        assert!(out[0] > 0.99 && out[0] < 1.0 + 1e-6, "arctan not normalised: {}", out[0]);
    }

    #[test]
    fn sine_curve_peaks_at_unity() {
        let mut n = node(SaturationCurve::Sine, 1.0, 0.0, 1.0, false);
        // Clamped input of 1.0 -> sin(pi/2) = 1.
        let out = run_mono(&mut n, &[1.0, 2.0]);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn curves_saturate_below_unity_for_hot_input() {
        for curve in [
            SaturationCurve::Tanh,
            SaturationCurve::Arctan,
            SaturationCurve::Reciprocal,
        ] {
            let mut n = node(curve, 1.0, 0.0, 1.0, false);
            let out = run_mono(&mut n, &[50.0, -50.0]);
            assert!(out[0] <= 1.0 && out[0] > 0.9, "curve {curve:?} high: {}", out[0]);
            assert!(out[1] >= -1.0 && out[1] < -0.9, "curve {curve:?} low: {}", out[1]);
        }
    }

    #[test]
    fn silent_input_with_bias_stays_zero() {
        // f(bias) - f(bias) == 0, so a biased curve does not offset silence.
        let mut n = node(SaturationCurve::Tanh, 1.0, 0.5, 1.0, false);
        let out = run_mono(&mut n, &[0.0, 0.0, 0.0]);
        for &y in &out {
            assert!(y.abs() < 1e-6, "bias leaked DC into silence: {y}");
        }
    }

    #[test]
    fn bias_breaks_odd_symmetry() {
        let mut n = node(SaturationCurve::Tanh, 2.0, 0.4, 1.0, false);
        let out = run_mono(&mut n, &[0.5, -0.5]);
        // With a non-zero bias the positive and negative responses differ in
        // magnitude (even harmonics appear).
        assert!((out[0].abs() - out[1].abs()).abs() > 1e-3, "bias did not break symmetry");
    }

    #[test]
    fn dc_blocker_removes_asymmetric_dc() {
        // A sine through a strongly biased curve pumps out DC; the blocker
        // should drive the long-run mean toward zero.
        let freq = 200.0;
        let sr = 48_000.0;
        let mut data = Vec::with_capacity(16_384);
        for i in 0..16_384 {
            data.push(0.8 * ops::sin(2.0 * PI * freq * (i as Sample) / sr));
        }

        let mut with_block = node(SaturationCurve::Tanh, 3.0, 0.6, 1.0, true);
        let mut without_block = node(SaturationCurve::Tanh, 3.0, 0.6, 1.0, false);
        let out_on = run_mono(&mut with_block, &data);
        let out_off = run_mono(&mut without_block, &data);

        // Measure the mean over the settled tail.
        let tail = 4800;
        let mean = |v: &[Sample]| -> Sample {
            let s: Sample = v[v.len() - tail..].iter().copied().sum();
            s / tail as Sample
        };
        let mean_on = mean(&out_on).abs();
        let mean_off = mean(&out_off).abs();
        assert!(mean_off > 0.01, "expected DC without blocker: {mean_off}");
        assert!(mean_on < mean_off * 0.1, "blocker did not remove DC: on={mean_on} off={mean_off}");
    }

    #[test]
    fn mix_blends_dry_and_wet() {
        let data = [0.5, -0.5, 0.8];
        let mut dry = node(SaturationCurve::Tanh, 4.0, 0.0, 0.0, false);
        let mut wet = node(SaturationCurve::Tanh, 4.0, 0.0, 1.0, false);
        let mut half = node(SaturationCurve::Tanh, 4.0, 0.0, 0.5, false);
        let d = run_mono(&mut dry, &data);
        let w = run_mono(&mut wet, &data);
        let h = run_mono(&mut half, &data);
        for i in 0..data.len() {
            assert!((d[i] - data[i]).abs() < 1e-6, "dry mix should pass input");
            let expect = 0.5 * w[i] + 0.5 * d[i];
            assert!((h[i] - expect).abs() < 1e-6, "half mix mismatch at {i}");
        }
    }

    #[test]
    fn output_gain_scales_wet_signal() {
        let mut unity = node(SaturationCurve::Tanh, 1.0, 0.0, 1.0, false);
        let mut doubled = SaturationNode::new(
            48_000,
            1,
            SaturationParams {
                output_gain: 2.0,
                dc_block: false,
                ..Default::default()
            },
        );
        let data = [0.1, -0.2];
        let u = run_mono(&mut unity, &data);
        let g = run_mono(&mut doubled, &data);
        for i in 0..data.len() {
            assert!((g[i] - 2.0 * u[i]).abs() < 1e-6, "gain mismatch at {i}");
        }
    }

    #[test]
    fn higher_drive_saturates_more() {
        let data = [0.3];
        let mut low = node(SaturationCurve::Tanh, 1.0, 0.0, 1.0, false);
        let mut high = node(SaturationCurve::Tanh, 5.0, 0.0, 1.0, false);
        let l = run_mono(&mut low, &data);
        let h = run_mono(&mut high, &data);
        assert!(h[0] > l[0], "more drive should push the output higher");
    }

    #[test]
    fn oversampling_reports_latency() {
        let n2 = SaturationNode::new(48_000, 1, SaturationParams { oversample: Oversample::X2, ..Default::default() });
        let n4 = SaturationNode::new(48_000, 1, SaturationParams { oversample: Oversample::X4, ..Default::default() });
        assert!(n2.filter_taps() > 0 && n2.latency_frames() > 0);
        assert!(n4.filter_taps() > n2.filter_taps());
    }

    #[test]
    fn x1_has_no_latency() {
        let n = SaturationNode::new(48_000, 1, SaturationParams::default());
        assert_eq!(n.latency_frames(), 0);
        assert_eq!(n.filter_taps(), 0);
    }

    #[test]
    fn oversampled_output_is_finite_and_bounded() {
        let sr = 48_000.0;
        let mut data = Vec::with_capacity(512);
        for i in 0..512 {
            data.push(2.0 * ops::sin(2.0 * PI * 9000.0 * (i as Sample) / sr));
        }
        let mut n = SaturationNode::new(
            48_000,
            1,
            SaturationParams { curve: SaturationCurve::Tanh, oversample: Oversample::X4, drive: 6.0, ..Default::default() },
        );
        let out = run_mono(&mut n, &data);
        for &y in &out {
            assert!(y.is_finite() && y.abs() <= 1.5, "oversampled output out of range: {y}");
        }
    }

    #[test]
    fn reset_restores_deterministic_output() {
        let mut n = node(SaturationCurve::Tanh, 3.0, 0.5, 1.0, true);
        let data = [0.4, -0.6, 0.2, 0.9, -0.3];
        let first = run_mono(&mut n, &data);
        n.reset();
        let second = run_mono(&mut n, &data);
        for i in 0..data.len() {
            assert!((first[i] - second[i]).abs() < 1e-6, "reset not deterministic at {i}");
        }
    }

    #[test]
    fn channels_are_processed_independently() {
        let mut n = SaturationNode::new(48_000, 2, SaturationParams { drive: 4.0, dc_block: false, ..Default::default() });
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 3);
        input.set_active_frames(3);
        input.channel_mut(0).copy_from_slice(&[0.5, -0.5, 0.9]);
        input.channel_mut(1).copy_from_slice(&[0.0, 0.0, 0.0]);
        let output = AudioBuffer::new(ChannelLayout::Stereo, 3);
        let inputs = [input];
        let mut outputs = [output];
        let c = ctx(48_000, 3);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        n.process(&c, &mut io);
        let [out] = outputs;
        // Right channel fed silence stays silent (no cross-channel bleed).
        for &y in out.channel(1) {
            assert!(y.abs() < 1e-6, "channel bleed: {y}");
        }
        // Left channel is shaped.
        assert!(out.channel(0)[0].abs() > 0.0);
    }

    #[test]
    fn non_finite_input_is_safe() {
        let mut n = node(SaturationCurve::Tanh, 2.0, 0.0, 1.0, true);
        let out = run_mono(&mut n, &[Sample::NAN, Sample::INFINITY, -Sample::INFINITY, 0.3]);
        for &y in &out {
            assert!(y.is_finite(), "non-finite leaked: {y}");
        }
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut n = node(SaturationCurve::Tanh, 2.0, 0.0, 1.0, true);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
        input.set_active_frames(0);
        let output = AudioBuffer::new(ChannelLayout::Mono, 8);
        let inputs = [input];
        let mut outputs = [output];
        let c = ctx(48_000, 0);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        n.process(&c, &mut io);
        // No panic and nothing written (the pre-zeroed buffer is untouched).
        let [out] = outputs;
        for &y in out.channel(0) {
            assert_eq!(y, 0.0);
        }
    }

    #[test]
    fn set_curve_switches_transfer_function() {
        let mut n = node(SaturationCurve::Tanh, 1.0, 0.0, 1.0, false);
        assert_eq!(n.curve(), SaturationCurve::Tanh);
        n.set_curve(SaturationCurve::Reciprocal);
        assert_eq!(n.curve(), SaturationCurve::Reciprocal);
        let out = run_mono(&mut n, &[2.0]);
        assert!((out[0] - 2.0 / 3.0).abs() < 1e-6, "did not switch to reciprocal");
    }
}
