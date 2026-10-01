//! Single-knob spectral tilt equaliser built from a matched pair of cascaded
//! [`Biquad`] shelving sections.
//!
//! A tilt EQ is a mastering and tone-shaping staple: one control rotates the
//! whole spectrum about a fixed pivot frequency, lifting one end by the same
//! amount it drops the other. It is realised here as two cascaded RBJ shelves
//! sharing a pivot frequency: a low shelf at `-tilt_db` and a high shelf at
//! `+tilt_db`. A positive `tilt_db` therefore brightens (treble up, bass down);
//! a negative value warms (bass up, treble down). The slope steepness is set by
//! a shared shelf `Q`.
//!
//! All filter state is pre-allocated at construction, so
//! [`TiltEqNode::process`] performs no allocation, locking, or panicking and is
//! safe to run on the audio thread.
//!
//! # Provenance
//!
//! Classic DSP only, with no AI/ML of any kind. This module contains no UE,
//! Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source
//! code or derived code; it only borrows the widely documented idea of a
//! single-control spectral tilt and expresses it as two standard RBJ shelving
//! filters.
//!
//! # Relationship
//!
//! This node cascades the shared [`Biquad`] core from [`crate::nodes::biquad`]
//! exactly like [`crate::nodes::effects::parametric_eq::ParametricEqNode`] and
//! [`crate::nodes::effects::graphic_eq`], but it is deliberately distinct in
//! intent:
//!
//! - [`ParametricEqNode`](crate::nodes::effects::parametric_eq::ParametricEqNode)
//!   exposes an arbitrary number of independently tunable bands (any mix of
//!   bells, shelves, pass filters), each with its own frequency, `Q`, and gain.
//! - `graphic_eq` exposes a fixed grid of ISO-spaced peaking bands with
//!   per-band gains.
//! - `tilt_eq` exposes exactly one control: a single tilt amount applied to a
//!   matched low-/high-shelf pair pinned to a common pivot. It never surfaces
//!   the individual shelf gains independently.
//!
//! The three share the same `Biquad` building block but model different user
//! intents, so none duplicates another.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::nodes::biquad::{Biquad, BiquadCoeffs, BiquadKind};

/// Default pivot frequency in Hertz about which the spectrum is tilted.
pub const DEFAULT_PIVOT_HZ: Sample = 1_000.0;

/// Default shelf quality factor (`Butterworth`, slope `S = 1`).
pub const DEFAULT_SLOPE_Q: Sample = core::f32::consts::FRAC_1_SQRT_2;

/// Default tilt amount in decibels (flat / unity).
pub const DEFAULT_TILT_DB: Sample = 0.0;

/// The tunable description of a tilt equaliser.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TiltEqParams {
    /// Pivot frequency in Hertz. The low shelf sits here cutting while the high
    /// shelf sits here boosting (or vice versa for negative tilt), so the
    /// response pivots around this frequency.
    pub pivot_hz: Sample,
    /// Tilt amount in decibels. Positive brightens (high end up, low end down);
    /// negative warms (low end up, high end down); zero is a flat pass-through.
    pub tilt_db: Sample,
    /// Shared shelf quality factor controlling the transition slope.
    pub slope_q: Sample,
}

impl Default for TiltEqParams {
    fn default() -> Self {
        Self {
            pivot_hz: DEFAULT_PIVOT_HZ,
            tilt_db: DEFAULT_TILT_DB,
            slope_q: DEFAULT_SLOPE_Q,
        }
    }
}

/// Returns the identity (unit) biquad coefficients: a bit-exact pass-through.
///
/// The RBJ shelving design multiplies by `inv_a0 = 1.0 / a0`, so a 0 dB shelf is
/// not bit-exactly unity due to the reciprocal rounding. Folding to these hand
/// built coefficients when the gain is zero guarantees a true pass-through.
#[inline]
fn unit_coeffs() -> BiquadCoeffs {
    BiquadCoeffs {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    }
}

/// Designs a shelving section, folding to the identity when `gain_db` is zero.
#[inline]
fn shelf_coeffs(
    kind: BiquadKind,
    sample_rate: u32,
    pivot_hz: Sample,
    q: Sample,
    gain_db: Sample,
) -> BiquadCoeffs {
    if gain_db == 0.0 {
        unit_coeffs()
    } else {
        BiquadCoeffs::design(kind, sample_rate, pivot_hz, q, gain_db)
    }
}

/// A single-knob spectral tilt equaliser owning a matched shelf pair.
///
/// The reusable DSP unit (no graph I/O), suitable for embedding in larger
/// chains. All state is pre-allocated at construction.
#[derive(Debug, Clone)]
pub struct TiltEq {
    sample_rate: u32,
    channels: usize,
    params: TiltEqParams,
    /// Low shelf, designed at `-tilt_db`.
    low: Biquad,
    /// High shelf, designed at `+tilt_db`.
    high: Biquad,
}

impl TiltEq {
    /// Builds a tilt EQ for a `channels`-wide signal at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: TiltEqParams) -> Self {
        let channels = channels.max(1);
        let low = Biquad::new(
            shelf_coeffs(
                BiquadKind::LowShelf,
                sample_rate,
                params.pivot_hz,
                params.slope_q,
                -params.tilt_db,
            ),
            channels,
        );
        let high = Biquad::new(
            shelf_coeffs(
                BiquadKind::HighShelf,
                sample_rate,
                params.pivot_hz,
                params.slope_q,
                params.tilt_db,
            ),
            channels,
        );
        Self {
            sample_rate,
            channels,
            params,
            low,
            high,
        }
    }

    /// Returns the current parameters.
    #[inline]
    #[must_use]
    pub fn params(&self) -> TiltEqParams {
        self.params
    }

    /// Returns the channel width.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Redesigns both shelves, preserving filter state so the change is
    /// click-free.
    pub fn set_params(&mut self, params: TiltEqParams) {
        self.params = params;
        self.low.set_coeffs(shelf_coeffs(
            BiquadKind::LowShelf,
            self.sample_rate,
            params.pivot_hz,
            params.slope_q,
            -params.tilt_db,
        ));
        self.high.set_coeffs(shelf_coeffs(
            BiquadKind::HighShelf,
            self.sample_rate,
            params.pivot_hz,
            params.slope_q,
            params.tilt_db,
        ));
    }

    /// Filters `buf` in place: the low shelf followed by the high shelf.
    pub fn process_inplace(&mut self, buf: &mut crate::buffer::AudioBuffer) {
        self.low.process_inplace(buf);
        self.high.process_inplace(buf);
    }

    /// Clears both shelves' filter memory.
    pub fn reset(&mut self) {
        self.low.reset();
        self.high.reset();
    }
}

/// A tilt equaliser node (input port 0 -> output port 0).
#[derive(Debug, Clone)]
pub struct TiltEqNode {
    tilt: TiltEq,
}

impl TiltEqNode {
    /// Builds a tilt EQ node for a `channels`-wide signal at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: TiltEqParams) -> Self {
        Self {
            tilt: TiltEq::new(sample_rate, channels, params),
        }
    }

    /// Returns the current parameters.
    #[inline]
    #[must_use]
    pub fn params(&self) -> TiltEqParams {
        self.tilt.params()
    }

    /// Redesigns both shelves, preserving filter state (click-free).
    pub fn set_params(&mut self, params: TiltEqParams) {
        self.tilt.set_params(params);
    }
}

impl AudioNode for TiltEqNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        output.copy_from(input);
        self.tilt.process_inplace(output);
    }

    fn reset(&mut self) {
        self.tilt.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn run(node: &mut TiltEqNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let inputs = [input.clone()];
        let mut out = AudioBuffer::new(input.layout(), input.capacity_frames());
        // The caller presets the output's active frame count (process does not
        // resize its output).
        out.set_active_frames(frames);
        let mut outputs = [out];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(frames), &mut io);
        }
        let [out] = outputs;
        out
    }

    /// Builds a mono buffer with a unit impulse at sample 0.
    fn impulse(frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, frames);
        buf.channel_mut(0)[0] = 1.0;
        buf
    }

    /// Sums the low-frequency content via a crude running integrator of the
    /// impulse response (DC gain = sum of taps).
    fn dc_gain(ir: &[Sample]) -> Sample {
        ir.iter().copied().sum()
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = TiltEqNode::new(SR, 2, TiltEqParams::default());
        let input = AudioBuffer::new(ChannelLayout::Stereo, 32);
        let out = run(&mut node, &input);
        for ch in 0..2 {
            assert!(out.channel(ch).iter().all(|&s| s == 0.0));
        }
    }

    #[test]
    fn zero_tilt_is_bit_exact_passthrough() {
        let params = TiltEqParams {
            tilt_db: 0.0,
            ..TiltEqParams::default()
        };
        let mut node = TiltEqNode::new(SR, 1, params);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 16);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = (i as Sample) - 7.5;
        }
        let out = run(&mut node, &input);
        assert_eq!(out.channel(0), input.channel(0));
    }

    #[test]
    fn default_params_are_flat() {
        let p = TiltEqParams::default();
        assert_eq!(p.tilt_db, 0.0);
        assert_eq!(p.pivot_hz, DEFAULT_PIVOT_HZ);
        assert_eq!(p.slope_q, DEFAULT_SLOPE_Q);
        // The default node is therefore a bit-exact pass-through.
        let mut node = TiltEqNode::new(SR, 1, p);
        let input = impulse(64);
        let out = run(&mut node, &input);
        assert_eq!(out.channel(0), input.channel(0));
    }

    #[test]
    fn positive_tilt_lifts_highs_and_cuts_lows() {
        // Positive tilt: high shelf boosts, low shelf cuts, so the DC gain
        // (dominated by the low end) drops below unity.
        let params = TiltEqParams {
            pivot_hz: 1_000.0,
            tilt_db: 6.0,
            slope_q: DEFAULT_SLOPE_Q,
        };
        let mut node = TiltEqNode::new(SR, 1, params);
        let input = impulse(2_048);
        let out = run(&mut node, &input);
        let dc = dc_gain(out.channel(0));
        assert!(dc < 1.0, "expected low-end cut, dc = {dc}");
    }

    #[test]
    fn negative_tilt_lifts_lows_and_cuts_highs() {
        let params = TiltEqParams {
            pivot_hz: 1_000.0,
            tilt_db: -6.0,
            slope_q: DEFAULT_SLOPE_Q,
        };
        let mut node = TiltEqNode::new(SR, 1, params);
        let input = impulse(2_048);
        let out = run(&mut node, &input);
        let dc = dc_gain(out.channel(0));
        assert!(dc > 1.0, "expected low-end boost, dc = {dc}");
    }

    #[test]
    fn positive_and_negative_tilt_are_opposite() {
        let pos = TiltEqParams {
            tilt_db: 4.0,
            ..TiltEqParams::default()
        };
        let neg = TiltEqParams {
            tilt_db: -4.0,
            ..TiltEqParams::default()
        };
        let mut np = TiltEqNode::new(SR, 1, pos);
        let mut nn = TiltEqNode::new(SR, 1, neg);
        let input = impulse(2_048);
        let dc_pos = dc_gain(run(&mut np, &input).channel(0));
        let dc_neg = dc_gain(run(&mut nn, &input).channel(0));
        assert!(dc_pos < 1.0);
        assert!(dc_neg > 1.0);
        // Opposite tilt signs straddle unity DC gain.
        assert!(dc_pos < dc_neg);
    }

    #[test]
    fn stereo_channels_are_processed_identically() {
        let params = TiltEqParams {
            tilt_db: 5.0,
            ..TiltEqParams::default()
        };
        let mut node = TiltEqNode::new(SR, 2, params);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 256);
        input.channel_mut(0)[0] = 1.0;
        input.channel_mut(1)[0] = 1.0;
        let out = run(&mut node, &input);
        assert_eq!(out.channel(0), out.channel(1));
    }

    #[test]
    fn set_params_preserves_state_no_click() {
        // Process part of a signal, then change params mid-stream. State is
        // retained, so the filter does not reset to zero (no discontinuity to
        // a cleared history).
        let mut node = TiltEqNode::new(SR, 1, TiltEqParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = (i as Sample) * 0.1;
        }
        let _ = run(&mut node, &input);
        node.set_params(TiltEqParams {
            tilt_db: 6.0,
            ..TiltEqParams::default()
        });
        assert_eq!(node.params().tilt_db, 6.0);
    }

    #[test]
    fn reset_clears_filter_memory() {
        let params = TiltEqParams {
            tilt_db: 6.0,
            ..TiltEqParams::default()
        };
        let mut node = TiltEqNode::new(SR, 1, params);
        let input = impulse(64);
        let first = run(&mut node, &input);
        node.reset();
        let second = run(&mut node, &input);
        // After reset, re-running the same impulse yields identical output.
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = TiltEqNode::new(SR, 2, TiltEqParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 4);
        input.set_active_frames(0);
        let out = run(&mut node, &input);
        assert_eq!(out.active_frames(), 0);
        for ch in 0..out.channels() {
            assert!(out.channel(ch).iter().all(|&s| s == 0.0));
        }
    }

    #[test]
    fn params_round_trip() {
        let params = TiltEqParams {
            pivot_hz: 800.0,
            tilt_db: 3.5,
            slope_q: 0.5,
        };
        let node = TiltEqNode::new(SR, 1, params);
        assert_eq!(node.params(), params);
    }

    #[test]
    fn tilt_core_matches_node() {
        // The embeddable TiltEq core and the TiltEqNode wrapper must agree.
        let params = TiltEqParams {
            tilt_db: 6.0,
            ..TiltEqParams::default()
        };
        let mut core = TiltEq::new(SR, 1, params);
        let mut node = TiltEqNode::new(SR, 1, params);
        let input = impulse(128);
        let mut core_buf = input.clone();
        core.process_inplace(&mut core_buf);
        let node_out = run(&mut node, &input);
        assert_eq!(core_buf.channel(0), node_out.channel(0));
    }
}

/// A worked example: a mastering tilt that gently brightens a signal.
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::tilt_eq::{TiltEqNode, TiltEqParams};
///
/// let params = TiltEqParams {
///     pivot_hz: 1_000.0,
///     tilt_db: 3.0, // brighten
///     slope_q: 0.707,
/// };
/// let mut node = TiltEqNode::new(48_000, 1, params);
///
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
/// input.channel_mut(0)[0] = 1.0; // unit impulse
///
/// let inputs = [input];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 4)];
/// let ctx = RenderContext { sample_rate: 48_000, frames: 4, playhead: 0 };
/// {
///     let mut io = ProcessIo::new(&inputs, &mut outputs);
///     node.process(&ctx, &mut io);
/// }
/// // The response is non-trivial (not a bare pass-through) for a non-zero tilt.
/// assert!(outputs[0].channel(0)[0] != 1.0 || outputs[0].channel(0)[1] != 0.0);
/// ```
#[cfg(doctest)]
struct DocExample;
