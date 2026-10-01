//! Mid-Side (M/S) matrix: a pure sum-and-difference stereo encoder / decoder
//! with independent mid and side trim gains.
//!
//! This node performs only the lossless linear transform between the ordinary
//! left/right (`L` / `R`) representation and the mid/side (`M` / `S`)
//! representation, so later nodes in a chain can process the centre and the
//! sides independently:
//!
//! ```text
//! encode:  M = (L + R) / 2        S = (L - R) / 2
//! decode:  L =  M + S             R =  M - S
//! ```
//!
//! A typical mastering insert is `encode -> (treat M and S separately) ->
//! decode`. In [`MidSideMode::Encode`] the node consumes an `L` / `R` pair and
//! emits `M` on channel 0 and `S` on channel 1. In [`MidSideMode::Decode`] it
//! consumes an `M` / `S` pair and reconstructs `L` on channel 0 and `R` on
//! channel 1. With both trim gains at unity, an encode followed by a decode is
//! an exact identity (to floating-point rounding).
//!
//! The mid and side trim gains let an engineer rebalance centre versus sides.
//! In encode mode they scale the produced `M` and `S` channels; in decode mode
//! they scale the incoming `M` and `S` before the sum/difference. Both gains
//! are [`Smoothed`] so automation is click-free.
//!
//! # Model
//!
//! The `(L, R) -> (M, S)` convention used here, `M = (L + R) / 2` and
//! `S = (L - R) / 2`, is the balanced engineering convention whose inverse is
//! simply `L = M + S`, `R = M - S` (perfect reconstruction). It is not a
//! unitary (energy-preserving) transform; the `1 / 2` scaling keeps a centred
//! mono signal (`L == R`) at unchanged level in `M` while `S` falls to zero.
//!
//! # Real-time contract
//!
//! All state is fixed at construction.
//! [`process`](crate::graph::AudioNode::process) performs no allocation, takes
//! no locks, and cannot panic: mismatched channel counts and zero-length
//! blocks degrade gracefully (a layout with fewer than two channels is passed
//! through unchanged, and channels beyond the processed stereo pair are copied
//! through).
//!
//! # Relationship
//!
//! This node is deliberately distinct from
//! [`stereo_width::StereoWidthNode`](crate::nodes::effects::stereo_width): the
//! stereo widener always does a full internal round trip (`L` / `R` in, `L` /
//! `R` out) applying a single `width` scale to the side plus an optional
//! bass-mono crossover, and never exposes `M` and `S` on its ports. This
//! matrix node instead exposes the raw `M` / `S` signals (encode) or rebuilds
//! `L` / `R` from them (decode) so arbitrary processors can be inserted between
//! the two stages. It reuses this crate's own [`Smoothed`] parameter and
//! [`db_to_linear`] helper.
//!
//! # Provenance
//!
//! Mid-Side (sum-and-difference) stereo is public, standard technique dating
//! back to Alan Blumlein's 1931 stereo patent and is described in every
//! audio-engineering text. This module contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or
//! derived code**; it is implemented purely from the widely documented M/S
//! matrix formulas.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Time constant for the mid / side trim-gain smoothers, in seconds.
///
/// Ten milliseconds is short enough to feel immediate yet long enough to avoid
/// zipper noise when a trim gain is automated.
pub const GAIN_SMOOTH_SECONDS: Sample = 0.01;

/// Direction of the Mid-Side matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum MidSideMode {
    /// Consume `L` / `R` on the input ports and emit `M` / `S`.
    #[default]
    Encode,
    /// Consume `M` / `S` on the input ports and reconstruct `L` / `R`.
    Decode,
}

/// Construction parameters for a [`MidSideMatrixNode`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MidSideMatrixParams {
    /// Matrix direction (encode `L` / `R` -> `M` / `S` or decode back).
    pub mode: MidSideMode,
    /// Trim applied to the mid component, in decibels. Defaults to `0` (unity).
    pub mid_gain_db: Sample,
    /// Trim applied to the side component, in decibels. Defaults to `0`
    /// (unity).
    pub side_gain_db: Sample,
}

/// A pure Mid-Side encoder / decoder with independent mid and side trim gains.
///
/// The trim gains are [`Smoothed`] for click-free automation. Construct with
/// [`MidSideMatrixNode::new`] and retune at runtime with
/// [`MidSideMatrixNode::set_params`].
///
/// # Examples
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::{MidSideMatrixNode, MidSideMatrixParams};
///
/// let mut node = MidSideMatrixNode::new(MidSideMatrixParams::default());
/// let mut input = AudioBuffer::new(ChannelLayout::Stereo, 1);
/// input.channel_mut(0)[0] = 1.0; // L
/// input.channel_mut(1)[0] = 0.0; // R
/// let mut output = AudioBuffer::new(ChannelLayout::Stereo, 1);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 1, playhead: 0 };
/// let inputs = [input];
/// let mut outputs = [output];
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
/// // M = (1 + 0) / 2 = 0.5, S = (1 - 0) / 2 = 0.5.
/// assert!((outputs[0].channel(0)[0] - 0.5).abs() < 1e-6);
/// assert!((outputs[0].channel(1)[0] - 0.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct MidSideMatrixNode {
    mode: MidSideMode,
    mid_gain: Smoothed,
    side_gain: Smoothed,
}

impl MidSideMatrixNode {
    /// Builds a node from `params`. The trim gains start fully settled at their
    /// configured values.
    #[must_use]
    pub fn new(params: MidSideMatrixParams) -> Self {
        Self {
            mode: params.mode,
            mid_gain: Smoothed::new(db_to_linear(params.mid_gain_db)),
            side_gain: Smoothed::new(db_to_linear(params.side_gain_db)),
        }
    }

    /// Updates the matrix direction and retargets the trim gains, ramping them
    /// over [`GAIN_SMOOTH_SECONDS`] at `sample_rate` for a click-free change.
    pub fn set_params(&mut self, params: MidSideMatrixParams, sample_rate: u32) {
        self.mode = params.mode;
        let ramp = Ramp::linear_seconds(GAIN_SMOOTH_SECONDS, sample_rate);
        self.mid_gain
            .set_target(db_to_linear(params.mid_gain_db), ramp);
        self.side_gain
            .set_target(db_to_linear(params.side_gain_db), ramp);
    }

    /// Current matrix direction.
    #[must_use]
    pub fn mode(&self) -> MidSideMode {
        self.mode
    }
}

impl AudioNode for MidSideMatrixNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels());
        let frames = output.active_frames().min(input.active_frames());

        // Fewer than two channels: there is no stereo pair to matrix, so pass
        // whatever exists through unchanged.
        if channels < 2 {
            for ch in 0..channels {
                let src = input.channel(ch);
                let dst = output.channel_mut(ch);
                dst[..frames].copy_from_slice(&src[..frames]);
            }
            return;
        }

        for f in 0..frames {
            let mid_g = self.mid_gain.next_sample();
            let side_g = self.side_gain.next_sample();
            let a = input.channel(0)[f];
            let b = input.channel(1)[f];

            let (o0, o1) = match self.mode {
                MidSideMode::Encode => {
                    let m = mid_g * (0.5 * (a + b));
                    let s = side_g * (0.5 * (a - b));
                    (m, s)
                }
                MidSideMode::Decode => {
                    let m = mid_g * a;
                    let s = side_g * b;
                    (m + s, m - s)
                }
            };

            output.channel_mut(0)[f] = flush_denormal(o0);
            output.channel_mut(1)[f] = flush_denormal(o1);
        }

        // Copy any surplus channels (beyond the processed stereo pair) through.
        for ch in 2..channels {
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            dst[..frames].copy_from_slice(&src[..frames]);
        }
    }

    fn reset(&mut self) {
        self.mid_gain = Smoothed::new(self.mid_gain.target());
        self.side_gain = Smoothed::new(self.side_gain.target());
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

    fn run(node: &mut MidSideMatrixNode, input: AudioBuffer, layout: ChannelLayout) -> AudioBuffer {
        let frames = input.active_frames();
        let output = AudioBuffer::new(layout, input.capacity_frames().max(1));
        let c = ctx(frames);
        let inputs = [input];
        let mut outputs = [output];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&c, &mut io);
        }
        outputs.into_iter().next().unwrap()
    }

    #[test]
    fn encode_produces_mid_and_side() {
        let mut node = MidSideMatrixNode::new(MidSideMatrixParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 4);
        for f in 0..4 {
            input.channel_mut(0)[f] = 0.8; // L
            input.channel_mut(1)[f] = 0.2; // R
        }
        let out = run(&mut node, input, ChannelLayout::Stereo);
        for f in 0..4 {
            assert!((out.channel(0)[f] - 0.5).abs() < 1e-6); // M = (0.8 + 0.2)/2
            assert!((out.channel(1)[f] - 0.3).abs() < 1e-6); // S = (0.8 - 0.2)/2
        }
    }

    #[test]
    fn decode_inverts_encode() {
        let mut encoder = MidSideMatrixNode::new(MidSideMatrixParams::default());
        let mut decoder = MidSideMatrixNode::new(MidSideMatrixParams {
            mode: MidSideMode::Decode,
            ..MidSideMatrixParams::default()
        });

        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 8);
        for f in 0..8 {
            let n = f as Sample;
            input.channel_mut(0)[f] = 0.1 * n - 0.3;
            input.channel_mut(1)[f] = -0.2 * n + 0.5;
        }
        let reference = input.clone();

        let encoded = run(&mut encoder, input, ChannelLayout::Stereo);
        let decoded = run(&mut decoder, encoded, ChannelLayout::Stereo);

        for f in 0..8 {
            assert!((decoded.channel(0)[f] - reference.channel(0)[f]).abs() < 1e-6);
            assert!((decoded.channel(1)[f] - reference.channel(1)[f]).abs() < 1e-6);
        }
    }

    #[test]
    fn centred_mono_has_no_side() {
        let mut node = MidSideMatrixNode::new(MidSideMatrixParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 4);
        for f in 0..4 {
            input.channel_mut(0)[f] = 0.7;
            input.channel_mut(1)[f] = 0.7;
        }
        let out = run(&mut node, input, ChannelLayout::Stereo);
        for f in 0..4 {
            assert!((out.channel(0)[f] - 0.7).abs() < 1e-6); // M unchanged
            assert!(out.channel(1)[f].abs() < 1e-6); // S == 0
        }
    }

    #[test]
    fn side_gain_scales_side_only() {
        // -6.0206 dB halves the side; mid stays put.
        let mut node = MidSideMatrixNode::new(MidSideMatrixParams {
            mode: MidSideMode::Encode,
            mid_gain_db: 0.0,
            side_gain_db: -6.020_6,
        });
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 4);
        for f in 0..4 {
            input.channel_mut(0)[f] = 1.0; // L
            input.channel_mut(1)[f] = -1.0; // R (pure side)
        }
        let out = run(&mut node, input, ChannelLayout::Stereo);
        for f in 0..4 {
            assert!(out.channel(0)[f].abs() < 1e-6); // M = 0
            assert!((out.channel(1)[f] - 0.5).abs() < 1e-3); // S = 1.0 * 0.5
        }
    }

    #[test]
    fn mono_layout_passes_through() {
        let mut node = MidSideMatrixNode::new(MidSideMatrixParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        for f in 0..4 {
            input.channel_mut(0)[f] = 0.42;
        }
        let out = run(&mut node, input, ChannelLayout::Mono);
        for f in 0..4 {
            assert!((out.channel(0)[f] - 0.42).abs() < 1e-6);
        }
    }

    #[test]
    fn surplus_channels_pass_through() {
        let mut node = MidSideMatrixNode::new(MidSideMatrixParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Quad, 2);
        for f in 0..2 {
            input.channel_mut(0)[f] = 0.6; // L
            input.channel_mut(1)[f] = 0.2; // R
            input.channel_mut(2)[f] = 0.9; // surplus
            input.channel_mut(3)[f] = -0.4; // surplus
        }
        let out = run(&mut node, input, ChannelLayout::Quad);
        for f in 0..2 {
            assert!((out.channel(0)[f] - 0.4).abs() < 1e-6); // M
            assert!((out.channel(1)[f] - 0.2).abs() < 1e-6); // S
            assert!((out.channel(2)[f] - 0.9).abs() < 1e-6); // passed through
            assert!((out.channel(3)[f] + 0.4).abs() < 1e-6); // passed through
        }
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = MidSideMatrixNode::new(MidSideMatrixParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 4);
        input.set_active_frames(0);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 4);
        output.set_active_frames(0);
        let c = ctx(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io); // must not panic
        assert_eq!(outputs[0].active_frames(), 0);
    }

    #[test]
    fn reset_snaps_gain_to_target() {
        let mut node = MidSideMatrixNode::new(MidSideMatrixParams::default());
        node.set_params(
            MidSideMatrixParams {
                mode: MidSideMode::Encode,
                mid_gain_db: -12.0,
                side_gain_db: 6.0,
            },
            SR,
        );
        assert!(!node.mid_gain.is_settled());
        node.reset();
        assert!(node.mid_gain.is_settled());
        assert!(node.side_gain.is_settled());
        assert!((node.mid_gain.current() - db_to_linear(-12.0)).abs() < 1e-6);
        assert!((node.side_gain.current() - db_to_linear(6.0)).abs() < 1e-6);
    }

    #[test]
    fn default_params_are_unity_encode() {
        let p = MidSideMatrixParams::default();
        assert_eq!(p.mode, MidSideMode::Encode);
        assert_eq!(p.mid_gain_db, 0.0);
        assert_eq!(p.side_gain_db, 0.0);
        let node = MidSideMatrixNode::new(p);
        assert_eq!(node.mode(), MidSideMode::Encode);
    }

    #[test]
    fn gain_ramp_settles_toward_target() {
        let mut node = MidSideMatrixNode::new(MidSideMatrixParams::default());
        node.set_params(
            MidSideMatrixParams {
                mode: MidSideMode::Encode,
                mid_gain_db: 0.0,
                side_gain_db: -6.0,
            },
            SR,
        );
        // Push one long block; the smoothed side gain should settle within the
        // ramp length (10 ms at 48 kHz is 480 samples).
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 1024);
        for f in 0..1024 {
            input.channel_mut(0)[f] = 1.0;
            input.channel_mut(1)[f] = -1.0;
        }
        let _ = run(&mut node, input, ChannelLayout::Stereo);
        assert!(node.side_gain.is_settled());
        assert!((node.side_gain.current() - db_to_linear(-6.0)).abs() < 1e-6);
    }
}
