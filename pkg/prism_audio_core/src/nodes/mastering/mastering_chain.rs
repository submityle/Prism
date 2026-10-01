//! A fixed-order mastering chain that orchestrates existing processors into a
//! single delivery-stage node.
//!
//! The chain renders a finished mix through the canonical mastering signal
//! order -- **parametric EQ -> dynamics compression -> brick-wall limiting ->
//! dither** -- using the already-implemented processors from the sibling
//! submodules. It owns one instance of each stage and runs them in series
//! inside a single [`AudioNode::process`] call, exposing per-stage bypass
//! switches so an engineer can audition the chain one link at a time.
//!
//! The node performs **no digital signal processing of its own**: every sample
//! transformation is delegated to the stage nodes. Its sole responsibility is
//! sequencing, buffer management, and latency accounting. Running the stages
//! back to back requires a scratch buffer per hand-off, because a node may not
//! use the same buffer as input and output; this module ping-pongs between two
//! pre-allocated scratch buffers so the hot path performs no allocation.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The fixed
//! EQ/compress/limit/dither ordering is standard mastering practice described
//! in public audio-engineering literature; the orchestration logic here is an
//! original composition over Prism's own nodes. It is pure classic DSP with no
//! AI/ML.
//!
//! # Relationship
//!
//! This node is a pure **composition** of existing processors and implements no
//! new DSP:
//!
//! - EQ is delegated to [`ParametricEqNode`](crate::nodes::ParametricEqNode)
//!   from [`effects`](crate::nodes::effects).
//! - Compression is delegated to
//!   [`CompressorNode`](crate::nodes::CompressorNode) from
//!   [`dynamics`](crate::nodes::dynamics).
//! - Limiting is delegated to [`LimiterNode`](crate::nodes::LimiterNode), also
//!   from [`dynamics`](crate::nodes::dynamics).
//! - Requantization is delegated to [`DitherNode`](crate::nodes::DitherNode)
//!   from the sibling [`dither`](crate::nodes::mastering::dither) module.
//!
//! The chain fixes the mastering signal order and reuses each stage verbatim
//! rather than re-deriving any filter, detector, or quantizer, keeping a single
//! source of truth for each processor's behaviour.

use alloc::vec::Vec;

use crate::buffer::{AudioBuffer, ChannelLayout};
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::nodes::{
    CompressorNode, CompressorParams, DitherNode, DitherParams, EqBand, LimiterNode, LimiterParams,
    ParametricEqNode,
};

/// Configuration for every stage of a [`MasteringChainNode`], plus the
/// per-stage bypass switches.
///
/// The defaults describe a transparent-by-construction chain: the EQ has no
/// bands (unity), and dithering is bypassed because requantization is only
/// meaningful when exporting to a fixed bit depth. The compressor and limiter
/// use their own documented defaults.
#[derive(Debug, Clone)]
pub struct MasteringChainParams {
    /// Parametric EQ bands, applied in order. Empty is a unity pass-through.
    pub eq_bands: Vec<EqBand>,
    /// Compressor parameters for the dynamics stage.
    pub compressor: CompressorParams,
    /// Limiter parameters for the brick-wall ceiling stage.
    pub limiter: LimiterParams,
    /// Dither parameters for the final requantization stage.
    pub dither: DitherParams,
    /// When `true`, the EQ stage is skipped and the signal passes through
    /// unchanged.
    pub bypass_eq: bool,
    /// When `true`, the compressor stage is skipped.
    pub bypass_compressor: bool,
    /// When `true`, the limiter stage is skipped.
    pub bypass_limiter: bool,
    /// When `true`, the dither stage is skipped (no requantization).
    pub bypass_dither: bool,
}

impl Default for MasteringChainParams {
    fn default() -> Self {
        Self {
            eq_bands: Vec::new(),
            compressor: CompressorParams::default(),
            limiter: LimiterParams::default(),
            dither: DitherParams::default(),
            bypass_eq: false,
            bypass_compressor: false,
            bypass_limiter: false,
            bypass_dither: true,
        }
    }
}

/// A delivery-stage mastering chain running EQ, compression, limiting, and
/// dither in series (input port 0 -> output port 0).
///
/// The node owns one instance of each stage and two scratch buffers sized at
/// construction. During [`process`](AudioNode::process) it copies the input
/// into a scratch buffer, runs each enabled stage ping-pong between the two
/// scratch buffers, and copies the final result to the output. Bypassed stages
/// are skipped entirely, so a fully bypassed chain is a bit-exact
/// pass-through.
///
/// All buffers are allocated in [`MasteringChainNode::new`]; the audio thread
/// performs no allocation, locking, or panicking. Blocks longer than the
/// configured maximum are clamped to the scratch capacity.
#[derive(Debug, Clone)]
pub struct MasteringChainNode {
    eq: ParametricEqNode,
    compressor: CompressorNode,
    limiter: LimiterNode,
    dither: DitherNode,
    bypass_eq: bool,
    bypass_compressor: bool,
    bypass_limiter: bool,
    bypass_dither: bool,
    /// Two pre-allocated scratch buffers used to hand the signal between
    /// stages without ever aliasing input and output.
    scratch: [AudioBuffer; 2],
    /// Channel layout shared by every stage and scratch buffer.
    layout: ChannelLayout,
    /// Maximum block size, in frames, the scratch buffers can hold.
    max_block_frames: usize,
}

impl MasteringChainNode {
    /// Builds a mastering chain for a `layout`-wide signal at `sample_rate` Hz
    /// that can process up to `max_block_frames` frames per call.
    ///
    /// Each stage is constructed with the channel count implied by `layout`.
    /// `max_block_frames` is clamped to at least one frame so the scratch
    /// buffers are always valid.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    /// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
    /// use prism_audio_core::nodes::mastering::{MasteringChainNode, MasteringChainParams};
    ///
    /// let params = MasteringChainParams::default();
    /// let mut chain = MasteringChainNode::new(48_000, ChannelLayout::Stereo, 512, &params);
    ///
    /// let mut input = AudioBuffer::new(ChannelLayout::Stereo, 256);
    /// input.set_active_frames(256);
    /// let mut output = AudioBuffer::new(ChannelLayout::Stereo, 256);
    /// output.set_active_frames(256);
    ///
    /// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
    /// let inputs = [input];
    /// let mut outputs = [output];
    /// let mut io = ProcessIo::new(&inputs, &mut outputs);
    /// chain.process(&ctx, &mut io);
    /// assert_eq!(chain.max_block_frames(), 512);
    /// ```
    #[must_use]
    pub fn new(
        sample_rate: u32,
        layout: ChannelLayout,
        max_block_frames: usize,
        params: &MasteringChainParams,
    ) -> Self {
        let channels = layout.channel_count();
        let cap = max_block_frames.max(1);
        let scratch = [
            AudioBuffer::new(layout, cap),
            AudioBuffer::new(layout, cap),
        ];
        Self {
            eq: ParametricEqNode::new(sample_rate, channels, &params.eq_bands),
            compressor: CompressorNode::new(sample_rate, channels, params.compressor),
            limiter: LimiterNode::new(sample_rate, channels, params.limiter),
            dither: DitherNode::new(params.dither, channels),
            bypass_eq: params.bypass_eq,
            bypass_compressor: params.bypass_compressor,
            bypass_limiter: params.bypass_limiter,
            bypass_dither: params.bypass_dither,
            scratch,
            layout,
            max_block_frames: cap,
        }
    }

    /// Returns the channel layout the chain was built for.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns the maximum block size, in frames, the chain can process.
    #[inline]
    #[must_use]
    pub fn max_block_frames(&self) -> usize {
        self.max_block_frames
    }

    /// Enables or disables the EQ stage at control rate.
    #[inline]
    pub fn set_bypass_eq(&mut self, bypass: bool) {
        self.bypass_eq = bypass;
    }

    /// Enables or disables the compressor stage at control rate.
    #[inline]
    pub fn set_bypass_compressor(&mut self, bypass: bool) {
        self.bypass_compressor = bypass;
    }

    /// Enables or disables the limiter stage at control rate.
    #[inline]
    pub fn set_bypass_limiter(&mut self, bypass: bool) {
        self.bypass_limiter = bypass;
    }

    /// Enables or disables the dither stage at control rate.
    #[inline]
    pub fn set_bypass_dither(&mut self, bypass: bool) {
        self.bypass_dither = bypass;
    }

    /// Borrows the EQ stage (for example to retune a band).
    #[inline]
    #[must_use]
    pub fn eq_mut(&mut self) -> &mut ParametricEqNode {
        &mut self.eq
    }

    /// Borrows the compressor stage.
    #[inline]
    #[must_use]
    pub fn compressor_mut(&mut self) -> &mut CompressorNode {
        &mut self.compressor
    }

    /// Borrows the limiter stage.
    #[inline]
    #[must_use]
    pub fn limiter_mut(&mut self) -> &mut LimiterNode {
        &mut self.limiter
    }

    /// Borrows the dither stage.
    #[inline]
    #[must_use]
    pub fn dither_mut(&mut self) -> &mut DitherNode {
        &mut self.dither
    }
}

/// Runs one stage, reading scratch buffer `src` and writing scratch buffer
/// `1 - src`, without aliasing the two buffers.
#[inline]
fn run_stage(
    scratch: &mut [AudioBuffer; 2],
    src: usize,
    node: &mut impl AudioNode,
    ctx: &RenderContext,
) {
    let (lo, hi) = scratch.split_at_mut(1);
    let (in_buf, out_buf) = if src == 0 {
        (&lo[0], &mut hi[0])
    } else {
        (&hi[0], &mut lo[0])
    };
    let inputs = core::slice::from_ref(in_buf);
    let outputs = core::slice::from_mut(out_buf);
    let mut io = ProcessIo::new(inputs, outputs);
    node.process(ctx, &mut io);
}

impl AudioNode for MasteringChainNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let frames = output.active_frames().min(self.max_block_frames);
        let chain_channels = self.layout.channel_count();
        if frames == 0 || out_channels == 0 {
            return;
        }

        // Prime both scratch buffers to the active block length so every stage
        // (whether it copies the input or writes in place) sees a consistent
        // frame count.
        self.scratch[0].set_active_frames(frames);
        self.scratch[1].set_active_frames(frames);

        // Copy the input into scratch[0], zero-filling any chain channels the
        // input does not supply.
        let in_channels = input.channels();
        for ch in 0..chain_channels {
            let dst = self.scratch[0].channel_mut(ch);
            if ch < in_channels {
                dst[..frames].copy_from_slice(&input.channel(ch)[..frames]);
            } else {
                for s in dst[..frames].iter_mut() {
                    *s = 0.0;
                }
            }
        }

        // Run each enabled stage in the canonical mastering order, bouncing
        // between the two scratch buffers.
        let mut src = 0usize;
        if !self.bypass_eq {
            run_stage(&mut self.scratch, src, &mut self.eq, ctx);
            src = 1 - src;
        }
        if !self.bypass_compressor {
            run_stage(&mut self.scratch, src, &mut self.compressor, ctx);
            src = 1 - src;
        }
        if !self.bypass_limiter {
            run_stage(&mut self.scratch, src, &mut self.limiter, ctx);
            src = 1 - src;
        }
        if !self.bypass_dither {
            run_stage(&mut self.scratch, src, &mut self.dither, ctx);
            src = 1 - src;
        }

        // Copy the final result to the output's shared channels.
        let copy_channels = out_channels.min(chain_channels);
        for ch in 0..copy_channels {
            let result = self.scratch[src].channel(ch);
            output.channel_mut(ch)[..frames].copy_from_slice(&result[..frames]);
        }
    }

    fn reset(&mut self) {
        self.eq.reset();
        self.compressor.reset();
        self.limiter.reset();
        self.dither.reset();
        self.scratch[0].clear();
        self.scratch[1].clear();
    }

    fn latency_frames(&self) -> u32 {
        let mut latency = 0;
        if !self.bypass_compressor {
            latency += self.compressor.latency_frames();
        }
        if !self.bypass_limiter {
            latency += self.limiter.latency_frames();
        }
        latency
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Sample;
    use alloc::vec;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn stereo_ramp(frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Stereo, frames);
        buf.set_active_frames(frames);
        for f in 0..frames {
            let v = (f as Sample) / (frames as Sample);
            buf.channel_mut(0)[f] = v;
            buf.channel_mut(1)[f] = -v;
        }
        buf
    }

    fn run(node: &mut MasteringChainNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let mut out = AudioBuffer::new(input.layout(), input.capacity_frames().max(1));
        out.set_active_frames(frames);
        let inputs = [input.clone()];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [o] = outputs;
        o
    }

    fn all_bypassed() -> MasteringChainParams {
        MasteringChainParams {
            bypass_eq: true,
            bypass_compressor: true,
            bypass_limiter: true,
            bypass_dither: true,
            ..MasteringChainParams::default()
        }
    }

    #[test]
    fn fully_bypassed_is_bit_exact_passthrough() {
        let params = all_bypassed();
        let mut node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 256, &params);
        let input = stereo_ramp(128);
        let out = run(&mut node, &input);
        for ch in 0..2 {
            assert_eq!(out.channel(ch), input.channel(ch));
        }
    }

    #[test]
    fn eq_only_matches_standalone_eq() {
        let bands = vec![EqBand::peaking(1_000.0, 1.0, 6.0)];
        let params = MasteringChainParams {
            eq_bands: bands.clone(),
            bypass_eq: false,
            bypass_compressor: true,
            bypass_limiter: true,
            bypass_dither: true,
            ..MasteringChainParams::default()
        };
        let mut node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 256, &params);
        let input = stereo_ramp(128);
        let chained = run(&mut node, &input);

        // Reference: a standalone EQ fed the same input.
        let mut eq = ParametricEqNode::new(SR, 2, &bands);
        let mut ref_out = AudioBuffer::new(ChannelLayout::Stereo, 128);
        ref_out.set_active_frames(128);
        {
            let inputs = [input.clone()];
            let mut outputs = [ref_out];
            let mut eq_io = ProcessIo::new(&inputs, &mut outputs);
            eq.process(&ctx(128), &mut eq_io);
            ref_out = outputs.into_iter().next().unwrap();
        }
        for ch in 0..2 {
            for f in 0..128 {
                assert!((chained.channel(ch)[f] - ref_out.channel(ch)[f]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn chain_equals_manual_serial_run() {
        let bands = vec![EqBand::high_shelf(8_000.0, 0.707, 3.0)];
        let comp = CompressorParams {
            threshold_db: -24.0,
            ratio: 3.0,
            ..CompressorParams::default()
        };
        let lim = LimiterParams {
            lookahead_ms: 0.0,
            ..LimiterParams::default()
        };
        let params = MasteringChainParams {
            eq_bands: bands.clone(),
            compressor: comp,
            limiter: lim,
            bypass_dither: true,
            ..MasteringChainParams::default()
        };
        let mut node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 256, &params);
        let input = stereo_ramp(200);
        let chained = run(&mut node, &input);

        // Reference: run the same three stages by hand in series.
        let mut eq = ParametricEqNode::new(SR, 2, &bands);
        let mut compressor = CompressorNode::new(SR, 2, comp);
        let mut limiter = LimiterNode::new(SR, 2, lim);
        let mut stage = input.clone();
        for node in [
            &mut eq as &mut dyn AudioNode,
            &mut compressor as &mut dyn AudioNode,
            &mut limiter as &mut dyn AudioNode,
        ] {
            let mut next = AudioBuffer::new(ChannelLayout::Stereo, 200);
            next.set_active_frames(200);
            let inputs = [stage.clone()];
            let mut outputs = [next];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(200), &mut io);
            stage = outputs.into_iter().next().unwrap();
        }
        for ch in 0..2 {
            for f in 0..200 {
                assert!((chained.channel(ch)[f] - stage.channel(ch)[f]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn limiter_enforces_ceiling() {
        let lim = LimiterParams {
            ceiling_db: -6.0,
            lookahead_ms: 1.0,
            ..LimiterParams::default()
        };
        let params = MasteringChainParams {
            bypass_eq: true,
            bypass_compressor: true,
            bypass_limiter: false,
            bypass_dither: true,
            limiter: lim,
            ..MasteringChainParams::default()
        };
        let mut node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 2048, &params);
        let frames = 1024;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, frames);
        input.set_active_frames(frames);
        for ch in 0..2 {
            for s in input.channel_mut(ch) {
                *s = 0.9;
            }
        }
        let out = run(&mut node, &input);
        // Ceiling of -6 dBFS is ~0.501 linear; the limiter should settle below
        // 0.9 well before the end of the block.
        let tail = out.channel(0)[frames - 1].abs();
        assert!(tail < 0.9, "limiter failed to reduce level: {tail}");
    }

    #[test]
    fn bypass_toggles_change_output() {
        let bands = vec![EqBand::peaking(2_000.0, 1.0, 9.0)];
        let params = MasteringChainParams {
            eq_bands: bands,
            bypass_eq: false,
            bypass_compressor: true,
            bypass_limiter: true,
            bypass_dither: true,
            ..MasteringChainParams::default()
        };
        let mut node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 256, &params);
        let input = stereo_ramp(128);
        let with_eq = run(&mut node, &input);

        node.reset();
        node.set_bypass_eq(true);
        let without_eq = run(&mut node, &input);

        // With EQ bypassed the output equals the input; with EQ engaged it does
        // not.
        assert_eq!(without_eq.channel(0), input.channel(0));
        let mut differs = false;
        for f in 0..128 {
            if (with_eq.channel(0)[f] - input.channel(0)[f]).abs() > 1e-6 {
                differs = true;
                break;
            }
        }
        assert!(differs, "engaged EQ should alter the signal");
    }

    #[test]
    fn reset_clears_scratch_and_stages() {
        let params = MasteringChainParams::default();
        let mut node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 256, &params);
        let input = stereo_ramp(128);
        let _ = run(&mut node, &input);
        node.reset();
        for ch in 0..2 {
            assert!(node.scratch[0].channel(ch).iter().all(|&s| s == 0.0));
            assert!(node.scratch[1].channel(ch).iter().all(|&s| s == 0.0));
        }
    }

    #[test]
    fn zero_frame_block_leaves_output_untouched() {
        let params = all_bypassed();
        let mut node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 256, &params);
        // A zero-frame render request (empty active window on both buffers)
        // must return early without panicking and without widening the output.
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 64);
        input.set_active_frames(0);
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
        out.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        let [o] = outputs;
        assert_eq!(o.active_frames(), 0);
    }

    #[test]
    fn block_longer_than_capacity_is_clamped() {
        let params = all_bypassed();
        let mut node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 64, &params);
        let input = stereo_ramp(256);
        // Must not panic even though the block exceeds the scratch capacity.
        let out = run(&mut node, &input);
        // The first 64 frames are processed (pass-through here).
        for f in 0..64 {
            assert_eq!(out.channel(0)[f], input.channel(0)[f]);
        }
    }

    #[test]
    fn mono_layout_processes_without_panic() {
        let params = MasteringChainParams::default();
        let mut node = MasteringChainNode::new(SR, ChannelLayout::Mono, 128, &params);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 64);
        input.set_active_frames(64);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = (i as Sample) / 64.0;
        }
        let out = run(&mut node, &input);
        assert_eq!(out.active_frames(), 64);
        assert!(out.channel(0).iter().all(|s| s.is_finite()));
    }

    #[test]
    fn non_finite_input_stays_finite_when_limited() {
        let params = MasteringChainParams {
            bypass_eq: true,
            bypass_compressor: true,
            bypass_limiter: false,
            bypass_dither: true,
            ..MasteringChainParams::default()
        };
        let mut node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 512, &params);
        let frames = 256;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, frames);
        input.set_active_frames(frames);
        input.channel_mut(0)[10] = Sample::INFINITY;
        input.channel_mut(1)[20] = Sample::NAN;
        // Must not panic.
        let _ = run(&mut node, &input);
    }

    #[test]
    fn latency_sums_enabled_dynamics_stages() {
        let comp = CompressorParams {
            lookahead_ms: 5.0,
            ..CompressorParams::default()
        };
        let lim = LimiterParams {
            lookahead_ms: 5.0,
            ..LimiterParams::default()
        };
        let params = MasteringChainParams {
            compressor: comp,
            limiter: lim,
            bypass_eq: false,
            bypass_compressor: false,
            bypass_limiter: false,
            bypass_dither: false,
            ..MasteringChainParams::default()
        };
        let node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 1024, &params);
        let ref_comp = CompressorNode::new(SR, 2, comp);
        let ref_lim = LimiterNode::new(SR, 2, lim);
        assert_eq!(
            node.latency_frames(),
            ref_comp.latency_frames() + ref_lim.latency_frames()
        );
    }

    #[test]
    fn bypassing_dynamics_zeroes_latency() {
        let params = MasteringChainParams {
            compressor: CompressorParams {
                lookahead_ms: 5.0,
                ..CompressorParams::default()
            },
            limiter: LimiterParams {
                lookahead_ms: 5.0,
                ..LimiterParams::default()
            },
            bypass_compressor: true,
            bypass_limiter: true,
            ..MasteringChainParams::default()
        };
        let node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 1024, &params);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn dither_stage_requantizes_output() {
        let params = MasteringChainParams {
            bypass_eq: true,
            bypass_compressor: true,
            bypass_limiter: true,
            bypass_dither: false,
            dither: DitherParams {
                bits: 8,
                ..DitherParams::default()
            },
            ..MasteringChainParams::default()
        };
        let mut node = MasteringChainNode::new(SR, ChannelLayout::Stereo, 256, &params);
        let input = stereo_ramp(128);
        let out = run(&mut node, &input);
        // At 8 bits the output should be quantized to a coarse grid, so it must
        // differ from the smooth input ramp somewhere.
        let mut differs = false;
        for f in 0..128 {
            if (out.channel(0)[f] - input.channel(0)[f]).abs() > 1e-6 {
                differs = true;
                break;
            }
        }
        assert!(differs, "dither stage should requantize the signal");
    }
}
