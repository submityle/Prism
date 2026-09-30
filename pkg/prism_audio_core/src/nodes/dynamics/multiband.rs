//! Multi-band dynamic-range compressor: a Linkwitz-Riley crossover feeding one
//! independent compressor per frequency band, whose outputs are summed back.
//!
//! Splitting the spectrum before compression lets each band react to its own
//! level without a loud event in one region pumping the gain of unrelated
//! regions. A kick drum can be tamed in the low band while cymbals in the high
//! band are left untouched -- the defining behaviour of a mastering-grade
//! multi-band compressor, and the reason it is preferred over a single
//! full-band unit for glue, de-essing, and loudness control.
//!
//! The band split is a [`LinkwitzRileyCrossover`](crate::nodes::crossover::LinkwitzRileyCrossover):
//! its all-pass phase compensation keeps the summed bands magnitude-flat, so
//! with every band bypassed the processor reconstructs the input (up to the
//! network's shared all-pass phase). Each band is a full
//! [`CompressorNode`](crate::nodes::dynamics::CompressorNode), inheriting its
//! soft knee, peak/RMS detection, look-ahead, make-up gain, and parallel mix.
//!
//! # Real-time contract
//!
//! Every buffer -- the crossover state, the per-band split scratch, and the
//! per-band compressor output scratch -- is allocated in
//! [`MultibandCompressorNode::new`]. [`process`](crate::graph::AudioNode::process)
//! performs no allocation, takes no locks, and cannot panic: mismatched channel
//! counts and zero-length blocks degrade gracefully.
//!
//! # Provenance
//!
//! The multi-band compressor is a standard studio and mastering topology
//! (band-split, per-band dynamics, recombine) described in any audio
//! signal-processing reference (e.g. Zoelzer, "DAFX"). The band split uses this
//! crate's own Linkwitz-Riley crossover and the per-band dynamics use this
//! crate's own compressor; no source or derivative code from any commercial or
//! open engine (UE, Unity, Godot, Wwise, FMOD, Steam Audio) was consulted or
//! copied.

use alloc::vec::Vec;

use crate::buffer::{AudioBuffer, ChannelLayout};
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::nodes::crossover::{LinkwitzRileyCrossover, MAX_BANDS};
use crate::nodes::dynamics::compressor::{CompressorNode, CompressorParams};

/// A Linkwitz-Riley multi-band compressor (input port 0 -> output port 0).
///
/// Built from an ascending list of crossover frequencies and a matching list of
/// per-band [`CompressorParams`]; `M` crossovers yield `M + 1` bands. Feed a
/// block through [`process`](crate::graph::AudioNode::process) to compress each
/// band independently and sum the results.
#[derive(Debug, Clone)]
pub struct MultibandCompressorNode {
    /// The band-splitting crossover.
    crossover: LinkwitzRileyCrossover,
    /// One compressor per band (`num_bands` entries).
    compressors: Vec<CompressorNode>,
    /// Scratch holding the crossover's band split (`num_bands` entries).
    bands: Vec<AudioBuffer>,
    /// Scratch holding each band compressor's output (`num_bands` entries).
    band_outs: Vec<AudioBuffer>,
    /// Number of active frequency bands.
    num_bands: usize,
}

impl MultibandCompressorNode {
    /// Builds a multi-band compressor for `layout` handling blocks up to
    /// `max_frames`.
    ///
    /// `crossover_freqs` is passed to the underlying crossover (copied, clamped,
    /// sorted; at most [`MAX_BANDS`]` - 1` are honoured). Band `j` uses
    /// `band_params[j]` when present, otherwise a [`CompressorParams::default`].
    #[must_use]
    pub fn new(
        sample_rate: u32,
        layout: ChannelLayout,
        crossover_freqs: &[Sample],
        band_params: &[CompressorParams],
        max_frames: usize,
    ) -> Self {
        let crossover = LinkwitzRileyCrossover::new(sample_rate, layout, crossover_freqs, max_frames);
        let num_bands = crossover.num_bands().min(MAX_BANDS);
        let channels = layout.channel_count().max(1);
        let frames = max_frames.max(1);

        let mut compressors: Vec<CompressorNode> = Vec::with_capacity(num_bands);
        let mut bands: Vec<AudioBuffer> = Vec::with_capacity(num_bands);
        let mut band_outs: Vec<AudioBuffer> = Vec::with_capacity(num_bands);
        for j in 0..num_bands {
            let params = band_params.get(j).copied().unwrap_or_default();
            compressors.push(CompressorNode::new(sample_rate, channels, params));
            bands.push(AudioBuffer::new(layout, frames));
            band_outs.push(AudioBuffer::new(layout, frames));
        }

        Self {
            crossover,
            compressors,
            bands,
            band_outs,
            num_bands,
        }
    }

    /// Number of frequency bands (`crossovers + 1`).
    #[inline]
    #[must_use]
    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    /// Current gain reduction of band `j` in decibels (`>= 0`), or `0` if the
    /// band index is out of range.
    #[inline]
    #[must_use]
    pub fn band_gain_reduction_db(&self, band: usize) -> Sample {
        match self.compressors.get(band) {
            Some(c) => c.gain_reduction_db(),
            None => 0.0,
        }
    }
}

impl AudioNode for MultibandCompressorNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let frames = output.active_frames().min(input.active_frames());

        // Split the input into per-band scratch buffers.
        self.crossover.process_block(input, &mut self.bands);

        // Compress each band independently into its own output scratch.
        for j in 0..self.num_bands {
            self.band_outs[j].set_active_frames(self.bands[j].active_frames());
            let mut band_io = ProcessIo::new(
                core::slice::from_ref(&self.bands[j]),
                core::slice::from_mut(&mut self.band_outs[j]),
            );
            self.compressors[j].process(ctx, &mut band_io);
        }

        // Sum the compressed bands back into the node output.
        let out_channels = output.channels();
        for ch in 0..out_channels {
            let out_ch = output.channel_mut(ch);
            for f in 0..frames.min(out_ch.len()) {
                out_ch[f] = 0.0;
            }
        }
        for band_out in &self.band_outs {
            let channels = out_channels.min(band_out.channels());
            for ch in 0..channels {
                let src = band_out.channel(ch);
                let dst = output.channel_mut(ch);
                let n = frames.min(dst.len()).min(src.len());
                for f in 0..n {
                    dst[f] += src[f];
                }
            }
        }
    }

    fn reset(&mut self) {
        self.crossover.reset();
        for c in &mut self.compressors {
            c.reset();
        }
        for b in &mut self.bands {
            b.clear();
        }
        for b in &mut self.band_outs {
            b.clear();
        }
    }

    fn latency_frames(&self) -> u32 {
        let mut max = 0;
        for c in &self.compressors {
            let l = c.latency_frames();
            if l > max {
                max = l;
            }
        }
        max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn mono(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Mono, frames)
    }

    fn fill_sine(buf: &mut AudioBuffer, freq: Sample, amp: Sample) {
        let step = 2.0 * core::f32::consts::PI * freq / SR as Sample;
        let data = buf.channel_mut(0);
        let mut phase = 0.0;
        for d in data.iter_mut() {
            *d = amp * ops::sin(phase);
            phase += step;
        }
    }

    fn tail_rms(buf: &AudioBuffer, skip: usize) -> Sample {
        let data = buf.channel(0);
        let start = skip.min(data.len());
        let slice = &data[start..];
        if slice.is_empty() {
            return 0.0;
        }
        let mut acc = 0.0;
        for &s in slice {
            acc += s * s;
        }
        ops::sqrt(acc / slice.len() as Sample)
    }

    fn run(node: &mut MultibandCompressorNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let inputs = [input.clone()];
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames.max(1));
        out.set_active_frames(frames);
        let mut outputs = [out];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(frames), &mut io);
        }
        let [out] = outputs;
        out
    }

    /// A generous-headroom parameter set that applies effectively no gain
    /// reduction, so the band recombination can be checked against the input.
    fn transparent_params() -> CompressorParams {
        CompressorParams {
            threshold_db: 12.0,
            ratio: 1.0,
            knee_db: 0.0,
            makeup_db: 0.0,
            lookahead_ms: 0.0,
            wet: 1.0,
            dry: 0.0,
            ..CompressorParams::default()
        }
    }

    #[test]
    fn single_band_transparent_is_near_identity() {
        let mut node =
            MultibandCompressorNode::new(SR, ChannelLayout::Mono, &[], &[transparent_params()], 1024);
        assert_eq!(node.num_bands(), 1);
        let mut input = mono(1024);
        fill_sine(&mut input, 1_000.0, 0.3);
        input.set_active_frames(1024);
        let out = run(&mut node, &input);
        let in_rms = tail_rms(&input, 256);
        let out_rms = tail_rms(&out, 256);
        assert!((out_rms - in_rms).abs() < 0.02, "in {in_rms} out {out_rms}");
    }

    #[test]
    fn transparent_bands_reconstruct_flat_magnitude() {
        let params = [transparent_params(), transparent_params(), transparent_params()];
        let mut node =
            MultibandCompressorNode::new(SR, ChannelLayout::Mono, &[300.0, 3_000.0], &params, 2048);
        assert_eq!(node.num_bands(), 3);
        for &freq in &[120.0, 800.0, 6_000.0] {
            let mut input = mono(2048);
            fill_sine(&mut input, freq, 0.4);
            input.set_active_frames(2048);
            let out = run(&mut node, &input);
            node.reset();
            let in_rms = tail_rms(&input, 512);
            let out_rms = tail_rms(&out, 512);
            let ratio = out_rms / in_rms.max(1.0e-9);
            assert!((ratio - 1.0).abs() < 0.08, "freq {freq}: ratio {ratio}");
        }
    }

    #[test]
    fn low_band_compresses_low_tone() {
        // Low band compresses hard; high band is transparent.
        let low = CompressorParams {
            threshold_db: -30.0,
            ratio: 8.0,
            knee_db: 0.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            ..CompressorParams::default()
        };
        let params = [low, transparent_params()];
        let mut node =
            MultibandCompressorNode::new(SR, ChannelLayout::Mono, &[1_000.0], &params, 4096);
        let mut input = mono(4096);
        fill_sine(&mut input, 120.0, 0.8);
        input.set_active_frames(4096);
        let out = run(&mut node, &input);
        let in_rms = tail_rms(&input, 2048);
        let out_rms = tail_rms(&out, 2048);
        assert!(out_rms < in_rms * 0.85, "expected reduction: in {in_rms} out {out_rms}");
        assert!(node.band_gain_reduction_db(0) > 1.0);
    }

    #[test]
    fn high_tone_survives_low_band_compression() {
        // Same setup as above, but the tone is in the untouched high band.
        let low = CompressorParams {
            threshold_db: -30.0,
            ratio: 8.0,
            knee_db: 0.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            ..CompressorParams::default()
        };
        let params = [low, transparent_params()];
        let mut node =
            MultibandCompressorNode::new(SR, ChannelLayout::Mono, &[1_000.0], &params, 4096);
        let mut input = mono(4096);
        fill_sine(&mut input, 6_000.0, 0.8);
        input.set_active_frames(4096);
        let out = run(&mut node, &input);
        let in_rms = tail_rms(&input, 2048);
        let out_rms = tail_rms(&out, 2048);
        assert!(out_rms > in_rms * 0.9, "high tone should survive: in {in_rms} out {out_rms}");
    }

    #[test]
    fn reset_clears_tail() {
        let mut node =
            MultibandCompressorNode::new(SR, ChannelLayout::Mono, &[500.0], &[], 512);
        let mut input = mono(512);
        fill_sine(&mut input, 700.0, 0.5);
        input.set_active_frames(512);
        let _ = run(&mut node, &input);
        node.reset();
        let silence = mono(512);
        let out = run(&mut node, &silence);
        assert!(tail_rms(&out, 0) < 1.0e-6);
    }

    #[test]
    fn output_is_finite() {
        let mut node =
            MultibandCompressorNode::new(SR, ChannelLayout::Mono, &[200.0, 2_000.0], &[], 1024);
        let mut input = mono(1024);
        fill_sine(&mut input, 1_000.0, 0.9);
        input.set_active_frames(1024);
        let out = run(&mut node, &input);
        for &s in out.channel(0) {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn zero_frames_does_not_panic() {
        let mut node = MultibandCompressorNode::new(SR, ChannelLayout::Mono, &[500.0], &[], 16);
        let mut input = mono(16);
        input.set_active_frames(0);
        let out = run(&mut node, &input);
        assert_eq!(out.active_frames(), 0);
    }

    #[test]
    fn band_count_matches_crossovers() {
        let node = MultibandCompressorNode::new(
            SR,
            ChannelLayout::Stereo,
            &[100.0, 500.0, 2_000.0, 8_000.0],
            &[],
            256,
        );
        assert_eq!(node.num_bands(), 5);
    }

    #[test]
    fn out_of_range_band_reduction_is_zero() {
        let node = MultibandCompressorNode::new(SR, ChannelLayout::Mono, &[500.0], &[], 64);
        assert_eq!(node.band_gain_reduction_db(99), 0.0);
    }
}
