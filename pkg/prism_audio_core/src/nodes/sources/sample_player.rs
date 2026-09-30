//! Pitched, loopable PCM sample-playback source node.
//!
//! [`SamplePlayerNode`] is a *source* node (zero inputs, one output) that reads
//! pre-loaded planar PCM data back at an arbitrary, automatable playback rate.
//! It is the primitive behind one-shots, looped ambience beds, granular grains,
//! and pitched instrument voices: the higher authoring layers instantiate one
//! per playing voice and feed it a slice of decoded audio.
//!
//! # Resampling
//!
//! The read position advances by a *fractional* number of source frames per
//! output frame, so the same data can be played at any pitch. Two boundary
//! concerns are handled here:
//!
//! - **Pitch / speed.** A user-facing playback `rate` (`1.0` = original pitch,
//!   `2.0` = one octave up / double speed) is smoothed to stay click-free.
//! - **Sample-rate conversion.** When the source was authored at a different
//!   sample rate than the graph renders at, the effective step is additionally
//!   scaled by `source_sample_rate / render_sample_rate` so a 22.05 kHz clip
//!   plays at the correct pitch inside a 48 kHz graph.
//!
//! Between integer source frames the output is reconstructed with a selectable
//! [`Interpolation`] kernel (linear or Catmull-Rom).
//!
//! # Real-time contract
//!
//! All PCM storage is captured at construction. Everything reachable from
//! [`SamplePlayerNode::process`](crate::graph::AudioNode::process) is
//! allocation-free, lock-free, and panic-free: every buffer index is clamped
//! and every loop boundary is range-checked, so malformed loop points can never
//! trigger an out-of-bounds access.

use alloc::vec::Vec;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

/// Smallest playback rate the node will accept.
///
/// Playback must always advance forward in source time, so requested rates are
/// clamped up to this small positive floor rather than allowed to reach zero
/// (which would stall the read head) or go negative.
const MIN_RATE: Sample = 1.0e-6;

/// How the read head behaves when it reaches the end of the playable region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// Play once from the current position to the end of the data, then stop
    /// and emit silence.
    OneShot,
    /// Loop forward forever, wrapping from `end` back to `start`.
    Forward {
        /// Inclusive first source frame of the loop region.
        start: usize,
        /// Inclusive last source frame of the loop region; playback wraps once
        /// the read head passes this frame.
        end: usize,
    },
    /// Bounce back and forth between `start` and `end`, reversing direction at
    /// each endpoint.
    PingPong {
        /// Inclusive lower source frame the read head reflects off of.
        start: usize,
        /// Inclusive upper source frame the read head reflects off of.
        end: usize,
    },
}

/// Fractional-position reconstruction kernel used between integer source
/// frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Two-point linear interpolation: cheapest, mild high-frequency loss.
    Linear,
    /// Four-point (Catmull-Rom) cubic interpolation: smoother, passes exactly
    /// through the original sample values at integer positions.
    CatmullRom,
}

/// A pitched, loopable PCM playback source (0 inputs, 1 output).
///
/// The output port carries one channel per stored PCM channel. When stopped,
/// when the data is empty, or when a [`LoopMode::OneShot`] has finished, the
/// node writes silence.
#[derive(Debug, Clone)]
pub struct SamplePlayerNode {
    /// Planar PCM: one owned buffer per channel, captured at construction.
    channels: Vec<Vec<Sample>>,
    /// Number of playable source frames (the shortest channel length).
    len: usize,
    /// Sample rate the PCM was authored at, in Hz.
    source_sample_rate: u32,
    /// Fractional read position in source frames.
    pos: f64,
    /// Current travel direction: `+1.0` forward, `-1.0` reversed (only ever
    /// negative under [`LoopMode::PingPong`]).
    direction: f64,
    /// Smoothed user-facing playback rate (pitch / speed multiplier).
    rate: Smoothed,
    /// Behaviour of the read head at the end of the playable region.
    loop_mode: LoopMode,
    /// Reconstruction kernel used for fractional positions.
    interpolation: Interpolation,
    /// Whether the read head is currently advancing.
    playing: bool,
}

impl SamplePlayerNode {
    /// Builds a player over `channels` of planar PCM authored at
    /// `source_sample_rate` Hz.
    ///
    /// The output port width equals `channels.len()`. The playable length is
    /// the shortest channel so ragged inputs can never over-read. The node
    /// starts stopped at position zero with `rate = 1.0`, [`LoopMode::OneShot`],
    /// and [`Interpolation::Linear`]; call [`SamplePlayerNode::play`] to begin.
    #[must_use]
    pub fn new(channels: Vec<Vec<Sample>>, source_sample_rate: u32) -> Self {
        let mut len = usize::MAX;
        for ch in &channels {
            len = len.min(ch.len());
        }
        if channels.is_empty() {
            len = 0;
        }
        Self {
            channels,
            len,
            source_sample_rate: source_sample_rate.max(1),
            pos: 0.0,
            direction: 1.0,
            rate: Smoothed::new(1.0),
            loop_mode: LoopMode::OneShot,
            interpolation: Interpolation::Linear,
            playing: false,
        }
    }

    /// Sets the playback rate (pitch / speed multiplier), clamped to a small
    /// positive floor and applied immediately (no glide) so scripted rate
    /// changes are sample-deterministic.
    #[inline]
    pub fn set_rate(&mut self, rate: Sample) {
        self.rate.set_target(rate.max(MIN_RATE), Ramp::Immediate);
    }

    /// Sets the playback rate with a click-free linear glide spanning
    /// `ramp_samples` frames (clamped to a small positive floor).
    #[inline]
    pub fn set_rate_smoothed(&mut self, rate: Sample, ramp_samples: u32) {
        self.rate.set_target(
            rate.max(MIN_RATE),
            Ramp::Linear {
                samples: ramp_samples,
            },
        );
    }

    /// Returns the target playback rate.
    #[inline]
    #[must_use]
    pub fn rate(&self) -> Sample {
        self.rate.target()
    }

    /// Selects the loop behaviour.
    #[inline]
    pub fn set_loop_mode(&mut self, mode: LoopMode) {
        self.loop_mode = mode;
    }

    /// Returns the current loop behaviour.
    #[inline]
    #[must_use]
    pub fn loop_mode(&self) -> LoopMode {
        self.loop_mode
    }

    /// Selects the fractional-position interpolation kernel.
    #[inline]
    pub fn set_interpolation(&mut self, interpolation: Interpolation) {
        self.interpolation = interpolation;
    }

    /// Returns the current interpolation kernel.
    #[inline]
    #[must_use]
    pub fn interpolation(&self) -> Interpolation {
        self.interpolation
    }

    /// Moves the read head to `frame` (fractional), clamped to the playable
    /// range, and resets the travel direction to forward.
    #[inline]
    pub fn seek(&mut self, frame: f64) {
        let max = last_index_f64(self.len);
        self.pos = frame.max(0.0).min(max);
        self.direction = 1.0;
    }

    /// Starts (or resumes) playback if there is any playable data.
    #[inline]
    pub fn play(&mut self) {
        if self.len > 0 {
            self.playing = true;
        }
    }

    /// Stops playback; subsequent blocks emit silence until the next
    /// [`SamplePlayerNode::play`].
    #[inline]
    pub fn stop(&mut self) {
        self.playing = false;
    }

    /// Returns whether the read head is currently advancing.
    #[inline]
    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Returns the current fractional read position in source frames.
    #[inline]
    #[must_use]
    pub fn position(&self) -> f64 {
        self.pos
    }

    /// Returns the number of output channels.
    #[inline]
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Returns the number of playable source frames.
    #[inline]
    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.len
    }

    /// Returns the source sample rate in Hz.
    #[inline]
    #[must_use]
    pub fn source_sample_rate(&self) -> u32 {
        self.source_sample_rate
    }

    /// Reads channel `ch` at integer source frame `index`, clamping the index
    /// into the channel's bounds so a lookup can never panic.
    #[inline]
    fn sample_at(&self, ch: usize, index: usize) -> Sample {
        if ch >= self.channels.len() {
            return 0.0;
        }
        let data = &self.channels[ch];
        if data.is_empty() {
            return 0.0;
        }
        let clamped = index.min(data.len() - 1);
        data[clamped]
    }

    /// Reconstructs channel `ch` at the fractional position `pos` using the
    /// selected interpolation kernel. All source indices are clamped, so
    /// boundary positions read the edge sample rather than over-running.
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the interpolated PCM value is finite and within the f32 Sample range; narrowing the f64 accumulator back to the engine's f32 sample type is intentional"
    )]
    fn interpolate(&self, ch: usize, pos: f64) -> Sample {
        let (idx, frac) = split_pos(pos);
        let value = match self.interpolation {
            Interpolation::Linear => {
                let a = f64::from(self.sample_at(ch, idx));
                let b = f64::from(self.sample_at(ch, idx + 1));
                a + (b - a) * frac
            }
            Interpolation::CatmullRom => {
                let p0 = f64::from(self.sample_at(ch, idx.saturating_sub(1)));
                let p1 = f64::from(self.sample_at(ch, idx));
                let p2 = f64::from(self.sample_at(ch, idx + 1));
                let p3 = f64::from(self.sample_at(ch, idx + 2));
                let t = frac;
                let t2 = t * t;
                let t3 = t2 * t;
                0.5 * ((2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
            }
        };
        value as Sample
    }

    /// Advances the read position by the (non-negative) magnitude `step` source
    /// frames, applying the active loop policy. May clear [`Self::playing`] when
    /// a [`LoopMode::OneShot`] reaches the end.
    #[inline]
    fn advance(&mut self, step: f64) {
        let last = last_index_f64(self.len);
        match self.loop_mode {
            LoopMode::OneShot => {
                self.pos += step;
                if self.pos > last {
                    self.pos = last;
                    self.playing = false;
                }
            }
            LoopMode::Forward { start, end } => {
                self.pos += step;
                let (s, e) = sanitize_region(start, end, self.len);
                if e > s {
                    // Region length in frames used to wrap the read head.
                    let span = to_f64(e - s);
                    let ef = to_f64(e);
                    let sf = to_f64(s);
                    while self.pos > ef {
                        self.pos -= span;
                    }
                    if self.pos < sf {
                        self.pos = sf;
                    }
                } else {
                    self.pos = to_f64(s);
                }
            }
            LoopMode::PingPong { start, end } => {
                let (s, e) = sanitize_region(start, end, self.len);
                if e > s {
                    self.pos += self.direction * step;
                    let sf = to_f64(s);
                    let ef = to_f64(e);
                    loop {
                        if self.pos > ef {
                            self.pos = 2.0 * ef - self.pos;
                            self.direction = -1.0;
                        } else if self.pos < sf {
                            self.pos = 2.0 * sf - self.pos;
                            self.direction = 1.0;
                        } else {
                            break;
                        }
                    }
                } else {
                    self.pos = to_f64(s);
                }
            }
        }
    }
}

/// Splits a non-negative fractional position into its integer frame index and
/// the `[0, 1)` fractional remainder. The integer part is obtained by
/// truncation, which equals `floor` for the non-negative positions this node
/// guarantees (avoiding `f64::floor`, which is unavailable under `no_std`).
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the read position is clamped non-negative and bounded by the buffer length, so truncating to usize is exact and never wraps"
)]
fn split_pos(pos: f64) -> (usize, f64) {
    let idx = pos as usize;
    (idx, pos - to_f64(idx))
}

/// Converts a source-frame count/index to `f64`.
#[inline]
#[expect(
    clippy::cast_precision_loss,
    reason = "source-frame indices stay far below f64's 2^53 exact-integer limit for any realistic audio buffer"
)]
fn to_f64(index: usize) -> f64 {
    index as f64
}

/// Returns the last valid source index as an `f64` (`0.0` when the buffer is
/// empty).
#[inline]
fn last_index_f64(len: usize) -> f64 {
    to_f64(len.saturating_sub(1))
}

/// Clamps a requested `[start, end]` loop region into the playable range,
/// guaranteeing `end <= last index` and `start <= end`.
#[inline]
fn sanitize_region(start: usize, end: usize, len: usize) -> (usize, usize) {
    let last = len.saturating_sub(1);
    let e = end.min(last);
    let s = start.min(e);
    (s, e)
}

impl AudioNode for SamplePlayerNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        let out_channels = out.channels();
        let frames = out.active_frames();

        // Silence fast path: nothing playing, or no data to play.
        if !self.playing || self.len == 0 {
            for ch in 0..out_channels {
                for s in out.channel_mut(ch) {
                    *s = 0.0;
                }
            }
            return;
        }

        // Combined pitch and sample-rate-conversion ratio.
        let ratio = f64::from(self.source_sample_rate) / f64::from(ctx.sample_rate.max(1));

        let mut f = 0usize;
        while f < frames {
            if !self.playing {
                // A OneShot finished mid-block: pad the remainder with silence.
                for ch in 0..out_channels {
                    let slice = out.channel_mut(ch);
                    for s in &mut slice[f..] {
                        *s = 0.0;
                    }
                }
                break;
            }

            let step = f64::from(self.rate.next_sample()) * ratio;
            let pos = self.pos;
            for ch in 0..out_channels {
                let value = self.interpolate(ch, pos);
                out.channel_mut(ch)[f] = value;
            }
            self.advance(step);
            f += 1;
        }
    }

    fn reset(&mut self) {
        self.pos = 0.0;
        self.direction = 1.0;
        self.playing = false;
        // Settle any in-flight rate glide onto its target.
        self.rate.set_target(self.rate.target(), Ramp::Immediate);
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use alloc::vec;
    use alloc::vec::Vec;

    /// Builds a one-frame-per-index mono ramp: `sample[i] == i as f32`.
    fn ramp(len: usize) -> Vec<Vec<Sample>> {
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push(i as Sample);
        }
        vec![data]
    }

    /// Renders `frames` frames of a mono player into a fresh buffer and returns
    /// the captured output channel.
    fn render_mono(node: &mut SamplePlayerNode, sample_rate: u32, frames: usize) -> Vec<Sample> {
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, frames)];
        let ctx = RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        outputs[0].channel(0).to_vec()
    }

    #[test]
    fn unity_rate_linear_replays_verbatim() {
        let mut node = SamplePlayerNode::new(ramp(8), 48_000);
        node.play();
        let out = render_mono(&mut node, 48_000, 6);
        for (i, v) in out.iter().enumerate() {
            assert!((v - i as Sample).abs() < 1e-6, "frame {i} = {v}");
        }
    }

    #[test]
    fn double_rate_skips_every_other_frame() {
        let mut node = SamplePlayerNode::new(ramp(16), 48_000);
        node.set_rate(2.0);
        node.play();
        let out = render_mono(&mut node, 48_000, 4);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 4.0).abs() < 1e-6);
        assert!((out[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn catmull_rom_is_exact_at_integers_and_bounded_between_neighbours() {
        let mut node = SamplePlayerNode::new(ramp(8), 48_000);
        node.set_interpolation(Interpolation::CatmullRom);
        node.play();

        // Exact at an integer position.
        node.seek(3.0);
        let at_integer = render_mono(&mut node, 48_000, 1);
        assert!((at_integer[0] - 3.0).abs() < 1e-5);

        // Midpoint lies strictly between the two bracketing samples.
        node.seek(3.5);
        let at_mid = render_mono(&mut node, 48_000, 1);
        assert!(at_mid[0] > 3.0 && at_mid[0] < 4.0, "mid = {}", at_mid[0]);
        // For linear source data Catmull-Rom reproduces the line exactly.
        assert!((at_mid[0] - 3.5).abs() < 1e-5);
    }

    #[test]
    fn forward_loop_wraps_past_end_back_to_start() {
        let mut node = SamplePlayerNode::new(ramp(8), 48_000);
        node.set_loop_mode(LoopMode::Forward { start: 0, end: 2 });
        node.play();
        let out = render_mono(&mut node, 48_000, 6);
        let expected = [0.0, 1.0, 2.0, 1.0, 2.0, 1.0];
        for (i, e) in expected.iter().enumerate() {
            assert!((out[i] - e).abs() < 1e-6, "frame {i} = {}", out[i]);
        }
        assert!(node.is_playing());
    }

    #[test]
    fn one_shot_stops_and_goes_silent_after_the_end() {
        let mut node = SamplePlayerNode::new(ramp(4), 48_000);
        node.play();
        let out = render_mono(&mut node, 48_000, 8);
        // First four frames replay the data.
        for (i, v) in out.iter().take(4).enumerate() {
            assert!((v - i as Sample).abs() < 1e-6, "frame {i} = {v}");
        }
        // Everything afterwards is silence and the node has stopped.
        for (i, v) in out.iter().enumerate().skip(4) {
            assert!(v.abs() < 1e-9, "frame {i} = {v}");
        }
        assert!(!node.is_playing());
    }

    #[test]
    fn ping_pong_reverses_at_endpoints() {
        let mut node = SamplePlayerNode::new(ramp(8), 48_000);
        node.set_loop_mode(LoopMode::PingPong { start: 0, end: 2 });
        node.play();
        let out = render_mono(&mut node, 48_000, 6);
        let expected = [0.0, 1.0, 2.0, 1.0, 0.0, 1.0];
        for (i, e) in expected.iter().enumerate() {
            assert!((out[i] - e).abs() < 1e-6, "frame {i} = {}", out[i]);
        }
        assert!(node.is_playing());
    }

    #[test]
    fn half_source_rate_halves_the_effective_step() {
        // Source authored at 22.05 kHz, graph rendering at 44.1 kHz -> ratio 0.5.
        let mut node = SamplePlayerNode::new(ramp(8), 22_050);
        node.play();
        let out = render_mono(&mut node, 44_100, 4);
        let expected = [0.0, 0.5, 1.0, 1.5];
        for (i, e) in expected.iter().enumerate() {
            assert!((out[i] - e).abs() < 1e-6, "frame {i} = {}", out[i]);
        }
    }

    #[test]
    fn reset_rewinds_and_stops() {
        let mut node = SamplePlayerNode::new(ramp(8), 48_000);
        node.play();
        let _rendered = render_mono(&mut node, 48_000, 3);
        assert!(node.position() > 0.0);
        node.reset();
        assert_eq!(node.position(), 0.0);
        assert!(!node.is_playing());
    }

    #[test]
    fn empty_data_is_always_silent() {
        let mut node = SamplePlayerNode::new(Vec::new(), 48_000);
        node.play();
        assert!(!node.is_playing());
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 4)];
        let ctx = RenderContext {
            sample_rate: 48_000,
            frames: 4,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        for v in outputs[0].channel(0) {
            assert!(v.abs() < 1e-9);
        }
    }
}
