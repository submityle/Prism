//! Look-ahead brick-wall peak limiter.
//!
//! A limiter is a compressor with an (effectively) infinite ratio and a hard
//! ceiling: it guarantees the output never exceeds a chosen level. This node
//! uses **look-ahead** — the audio is delayed by a few milliseconds while the
//! detector reads the un-delayed signal — so the gain can slew down smoothly
//! and reach full attenuation exactly as a transient arrives, avoiding the
//! audible distortion of a zero-attack clipper. A final clamp at the ceiling
//! guarantees a true brick wall even for the residual overshoot that smoothing
//! alone would let through.
//!
//! Storage is allocated at construction, so
//! [`LimiterNode::process`](crate::graph::AudioNode::process) is real-time safe.

use alloc::vec::Vec;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};
use crate::nodes::dynamics::detector::GainBallistics;
use crate::param::Smoothed;

/// Construction parameters for a [`LimiterNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LimiterParams {
    /// Output ceiling in dBFS; the output magnitude never exceeds this.
    pub ceiling_db: Sample,
    /// Release time in milliseconds.
    pub release_ms: Sample,
    /// Look-ahead time in milliseconds (`0` = none, but then the ceiling is
    /// only enforced by the safety clamp, which can distort transients).
    pub lookahead_ms: Sample,
    /// Input gain in dB applied before limiting (drive into the ceiling).
    pub input_gain_db: Sample,
}

impl Default for LimiterParams {
    fn default() -> Self {
        Self {
            ceiling_db: -0.3,
            release_ms: 100.0,
            lookahead_ms: 5.0,
            input_gain_db: 0.0,
        }
    }
}

/// A look-ahead brick-wall peak limiter (input port 0 -> output port 0).
#[derive(Debug, Clone)]
pub struct LimiterNode {
    /// Ceiling in linear amplitude.
    ceiling_lin: Sample,
    /// Ceiling in dBFS (for the gain computer).
    ceiling_db: Sample,
    /// Smoothed input drive gain (linear).
    input_gain: Smoothed,
    /// Attack / release ballistics on the gain reduction.
    ballistics: GainBallistics,
    /// Look-ahead length in frames (`0` = disabled).
    lookahead: usize,
    /// One look-ahead ring per channel.
    rings: Vec<Vec<Sample>>,
    /// Shared write cursor.
    write_pos: usize,
}

impl LimiterNode {
    /// Builds a limiter for `channels` channels at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: LimiterParams) -> Self {
        let channels = channels.max(1);
        let lookahead = bevy_math::ops::round(params.lookahead_ms.max(0.0) * (sample_rate as Sample) * 0.001)
            as usize;
        let rings = if lookahead > 0 {
            let mut v = Vec::with_capacity(channels);
            for _ in 0..channels {
                let mut r = Vec::with_capacity(lookahead);
                r.resize(lookahead, 0.0);
                v.push(r);
            }
            v
        } else {
            Vec::new()
        };
        // Attack reaches full attenuation within the look-ahead window so the
        // gain bottoms out just as the transient reaches the output.
        let attack_ms = if params.lookahead_ms > 0.0 {
            params.lookahead_ms * 0.5
        } else {
            0.05
        };

        Self {
            ceiling_lin: db_to_linear(params.ceiling_db),
            ceiling_db: params.ceiling_db,
            input_gain: Smoothed::new(db_to_linear(params.input_gain_db)),
            ballistics: GainBallistics::new(attack_ms, params.release_ms, sample_rate),
            lookahead,
            rings,
            write_pos: 0,
        }
    }

    /// Returns the current smoothed gain reduction in decibels (`>= 0`).
    #[inline]
    #[must_use]
    pub fn gain_reduction_db(&self) -> Sample {
        self.ballistics.current_db()
    }
}

impl AudioNode for LimiterNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels());
        let frames = output.active_frames();

        for f in 0..frames {
            let drive = self.input_gain.next_sample();

            // Peak of the loudest driven, un-delayed channel.
            let mut peak = 0.0;
            for ch in 0..channels {
                let a = (input.channel(ch)[f] * drive).abs();
                if a > peak {
                    peak = a;
                }
            }

            // Required reduction to bring the peak down to the ceiling.
            let target_db = if peak > self.ceiling_lin && peak > 0.0 {
                crate::math::linear_to_db(peak) - self.ceiling_db
            } else {
                0.0
            };
            let reduction_db = self.ballistics.process(target_db.max(0.0));
            let gain = db_to_linear(-reduction_db);

            let w = self.write_pos;
            for ch in 0..channels {
                let x = input.channel(ch)[f] * drive;
                let delayed = if self.lookahead > 0 {
                    let d = self.rings[ch][w];
                    self.rings[ch][w] = flush_denormal(x);
                    d
                } else {
                    x
                };
                // Smoothed limiting plus a hard ceiling safety clamp.
                let limited = (delayed * gain).clamp(-self.ceiling_lin, self.ceiling_lin);
                output.channel_mut(ch)[f] = limited;
            }

            if self.lookahead > 0 {
                self.write_pos = if w + 1 == self.lookahead { 0 } else { w + 1 };
            }
        }
    }

    fn reset(&mut self) {
        self.ballistics.reset();
        for ring in &mut self.rings {
            for s in ring.iter_mut() {
                *s = 0.0;
            }
        }
        self.write_pos = 0;
        self.input_gain = Smoothed::new(self.input_gain.target());
    }

    fn latency_frames(&self) -> u32 {
        self.lookahead as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: 48_000,
            frames,
            playhead: 0,
        }
    }

    fn mono(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Mono, frames)
    }

    #[test]
    fn output_never_exceeds_ceiling() {
        let params = LimiterParams {
            ceiling_db: -6.0,
            ..LimiterParams::default()
        };
        let ceiling = db_to_linear(-6.0);
        let mut node = LimiterNode::new(48_000, 1, params);
        let mut input = mono(9_600);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            // Hot 0 dBFS sine plus a spike.
            *s = bevy_math::ops::sin(0.05 * i as Sample);
        }
        input.channel_mut(0)[5_000] = 4.0;
        let inputs = [input];
        let mut outputs = [mono(9_600)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(9_600), &mut io);
        for &y in outputs[0].channel(0) {
            assert!(y.abs() <= ceiling + 1e-4, "exceeded ceiling: {y}");
        }
    }

    #[test]
    fn quiet_signal_is_untouched() {
        let mut node = LimiterNode::new(48_000, 1, LimiterParams::default());
        let mut input = mono(512);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = 0.05 * bevy_math::ops::sin(0.05 * i as Sample);
        }
        let inputs = [input.clone()];
        let mut outputs = [mono(512)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(512), &mut io);
        // Latency-aligned comparison: output is delayed by look-ahead.
        let lat = node.latency_frames() as usize;
        for i in 0..(512 - lat) {
            let o = outputs[0].channel(0)[i + lat];
            let x = inputs[0].channel(0)[i];
            assert!((o - x).abs() < 1e-3, "{o} vs {x}");
        }
    }

    #[test]
    fn lookahead_reports_latency() {
        let node = LimiterNode::new(48_000, 2, LimiterParams::default());
        assert_eq!(node.latency_frames(), 240);
    }

    #[test]
    fn reset_clears_state() {
        let mut node = LimiterNode::new(48_000, 1, LimiterParams::default());
        let mut input = mono(256);
        for s in input.channel_mut(0).iter_mut() {
            *s = 2.0;
        }
        let inputs = [input];
        let mut outputs = [mono(256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        node.reset();
        assert!(node.gain_reduction_db().abs() < 1e-6);
    }
}
