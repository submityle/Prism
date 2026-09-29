//! Feed-forward dynamic-range compressor with soft knee, peak/RMS detection,
//! optional look-ahead, make-up gain, and wet/dry (parallel) mixing.
//!
//! The detector is **stereo-linked**: a single control signal derived from the
//! loudest channel drives the gain of every channel, so the stereo image never
//! shifts under compression. Look-ahead delays the audio (but not the
//! detector) so the gain reduction can begin *before* a transient arrives,
//! catching peaks a naive compressor would miss.
//!
//! All storage is allocated at construction, so
//! [`CompressorNode::process`](crate::graph::AudioNode::process) is real-time
//! safe.

use alloc::vec::Vec;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};
use crate::nodes::dynamics::detector::{
    DetectionMode, GainBallistics, LevelDetector, compressor_reduction_db,
};
use crate::param::{Ramp, Smoothed};

/// Construction parameters for a [`CompressorNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CompressorParams {
    /// Threshold in dBFS above which compression begins.
    pub threshold_db: Sample,
    /// Compression ratio (`>= 1`); `4.0` means 4 dB in per 1 dB out.
    pub ratio: Sample,
    /// Soft-knee width in dB centered on the threshold (`0` = hard knee).
    pub knee_db: Sample,
    /// Attack time in milliseconds.
    pub attack_ms: Sample,
    /// Release time in milliseconds.
    pub release_ms: Sample,
    /// Make-up gain in dB applied after compression.
    pub makeup_db: Sample,
    /// Look-ahead time in milliseconds (`0` = none).
    pub lookahead_ms: Sample,
    /// Side-chain detection mode (peak or RMS).
    pub detection: DetectionMode,
    /// RMS averaging window in milliseconds (ignored for peak detection).
    pub rms_window_ms: Sample,
    /// Wet (compressed) mix gain for parallel compression.
    pub wet: Sample,
    /// Dry (uncompressed) mix gain for parallel compression.
    pub dry: Sample,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 4.0,
            knee_db: 6.0,
            attack_ms: 10.0,
            release_ms: 120.0,
            makeup_db: 0.0,
            lookahead_ms: 0.0,
            detection: DetectionMode::Peak,
            rms_window_ms: 10.0,
            wet: 1.0,
            dry: 0.0,
        }
    }
}

/// A stereo-linked feed-forward compressor (input port 0 -> output port 0).
#[derive(Debug, Clone)]
pub struct CompressorNode {
    /// Threshold in dBFS.
    threshold_db: Sample,
    /// Compression ratio.
    ratio: Sample,
    /// Soft-knee width in dB.
    knee_db: Sample,
    /// Level detector shared across channels (stereo-linked).
    detector: LevelDetector,
    /// Attack / release ballistics on the gain-reduction control.
    ballistics: GainBallistics,
    /// Smoothed make-up gain (linear).
    makeup: Smoothed,
    /// Smoothed wet mix gain.
    wet: Smoothed,
    /// Smoothed dry mix gain.
    dry: Smoothed,
    /// Look-ahead length in frames (`0` = disabled).
    lookahead: usize,
    /// One look-ahead ring per channel (empty when look-ahead is disabled).
    rings: Vec<Vec<Sample>>,
    /// Shared write cursor into every look-ahead ring.
    write_pos: usize,
}

impl CompressorNode {
    /// Builds a compressor for `channels` channels at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: CompressorParams) -> Self {
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

        Self {
            threshold_db: params.threshold_db,
            ratio: params.ratio.max(1.0),
            knee_db: params.knee_db.max(0.0),
            detector: LevelDetector::new(params.detection, params.rms_window_ms, sample_rate),
            ballistics: GainBallistics::new(params.attack_ms, params.release_ms, sample_rate),
            makeup: Smoothed::new(db_to_linear(params.makeup_db)),
            wet: Smoothed::new(params.wet),
            dry: Smoothed::new(params.dry),
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

    /// Sets the make-up gain in dB.
    #[inline]
    pub fn set_makeup_db(&mut self, makeup_db: Sample, ramp: Ramp) {
        self.makeup.set_target(db_to_linear(makeup_db), ramp);
    }

    /// Sets the wet mix gain.
    #[inline]
    pub fn set_wet(&mut self, wet: Sample, ramp: Ramp) {
        self.wet.set_target(wet, ramp);
    }

    /// Sets the dry mix gain.
    #[inline]
    pub fn set_dry(&mut self, dry: Sample, ramp: Ramp) {
        self.dry.set_target(dry, ramp);
    }
}

impl AudioNode for CompressorNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels());
        let frames = output.active_frames();

        for f in 0..frames {
            // Stereo-linked detection uses the loudest *undelayed* channel.
            let mut peak = 0.0;
            for ch in 0..channels {
                let a = input.channel(ch)[f].abs();
                if a > peak {
                    peak = a;
                }
            }

            let level_db = self.detector.level_db(peak);
            let target =
                compressor_reduction_db(level_db, self.threshold_db, self.ratio, self.knee_db);
            let reduction_db = self.ballistics.process(target);
            let gain = db_to_linear(-reduction_db) * self.makeup.next_sample();
            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();

            let w = self.write_pos;
            for ch in 0..channels {
                let x = input.channel(ch)[f];
                let delayed = if self.lookahead > 0 {
                    let d = self.rings[ch][w];
                    self.rings[ch][w] = flush_denormal(x);
                    d
                } else {
                    x
                };
                let compressed = delayed * gain;
                output.channel_mut(ch)[f] = dry * delayed + wet * compressed;
            }

            if self.lookahead > 0 {
                self.write_pos = if w + 1 == self.lookahead { 0 } else { w + 1 };
            }
        }
    }

    fn reset(&mut self) {
        self.detector.reset();
        self.ballistics.reset();
        for ring in &mut self.rings {
            for s in ring.iter_mut() {
                *s = 0.0;
            }
        }
        self.write_pos = 0;
        self.makeup = Smoothed::new(self.makeup.target());
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
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
    fn quiet_signal_passes_unchanged() {
        let mut node = CompressorNode::new(48_000, 1, CompressorParams::default());
        let mut input = mono(512);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            // -40 dBFS sine, well below the -18 dB threshold.
            *s = 0.01 * bevy_math::ops::sin(0.05 * i as Sample);
        }
        let inputs = [input.clone()];
        let mut outputs = [mono(512)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(512), &mut io);
        for (o, i) in outputs[0].channel(0).iter().zip(inputs[0].channel(0)) {
            assert!((o - i).abs() < 1e-3, "{o} vs {i}");
        }
    }

    #[test]
    fn loud_signal_is_attenuated() {
        let params = CompressorParams {
            attack_ms: 0.1,
            ..CompressorParams::default()
        };
        let mut node = CompressorNode::new(48_000, 1, params);
        let mut input = mono(4_800);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            // 0 dBFS sine, well above the -18 dB threshold.
            *s = bevy_math::ops::sin(0.05 * i as Sample);
        }
        let inputs = [input.clone()];
        let mut outputs = [mono(4_800)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(4_800), &mut io);
        // Compare peak of the settled tail.
        let peak_in = inputs[0].channel(0)[2_400..]
            .iter()
            .fold(0.0, |m, &v| v.abs().max(m));
        let peak_out = outputs[0].channel(0)[2_400..]
            .iter()
            .fold(0.0, |m, &v| v.abs().max(m));
        assert!(peak_out < peak_in * 0.8, "in={peak_in} out={peak_out}");
        assert!(node.gain_reduction_db() > 3.0);
    }

    #[test]
    fn lookahead_reports_latency() {
        let params = CompressorParams {
            lookahead_ms: 5.0,
            ..CompressorParams::default()
        };
        let node = CompressorNode::new(48_000, 2, params);
        assert_eq!(node.latency_frames(), 240);
    }

    #[test]
    fn reset_clears_state() {
        let params = CompressorParams {
            lookahead_ms: 2.0,
            ..CompressorParams::default()
        };
        let mut node = CompressorNode::new(48_000, 1, params);
        let mut input = mono(256);
        for s in input.channel_mut(0).iter_mut() {
            *s = 1.0;
        }
        let inputs = [input];
        let mut outputs = [mono(256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        node.reset();
        assert!(node.gain_reduction_db().abs() < 1e-6);
    }

    #[test]
    fn parallel_mix_blends_dry() {
        let params = CompressorParams {
            wet: 0.0,
            dry: 1.0,
            ..CompressorParams::default()
        };
        let mut node = CompressorNode::new(48_000, 1, params);
        let mut input = mono(128);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = bevy_math::ops::sin(0.05 * i as Sample);
        }
        let inputs = [input.clone()];
        let mut outputs = [mono(128)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(128), &mut io);
        for (o, i) in outputs[0].channel(0).iter().zip(inputs[0].channel(0)) {
            assert!((o - i).abs() < 1e-6, "dry passthrough {o} vs {i}");
        }
    }
}
