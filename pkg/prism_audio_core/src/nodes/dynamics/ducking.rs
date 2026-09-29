//! Side-chain ducking: attenuate a main signal in response to a separate key
//! (side-chain) signal.
//!
//! Ducking is the "lower the music while the voice-over plays" effect and the
//! engine backbone of dialogue-priority mixing in games (UE's source-bus
//! side-chain, Unity's snapshot ducking, Godot's bus send routing). It is a
//! compressor whose detector listens to a *different* input — input port 1 (the
//! key) — while the gain is applied to input port 0 (the main). The attenuation
//! is bounded by `range_db` so the main never disappears entirely.
//!
//! - **Input port 0**: main signal (e.g. music / ambience).
//! - **Input port 1**: key / side-chain signal (e.g. dialogue).
//! - **Output port 0**: the ducked main signal.
//!
//! Storage is allocated at construction, so
//! [`DuckingNode::process`](crate::graph::AudioNode::process) is real-time
//! safe.

use crate::buffer::AudioBuffer;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear};
use crate::nodes::dynamics::detector::{
    DetectionMode, GainBallistics, LevelDetector, compressor_reduction_db,
};

/// Construction parameters for a [`DuckingNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DuckingParams {
    /// Key-level threshold in dBFS above which the main is ducked.
    pub threshold_db: Sample,
    /// Ducking ratio (`>= 1`); higher ducks harder per dB of key over threshold.
    pub ratio: Sample,
    /// Soft-knee width in dB centered on the threshold.
    pub knee_db: Sample,
    /// Maximum attenuation applied to the main signal, in dB.
    pub range_db: Sample,
    /// Attack (duck-down) time in milliseconds.
    pub attack_ms: Sample,
    /// Release (recover) time in milliseconds.
    pub release_ms: Sample,
    /// Key detection mode (peak or RMS).
    pub detection: DetectionMode,
    /// RMS averaging window in milliseconds (ignored for peak detection).
    pub rms_window_ms: Sample,
}

impl Default for DuckingParams {
    fn default() -> Self {
        Self {
            threshold_db: -30.0,
            ratio: 8.0,
            knee_db: 6.0,
            range_db: 18.0,
            attack_ms: 15.0,
            release_ms: 250.0,
            detection: DetectionMode::Rms,
            rms_window_ms: 15.0,
        }
    }
}

/// A side-chain ducker (main on input 0, key on input 1 -> output 0).
#[derive(Debug, Clone)]
pub struct DuckingNode {
    /// Threshold in dBFS.
    threshold_db: Sample,
    /// Ducking ratio.
    ratio: Sample,
    /// Soft-knee width in dB.
    knee_db: Sample,
    /// Maximum attenuation in dB.
    range_db: Sample,
    /// Level detector on the key input.
    detector: LevelDetector,
    /// Attack / release ballistics on the gain reduction.
    ballistics: GainBallistics,
}

impl DuckingNode {
    /// Builds a ducker at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, params: DuckingParams) -> Self {
        Self {
            threshold_db: params.threshold_db,
            ratio: params.ratio.max(1.0),
            knee_db: params.knee_db.max(0.0),
            range_db: params.range_db.max(0.0),
            detector: LevelDetector::new(params.detection, params.rms_window_ms, sample_rate),
            ballistics: GainBallistics::new(params.attack_ms, params.release_ms, sample_rate),
        }
    }

    /// Returns the current smoothed attenuation applied to the main, in dB.
    #[inline]
    #[must_use]
    pub fn gain_reduction_db(&self) -> Sample {
        self.ballistics.current_db()
    }
}

impl AudioNode for DuckingNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (inputs, outputs) = io.split();
        let main = &inputs[0];
        // The key is optional: if no second input is wired, nothing ducks.
        let key = inputs.get(1);
        let out = &mut outputs[0];

        let channels = out.channels().min(main.channels());
        let frames = out.active_frames();
        let key_channels = key.map(AudioBuffer::channels).unwrap_or(0);

        for f in 0..frames {
            // Detect the loudest key channel at this frame.
            let mut key_peak = 0.0;
            if let Some(k) = key {
                for ch in 0..key_channels {
                    let a = k.channel(ch)[f].abs();
                    if a > key_peak {
                        key_peak = a;
                    }
                }
            }

            let level_db = self.detector.level_db(key_peak);
            let raw = compressor_reduction_db(level_db, self.threshold_db, self.ratio, self.knee_db);
            let target = raw.min(self.range_db);
            let reduction_db = self.ballistics.process(target);
            let gain = db_to_linear(-reduction_db);

            for ch in 0..channels {
                out.channel_mut(ch)[f] = main.channel(ch)[f] * gain;
            }
        }
    }

    fn reset(&mut self) {
        self.detector.reset();
        self.ballistics.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::ChannelLayout;

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
    fn silent_key_leaves_main_untouched() {
        let mut node = DuckingNode::new(48_000, DuckingParams::default());
        let mut main = mono(512);
        for (i, s) in main.channel_mut(0).iter_mut().enumerate() {
            *s = 0.5 * bevy_math::ops::sin(0.05 * i as Sample);
        }
        let key = mono(512); // silent
        let inputs = [main.clone(), key];
        let mut outputs = [mono(512)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(512), &mut io);
        for (o, i) in outputs[0].channel(0).iter().zip(inputs[0].channel(0)) {
            assert!((o - i).abs() < 1e-4, "{o} vs {i}");
        }
    }

    #[test]
    fn loud_key_ducks_the_main() {
        let params = DuckingParams {
            attack_ms: 1.0,
            range_db: 18.0,
            ..DuckingParams::default()
        };
        let mut node = DuckingNode::new(48_000, params);
        let mut main = mono(9_600);
        let mut key = mono(9_600);
        for i in 0..9_600 {
            main.channel_mut(0)[i] = 0.5 * bevy_math::ops::sin(0.05 * i as Sample);
            key.channel_mut(0)[i] = 0.9 * bevy_math::ops::sin(0.07 * i as Sample);
        }
        let inputs = [main.clone(), key];
        let mut outputs = [mono(9_600)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(9_600), &mut io);
        let peak_in = inputs[0].channel(0)[4_800..].iter().fold(0.0, |m, &v| v.abs().max(m));
        let peak_out = outputs[0].channel(0)[4_800..].iter().fold(0.0, |m, &v| v.abs().max(m));
        assert!(peak_out < peak_in * 0.6, "not ducked: in={peak_in} out={peak_out}");
        assert!(node.gain_reduction_db() > 3.0);
        // Attenuation is bounded by range_db.
        assert!(node.gain_reduction_db() <= 18.0 + 1e-3);
    }

    #[test]
    fn missing_key_is_a_passthrough() {
        let mut node = DuckingNode::new(48_000, DuckingParams::default());
        let mut main = mono(128);
        for (i, s) in main.channel_mut(0).iter_mut().enumerate() {
            *s = 0.5 * bevy_math::ops::sin(0.05 * i as Sample);
        }
        let inputs = [main.clone()]; // no key input
        let mut outputs = [mono(128)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(128), &mut io);
        for (o, i) in outputs[0].channel(0).iter().zip(inputs[0].channel(0)) {
            assert!((o - i).abs() < 1e-6, "{o} vs {i}");
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut node = DuckingNode::new(48_000, DuckingParams::default());
        let mut main = mono(256);
        let mut key = mono(256);
        for i in 0..256 {
            main.channel_mut(0)[i] = 0.5;
            key.channel_mut(0)[i] = 1.0;
        }
        let inputs = [main, key];
        let mut outputs = [mono(256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        node.reset();
        assert!(node.gain_reduction_db().abs() < 1e-6);
    }
}
