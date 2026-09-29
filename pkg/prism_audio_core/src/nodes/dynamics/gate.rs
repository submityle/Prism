//! Downward expander / noise gate.
//!
//! Below the threshold the signal is attenuated (an *expander* when the ratio
//! is modest, a hard *gate* when it is steep), which pushes quiet noise, bleed,
//! and hum further down while leaving louder material untouched. A **hold**
//! time keeps the gate open briefly after the signal drops so it does not
//! chatter on decaying tails, and separate attack (open) and release (close)
//! times shape the transition. The maximum attenuation is bounded by `range`.
//!
//! Storage is allocated at construction, so
//! [`ExpanderGateNode::process`](crate::graph::AudioNode::process) is real-time
//! safe.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear};
use crate::nodes::dynamics::detector::{
    DetectionMode, LevelDetector, expander_reduction_db, time_to_coef,
};

/// Construction parameters for an [`ExpanderGateNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GateParams {
    /// Threshold in dBFS; signal below this is attenuated.
    pub threshold_db: Sample,
    /// Downward-expansion ratio (`>= 1`); large values approach a hard gate.
    pub ratio: Sample,
    /// Soft-knee width in dB centered on the threshold.
    pub knee_db: Sample,
    /// Maximum attenuation in dB applied when fully closed.
    pub range_db: Sample,
    /// Open (attack) time in milliseconds.
    pub attack_ms: Sample,
    /// Hold time in milliseconds after the signal drops below threshold.
    pub hold_ms: Sample,
    /// Close (release) time in milliseconds.
    pub release_ms: Sample,
    /// Side-chain detection mode (peak or RMS).
    pub detection: DetectionMode,
    /// RMS averaging window in milliseconds (ignored for peak detection).
    pub rms_window_ms: Sample,
}

impl Default for GateParams {
    fn default() -> Self {
        Self {
            threshold_db: -45.0,
            ratio: 4.0,
            knee_db: 3.0,
            range_db: 60.0,
            attack_ms: 1.0,
            hold_ms: 50.0,
            release_ms: 120.0,
            detection: DetectionMode::Peak,
            rms_window_ms: 5.0,
        }
    }
}

/// A downward expander / noise gate (input port 0 -> output port 0).
#[derive(Debug, Clone)]
pub struct ExpanderGateNode {
    /// Threshold in dBFS.
    threshold_db: Sample,
    /// Downward-expansion ratio.
    ratio: Sample,
    /// Soft-knee width in dB.
    knee_db: Sample,
    /// Maximum attenuation in dB.
    range_db: Sample,
    /// Stereo-linked level detector.
    detector: LevelDetector,
    /// One-pole coefficient used while opening.
    attack_coef: Sample,
    /// One-pole coefficient used while closing.
    release_coef: Sample,
    /// Hold length in frames.
    hold_frames: u32,
    /// Frames remaining in the current hold window.
    hold_counter: u32,
    /// Current linear gain applied to the signal.
    gain: Sample,
    /// Fully-closed linear gain (`db_to_linear(-range_db)`).
    closed_gain: Sample,
}

impl ExpanderGateNode {
    /// Builds a gate for `channels` channels at `sample_rate` Hz.
    ///
    /// (The channel count only affects stereo-linked detection; the node adapts
    /// to the buffer's channel count at render time.)
    #[must_use]
    pub fn new(sample_rate: u32, _channels: usize, params: GateParams) -> Self {
        let closed_gain = db_to_linear(-params.range_db.max(0.0));
        let hold_frames = bevy_math::ops::round(params.hold_ms.max(0.0) * (sample_rate as Sample) * 0.001) as u32;
        Self {
            threshold_db: params.threshold_db,
            ratio: params.ratio.max(1.0),
            knee_db: params.knee_db.max(0.0),
            range_db: params.range_db.max(0.0),
            detector: LevelDetector::new(params.detection, params.rms_window_ms, sample_rate),
            attack_coef: time_to_coef(params.attack_ms, sample_rate),
            release_coef: time_to_coef(params.release_ms, sample_rate),
            hold_frames,
            hold_counter: 0,
            gain: closed_gain,
            closed_gain,
        }
    }

    /// Returns the current linear gain the gate is applying (`closed..=1`).
    #[inline]
    #[must_use]
    pub fn current_gain(&self) -> Sample {
        self.gain
    }
}

impl AudioNode for ExpanderGateNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels());
        let frames = output.active_frames();

        for f in 0..frames {
            let mut peak = 0.0;
            for ch in 0..channels {
                let a = input.channel(ch)[f].abs();
                if a > peak {
                    peak = a;
                }
            }

            let level_db = self.detector.level_db(peak);
            let reduction =
                expander_reduction_db(level_db, self.threshold_db, self.ratio, self.knee_db, self.range_db);
            let target_gain = db_to_linear(-reduction);

            if target_gain >= self.gain {
                // Opening (or already more open than the target): move quickly
                // and re-arm the hold window.
                self.gain += (1.0 - self.attack_coef) * (target_gain - self.gain);
                self.hold_counter = self.hold_frames;
            } else if self.hold_counter > 0 {
                // Hold: keep the gate open to ride out short dips.
                self.hold_counter -= 1;
            } else {
                // Closing: ease toward the (quieter) target.
                self.gain += (1.0 - self.release_coef) * (target_gain - self.gain);
            }

            let g = self.gain;
            for ch in 0..channels {
                output.channel_mut(ch)[f] = input.channel(ch)[f] * g;
            }
        }
    }

    fn reset(&mut self) {
        self.detector.reset();
        self.gain = self.closed_gain;
        self.hold_counter = 0;
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
    fn loud_signal_opens_the_gate() {
        let params = GateParams {
            attack_ms: 0.1,
            ..GateParams::default()
        };
        let mut node = ExpanderGateNode::new(48_000, 1, params);
        let mut input = mono(4_800);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = 0.8 * bevy_math::ops::sin(0.05 * i as Sample);
        }
        let inputs = [input.clone()];
        let mut outputs = [mono(4_800)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(4_800), &mut io);
        assert!(node.current_gain() > 0.9, "gate did not open: {}", node.current_gain());
        // Settled tail should pass at (near) unity.
        let peak_in = inputs[0].channel(0)[2_400..].iter().fold(0.0, |m, &v| v.abs().max(m));
        let peak_out = outputs[0].channel(0)[2_400..].iter().fold(0.0, |m, &v| v.abs().max(m));
        assert!(peak_out > peak_in * 0.9, "in={peak_in} out={peak_out}");
    }

    #[test]
    fn quiet_noise_is_attenuated() {
        let params = GateParams {
            range_db: 40.0,
            hold_ms: 0.0,
            release_ms: 1.0,
            ..GateParams::default()
        };
        let mut node = ExpanderGateNode::new(48_000, 1, params);
        let mut input = mono(4_800);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            // -60 dBFS hiss, well below the -45 dB threshold.
            *s = 0.001 * bevy_math::ops::sin(0.3 * i as Sample);
        }
        let inputs = [input.clone()];
        let mut outputs = [mono(4_800)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(4_800), &mut io);
        let peak_in = inputs[0].channel(0)[2_400..].iter().fold(0.0, |m, &v| v.abs().max(m));
        let peak_out = outputs[0].channel(0)[2_400..].iter().fold(0.0, |m, &v| v.abs().max(m));
        assert!(peak_out < peak_in * 0.2, "gate did not attenuate: in={peak_in} out={peak_out}");
    }

    #[test]
    fn reset_recloses_the_gate() {
        let mut node = ExpanderGateNode::new(48_000, 1, GateParams::default());
        let mut input = mono(256);
        for s in input.channel_mut(0).iter_mut() {
            *s = 1.0;
        }
        let inputs = [input];
        let mut outputs = [mono(256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        node.reset();
        assert!(node.current_gain() <= db_to_linear(-60.0) + 1e-6);
    }
}
