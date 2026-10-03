//! Smoothed live tuning of parameters with an asset write-back set.
//!
//! Live tuning lets a tool nudge an RTPC, a bus gain, or an attenuation-curve
//! control and hear the result immediately. Applying a jump straight to a
//! per-sample control would click, so every tuned target glides through a
//! [`Smoothed`] value from `prism_audio_core`. The [`LiveTuner`] also records
//! the final chosen value of each edit in a pending write-back set, so once the
//! sound designer is satisfied the edits can be flushed back onto the authoring
//! asset in one batch.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the live-tuning path of design section 38. Smoothing reuses the
//! parameter-ramp machinery of design section 7 ([`Smoothed`] and [`Ramp`])
//! rather than re-deriving it, and the pending write-back set is the hand-off
//! toward the authoring asset edits of design section 38.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_audio_core::param::{Ramp, Smoothed};

/// Which kind of control a tuning edit targets.
///
/// The inner `u64` is the stable identifier of the specific control within its
/// kind, so an RTPC and a bus with the same numeric id are still distinct
/// targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum TuningTarget {
    /// A real-time parameter control (an RTPC).
    Rtpc(u64),
    /// A mix-bus gain in decibels.
    BusGain(u64),
    /// An attenuation-curve control value.
    Attenuation(u64),
}

/// How a tuning edit should glide toward its new value.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Smoothing {
    /// Jump immediately, with no glide (safe only for non per-sample controls).
    Immediate,
    /// Glide linearly over the given number of seconds.
    LinearSeconds(f32),
    /// Glide with a one-pole response whose time constant is given in seconds.
    ExponentialSeconds(f32),
}

impl Smoothing {
    /// Converts this request into a [`Ramp`] at the given `sample_rate`.
    fn to_ramp(self, sample_rate: u32) -> Ramp {
        match self {
            Smoothing::Immediate => Ramp::Immediate,
            Smoothing::LinearSeconds(seconds) => {
                Ramp::linear_seconds(seconds, sample_rate)
            }
            Smoothing::ExponentialSeconds(seconds) => {
                let tau = (seconds.max(0.0) * sample_rate as f32).max(1.0);
                Ramp::Exponential { tau_samples: tau }
            }
        }
    }
}

/// A smoothed live tuner with a pending write-back set.
///
/// Each tuned [`TuningTarget`] owns a [`Smoothed`] value that is advanced by
/// [`LiveTuner::advance`]; the latest requested value for each target is also
/// recorded for later write-back via [`LiveTuner::take_pending`].
#[derive(Debug, Clone)]
pub struct LiveTuner {
    sample_rate: u32,
    channels: BTreeMap<TuningTarget, Smoothed>,
    pending: BTreeMap<TuningTarget, f32>,
}

impl LiveTuner {
    /// Creates a tuner that smooths at the given `sample_rate` in hertz.
    ///
    /// A `sample_rate` of zero is treated as one so ramp math stays finite.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            channels: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }

    /// Returns the sample rate used for smoothing.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Applies a tuning edit to `target`, gliding toward `value`.
    ///
    /// The first edit of a target seeds its smoother settled at `value`; later
    /// edits glide from the current value using `smoothing`. Either way the
    /// requested `value` is recorded in the pending write-back set.
    pub fn tune(&mut self, target: TuningTarget, value: f32, smoothing: Smoothing) {
        let ramp = smoothing.to_ramp(self.sample_rate);
        match self.channels.get_mut(&target) {
            Some(smoothed) => smoothed.set_target(value, ramp),
            None => {
                self.channels.insert(target, Smoothed::new(value));
            }
        }
        self.pending.insert(target, value);
    }

    /// Advances every live smoother by `frames` samples.
    ///
    /// This is the per-block tick a tool calls so the audible value catches up
    /// to its target without clicking.
    pub fn advance(&mut self, frames: u32) {
        for smoothed in self.channels.values_mut() {
            for _ in 0..frames {
                smoothed.next_sample();
            }
        }
    }

    /// Returns the current audible value of `target`, if it is being tuned.
    #[must_use]
    pub fn current(&self, target: TuningTarget) -> Option<f32> {
        self.channels.get(&target).map(Smoothed::current)
    }

    /// Returns the value `target` is gliding toward, if it is being tuned.
    #[must_use]
    pub fn goal(&self, target: TuningTarget) -> Option<f32> {
        self.channels.get(&target).map(Smoothed::target)
    }

    /// Returns `true` when every live smoother has reached its target.
    #[must_use]
    pub fn is_settled(&self) -> bool {
        self.channels.values().all(Smoothed::is_settled)
    }

    /// Returns the number of pending write-back edits.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Drains and returns the pending write-back edits, sorted by target.
    ///
    /// Each entry is the final requested value for a target; applying them to
    /// the authoring asset commits the live session's edits.
    #[must_use]
    pub fn take_pending(&mut self) -> Vec<(TuningTarget, f32)> {
        let drained: Vec<(TuningTarget, f32)> = self
            .pending
            .iter()
            .map(|(target, value)| (*target, *value))
            .collect();
        self.pending.clear();
        drained
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-4;

    #[test]
    fn first_edit_seeds_settled() {
        let mut tuner = LiveTuner::new(48_000);
        tuner.tune(TuningTarget::Rtpc(1), 0.75, Smoothing::LinearSeconds(0.1));
        assert!(tuner.is_settled());
        let current = tuner.current(TuningTarget::Rtpc(1)).unwrap();
        assert!((current - 0.75).abs() < EPS);
    }

    #[test]
    fn second_edit_glides_without_jumping() {
        let mut tuner = LiveTuner::new(48_000);
        tuner.tune(TuningTarget::BusGain(2), 0.0, Smoothing::Immediate);
        tuner.tune(TuningTarget::BusGain(2), 1.0, Smoothing::LinearSeconds(0.01));
        // One frame in, the audible value should have moved but not arrived.
        tuner.advance(1);
        let mid = tuner.current(TuningTarget::BusGain(2)).unwrap();
        assert!(mid > 0.0 && mid < 1.0);
        assert!(!tuner.is_settled());
    }

    #[test]
    fn glide_reaches_goal_after_enough_frames() {
        let mut tuner = LiveTuner::new(1_000);
        tuner.tune(TuningTarget::Rtpc(3), 0.0, Smoothing::Immediate);
        tuner.tune(TuningTarget::Rtpc(3), 2.0, Smoothing::LinearSeconds(0.01));
        tuner.advance(32);
        assert!(tuner.is_settled());
        let value = tuner.current(TuningTarget::Rtpc(3)).unwrap();
        assert!((value - 2.0).abs() < EPS);
        let goal = tuner.goal(TuningTarget::Rtpc(3)).unwrap();
        assert!((goal - 2.0).abs() < EPS);
    }

    #[test]
    fn exponential_smoothing_converges() {
        let mut tuner = LiveTuner::new(1_000);
        tuner.tune(TuningTarget::Attenuation(4), 0.0, Smoothing::Immediate);
        tuner.tune(
            TuningTarget::Attenuation(4),
            1.0,
            Smoothing::ExponentialSeconds(0.005),
        );
        tuner.advance(512);
        let value = tuner.current(TuningTarget::Attenuation(4)).unwrap();
        assert!((value - 1.0).abs() < 1.0e-2);
    }

    #[test]
    fn pending_records_latest_value_per_target() {
        let mut tuner = LiveTuner::new(48_000);
        tuner.tune(TuningTarget::Rtpc(1), 0.1, Smoothing::Immediate);
        tuner.tune(TuningTarget::Rtpc(1), 0.9, Smoothing::Immediate);
        tuner.tune(TuningTarget::BusGain(1), -6.0, Smoothing::Immediate);
        assert_eq!(tuner.pending_len(), 2);

        let drained = tuner.take_pending();
        assert_eq!(drained.len(), 2);
        let rtpc = drained
            .iter()
            .find(|(t, _)| *t == TuningTarget::Rtpc(1))
            .map(|(_, v)| *v)
            .unwrap();
        assert!((rtpc - 0.9).abs() < EPS);
        assert_eq!(tuner.pending_len(), 0);
    }

    #[test]
    fn unknown_target_reads_none() {
        let tuner = LiveTuner::new(48_000);
        assert!(tuner.current(TuningTarget::Rtpc(99)).is_none());
        assert!(tuner.goal(TuningTarget::Rtpc(99)).is_none());
        assert!(tuner.is_settled());
    }
}
