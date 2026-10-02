//! **Resolved music actions**: the flat, sample-timestamped planner output.
//!
//! A [`MusicAction`] is one concrete instruction naming *what* to do, *when*
//! (an absolute sample position), and *at what gain*. The live
//! [`crate::system::MusicSystem`] resolves every high-level music request
//! (start a playlist, transition to a segment, raise the intensity, fire a
//! stinger, step a clip graph) into an ordered stream of these actions. The
//! stream is the sole contract with the lower runtime: this crate performs no
//! DSP and spawns no voices, it only decides the plan.
//!
//! Each action carries fully resolved sample fields so the runtime never has to
//! redo tempo arithmetic: fade lengths are already in samples and the switch
//! point is already quantized.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is an
//! original, data-only command vocabulary. No AI/ML.
//!
//! # Relationship
//!
//! [`MusicAction`]s are produced by [`crate::system::MusicSystem`] from the
//! segments, layers, stingers, and clip graphs in
//! [`crate::model::MusicModel`]; fade lengths and [`crate::transition::FadeCurve`]
//! kinds are forwarded verbatim from the resolved
//! [`crate::transition::Transition`]. The lower runtime consumes them to spawn
//! and retire voices.

use prism_audio_core::math::Sample;

use crate::id::{LayerId, SegmentId, SoundId, StingerId};
use crate::transition::FadeCurve;

/// One fully resolved, sample-timestamped music instruction.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum MusicAction {
    /// Begin playing a segment's sound at `at_sample`, ramping in over
    /// `fade_in` samples.
    PlaySegment {
        /// Segment being started.
        segment: SegmentId,
        /// Leaf sound the segment plays.
        sound: SoundId,
        /// Absolute sample position playback begins.
        at_sample: u64,
        /// Playback gain in decibels.
        gain_db: Sample,
        /// Fade-in length in samples.
        fade_in: u64,
        /// Gain shape of the fade-in.
        curve: FadeCurve,
    },
    /// Stop a playing segment at `at_sample`, ramping out over `fade_out`
    /// samples.
    StopSegment {
        /// Segment being stopped.
        segment: SegmentId,
        /// Leaf sound the segment plays.
        sound: SoundId,
        /// Absolute sample position the fade-out begins.
        at_sample: u64,
        /// Fade-out length in samples.
        fade_out: u64,
        /// Gain shape of the fade-out.
        curve: FadeCurve,
    },
    /// Add a vertical layer at `at_sample`, ramping in over `fade_in` samples.
    StartLayer {
        /// Layer being started.
        layer: LayerId,
        /// Leaf sound the layer plays.
        sound: SoundId,
        /// Absolute sample position playback begins.
        at_sample: u64,
        /// Playback gain in decibels.
        gain_db: Sample,
        /// Fade-in length in samples.
        fade_in: u64,
        /// Gain shape of the fade-in.
        curve: FadeCurve,
    },
    /// Remove a vertical layer at `at_sample`, ramping out over `fade_out`
    /// samples.
    StopLayer {
        /// Layer being stopped.
        layer: LayerId,
        /// Leaf sound the layer plays.
        sound: SoundId,
        /// Absolute sample position the fade-out begins.
        at_sample: u64,
        /// Fade-out length in samples.
        fade_out: u64,
        /// Gain shape of the fade-out.
        curve: FadeCurve,
    },
    /// Set an already-playing layer's gain at `at_sample`.
    SetLayerGain {
        /// Layer whose gain changes.
        layer: LayerId,
        /// Absolute sample position the gain applies.
        at_sample: u64,
        /// New playback gain in decibels.
        gain_db: Sample,
    },
    /// Play a one-shot stinger overlay at `at_sample`.
    PlayStinger {
        /// Stinger being played.
        stinger: StingerId,
        /// Leaf sound the stinger plays.
        sound: SoundId,
        /// Absolute sample position the overlay begins.
        at_sample: u64,
        /// Playback gain in decibels.
        gain_db: Sample,
    },
}

impl MusicAction {
    /// Returns the absolute sample position this action is scheduled at.
    #[must_use]
    pub fn at_sample(&self) -> u64 {
        match *self {
            Self::PlaySegment { at_sample, .. }
            | Self::StopSegment { at_sample, .. }
            | Self::StartLayer { at_sample, .. }
            | Self::StopLayer { at_sample, .. }
            | Self::SetLayerGain { at_sample, .. }
            | Self::PlayStinger { at_sample, .. } => at_sample,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_sample_reads_timestamp_of_every_variant() {
        let play = MusicAction::PlaySegment {
            segment: SegmentId::new(1),
            sound: SoundId::new(2),
            at_sample: 100,
            gain_db: 0.0,
            fade_in: 10,
            curve: FadeCurve::EqualPower,
        };
        assert_eq!(play.at_sample(), 100);

        let sting = MusicAction::PlayStinger {
            stinger: StingerId::new(1),
            sound: SoundId::new(2),
            at_sample: 250,
            gain_db: -3.0,
        };
        assert_eq!(sting.at_sample(), 250);

        let gain = MusicAction::SetLayerGain {
            layer: LayerId::new(1),
            at_sample: 42,
            gain_db: -6.0,
        };
        assert_eq!(gain.at_sample(), 42);
    }

    #[test]
    fn actions_compare_by_value() {
        let a = MusicAction::StopLayer {
            layer: LayerId::new(1),
            sound: SoundId::new(2),
            at_sample: 10,
            fade_out: 5,
            curve: FadeCurve::Linear,
        };
        let b = a;
        assert_eq!(a, b);
    }
}
