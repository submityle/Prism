//! Arbitrary loudspeaker-array geometry (design section 49.3).
//!
//! A physical array is an ordered set of loudspeakers, each with a physical
//! position (used by the wave-field and beamforming driving functions), a unit
//! direction in the listener-local frame (used by the amplitude panners), an
//! output channel index, and a low-frequency-effects flag. This module models
//! that geometry and derives the quantities the renderers need: the directional
//! speaker subset, the planar/dome classification, and a helper that lifts a
//! fixed bed layout onto a nominal sphere.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! The irregular-layout model follows the open literature on amplitude panning
//! (Pulkki 1997) and wave-field synthesis (Berkhout 1988).
//!
//! # Relationship
//!
//! Consumed by [`crate::array::vbap`], [`crate::array::allrad`],
//! [`crate::array::wfs`], and [`crate::array::beamforming`]. Reuses
//! [`crate::eif::ChannelLabel`] for speaker labels and
//! `prism_audio_object::bed::BedLayout` as a bootstrap source of regular
//! layouts.

use alloc::vec::Vec;

use bevy_math::{ops, Vec3};
use prism_audio_core::math::Sample;
use prism_audio_object::bed::BedLayout;

use crate::eif::ChannelLabel;

/// The default radius in metres used when lifting a direction-only layout onto
/// a physical sphere.
pub const DEFAULT_RADIUS_M: Sample = 2.0;

/// The tolerance (in the same units as positions) within which a layout counts
/// as planar.
pub const PLANAR_EPSILON: Sample = 1.0e-3;

/// One loudspeaker in an [`ArrayLayout`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ArraySpeaker {
    /// Output channel index this speaker is wired to.
    pub channel: usize,
    /// Physical position in metres in the listener-local frame.
    pub position: Vec3,
    /// Unit direction toward the speaker from the listener origin.
    pub direction: Vec3,
    /// Short label (for example `"FL"`, `"SUB"`).
    pub label: ChannelLabel,
    /// Whether this is a low-frequency-effects feed (never a panning target).
    pub is_lfe: bool,
}

impl ArraySpeaker {
    /// Builds a directional speaker at `position`; the direction is the
    /// normalised position (or forward if the position is at the origin).
    #[must_use]
    pub fn at(channel: usize, position: Vec3, label: &str) -> Self {
        Self {
            channel,
            position,
            direction: unit_or_forward(position),
            label: ChannelLabel::new(label),
            is_lfe: false,
        }
    }

    /// Builds a speaker from a unit `direction`, placing it at `radius` metres.
    #[must_use]
    pub fn from_direction(channel: usize, direction: Vec3, radius: Sample, label: &str) -> Self {
        let unit = unit_or_forward(direction);
        Self {
            channel,
            position: unit * radius,
            direction: unit,
            label: ChannelLabel::new(label),
            is_lfe: false,
        }
    }

    /// Builds a low-frequency-effects feed (no direction).
    #[must_use]
    pub fn lfe(channel: usize, label: &str) -> Self {
        Self {
            channel,
            position: Vec3::ZERO,
            direction: Vec3::ZERO,
            label: ChannelLabel::new(label),
            is_lfe: true,
        }
    }
}

/// Returns the normalised vector, or `-Z` (forward) if it is degenerate.
fn unit_or_forward(vector: Vec3) -> Vec3 {
    let normalized = vector.normalize_or_zero();
    if normalized == Vec3::ZERO {
        Vec3::new(0.0, 0.0, -1.0)
    } else {
        normalized
    }
}

/// An ordered, arbitrary loudspeaker array.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ArrayLayout {
    /// The speakers in output-channel order.
    pub speakers: Vec<ArraySpeaker>,
}

impl ArrayLayout {
    /// An empty layout.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a layout from an explicit speaker list.
    #[must_use]
    pub fn from_speakers(speakers: Vec<ArraySpeaker>) -> Self {
        Self { speakers }
    }

    /// Appends a speaker.
    pub fn push(&mut self, speaker: ArraySpeaker) {
        self.speakers.push(speaker);
    }

    /// The number of output channels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.speakers.len()
    }

    /// Whether the layout has no speakers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.speakers.is_empty()
    }

    /// The indices of the directional (non-LFE) speakers.
    #[must_use]
    pub fn directional_indices(&self) -> Vec<usize> {
        self.speakers
            .iter()
            .enumerate()
            .filter(|(_, speaker)| !speaker.is_lfe)
            .map(|(index, _)| index)
            .collect()
    }

    /// The unit directions of the directional speakers, in the same order as
    /// [`ArrayLayout::directional_indices`].
    #[must_use]
    pub fn directional_directions(&self) -> Vec<Vec3> {
        self.speakers
            .iter()
            .filter(|speaker| !speaker.is_lfe)
            .map(|speaker| speaker.direction)
            .collect()
    }

    /// The number of directional (non-LFE) speakers.
    #[must_use]
    pub fn directional_count(&self) -> usize {
        self.speakers.iter().filter(|s| !s.is_lfe).count()
    }

    /// The physical positions of every speaker, in channel order.
    #[must_use]
    pub fn positions(&self) -> Vec<Vec3> {
        self.speakers.iter().map(|speaker| speaker.position).collect()
    }

    /// Whether every directional speaker lies within [`PLANAR_EPSILON`] of the
    /// horizontal (`y = 0`) plane.
    #[must_use]
    pub fn is_planar(&self) -> bool {
        self.speakers
            .iter()
            .filter(|speaker| !speaker.is_lfe)
            .all(|speaker| ops::abs(speaker.direction.y) <= PLANAR_EPSILON)
    }

    /// Lifts a fixed [`BedLayout`] onto a sphere of [`DEFAULT_RADIUS_M`].
    ///
    /// Directional channels take their bed direction; the LFE channel (if any)
    /// becomes an [`ArraySpeaker::lfe`] feed so channel indices line up with
    /// the bed's own channel order.
    #[must_use]
    pub fn from_bed(bed: BedLayout) -> Self {
        let mut speakers = Vec::with_capacity(bed.channel_count());
        for (channel, bed_channel) in bed.channels().into_iter().enumerate() {
            if bed_channel.is_lfe {
                speakers.push(ArraySpeaker::lfe(channel, bed_channel.label));
            } else {
                speakers.push(ArraySpeaker::from_direction(
                    channel,
                    bed_channel.direction,
                    DEFAULT_RADIUS_M,
                    bed_channel.label,
                ));
            }
        }
        Self { speakers }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    const EPS: Sample = 1.0e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    #[test]
    fn from_bed_preserves_channel_count_and_lfe() {
        let layout = ArrayLayout::from_bed(BedLayout::Surround5_1_4);
        assert_eq!(layout.len(), BedLayout::Surround5_1_4.channel_count());
        assert_eq!(
            layout.directional_count(),
            BedLayout::Surround5_1_4.directional_count()
        );
        assert!(layout.speakers.iter().any(|speaker| speaker.is_lfe));
    }

    #[test]
    fn directional_directions_are_unit_length() {
        let layout = ArrayLayout::from_bed(BedLayout::Surround7_1_4);
        for direction in layout.directional_directions() {
            assert!(close(direction.length(), 1.0));
        }
    }

    #[test]
    fn stereo_is_planar_dome_is_not() {
        assert!(ArrayLayout::from_bed(BedLayout::Stereo).is_planar());
        assert!(!ArrayLayout::from_bed(BedLayout::Surround5_1_4).is_planar());
    }

    #[test]
    fn radius_scales_position() {
        let speaker = ArraySpeaker::from_direction(0, Vec3::new(0.0, 0.0, -1.0), 3.0, "C");
        assert!(close(speaker.position.length(), 3.0));
        assert!(close(speaker.direction.length(), 1.0));
    }

    #[test]
    fn degenerate_direction_falls_back_to_forward() {
        let speaker = ArraySpeaker::at(0, Vec3::ZERO, "X");
        assert!(close(speaker.direction.z, -1.0));
    }
}
