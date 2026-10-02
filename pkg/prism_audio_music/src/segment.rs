//! **Segments**: the atomic unit of interactive music.
//!
//! A [`Segment`] is one musically coherent chunk (an intro, a loop, a fill)
//! with its own tempo and time signature. Following the Wwise/FMOD music
//! vocabulary (modelled from scratch), a segment carries:
//!
//! - a **pre-roll** lead-in before its entry cue (pickup notes, anticipation),
//! - a musical **body** between the entry cue and the exit cue (the part that
//!   loops seamlessly and that transitions align to),
//! - a **post-roll** tail after the exit cue (reverb/ring-out that overlaps the
//!   next segment),
//! - named **markers** at sample offsets for marker-aligned transitions.
//!
//! All offsets are in samples at the project sample rate, so timing is exact
//! and platform independent. The segment is pure data; it owns no runtime
//! state and performs no DSP.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The entry/exit
//! cue plus pre/post-roll model is reconstructed from first principles as plain
//! data. No AI/ML.
//!
//! # Relationship
//!
//! Segments are stored in [`crate::model::MusicModel`], ordered by
//! [`crate::playlist::Playlist`], referenced by [`crate::clip_graph::Clip`]
//! nodes, and scheduled by [`crate::system::MusicSystem`], whose
//! [`prism_audio_core::scheduler::NamedClock`] is anchored at a segment start
//! using the segment's [`Segment::tempo_bpm`] and [`Segment::signature`].

use alloc::vec::Vec;

use prism_audio_core::time::TimeSignature;

use crate::id::{MarkerId, SegmentId, SoundId};

/// A named cue point inside a segment, at a sample offset from segment start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Marker {
    /// Stable id a marker-aligned transition names.
    pub id: MarkerId,
    /// Offset in samples from the start of the segment (pre-roll included).
    pub offset: u64,
}

impl Marker {
    /// Builds a marker at `offset` samples from the segment start.
    #[must_use]
    pub fn new(id: MarkerId, offset: u64) -> Self {
        Self { id, offset }
    }
}

/// One musically coherent chunk of interactive music.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Segment {
    /// Stable id this segment is referenced by.
    pub id: SegmentId,
    /// Leaf audio asset the segment plays.
    pub sound: SoundId,
    /// Tempo in beats per minute used to anchor the quantization grid.
    pub tempo_bpm: f32,
    /// Time signature used to compute bar length from beat length.
    pub signature: TimeSignature,
    /// Lead-in samples before the entry cue (pickup / anticipation).
    pub pre_roll: u64,
    /// Samples of musical body between the entry cue and the exit cue.
    pub body: u64,
    /// Tail samples after the exit cue (ring-out that overlaps the next
    /// segment).
    pub post_roll: u64,
    /// Named cue points in authored order, offsets from the segment start.
    pub markers: Vec<Marker>,
}

impl Segment {
    /// Builds a segment with the given tempo/meter and margins and no markers.
    ///
    /// `tempo_bpm` is clamped to the musical range `[1, 1000]` so a corrupt
    /// authoring value can never produce a zero or negative beat length.
    #[must_use]
    pub fn new(
        id: SegmentId,
        sound: SoundId,
        tempo_bpm: f32,
        signature: TimeSignature,
        pre_roll: u64,
        body: u64,
        post_roll: u64,
    ) -> Self {
        Self {
            id,
            sound,
            tempo_bpm: tempo_bpm.clamp(1.0, 1000.0),
            signature,
            pre_roll,
            body,
            post_roll,
            markers: Vec::new(),
        }
    }

    /// Appends a marker, returning `self` for builder-style chaining.
    #[must_use]
    pub fn with_marker(mut self, marker: Marker) -> Self {
        self.markers.push(marker);
        self
    }

    /// Sample offset of the entry cue from the segment start (the pre-roll
    /// length).
    #[inline]
    #[must_use]
    pub fn entry_cue(&self) -> u64 {
        self.pre_roll
    }

    /// Sample offset of the exit cue from the segment start.
    ///
    /// This is the seamless hand-off point: the next segment's entry cue is
    /// scheduled here so the musical body loops/chains without a gap.
    #[inline]
    #[must_use]
    pub fn exit_cue(&self) -> u64 {
        self.pre_roll + self.body
    }

    /// Total length of the segment in samples (pre-roll + body + post-roll).
    #[inline]
    #[must_use]
    pub fn total(&self) -> u64 {
        self.pre_roll + self.body + self.post_roll
    }

    /// Looks up a marker's offset by id.
    #[must_use]
    pub fn marker_offset(&self, id: MarkerId) -> Option<u64> {
        self.markers.iter().find(|m| m.id == id).map(|m| m.offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg() -> Segment {
        Segment::new(
            SegmentId::new(1),
            SoundId::new(10),
            120.0,
            TimeSignature::default(),
            4_800,  // pre-roll
            96_000, // body (one 4/4 bar at 120 BPM @ 48k is 96000)
            2_400,  // post-roll
        )
    }

    #[test]
    fn cue_arithmetic() {
        let s = seg();
        assert_eq!(s.entry_cue(), 4_800);
        assert_eq!(s.exit_cue(), 4_800 + 96_000);
        assert_eq!(s.total(), 4_800 + 96_000 + 2_400);
    }

    #[test]
    fn tempo_is_clamped() {
        let s = Segment::new(
            SegmentId::new(2),
            SoundId::new(1),
            -5.0,
            TimeSignature::default(),
            0,
            100,
            0,
        );
        assert!(s.tempo_bpm >= 1.0);
    }

    #[test]
    fn markers_look_up_by_id() {
        let s = seg()
            .with_marker(Marker::new(MarkerId::new(1), 24_000))
            .with_marker(Marker::new(MarkerId::new(2), 48_000));
        assert_eq!(s.marker_offset(MarkerId::new(2)), Some(48_000));
        assert_eq!(s.marker_offset(MarkerId::new(9)), None);
    }
}
