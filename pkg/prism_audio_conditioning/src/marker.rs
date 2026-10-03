//! Labeled transient / beat / bar marker timelines.
//!
//! A [`MarkerTimeline`] is an ordered list of [`Marker`]s. Markers can be
//! inserted one at a time (kept sorted by frame, ties broken by kind) or built
//! in bulk from transient onsets plus a beat grid, with a bar marker placed on
//! every `beats_per_bar`-th beat.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the marker-timeline leg of design section 51; it consumes the
//! onsets of [`crate::transient`] and the beat grid of [`crate::tempo`].

use alloc::vec::Vec;

/// The semantic class of a marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum MarkerKind {
    /// A detected transient / onset.
    Transient,
    /// A beat-grid position.
    Beat,
    /// A bar (downbeat) position.
    Bar,
}

/// A single marker at an audio-frame position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Marker {
    /// Audio-frame position of the marker.
    pub frame: usize,
    /// Semantic class of the marker.
    pub kind: MarkerKind,
}

impl Marker {
    /// Creates a marker.
    #[must_use]
    pub const fn new(frame: usize, kind: MarkerKind) -> Self {
        Self { frame, kind }
    }
}

/// An ordered collection of markers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MarkerTimeline {
    /// Markers sorted by `(frame, kind)`.
    markers: Vec<Marker>,
}

impl MarkerTimeline {
    /// Creates an empty timeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            markers: Vec::new(),
        }
    }

    /// Inserts a marker, keeping the timeline sorted by `(frame, kind)`.
    pub fn insert(&mut self, marker: Marker) {
        let key = (marker.frame, marker.kind);
        let index = self
            .markers
            .partition_point(|m| (m.frame, m.kind) < key);
        self.markers.insert(index, marker);
    }

    /// Borrows the ordered markers.
    #[must_use]
    pub fn markers(&self) -> &[Marker] {
        &self.markers
    }

    /// Returns the number of markers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.markers.len()
    }

    /// Returns `true` when the timeline holds no markers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }

    /// Builds a timeline from transient onsets and a beat grid.
    ///
    /// Each onset becomes a [`MarkerKind::Transient`]; each beat becomes a
    /// [`MarkerKind::Beat`], except every `beats_per_bar`-th beat (starting at
    /// the first) which becomes a [`MarkerKind::Bar`]. A `beats_per_bar` of
    /// zero disables bar promotion.
    #[must_use]
    pub fn build(onsets: &[usize], beat_grid: &[usize], beats_per_bar: u32) -> Self {
        let mut timeline = Self::new();
        for &onset in onsets {
            timeline.insert(Marker::new(onset, MarkerKind::Transient));
        }
        for (index, &beat) in beat_grid.iter().enumerate() {
            let is_bar = beats_per_bar > 0 && (index as u32).is_multiple_of(beats_per_bar);
            let kind = if is_bar {
                MarkerKind::Bar
            } else {
                MarkerKind::Beat
            };
            timeline.insert(Marker::new(beat, kind));
        }
        timeline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_keeps_sorted_order() {
        let mut timeline = MarkerTimeline::new();
        timeline.insert(Marker::new(100, MarkerKind::Beat));
        timeline.insert(Marker::new(10, MarkerKind::Transient));
        timeline.insert(Marker::new(50, MarkerKind::Bar));
        let frames: Vec<usize> =
            timeline.markers().iter().map(|m| m.frame).collect();
        assert_eq!(frames, alloc::vec![10, 50, 100]);
    }

    #[test]
    fn build_promotes_every_fourth_beat_to_bar() {
        let beats = [0usize, 100, 200, 300, 400, 500, 600, 700];
        let timeline = MarkerTimeline::build(&[], &beats, 4);
        let bars: Vec<usize> = timeline
            .markers()
            .iter()
            .filter(|m| m.kind == MarkerKind::Bar)
            .map(|m| m.frame)
            .collect();
        assert_eq!(bars, alloc::vec![0, 400]);
    }

    #[test]
    fn build_merges_onsets_and_beats() {
        let onsets = [25usize, 125];
        let beats = [0usize, 100, 200];
        let timeline = MarkerTimeline::build(&onsets, &beats, 0);
        assert_eq!(timeline.len(), 5);
        // All beats are plain beats when beats_per_bar is zero.
        assert!(timeline
            .markers()
            .iter()
            .all(|m| m.kind != MarkerKind::Bar));
    }
}
