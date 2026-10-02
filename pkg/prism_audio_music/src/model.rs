//! **Music model**: the immutable registry of all authored music content.
//!
//! The model is a flat set of id-keyed tables -- segments, playlists, layer
//! sets, stingers, and clip graphs -- so it is cheap to clone, serialise, and
//! share. It owns no runtime state: the live playlist cursor, intensity, active
//! layers, graph position, clock, and RNG all live in
//! [`crate::system::MusicSystem`], which reads this model to resolve music
//! requests. Keeping authored data and live state apart lets one model drive
//! many independent, deterministic music systems.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is an
//! original, data-only registry built on ordered maps. No AI/ML.
//!
//! # Relationship
//!
//! [`MusicModel`] aggregates [`crate::segment::Segment`],
//! [`crate::playlist::Playlist`], [`crate::layer::LayerSet`],
//! [`crate::stinger::Stinger`], and [`crate::clip_graph::ClipGraph`].
//! [`crate::system::MusicSystem`] looks every one of them up by id while
//! resolving a request into [`crate::action::MusicAction`]s.

use alloc::collections::BTreeMap;

use crate::clip_graph::ClipGraph;
use crate::id::{GraphId, LayerSetId, PlaylistId, SegmentId, StingerId};
use crate::layer::LayerSet;
use crate::playlist::Playlist;
use crate::segment::Segment;
use crate::stinger::Stinger;

/// Immutable, id-keyed registry of all authored interactive music content.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MusicModel {
    /// Segments keyed by id.
    segments: BTreeMap<SegmentId, Segment>,
    /// Playlists keyed by id.
    playlists: BTreeMap<PlaylistId, Playlist>,
    /// Vertical layer sets keyed by id.
    layer_sets: BTreeMap<LayerSetId, LayerSet>,
    /// Stingers keyed by id.
    stingers: BTreeMap<StingerId, Stinger>,
    /// Clip graphs keyed by id.
    graphs: BTreeMap<GraphId, ClipGraph>,
}

impl MusicModel {
    /// Builds an empty model.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers (or replaces) a segment, keyed by its id.
    pub fn add_segment(&mut self, segment: Segment) {
        self.segments.insert(segment.id, segment);
    }

    /// Registers (or replaces) a playlist, keyed by its id.
    pub fn add_playlist(&mut self, playlist: Playlist) {
        self.playlists.insert(playlist.id, playlist);
    }

    /// Registers (or replaces) a layer set, keyed by its id.
    pub fn add_layer_set(&mut self, layer_set: LayerSet) {
        self.layer_sets.insert(layer_set.id, layer_set);
    }

    /// Registers (or replaces) a stinger, keyed by its id.
    pub fn add_stinger(&mut self, stinger: Stinger) {
        self.stingers.insert(stinger.id, stinger);
    }

    /// Registers (or replaces) a clip graph, keyed by its id.
    pub fn add_graph(&mut self, graph: ClipGraph) {
        self.graphs.insert(graph.id, graph);
    }

    /// Looks up a segment by id.
    #[must_use]
    pub fn segment(&self, id: SegmentId) -> Option<&Segment> {
        self.segments.get(&id)
    }

    /// Looks up a playlist by id.
    #[must_use]
    pub fn playlist(&self, id: PlaylistId) -> Option<&Playlist> {
        self.playlists.get(&id)
    }

    /// Looks up a layer set by id.
    #[must_use]
    pub fn layer_set(&self, id: LayerSetId) -> Option<&LayerSet> {
        self.layer_sets.get(&id)
    }

    /// Looks up a stinger by id.
    #[must_use]
    pub fn stinger(&self, id: StingerId) -> Option<&Stinger> {
        self.stingers.get(&id)
    }

    /// Iterates every registered layer set in ascending id order.
    ///
    /// Used by [`crate::system::MusicSystem::set_intensity`] to apply every
    /// set's hysteresis bands deterministically.
    pub fn layer_sets(&self) -> alloc::collections::btree_map::Values<'_, LayerSetId, LayerSet> {
        self.layer_sets.values()
    }


    /// Looks up a clip graph by id.
    #[must_use]
    pub fn graph(&self, id: GraphId) -> Option<&ClipGraph> {
        self.graphs.get(&id)
    }

    /// Returns the number of registered segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns the number of registered playlists.
    #[must_use]
    pub fn playlist_count(&self) -> usize {
        self.playlists.len()
    }

    /// Returns the number of registered layer sets.
    #[must_use]
    pub fn layer_set_count(&self) -> usize {
        self.layer_sets.len()
    }

    /// Returns the number of registered stingers.
    #[must_use]
    pub fn stinger_count(&self) -> usize {
        self.stingers.len()
    }

    /// Returns the number of registered clip graphs.
    #[must_use]
    pub fn graph_count(&self) -> usize {
        self.graphs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip_graph::{Clip, ClipGraph};
    use crate::id::{ClipId, LayerId, SoundId};
    use crate::layer::Layer;
    use crate::playlist::{PlaylistItem, PlaylistMode};
    use crate::transition::TransitionType;
    use prism_audio_core::time::TimeSignature;

    fn segment(id: u32) -> Segment {
        Segment::new(
            SegmentId::new(id),
            SoundId::new(id + 100),
            120.0,
            TimeSignature::default(),
            0,
            96_000,
            0,
        )
    }

    #[test]
    fn add_and_lookup_every_table() {
        let mut model = MusicModel::new();
        model.add_segment(segment(1));
        model.add_playlist(
            Playlist::new(PlaylistId::new(1), PlaylistMode::Loop)
                .with_item(PlaylistItem::once(SegmentId::new(1))),
        );
        model.add_layer_set(
            LayerSet::new(LayerSetId::new(1))
                .with_layer(Layer::new(LayerId::new(1), SoundId::new(9), 0.5, 0.4)),
        );
        model.add_stinger(Stinger::new(
            StingerId::new(1),
            SoundId::new(8),
            TransitionType::NextBar,
        ));
        model.add_graph(
            ClipGraph::new(GraphId::new(1), ClipId::new(1))
                .with_clip(Clip::new(ClipId::new(1), SegmentId::new(1))),
        );

        assert_eq!(model.segment_count(), 1);
        assert_eq!(model.playlist_count(), 1);
        assert_eq!(model.layer_set_count(), 1);
        assert_eq!(model.stinger_count(), 1);
        assert_eq!(model.graph_count(), 1);
        assert!(model.segment(SegmentId::new(1)).is_some());
        assert!(model.playlist(PlaylistId::new(1)).is_some());
        assert!(model.layer_set(LayerSetId::new(1)).is_some());
        assert!(model.stinger(StingerId::new(1)).is_some());
        assert!(model.graph(GraphId::new(1)).is_some());
    }

    #[test]
    fn add_replaces_existing_entry_by_id() {
        let mut model = MusicModel::new();
        model.add_segment(segment(1));
        let mut replacement = segment(1);
        replacement.body = 48_000;
        model.add_segment(replacement);
        assert_eq!(model.segment_count(), 1);
        assert_eq!(model.segment(SegmentId::new(1)).map(|s| s.body), Some(48_000));
    }

    #[test]
    fn empty_model_has_zero_counts() {
        let model = MusicModel::new();
        assert_eq!(model.segment_count(), 0);
        assert!(model.segment(SegmentId::new(1)).is_none());
        assert!(model.graph(GraphId::new(1)).is_none());
    }
}
