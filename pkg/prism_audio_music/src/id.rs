//! Lightweight, copyable identifier newtypes used throughout the music model.
//!
//! Authoring data references other authoring data by stable integer id rather
//! than by owning pointers, so the whole model is a flat, serialisable,
//! cache-friendly set of tables. Ids are opaque `u32` handles minted by the
//! authoring tool (or by [`crate::model::MusicModel`] builders); the planner
//! never interprets their numeric value beyond equality and table lookup.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. Opaque integer
//! handles are a universal data-modelling idiom. Pure data, no AI/ML.
//!
//! # Relationship
//!
//! These ids are the join keys between the registries in
//! [`crate::model::MusicModel`] and the live state in
//! [`crate::system::MusicSystem`]. They carry no behaviour.

/// Declares a transparent `u32` identifier newtype with the common helpers.
macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[cfg_attr(
            feature = "serialize",
            derive(serde::Serialize, serde::Deserialize)
        )]
        #[repr(transparent)]
        pub struct $name(pub u32);

        impl $name {
            /// Wraps a raw numeric handle.
            #[must_use]
            pub const fn new(raw: u32) -> Self {
                Self(raw)
            }

            /// Returns the raw numeric handle.
            #[must_use]
            pub const fn get(self) -> u32 {
                self.0
            }
        }

        impl From<u32> for $name {
            fn from(raw: u32) -> Self {
                Self(raw)
            }
        }

        impl From<$name> for u32 {
            fn from(id: $name) -> Self {
                id.0
            }
        }
    };
}

define_id!(
    /// Identifies a music [`crate::segment::Segment`].
    SegmentId
);
define_id!(
    /// Identifies a [`crate::playlist::Playlist`].
    PlaylistId
);
define_id!(
    /// Identifies a leaf audio asset a segment, layer, or stinger plays
    /// (resolved by a higher asset layer).
    SoundId
);
define_id!(
    /// Identifies a named [`crate::segment::Marker`] inside a segment.
    MarkerId
);
define_id!(
    /// Identifies a [`crate::stinger::Stinger`].
    StingerId
);
define_id!(
    /// Identifies one [`crate::layer::Layer`] of a vertical layer set.
    LayerId
);
define_id!(
    /// Identifies a [`crate::layer::LayerSet`] (a group of layers driven by one
    /// intensity value).
    LayerSetId
);
define_id!(
    /// Identifies a [`crate::clip_graph::Clip`] node in a clip graph.
    ClipId
);
define_id!(
    /// Identifies a [`crate::clip_graph::ClipGraph`].
    GraphId
);
define_id!(
    /// Identifies a gameplay branch value used by clip-graph trigger
    /// conditions (horizontal re-sequencing).
    BranchId
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_raw_handle() {
        let id = SegmentId::new(7);
        assert_eq!(id.get(), 7);
        assert_eq!(u32::from(id), 7);
        assert_eq!(SegmentId::from(7u32), id);
    }

    #[test]
    fn ids_are_ordered_for_btree_keys() {
        assert!(ClipId::new(1) < ClipId::new(2));
        assert_eq!(LayerId::new(3), LayerId::new(3));
    }
}
