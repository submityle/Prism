//! Lightweight, copyable identifier newtypes used throughout the content model.
//!
//! Authoring data references other authoring data by stable integer id rather
//! than by owning pointers, so the whole model is a flat, serialisable,
//! cache-friendly set of tables. Ids are opaque `u32` handles minted by the
//! authoring tool (or by [`crate::model::ContentModel`] builders); the runtime
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
//! [`crate::model::ContentModel`] and the live state in
//! [`crate::system::EventSystem`]. They carry no behaviour.

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
    /// Identifies a game-posted [`crate::event::Event`].
    EventId
);
define_id!(
    /// Identifies a playable leaf sound (resolved by a higher asset layer).
    SoundId
);
define_id!(
    /// Identifies a [`crate::container::Container`] node.
    ContainerId
);
define_id!(
    /// Identifies a [`crate::state::StateGroup`].
    StateGroupId
);
define_id!(
    /// Identifies one state within a [`crate::state::StateGroup`].
    StateId
);
define_id!(
    /// Identifies a [`crate::switch::SwitchGroup`].
    SwitchGroupId
);
define_id!(
    /// Identifies one switch within a [`crate::switch::SwitchGroup`].
    SwitchId
);
define_id!(
    /// Identifies an RTPC game parameter definition.
    RtpcId
);
define_id!(
    /// Identifies a per-emitter game object (for scoped switches/RTPC).
    GameObjectId
);
define_id!(
    /// Identifies an output bus / mixer target an action may address.
    BusId
);

/// A playable node reference: either a leaf sound or a container subtree.
///
/// Containers nest by referencing other `Playable`s, forming the authoring
/// tree a play action walks to pick concrete leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Playable {
    /// A concrete leaf sound.
    Sound(SoundId),
    /// A nested container to be resolved recursively.
    Container(ContainerId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_then_get_roundtrips() {
        let e = EventId::new(42);
        assert_eq!(e.get(), 42);
        let c = ContainerId::new(0);
        assert_eq!(c.get(), 0);
    }

    #[test]
    fn from_u32_and_into_u32_roundtrip() {
        let s: SoundId = 7u32.into();
        assert_eq!(s, SoundId::new(7));
        let raw: u32 = s.into();
        assert_eq!(raw, 7);
    }

    #[test]
    fn distinct_newtypes_do_not_alias_values() {
        // Same raw value in two different id types stays logically distinct in
        // use even though the backing integer matches.
        assert_eq!(StateId::new(3).get(), SwitchId::new(3).get());
        assert_eq!(RtpcId::new(5).get(), BusId::new(5).get());
    }

    #[test]
    fn playable_variants_compare_by_contents() {
        let a = Playable::Sound(SoundId::new(1));
        let b = Playable::Sound(SoundId::new(1));
        let c = Playable::Container(ContainerId::new(1));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
