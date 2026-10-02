//! **Actions**: the atomic verbs an [`crate::event::Event`] performs when
//! posted, plus the flattened [`ResolvedAction`] stream the runtime consumes.
//!
//! An [`Action`] is pure authoring data: "play this playable", "set this
//! state", "nudge this bus". Posting an event walks its actions, mutates the
//! live game-sync state held by [`crate::system::EventSystem`], and resolves
//! every container tree into a flat list of [`ResolvedAction`]s that name
//! concrete leaf sounds and parameter changes. The content layer deliberately
//! stops at *what* to play; *how* a sound becomes a voice is the job of the
//! lower runtime (`prism_audio_rt`) and voice/graph layers.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The verb set is
//! a from-scratch, data-only model of the universal "event posts actions"
//! idiom. No AI/ML.
//!
//! # Relationship
//!
//! [`Action`]s are owned by [`crate::event::Event`] and interpreted by
//! [`crate::system::EventSystem::post_event`], which turns [`Action::Play`]
//! into [`ResolvedAction::PlaySound`] by walking [`crate::container`] trees,
//! and turns [`Action::SetRtpc`] into [`ResolvedAction::SetParameter`] via the
//! [`crate::rtpc::RtpcRegistry`] bindings.

use prism_audio_core::math::Sample;

use crate::id::{
    BusId, GameObjectId, Playable, RtpcId, SoundId, StateGroupId, StateId, SwitchGroupId,
    SwitchId,
};
use crate::parameter::ParameterSetting;

/// A single authored verb executed when an event is posted.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Action {
    /// Start the given playable (leaf sound or container subtree).
    Play(Playable),
    /// Stop every sound reachable from the given playable.
    Stop(Playable),
    /// Stop every sound the posting object currently plays.
    StopAll,
    /// Set the active state of a global state group.
    SetState {
        /// The state group to change.
        group: StateGroupId,
        /// The state to activate.
        state: StateId,
    },
    /// Set the active switch of a group, scoped to the posting game object.
    SetSwitch {
        /// The switch group to change.
        group: SwitchGroupId,
        /// The switch to activate.
        switch: SwitchId,
    },
    /// Set a game parameter (RTPC) value, scoped to the posting game object.
    SetRtpc {
        /// The parameter to change.
        rtpc: RtpcId,
        /// The new raw value (clamped to the definition range on apply).
        value: Sample,
    },
    /// Override an output bus volume in decibels.
    SetBusVolumeDb {
        /// The target bus.
        bus: BusId,
        /// The new bus volume in decibels.
        volume_db: Sample,
    },
}

/// A flattened, runtime-ready instruction emitted while posting an event.
///
/// Every container selection and RTPC binding has already been resolved; the
/// variants name concrete [`SoundId`]s, [`crate::parameter::ParameterSetting`]s,
/// and buses, so a runtime can act on them without any further authoring-model
/// knowledge.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ResolvedAction {
    /// Play one concrete leaf sound at the given relative gain.
    PlaySound {
        /// Owning game object (for scoped stops and parameter routing).
        object: GameObjectId,
        /// The concrete leaf sound to spawn.
        sound: SoundId,
        /// Relative gain in decibels accumulated down the container tree.
        gain_db: Sample,
    },
    /// Stop one concrete leaf sound on the given object.
    StopSound {
        /// Owning game object.
        object: GameObjectId,
        /// The concrete leaf sound to stop.
        sound: SoundId,
    },
    /// Stop everything the given object is playing.
    StopAll {
        /// Owning game object.
        object: GameObjectId,
    },
    /// Apply a resolved parameter change to the object's live voices.
    SetParameter {
        /// Owning game object.
        object: GameObjectId,
        /// The resolved target/value pair.
        setting: ParameterSetting,
    },
    /// Apply a bus volume override.
    SetBusVolumeDb {
        /// The target bus.
        bus: BusId,
        /// The new bus volume in decibels.
        volume_db: Sample,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_variants_are_copy_and_comparable() {
        let a = Action::Play(Playable::Sound(SoundId::new(1)));
        let b = a; // Copy
        assert_eq!(a, b);
        let c = Action::SetState { group: StateGroupId::new(1), state: StateId::new(2) };
        assert_ne!(a, c);
    }

    #[test]
    fn resolved_actions_carry_object_and_payload() {
        let setting = ParameterSetting::new(crate::parameter::ParameterTarget::VolumeDb, -6.0);
        let r = ResolvedAction::SetParameter { object: GameObjectId::new(7), setting };
        match r {
            ResolvedAction::SetParameter { object, setting } => {
                assert_eq!(object, GameObjectId::new(7));
                assert_eq!(setting.target, crate::parameter::ParameterTarget::VolumeDb);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn bus_volume_action_roundtrips() {
        let a = Action::SetBusVolumeDb { bus: BusId::new(3), volume_db: -3.0 };
        match a {
            Action::SetBusVolumeDb { bus, .. } => assert_eq!(bus, BusId::new(3)),
            _ => panic!("wrong variant"),
        }
    }
}
