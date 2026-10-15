//! The [`PlaybackMode`] enum and [`PlaybackSettings`] component: a declarative,
//! component-shaped description of *how* an [`AudioPlayer`](crate::player::AudioPlayer)
//! should be scheduled and what becomes of its entity on an explicit stop.
//!
//! This is the ergonomic authoring facade over the lower-level
//! [`AudioEmitter`](crate::emitter::AudioEmitter): instead of hand-choosing a
//! [`VirtualBehavior`] and an attenuation band, a caller spawns an
//! [`AudioPlayer`](crate::player::AudioPlayer) with these settings and the
//! [`sync_audio_players`](crate::player_systems::sync_audio_players) system
//! derives the emitter. The field set is deliberately limited to knobs the
//! current voice runtime can honor. The runtime has no per-source decoder,
//! duration, or natural-completion event; these modes select virtualization
//! policy and explicit-stop disposition, not automatic sample looping/EOS.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Read by [`crate::player_systems`] alongside
//! [`crate::player::AudioPlayer`]; its [`Volume`] folds into the base
//! [`Importance`](prism_audio_core::voice::Importance) and its [`PlaybackMode`]
//! chooses the [`VirtualBehavior`] on the derived
//! [`AudioEmitter`](crate::emitter::AudioEmitter).

use prism_audio_core::voice::VirtualBehavior;
use prism_ecs::prelude::Component;

use crate::volume::Volume;

/// Default reference distance for a spatial player, in world units: inside this
/// radius the player sounds at its full authored volume.
const DEFAULT_REFERENCE_DISTANCE: f32 = 1.0;

/// Default maximum distance for a spatial player, in world units: at and beyond
/// this radius the player fades to silence.
const DEFAULT_MAX_DISTANCE: f32 = 100.0;

/// Explicit-stop disposition and virtual-voice playhead policy.
///
/// There is no natural-completion event in the underlying runtime. Source
/// decoding and sample looping are not provided by this component; the caller
/// requests disposition through [`AudioPlayer::stop`](crate::AudioPlayer::stop).
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlaybackMode {
    /// Restart the playhead on virtual revival; retain components on stop.
    #[default]
    Once,
    /// Advance the virtual playhead continuously; retain components on stop.
    Loop,
    /// Despawn the whole entity on an explicit stop request.
    Despawn,
    /// On explicit stop, remove this crate's playback components from the entity,
    /// leaving the rest of the entity intact.
    Remove,
}

impl PlaybackMode {
    /// Whether continuous virtual playhead advancement is requested.
    #[must_use]
    #[inline]
    pub const fn loops(self) -> bool {
        matches!(self, PlaybackMode::Loop)
    }

    /// Whether an explicit stop should despawn the entity.
    #[must_use]
    #[inline]
    pub const fn despawns_entity(self) -> bool {
        matches!(self, PlaybackMode::Despawn)
    }

    /// Whether an explicit stop should remove playback components, leaving
    /// the entity otherwise intact.
    #[must_use]
    #[inline]
    pub const fn removes_components(self) -> bool {
        matches!(self, PlaybackMode::Remove)
    }

    /// The culled-voice behavior this mode implies: looping sources keep
    /// advancing silently so they stay in sync, while one-shots restart from
    /// the beginning when revived.
    #[must_use]
    #[inline]
    pub const fn virtual_behavior(self) -> VirtualBehavior {
        if self.loops() {
            VirtualBehavior::ContinueVirtual
        } else {
            VirtualBehavior::RestartFromBeginning
        }
    }
}

/// Declarative playback configuration attached beside an
/// [`AudioPlayer`](crate::player::AudioPlayer).
///
/// Build one from a mode preset ([`PlaybackSettings::ONCE`],
/// [`PlaybackSettings::LOOP`], [`PlaybackSettings::DESPAWN`],
/// [`PlaybackSettings::REMOVE`]) and refine it with the chaining setters.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PlaybackSettings {
    /// Virtual playhead behavior and explicit-stop disposition of the entity.
    pub mode: PlaybackMode,
    /// Authored loudness, folded into the derived emitter's base importance.
    pub volume: Volume,
    /// When `true`, drive importance to silence without tearing the voice
    /// down. This is not a sample/playhead pause; the runtime still advances it.
    pub paused: bool,
    /// When `true`, the voice is driven to silence but otherwise kept alive and
    /// scheduled, matching a mute that preserves playback position.
    pub muted: bool,
    /// When `true`, the player attenuates with listener distance using
    /// [`PlaybackSettings::reference_distance`] and
    /// [`PlaybackSettings::max_distance`]; when `false`, it plays at full
    /// volume regardless of position (a non-positional / "2D" source).
    pub spatial: bool,
    /// Distance, in world units, within which a spatial player sounds at full
    /// volume. Ignored when [`PlaybackSettings::spatial`] is `false`.
    pub reference_distance: f32,
    /// Distance, in world units, at and beyond which a spatial player fades to
    /// silence. Ignored when [`PlaybackSettings::spatial`] is `false`.
    pub max_distance: f32,
}

impl PlaybackSettings {
    /// Restart virtual voices on revival and retain components on explicit stop.
    pub const ONCE: Self = Self::with_mode(PlaybackMode::Once);
    /// Advance virtual voices continuously and retain components on explicit stop.
    pub const LOOP: Self = Self::with_mode(PlaybackMode::Loop);
    /// Despawn the entity on explicit stop.
    pub const DESPAWN: Self = Self::with_mode(PlaybackMode::Despawn);
    /// Remove playback components on explicit stop.
    pub const REMOVE: Self = Self::with_mode(PlaybackMode::Remove);

    /// Builds settings in `mode` at unity volume, unpaused, unmuted, and
    /// spatial with the default distance band.
    #[must_use]
    #[inline]
    pub const fn with_mode(mode: PlaybackMode) -> Self {
        Self {
            mode,
            volume: Volume::UNITY,
            paused: false,
            muted: false,
            spatial: true,
            reference_distance: DEFAULT_REFERENCE_DISTANCE,
            max_distance: DEFAULT_MAX_DISTANCE,
        }
    }

    /// Sets the authored volume.
    #[must_use]
    #[inline]
    pub const fn with_volume(mut self, volume: Volume) -> Self {
        self.volume = volume;
        self
    }

    /// Marks the player as initially paused.
    #[must_use]
    #[inline]
    pub const fn paused(mut self) -> Self {
        self.paused = true;
        self
    }

    /// Marks the player as initially muted.
    #[must_use]
    #[inline]
    pub const fn muted(mut self) -> Self {
        self.muted = true;
        self
    }

    /// Makes the player non-positional: it plays at full volume regardless of
    /// listener distance.
    #[must_use]
    #[inline]
    pub const fn non_spatial(mut self) -> Self {
        self.spatial = false;
        self
    }

    /// Sets the spatial attenuation band and marks the player spatial.
    #[must_use]
    #[inline]
    pub const fn with_distance(mut self, reference_distance: f32, max_distance: f32) -> Self {
        self.spatial = true;
        self.reference_distance = reference_distance;
        self.max_distance = max_distance;
        self
    }

    /// The volume actually applied this frame: silent while paused or muted,
    /// otherwise the authored [`PlaybackSettings::volume`].
    #[must_use]
    #[inline]
    pub fn effective_volume(&self) -> Volume {
        if self.paused || self.muted {
            Volume::SILENT
        } else {
            self.volume
        }
    }
}

impl Default for PlaybackSettings {
    /// Equivalent to [`PlaybackSettings::ONCE`].
    #[inline]
    fn default() -> Self {
        Self::ONCE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_presets_select_the_right_mode() {
        assert_eq!(PlaybackSettings::ONCE.mode, PlaybackMode::Once);
        assert_eq!(PlaybackSettings::LOOP.mode, PlaybackMode::Loop);
        assert_eq!(PlaybackSettings::DESPAWN.mode, PlaybackMode::Despawn);
        assert_eq!(PlaybackSettings::REMOVE.mode, PlaybackMode::Remove);
    }

    #[test]
    fn loop_mode_keeps_voice_advancing() {
        assert_eq!(
            PlaybackMode::Loop.virtual_behavior(),
            VirtualBehavior::ContinueVirtual
        );
        assert!(PlaybackMode::Loop.loops());
    }

    #[test]
    fn one_shot_modes_restart_on_revival() {
        for mode in [
            PlaybackMode::Once,
            PlaybackMode::Despawn,
            PlaybackMode::Remove,
        ] {
            assert_eq!(
                mode.virtual_behavior(),
                VirtualBehavior::RestartFromBeginning
            );
            assert!(!mode.loops());
        }
    }

    #[test]
    fn disposition_predicates_are_exclusive() {
        assert!(PlaybackMode::Despawn.despawns_entity());
        assert!(!PlaybackMode::Despawn.removes_components());
        assert!(PlaybackMode::Remove.removes_components());
        assert!(!PlaybackMode::Remove.despawns_entity());
        assert!(!PlaybackMode::Once.despawns_entity());
        assert!(!PlaybackMode::Once.removes_components());
    }

    #[test]
    fn paused_or_muted_forces_silence() {
        assert!(PlaybackSettings::ONCE
            .paused()
            .effective_volume()
            .is_silent());
        assert!(PlaybackSettings::ONCE
            .muted()
            .effective_volume()
            .is_silent());
        assert!(!PlaybackSettings::ONCE.effective_volume().is_silent());
    }

    #[test]
    fn builders_set_fields() {
        let settings = PlaybackSettings::LOOP
            .with_volume(Volume::Decibels(-6.0))
            .with_distance(2.0, 50.0);
        assert_eq!(settings.mode, PlaybackMode::Loop);
        assert_eq!(settings.volume, Volume::Decibels(-6.0));
        assert!(settings.spatial);
        assert_eq!(settings.reference_distance, 2.0);
        assert_eq!(settings.max_distance, 50.0);
    }

    #[test]
    fn non_spatial_clears_spatial_flag() {
        assert!(!PlaybackSettings::ONCE.non_spatial().spatial);
    }
}
