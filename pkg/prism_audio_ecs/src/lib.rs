//! Prism ECS front-end for the lock-free [`prism_audio_rt`] runtime.
//!
//! The crate owns the control-plane resources and systems that translate ECS
//! components into runtime commands. The real-time runtime remains parked in a
//! non-send [`AudioRuntimeHost`] for device-
//! less and offline execution; a production device callback can take the same
//! runtime ownership model without changing ECS-facing components.
//!
//! Voice lifecycle commands are serialized through the ordered audio system and
//! mirrored locally by [`VoiceMirror`]. This keeps entity handles deterministic
//! while preserving the runtime's bounded, non-blocking command-ring contract.
#![forbid(unsafe_code)]

extern crate alloc;

pub mod assets;
pub mod budget;
pub mod client_resource;
pub mod commands;
pub mod emitter;
pub mod listener;
pub mod master_gain;
pub mod playback;
pub mod player;
pub mod player_systems;
pub mod plugin;
pub mod scene_adapter;
pub mod source_route;
pub mod systems;
pub mod telemetry;
pub mod voice_registry;
pub mod volume;

pub use assets::{
    AudioAssetLoadError, AudioAssetStore, AudioSourceRouteError, ResidentAudioSource,
};
pub use budget::PhysicalVoiceBudget;
pub use client_resource::{AudioClient, AudioRuntimeHost};
pub use emitter::AudioEmitter;
pub use listener::AudioListener;
pub use master_gain::MasterGain;
pub use playback::{PlaybackMode, PlaybackSettings};
pub use player::{AudioPlaybackState, AudioPlayer};
pub use player_systems::{apply_player_disposition, sync_audio_players};
pub use plugin::AudioRuntimePlugin;
pub use scene_adapter::materialize_scene_audio_sources;
pub use source_route::{
    sync_audio_source_routes, AudioGraphRouteError, AudioGraphRouteState, AudioOutputLayout,
};
pub use systems::AudioSystems;
pub use telemetry::AudioTelemetry;
pub use voice_registry::{LiveVoices, PendingStops, RemovedAudioEmitters, VoiceMirror};
pub use volume::Volume;

/// Common public audio ECS types.
pub mod prelude {
    pub use crate::assets::{
        AudioAssetLoadError, AudioAssetStore, AudioSourceRouteError, ResidentAudioSource,
    };
    pub use crate::budget::PhysicalVoiceBudget;
    pub use crate::client_resource::{AudioClient, AudioRuntimeHost};
    pub use crate::emitter::AudioEmitter;
    pub use crate::listener::AudioListener;
    pub use crate::master_gain::MasterGain;
    pub use crate::playback::{PlaybackMode, PlaybackSettings};
    pub use crate::player::{AudioPlaybackState, AudioPlayer};
    pub use crate::player_systems::{apply_player_disposition, sync_audio_players};
    pub use crate::plugin::AudioRuntimePlugin;
    pub use crate::scene_adapter::materialize_scene_audio_sources;
    pub use crate::source_route::{
        sync_audio_source_routes, AudioGraphRouteError, AudioGraphRouteState, AudioOutputLayout,
    };
    pub use crate::systems::AudioSystems;
    pub use crate::telemetry::AudioTelemetry;
    pub use crate::voice_registry::{LiveVoices, PendingStops, RemovedAudioEmitters, VoiceMirror};
    pub use crate::volume::Volume;
}
