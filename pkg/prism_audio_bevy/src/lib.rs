//! Bevy ECS front-end for Prism's next-generation audio engine.
//!
//! This crate is the client (control-plane) half of the lock-free ECS
//! integration layer described in the engine design (section 21). It owns an
//! [`AudioRuntimeClient`](prism_audio_rt::AudioRuntimeClient) and translates
//! Bevy world state into [`AudioCommand`](prism_audio_rt::AudioCommand)s pushed
//! onto the command ring, then surfaces the audio thread's
//! [`TelemetryFrame`](prism_audio_rt::TelemetryFrame)s back as the
//! [`AudioTelemetry`] ECS resource.
//!
//! # Threading and ownership model
//!
//! Construction of a runtime yields three cooperating halves: the real-time
//! [`AudioRuntime`](prism_audio_rt::AudioRuntime) (audio-callback thread), the
//! clonable [`AudioRuntimeClient`](prism_audio_rt::AudioRuntimeClient) (any ECS
//! or task thread), and the [`Collector`](prism_audio_rt::Collector) (parked on
//! a task thread to drop retired resources). This crate is strictly the client
//! side: its systems only ever push commands and read telemetry, never render.
//!
//! Because this crate has no audio-device backend, the audio-thread half is
//! parked in the [`AudioRuntimeHost`] non-`Send` resource. In production a
//! device backend (for example a `cpal` callback) would take ownership of the
//! [`AudioRuntime`](prism_audio_rt::AudioRuntime) and drive it from the
//! real-time thread; the parked host exists so headless, offline, and test
//! setups can still advance the runtime deterministically off the ECS thread.
//!
//! # Handle tracking
//!
//! The runtime mints [`VoiceHandle`](prism_audio_core::voice::VoiceHandle)s on
//! the audio thread, and the current `prism_audio_rt` API does not return them
//! across the ring. This crate therefore keeps a deterministic client-side
//! mirror of the runtime's voice pool in [`VoiceMirror`]. A command mutates the
//! mirror only after it has been accepted by the ring, in the exact order it
//! was enqueued, so the mirror reproduces every allocation, release, and steal
//! the runtime performs and yields the identical handle. This exactness relies
//! on this front-end being the sole producer of voice-lifecycle commands.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Bridges [`bevy_ecs`] world state onto `prism_audio_rt`'s command ring and
//! telemetry ring, using real [`bevy_transform`] poses for listener/emitter
//! placement and [`bevy_math`] deterministic math for distance attenuation.
#![forbid(unsafe_code)]

extern crate alloc;

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
pub mod systems;
pub mod telemetry;
pub mod voice_registry;
pub mod volume;

pub use budget::PhysicalVoiceBudget;
pub use client_resource::{AudioClient, AudioRuntimeHost};
pub use emitter::AudioEmitter;
pub use listener::AudioListener;
pub use master_gain::MasterGain;
pub use playback::{PlaybackMode, PlaybackSettings};
pub use player::AudioPlayer;
pub use player_systems::{apply_player_disposition, sync_audio_players};
pub use plugin::AudioRuntimePlugin;
pub use systems::AudioSystems;
pub use telemetry::AudioTelemetry;
pub use voice_registry::{LiveVoices, PendingStops, VoiceMirror};
pub use volume::Volume;

/// Commonly used items, re-exported for convenient glob import.
pub mod prelude {
    pub use crate::budget::PhysicalVoiceBudget;
    pub use crate::client_resource::{AudioClient, AudioRuntimeHost};
    pub use crate::emitter::AudioEmitter;
    pub use crate::listener::AudioListener;
    pub use crate::master_gain::MasterGain;
    pub use crate::playback::{PlaybackMode, PlaybackSettings};
    pub use crate::player::AudioPlayer;
    pub use crate::plugin::AudioRuntimePlugin;
    pub use crate::systems::AudioSystems;
    pub use crate::telemetry::AudioTelemetry;
    pub use crate::voice_registry::{LiveVoices, PendingStops, VoiceMirror};
    pub use crate::volume::Volume;
}
