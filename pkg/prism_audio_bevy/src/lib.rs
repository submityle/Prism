//! Compatibility facade for Prism's native ECS audio bridge.
//!
//! The implementation and its single ownership boundary live in
//! [`prism_audio_ecs`]. This package name remains available to existing
//! callers while the audio control plane uses [`prism_app`], [`prism_ecs`],
//! [`prism_math`], and [`prism_transform`] instead of Bevy runtime crates.
//! Re-exporting the native crate keeps the public component, resource, system,
//! and plugin paths stable without maintaining a second audio implementation.
#![forbid(unsafe_code)]

pub use prism_audio_ecs::*;

// Keep the old public module paths available while the implementation moves
// behind the native ECS crate. This is a path-level compatibility bridge, not
// a second owner of any audio state or systems.
pub use prism_audio_ecs::{
    budget, client_resource, commands, emitter, listener, master_gain, playback, player,
    player_systems, plugin, systems, telemetry, voice_registry, volume,
};
