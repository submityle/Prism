//! The [`AudioListener`] component: a marker whose
//! [`GlobalTransform`](bevy_transform::components::GlobalTransform) defines the
//! listener pose used for distance-based importance.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Read by the spawn/importance systems to locate the listener; its pose comes
//! from [`bevy_transform`].

use bevy_ecs::component::Component;

/// Marks the entity whose
/// [`GlobalTransform`](bevy_transform::components::GlobalTransform) is the
/// active listener pose.
///
/// At most one listener is used; when several exist the first encountered in
/// archetype order wins. When no listener exists, emitters fall back to their
/// unattenuated base importance.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AudioListener;
