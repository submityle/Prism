//! The [`AudioListener`] component: a marker whose
//! [`GlobalTransform`](prism_transform::GlobalTransform) defines the
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
//! from [`prism_transform`].

use prism_ecs::prelude::Component;

/// Marks the entity whose
/// [`GlobalTransform`](prism_transform::GlobalTransform) is the
/// active listener pose.
///
/// At most one listener is used; when several exist the lowest complete Entity
/// identifier wins, independent of archetype iteration order. Without a listener,
/// emitters fall back to their
/// unattenuated base importance.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AudioListener;
