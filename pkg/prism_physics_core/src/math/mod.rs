//! Math primitives shared across the physics core.
//!
//! This module fixes the engine's scalar precision ([`scalar::Real`]) and
//! provides the [`transform::Isometry`] rigid-body transform used to place
//! bodies and colliders in world space.

pub mod scalar;
pub mod transform;
