//! Compatibility re-export for the scene-hosted crate.
//!
//! The implementation lives in [`prism_particle_render`], whose dependency
//! closure contains no engine-framework crate. This module deliberately contains no ECS,
//! plugin, or render-world adapter code.

pub use prism_particle_render::{
    ParticleExtractBatch, ParticleFabricError, ParticleFabricFrame, ParticleInstanceKey,
    ParticleInstanceSnapshot, PARTICLE_INSTANCE_PAYLOAD_ROLE,
};
