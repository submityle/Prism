//! Compatibility re-export for the scene-hosted crate.
//!
//! The implementation lives in [`prism_particle_render`], whose dependency
//! closure contains no engine-framework crate. This module deliberately contains no ECS,
//! plugin, or render-world adapter code.

pub use prism_particle_render::{
    ParticleExtractBatch, ParticleFabricError, ParticleFabricFrame, ParticleInstanceKey,
    ParticleInstanceSnapshot, PARTICLE_INSTANCE_PAYLOAD_ROLE,
};

#[cfg(feature = "legacy-bevy")]
mod fabric;
#[cfg(feature = "legacy-bevy")]
pub(crate) use fabric::append_particle_fabric_parts;
#[cfg(feature = "legacy-bevy")]
#[cfg(test)]
pub(crate) use fabric::prepare_particle_fabric;
#[cfg(feature = "legacy-bevy")]
pub use fabric::{
    install_particle_fabric, ParticleSceneFabricBuffer, ParticleSceneFabricBufferOwned,
    ParticleSceneFabricDispatch, ParticleSceneFabricDispatchOwned, ParticleSceneFabricError,
    ParticleSceneFabricHost, ParticleSceneFabricHostInput, ParticleSceneFabricInput,
    ParticleSceneFabricSlot,
};
