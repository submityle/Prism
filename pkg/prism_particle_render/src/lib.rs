//! Framework-independent particle render data and Render Fabric lowering.
//!
//! The application/runtime side owns ECS components and simulation state. This
//! crate accepts an immutable Prism snapshot and emits typed per-frame records;
//! it does not install a plugin, access an ECS world, or depend on a renderer
//! framework.

mod adapter;

pub use adapter::{
    ParticleExtractBatch, ParticleFabricError, ParticleFabricFrame, ParticleInstanceKey,
    ParticleInstanceSnapshot, PARTICLE_INSTANCE_PAYLOAD_ROLE,
};
