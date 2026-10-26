//! Framework-independent particle render data and Render Fabric lowering.
//!
//! The application/runtime side owns ECS components and simulation state. This
//! crate accepts an immutable Prism snapshot and emits typed per-frame records;
//! it does not install a plugin, access an ECS world, or depend on a renderer
//! framework.

mod adapter;
mod fabric_gpu;
mod transport;

pub use adapter::{
    particle_resource_payload, particle_resource_template, ParticleExtractBatch,
    ParticleFabricError, ParticleFabricFrame, ParticleInstanceKey, ParticleInstanceSnapshot,
    PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE, PARTICLE_EMITTER_COUNT_PAYLOAD_ROLE,
    PARTICLE_EMITTER_POSITION_BOUNDS_PAYLOAD_ROLE, PARTICLE_INSTANCE_PAYLOAD_ROLE,
    PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
};
pub use fabric_gpu::{
    ParticleFabricGpuBuffer, ParticleFabricGpuDraw, ParticleFabricScenePublication,
};
pub use transport::{
    ParticleFabricBufferLease, ParticleFabricDeviceGeneration, ParticleFabricLeaseError,
    ParticleFabricLeaseId, ParticleFabricLeaseIssuer, ParticleFabricLeaseProvenance,
    ParticleFabricSceneTransport, ParticleFabricTransportError,
};
