//! Particles: the shared point-mass primitive of the unified soft-body kernel.
//!
//! Cloth, rope/hair, and volumetric soft bodies are all represented as a set of
//! point-mass [`ParticleHandle`]s stored in a [`ParticleStorage`], coupled by
//! XPBD constraints. This module owns that particle representation.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**.

pub mod handle;
pub mod storage;

pub use handle::ParticleHandle;
pub use storage::ParticleStorage;
