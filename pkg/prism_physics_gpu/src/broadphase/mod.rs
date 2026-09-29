//! Uniform-grid spatial-hash broad phase (Teschner et al. 2003).
//!
//! The broad phase reduces the `O(n^2)` all-pairs collision test to the pairs
//! whose bounding spheres can actually touch. Each particle is hashed into an
//! infinite virtual grid of `cell_size` cubes via the classic three-prime hash;
//! a candidate pair is emitted when two particles fall in adjacent cells and
//! their spheres overlap. With `cell_size` at least the largest sphere
//! diameter, every genuine overlap lies in the `3x3x3` neighbourhood, so the
//! result is *exact* (no missed pairs), which the unit tests pin against an
//! independent brute-force reference.
//!
//! This module hosts the shared integer/float hashing math ([`hash`]) used by
//! both the [`cpu`] golden twin and the [`gpu`] `WGSL` kernel, guaranteeing the
//! two paths bucket every particle identically.
//!
//! Provenance: Teschner, Heidelberger, Müller, Pomeranets, Gross,
//! "Optimized Spatial Hashing for Collision Detection of Deformable Objects"
//! (VMV 2003). No Unreal Engine source or derived code.

mod config;
mod cpu;
mod gpu;
mod hash;
mod pair;
mod particle;

pub use config::{BroadphaseConfig, BroadphaseError};
pub use cpu::cpu_broadphase;
pub use gpu::GpuBroadphase;
pub use hash::{cell_coord, hash_cell};
pub use pair::CandidatePair;
pub use particle::Particle;
