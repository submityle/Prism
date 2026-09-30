//! `GPU` linear bounding volume hierarchy (`LBVH`) construction.
//!
//! An `LBVH` is the standard GPU-built acceleration structure for broad-phase
//! culling, ray queries, and collision-pair generation over static or rebuilt
//! primitive sets. Every leaf is reduced to an axis-aligned box (see
//! [`config::Aabb`]); the builder quantises each centroid to a 30-bit Morton
//! code, stably sorts leaves by that code so spatially near primitives are
//! adjacent, threads Karras' binary radix tree over the sorted codes in one
//! parallel pass, and unions child boxes bottom-up into internal-node bounds.
//!
//! The build has four device stages, each mirrored by a [`cpu`] golden twin: a
//! Morton kernel writes each leaf's code and identity index, the sibling
//! [`GpuRadixSort`](crate::radix::GpuRadixSort) stably orders leaves by code
//! entirely on device, a tree kernel derives every internal node's range,
//! split, and children independently, and a bounds kernel climbs from the
//! leaves to fill internal-node boxes with an atomic per-node arrival counter.
//! Because the pipeline is an integer permutation plus exact `min`/`max`
//! reductions, a passing real-device parity test against [`cpu_build_lbvh`] is
//! bit-for-bit evidence of a faithful port.
//!
//! # Provenance
//!
//! The Morton-code sort plus binary radix tree is Karras, "Maximizing
//! Parallelism in the Construction of BVHs, Octrees, and k-d Trees" (High
//! Performance Graphics 2012); the sibling radix sort follows Blelloch (1990)
//! and Satish, Harris, Garland (2009). This module contains no Unreal Engine
//! source or derived code.

pub mod config;
pub mod cpu;
pub mod gpu;
pub mod layout;
pub mod morton;
pub mod query;
pub mod query_gpu;
pub mod ray;
pub mod resident;

pub use config::{Aabb, SceneBounds};
pub use cpu::{cpu_build_lbvh, Lbvh, NO_PARENT};
pub use gpu::GpuLbvh;
pub use query::{cpu_bvh_pairs, BvhQueryError};
pub use query_gpu::GpuBvhQuery;
pub use ray::{cpu_bvh_raycast_any, cpu_bvh_raycast_closest, Ray, RayHit};
pub use resident::GpuResidentLbvh;
