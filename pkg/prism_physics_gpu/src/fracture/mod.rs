//! `GPU`-accelerated fragment classification and aggregation for destruction.
//!
//! The convex Voronoi *carving* in [`prism_physics_core::fracture`] decides the
//! shape of each fragment on the `CPU`; this module supplies the mass-parallel
//! companions that bin a large point cloud (debris particles, surface samples,
//! or voxel centres) into those fragments on the `GPU` and then reduce each
//! fragment to a rigid-body seed.
//!
//! - [`GpuVoronoiAssign`] reports, for every query point, the owning Voronoi
//!   cell and the distance to the nearest cell wall.
//! - [`GpuFragmentAggregate`] reduces a classified point cloud to one
//!   rigid-body seed per fragment: mass, centre of mass, and inertia tensor.
//! - [`GpuFragmentBounds`] reduces a classified point cloud to one broad-phase
//!   proxy per fragment: an axis-aligned bounding box and a bounding sphere.
//!
//! Together they cover the classify-then-seed pipeline a real-time destruction
//! system runs at the hundred-thousand-to-million point scale.
//!
//! # Layout
//!
//! - [`config`] — the [`VoronoiAssignConfig`] tunable and the [`NO_CELL`]
//!   sentinel.
//! - [`cpu`] — the [`cpu_assign_cells`] golden twin and its [`CellAssignment`]
//!   output.
//! - [`gpu`] — the real-device [`GpuVoronoiAssign`] classifier.
//! - [`aggregate`] — the per-fragment [`GpuFragmentAggregate`] reducer, its
//!   [`cpu_aggregate_fragments`] twin, and the [`FragmentAggregate`] output.
//! - [`bounds`] — the per-fragment [`GpuFragmentBounds`] proxy builder, its
//!   [`cpu_bounds_fragments`] twin, and the [`FragmentBounds`] output.
//!
//! # Correctness model
//!
//! Each kernel is paired with a `CPU` twin running the identical arithmetic. The
//! owning cell index is integer-exact whenever the nearest site is unambiguous
//! (the argmin is taken over squared distances with a strict comparison, so both
//! sides keep the lowest index on a tie); the classifier clearance and the
//! aggregated centroid and inertia carry floating-point roots and divisions and
//! are therefore checked within a tight tolerance, while the aggregated integer
//! moments are bit-identical.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It reuses
//! [`prism_physics_core::fracture::Plane`] for the bisector cell walls;
//! nearest-site Voronoi membership and the rigid-body mass/centroid/inertia
//! formulas are standard, publicly documented results.

pub mod aggregate;
pub mod bounds;
pub mod config;
pub mod cpu;
pub mod gpu;

pub use aggregate::{
    cpu_aggregate_fragments, AggregateConfig, FragmentAggregate, GpuFragmentAggregate,
};
pub use bounds::{cpu_bounds_fragments, BoundsConfig, FragmentBounds, GpuFragmentBounds};
pub use config::{VoronoiAssignConfig, NO_CELL};
pub use cpu::{cpu_assign_cells, CellAssignment};
pub use gpu::GpuVoronoiAssign;
