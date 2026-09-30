//! `GPU` per-fragment rigid-body seed aggregation for destruction.
//!
//! Once the Voronoi classifier in [`super`] has binned a debris point cloud
//! into fragment cells, a destruction system spawns one rigid body per fragment
//! and needs each fragment's mass, centre of mass, and inertia tensor to seed
//! the rigid-body solver. [`GpuFragmentAggregate`] computes all three in two
//! mass-parallel passes: an atomic scatter of the mass moments followed by a
//! per-fragment finalise.
//!
//! # Layout
//!
//! - [`config`] — the [`AggregateConfig`] tunable and the shared fixed-point
//!   quantisation helpers.
//! - [`cpu`] — the [`cpu_aggregate_fragments`] golden twin and its
//!   [`FragmentAggregate`] output.
//! - [`gpu`] — the real-device [`GpuFragmentAggregate`] aggregator.
//! - [`layout`] — the shared bind-group layout helpers.
//!
//! # Correctness model
//!
//! The kernels are paired with the [`cpu_aggregate_fragments`] twin running the
//! identical fixed-point accumulation. Integer atomic addition is exact and
//! order independent, so the accumulated mass and moments are bit-identical
//! across devices; the finalised centroid and inertia carry floating-point
//! division and are therefore checked within a tight tolerance.
//!
//! # Provenance
//!
//! Rigid-body mass/centroid/inertia formulas are textbook mechanics; fixed-point
//! atomic accumulation is a standard `GPU` reduction. No Unreal Engine source or
//! derived code.

pub mod config;
pub mod cpu;
pub mod gpu;
pub mod layout;

pub use config::AggregateConfig;
pub use cpu::{cpu_aggregate_fragments, FragmentAggregate};
pub use gpu::GpuFragmentAggregate;
