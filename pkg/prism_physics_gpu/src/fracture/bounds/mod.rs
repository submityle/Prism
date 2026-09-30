//! `GPU` per-fragment broad-phase bounds for destruction.
//!
//! Once the Voronoi classifier in [`super`] has binned a debris point cloud
//! into fragment cells, a destruction system needs each fragment's broad-phase
//! proxy — an axis-aligned bounding box and a bounding sphere — to seed the
//! broad phase before the fragment's rigid body is even spawned.
//! [`GpuFragmentBounds`] computes both in five mass-parallel passes: prime the
//! extrema, scatter the box, finalise the box centre, scatter the squared
//! radius, and finalise the radius.
//!
//! # Layout
//!
//! - [`config`] — the [`BoundsConfig`] tunable and the shared fixed-point
//!   quantisation helpers.
//! - [`cpu`] — the [`cpu_bounds_fragments`] golden twin and its
//!   [`FragmentBounds`] output.
//! - [`gpu`] — the real-device [`GpuFragmentBounds`] builder.
//! - [`layout`] — the shared bind-group layout helpers.
//!
//! # Correctness model
//!
//! The kernels are paired with the [`cpu_bounds_fragments`] twin running the
//! identical fixed-point reduction. Integer extrema are exact and order
//! independent, so the de-quantised box is bit-identical across devices; the
//! finalised sphere radius carries a square root and is therefore checked
//! within a tight tolerance.
//!
//! # Provenance
//!
//! Axis-aligned extrema and a box-centred bounding sphere are elementary
//! geometry; fixed-point atomic reduction is a standard `GPU` technique. No
//! Unreal Engine source or derived code.

pub mod config;
pub mod cpu;
pub mod gpu;
pub mod layout;

pub use config::BoundsConfig;
pub use cpu::{cpu_bounds_fragments, FragmentBounds};
pub use gpu::GpuFragmentBounds;
