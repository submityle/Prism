//! Deferred decal projection and blending (CPU golden reference).
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; never `f32::exp()` style.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Defensive clamping everywhere: guard divide-by-zero, enforce
//!   non-negativity and finiteness, and fall back gracefully on degenerate
//!   input so no `NaN`/`inf` ever escapes.
//!
//! Implements box/clip-space decal projection, angle-based edge fade,
//! G-buffer attribute blending, and clustered froxel binning.


pub mod blend;
pub mod cluster;
pub mod projection;

pub use blend::{blend_albedo, blend_normal, blend_scalar, DecalBlendMode};
pub use cluster::{Aabb, ClusterGrid};
pub use projection::{DecalHit, DecalProjector};
