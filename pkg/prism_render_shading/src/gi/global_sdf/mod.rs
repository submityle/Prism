//! Global signed distance field: merge per-object distance fields into a sparse
//! brick grid and cone/sphere trace it for ambient occlusion and GI gather.
//!
//! This is the backend-neutral, GPU-free reference for Lumen's *global distance
//! field* half of the GI pipeline. It is distinct from the per-mesh
//! [`crate::gi::scene::mesh_sdf`] field: here many placed objects are unioned
//! into one world-space structure and traced as a whole.
//!
//! # Conventions
//! * [`brick_grid`] — a sparse, world-space grid of fixed-size bricks
//!   ([`BrickGrid`]), each an `(brick_dim + 1)^3` cube of signed-distance
//!   corners with `world <-> voxel <-> brick` mapping, clamp-to-edge trilinear
//!   sampling, central-difference gradients, and a conservative empty-space
//!   distance so marches step safely across unallocated gaps.
//! * [`merge`] — closed-form primitives ([`SdfPrimitive`]) placed by a rigid
//!   transform plus uniform scale ([`SdfObject`]), unioned by the per-point
//!   minimum (`d(p) = min_i scale_i * sdf_i(T_i^{-1} p)`) via [`merge_distance`]
//!   and baked into a [`BrickGrid`] with [`bake_merged`].
//! * [`cone_trace`] — [`sphere_march`] for first-hit ray marching, plus cosine
//!   hemisphere cone tracing for soft ambient occlusion ([`cone_trace_ao`]) and
//!   a front-to-back one-bounce irradiance gather ([`cone_gather_gi`]).
//! * The merged field is the per-sample minimum of transformed object SDFs
//!   (union); tracing uses sphere marching with a cone-tapered occupancy
//!   estimate. All transcendental math routes through [`bevy_math::ops`].
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe)
//!   with defensive clamps so no path yields `NaN`.

pub mod brick_grid;
pub mod cone_trace;
pub mod merge;

pub use brick_grid::{BrickGrid, SparseBrick, DEFAULT_BRICK_DIM, FAR_DISTANCE};
pub use cone_trace::{cone_gather_gi, cone_trace_ao, sphere_march, ConeConfig, MarchHit};
pub use merge::{bake_merged, box_sdf, merge_distance, sphere_sdf, SdfObject, SdfPrimitive};
