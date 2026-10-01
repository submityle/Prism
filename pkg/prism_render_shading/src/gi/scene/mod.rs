//! Scene-level signed-distance structures for GI (CPU golden).
//!
//! A backend-neutral, GPU-free reference for the distance-field half of the GI
//! pipeline, mirroring Unreal Engine's *Mesh Distance Fields* and *Distance
//! Field Ambient Occlusion*:
//!
//! * [`mesh_sdf`] — a uniform voxel lattice of signed distances over a
//!   world-space box ([`MeshSdf`]), with world/grid mapping, trilinear
//!   sampling, central-difference gradients, sphere tracing, and analytic
//!   primitive bakes ([`sphere_sdf`], [`box_sdf`]) for ground-truth tests.
//! * [`dfao`] — cosine-weighted hemispherical cone tracing of that field
//!   ([`cone_trace_ao`]) to produce distance-field ambient occlusion.
//!
//! Every item is a deterministic pure function with unit tests; its output is
//! the numerical reference the WESL/GPU twin passes must reproduce.  The voxel
//! layout (row-major `x` fastest), the clamp-to-edge sampling, and the
//! `[0, 1]` visibility convention are all chosen to match a 3D-texture GPU
//! twin addressed with a clamped sampler.

pub mod dfao;
pub mod mesh_sdf;

pub use dfao::{cone_trace_ao, AoConfig};
pub use mesh_sdf::{box_sdf, sphere_sdf, MeshSdf, SphereTraceHit};
