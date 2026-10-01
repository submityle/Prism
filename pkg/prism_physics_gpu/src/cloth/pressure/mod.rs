//! `GPU` closed-mesh cloth pressure (volume preservation / inflation).
//!
//! A pressure constraint keeps a *closed* triangle shell at a target enclosed
//! volume `target = overpressure * rest_volume`, inflating a garment into a
//! balloon, a down jacket, or an air bladder. Unlike the per-edge or per-pair
//! constraints in the sibling modules, pressure is a *single global* constraint
//! coupling every vertex of the shell through the divergence-theorem volume
//! functional, so there is no graph colouring: one iteration is one coordinated
//! sweep over the whole mesh with a single accumulated Lagrange multiplier.
//!
//! The sequential golden lives in [`prism_physics_core`] as `project_pressure`;
//! this module runs the exact same compliant-`XPBD` projection on the `GPU`.
//! Because a vertex gradient is the sum over every incident triangle corner, a
//! race-free device pass cannot scatter without atomics and instead *gathers*
//! through the host-built [`VertexTriangleAdjacency`] (see [`adjacency`]).
//!
//! # Layout
//!
//! - [`adjacency`] — the host-side `CSR` vertex→triangle-corner table that turns
//!   the per-triangle gradient scatter into a race-free per-vertex gather.
//! - [`cpu`] — the [`cpu_cloth_pressure`] golden twin, delegating each iteration
//!   to [`prism_physics_core`]'s `project_pressure`, anchored in its tests
//!   against an independent brute-force signed-volume reference.
//! - [`gpu`] — the real-device [`GpuClothPressure`] multi-pass pipeline.
//!
//! # Provenance
//!
//! The signed-volume-via-divergence-theorem pressure constraint, its compliant
//! `XPBD` projection, and the `CSR` scatter→gather reformulation are standard,
//! publicly documented techniques. No Unreal Engine source or derived code.

pub mod adjacency;
pub mod cpu;
pub mod gpu;

pub use adjacency::{
    build_vertex_triangle_adjacency, pack_corner, unpack_corner, VertexTriangleAdjacency,
};
pub use cpu::cpu_cloth_pressure;
pub use gpu::GpuClothPressure;
