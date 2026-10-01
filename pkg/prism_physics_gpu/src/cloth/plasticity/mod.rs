//! `GPU` cloth plasticity: permanent rest-length creep for over-stretched edges.
//!
//! Plasticity lets a distance edge stretched past a yield strain creep its rest
//! length toward the current length, capturing permanent wrinkles and sag while
//! retaining a bounded residual elastic strain. The authoritative per-edge
//! scalar kernel lives in [`prism_physics_core`] as
//! [`plastic_rest_length`](prism_physics_core::plastic_rest_length); this module
//! runs that exact kernel over every edge in parallel on the device.
//!
//! # Why no colouring
//!
//! Unlike the distance/strain-limit projections, plasticity never writes a
//! particle position — each edge updates only *its own* `rest_length` from a
//! read-only position snapshot. The edges therefore share no mutable state, so
//! the pass is embarrassingly parallel: one thread owns one edge with no
//! colouring and no inter-thread barrier. A single atomic counter tallies how
//! many edges crept, matching the sequential golden's `modified` return.
//!
//! # Layout
//!
//! - [`ClothPlasticEdge`] — the `GPU` crate's public, `Pod` re-statement of a
//!   distance edge (two particle indices plus a rest length), since
//!   [`prism_physics_core`]'s `DistanceConstraint` stores crate-private handles.
//! - [`cpu`] — the [`cpu_cloth_plasticity`] golden twin, delegating the creep
//!   arithmetic to [`prism_physics_core::plastic_rest_length`].
//! - [`gpu`] — the real-device [`GpuClothPlasticity`] pipeline.
//!
//! # Provenance
//!
//! Rest-length creep past a yield strain is a standard, publicly documented
//! plastic-set model for position-based cloth. No Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};

pub mod cpu;
pub mod gpu;

pub use cpu::cpu_cloth_plasticity;
pub use gpu::GpuClothPlasticity;

/// One plastic distance edge, over raw particle indices.
///
/// This is the `GPU` crate's public re-statement of a
/// [`prism_physics_core`] `DistanceConstraint` for the plasticity pass, which
/// only needs the two endpoints and the mutable rest length. The field layout
/// is also the exact bytes uploaded to the device, so it is `Pod`.
///
/// All three fields are 4-byte scalars, so the `std430` struct alignment is `4`
/// and the array stride is a packed `12` bytes with no padding — matching this
/// `#[repr(C)]` layout and the `WGSL` `Edge` struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClothPlasticEdge {
    /// First coupled particle index.
    pub a: u32,
    /// Second coupled particle index.
    pub b: u32,
    /// Current rest (reference) length of the edge, in metres. This is the value
    /// the pass creeps; the solve returns the updated length per edge.
    pub rest_length: f32,
}

impl ClothPlasticEdge {
    /// Creates a plastic edge over `a`-`b` with the given `rest_length`.
    ///
    /// Mirrors [`prism_physics_core`]'s `DistanceConstraint::new` floor:
    /// `rest_length` is clamped to be non-negative. A degenerate (`<= EPS_REST`)
    /// rest length makes the edge inert in the pass.
    #[must_use]
    pub fn new(a: u32, b: u32, rest_length: f32) -> Self {
        ClothPlasticEdge {
            a,
            b,
            rest_length: rest_length.max(0.0),
        }
    }
}
