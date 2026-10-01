//! `GPU` cloth tearing: per-edge break-flag evaluation for over-stretched edges.
//!
//! Tearing drops a distance edge from the solve once its tensile strain exceeds
//! a break threshold, so an over-stretched garment rips instead of stretching
//! without bound. The authoritative per-edge decision lives in
//! [`prism_physics_core`] as
//! [`tear_flag`](prism_physics_core::tear_flag); this module runs that exact
//! predicate over every edge in parallel on the device and reports, per edge,
//! whether it would tear.
//!
//! # Flags, not compaction
//!
//! Removing an edge changes the constraint graph (and any downstream colouring),
//! which is a host-side bookkeeping step, not an embarrassingly parallel one.
//! This kernel therefore computes only the per-edge *break flag* (`1` = tears)
//! plus an atomic tally of torn edges — the exact `tear_flags` /
//! `apply_tearing` return the host consumes before it compacts the graph. The
//! flag pass is embarrassingly parallel: each edge reads a shared, read-only
//! position snapshot and writes only *its own* flag, so there is no colouring
//! and no inter-thread barrier.
//!
//! # Layout
//!
//! - [`ClothTearEdge`] — the `GPU` crate's public, `Pod` re-statement of a
//!   distance edge (two particle indices plus a rest length), since
//!   [`prism_physics_core`]'s `DistanceConstraint` stores crate-private handles.
//! - [`cpu`] — the [`cpu_cloth_tearing`] golden twin, delegating the break
//!   decision to [`prism_physics_core::tear_flag`].
//! - [`gpu`] — the real-device [`GpuClothTearing`] pipeline.
//!
//! # Provenance
//!
//! Removing a constraint whose strain exceeds a threshold is a standard,
//! publicly documented position-based-dynamics technique. No Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};

pub mod cpu;
pub mod gpu;

pub use cpu::cpu_cloth_tearing;
pub use gpu::GpuClothTearing;

/// One tearable distance edge, over raw particle indices.
///
/// This is the `GPU` crate's public re-statement of a
/// [`prism_physics_core`] `DistanceConstraint` for the tearing pass, which only
/// needs the two endpoints and the (immutable) rest length against which strain
/// is measured. The field layout is also the exact bytes uploaded to the
/// device, so it is `Pod`.
///
/// All three fields are 4-byte scalars, so the `std430` struct alignment is `4`
/// and the array stride is a packed `12` bytes with no padding — matching this
/// `#[repr(C)]` layout and the `WGSL` `Edge` struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClothTearEdge {
    /// First coupled particle index.
    pub a: u32,
    /// Second coupled particle index.
    pub b: u32,
    /// Rest (reference) length of the edge, in metres. Tensile strain is
    /// measured against this; a degenerate (`<= EPS_REST`) rest length never
    /// tears.
    pub rest_length: f32,
}

impl ClothTearEdge {
    /// Creates a tearable edge over `a`-`b` with the given `rest_length`.
    ///
    /// Mirrors [`prism_physics_core`]'s `DistanceConstraint::new` floor:
    /// `rest_length` is clamped to be non-negative. A degenerate (`<= EPS_REST`)
    /// rest length makes the edge inert in the pass (it never tears).
    #[must_use]
    pub fn new(a: u32, b: u32, rest_length: f32) -> Self {
        ClothTearEdge {
            a,
            b,
            rest_length: rest_length.max(0.0),
        }
    }
}
