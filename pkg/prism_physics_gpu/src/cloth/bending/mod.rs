//! `GPU` colour-batched cloth bending constraints.
//!
//! Cloth resists folding through a point-to-midpoint bending constraint: the
//! hinge particle `center` is pulled toward the midpoint of its two neighbours
//! `a`/`b` with a compliant `XPBD` projection. The sequential Gauss-Seidel
//! golden lives in [`prism_physics_core`] as `project_bending`; this module runs
//! the exact same projection on the `GPU` as a sequence of *graph-coloured*
//! batches so the hardware stays race-free without an intra-pass barrier.
//!
//! # Layout
//!
//! - [`coloring`] — the three-endpoint greedy colourer that partitions the
//!   constraints into independent sets (one race-free pass each).
//! - [`cpu`] — the [`cpu_cloth_bending`] golden twin (delegating the per-joint
//!   arithmetic to [`prism_physics_core`]'s `project_bending`) plus an
//!   independent brute-force anchor.
//! - [`gpu`] — the real-device [`GpuClothBending`] pipeline.
//!
//! # Correctness model
//!
//! Within one colour class the constraints share no particle, so a parallel
//! pass and a sequential sweep produce bit-identical writes; across colours the
//! device runs the classes in the same ascending order as the twin. The only
//! divergence is the per-projection fused-multiply-add and division/square-root
//! rounding, so parity is verified within a tight tolerance rather than
//! bit-for-bit — the same model the rest of the solver uses.
//!
//! # Provenance
//!
//! The compliant `XPBD` bending projection is the published Müller et al.
//! position-based-dynamics technique; greedy graph colouring of the constraint
//! conflict graph is standard batched-PBD practice. No Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};

pub mod coloring;
pub mod cpu;
pub mod gpu;

pub use coloring::{colour_bending, BendingColoring};
pub use cpu::cpu_cloth_bending;
pub use gpu::GpuClothBending;

/// One point-to-midpoint cloth bending constraint, over raw particle indices.
///
/// This is the `GPU` crate's public re-statement of
/// [`prism_physics_core`]'s `BendingConstraint`, which stores crate-private
/// particle handles that callers outside the engine cannot build. The field
/// layout is also the exact bytes uploaded to the device, so it is `Pod`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClothBendingConstraint {
    /// First neighbour particle index.
    pub a: u32,
    /// Hinge particle index (pulled toward the `a`/`b` midpoint).
    pub center: u32,
    /// Second neighbour particle index.
    pub b: u32,
    /// Rest centre-to-midpoint offset length.
    pub rest_offset: f32,
    /// `XPBD` compliance (`0` is a rigid hinge).
    pub compliance: f32,
}
