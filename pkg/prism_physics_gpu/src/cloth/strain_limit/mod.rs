//! `GPU` colour-batched cloth strain limiting (biphasic length clamp).
//!
//! After the compliant distance/bending sweeps approach the rest length they
//! can still overshoot under fast motion, letting cloth stretch "like rubber".
//! A strain limiter is a *geometric* post-projection clamp that guarantees no
//! structural edge exceeds `rest_length * max_scale` (and, optionally, never
//! compresses below `rest_length * min_scale`). The sequential golden lives in
//! [`prism_physics_core`] as `project_strain_limit`; this module runs the exact
//! same clamp on the `GPU` as a sequence of *graph-coloured* batches so the
//! hardware stays race-free without an intra-pass barrier.
//!
//! # Layout
//!
//! - [`coloring`] — the two-endpoint greedy colourer that partitions the edges
//!   into independent sets (one race-free pass each).
//! - [`cpu`] — the [`cpu_cloth_strain_limit`] golden twin (delegating the
//!   per-edge arithmetic to [`prism_physics_core`]'s `project_strain_limit`)
//!   plus an independent brute-force anchor.
//! - [`gpu`] — the real-device [`GpuClothStrainLimit`] pipeline.
//!
//! # Correctness model
//!
//! Within one colour class the edges share no particle, so a parallel pass and
//! a sequential sweep produce identical writes; across colours the device runs
//! the classes in the same ascending order as the twin. The clamp is stateless
//! (a pure projection with no Lagrange multiplier), so there is no accumulator
//! to carry. The only divergence is the per-projection fused-multiply-add and
//! division/square-root rounding, so parity is verified within a tight
//! tolerance rather than bit-for-bit — the same model the rest of the solver
//! uses.
//!
//! # Provenance
//!
//! Biphasic strain limiting is a standard, publicly documented cloth technique
//! (Provot 1995; Thomaszewski et al. 2009); greedy graph colouring of the
//! constraint conflict graph is standard batched-PBD practice. No Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};

pub mod coloring;
pub mod cpu;
pub mod gpu;

pub use coloring::{colour_strain_limit, StrainLimitColoring};
pub use cpu::cpu_cloth_strain_limit;
pub use gpu::GpuClothStrainLimit;

/// One biphasic strain-limiting edge, over raw particle indices.
///
/// This is the `GPU` crate's public re-statement of
/// [`prism_physics_core`]'s `StrainLimitConstraint`, which stores crate-private
/// particle handles that callers outside the engine cannot build. The field
/// layout is also the exact bytes uploaded to the device, so it is `Pod`.
///
/// All five fields are 4-byte scalars, so the `std430` struct alignment is `4`
/// and the array stride is a packed `20` bytes with no padding — matching this
/// `#[repr(C)]` layout exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClothStrainLimitConstraint {
    /// First coupled particle index.
    pub a: u32,
    /// Second coupled particle index.
    pub b: u32,
    /// Rest (reference) length of the edge, in metres.
    pub rest_length: f32,
    /// Maximum allowed length as a multiple of `rest_length` (`>= 1`).
    pub max_scale: f32,
    /// Minimum allowed length as a multiple of `rest_length`; `0` disables the
    /// compression clamp.
    pub min_scale: f32,
}

impl ClothStrainLimitConstraint {
    /// Creates a strain limiter over edge `a`-`b` with the given `rest_length`.
    ///
    /// Mirrors [`prism_physics_core`]'s `StrainLimitConstraint::new` clamps:
    /// `rest_length` is floored at `0`, `max_scale` at `1`, and `min_scale` is
    /// clamped into `[0, 1]`. Pass `min_scale = 0` to disable the compression
    /// clamp.
    #[must_use]
    pub fn new(a: u32, b: u32, rest_length: f32, max_scale: f32, min_scale: f32) -> Self {
        ClothStrainLimitConstraint {
            a,
            b,
            rest_length: rest_length.max(0.0),
            max_scale: max_scale.max(1.0),
            min_scale: min_scale.clamp(0.0, 1.0),
        }
    }

    /// Convenience constructor for a max-stretch-only limiter expressed as a
    /// fractional strain `limit` (e.g. `0.1` for a 10% stretch cap), matching
    /// the render-side authoring convention `max_scale = 1 + limit`.
    #[must_use]
    pub fn from_stretch_limit(a: u32, b: u32, rest_length: f32, limit: f32) -> Self {
        ClothStrainLimitConstraint::new(a, b, rest_length, 1.0 + limit.max(0.0), 0.0)
    }
}
