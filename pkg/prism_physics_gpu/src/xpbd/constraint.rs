//! The distance (stretch) constraint shared by the `CPU` golden and `GPU`
//! kernels.
//!
//! A [`DistanceConstraint`] holds two particle indices at a target *rest
//! length* with a compliance (inverse stiffness). Its constraint function is
//!
//! ```text
//! C = |p_a - p_b| - rest_length
//! ```
//!
//! with unit gradients `+n` on `a` and `-n` on `b`, so the `XPBD` denominator
//! reduces to `w_a + w_b + alpha_tilde`. This is the exact stretch constraint
//! implemented by [`prism_physics_core`](prism_physics_core::soft::constraint);
//! the type here adds only the flat, `std430`-compatible upload layout the
//! device kernel consumes.
//!
//! Provenance: canonical `XPBD` stretch constraint (Müller et al.). No Unreal
//! Engine source or derived code.

use bytemuck::{Pod, Zeroable};

/// A compliant distance constraint keeping two particles at a rest length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceConstraint {
    /// Index of the first coupled particle.
    pub a: u32,
    /// Index of the second coupled particle.
    pub b: u32,
    /// Target separation between the two particles, in metres.
    pub rest_length: f32,
    /// Compliance (inverse stiffness); `0` is perfectly rigid.
    pub compliance: f32,
}

impl DistanceConstraint {
    /// Creates a distance constraint between particles `a` and `b`.
    ///
    /// Negative rest lengths and compliances are clamped to `0` so a caller
    /// cannot construct a constraint that pulls particles through each other or
    /// applies negative stiffness.
    #[must_use]
    pub fn new(a: u32, b: u32, rest_length: f32, compliance: f32) -> DistanceConstraint {
        DistanceConstraint {
            a,
            b,
            rest_length: rest_length.max(0.0),
            compliance: compliance.max(0.0),
        }
    }

    /// Packs the constraint into its `std430` upload form.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuConstraint {
        GpuConstraint {
            a: self.a,
            b: self.b,
            rest_length: self.rest_length,
            compliance: self.compliance,
        }
    }
}

/// `std430`-compatible upload form of [`DistanceConstraint`] (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuConstraint {
    /// Index of the first coupled particle.
    pub a: u32,
    /// Index of the second coupled particle.
    pub b: u32,
    /// Target separation, in metres.
    pub rest_length: f32,
    /// Compliance (inverse stiffness).
    pub compliance: f32,
}
