//! The one-sided non-penetration contact constraint.
//!
//! A [`ContactConstraint`] couples two particles that the narrow phase found
//! overlapping and keeps them from interpenetrating. Its constraint function is
//! the *inequality*
//!
//! ```text
//! C = |p_a - p_b| - rest >= 0
//! ```
//!
//! where `rest` is the sum of the two particle radii. Unlike the bidirectional
//! [`DistanceConstraint`](crate::xpbd::DistanceConstraint) — which drives the
//! separation *to* the rest length from either side — a contact only ever pushes
//! the pair *apart*: when `C >= 0` the pair is already separated and the
//! projection is skipped, and the accumulated Lagrange multiplier is clamped to
//! be non-negative so the constraint can never pull the particles together
//! (a contact force cannot be attractive).
//!
//! The unit gradients are `+n` on `a` and `-n` on `b` (with `n` pointing from
//! `b` toward `a`), identical in magnitude to the stretch constraint, so the
//! `XPBD` denominator is again `w_a + w_b + alpha_tilde`. Reusing that shared
//! structure is what lets the contact solver borrow the proven colouring and
//! substep machinery from the distance solver unchanged.
//!
//! Provenance: canonical `XPBD` inequality (contact) constraint of Müller et
//! al. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};

use crate::xpbd::ColouredEdge;

/// A compliant one-sided constraint keeping two particles from overlapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactConstraint {
    /// Index of the first coupled particle.
    pub a: u32,
    /// Index of the second coupled particle.
    pub b: u32,
    /// Minimum allowed separation (the sum of the two radii), in metres.
    pub rest: f32,
    /// Compliance (inverse stiffness); `0` is a perfectly rigid contact.
    pub compliance: f32,
}

impl ContactConstraint {
    /// Creates a contact constraint between particles `a` and `b`.
    ///
    /// Negative rest separations and compliances are clamped to `0` so a caller
    /// cannot construct a contact that pulls particles together or applies
    /// negative stiffness.
    #[must_use]
    pub fn new(a: u32, b: u32, rest: f32, compliance: f32) -> ContactConstraint {
        ContactConstraint {
            a,
            b,
            rest: rest.max(0.0),
            compliance: compliance.max(0.0),
        }
    }

    /// Packs the constraint into its `std430` upload form.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuContactConstraint {
        GpuContactConstraint {
            a: self.a,
            b: self.b,
            rest: self.rest,
            compliance: self.compliance,
        }
    }
}

impl ColouredEdge for ContactConstraint {
    fn endpoints(&self) -> (u32, u32) {
        (self.a, self.b)
    }
}

/// `std430`-compatible upload form of [`ContactConstraint`] (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuContactConstraint {
    /// Index of the first coupled particle.
    pub a: u32,
    /// Index of the second coupled particle.
    pub b: u32,
    /// Minimum allowed separation, in metres.
    pub rest: f32,
    /// Compliance (inverse stiffness).
    pub compliance: f32,
}
