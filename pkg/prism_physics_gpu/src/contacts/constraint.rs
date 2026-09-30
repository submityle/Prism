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
//! # Coulomb friction
//!
//! Beyond non-penetration a contact also carries a two-coefficient Coulomb
//! friction model (`static_friction` and `dynamic_friction`). Friction is a
//! *positional* correction applied by the solver after the normal projection:
//! the tangential drift the pair accumulated during the substep is either fully
//! cancelled (static, while it stays inside the cone `mu_s * penetration`) or
//! clamped to the dynamic cone `mu_d * penetration` (kinetic sliding). Both
//! coefficients default to `0` in [`ContactConstraint::new`], so a plain
//! contact is frictionless and byte-for-byte identical to the pre-friction
//! solver; friction is opted into with [`ContactConstraint::with_friction`].
//!
//! Provenance: canonical `XPBD` inequality (contact) constraint of Müller et
//! al., with the positional Coulomb friction of Müller et al. 2020 ("Detailed
//! Rigid Body Simulation with `XPBD`") bounded by penetration depth after
//! Macklin et al. 2014. No Unreal Engine source or derived code.

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
    /// Static (stick) Coulomb coefficient; the tangential drift is fully
    /// cancelled while it stays inside the cone `static_friction * penetration`.
    pub static_friction: f32,
    /// Dynamic (slip) Coulomb coefficient bounding the tangential correction to
    /// the cone `dynamic_friction * penetration` once the pair is sliding.
    pub dynamic_friction: f32,
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
            static_friction: 0.0,
            dynamic_friction: 0.0,
        }
    }

    /// Returns a copy of this contact with Coulomb friction enabled.
    ///
    /// `static_friction` bounds the stick cone and `dynamic_friction` the slip
    /// cone; both are clamped to be non-negative. A dynamic coefficient larger
    /// than the static one is permitted (the caller owns that choice) — the
    /// solver simply applies whichever cone the current tangential drift falls
    /// in. Leaving this un-called (or passing zeros) keeps the contact
    /// frictionless.
    #[must_use]
    pub fn with_friction(self, static_friction: f32, dynamic_friction: f32) -> ContactConstraint {
        ContactConstraint {
            static_friction: static_friction.max(0.0),
            dynamic_friction: dynamic_friction.max(0.0),
            ..self
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
            static_friction: self.static_friction,
            dynamic_friction: self.dynamic_friction,
        }
    }
}

impl ColouredEdge for ContactConstraint {
    fn endpoints(&self) -> (u32, u32) {
        (self.a, self.b)
    }
}

/// `std430`-compatible upload form of [`ContactConstraint`] (24 bytes).
///
/// All six members are 4-byte scalars, so the `std430` array stride is a tight
/// 24 bytes with no interior or trailing padding — matching the `Contact`
/// struct in `shaders/contacts_resolve.wgsl` field for field.
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
    /// Static (stick) Coulomb coefficient.
    pub static_friction: f32,
    /// Dynamic (slip) Coulomb coefficient.
    pub dynamic_friction: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_frictionless() {
        let con = ContactConstraint::new(0, 1, 2.0, 0.0);
        assert_eq!(con.static_friction, 0.0);
        assert_eq!(con.dynamic_friction, 0.0);
    }

    #[test]
    fn with_friction_sets_and_preserves_other_fields() {
        let con = ContactConstraint::new(3, 7, 2.0, 1.0e-6).with_friction(0.8, 0.5);
        assert_eq!(con.a, 3);
        assert_eq!(con.b, 7);
        assert!((con.rest - 2.0).abs() < 1e-6);
        assert!((con.compliance - 1.0e-6).abs() < 1e-12);
        assert!((con.static_friction - 0.8).abs() < 1e-6);
        assert!((con.dynamic_friction - 0.5).abs() < 1e-6);
    }

    #[test]
    fn with_friction_clamps_negatives_to_zero() {
        let con = ContactConstraint::new(0, 1, 2.0, 0.0).with_friction(-1.0, -2.0);
        assert_eq!(con.static_friction, 0.0);
        assert_eq!(con.dynamic_friction, 0.0);
    }

    #[test]
    fn to_gpu_carries_friction_and_is_twenty_four_bytes() {
        assert_eq!(size_of::<GpuContactConstraint>(), 24);
        let gpu = ContactConstraint::new(1, 2, 2.0, 0.0)
            .with_friction(0.6, 0.4)
            .to_gpu();
        assert_eq!(gpu.a, 1);
        assert_eq!(gpu.b, 2);
        assert!((gpu.static_friction - 0.6).abs() < 1e-6);
        assert!((gpu.dynamic_friction - 0.4).abs() < 1e-6);
    }
}
