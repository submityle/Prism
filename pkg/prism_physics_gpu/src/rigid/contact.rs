//! The 6-DOF rigid-body contact constraint and its device-packed form.
//!
//! A [`RigidContact`] couples two rigid bodies the narrow phase found touching
//! and keeps them from interpenetrating, resolving both the linear and the
//! *angular* response: unlike the particle contact in
//! [`ContactConstraint`](crate::ContactConstraint) — which only exchanges linear
//! momentum between two point masses — a rigid contact is applied at an offset
//! `anchor` from each body's centre of mass, so the resolving impulse also
//! produces a torque `r x P` through the body's inverse inertia. That angular
//! arm is what makes a struck box spin, a stacked crate settle flat, and a
//! glancing hit impart tumble rather than only slowing the bodies down.
//!
//! # Frame conventions
//!
//! * `anchor_a` / `anchor_b` are world-space vectors **from each body's centre
//!   of mass to the shared contact point**, captured at the instant the contact
//!   was generated. The velocity solver never moves bodies, so the anchors stay
//!   valid for the whole solve.
//! * `normal` points **from body `b` toward body `a`** and is unit length. A
//!   positive accumulated normal impulse therefore pushes `a` along `+normal`
//!   and `b` along `-normal` — a contact force that can only ever separate the
//!   pair, never pull it together.
//! * `penetration` is positive when the bodies overlap (the depth the solver
//!   must resolve) and non-positive when they are already separated.
//!
//! # Combined material coefficients
//!
//! `friction` is a single combined Coulomb coefficient (the narrow phase or the
//! material table is expected to have already combined the two bodies' values,
//! e.g. by `sqrt(mu_a * mu_b)`), and `restitution` is a single combined
//! coefficient in `[0, 1]`. Keeping one combined value per contact — rather than
//! two per-body values — matches what a production contact buffer carries into
//! the solver and keeps the device struct compact.
//!
//! # Warm-start accumulators
//!
//! The solver is *accumulating*: `normal_impulse` and the two `tangent_impulse`
//! components persist the total impulse applied over a frame's iterations, both
//! so friction can be clamped against the running normal impulse and so the next
//! frame can seed ("warm start") from this frame's converged solution. A caller
//! that carries the same [`RigidContact`] across frames therefore gets
//! warm-started solves for free; a caller that rebuilds contacts each frame
//! simply leaves the accumulators at their [`RigidContact::new`] zero.
//!
//! Provenance: the sequential-impulse rigid contact of Catto ("Iterative
//! Dynamics with Temporal Coherence", 2005; `Box2D`) with the box-clamped Coulomb
//! friction and warm-starting standard to that method. No Unreal Engine source
//! or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// A one-sided non-penetration contact between two rigid bodies, resolved with
/// both a linear and an angular impulse response.
///
/// All vectors are world-space. See the [module documentation](self) for the
/// full frame conventions; in short, `normal` points from `b` to `a`, the
/// `anchor`s run from each centre of mass to the contact point, and the three
/// impulse fields accumulate the solver's running solution for clamping and
/// warm starting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidContact {
    /// Index of the first body (the one `normal` points toward).
    pub body_a: u32,
    /// Index of the second body.
    pub body_b: u32,
    /// World-space arm from body `a`'s centre of mass to the contact point.
    pub anchor_a: Vec3,
    /// World-space arm from body `b`'s centre of mass to the contact point.
    pub anchor_b: Vec3,
    /// Unit contact normal, pointing from body `b` toward body `a`.
    pub normal: Vec3,
    /// Penetration depth; positive when the bodies overlap.
    pub penetration: f32,
    /// Combined Coulomb friction coefficient (non-negative).
    pub friction: f32,
    /// Combined restitution coefficient in `[0, 1]`.
    pub restitution: f32,
    /// Accumulated normal impulse (clamped non-negative), persisted for
    /// friction clamping and cross-frame warm starting.
    pub normal_impulse: f32,
    /// Accumulated friction impulse along the first contact tangent.
    pub tangent_impulse_0: f32,
    /// Accumulated friction impulse along the second contact tangent.
    pub tangent_impulse_1: f32,
}

impl RigidContact {
    /// Creates a contact between bodies `a` and `b` with zeroed warm-start
    /// accumulators.
    ///
    /// `normal` is normalised (falling back to `+Y` if degenerate) so the solver
    /// can rely on a unit normal, `penetration` is kept as given (a non-positive
    /// value marks an already-separated pair the solver's bias term ignores),
    /// and `friction` and `restitution` are clamped to their valid ranges so a
    /// caller cannot construct an attractive or energy-adding contact.
    #[must_use]
    pub fn new(
        body_a: u32,
        body_b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        normal: Vec3,
        penetration: f32,
    ) -> RigidContact {
        let normal = if normal.length_squared() > 0.0 {
            normal.normalize()
        } else {
            Vec3::Y
        };
        RigidContact {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            normal,
            penetration,
            friction: 0.0,
            restitution: 0.0,
            normal_impulse: 0.0,
            tangent_impulse_0: 0.0,
            tangent_impulse_1: 0.0,
        }
    }

    /// Returns a copy of this contact with the given combined `friction`
    /// coefficient (clamped non-negative).
    #[must_use]
    pub fn with_friction(self, friction: f32) -> RigidContact {
        RigidContact {
            friction: friction.max(0.0),
            ..self
        }
    }

    /// Returns a copy of this contact with the given combined `restitution`
    /// coefficient (clamped to `[0, 1]`).
    #[must_use]
    pub fn with_restitution(self, restitution: f32) -> RigidContact {
        RigidContact {
            restitution: restitution.clamp(0.0, 1.0),
            ..self
        }
    }

    /// Packs this contact into its device-upload form.
    #[must_use]
    pub(crate) fn to_gpu(self) -> GpuRigidContact {
        GpuRigidContact {
            anchor_a: [self.anchor_a.x, self.anchor_a.y, self.anchor_a.z, 0.0],
            anchor_b: [self.anchor_b.x, self.anchor_b.y, self.anchor_b.z, 0.0],
            normal: [self.normal.x, self.normal.y, self.normal.z, 0.0],
            body_a: self.body_a,
            body_b: self.body_b,
            penetration: self.penetration,
            friction: self.friction,
            restitution: self.restitution,
            normal_impulse: self.normal_impulse,
            tangent_impulse_0: self.tangent_impulse_0,
            tangent_impulse_1: self.tangent_impulse_1,
        }
    }
}

/// The device-packed, 16-byte-aligned form of a [`RigidContact`].
///
/// Layout matches `Contact` in `shaders/rigid_contact.wgsl`: three padded
/// `vec4` arms/normal followed by the two body indices and the six scalar
/// fields, for a total of 80 bytes. The accumulated-impulse fields are
/// read-write on device so the solver's warm-start seed and converged solution
/// round-trip through the same buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuRigidContact {
    /// `anchor_a` padded to a `vec4`.
    pub anchor_a: [f32; 4],
    /// `anchor_b` padded to a `vec4`.
    pub anchor_b: [f32; 4],
    /// `normal` padded to a `vec4`.
    pub normal: [f32; 4],
    /// Index of body `a`.
    pub body_a: u32,
    /// Index of body `b`.
    pub body_b: u32,
    /// Penetration depth.
    pub penetration: f32,
    /// Combined friction coefficient.
    pub friction: f32,
    /// Combined restitution coefficient.
    pub restitution: f32,
    /// Accumulated normal impulse.
    pub normal_impulse: f32,
    /// Accumulated first-tangent friction impulse.
    pub tangent_impulse_0: f32,
    /// Accumulated second-tangent friction impulse.
    pub tangent_impulse_1: f32,
}

/// Builds a right-handed orthonormal tangent basis `(t1, t2)` spanning the
/// contact plane perpendicular to the unit normal `n`.
///
/// Uses the branchless construction of Frisvad / Catto: pick the smaller of
/// `n`'s components to seed a vector guaranteed not to be parallel to `n`, take
/// one cross product for `t1`, and a second for `t2`. The device shader runs the
/// identical selection and arithmetic so both engines agree on the friction
/// axes bit-for-intent.
#[must_use]
pub(crate) fn contact_tangents(n: Vec3) -> (Vec3, Vec3) {
    // Seed from whichever of the first two components keeps the cross product
    // well conditioned (the standard 0.57735 ~= 1/sqrt(3) threshold).
    let t1 = if n.x.abs() >= 0.577_350_26 {
        Vec3::new(n.y, -n.x, 0.0)
    } else {
        Vec3::new(0.0, n.z, -n.y)
    };
    let len = t1.length();
    let t1 = if len > 0.0 { t1 / len } else { Vec3::X };
    // `n` and `t1` are unit and orthogonal, so their cross product is already
    // unit length; no second normalisation is needed.
    let t2 = n.cross(t1);
    (t1, t2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normalises_normal_and_clamps_coefficients() {
        let c = RigidContact::new(0, 1, Vec3::X, -Vec3::X, Vec3::new(0.0, 2.0, 0.0), 0.1)
            .with_friction(-1.0)
            .with_restitution(5.0);
        assert!((c.normal.length() - 1.0).abs() < 1e-6);
        assert_eq!(c.friction, 0.0);
        assert_eq!(c.restitution, 1.0);
    }

    #[test]
    fn degenerate_normal_falls_back_to_up() {
        let c = RigidContact::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0.0);
        assert_eq!(c.normal, Vec3::Y);
    }

    #[test]
    fn tangents_are_orthonormal_to_normal() {
        for n in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(-0.3, 0.8, 0.5).normalize(),
        ] {
            let (t1, t2) = contact_tangents(n);
            assert!((t1.length() - 1.0).abs() < 1e-5, "t1 not unit for {n:?}");
            assert!((t2.length() - 1.0).abs() < 1e-5, "t2 not unit for {n:?}");
            assert!(t1.dot(n).abs() < 1e-5, "t1 not perpendicular for {n:?}");
            assert!(t2.dot(n).abs() < 1e-5, "t2 not perpendicular for {n:?}");
            assert!(t1.dot(t2).abs() < 1e-5, "t1,t2 not orthogonal for {n:?}");
        }
    }

    #[test]
    fn gpu_packing_round_trips_fields() {
        let c = RigidContact::new(
            3,
            7,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::Y,
            0.25,
        )
        .with_friction(0.5)
        .with_restitution(0.3);
        let g = c.to_gpu();
        assert_eq!(g.body_a, 3);
        assert_eq!(g.body_b, 7);
        assert_eq!(g.anchor_a, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(g.normal, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(g.penetration, 0.25);
        assert_eq!(g.friction, 0.5);
        assert_eq!(g.restitution, 0.3);
    }
}
