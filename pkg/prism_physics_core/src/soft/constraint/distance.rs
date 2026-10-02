//! Distance (stretch) constraint between two particles.
//!
//! A [`DistanceConstraint`] holds two particles at a target *rest length*. It
//! is the workhorse of the unified kernel: cloth structural/shear edges, rope
//! links, and soft-body lattice edges are all distance constraints. The
//! constraint function is
//!
//! ```text
//! C = |p_a - p_b| - rest_length
//! ```
//!
//! with unit gradients `+n` on `a` and `-n` on `b`, where `n` is the
//! normalized separation. Because both gradients are unit vectors, the XPBD
//! denominator reduces to `w_a + w_b + alpha_tilde`.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! distance-constraint projection is the canonical XPBD stretch constraint
//! (Müller et al.).

use glam::Vec3;

use crate::math::scalar::{Real, EPSILON};
use crate::soft::particle::ParticleHandle;

use super::{ParticleConstraint, SoftConstraintKind};

/// A compliant XPBD constraint keeping two particles at a rest length.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DistanceConstraint {
    /// First coupled particle.
    pub a: ParticleHandle,
    /// Second coupled particle.
    pub b: ParticleHandle,
    /// Target separation between the two particles, in metres.
    pub rest_length: Real,
    /// Compliance (inverse stiffness); `0` is perfectly rigid.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier for the current substep.
    lambda: Real,
}

impl DistanceConstraint {
    /// Creates a distance constraint between `a` and `b` with an explicit
    /// `rest_length` and `compliance`.
    #[must_use]
    pub fn new(a: ParticleHandle, b: ParticleHandle, rest_length: Real, compliance: Real) -> Self {
        DistanceConstraint {
            a,
            b,
            rest_length: rest_length.max(0.0),
            compliance: compliance.max(0.0),
            lambda: 0.0,
        }
    }

    /// Creates a distance constraint whose rest length is the current
    /// separation between `a` and `b` in `positions`.
    ///
    /// This is the common builder path: sample the authored geometry to fix the
    /// rest length, then simulate. Returns `None` if either handle is out of
    /// range.
    #[must_use]
    pub fn from_positions(
        a: ParticleHandle,
        b: ParticleHandle,
        positions: &[Vec3],
        compliance: Real,
    ) -> Option<Self> {
        let pa = positions.get(a.index())?;
        let pb = positions.get(b.index())?;
        Some(DistanceConstraint::new(
            a,
            b,
            (*pa - *pb).length(),
            compliance,
        ))
    }

    /// Returns the current accumulated Lagrange multiplier (for diagnostics and
    /// tests).
    #[must_use]
    pub fn lambda(&self) -> Real {
        self.lambda
    }
}

impl ParticleConstraint for DistanceConstraint {
    fn kind(&self) -> SoftConstraintKind {
        SoftConstraintKind::Distance
    }

    fn compliance(&self) -> Real {
        self.compliance
    }

    fn reset(&mut self) {
        self.lambda = 0.0;
    }

    fn project(&mut self, positions: &mut [Vec3], inverse_masses: &[Real], dt: Real) {
        // The projection arithmetic lives in the free `project_distance_constraint`
        // below so the parallel GPU twin and the render-side cloth solver can share
        // this one, authoritative XPBD step; the constraint object only owns the
        // warm-started Lagrange multiplier that is threaded through it.
        self.lambda = project_distance_constraint(
            positions,
            inverse_masses,
            self.a.index(),
            self.b.index(),
            self.rest_length,
            self.compliance,
            self.lambda,
            dt,
        );
    }
}

/// Projects a single two-sided distance constraint between particles `a` and
/// `b` in place over raw particle indices, returning the updated accumulated
/// Lagrange multiplier.
///
/// This is the one, authoritative XPBD distance step. [`DistanceConstraint`]
/// wraps it to own a warm-started multiplier across solver iterations, while a
/// parallel GPU twin (and the render-side cloth solver) drive it directly from
/// their own index space with a fresh `lambda` of `0.0` per projection — so
/// there is exactly one copy of the arithmetic.
///
/// The projection is inert (the input `lambda` is returned unchanged) for an
/// out-of-range pair, a pair with no free inverse mass, or a coincident pair
/// with no defined separation direction. The correction is split between the
/// endpoints by inverse mass; the separation direction is `delta / length`
/// (component-wise), matching the GPU twin bit-for-bit.
///
/// `rest_length` and `compliance` are used as given (the
/// [`DistanceConstraint`] constructor already clamps them non-negative, and the
/// render cloth solver passes its clamped `Compliance::value()`).
///
/// # Provenance
///
/// XPBD distance projection is a published position-based-dynamics technique
/// (Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained
/// Dynamics"). No Unreal Engine source or derived code.
#[must_use]
pub fn project_distance_constraint(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    a: usize,
    b: usize,
    rest_length: Real,
    compliance: Real,
    lambda: Real,
    dt: Real,
) -> Real {
    // Bounds guard: an out-of-range pair is inert rather than a panic.
    let (Some(&wa), Some(&wb)) = (inverse_masses.get(a), inverse_masses.get(b)) else {
        return lambda;
    };
    if a >= positions.len() || b >= positions.len() {
        return lambda;
    }
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return lambda;
    }
    let delta = positions[a] - positions[b];
    let length = delta.length();
    if length < EPSILON {
        return lambda;
    }
    let normal = delta / length;
    let c = length - rest_length;
    let alpha_tilde = compliance / (dt * dt);
    let delta_lambda = (-c - alpha_tilde * lambda) / (w_sum + alpha_tilde);
    let correction = normal * delta_lambda;
    positions[a] += correction * wa;
    positions[b] -= correction * wb;
    lambda + delta_lambda
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    #[test]
    fn rigid_constraint_restores_rest_length_in_one_projection() {
        // Two equal masses pulled apart to length 2 with rest length 1.
        let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let inv = [1.0, 1.0];
        let mut c = DistanceConstraint::new(h(0), h(1), 1.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        let length = (positions[0] - positions[1]).length();
        assert!((length - 1.0).abs() < 1e-5, "length was {length}");
        // Symmetric masses => symmetric correction about the midpoint.
        assert!((positions[0] - Vec3::new(0.5, 0.0, 0.0)).length() < 1e-5);
        assert!((positions[1] - Vec3::new(1.5, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn pinned_partner_absorbs_all_correction() {
        // a is pinned (inv mass 0); only b moves.
        let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let inv = [0.0, 1.0];
        let mut c = DistanceConstraint::new(h(0), h(1), 1.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::ZERO);
        assert!((positions[1] - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn both_pinned_is_inert() {
        let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let inv = [0.0, 0.0];
        let mut c = DistanceConstraint::new(h(0), h(1), 1.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[1], Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn out_of_range_is_inert() {
        let mut positions = [Vec3::ZERO];
        let inv = [1.0];
        let mut c = DistanceConstraint::new(h(0), h(9), 1.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::ZERO);
    }

    #[test]
    fn from_positions_captures_current_separation() {
        let positions = [Vec3::ZERO, Vec3::new(3.0, 4.0, 0.0)];
        let c = DistanceConstraint::from_positions(h(0), h(1), &positions, 0.0).unwrap();
        assert!((c.rest_length - 5.0).abs() < 1e-6);
    }

    #[test]
    fn compliant_constraint_is_softer_than_rigid() {
        let dt = 1.0 / 60.0;
        let inv = [1.0, 1.0];
        let start = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];

        let mut rigid = start;
        let mut cr = DistanceConstraint::new(h(0), h(1), 1.0, 0.0);
        cr.project(&mut rigid, &inv, dt);
        let rigid_len = (rigid[0] - rigid[1]).length();

        let mut soft = start;
        let mut cs = DistanceConstraint::new(h(0), h(1), 1.0, 1.0);
        cs.project(&mut soft, &inv, dt);
        let soft_len = (soft[0] - soft[1]).length();

        // Rigid snaps closer to rest length; soft barely moves.
        assert!((rigid_len - 1.0).abs() < (soft_len - 1.0).abs());
        assert!(soft_len > 1.9);
    }

    #[test]
    fn degenerate_coincident_particles_are_inert() {
        let mut positions = [Vec3::ZERO, Vec3::ZERO];
        let inv = [1.0, 1.0];
        let mut c = DistanceConstraint::new(h(0), h(1), 1.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::ZERO);
        assert_eq!(positions[1], Vec3::ZERO);
    }
}
