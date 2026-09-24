//! Bending constraint resisting folding at a three-particle joint.
//!
//! A [`BendingConstraint`] couples three particles `a`, `center`, `b` and
//! resists the joint at `center` folding away from its rest configuration. It
//! is the trig-free "point-to-midpoint" bending model: the constraint measures
//! how far `center` sits from the midpoint `M = (a + b) / 2` of its neighbours
//! and drives that offset back to the rest value captured at build time,
//!
//! ```text
//! M = (a + b) / 2
//! C = |center - M| - rest_offset
//! ```
//!
//! Applied along each row and column of a cloth grid it resists out-of-plane
//! folding; applied along a chain it gives rope and hair their stiffness. It
//! deliberately avoids any inverse-trigonometric dihedral-angle evaluation so
//! that results are portable and deterministic across platforms.
//!
//! The gradients are `+n` on `center` and `-n/2` on each of `a` and `b`, where
//! `n = (center - M) / |center - M|`, so the XPBD denominator is
//! `w_center + (w_a + w_b) / 4 + alpha_tilde`.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! point-to-midpoint bending constraint is a standard, publicly documented
//! position-based-dynamics technique.

use glam::Vec3;

use crate::math::scalar::{Real, EPSILON};
use crate::soft::particle::ParticleHandle;

use super::{ParticleConstraint, SoftConstraintKind};

/// A compliant XPBD constraint resisting folding at the `center` particle of a
/// three-particle joint.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BendingConstraint {
    /// First neighbour particle.
    pub a: ParticleHandle,
    /// The joint particle whose folding is resisted.
    pub center: ParticleHandle,
    /// Second neighbour particle.
    pub b: ParticleHandle,
    /// Rest offset of `center` from the midpoint of `a` and `b`, in metres.
    pub rest_offset: Real,
    /// Compliance (inverse stiffness); `0` is perfectly rigid.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier for the current substep.
    lambda: Real,
}

impl BendingConstraint {
    /// Creates a bending constraint with an explicit `rest_offset` and
    /// `compliance`.
    #[must_use]
    pub fn new(
        a: ParticleHandle,
        center: ParticleHandle,
        b: ParticleHandle,
        rest_offset: Real,
        compliance: Real,
    ) -> Self {
        BendingConstraint {
            a,
            center,
            b,
            rest_offset: rest_offset.max(0.0),
            compliance: compliance.max(0.0),
            lambda: 0.0,
        }
    }

    /// Creates a bending constraint whose rest offset is sampled from the
    /// current geometry in `positions`. Returns `None` if any handle is out of
    /// range.
    #[must_use]
    pub fn from_positions(
        a: ParticleHandle,
        center: ParticleHandle,
        b: ParticleHandle,
        positions: &[Vec3],
        compliance: Real,
    ) -> Option<Self> {
        let pa = positions.get(a.index())?;
        let pc = positions.get(center.index())?;
        let pb = positions.get(b.index())?;
        let midpoint = (*pa + *pb) * 0.5;
        Some(BendingConstraint::new(
            a,
            center,
            b,
            (*pc - midpoint).length(),
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

impl ParticleConstraint for BendingConstraint {
    fn kind(&self) -> SoftConstraintKind {
        SoftConstraintKind::Bending
    }

    fn compliance(&self) -> Real {
        self.compliance
    }

    fn reset(&mut self) {
        self.lambda = 0.0;
    }

    fn project(&mut self, positions: &mut [Vec3], inverse_masses: &[Real], dt: Real) {
        let ia = self.a.index();
        let ic = self.center.index();
        let ib = self.b.index();
        let (Some(&wa), Some(&wc), Some(&wb)) = (
            inverse_masses.get(ia),
            inverse_masses.get(ic),
            inverse_masses.get(ib),
        ) else {
            return;
        };
        // Gradient magnitudes: |grad center| = 1, |grad a| = |grad b| = 1/2.
        let denom_mass = wc + 0.25 * (wa + wb);
        if denom_mass <= 0.0 {
            return;
        }
        let midpoint = (positions[ia] + positions[ib]) * 0.5;
        let delta = positions[ic] - midpoint;
        let length = delta.length();
        if length < EPSILON {
            return;
        }
        let normal = delta / length;
        let c = length - self.rest_offset;
        let alpha_tilde = self.compliance / (dt * dt);
        let delta_lambda = (-c - alpha_tilde * self.lambda) / (denom_mass + alpha_tilde);
        self.lambda += delta_lambda;
        positions[ic] += normal * (delta_lambda * wc);
        positions[ia] -= normal * (delta_lambda * wa * 0.5);
        positions[ib] -= normal * (delta_lambda * wb * 0.5);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    #[test]
    fn from_positions_captures_zero_offset_for_straight_chain() {
        let positions = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
        ];
        let c = BendingConstraint::from_positions(h(0), h(1), h(2), &positions, 0.0).unwrap();
        assert!(c.rest_offset < 1e-6);
    }

    #[test]
    fn rigid_bend_pulls_folded_center_back_toward_midpoint() {
        // Straight rest configuration (rest offset 0), then fold the center up.
        let mut positions = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        let inv = [1.0, 1.0, 1.0];
        let mut c = BendingConstraint::new(h(0), h(1), h(2), 0.0, 0.0);
        let before = (positions[1] - (positions[0] + positions[2]) * 0.5).length();
        c.project(&mut positions, &inv, 1.0 / 60.0);
        let after = (positions[1] - (positions[0] + positions[2]) * 0.5).length();
        assert!(after < before, "offset {before} -> {after}");
    }

    #[test]
    fn straight_chain_at_rest_is_inert() {
        let mut positions = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
        ];
        let inv = [1.0, 1.0, 1.0];
        let mut c = BendingConstraint::new(h(0), h(1), h(2), 0.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert!(positions[1].length() < 1e-6);
    }

    #[test]
    fn all_pinned_is_inert() {
        let mut positions = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        let inv = [0.0, 0.0, 0.0];
        let mut c = BendingConstraint::new(h(0), h(1), h(2), 0.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[1], Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn out_of_range_is_inert() {
        let mut positions = [Vec3::ZERO, Vec3::Y];
        let inv = [1.0, 1.0];
        let mut c = BendingConstraint::new(h(0), h(1), h(9), 0.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[1], Vec3::Y);
    }
}
