//! Volume-preservation constraint over a tetrahedron.
//!
//! A [`TetraVolumeConstraint`] couples the four particles of a tetrahedron and
//! drives its signed volume back to a rest value, giving volumetric soft bodies
//! their resistance to squashing and inflation. With `e1 = p1 - p0`,
//! `e2 = p2 - p0`, `e3 = p3 - p0`, the signed volume and constraint are
//!
//! ```text
//! V = (1/6) * dot(e1, cross(e2, e3))
//! C = V - rest_volume
//! ```
//!
//! with gradients (all trig-free, built from cross products)
//!
//! ```text
//! grad p1 = (1/6) * cross(e2, e3)
//! grad p2 = (1/6) * cross(e3, e1)
//! grad p3 = (1/6) * cross(e1, e2)
//! grad p0 = -(grad p1 + grad p2 + grad p3)
//! ```
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! tetrahedral volume-preservation constraint is a standard, publicly
//! documented position-based-dynamics technique.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::particle::ParticleHandle;

use super::{ParticleConstraint, SoftConstraintKind};

/// A compliant XPBD constraint preserving the signed volume of a tetrahedron.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TetraVolumeConstraint {
    /// The four particles forming the tetrahedron, in a fixed winding.
    pub particles: [ParticleHandle; 4],
    /// Target signed volume, in cubic metres.
    pub rest_volume: Real,
    /// Compliance (inverse stiffness); `0` is perfectly rigid.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier for the current substep.
    lambda: Real,
}

/// Returns the signed volume of the tetrahedron with the given corner
/// positions, using the standard scalar-triple-product formula.
#[must_use]
fn signed_volume(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> Real {
    (p1 - p0).dot((p2 - p0).cross(p3 - p0)) / 6.0
}

impl TetraVolumeConstraint {
    /// Creates a volume constraint with an explicit `rest_volume` and
    /// `compliance`.
    #[must_use]
    pub fn new(particles: [ParticleHandle; 4], rest_volume: Real, compliance: Real) -> Self {
        TetraVolumeConstraint {
            particles,
            rest_volume,
            compliance: compliance.max(0.0),
            lambda: 0.0,
        }
    }

    /// Creates a volume constraint whose rest volume is sampled from the current
    /// geometry in `positions`. Returns `None` if any handle is out of range.
    #[must_use]
    pub fn from_positions(
        particles: [ParticleHandle; 4],
        positions: &[Vec3],
        compliance: Real,
    ) -> Option<Self> {
        let p0 = *positions.get(particles[0].index())?;
        let p1 = *positions.get(particles[1].index())?;
        let p2 = *positions.get(particles[2].index())?;
        let p3 = *positions.get(particles[3].index())?;
        Some(TetraVolumeConstraint::new(
            particles,
            signed_volume(p0, p1, p2, p3),
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

impl ParticleConstraint for TetraVolumeConstraint {
    fn kind(&self) -> SoftConstraintKind {
        SoftConstraintKind::Volume
    }

    fn compliance(&self) -> Real {
        self.compliance
    }

    fn reset(&mut self) {
        self.lambda = 0.0;
    }

    fn project(&mut self, positions: &mut [Vec3], inverse_masses: &[Real], dt: Real) {
        let idx = [
            self.particles[0].index(),
            self.particles[1].index(),
            self.particles[2].index(),
            self.particles[3].index(),
        ];
        let mut w = [0.0; 4];
        for (slot, &i) in w.iter_mut().zip(idx.iter()) {
            let Some(&wi) = inverse_masses.get(i) else {
                return;
            };
            *slot = wi;
        }
        // Bounds already validated above via inverse_masses; positions share len.
        let p0 = positions[idx[0]];
        let p1 = positions[idx[1]];
        let p2 = positions[idx[2]];
        let p3 = positions[idx[3]];
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let e3 = p3 - p0;
        let grad1 = e2.cross(e3) / 6.0;
        let grad2 = e3.cross(e1) / 6.0;
        let grad3 = e1.cross(e2) / 6.0;
        let grad0 = -(grad1 + grad2 + grad3);
        let denom_mass = w[0] * grad0.length_squared()
            + w[1] * grad1.length_squared()
            + w[2] * grad2.length_squared()
            + w[3] * grad3.length_squared();
        if denom_mass <= 0.0 {
            return;
        }
        let volume = e1.dot(e2.cross(e3)) / 6.0;
        let c = volume - self.rest_volume;
        let alpha_tilde = self.compliance / (dt * dt);
        let delta_lambda = (-c - alpha_tilde * self.lambda) / (denom_mass + alpha_tilde);
        self.lambda += delta_lambda;
        positions[idx[0]] += grad0 * (delta_lambda * w[0]);
        positions[idx[1]] += grad1 * (delta_lambda * w[1]);
        positions[idx[2]] += grad2 * (delta_lambda * w[2]);
        positions[idx[3]] += grad3 * (delta_lambda * w[3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    fn unit_tetra() -> [Vec3; 4] {
        [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
    }

    #[test]
    fn signed_volume_of_unit_corner_tetra_is_one_sixth() {
        let [p0, p1, p2, p3] = unit_tetra();
        assert!((signed_volume(p0, p1, p2, p3) - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn from_positions_captures_rest_volume() {
        let positions = unit_tetra();
        let c = TetraVolumeConstraint::from_positions([h(0), h(1), h(2), h(3)], &positions, 0.0)
            .unwrap();
        assert!((c.rest_volume - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn rigid_constraint_restores_compressed_volume() {
        let mut positions = unit_tetra();
        let rest = signed_volume(positions[0], positions[1], positions[2], positions[3]);
        let mut c = TetraVolumeConstraint::new([h(0), h(1), h(2), h(3)], rest, 0.0);
        // Compress the apex toward the base, shrinking the volume.
        positions[3] = Vec3::new(0.0, 0.0, 0.5);
        let inv = [1.0, 1.0, 1.0, 1.0];
        for _ in 0..8 {
            c.reset();
            c.project(&mut positions, &inv, 1.0 / 60.0);
        }
        let restored = signed_volume(positions[0], positions[1], positions[2], positions[3]);
        assert!(
            (restored - rest).abs() < 1e-3,
            "restored {restored} vs {rest}"
        );
    }

    #[test]
    fn all_pinned_is_inert() {
        let mut positions = unit_tetra();
        positions[3] = Vec3::new(0.0, 0.0, 0.5);
        let snapshot = positions;
        let inv = [0.0, 0.0, 0.0, 0.0];
        let mut c = TetraVolumeConstraint::new([h(0), h(1), h(2), h(3)], 1.0 / 6.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions, snapshot);
    }

    #[test]
    fn out_of_range_is_inert() {
        let mut positions = unit_tetra();
        let snapshot = positions;
        let inv = [1.0, 1.0, 1.0, 1.0];
        let mut c = TetraVolumeConstraint::new([h(0), h(1), h(2), h(9)], 1.0 / 6.0, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions, snapshot);
    }
}
