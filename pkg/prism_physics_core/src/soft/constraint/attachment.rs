//! Attachment constraint pinning a particle toward a fixed world point.
//!
//! An [`AttachmentConstraint`] softly (or rigidly) drives one particle toward a
//! fixed target position in world space. It complements the "infinite mass"
//! pinning offered by [`ParticleStorage::spawn_pinned`], which is unconditional:
//! an attachment keeps the particle *dynamic* (it still responds to other
//! constraints) while being pulled toward an anchor with a tunable compliance.
//! This is what lets a cloth corner be nailed to a moving hook, or a rope end
//! follow a character's hand.
//!
//! The constraint function is the distance to the target,
//!
//! ```text
//! C = |p - target|
//! ```
//!
//! with unit gradient `n = (p - target) / |p - target|`, so the XPBD
//! denominator is `w + alpha_tilde`.
//!
//! [`ParticleStorage::spawn_pinned`]: crate::soft::particle::ParticleStorage::spawn_pinned
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Pinning a
//! particle to a target point via a zero-rest-length XPBD distance constraint
//! is a standard position-based-dynamics technique.

use glam::Vec3;

use crate::math::scalar::{Real, EPSILON};
use crate::soft::particle::ParticleHandle;

use super::{ParticleConstraint, SoftConstraintKind};

/// A compliant XPBD constraint pulling one particle toward a fixed world point.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AttachmentConstraint {
    /// The particle being attached.
    pub particle: ParticleHandle,
    /// The fixed world-space anchor the particle is driven toward.
    pub target: Vec3,
    /// Compliance (inverse stiffness); `0` snaps the particle onto the target.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier for the current substep.
    lambda: Real,
}

impl AttachmentConstraint {
    /// Creates an attachment driving `particle` toward `target` with the given
    /// `compliance` (`0` for a rigid pin).
    #[must_use]
    pub fn new(particle: ParticleHandle, target: Vec3, compliance: Real) -> Self {
        AttachmentConstraint {
            particle,
            target,
            compliance: compliance.max(0.0),
            lambda: 0.0,
        }
    }

    /// Moves the anchor to a new world position. Use this each frame to drag an
    /// attached particle along a kinematic path.
    pub fn set_target(&mut self, target: Vec3) {
        self.target = target;
    }

    /// Returns the current accumulated Lagrange multiplier (for diagnostics and
    /// tests).
    #[must_use]
    pub fn lambda(&self) -> Real {
        self.lambda
    }
}

impl ParticleConstraint for AttachmentConstraint {
    fn kind(&self) -> SoftConstraintKind {
        SoftConstraintKind::Attachment
    }

    fn compliance(&self) -> Real {
        self.compliance
    }

    fn reset(&mut self) {
        self.lambda = 0.0;
    }

    fn project(&mut self, positions: &mut [Vec3], inverse_masses: &[Real], dt: Real) {
        let i = self.particle.index();
        let (Some(&w), Some(&position)) = (inverse_masses.get(i), positions.get(i)) else {
            return;
        };
        if w <= 0.0 {
            return;
        }
        let delta = position - self.target;
        let length = delta.length();
        if length < EPSILON {
            return;
        }
        let normal = delta / length;
        let alpha_tilde = self.compliance / (dt * dt);
        let delta_lambda = (-length - alpha_tilde * self.lambda) / (w + alpha_tilde);
        self.lambda += delta_lambda;
        positions[i] += normal * (delta_lambda * w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    #[test]
    fn rigid_attachment_snaps_particle_onto_target() {
        let mut positions = [Vec3::new(3.0, 0.0, 0.0)];
        let inv = [1.0];
        let mut c = AttachmentConstraint::new(h(0), Vec3::ZERO, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert!(positions[0].length() < 1e-5, "pos was {:?}", positions[0]);
    }

    #[test]
    fn pinned_particle_is_not_moved() {
        let mut positions = [Vec3::new(3.0, 0.0, 0.0)];
        let inv = [0.0];
        let mut c = AttachmentConstraint::new(h(0), Vec3::ZERO, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn already_at_target_is_inert() {
        let mut positions = [Vec3::ZERO];
        let inv = [1.0];
        let mut c = AttachmentConstraint::new(h(0), Vec3::ZERO, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::ZERO);
    }

    #[test]
    fn out_of_range_is_inert() {
        let mut positions = [Vec3::ZERO];
        let inv = [1.0];
        let mut c = AttachmentConstraint::new(h(5), Vec3::ONE, 0.0);
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[0], Vec3::ZERO);
    }

    #[test]
    fn set_target_moves_anchor() {
        let mut c = AttachmentConstraint::new(h(0), Vec3::ZERO, 0.0);
        c.set_target(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(c.target, Vec3::new(1.0, 2.0, 3.0));
    }
}
