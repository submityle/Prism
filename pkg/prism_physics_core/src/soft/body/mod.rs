//! A self-contained deformable body: particles, constraints, and solver config.
//!
//! [`SoftBody`] is the top-level unit the rest of the engine (and the cloth /
//! rope / soft-body builders) works with. It bundles a [`ParticleStorage`], a
//! [`ConstraintSet`], and a [`SoftSolverConfig`], and advances them with the
//! shared [`SoftSolver`]. Cloth, rope, and volumetric soft bodies are all just
//! a [`SoftBody`] populated with different particles and constraints.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::constraint::{
    AttachmentConstraint, BendingConstraint, ConstraintSet, DistanceConstraint,
    TetraVolumeConstraint,
};
use crate::soft::particle::{ParticleHandle, ParticleStorage};
use crate::soft::solver::{SoftSolver, SoftSolverConfig};

/// A deformable body advanced by the substep XPBD [`SoftSolver`].
///
/// Construct one directly and add particles and constraints, or use the
/// builders in [`crate::soft::build`] to author cloth, rope, and soft bodies.
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SoftBody {
    /// The point-mass particles making up this body.
    pub particles: ParticleStorage,
    /// The constraints coupling this body's particles.
    pub constraints: ConstraintSet,
    /// Solver tunables (gravity, substeps, iterations, damping).
    pub config: SoftSolverConfig,
}

impl SoftBody {
    /// Creates an empty soft body with the given solver configuration.
    #[must_use]
    pub fn new(config: SoftSolverConfig) -> SoftBody {
        SoftBody {
            particles: ParticleStorage::new(),
            constraints: ConstraintSet::new(),
            config,
        }
    }

    /// Creates an empty soft body with the default solver configuration.
    #[must_use]
    pub fn with_default_config() -> SoftBody {
        SoftBody::new(SoftSolverConfig::default())
    }

    /// Spawns a dynamic particle at `position` with the given `mass`.
    pub fn spawn(&mut self, position: Vec3, mass: Real) -> ParticleHandle {
        self.particles.spawn(position, mass)
    }

    /// Spawns a pinned (immovable) particle at `position`.
    pub fn spawn_pinned(&mut self, position: Vec3) -> ParticleHandle {
        self.particles.spawn_pinned(position)
    }

    /// Adds a distance constraint between `a` and `b` whose rest length is the
    /// current separation. Returns `false` (adding nothing) if either handle is
    /// out of range.
    pub fn connect(&mut self, a: ParticleHandle, b: ParticleHandle, compliance: Real) -> bool {
        match DistanceConstraint::from_positions(a, b, self.particles.positions(), compliance) {
            Some(c) => {
                self.constraints.distance.push(c);
                true
            }
            None => false,
        }
    }

    /// Adds a pre-built distance constraint.
    pub fn add_distance(&mut self, constraint: DistanceConstraint) {
        self.constraints.distance.push(constraint);
    }

    /// Adds a pre-built bending constraint.
    pub fn add_bending(&mut self, constraint: BendingConstraint) {
        self.constraints.bending.push(constraint);
    }

    /// Adds a pre-built tetrahedral volume constraint.
    pub fn add_volume(&mut self, constraint: TetraVolumeConstraint) {
        self.constraints.volume.push(constraint);
    }

    /// Adds a pre-built attachment constraint.
    pub fn add_attachment(&mut self, constraint: AttachmentConstraint) {
        self.constraints.attachment.push(constraint);
    }

    /// Advances the body by `dt` seconds using the substep XPBD solver.
    pub fn step(&mut self, dt: Real) {
        SoftSolver::new().step(&mut self.particles, &mut self.constraints, &self.config, dt);
    }

    /// Returns the axis-aligned bounding box of the current particle positions
    /// as `(min, max)`, or `None` when the body has no particles.
    #[must_use]
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let positions = self.particles.positions();
        let (first, rest) = positions.split_first()?;
        let mut min = *first;
        let mut max = *first;
        for &p in rest {
            min = min.min(p);
            max = max.max(p);
        }
        Some((min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_body_is_empty() {
        let body = SoftBody::with_default_config();
        assert!(body.particles.is_empty());
        assert!(body.constraints.is_empty());
        assert!(body.bounds().is_none());
    }

    #[test]
    fn connect_uses_current_separation_as_rest_length() {
        let mut body = SoftBody::with_default_config();
        let a = body.spawn(Vec3::ZERO, 1.0);
        let b = body.spawn(Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(body.connect(a, b, 0.0));
        assert_eq!(body.constraints.distance.len(), 1);
        assert!((body.constraints.distance[0].rest_length - 3.0).abs() < 1e-6);
    }

    #[test]
    fn connect_rejects_out_of_range_handle() {
        let mut body = SoftBody::with_default_config();
        let a = body.spawn(Vec3::ZERO, 1.0);
        assert!(!body.connect(a, ParticleHandle::from_index(9), 0.0));
        assert!(body.constraints.distance.is_empty());
    }

    #[test]
    fn bounds_span_all_particles() {
        let mut body = SoftBody::with_default_config();
        body.spawn(Vec3::new(-1.0, 0.0, 2.0), 1.0);
        body.spawn(Vec3::new(3.0, -4.0, 1.0), 1.0);
        let (min, max) = body.bounds().unwrap();
        assert_eq!(min, Vec3::new(-1.0, -4.0, 1.0));
        assert_eq!(max, Vec3::new(3.0, 0.0, 2.0));
    }

    #[test]
    fn step_moves_dynamic_particles() {
        let mut body = SoftBody::new(SoftSolverConfig {
            damping: 0.0,
            ..SoftSolverConfig::default()
        });
        let p = body.spawn(Vec3::ZERO, 1.0);
        body.step(1.0 / 60.0);
        assert!(body.particles.position(p).unwrap().y < 0.0);
    }
}
