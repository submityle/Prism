//! A self-contained Vertex Block Descent body: particles, springs, config.
//!
//! [`VbdBody`] is the convenience unit callers work with when they want the
//! unconditional stability of Vertex Block Descent instead of the XPBD
//! [`SoftBody`](crate::soft::body::SoftBody). It bundles a [`ParticleStorage`],
//! a [`SpringSet`], and a [`VbdConfig`], and advances them with the shared
//! [`VbdSolver`]. Cloth, rope, and lattice soft bodies are all just a
//! [`VbdBody`] populated with different particles and springs.
//!
//! Existing XPBD scenes can be reused verbatim: [`VbdBody::from_soft`] converts
//! a soft body's distance constraints into equivalent springs, so the same
//! authored topology can be driven by either solver.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::body::SoftBody;
use crate::soft::particle::{ParticleHandle, ParticleStorage};

use super::config::VbdConfig;
use super::element::{SpringElement, SpringSet};
use super::solver::VbdSolver;

/// A deformable body advanced by the [`VbdSolver`].
///
/// Construct one directly and add particles and springs, or convert an existing
/// [`SoftBody`] with [`from_soft`](Self::from_soft).
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VbdBody {
    /// The point-mass particles making up this body.
    pub particles: ParticleStorage,
    /// The springs coupling this body's particles.
    pub springs: SpringSet,
    /// Solver tunables (gravity, substeps, iterations, damping).
    pub config: VbdConfig,
}

impl VbdBody {
    /// Creates an empty body with the given solver configuration.
    #[must_use]
    pub fn new(config: VbdConfig) -> VbdBody {
        VbdBody {
            particles: ParticleStorage::new(),
            springs: SpringSet::new(),
            config,
        }
    }

    /// Creates an empty body with the default solver configuration.
    #[must_use]
    pub fn with_default_config() -> VbdBody {
        VbdBody::new(VbdConfig::default())
    }

    /// Builds a body from an existing XPBD [`SoftBody`], cloning its particles
    /// and converting its distance constraints into springs via
    /// [`SpringSet::from_constraint_set`]. Rigid (zero-compliance) constraints
    /// are mapped to `default_stiffness`.
    #[must_use]
    pub fn from_soft(soft: &SoftBody, default_stiffness: Real) -> VbdBody {
        VbdBody {
            particles: soft.particles.clone(),
            springs: SpringSet::from_constraint_set(&soft.constraints, default_stiffness),
            config: VbdConfig {
                gravity: soft.config.gravity,
                ..VbdConfig::default()
            },
        }
    }

    /// Spawns a dynamic particle at `position` with the given `mass`.
    pub fn spawn(&mut self, position: Vec3, mass: Real) -> ParticleHandle {
        self.particles.spawn(position, mass)
    }

    /// Spawns a pinned (immovable) particle at `position`.
    pub fn spawn_pinned(&mut self, position: Vec3) -> ParticleHandle {
        self.particles.spawn_pinned(position)
    }

    /// Connects `a` and `b` with a spring whose rest length is their current
    /// separation. Returns `false` (adding nothing) if either handle is out of
    /// range.
    pub fn connect(&mut self, a: ParticleHandle, b: ParticleHandle, stiffness: Real) -> bool {
        if !self.particles.contains(a) || !self.particles.contains(b) {
            return false;
        }
        let rest_length = self.springs_rest_length(a, b);
        self.springs
            .push(SpringElement::new(a, b, rest_length, stiffness));
        true
    }

    /// Returns the current separation between two (valid) particles.
    fn springs_rest_length(&self, a: ParticleHandle, b: ParticleHandle) -> Real {
        let positions = self.particles.positions();
        (positions[a.index()] - positions[b.index()]).length()
    }

    /// Adds a pre-built spring.
    pub fn add_spring(&mut self, spring: SpringElement) {
        self.springs.push(spring);
    }

    /// Advances the body by `dt` seconds using the Vertex Block Descent solver.
    pub fn step(&mut self, dt: Real) {
        VbdSolver::new().step(&mut self.particles, &self.springs, &self.config, dt);
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
    use crate::soft::constraint::DistanceConstraint;

    #[test]
    fn new_body_is_empty() {
        let body = VbdBody::with_default_config();
        assert!(body.particles.is_empty());
        assert!(body.springs.is_empty());
        assert!(body.bounds().is_none());
    }

    #[test]
    fn connect_uses_current_separation_as_rest_length() {
        let mut body = VbdBody::with_default_config();
        let a = body.spawn(Vec3::ZERO, 1.0);
        let b = body.spawn(Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(body.connect(a, b, 1000.0));
        assert_eq!(body.springs.len(), 1);
        assert!((body.springs.springs[0].rest_length - 3.0).abs() < 1e-6);
    }

    #[test]
    fn connect_rejects_out_of_range_handle() {
        let mut body = VbdBody::with_default_config();
        let a = body.spawn(Vec3::ZERO, 1.0);
        assert!(!body.connect(a, ParticleHandle::from_index(9), 1000.0));
        assert!(body.springs.is_empty());
    }

    #[test]
    fn from_soft_clones_particles_and_maps_constraints() {
        let mut soft = SoftBody::with_default_config();
        let a = soft.spawn_pinned(Vec3::ZERO);
        let b = soft.spawn(Vec3::new(0.0, -1.0, 0.0), 1.0);
        soft.add_distance(DistanceConstraint::new(a, b, 1.0, 0.01));
        let body = VbdBody::from_soft(&soft, 5000.0);
        assert_eq!(body.particles.len(), 2);
        assert_eq!(body.springs.len(), 1);
        assert!((body.springs.springs[0].stiffness - 100.0).abs() < 1e-3);
    }

    #[test]
    fn step_settles_a_stiff_rope_near_rest_length() {
        let mut body = VbdBody::with_default_config();
        let top = body.spawn_pinned(Vec3::ZERO);
        let bottom = body.spawn(Vec3::new(0.0, -1.0, 0.0), 1.0);
        assert!(body.connect(top, bottom, 1.0e6));
        for _ in 0..300 {
            body.step(1.0 / 60.0);
        }
        let (min, max) = body.bounds().unwrap();
        assert!(min.y.is_finite() && max.y.is_finite());
        let length = (body.particles.position(top).unwrap()
            - body.particles.position(bottom).unwrap())
        .length();
        assert!((length - 1.0).abs() < 0.05, "settled at {length}");
    }
}
