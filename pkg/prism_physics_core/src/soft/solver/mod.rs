//! The substep XPBD soft-body solver.
//!
//! [`SoftSolver`] advances a set of particles and constraints by one frame time
//! step using the substep XPBD scheme: the frame is split into equal substeps,
//! and each substep predicts positions under gravity, projects the constraints
//! for a few Gauss-Seidel iterations, then recovers velocities from the net
//! motion. Splitting the frame into many small substeps (rather than iterating
//! many times within one large step) is the key insight of substep XPBD: it
//! makes stiffness converge robustly and keeps behaviour largely independent of
//! the time step.
//!
//! The solver is stateless; all mutable state lives in the [`ParticleStorage`]
//! and [`ConstraintSet`] it is handed, so the same solver instance can advance
//! many soft bodies.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! substep XPBD loop (predict, project, recover velocities) is the standard
//! formulation published by Müller et al.

pub mod config;
pub mod contacts;
pub mod integrate;

pub use config::{SelfCollisionParams, SoftSolverConfig};
pub use contacts::SoftContacts;

use crate::math::scalar::Real;
use crate::soft::collision::{
    resolve_backstops, resolve_body_collisions, resolve_body_collisions_with_friction,
    resolve_self_collision, resolve_self_collision_with_friction,
};
use crate::soft::constraint::ConstraintSet;
use crate::soft::particle::ParticleStorage;

/// A stateless substep XPBD solver for soft bodies, cloth, and rope.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SoftSolver;

impl SoftSolver {
    /// Creates a solver. The solver holds no state; it is a zero-sized handle to
    /// the stepping logic.
    #[must_use]
    pub const fn new() -> SoftSolver {
        SoftSolver
    }

    /// Advances `particles` under `constraints` by `dt` seconds using the
    /// substep XPBD scheme configured by `config`, with no external contacts.
    ///
    /// This is [`step_with_contacts`](Self::step_with_contacts) with an empty
    /// [`SoftContacts`] bundle, so only the internal constraints and the
    /// optional self-collision pass run. Does nothing when there are no
    /// particles or when `dt` is non-positive; the substep and iteration counts
    /// are clamped to at least `1`.
    pub fn step(
        &self,
        particles: &mut ParticleStorage,
        constraints: &mut ConstraintSet,
        config: &SoftSolverConfig,
        dt: Real,
    ) {
        self.step_with_contacts(particles, constraints, config, &SoftContacts::EMPTY, dt);
    }

    /// Advances `particles` under `constraints` by `dt` seconds, additionally
    /// resolving the per-frame external `contacts` (body-proxy colliders and
    /// per-particle backstops) each substep.
    ///
    /// Each substep predicts under gravity, projects the constraints, then runs
    /// the collide stage in a fixed order so later passes win where they
    /// overlap: optional self-collision (from `config`), then body-proxy
    /// collision, then backstops. Running the body passes last keeps the
    /// garment out of the animated body and off its backstop planes at the end
    /// of the substep. Finally velocities are recovered from the net motion, so
    /// a particle dragged along by a moving collider or backstop keeps the
    /// implied velocity.
    ///
    /// Does nothing when there are no particles or when `dt` is non-positive;
    /// the substep and iteration counts are clamped to at least `1`. An empty
    /// `contacts` bundle makes this identical to [`step`](Self::step).
    pub fn step_with_contacts(
        &self,
        particles: &mut ParticleStorage,
        constraints: &mut ConstraintSet,
        config: &SoftSolverConfig,
        contacts: &SoftContacts<'_>,
        dt: Real,
    ) {
        if particles.is_empty() || dt <= 0.0 {
            return;
        }
        let substeps = config.substeps.max(1);
        let iterations = config.iterations.max(1);
        let h = dt / substeps as Real;
        if h <= 0.0 {
            return;
        }
        for _ in 0..substeps {
            let mut columns = particles.columns_mut();
            integrate::predict(&mut columns, config.gravity, config.damping, h);
            constraints.reset();
            for _ in 0..iterations {
                constraints.project(columns.positions, columns.inverse_masses, h);
            }
            if let Some(contact) = config.self_collision {
                if contact.friction > 0.0 {
                    resolve_self_collision_with_friction(
                        columns.positions,
                        columns.prev_positions,
                        columns.inverse_masses,
                        contact.cell_size,
                        contact.thickness,
                        contact.friction,
                    );
                } else {
                    resolve_self_collision(
                        columns.positions,
                        columns.inverse_masses,
                        contact.cell_size,
                        contact.thickness,
                    );
                }
            }
            if !contacts.body_colliders.is_empty() {
                if contacts.body_friction > 0.0 {
                    resolve_body_collisions_with_friction(
                        columns.positions,
                        columns.prev_positions,
                        columns.inverse_masses,
                        contacts.body_colliders,
                        contacts.body_friction,
                    );
                } else {
                    resolve_body_collisions(
                        columns.positions,
                        columns.inverse_masses,
                        contacts.body_colliders,
                    );
                }
            }
            if !contacts.backstops.is_empty() {
                resolve_backstops(
                    columns.positions,
                    columns.inverse_masses,
                    contacts.backstops,
                );
            }
            integrate::finalize_velocities(&mut columns, h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::collision::{Backstop, BodyCollider};
    use crate::soft::constraint::DistanceConstraint;
    use crate::soft::particle::ParticleHandle;
    use glam::Vec3;

    #[test]
    fn empty_particles_is_a_no_op() {
        let solver = SoftSolver::new();
        let mut particles = ParticleStorage::new();
        let mut constraints = ConstraintSet::new();
        solver.step(
            &mut particles,
            &mut constraints,
            &SoftSolverConfig::default(),
            1.0 / 60.0,
        );
        assert!(particles.is_empty());
    }

    #[test]
    fn non_positive_dt_is_a_no_op() {
        let solver = SoftSolver::new();
        let mut particles = ParticleStorage::new();
        let p = particles.spawn(Vec3::ZERO, 1.0);
        let mut constraints = ConstraintSet::new();
        solver.step(
            &mut particles,
            &mut constraints,
            &SoftSolverConfig::default(),
            0.0,
        );
        assert_eq!(particles.position(p), Some(Vec3::ZERO));
    }

    #[test]
    fn free_particle_falls_under_gravity() {
        let solver = SoftSolver::new();
        let mut particles = ParticleStorage::new();
        let p = particles.spawn(Vec3::ZERO, 1.0);
        let mut constraints = ConstraintSet::new();
        let config = SoftSolverConfig {
            damping: 0.0,
            ..SoftSolverConfig::default()
        };
        solver.step(&mut particles, &mut constraints, &config, 1.0 / 60.0);
        let pos = particles.position(p).unwrap();
        assert!(pos.y < 0.0, "expected downward motion, got {pos:?}");
        assert!(pos.x.abs() < 1e-6 && pos.z.abs() < 1e-6);
    }

    #[test]
    fn pinned_particle_holds_dynamic_neighbour_within_rest_length() {
        // A two-particle rope: top pinned, bottom hangs under gravity but the
        // distance constraint keeps them within (roughly) the rest length.
        let solver = SoftSolver::new();
        let mut particles = ParticleStorage::new();
        let top = particles.spawn_pinned(Vec3::ZERO);
        let bottom = particles.spawn(Vec3::new(0.0, -1.0, 0.0), 1.0);
        let mut constraints = ConstraintSet::new();
        constraints
            .distance
            .push(DistanceConstraint::new(top, bottom, 1.0, 0.0));
        let config = SoftSolverConfig::default();
        for _ in 0..240 {
            solver.step(&mut particles, &mut constraints, &config, 1.0 / 60.0);
        }
        assert_eq!(particles.position(top), Some(Vec3::ZERO));
        let length =
            (particles.position(top).unwrap() - particles.position(bottom).unwrap()).length();
        assert!((length - 1.0).abs() < 0.05, "rope stretched to {length}");
    }

    #[test]
    fn settled_rope_is_deterministic() {
        let run = || {
            let solver = SoftSolver::new();
            let mut particles = ParticleStorage::new();
            let top = particles.spawn_pinned(Vec3::ZERO);
            let bottom = particles.spawn(Vec3::new(0.0, -1.0, 0.0), 1.0);
            let mut constraints = ConstraintSet::new();
            constraints
                .distance
                .push(DistanceConstraint::new(top, bottom, 1.0, 0.0));
            let config = SoftSolverConfig::default();
            for _ in 0..120 {
                solver.step(&mut particles, &mut constraints, &config, 1.0 / 60.0);
            }
            particles.position(bottom).unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn self_collision_disabled_by_default_lets_particles_overlap() {
        // Two coincident free particles with no constraints: with the collide
        // stage off (the default), nothing pushes them apart.
        let solver = SoftSolver::new();
        let mut particles = ParticleStorage::new();
        particles.spawn(Vec3::ZERO, 1.0);
        particles.spawn(Vec3::ZERO, 1.0);
        let mut constraints = ConstraintSet::new();
        let config = SoftSolverConfig {
            gravity: Vec3::ZERO,
            self_collision: None,
            ..SoftSolverConfig::default()
        };
        solver.step(&mut particles, &mut constraints, &config, 1.0 / 60.0);
        let gap = (particles.position(ParticleHandle::from_index(0)).unwrap()
            - particles.position(ParticleHandle::from_index(1)).unwrap())
        .length();
        assert!(gap < 1e-6, "particles drifted apart without a collide stage: {gap}");
    }

    #[test]
    fn self_collision_stage_pushes_overlapping_particles_to_thickness() {
        // Same pair, but with the collide stage enabled: the substep's collide
        // pass must separate them to (at least) the contact thickness.
        let solver = SoftSolver::new();
        let mut particles = ParticleStorage::new();
        particles.spawn(Vec3::new(0.0, 0.0, 0.0), 1.0);
        particles.spawn(Vec3::new(0.1, 0.0, 0.0), 1.0);
        let mut constraints = ConstraintSet::new();
        let config = SoftSolverConfig {
            gravity: Vec3::ZERO,
            damping: 0.0,
            self_collision: Some(SelfCollisionParams::new(1.0, 1.0)),
            ..SoftSolverConfig::default()
        };
        solver.step(&mut particles, &mut constraints, &config, 1.0 / 60.0);
        let gap = (particles.position(ParticleHandle::from_index(0)).unwrap()
            - particles.position(ParticleHandle::from_index(1)).unwrap())
        .length();
        assert!(gap >= 1.0 - 1e-5, "pair not separated to thickness: {gap}");
    }

    #[test]
    fn self_collision_stage_is_deterministic() {
        let run = || {
            let solver = SoftSolver::new();
            let mut particles = ParticleStorage::new();
            particles.spawn(Vec3::new(0.0, 0.0, 0.0), 1.0);
            particles.spawn(Vec3::new(0.2, 0.1, 0.0), 1.0);
            particles.spawn(Vec3::new(0.1, 0.2, 0.1), 1.0);
            let mut constraints = ConstraintSet::new();
            let config = SoftSolverConfig {
                self_collision: Some(SelfCollisionParams::new(1.0, 0.5).with_friction(0.3)),
                ..SoftSolverConfig::default()
            };
            for _ in 0..30 {
                solver.step(&mut particles, &mut constraints, &config, 1.0 / 60.0);
            }
            particles.position(ParticleHandle::from_index(2)).unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn step_with_empty_contacts_matches_plain_step() {
        let solver = SoftSolver::new();
        let config = SoftSolverConfig::default();
        let build = || {
            let mut particles = ParticleStorage::new();
            let top = particles.spawn_pinned(Vec3::ZERO);
            let bottom = particles.spawn(Vec3::new(0.0, -1.0, 0.0), 1.0);
            let mut constraints = ConstraintSet::new();
            constraints
                .distance
                .push(DistanceConstraint::new(top, bottom, 1.0, 0.0));
            (particles, constraints, bottom)
        };
        let (mut pa, mut ca, ba) = build();
        let (mut pb, mut cb, bb) = build();
        for _ in 0..60 {
            solver.step(&mut pa, &mut ca, &config, 1.0 / 60.0);
            solver.step_with_contacts(&mut pb, &mut cb, &config, &SoftContacts::EMPTY, 1.0 / 60.0);
        }
        assert_eq!(pa.position(ba).unwrap(), pb.position(bb).unwrap());
    }

    #[test]
    fn body_collider_keeps_particle_out_of_sphere() {
        // A free particle falling onto a sphere centred below it must come to
        // rest on (or above) the sphere surface, never inside it.
        let solver = SoftSolver::new();
        let mut particles = ParticleStorage::new();
        let p = particles.spawn(Vec3::new(0.0, 1.2, 0.0), 1.0);
        let mut constraints = ConstraintSet::new();
        let config = SoftSolverConfig::default();
        let colliders = [BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        let contacts = SoftContacts::new(&colliders, &[]);
        for _ in 0..240 {
            solver.step_with_contacts(&mut particles, &mut constraints, &config, &contacts, 1.0 / 60.0);
        }
        let pos = particles.position(p).unwrap();
        assert!(
            pos.length() >= 1.0 - 1e-3,
            "particle sank into the sphere: {pos:?} (|pos| = {})",
            pos.length()
        );
    }

    #[test]
    fn backstop_holds_particle_on_front_side() {
        // A particle pulled toward -Y by gravity, with a backstop plane facing
        // +Y anchored at the origin (zero slack), must not sink below y = 0.
        let solver = SoftSolver::new();
        let mut particles = ParticleStorage::new();
        let p = particles.spawn(Vec3::new(0.0, 0.5, 0.0), 1.0);
        let mut constraints = ConstraintSet::new();
        let config = SoftSolverConfig::default();
        let backstops = [Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::Y,
            distance: 0.0,
        }];
        let contacts = SoftContacts::new(&[], &backstops);
        for _ in 0..240 {
            solver.step_with_contacts(&mut particles, &mut constraints, &config, &contacts, 1.0 / 60.0);
        }
        let pos = particles.position(p).unwrap();
        assert!(pos.y >= -1e-3, "particle sank behind its backstop: {pos:?}");
    }

    #[test]
    fn step_with_contacts_is_deterministic() {
        let colliders = [BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        let run = || {
            let solver = SoftSolver::new();
            let mut particles = ParticleStorage::new();
            particles.spawn(Vec3::new(0.0, 1.2, 0.0), 1.0);
            particles.spawn(Vec3::new(0.3, 1.0, 0.1), 1.0);
            let mut constraints = ConstraintSet::new();
            let config = SoftSolverConfig::default();
            let contacts = SoftContacts::new(&colliders, &[]).with_body_friction(0.3);
            for _ in 0..60 {
                solver.step_with_contacts(&mut particles, &mut constraints, &config, &contacts, 1.0 / 60.0);
            }
            particles.position(ParticleHandle::from_index(0)).unwrap()
        };
        assert_eq!(run(), run());
    }
}
