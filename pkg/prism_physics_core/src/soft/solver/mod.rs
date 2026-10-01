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

pub use config::{SelfCollisionParams, SoftSolverConfig, VirtualSelfCollisionParams};
pub use contacts::SoftContacts;

use crate::math::scalar::Real;
use crate::soft::collision::{
    resolve_backstops, resolve_body_collisions, resolve_body_collisions_with_friction, resolve_ccd,
    resolve_self_ccd, resolve_self_collision, resolve_self_collision_virtual,
    resolve_self_collision_virtual_augment, resolve_self_collision_with_friction,
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
    /// overlap: optional discrete self-collision, then optional `NvCloth`-style
    /// virtual-particle self-collision (gated by
    /// [`virtual_self_collision`](SoftSolverConfig::virtual_self_collision) with
    /// the per-frame
    /// [`contacts.virtual_particles`](SoftContacts::virtual_particles); run in
    /// augment mode when discrete self-collision is also on so it preserves that
    /// pass's friction), then optional continuous self-collision (self-CCD), all
    /// in the self-collision tier; then discrete body-proxy collision then
    /// optional continuous body collision (CCD) against the same `contacts`
    /// colliders; then backstops. Running the body passes last keeps
    /// the garment out of the animated body and off its backstop planes at the
    /// end of the substep, and the continuous passes close the thin-sheet or
    /// fast-proxy tunneling gaps their discrete counterparts can miss. Finally
    /// velocities are recovered from the net motion, so a particle dragged along
    /// by a moving collider or backstop keeps the implied velocity.
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
            if let Some(vp_params) = config.virtual_self_collision
                && !contacts.virtual_particles.is_empty()
            {
                if config.self_collision.is_some() {
                    resolve_self_collision_virtual_augment(
                        columns.positions,
                        columns.inverse_masses,
                        contacts.virtual_particles,
                        vp_params.cell_size,
                        vp_params.thickness,
                    );
                } else {
                    resolve_self_collision_virtual(
                        columns.positions,
                        columns.inverse_masses,
                        contacts.virtual_particles,
                        vp_params.cell_size,
                        vp_params.thickness,
                    );
                }
            }
            if let Some(params) = config.self_ccd {
                resolve_self_ccd(
                    columns.positions,
                    columns.prev_positions,
                    columns.velocities,
                    columns.inverse_masses,
                    params,
                    h,
                );
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
            if let Some(params) = config.ccd
                && !contacts.body_colliders.is_empty()
            {
                resolve_ccd(
                    columns.positions,
                    columns.prev_positions,
                    columns.velocities,
                    columns.inverse_masses,
                    contacts.body_colliders,
                    params,
                    h,
                    contacts.body_friction,
                );
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
    use crate::soft::collision::{
        Backstop, BodyCollider, CcdParams, SelfCcdParams, VirtualParticlePattern,
        generate_virtual_particles,
    };
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

    #[test]
    fn ccd_catches_fast_tunneling_through_sphere() {
        // One large substep: a particle moving fast enough to cross a small
        // sphere in a single step starts and ends outside it, so the discrete
        // body pass misses it. The CCD sweep must catch the crossing.
        let build = || {
            let mut particles = ParticleStorage::new();
            let p = particles.spawn(Vec3::new(-2.0, 0.0, 0.0), 1.0);
            particles.set_velocity(p, Vec3::new(240.0, 0.0, 0.0));
            (particles, p)
        };
        let colliders = [BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 0.5,
        }];
        let contacts = SoftContacts::new(&colliders, &[]);
        let base = SoftSolverConfig {
            gravity: Vec3::ZERO,
            substeps: 1,
            iterations: 1,
            damping: 0.0,
            ..SoftSolverConfig::default()
        };
        let solver = SoftSolver::new();

        // Without CCD the particle tunnels clean through to the far side.
        let (mut no_ccd, p0) = build();
        let mut c0 = ConstraintSet::new();
        solver.step_with_contacts(&mut no_ccd, &mut c0, &base, &contacts, 1.0 / 60.0);
        assert!(
            no_ccd.position(p0).unwrap().x > 1.0,
            "expected tunneling without CCD, got {:?}",
            no_ccd.position(p0).unwrap()
        );

        // With CCD the particle is stopped at the entry surface of the sphere.
        let (mut with_ccd, p1) = build();
        let mut c1 = ConstraintSet::new();
        let cfg = SoftSolverConfig {
            ccd: Some(CcdParams::default()),
            ..base
        };
        solver.step_with_contacts(&mut with_ccd, &mut c1, &cfg, &contacts, 1.0 / 60.0);
        let pos = with_ccd.position(p1).unwrap();
        assert!(
            pos.length() >= 0.5 - 1e-3,
            "CCD let the particle inside the sphere: {pos:?}"
        );
        assert!(pos.x < 0.5, "CCD failed to stop the crossing: {pos:?}");
    }

    #[test]
    fn self_ccd_catches_fast_pair_tunneling() {
        // Two particles rushing through each other in one big substep: the
        // discrete self-collision pass (disabled here) would miss the crossing,
        // but self-CCD must keep them from swapping sides.
        let thickness = 0.5;
        let build = || {
            let mut particles = ParticleStorage::new();
            let a = particles.spawn(Vec3::new(-1.0, 0.0, 0.0), 1.0);
            let b = particles.spawn(Vec3::new(1.0, 0.0, 0.0), 1.0);
            particles.set_velocity(a, Vec3::new(180.0, 0.0, 0.0));
            particles.set_velocity(b, Vec3::new(-180.0, 0.0, 0.0));
            (particles, a, b)
        };
        let base = SoftSolverConfig {
            gravity: Vec3::ZERO,
            substeps: 1,
            iterations: 1,
            damping: 0.0,
            ..SoftSolverConfig::default()
        };
        let solver = SoftSolver::new();

        // Without self-CCD the pair swaps sides (a ends to the right of b).
        let (mut plain, a0, b0) = build();
        let mut c0 = ConstraintSet::new();
        solver.step(&mut plain, &mut c0, &base, 1.0 / 60.0);
        assert!(
            plain.position(a0).unwrap().x > plain.position(b0).unwrap().x,
            "expected the pair to tunnel past each other without self-CCD"
        );

        // With self-CCD the ordering is preserved and the gap respects thickness.
        let (mut swept, a1, b1) = build();
        let mut c1 = ConstraintSet::new();
        let cfg = SoftSolverConfig {
            self_ccd: Some(SelfCcdParams::new(thickness * 2.0, thickness)),
            ..base
        };
        solver.step(&mut swept, &mut c1, &cfg, 1.0 / 60.0);
        let xa = swept.position(a1).unwrap().x;
        let xb = swept.position(b1).unwrap().x;
        assert!(xa <= xb + 1e-4, "self-CCD let the pair swap sides: a={xa} b={xb}");
        assert!(
            (xb - xa) >= thickness - 1e-3,
            "self-CCD did not keep thickness gap: a={xa} b={xb}"
        );
    }

    #[test]
    fn disabled_continuous_passes_match_plain_step() {
        // self_ccd/ccd set to disabled params must be exact no-ops versus a
        // config that leaves them None.
        let solver = SoftSolver::new();
        let colliders = [BodyCollider::Sphere {
            center: Vec3::new(0.0, -2.0, 0.0),
            radius: 1.0,
        }];
        let contacts = SoftContacts::new(&colliders, &[]);
        let build = || {
            let mut particles = ParticleStorage::new();
            particles.spawn(Vec3::new(0.0, 1.0, 0.0), 1.0);
            particles.spawn(Vec3::new(0.2, 1.1, 0.0), 1.0);
            (particles, ConstraintSet::new())
        };
        let plain = SoftSolverConfig::default();
        let disabled = SoftSolverConfig {
            self_ccd: Some(SelfCcdParams::default()), // enabled == false by default
            ccd: Some(CcdParams {
                enabled: false,
                ..CcdParams::default()
            }),
            ..SoftSolverConfig::default()
        };
        let (mut pa, mut ca) = build();
        let (mut pb, mut cb) = build();
        for _ in 0..30 {
            solver.step_with_contacts(&mut pa, &mut ca, &plain, &contacts, 1.0 / 60.0);
            solver.step_with_contacts(&mut pb, &mut cb, &disabled, &contacts, 1.0 / 60.0);
        }
        assert_eq!(
            pa.position(ParticleHandle::from_index(0)).unwrap(),
            pb.position(ParticleHandle::from_index(0)).unwrap()
        );
        assert_eq!(
            pa.position(ParticleHandle::from_index(1)).unwrap(),
            pb.position(ParticleHandle::from_index(1)).unwrap()
        );
    }

    // A big triangle in the z=0 plane (corners pinned) plus a free intruder
    // hovering just above its centroid, inside `thickness` of the face but far
    // from every corner. Gravity is zeroed so the only motion is the collide
    // stage, isolating the virtual-particle tier.
    fn build_vertex_through_triangle() -> (ParticleStorage, ConstraintSet, ParticleHandle) {
        let mut particles = ParticleStorage::new();
        particles.spawn_pinned(Vec3::new(0.0, 0.0, 0.0));
        particles.spawn_pinned(Vec3::new(4.0, 0.0, 0.0));
        particles.spawn_pinned(Vec3::new(0.0, 4.0, 0.0));
        let intruder = particles.spawn(Vec3::new(4.0 / 3.0, 4.0 / 3.0, 0.05), 1.0);
        (particles, ConstraintSet::new(), intruder)
    }

    #[test]
    fn virtual_self_collision_catches_vertex_through_triangle() {
        let solver = SoftSolver::new();
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        let contacts = SoftContacts::EMPTY.with_virtual_particles(&virtuals);
        let base = SoftSolverConfig {
            gravity: Vec3::ZERO,
            ..SoftSolverConfig::default()
        };

        // Gate off: the intruder is far from every real corner, so the solver's
        // self-collision tier never touches it and it stays at z=0.05.
        let (mut off_p, mut off_c, off_h) = build_vertex_through_triangle();
        solver.step_with_contacts(&mut off_p, &mut off_c, &base, &contacts, 1.0 / 60.0);
        let off_z = off_p.position(off_h).unwrap().z;
        assert!(
            (off_z - 0.05).abs() < 1e-5,
            "no virtual gate should leave the intruder in place, got z={off_z}"
        );

        // Gate on: the centroid virtual particle sits under the intruder and the
        // solver pushes it back out along +z past the 0.2 contact thickness.
        let cfg = SoftSolverConfig {
            virtual_self_collision: Some(VirtualSelfCollisionParams::new(1.0, 0.2)),
            ..base
        };
        let (mut on_p, mut on_c, on_h) = build_vertex_through_triangle();
        solver.step_with_contacts(&mut on_p, &mut on_c, &cfg, &contacts, 1.0 / 60.0);
        let on_z = on_p.position(on_h).unwrap().z;
        assert!(
            on_z > 0.05 + 1e-4,
            "virtual gate should push the intruder out along +z, got z={on_z}"
        );
    }

    #[test]
    fn virtual_self_collision_noop_without_particles_or_gate() {
        // Enabling the gate with an empty virtual slice, and leaving the gate
        // None with virtual particles present, must both exactly match the plain
        // step with no virtual tier at all.
        let solver = SoftSolver::new();
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        let base = SoftSolverConfig {
            gravity: Vec3::ZERO,
            ..SoftSolverConfig::default()
        };

        let run = |cfg: &SoftSolverConfig, contacts: &SoftContacts<'_>| {
            let (mut p, mut c, h) = build_vertex_through_triangle();
            solver.step_with_contacts(&mut p, &mut c, cfg, contacts, 1.0 / 60.0);
            p.position(h).unwrap()
        };

        let plain = run(&base, &SoftContacts::EMPTY);

        // Gate on but no virtual particles supplied -> no-op.
        let gate_on = SoftSolverConfig {
            virtual_self_collision: Some(VirtualSelfCollisionParams::new(1.0, 0.2)),
            ..base
        };
        let gate_on_no_vps = run(&gate_on, &SoftContacts::EMPTY);
        assert_eq!(plain, gate_on_no_vps);

        // Virtual particles supplied but gate None -> no-op.
        let vps_no_gate = run(&base, &SoftContacts::EMPTY.with_virtual_particles(&virtuals));
        assert_eq!(plain, vps_no_gate);
    }
}
