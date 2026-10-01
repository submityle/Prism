//! Two-way rigid coupling between soft-body particles and rigid proxies.
//!
//! One-way body collision ([`super::resolve_body_collisions`]) only ever pushes
//! particles *out* of a rigid proxy; the body never notices. Production solvers
//! close the loop: a light prop resting on a hammock dents the cloth and the
//! cloth pushes the prop back up. The authoritative rigid-body integrator is a
//! separate concern, so this pass contributes only the contact half:
//!
//! * a **mass-weighted** contact that splits each push-out between the particle
//!   and the rigid proxy by inverse mass, so a light body is displaced far more
//!   than the heavy sheet it rests on, and
//! * the **reaction impulse** each body accumulates from the particles pressing
//!   on it, which a rigid-body integrator reads (and clears) to drive its own
//!   step.
//!
//! The proxy is a translatable [`BodyCollider`] plus an inverse mass; a zero
//! inverse mass is a kinematic body that never moves but still records the
//! reaction impulse (useful for force feedback and the "prop denting the cloth"
//! report). Coupling is linear only — angular response is the rigid-body
//! integrator's job. The pass is Jacobi (every particle correction for a body
//! uses that body's pre-pass pose, then the body is translated once), so it is
//! deterministic array-in / array-out, only uses `sqrt`, and skips degenerate
//! or out-of-contact cases instead of panicking.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! inverse-mass-weighted contact split and the Newton reaction impulse are
//! textbook position-based-dynamics / rigid-body contact mechanics.

use glam::Vec3;

use crate::math::scalar::Real;

use super::{BodyCollider, EPS_LEN_SQ};

/// A rigid proxy that couples both ways with the particles.
///
/// The proxy carries a [`BodyCollider`] shape, an `inverse_mass` (zero for a
/// kinematic/immovable body), and a running `reaction_impulse` accumulator that
/// [`resolve_two_way_coupling`] adds into. A rigid-body integrator is expected
/// to consume `reaction_impulse` each frame and reset it via
/// [`CouplingBody::clear_reaction`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingBody {
    /// The rigid shape particles collide against; translated in place by the
    /// pass.
    pub collider: BodyCollider,
    /// Inverse mass of the rigid proxy. Zero is a kinematic body that never
    /// moves but still records the reaction impulse.
    pub inverse_mass: Real,
    /// Accumulated impulse (momentum) the particles exerted on this body over
    /// the pass, for a rigid-body integrator to apply and then clear.
    pub reaction_impulse: Vec3,
}

impl CouplingBody {
    /// Creates a movable proxy with the given collider and inverse mass and a
    /// zeroed reaction accumulator. A negative inverse mass is clamped to zero.
    #[must_use]
    pub fn new(collider: BodyCollider, inverse_mass: Real) -> Self {
        Self {
            collider,
            inverse_mass: inverse_mass.max(0.0),
            reaction_impulse: Vec3::ZERO,
        }
    }

    /// Creates a kinematic proxy (zero inverse mass) that never moves but still
    /// records the reaction impulse the particles apply to it.
    #[must_use]
    pub fn kinematic(collider: BodyCollider) -> Self {
        Self {
            collider,
            inverse_mass: 0.0,
            reaction_impulse: Vec3::ZERO,
        }
    }

    /// Zeroes the accumulated reaction impulse; call after a rigid-body
    /// integrator has applied it.
    pub fn clear_reaction(&mut self) {
        self.reaction_impulse = Vec3::ZERO;
    }
}

/// Returns `collider` translated by `delta`.
///
/// Spheres and capsules move their centers/endpoints rigidly. A half-space is
/// translated by shifting its `offset` along the normal (`normal.dot(delta)`),
/// which is the plane displacement for a rigid translation; the normal itself
/// is unchanged. Half-spaces are normally kinematic ground planes, so this
/// branch is exercised only when an author gives one a non-zero inverse mass.
fn translate_collider(collider: BodyCollider, delta: Vec3) -> BodyCollider {
    match collider {
        BodyCollider::Sphere { center, radius } => BodyCollider::Sphere {
            center: center + delta,
            radius,
        },
        BodyCollider::Capsule { p0, p1, radius } => BodyCollider::Capsule {
            p0: p0 + delta,
            p1: p1 + delta,
            radius,
        },
        BodyCollider::HalfSpace { normal, offset } => BodyCollider::HalfSpace {
            normal,
            offset: offset + normal.dot(delta),
        },
    }
}

/// Resolves two-way particle/rigid contact for every particle against every
/// body.
///
/// Particle state is the standard structure-of-arrays pair: `positions` is
/// corrected in place and `inverse_masses` is read-only. For each contact the
/// push-out vector `c` (the correction that would move the particle fully to
/// the collider surface) is split by inverse mass: the particle moves
/// `w_particle / (w_particle + w_body)` of it and the body the complementary
/// share in the opposite direction. Each body accumulates the Newton reaction
/// impulse `-c / ((w_particle + w_body) * dt)` from every particle it touches.
/// The body is translated once per pass (Jacobi), so all of its particle
/// corrections see the same pre-pass pose and the result is order-independent
/// and deterministic.
///
/// A non-positive `dt`, an empty particle or body list, a contact where both
/// masses are infinite (`w_particle + w_body == 0`, e.g. a pinned particle
/// against a kinematic body), or a particle already outside the collider are
/// all no-ops. Particles beyond the shorter of the two slices are ignored.
pub fn resolve_two_way_coupling(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    bodies: &mut [CouplingBody],
    dt: Real,
) {
    let count = positions.len().min(inverse_masses.len());
    if dt <= 0.0 || count == 0 || bodies.is_empty() {
        return;
    }

    for body in bodies.iter_mut() {
        let collider = body.collider;
        let w_body = body.inverse_mass.max(0.0);
        // Jacobi accumulators against this body's pre-pass pose.
        let mut body_delta = Vec3::ZERO;
        let mut impulse = Vec3::ZERO;

        for i in 0..count {
            let w_particle = inverse_masses[i].max(0.0);
            let w_sum = w_particle + w_body;
            if w_sum <= 0.0 {
                continue;
            }
            let projected = collider.project(positions[i]);
            let correction = projected - positions[i];
            if correction.length_squared() <= EPS_LEN_SQ {
                continue;
            }
            // Mass-weighted split: the lighter side moves more.
            positions[i] += correction * (w_particle / w_sum);
            body_delta += correction * (-(w_body / w_sum));
            // Newton reaction on the body (momentum), opposing the push.
            impulse += correction * (-(1.0 / (w_sum * dt)));
        }

        if w_body > 0.0 && body_delta.length_squared() > EPS_LEN_SQ {
            body.collider = translate_collider(collider, body_delta);
        }
        body.reaction_impulse += impulse;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit sphere collider centered at the origin.
    fn unit_sphere() -> BodyCollider {
        BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }
    }

    #[test]
    fn kinematic_body_holds_still_and_pushes_particle_fully_out() {
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0];
        let mut bodies = [CouplingBody::kinematic(unit_sphere())];
        resolve_two_way_coupling(&mut positions, &inverse_masses, &mut bodies, 1.0 / 60.0);
        assert!((positions[0].x - 1.0).abs() < 1e-5);
        let BodyCollider::Sphere { center, .. } = bodies[0].collider else {
            panic!("collider shape changed");
        };
        assert!(center.distance(Vec3::ZERO) < 1e-9);
        assert!(bodies[0].reaction_impulse.x < -1e-6);
    }

    #[test]
    fn light_body_moves_more_than_heavy_particle() {
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [0.1];
        let mut bodies = [CouplingBody::new(unit_sphere(), 10.0)];
        resolve_two_way_coupling(&mut positions, &inverse_masses, &mut bodies, 1.0 / 60.0);
        let particle_move = (positions[0].x - 0.5).abs();
        let BodyCollider::Sphere { center, .. } = bodies[0].collider else {
            panic!("collider shape changed");
        };
        let body_move = center.distance(Vec3::ZERO);
        assert!(
            body_move > particle_move * 50.0,
            "body {body_move} particle {particle_move}"
        );
    }

    #[test]
    fn equal_mass_splits_the_correction() {
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0];
        let mut bodies = [CouplingBody::new(unit_sphere(), 1.0)];
        resolve_two_way_coupling(&mut positions, &inverse_masses, &mut bodies, 1.0 / 60.0);
        assert!((positions[0].x - 0.75).abs() < 1e-5);
        let BodyCollider::Sphere { center, .. } = bodies[0].collider else {
            panic!("collider shape changed");
        };
        assert!((center.x - (-0.25)).abs() < 1e-5);
    }

    #[test]
    fn reaction_impulse_opposes_the_push() {
        let mut positions = [Vec3::new(0.0, 0.5, 0.0)];
        let inverse_masses = [1.0];
        let mut bodies = [CouplingBody::new(unit_sphere(), 1.0)];
        resolve_two_way_coupling(&mut positions, &inverse_masses, &mut bodies, 1.0 / 60.0);
        assert!(bodies[0].reaction_impulse.y < -1e-6);
        assert!(bodies[0].reaction_impulse.x.abs() < 1e-6);
        assert!(bodies[0].reaction_impulse.z.abs() < 1e-6);
    }

    #[test]
    fn no_contact_leaves_everything_untouched() {
        let mut positions = [Vec3::new(3.0, 0.0, 0.0)];
        let inverse_masses = [1.0];
        let mut bodies = [CouplingBody::new(unit_sphere(), 1.0)];
        resolve_two_way_coupling(&mut positions, &inverse_masses, &mut bodies, 1.0 / 60.0);
        assert!((positions[0].x - 3.0).abs() < 1e-9);
        let BodyCollider::Sphere { center, .. } = bodies[0].collider else {
            panic!("collider shape changed");
        };
        assert!(center.distance(Vec3::ZERO) < 1e-9);
        assert!(bodies[0].reaction_impulse.length_squared() < 1e-18);
    }

    #[test]
    fn non_positive_dt_is_a_noop() {
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0];
        let mut bodies = [CouplingBody::new(unit_sphere(), 1.0)];
        resolve_two_way_coupling(&mut positions, &inverse_masses, &mut bodies, 0.0);
        assert!((positions[0].x - 0.5).abs() < 1e-9);
        assert!(bodies[0].reaction_impulse.length_squared() < 1e-18);
    }

    #[test]
    fn pinned_particle_against_kinematic_body_is_skipped() {
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [0.0];
        let mut bodies = [CouplingBody::kinematic(unit_sphere())];
        resolve_two_way_coupling(&mut positions, &inverse_masses, &mut bodies, 1.0 / 60.0);
        assert!((positions[0].x - 0.5).abs() < 1e-9);
        assert!(bodies[0].reaction_impulse.length_squared() < 1e-18);
    }

    #[test]
    fn pinned_particle_moves_only_the_body() {
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [0.0];
        let mut bodies = [CouplingBody::new(unit_sphere(), 1.0)];
        resolve_two_way_coupling(&mut positions, &inverse_masses, &mut bodies, 1.0 / 60.0);
        assert!((positions[0].x - 0.5).abs() < 1e-9);
        let BodyCollider::Sphere { center, .. } = bodies[0].collider else {
            panic!("collider shape changed");
        };
        assert!((center.x - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn capsule_body_translates_rigidly() {
        let capsule = BodyCollider::Capsule {
            p0: Vec3::new(0.0, -1.0, 0.0),
            p1: Vec3::new(0.0, 1.0, 0.0),
            radius: 1.0,
        };
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0];
        let mut bodies = [CouplingBody::new(capsule, 1.0)];
        resolve_two_way_coupling(&mut positions, &inverse_masses, &mut bodies, 1.0 / 60.0);
        let BodyCollider::Capsule { p0, p1, .. } = bodies[0].collider else {
            panic!("collider shape changed");
        };
        assert!((p0.x - p1.x).abs() < 1e-9);
        assert!(p0.x < -1e-6);
        assert!((p0.y - (-1.0)).abs() < 1e-9);
        assert!((p1.y - 1.0).abs() < 1e-9);
    }

    #[test]
    fn clear_reaction_zeroes_the_accumulator() {
        let mut body = CouplingBody::new(unit_sphere(), 1.0);
        body.reaction_impulse = Vec3::new(1.0, 2.0, 3.0);
        body.clear_reaction();
        assert!(body.reaction_impulse.length_squared() < 1e-18);
    }

    #[test]
    fn coupling_is_deterministic() {
        let build = || {
            [
                Vec3::new(0.5, 0.0, 0.0),
                Vec3::new(0.0, 0.4, 0.0),
                Vec3::new(-0.3, 0.0, 0.2),
            ]
        };
        let inverse_masses = [1.0, 1.0, 1.0];
        let mut a = build();
        let mut b = build();
        let mut bodies_a = [CouplingBody::new(unit_sphere(), 1.0)];
        let mut bodies_b = [CouplingBody::new(unit_sphere(), 1.0)];
        resolve_two_way_coupling(&mut a, &inverse_masses, &mut bodies_a, 1.0 / 60.0);
        resolve_two_way_coupling(&mut b, &inverse_masses, &mut bodies_b, 1.0 / 60.0);
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert!(pa.distance(*pb) < 1e-9);
        }
        assert!(
            bodies_a[0]
                .reaction_impulse
                .distance(bodies_b[0].reaction_impulse)
                < 1e-9
        );
    }

    #[test]
    fn mismatched_slice_lengths_use_the_shorter() {
        // Three positions but only one inverse mass: only particle 0 is touched.
        let mut positions = [
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.4, 0.0, 0.0),
            Vec3::new(0.3, 0.0, 0.0),
        ];
        let inverse_masses = [1.0];
        let mut bodies = [CouplingBody::kinematic(unit_sphere())];
        resolve_two_way_coupling(&mut positions, &inverse_masses, &mut bodies, 1.0 / 60.0);
        assert!((positions[0].x - 1.0).abs() < 1e-5);
        assert!((positions[1].x - 0.4).abs() < 1e-9);
        assert!((positions[2].x - 0.3).abs() < 1e-9);
    }
}
