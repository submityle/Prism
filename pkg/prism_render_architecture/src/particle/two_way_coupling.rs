//! Particle<->rigid-body two-way coupling `CPU` gold standard (design §10).
//!
//! Design §10 lists *two-way coupling* ("两向耦合") as the high-end feature where
//! fluid / cloth / debris particles push back on the rigid bodies they touch,
//! "经 §23 少量回读或 GPU 侧共享缓冲对接 `physics_core`". A one-way coupling only
//! lets the particle *read* the body (sample its surface velocity and get
//! pushed); a two-way coupling additionally feeds the equal-and-opposite
//! reaction back into the body so the closed particle+body system conserves
//! linear momentum. This module owns the `CPU` reference for that exchange so a
//! future `GPU` compute kernel (or the shared `prism_physics_core` solver) can
//! match it bit for bit.
//!
//! The exchange is the textbook rigid-body contact-impulse resolution used by
//! position-based dynamics — Müller et al., "Detailed Rigid Body Simulation
//! with Extended Position Based Dynamics" (2020) — and is the same impulse
//! algebra `prism_physics_core`'s `solver::xpbd::rigid` applies internally
//! (`generalized_inverse_mass` / `apply_velocity_impulse`). It contains no
//! Unreal / Houdini / `EmberGen` source or derived code.
//!
//! # The exchange
//!
//! For a point particle touching a body at a world contact point, with a unit
//! contact `normal` pointing from the body toward the particle:
//!
//! 1. The *relative normal velocity* `vn = (v_particle - v_body_at_contact) . n`
//!    measures how fast the particle approaches the body along the normal, where
//!    the body's surface velocity at the contact is
//!    `v_linear + omega x (contact - center_of_mass)`
//!    (see [`body_point_velocity`]). This is the "particle reads the body" half.
//! 2. The *effective inverse mass* along the normal,
//!    `w = inv_mass_particle + inv_mass_body
//!         + (r x n) . I_inv_world (r x n)`, folds in the body's rotational
//!    response at the lever arm `r = contact - center_of_mass`
//!    (see [`generalized_inverse_mass`]).
//! 3. The contact impulse magnitude is `j = -(1 + e) * vn / w` for a restitution
//!    `e` in `0..=1`; a *separating* contact (`vn >= 0`) yields the zero impulse
//!    identity (see [`coupling_impulse`]).
//! 4. The impulse `J = j * n` is added to the particle while the equal-and-
//!    opposite reaction `-J` is applied to the body *at the same contact point*
//!    (see [`apply_coupling`]). Because both impulses act at one world point, the
//!    net momentum injected into the particle+body system is exactly zero: linear
//!    momentum is conserved, which the gold-standard tests verify to `f32`
//!    tolerance.
//!
//! # Determinism
//!
//! Matching the sibling particle modules, the only floating-point primitive
//! beyond ordinary `+ - * /` arithmetic is `sqrt` (reached through [`Vec3`]'s
//! normalization and length). There are no transcendental calls, no hashing, and
//! no randomness in the solver itself, so the result is bit-reproducible against
//! a `GPU` kernel. Degenerate inputs (zero-length normal, a wholly static
//! particle+body pair) collapse to the zero impulse instead of dividing by zero.

use super::{Vec3, EPS_LEN_SQ};

/// Effective-inverse-mass floor below which a particle+body pair is treated as
/// jointly immovable (both masses infinite), so the impulse divide collapses to
/// the zero impulse instead of producing a `NaN`.
const MIN_EFFECTIVE_INV_MASS: f32 = 1e-12;

/// A column-major 3x3 matrix built from three [`Vec3`] columns.
///
/// Only the handful of operations the coupling needs are provided: a
/// matrix-vector product, a transpose, a matrix-matrix product, and the
/// constructors used to assemble a world-space inverse inertia tensor. The
/// crate is dependency-free, so (like [`Vec3`]) the math is spelled out here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    /// First (x) column.
    pub col_x: Vec3,
    /// Second (y) column.
    pub col_y: Vec3,
    /// Third (z) column.
    pub col_z: Vec3,
}

impl Mat3 {
    /// The identity matrix (unit inverse inertia: a body that spins freely with
    /// unit response about every axis).
    pub const IDENTITY: Self = Self {
        col_x: Vec3::new(1.0, 0.0, 0.0),
        col_y: Vec3::new(0.0, 1.0, 0.0),
        col_z: Vec3::new(0.0, 0.0, 1.0),
    };

    /// Builds a matrix from three explicit columns.
    #[must_use]
    pub const fn from_columns(col_x: Vec3, col_y: Vec3, col_z: Vec3) -> Self {
        Self {
            col_x,
            col_y,
            col_z,
        }
    }

    /// Builds a diagonal matrix from `d`, used for a principal-frame (axis-
    /// aligned) inverse inertia tensor.
    #[must_use]
    pub const fn from_diagonal(d: Vec3) -> Self {
        Self {
            col_x: Vec3::new(d.x, 0.0, 0.0),
            col_y: Vec3::new(0.0, d.y, 0.0),
            col_z: Vec3::new(0.0, 0.0, d.z),
        }
    }

    /// Matrix-vector product `self * v`.
    #[must_use]
    pub fn mul_vec3(self, v: Vec3) -> Vec3 {
        self.col_x
            .scale(v.x)
            .add(self.col_y.scale(v.y))
            .add(self.col_z.scale(v.z))
    }

    /// Transpose (rows become columns).
    #[must_use]
    pub fn transpose(self) -> Self {
        Self {
            col_x: Vec3::new(self.col_x.x, self.col_y.x, self.col_z.x),
            col_y: Vec3::new(self.col_x.y, self.col_y.y, self.col_z.y),
            col_z: Vec3::new(self.col_x.z, self.col_y.z, self.col_z.z),
        }
    }

    /// Matrix-matrix product `self * rhs`, column by column.
    #[must_use]
    pub fn mul_mat3(self, rhs: Self) -> Self {
        Self {
            col_x: self.mul_vec3(rhs.col_x),
            col_y: self.mul_vec3(rhs.col_y),
            col_z: self.mul_vec3(rhs.col_z),
        }
    }
}

/// Builds the world-space inverse inertia tensor from the body's principal-frame
/// inverse inertia and its orientation as a rotation matrix.
///
/// `principal_inv_inertia` is the diagonal inverse inertia in the body's own
/// (principal) frame; `rotation` has the body's principal axes as its columns.
/// The world tensor is `R * diag(principal_inv_inertia) * R^T`, matching the
/// `prism_physics_core` `inv_inertia_world` helper (which does the same with a
/// quaternion-derived rotation).
#[must_use]
pub fn inv_inertia_world(principal_inv_inertia: Vec3, rotation: Mat3) -> Mat3 {
    // R * diag(principal_inv_inertia): scale each rotation column by the matching
    // principal entry.
    let scaled = Mat3::from_columns(
        rotation.col_x.scale(principal_inv_inertia.x),
        rotation.col_y.scale(principal_inv_inertia.y),
        rotation.col_z.scale(principal_inv_inertia.z),
    );
    scaled.mul_mat3(rotation.transpose())
}

/// Returns the generalized inverse mass of a body for a unit `direction` applied
/// at world lever arm `r` (the contact offset from the center of mass).
///
/// This is `inv_mass + (r x direction) . I_inv_world (r x direction)`, the
/// effective inverse mass a constraint acting along `direction` sees once the
/// body's rotational response is folded in. Identical in form to the
/// `prism_physics_core` helper of the same name.
#[must_use]
pub fn generalized_inverse_mass(
    inv_mass: f32,
    inv_inertia_world: Mat3,
    r: Vec3,
    direction: Vec3,
) -> f32 {
    let rn = r.cross(direction);
    inv_mass + rn.dot(inv_inertia_world.mul_vec3(rn))
}

/// A point particle participating in the coupling, in world space.
///
/// Momentum lives in `inv_mass` form (`inv_mass = 1 / mass`) to match the
/// rigid-body impulse algebra; `inv_mass = 0` models an immovable (infinite-mass)
/// particle. Position doubles as the contact point for a point particle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingParticle {
    /// Inverse mass (`1 / mass`); `0` is an immovable particle.
    pub inv_mass: f32,
    /// World-space position, also used as the contact point.
    pub position: Vec3,
    /// World-space linear velocity.
    pub velocity: Vec3,
}

impl CouplingParticle {
    /// Builds a particle from its inverse mass, position, and velocity.
    #[must_use]
    pub const fn new(inv_mass: f32, position: Vec3, velocity: Vec3) -> Self {
        Self {
            inv_mass,
            position,
            velocity,
        }
    }
}

/// A rigid body participating in the coupling, in world space.
///
/// `inv_inertia_world` is the world-space inverse inertia tensor (see
/// [`inv_inertia_world`]); `inv_mass = 0` models a static (infinite-mass) body
/// such as the floor or a kinematic collider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingBody {
    /// Inverse mass (`1 / mass`); `0` is a static body.
    pub inv_mass: f32,
    /// World-space inverse inertia tensor.
    pub inv_inertia_world: Mat3,
    /// World-space center of mass (the lever-arm origin).
    pub center_of_mass: Vec3,
    /// World-space linear velocity of the center of mass.
    pub linear_velocity: Vec3,
    /// World-space angular velocity.
    pub angular_velocity: Vec3,
}

impl CouplingBody {
    /// Builds a body from its inverse mass, world inverse inertia tensor, center
    /// of mass, and linear / angular velocities.
    #[must_use]
    pub const fn new(
        inv_mass: f32,
        inv_inertia_world: Mat3,
        center_of_mass: Vec3,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
    ) -> Self {
        Self {
            inv_mass,
            inv_inertia_world,
            center_of_mass,
            linear_velocity,
            angular_velocity,
        }
    }
}

/// Returns the body's surface velocity at a world `point`.
///
/// A rigid body's velocity at an offset is `v_linear + omega x r`, where
/// `r = point - center_of_mass`. This is the quantity the particle "reads" from
/// the body in the coupling's first half.
#[must_use]
pub fn body_point_velocity(body: &CouplingBody, point: Vec3) -> Vec3 {
    let r = point.sub(body.center_of_mass);
    body.linear_velocity.add(body.angular_velocity.cross(r))
}

/// Computes the contact impulse the body applies *to the particle* along the
/// unit contact `normal`, resolving their relative approach with restitution
/// `restitution` (clamped to `0..=1`).
///
/// The returned impulse `J = j * normal` is the vector added to the particle's
/// momentum; the body receives the reaction `-J` (see [`apply_coupling`]). The
/// result is the zero vector for a *separating* contact (relative normal
/// velocity `>= 0`), a degenerate (near-zero-length) `normal`, or a jointly
/// immovable pair (effective inverse mass at or below
/// [`MIN_EFFECTIVE_INV_MASS`]) — none of which should perturb the state.
#[must_use]
pub fn coupling_impulse(
    particle: &CouplingParticle,
    body: &CouplingBody,
    normal: Vec3,
    restitution: f32,
) -> Vec3 {
    // A degenerate normal carries no contact direction: no impulse.
    if normal.length_squared() <= EPS_LEN_SQ {
        return Vec3::ZERO;
    }
    let n = normal.normalize_or_zero();
    let contact = particle.position;
    let r = contact.sub(body.center_of_mass);

    // Relative velocity of the particle with respect to the body surface at the
    // contact, projected onto the normal.
    let relative = particle.velocity.sub(body_point_velocity(body, contact));
    let vn = relative.dot(n);
    // Separating or grazing contact: nothing to resolve (zero impulse identity).
    if vn >= 0.0 {
        return Vec3::ZERO;
    }

    let effective_inv_mass = particle.inv_mass
        + generalized_inverse_mass(body.inv_mass, body.inv_inertia_world, r, n);
    // Both particle and body effectively immovable: no finite impulse exists.
    if effective_inv_mass <= MIN_EFFECTIVE_INV_MASS {
        return Vec3::ZERO;
    }

    // Restitution is a physical reflection coefficient; outside 0..=1 it would
    // either absorb impossible energy or inject it, so it is clamped.
    let e = restitution.clamp(0.0, 1.0);
    // j = -(1 + e) * vn / w. With vn < 0 this is positive, pushing the particle
    // back along +normal.
    let magnitude = -(1.0 + e) * vn / effective_inv_mass;
    n.scale(magnitude)
}

/// Applies an already-computed coupling `impulse` to the particle and the
/// equal-and-opposite reaction to the body, in place.
///
/// The particle gains `impulse` (linear velocity change `inv_mass * impulse`).
/// The body gains the reaction `-impulse` at the particle's position: its linear
/// velocity changes by `inv_mass_body * (-impulse)` and its angular velocity by
/// `I_inv_world * (r x -impulse)`, with `r = particle.position -
/// center_of_mass`. Because both impulses act at the one contact point, the net
/// momentum added to the particle+body system is zero.
pub fn apply_coupling(particle: &mut CouplingParticle, body: &mut CouplingBody, impulse: Vec3) {
    let r = particle.position.sub(body.center_of_mass);
    let reaction = impulse.scale(-1.0);

    particle.velocity = particle.velocity.add(impulse.scale(particle.inv_mass));
    body.linear_velocity = body.linear_velocity.add(reaction.scale(body.inv_mass));
    body.angular_velocity = body
        .angular_velocity
        .add(body.inv_inertia_world.mul_vec3(r.cross(reaction)));
}

/// Resolves one particle<->body coupling step along the unit contact `normal`
/// with the given `restitution`, mutating both in place, and returns the impulse
/// applied to the particle.
///
/// This is [`coupling_impulse`] followed by [`apply_coupling`]: the single entry
/// point a simulation step calls per contact. The returned impulse is the
/// particle's momentum change; its negation is the body's.
pub fn resolve_coupling(
    particle: &mut CouplingParticle,
    body: &mut CouplingBody,
    normal: Vec3,
    restitution: f32,
) -> Vec3 {
    let impulse = coupling_impulse(particle, body, normal, restitution);
    apply_coupling(particle, body, impulse);
    impulse
}

/// Linear momentum `mass * velocity` of a body with the given finite `mass`.
///
/// Provided for conservation accounting: the coupling stores inverse mass, so a
/// caller summing system momentum passes the finite `mass` (for a dynamic body,
/// `1 / inv_mass`). Infinite-mass (static) participants are excluded from such a
/// sum rather than passed here.
#[must_use]
pub fn linear_momentum(mass: f32, velocity: Vec3) -> Vec3 {
    velocity.scale(mass)
}

#[cfg(test)]
mod tests {
    use super::super::determinism::unit_f32_from_bits;
    use super::super::squares_rng::squares32;
    use super::*;
    use alloc::vec::Vec;

    /// Absolute tolerance for scalar near-equality in the gold-standard checks.
    const EPS: f32 = 1e-4;

    /// Finite mass from an inverse mass; used only by the test harness to form
    /// momentum sums for dynamic (finite-mass) participants.
    fn mass_of(inv_mass: f32) -> f32 {
        1.0 / inv_mass
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    /// Vector near-equality with a magnitude-relative tolerance, so conservation
    /// checks on large momenta are not held to an unrealistic absolute bound.
    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        let scale = 1.0 + a.length().max(b.length());
        a.distance(b) <= EPS * scale
    }

    /// A deterministic signed sample in `-1.0..1.0` from the counter-based
    /// `squares32` generator mapped through the shared uniform-`f32` helper (no
    /// bespoke `RNG`).
    fn signed_unit(ctr: u64, key: u64) -> f32 {
        unit_f32_from_bits(squares32(ctr, key)) * 2.0 - 1.0
    }

    #[test]
    fn body_point_velocity_adds_rotational_term() {
        // Spin about +Z at 2 rad/s; a point at +X on the body moves toward +Y.
        let body = CouplingBody::new(
            1.0,
            Mat3::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 2.0),
        );
        let v = body_point_velocity(&body, Vec3::new(1.0, 0.0, 0.0));
        assert!(approx(v.x, 0.0));
        assert!(approx(v.y, 2.0));
        assert!(approx(v.z, 0.0));
    }

    #[test]
    fn single_particle_single_body_conserves_linear_momentum() {
        // Particle (mass 2) falling onto a free body (mass 4) with its center of
        // mass offset so the lever arm is non-trivial.
        let mut particle = CouplingParticle::new(
            0.5,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -3.0, 0.0),
        );
        let mut body = CouplingBody::new(
            0.25,
            Mat3::from_diagonal(Vec3::new(0.5, 0.5, 0.5)),
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.5),
        );
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let before = linear_momentum(mass_of(particle.inv_mass), particle.velocity)
            .add(linear_momentum(mass_of(body.inv_mass), body.linear_velocity));
        let impulse = resolve_coupling(&mut particle, &mut body, normal, 0.5);
        let after = linear_momentum(mass_of(particle.inv_mass), particle.velocity)
            .add(linear_momentum(mass_of(body.inv_mass), body.linear_velocity));

        // A real, non-zero impulse was exchanged, and total linear momentum is
        // unchanged to f32 tolerance.
        assert!(impulse.length() > EPS);
        assert!(approx_vec(before, after));
    }

    #[test]
    fn restitution_sets_the_post_contact_relative_velocity() {
        // The defining gold standard: after resolution the normal relative
        // velocity equals -e * vn, exercising the full linear + angular response.
        let mut particle = CouplingParticle::new(
            0.5,
            Vec3::new(0.3, 0.8, -0.2),
            Vec3::new(0.1, -2.0, 0.4),
        );
        let mut body = CouplingBody::new(
            0.2,
            Mat3::from_diagonal(Vec3::new(0.7, 0.4, 0.9)),
            Vec3::new(0.0, -0.1, 0.0),
            Vec3::new(-0.2, 0.3, 0.1),
            Vec3::new(0.1, -0.2, 0.3),
        );
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let restitution = 0.6;

        let contact = particle.position;
        let vn_before = particle
            .velocity
            .sub(body_point_velocity(&body, contact))
            .dot(normal);
        resolve_coupling(&mut particle, &mut body, normal, restitution);
        // Re-evaluate the body's surface velocity at the same world point.
        let vn_after = particle
            .velocity
            .sub(body_point_velocity(&body, contact))
            .dot(normal);

        assert!(vn_before < 0.0, "contact must be approaching");
        assert!(approx(vn_after, -restitution * vn_before));
    }

    #[test]
    fn separating_and_degenerate_contacts_are_identities() {
        let base_particle =
            CouplingParticle::new(0.5, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 2.0, 0.0));
        let base_body = CouplingBody::new(
            0.25,
            Mat3::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        );
        let normal = Vec3::new(0.0, 1.0, 0.0);

        // Particle already receding along +normal: zero impulse, no state change.
        {
            let mut particle = base_particle;
            let mut body = base_body;
            let impulse = resolve_coupling(&mut particle, &mut body, normal, 0.5);
            assert_eq!(impulse, Vec3::ZERO);
            assert_eq!(particle, base_particle);
            assert_eq!(body, base_body);
        }
        // Degenerate (zero-length) normal: zero impulse, no state change, no NaN.
        {
            let mut particle =
                CouplingParticle::new(0.5, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -2.0, 0.0));
            let approaching = particle;
            let mut body = base_body;
            let impulse = resolve_coupling(&mut particle, &mut body, Vec3::ZERO, 0.5);
            assert_eq!(impulse, Vec3::ZERO);
            assert_eq!(particle, approaching);
            assert_eq!(body, base_body);
        }
    }

    #[test]
    fn degenerate_masses_behave_physically() {
        let normal = Vec3::new(0.0, 1.0, 0.0);

        // Static body (inv_mass = 0, infinite inertia): the body never moves and
        // a unit-mass particle bounces with the full restitution response.
        {
            let mut particle = CouplingParticle::new(
                1.0,
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, -2.0, 0.0),
            );
            let mut body =
                CouplingBody::new(0.0, Mat3::from_diagonal(Vec3::ZERO), Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            let before = body;
            resolve_coupling(&mut particle, &mut body, normal, 1.0);
            // Body is untouched; particle reflects (-2 -> +2 at e = 1).
            assert_eq!(body, before);
            assert!(approx(particle.velocity.y, 2.0));
        }

        // Immovable particle (inv_mass = 0) hitting a free body: the particle is
        // unchanged and the body absorbs the whole exchange.
        {
            let mut particle = CouplingParticle::new(
                0.0,
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, -2.0, 0.0),
            );
            let fixed_particle = particle;
            let mut body = CouplingBody::new(
                0.5,
                Mat3::from_diagonal(Vec3::new(1.0, 1.0, 1.0)),
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::ZERO,
            );
            let impulse = resolve_coupling(&mut particle, &mut body, normal, 0.0);
            assert_eq!(particle, fixed_particle);
            assert!(impulse.length() > EPS);
            // Reaction -J pushes the body down along -normal.
            assert!(body.linear_velocity.y < 0.0);
        }

        // Both immovable: no finite impulse exists, so nothing changes.
        {
            let mut particle = CouplingParticle::new(
                0.0,
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, -2.0, 0.0),
            );
            let fixed_particle = particle;
            let mut body =
                CouplingBody::new(0.0, Mat3::from_diagonal(Vec3::ZERO), Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            let fixed_body = body;
            let impulse = resolve_coupling(&mut particle, &mut body, normal, 0.5);
            assert_eq!(impulse, Vec3::ZERO);
            assert_eq!(particle, fixed_particle);
            assert_eq!(body, fixed_body);
        }
    }

    #[test]
    fn many_particles_accumulate_reaction_and_conserve_momentum() {
        // A swarm of particles couples sequentially to one shared body. Each step
        // conserves momentum, so the whole batch does too, and the body's total
        // momentum change equals the negated sum of the particle impulses.
        let mut body = CouplingBody::new(
            0.2,
            Mat3::from_diagonal(Vec3::new(0.6, 0.6, 0.6)),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        );
        let body_mass = mass_of(body.inv_mass);

        let mut particles = Vec::new();
        for i in 0..8_u32 {
            let x = f32::from(i as u16) * 0.25 - 1.0;
            particles.push(CouplingParticle::new(
                0.5,
                Vec3::new(x, 1.0, 0.0),
                Vec3::new(0.0, -1.5, 0.0),
            ));
        }
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let mut momentum_before = linear_momentum(body_mass, body.linear_velocity);
        for p in &particles {
            momentum_before = momentum_before.add(linear_momentum(mass_of(p.inv_mass), p.velocity));
        }
        let body_momentum_before = linear_momentum(body_mass, body.linear_velocity);

        let mut summed_impulse = Vec3::ZERO;
        for p in &mut particles {
            let impulse = resolve_coupling(p, &mut body, normal, 0.3);
            summed_impulse = summed_impulse.add(impulse);
        }

        let mut momentum_after = linear_momentum(body_mass, body.linear_velocity);
        for p in &particles {
            momentum_after = momentum_after.add(linear_momentum(mass_of(p.inv_mass), p.velocity));
        }
        let body_momentum_after = linear_momentum(body_mass, body.linear_velocity);

        // Total system momentum conserved.
        assert!(approx_vec(momentum_before, momentum_after));
        // The body's momentum change is exactly the reaction to the particle sum.
        let body_delta = body_momentum_after.sub(body_momentum_before);
        assert!(approx_vec(body_delta, summed_impulse.scale(-1.0)));
    }

    #[test]
    fn random_batch_conserves_linear_momentum() {
        // Deterministic pseudo-random swarm (via squares32) coupling to a free,
        // spinning, offset body. The headline invariant: total linear momentum is
        // conserved across the whole batch to f32 tolerance.
        let key = 0x9E37_79B9_7F4A_7C15;
        let mut body = CouplingBody::new(
            0.3,
            inv_inertia_world(Vec3::new(0.8, 0.5, 0.9), Mat3::IDENTITY),
            Vec3::new(0.2, -0.3, 0.1),
            Vec3::new(0.1, 0.2, -0.1),
            Vec3::new(-0.2, 0.1, 0.3),
        );
        let body_mass = mass_of(body.inv_mass);

        let mut particles = Vec::new();
        for i in 0..24_u64 {
            // Positive inverse mass in 0.25..=1.25 keeps every particle dynamic.
            let inv_mass = 0.75 + 0.5 * signed_unit(i, key);
            let position = Vec3::new(
                signed_unit(i.wrapping_mul(7).wrapping_add(1), key),
                signed_unit(i.wrapping_mul(7).wrapping_add(2), key),
                signed_unit(i.wrapping_mul(7).wrapping_add(3), key),
            );
            let velocity = Vec3::new(
                signed_unit(i.wrapping_mul(7).wrapping_add(4), key),
                signed_unit(i.wrapping_mul(7).wrapping_add(5), key),
                signed_unit(i.wrapping_mul(7).wrapping_add(6), key),
            );
            particles.push(CouplingParticle::new(inv_mass, position, velocity));
        }

        let mut momentum_before = linear_momentum(body_mass, body.linear_velocity);
        for p in &particles {
            momentum_before = momentum_before.add(linear_momentum(mass_of(p.inv_mass), p.velocity));
        }

        for (i, p) in particles.iter_mut().enumerate() {
            // A random (occasionally degenerate) contact normal per particle.
            let raw = Vec3::new(
                signed_unit((i as u64).wrapping_mul(11).wrapping_add(100), key),
                signed_unit((i as u64).wrapping_mul(11).wrapping_add(101), key),
                signed_unit((i as u64).wrapping_mul(11).wrapping_add(102), key),
            );
            let normal = raw.normalize_or_zero();
            let restitution = 0.5 + 0.5 * signed_unit((i as u64).wrapping_add(200), key);
            resolve_coupling(p, &mut body, normal, restitution);
        }

        let mut momentum_after = linear_momentum(body_mass, body.linear_velocity);
        for p in &particles {
            momentum_after = momentum_after.add(linear_momentum(mass_of(p.inv_mass), p.velocity));
        }

        assert!(approx_vec(momentum_before, momentum_after));
    }

    #[test]
    fn resolution_is_deterministic() {
        let make = || {
            (
                CouplingParticle::new(0.5, Vec3::new(0.1, 0.9, -0.2), Vec3::new(0.2, -1.7, 0.3)),
                CouplingBody::new(
                    0.25,
                    Mat3::from_diagonal(Vec3::new(0.6, 0.4, 0.8)),
                    Vec3::new(0.0, -0.1, 0.0),
                    Vec3::new(0.05, 0.1, -0.05),
                    Vec3::new(0.1, -0.1, 0.2),
                ),
            )
        };
        let normal = Vec3::new(0.1, 0.95, -0.05);
        let (mut p1, mut b1) = make();
        let (mut p2, mut b2) = make();
        let j1 = resolve_coupling(&mut p1, &mut b1, normal, 0.4);
        let j2 = resolve_coupling(&mut p2, &mut b2, normal, 0.4);
        assert_eq!(j1, j2);
        assert_eq!(p1, p2);
        assert_eq!(b1, b2);
    }

    #[test]
    fn inv_inertia_world_identity_rotation_is_diagonal() {
        let tensor = inv_inertia_world(Vec3::new(2.0, 3.0, 4.0), Mat3::IDENTITY);
        assert_eq!(tensor, Mat3::from_diagonal(Vec3::new(2.0, 3.0, 4.0)));
    }
}
