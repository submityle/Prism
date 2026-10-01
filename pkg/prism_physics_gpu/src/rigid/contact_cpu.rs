//! The `CPU` golden reference for the 6-DOF rigid-body contact solver.
//!
//! [`cpu_solve_contacts`] resolves a set of [`RigidContact`]s between the bodies
//! of a [`RigidBodyState`] with velocity-level sequential impulses: it drives
//! the relative velocity at each contact to a non-penetrating (and, for a
//! bouncy pair, separating) target, exchanging both linear and angular momentum
//! through each body's inverse mass and world-space inverse inertia. It is the
//! authoritative twin the device kernel in [`contact_gpu`](super::contact_gpu)
//! must reproduce; the parity test (`tests/rigid_contact_parity.rs`) bounds
//! their divergence to floating-point reassociation noise.
//!
//! # The solve, step by step
//!
//! One frame of duration `dt` is resolved as:
//!
//! 1. **Prepare.** Measure each contact's initial relative normal velocity
//!    `vn0` from the incoming state. Restitution is driven from `vn0` (the true
//!    approach speed) rather than from the velocity after warm starting, so a
//!    warm-started separating velocity cannot be mistaken for an approach.
//! 2. **Warm start.** Re-apply each contact's accumulated impulse from the
//!    previous frame, so a loaded stack starts near its converged solution.
//! 3. **Iterate.** Run `iterations` sequential-impulse sweeps. Each sweep, per
//!    contact: solve the non-penetration normal impulse (accumulated and clamped
//!    non-negative, with a Baumgarte position bias and the restitution target),
//!    then solve two tangential friction impulses clamped as a 2-D disc to the
//!    Coulomb cone `friction * normal_impulse`.
//!
//! Steps 2 and 3 walk the contacts in the batch order produced by
//! [`RigidContactColouring`](super::RigidContactColouring); within a batch the
//! contacts write disjoint movable bodies, so the order inside a batch does not
//! change the result and the device can run the whole batch in parallel, while
//! the barrier between batches reproduces a Gauss-Seidel sweep. The `CPU` twin
//! walks the identical batch order so the two engines converge to the same
//! fixed point.
//!
//! # Effective mass with the angular arm
//!
//! The impulse `lambda` along a unit direction `d` divides the velocity error by
//! the contact's effective mass along `d`,
//!
//! ```text
//! k = inv_m_a + inv_m_b
//!   + d . ((I_a^-1 (r_a x d)) x r_a)
//!   + d . ((I_b^-1 (r_b x d)) x r_b)
//! ```
//!
//! where `I^-1` is the **world-space** inverse inertia `R diag(inv_inertia) R^T`
//! applied by rotating the argument into the body frame, scaling by the diagonal
//! inverse inertia, and rotating back. The two angular terms are exactly what a
//! particle contact lacks; they are why an off-centre hit (`r x P != 0`) spins a
//! body instead of only translating it.
//!
//! # Honest scope
//!
//! This is a velocity-level sequential-impulse solver with a Baumgarte position
//! bias — the `Box2D` / Bullet baseline and the floor `PhysX` built its
//! temporal-Gauss-Seidel (`TGS`) solver on. It is not itself `TGS`: it does not
//! re-linearise the contact as the bodies move within the frame (the bodies are
//! not integrated here), and it leaves a small Baumgarte-driven energy leak that
//! `TGS` and split-impulse schemes remove. Those, plus joints and continuous
//! collision, are later slices. What is complete here is a genuine 6-DOF contact
//! response: linear and angular, friction and restitution, warm-started and
//! parallel-coloured, with a device twin verified on real hardware.
//!
//! Provenance: the sequential-impulse rigid contact of Catto ("Iterative
//! Dynamics with Temporal Coherence", 2005; `Box2D`), the Baumgarte position bias,
//! and box-clamped Coulomb friction, over the world-space inverse inertia of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use glam::{Quat, Vec3};

use super::body::RigidBodyState;
use super::config::{ContactSolverConfig, RigidError};
use super::contact::{contact_tangents, RigidContact};
use super::contact_coloring::RigidContactColouring;

/// Guard below which an effective mass is treated as infinite (both bodies
/// static along this direction) and the impulse is skipped. Matches `EPSILON`
/// in `shaders/rigid_contact.wgsl`.
const EPSILON: f32 = 1.192_092_9e-7;

/// Resolves `contacts` between the bodies of `state` over one frame of `dt`
/// seconds, mutating body velocities and the contacts' accumulated impulses in
/// place.
///
/// The accumulated-impulse fields of each [`RigidContact`] are both read (as the
/// warm-start seed) and written (with the converged solution), so a caller that
/// carries contacts across frames gets warm starting for free. Body *positions*
/// are untouched — this is a velocity solver; integrate the bodies separately.
///
/// # Errors
///
/// Returns [`RigidError::InvalidConfig`] when `config` fails validation,
/// [`RigidError::InconsistentState`] when the body arrays disagree in length or
/// a contact indexes a missing body, or [`RigidError::TooManyContactBatches`]
/// when the contact graph needs more parallel batches than supported. Does
/// nothing (returns `Ok`) when there are no bodies, no contacts, or `dt` is
/// non-positive.
pub fn cpu_solve_contacts(
    state: &mut RigidBodyState,
    contacts: &mut [RigidContact],
    config: &ContactSolverConfig,
    dt: f32,
) -> Result<(), RigidError> {
    config.validate()?;
    if !state.is_consistent() {
        return Err(RigidError::InconsistentState {
            reason: "per-body arrays must have equal length",
        });
    }
    if state.is_empty() || contacts.is_empty() || dt <= 0.0 {
        return Ok(());
    }

    let movable = movable_mask(state);
    let colouring = RigidContactColouring::build(contacts, &movable)?;
    let inv_dt = 1.0 / dt;
    let iterations = config.effective_iterations();

    // 1. Prepare: capture the approach speed before any impulse is applied.
    let mut vn_initial = vec![0.0f32; contacts.len()];
    for (ci, contact) in contacts.iter().enumerate() {
        vn_initial[ci] = relative_velocity(state, contact).dot(contact.normal);
    }

    // 2. Warm start: re-apply last frame's accumulated impulse, in batch order.
    for &(start, end) in colouring.ranges() {
        for &ci in &colouring.order()[start as usize..end as usize] {
            apply_warm_start(state, &contacts[ci as usize]);
        }
    }

    // 3. Iterate the sequential-impulse sweeps, in batch order.
    for _ in 0..iterations {
        for &(start, end) in colouring.ranges() {
            for &ci in &colouring.order()[start as usize..end as usize] {
                solve_one(
                    state,
                    &mut contacts[ci as usize],
                    vn_initial[ci as usize],
                    config,
                    inv_dt,
                );
            }
        }
    }

    Ok(())
}

/// Returns a per-body flag that is `true` when the solver may write the body,
/// i.e. it has a non-zero inverse mass or any non-zero inverse-inertia axis.
fn movable_mask(state: &RigidBodyState) -> Vec<bool> {
    (0..state.len())
        .map(|i| {
            let inv_i = state.inverse_inertias[i];
            state.inverse_masses[i] > 0.0 || inv_i.x > 0.0 || inv_i.y > 0.0 || inv_i.z > 0.0
        })
        .collect()
}

/// Relative velocity of the contact point on body `a` with respect to body `b`.
fn relative_velocity(state: &RigidBodyState, c: &RigidContact) -> Vec3 {
    let a = c.body_a as usize;
    let b = c.body_b as usize;
    let va = state.linear_velocities[a] + state.angular_velocities[a].cross(c.anchor_a);
    let vb = state.linear_velocities[b] + state.angular_velocities[b].cross(c.anchor_b);
    va - vb
}

/// Re-applies a contact's accumulated impulse to the two bodies' velocities.
fn apply_warm_start(state: &mut RigidBodyState, c: &RigidContact) {
    let (t1, t2) = contact_tangents(c.normal);
    let impulse = c.normal * c.normal_impulse + t1 * c.tangent_impulse_0 + t2 * c.tangent_impulse_1;
    apply_impulse(state, c, impulse);
}

/// Solves one contact's normal and friction impulses for a single sweep.
fn solve_one(
    state: &mut RigidBodyState,
    c: &mut RigidContact,
    vn_initial: f32,
    config: &ContactSolverConfig,
    inv_dt: f32,
) {
    let n = c.normal;
    let (t1, t2) = contact_tangents(n);

    // --- Normal impulse ---
    let k_n = effective_mass(state, c, n);
    if k_n > EPSILON {
        let vn = relative_velocity(state, c).dot(n);
        let position_bias = config.baumgarte * inv_dt * (c.penetration - config.slop).max(0.0);
        let restitution_bias = if vn_initial < -config.restitution_threshold {
            -c.restitution * vn_initial
        } else {
            0.0
        };
        let target = position_bias.max(restitution_bias);
        let lambda = (target - vn) / k_n;
        let new_impulse = (c.normal_impulse + lambda).max(0.0);
        let applied = new_impulse - c.normal_impulse;
        c.normal_impulse = new_impulse;
        apply_impulse(state, c, n * applied);
    }

    // --- Friction impulses (2-D cone, clamped to friction * normal_impulse) ---
    let k_t1 = effective_mass(state, c, t1);
    let k_t2 = effective_mass(state, c, t2);
    if k_t1 > EPSILON && k_t2 > EPSILON {
        let v_rel = relative_velocity(state, c);
        let delta_0 = -v_rel.dot(t1) / k_t1;
        let delta_1 = -v_rel.dot(t2) / k_t2;
        let max_friction = c.friction * c.normal_impulse;
        let old_0 = c.tangent_impulse_0;
        let old_1 = c.tangent_impulse_1;
        let mut new_0 = old_0 + delta_0;
        let mut new_1 = old_1 + delta_1;
        let magnitude = (new_0 * new_0 + new_1 * new_1).sqrt();
        if magnitude > max_friction {
            let scale = max_friction / magnitude;
            new_0 *= scale;
            new_1 *= scale;
        }
        let applied_0 = new_0 - old_0;
        let applied_1 = new_1 - old_1;
        c.tangent_impulse_0 = new_0;
        c.tangent_impulse_1 = new_1;
        apply_impulse(state, c, t1 * applied_0 + t2 * applied_1);
    }
}

/// The contact's effective mass along the unit direction `d`, including both
/// bodies' angular arms through their world-space inverse inertia.
fn effective_mass(state: &RigidBodyState, c: &RigidContact, d: Vec3) -> f32 {
    let a = c.body_a as usize;
    let b = c.body_b as usize;
    let mut k = state.inverse_masses[a] + state.inverse_masses[b];
    let arm_a = world_inv_inertia_apply(
        state.orientations[a],
        state.inverse_inertias[a],
        c.anchor_a.cross(d),
    )
    .cross(c.anchor_a);
    k += d.dot(arm_a);
    let arm_b = world_inv_inertia_apply(
        state.orientations[b],
        state.inverse_inertias[b],
        c.anchor_b.cross(d),
    )
    .cross(c.anchor_b);
    k += d.dot(arm_b);
    k
}

/// Applies the world-space impulse `p` at the contact (`+p` to body `a` at
/// `anchor_a`, `-p` to body `b` at `anchor_b`), updating both linear and angular
/// velocities. Static or locked axes contribute zero through their zero inverse
/// mass or inertia, so no explicit guard is needed for correctness.
fn apply_impulse(state: &mut RigidBodyState, c: &RigidContact, p: Vec3) {
    let a = c.body_a as usize;
    let b = c.body_b as usize;

    state.linear_velocities[a] += p * state.inverse_masses[a];
    let dw_a = world_inv_inertia_apply(
        state.orientations[a],
        state.inverse_inertias[a],
        c.anchor_a.cross(p),
    );
    state.angular_velocities[a] += dw_a;

    state.linear_velocities[b] -= p * state.inverse_masses[b];
    let dw_b = world_inv_inertia_apply(
        state.orientations[b],
        state.inverse_inertias[b],
        c.anchor_b.cross(p),
    );
    state.angular_velocities[b] -= dw_b;
}

/// Applies the world-space inverse inertia `R diag(inv_inertia) R^T` to `v`:
/// rotate `v` into the body frame, scale by the diagonal inverse inertia, and
/// rotate back.
fn world_inv_inertia_apply(orientation: Quat, inv_inertia: Vec3, v: Vec3) -> Vec3 {
    let q = [orientation.x, orientation.y, orientation.z, orientation.w];
    let body = quat_rotate(quat_conj(q), v);
    let scaled = inv_inertia * body;
    quat_rotate(q, scaled)
}

/// Conjugate (inverse rotation) of a unit quaternion stored as `(x, y, z, w)`.
fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Rotates `v` by the unit quaternion `q` via the expanded sandwich product
/// `2 (u . v) u + (s^2 - u . u) v + 2 s (u x v)` with `u = q.xyz`, `s = q.w`.
fn quat_rotate(q: [f32; 4], v: Vec3) -> Vec3 {
    let u = Vec3::new(q[0], q[1], q[2]);
    let s = q[3];
    u * (2.0 * u.dot(v)) + v * (s * s - u.dot(u)) + u.cross(v) * (2.0 * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A free body at `position` with unit mass and the given isotropic inverse
    /// inertia.
    fn push_body(state: &mut RigidBodyState, position: Vec3, inv_mass: f32, inv_inertia: Vec3) {
        state.push(position, Quat::IDENTITY, inv_mass, inv_inertia);
    }

    #[test]
    fn head_on_point_masses_exchange_linear_momentum() {
        // Two equal point masses (zero inertia, anchors at the centres so no
        // angular arm) approach head-on; a perfectly elastic frictionless
        // contact must swap their velocities and conserve linear momentum.
        let mut state = RigidBodyState::new();
        push_body(&mut state, Vec3::new(-1.0, 0.0, 0.0), 1.0, Vec3::ZERO);
        push_body(&mut state, Vec3::new(1.0, 0.0, 0.0), 1.0, Vec3::ZERO);
        state.linear_velocities[0] = Vec3::new(2.0, 0.0, 0.0);
        state.linear_velocities[1] = Vec3::new(-2.0, 0.0, 0.0);

        // Contact point between them; normal points from b (right) to a (left).
        let mut contacts =
            [
                RigidContact::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0), 0.0)
                    .with_restitution(1.0),
            ];

        let momentum_before = state.linear_velocities[0] + state.linear_velocities[1];
        let config = ContactSolverConfig::new(16, 0.0, 0.0, 0.0);
        cpu_solve_contacts(&mut state, &mut contacts, &config, 1.0 / 60.0).expect("solve");

        let momentum_after = state.linear_velocities[0] + state.linear_velocities[1];
        assert!(
            (momentum_after - momentum_before).length() < 1e-4,
            "linear momentum not conserved: {momentum_before:?} -> {momentum_after:?}"
        );
        // Equal masses, elastic: velocities swap.
        assert!(
            (state.linear_velocities[0].x - (-2.0)).abs() < 1e-3,
            "a velocity {:?}",
            state.linear_velocities[0]
        );
        assert!(
            (state.linear_velocities[1].x - 2.0).abs() < 1e-3,
            "b velocity {:?}",
            state.linear_velocities[1]
        );
    }

    #[test]
    fn resting_contact_pushes_out_of_penetration() {
        // A dynamic body penetrating a static floor; the Baumgarte bias must
        // leave it with a small separating normal velocity.
        let mut state = RigidBodyState::new();
        push_body(&mut state, Vec3::ZERO, 0.0, Vec3::ZERO); // static floor
        push_body(&mut state, Vec3::new(0.0, 0.5, 0.0), 1.0, Vec3::ZERO);
        // Floor is body a, box is body b; normal points from b (box) to a
        // (floor) = -Y, so a positive impulse pushes the box up (-(-Y) = +Y).
        let mut contacts = [RigidContact::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            0.1,
        )];
        let config = ContactSolverConfig::default();
        cpu_solve_contacts(&mut state, &mut contacts, &config, 1.0 / 60.0).expect("solve");

        assert!(state.linear_velocities[1].y > 0.0, "box should move upward");
        assert_eq!(state.linear_velocities[0], Vec3::ZERO, "floor is static");
        assert!(
            contacts[0].normal_impulse > 0.0,
            "normal impulse accumulated"
        );
    }

    #[test]
    fn friction_opposes_sliding_but_stays_in_cone() {
        // A box sliding sideways while pressed into a static floor by gravity's
        // downward velocity. Friction must reduce the slide without reversing it.
        let mut state = RigidBodyState::new();
        push_body(&mut state, Vec3::ZERO, 0.0, Vec3::ZERO); // floor
        push_body(&mut state, Vec3::new(0.0, 0.5, 0.0), 1.0, Vec3::ZERO);
        state.linear_velocities[1] = Vec3::new(3.0, -1.0, 0.0);
        let mut contacts = [RigidContact::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            0.02,
        )
        .with_friction(0.5)];
        let config = ContactSolverConfig::default();
        cpu_solve_contacts(&mut state, &mut contacts, &config, 1.0 / 60.0).expect("solve");

        let vx = state.linear_velocities[1].x;
        assert!(vx < 3.0, "friction should slow the slide: {vx}");
        assert!(vx > 0.0, "friction must not reverse the slide: {vx}");
    }

    #[test]
    fn off_centre_hit_induces_spin() {
        // A free body (unit inverse inertia) struck off-centre: the impulse arm
        // r x P must produce angular velocity, not just translation.
        let mut state = RigidBodyState::new();
        push_body(&mut state, Vec3::new(-1.0, 0.0, 0.0), 1.0, Vec3::ONE); // dynamic
        push_body(&mut state, Vec3::new(1.0, 0.0, 0.0), 0.0, Vec3::ZERO); // static striker
        state.linear_velocities[0] = Vec3::new(2.0, 0.0, 0.0);
        // Contact offset along +Y from the dynamic body's centre, normal along
        // -X (from the static striker toward the dynamic body is +X; here the
        // striker is b so normal from b to a is -X).
        let mut contacts = [RigidContact::new(
            0,
            1,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            0.0,
        )
        .with_restitution(1.0)];
        let config = ContactSolverConfig::new(16, 0.0, 0.0, 0.0);
        cpu_solve_contacts(&mut state, &mut contacts, &config, 1.0 / 60.0).expect("solve");

        // Impulse P ~ -X applied at +Y arm: torque r x P = (+Y) x (-X) = +Z.
        assert!(
            state.angular_velocities[0].z > 1e-3,
            "expected +Z spin, got {:?}",
            state.angular_velocities[0]
        );
    }

    #[test]
    fn two_static_bodies_are_a_noop() {
        let mut state = RigidBodyState::new();
        push_body(&mut state, Vec3::ZERO, 0.0, Vec3::ZERO);
        push_body(&mut state, Vec3::new(0.0, 0.5, 0.0), 0.0, Vec3::ZERO);
        let mut contacts = [RigidContact::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            0.3,
        )];
        let config = ContactSolverConfig::default();
        cpu_solve_contacts(&mut state, &mut contacts, &config, 1.0 / 60.0).expect("solve");
        assert_eq!(state.linear_velocities[0], Vec3::ZERO);
        assert_eq!(state.linear_velocities[1], Vec3::ZERO);
        assert_eq!(contacts[0].normal_impulse, 0.0);
    }

    #[test]
    fn slow_approach_below_threshold_does_not_bounce() {
        // Two point masses closing slower than the restitution threshold: even
        // with restitution 1 the contact must not add separating speed beyond
        // gently removing penetration.
        let mut state = RigidBodyState::new();
        push_body(&mut state, Vec3::new(-1.0, 0.0, 0.0), 1.0, Vec3::ZERO);
        push_body(&mut state, Vec3::new(1.0, 0.0, 0.0), 1.0, Vec3::ZERO);
        state.linear_velocities[0] = Vec3::new(0.1, 0.0, 0.0);
        state.linear_velocities[1] = Vec3::new(-0.1, 0.0, 0.0);
        let mut contacts =
            [
                RigidContact::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0), 0.0)
                    .with_restitution(1.0),
            ];
        // Threshold 0.5 > approach speed 0.2: restitution suppressed.
        let config = ContactSolverConfig::new(16, 0.0, 0.0, 0.5);
        cpu_solve_contacts(&mut state, &mut contacts, &config, 1.0 / 60.0).expect("solve");

        // Relative separating speed should be ~0 (contact just stops them), not
        // the ~0.2 an elastic bounce would restore.
        let rel = state.linear_velocities[0].x - state.linear_velocities[1].x;
        assert!(rel.abs() < 1e-2, "unexpected bounce, rel speed {rel}");
    }

    #[test]
    fn warm_start_accelerates_convergence_of_a_coupled_stack() {
        // Warm-starting re-applies the previous solve's accumulated impulse as
        // the initial guess. For a single contact sequential impulses converge
        // in one sweep regardless, so the observable contract only appears in a
        // *coupled* system solved with too few iterations: a cold start under-
        // converges while a seed from the converged solution holds immediately.
        //
        // Stack along Y: static floor (body 0), box1 (body 1) resting on it,
        // box2 (body 2) resting on box1. Both boxes fall at 1 m/s. Point masses
        // (zero inertia, zero anchors) keep the response purely linear.
        /// Builds the three-body falling stack from scratch.
        fn build_stack() -> (RigidBodyState, [RigidContact; 2]) {
            let mut state = RigidBodyState::new();
            push_body(&mut state, Vec3::ZERO, 0.0, Vec3::ZERO); // static floor
            push_body(&mut state, Vec3::new(0.0, 1.0, 0.0), 1.0, Vec3::ZERO); // box1
            push_body(&mut state, Vec3::new(0.0, 2.0, 0.0), 1.0, Vec3::ZERO); // box2
            state.linear_velocities[1] = Vec3::new(0.0, -1.0, 0.0);
            state.linear_velocities[2] = Vec3::new(0.0, -1.0, 0.0);
            let down = Vec3::new(0.0, -1.0, 0.0);
            let floor_box = RigidContact::new(0, 1, Vec3::ZERO, Vec3::ZERO, down, 0.0);
            let box_box = RigidContact::new(1, 2, Vec3::ZERO, Vec3::ZERO, down, 0.0);
            (state, [floor_box, box_box])
        }

        // No penetration bias and no restitution: the converged state is both
        // boxes at rest.
        let config_many = ContactSolverConfig::new(64, 0.0, 0.0, 0.0);
        let config_one = ContactSolverConfig::new(1, 0.0, 0.0, 0.0);
        let dt = 1.0 / 60.0;

        // 1. Converge fully to recover the steady-state accumulated impulses.
        let (mut converged_state, mut converged_contacts) = build_stack();
        cpu_solve_contacts(
            &mut converged_state,
            &mut converged_contacts,
            &config_many,
            dt,
        )
        .expect("converge");
        assert!(
            converged_state.linear_velocities[1].y.abs() < 1e-4
                && converged_state.linear_velocities[2].y.abs() < 1e-4,
            "stack should settle: {:?} {:?}",
            converged_state.linear_velocities[1],
            converged_state.linear_velocities[2]
        );

        // 2. One cold iteration under-converges: box1 is still driven downward
        //    because box2's load has not yet propagated through the stack.
        let (mut cold_state, mut cold_contacts) = build_stack();
        cpu_solve_contacts(&mut cold_state, &mut cold_contacts, &config_one, dt).expect("cold");
        assert!(
            cold_state.linear_velocities[1].y < -0.1,
            "cold single-iteration solve should leave box1 moving down, got {:?}",
            cold_state.linear_velocities[1]
        );

        // 3. One warm-started iteration (seeded with the converged impulses)
        //    reaches the settled state immediately — proof the accumulated
        //    impulse was re-applied.
        let (mut warm_state, mut warm_contacts) = build_stack();
        warm_contacts[0].normal_impulse = converged_contacts[0].normal_impulse;
        warm_contacts[1].normal_impulse = converged_contacts[1].normal_impulse;
        cpu_solve_contacts(&mut warm_state, &mut warm_contacts, &config_one, dt).expect("warm");
        assert!(
            warm_state.linear_velocities[1].y.abs() < 1e-4
                && warm_state.linear_velocities[2].y.abs() < 1e-4,
            "warm-started single iteration should already hold the stack, got {:?} {:?}",
            warm_state.linear_velocities[1],
            warm_state.linear_velocities[2]
        );
    }
}
