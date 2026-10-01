//! The `CPU` golden reference for the soft-constraint Temporal Gauss-Seidel
//! (`TGS`) rigid-body contact stepper.
//!
//! [`cpu_solve_contacts_tgs`] is the authoritative twin the device kernel in
//! [`contact_tgs_gpu`](super) must reproduce. Unlike the velocity-only
//! [`cpu_solve_contacts`](super::cpu_solve_contacts), this routine is a *full
//! stepper*: it owns the frame's integration. It substeps the frame and, within
//! each substep, integrates the bodies' velocities under gravity, warm starts
//! and re-linearises the contacts against the moved geometry, resolves them as
//! damped springs, integrates the bodies' positions and orientations, and runs
//! bias-free relaxation sweeps. The caller must therefore **not** integrate the
//! bodies separately — doing so would advance them twice.
//!
//! # Why a soft-constraint substepping stepper
//!
//! The velocity-level solver corrects penetration with a raw Baumgarte bias:
//! it injects a position-correction velocity but never removes it, so a resting
//! stack slowly gains energy and jitters. The `TGS`-soft scheme fixes this in
//! two ways that only a stepper (not a pure velocity solver) can provide:
//!
//! 1. **Soft constraints.** The penetration error is pulled out as a damped
//!    spring at a target frequency, scaling the corrective impulse by
//!    [`SoftParams::mass_scale`](crate::SoftParams) and decaying the
//!    accumulated impulse by [`SoftParams::impulse_scale`](crate::SoftParams).
//!    Taking the frequency to infinity recovers the rigid Baumgarte limit.
//! 2. **Relaxation.** After the bodies are integrated, bias-free sweeps remove
//!    exactly the position-correction velocity the biased pass injected, so the
//!    frame ends at the true physical velocity and no energy is manufactured.
//!
//! Between substeps the contact anchors and separation are re-linearised
//! against the moved bodies (the "`TGS`" of the name), so a tall stack settles
//! without the drift a frame-constant linearisation would accumulate.
//!
//! # Per-substep schedule
//!
//! Each of the `substeps` sub-intervals of size `h = dt / substeps` runs, in
//! order:
//!
//! 1. **Integrate velocities.** Add `gravity * h` to each translatable body and
//!    apply linear and angular damping, matching
//!    [`cpu_integrate`](super::cpu_integrate)'s per-substep damping.
//! 2. **Warm start.** Re-apply every contact's accumulated impulse to the
//!    velocities, in batch order.
//! 3. **Re-linearise.** Rotate each contact's world arms by the body's
//!    incremental rotation and recompute its separation from the bodies'
//!    centre-of-mass (`COM`) translation and arm rotation.
//! 4. **Biased solve.** One Gauss-Seidel sweep in batch order: the soft normal
//!    impulse (accumulated, clamped non-negative) then the two friction
//!    impulses clamped as a disc to the Coulomb cone.
//! 5. **Integrate positions.** Advance each body's position and orientation by
//!    its current velocity over `h`.
//! 6. **Relax.** [`TgsContactConfig::effective_relax_iterations`] bias-free
//!    sweeps that remove the injected bias velocity.
//!
//! After the substep loop, a single restitution pass restores any elastic
//! bounce from the pre-step approach speed, and the converged accumulated
//! impulses are written back into the caller's contacts for next frame's warm
//! start.
//!
//! # Honest scope
//!
//! This stepper resolves gravity and contacts only: it takes no external forces
//! or torques, and it omits the explicit gyroscopic coupling that
//! [`cpu_integrate`](super::cpu_integrate) carries, because within a contact
//! substep the free-flight gyroscopic term is negligible next to the contact
//! response (this matches `Box2D`'s and Chaos's contact substep). Joints,
//! continuous collision, and an external-force path are later slices.
//!
//! Provenance: the soft-constraint contact of Catto ("Soft Constraints", GDC
//! 2011) and the substepping solver loop it feeds (`Box2D` TGS Soft), over the
//! world-space inverse inertia and quaternion kinematics of Baraff & Witkin. No
//! Unreal Engine source or derived code.

use glam::{Quat, Vec3};

use super::body::RigidBodyState;
use super::config::{IntegratorConfig, RigidError};
use super::contact::{contact_tangents, RigidContact};
use super::contact_coloring::RigidContactColouring;
use super::contact_cpu::{apply_impulse, effective_mass, movable_mask, relative_velocity, EPSILON};
use super::contact_tgs_config::TgsContactConfig;
use crate::xpbd::SoftParams;

/// Maximum speed (metres per second) at which the biased solve is allowed to
/// push two deeply penetrating bodies apart. Clamping the recovery speed keeps
/// a large initial overlap from launching the bodies; it matches the
/// `maxBiasVelocity` constant of `Box2D` v3's contact solver.
const MAX_RECOVERY_SPEED: f32 = 3.0;

/// Advances `state` by one frame of `dt` seconds, resolving `contacts` with the
/// soft-constraint `TGS` stepper and integrating the bodies in place.
///
/// This routine integrates the bodies itself: it adds gravity, applies damping,
/// and advances positions and orientations across `integrator.substeps`
/// sub-intervals, resolving the contacts between each. **Do not** call
/// [`cpu_integrate`](super::cpu_integrate) on the same bodies in the same frame.
///
/// The accumulated-impulse fields of each [`RigidContact`] are both read (as the
/// warm-start seed) and written (with the converged solution). These impulses
/// are *per substep* quantities and are not interchangeable with the *per
/// frame* impulses stored by [`cpu_solve_contacts`](super::cpu_solve_contacts);
/// a contact carried across frames must be driven by only one of the two
/// solvers.
///
/// # Errors
///
/// Returns [`RigidError::InvalidConfig`] when either configuration fails
/// validation, [`RigidError::InconsistentState`] when the body arrays disagree
/// in length or a contact indexes a missing body, or
/// [`RigidError::TooManyContactBatches`] when the contact graph needs more
/// parallel batches than supported. Does nothing (returns `Ok`) when there are
/// no bodies or `dt` is non-positive; a body with no contacts is still
/// integrated.
pub fn cpu_solve_contacts_tgs(
    state: &mut RigidBodyState,
    contacts: &mut [RigidContact],
    integrator: &IntegratorConfig,
    tgs: &TgsContactConfig,
    dt: f32,
) -> Result<(), RigidError> {
    integrator.validate()?;
    tgs.validate()?;
    if !state.is_consistent() {
        return Err(RigidError::InconsistentState {
            reason: "per-body arrays must have equal length",
        });
    }
    if state.is_empty() || dt <= 0.0 {
        return Ok(());
    }

    let movable = movable_mask(state);
    let colouring = RigidContactColouring::build(contacts, &movable)?;

    let substeps = integrator.effective_substeps();
    let h = dt / substeps as f32;
    if h <= 0.0 {
        return Ok(());
    }
    let inv_h = 1.0 / h;
    let soft = SoftParams::from_hertz(tgs.contact_hertz, tgs.contact_damping_ratio, h);
    let relax = SoftParams::relax();
    let relax_iterations = tgs.effective_relax_iterations();

    // Snapshot the frame-initial transforms so each re-linearisation measures
    // the geometry's drift from the start of the frame.
    let initial_positions: Vec<Vec3> = state.positions.clone();
    let initial_orientations: Vec<Quat> = state.orientations.clone();

    // Per-contact frame-initial data. The working contacts carry the live arms,
    // separation (as `penetration = -separation`), and accumulated impulses.
    let mut work: Vec<RigidContact> = contacts.to_vec();
    let initial_arm_a: Vec<Vec3> = contacts.iter().map(|c| c.anchor_a).collect();
    let initial_arm_b: Vec<Vec3> = contacts.iter().map(|c| c.anchor_b).collect();
    let base_separation: Vec<f32> = contacts.iter().map(|c| -c.penetration).collect();
    let approach_speed: Vec<f32> = contacts
        .iter()
        .map(|c| relative_velocity(state, c).dot(c.normal))
        .collect();

    // Per-substep damping scales, matching `cpu_integrate`.
    let linear_damping_scale = (1.0 - integrator.linear_damping * h).max(0.0);
    let angular_damping_scale = (1.0 - integrator.angular_damping * h).max(0.0);

    for _ in 0..substeps {
        integrate_velocities(
            state,
            &movable,
            integrator.gravity,
            linear_damping_scale,
            angular_damping_scale,
            h,
        );
        warm_start(state, &work, &colouring);
        relinearise(
            state,
            &mut work,
            &initial_positions,
            &initial_orientations,
            &initial_arm_a,
            &initial_arm_b,
            &base_separation,
        );
        solve_sweep(state, &mut work, &colouring, &soft, inv_h, tgs.slop, true);
        integrate_positions(state, &movable, h);
        for _ in 0..relax_iterations {
            solve_sweep(state, &mut work, &colouring, &relax, inv_h, tgs.slop, false);
        }
    }

    apply_restitution(state, &mut work, &approach_speed, tgs.restitution_threshold);

    // Store the converged impulses back for next frame's warm start.
    for (dst, src) in contacts.iter_mut().zip(work.iter()) {
        dst.normal_impulse = src.normal_impulse;
        dst.tangent_impulse_0 = src.tangent_impulse_0;
        dst.tangent_impulse_1 = src.tangent_impulse_1;
    }

    Ok(())
}

/// Adds `gravity * h` to every translatable body and applies the per-substep
/// linear and angular damping scales, mutating velocities in place.
fn integrate_velocities(
    state: &mut RigidBodyState,
    movable: &[bool],
    gravity: Vec3,
    linear_damping_scale: f32,
    angular_damping_scale: f32,
    h: f32,
) {
    for (i, &is_movable) in movable.iter().enumerate() {
        if !is_movable {
            continue;
        }
        if state.inverse_masses[i] > 0.0 {
            state.linear_velocities[i] =
                (state.linear_velocities[i] + gravity * h) * linear_damping_scale;
        }
        state.angular_velocities[i] *= angular_damping_scale;
    }
}

/// Re-applies every contact's accumulated impulse to the body velocities, in
/// batch order, so the substep starts near its warm-started solution.
fn warm_start(
    state: &mut RigidBodyState,
    work: &[RigidContact],
    colouring: &RigidContactColouring,
) {
    for &(start, end) in colouring.ranges() {
        for &ci in &colouring.order()[start as usize..end as usize] {
            let c = &work[ci as usize];
            let (t1, t2) = contact_tangents(c.normal);
            let impulse =
                c.normal * c.normal_impulse + t1 * c.tangent_impulse_0 + t2 * c.tangent_impulse_1;
            apply_impulse(state, c, impulse);
        }
    }
}

/// Rotates each contact's world arms by its bodies' incremental rotation since
/// the frame start and recomputes the separation from the bodies' `COM`
/// translation and arm rotation, writing the live arms and `penetration` into
/// the working contacts.
fn relinearise(
    state: &RigidBodyState,
    work: &mut [RigidContact],
    initial_positions: &[Vec3],
    initial_orientations: &[Quat],
    initial_arm_a: &[Vec3],
    initial_arm_b: &[Vec3],
    base_separation: &[f32],
) {
    for (ci, c) in work.iter_mut().enumerate() {
        let a = c.body_a as usize;
        let b = c.body_b as usize;

        let delta_rot_a = state.orientations[a] * initial_orientations[a].conjugate();
        let delta_rot_b = state.orientations[b] * initial_orientations[b].conjugate();
        let arm_a = delta_rot_a.mul_vec3(initial_arm_a[ci]);
        let arm_b = delta_rot_b.mul_vec3(initial_arm_b[ci]);

        let shift_a = (state.positions[a] - initial_positions[a]) + (arm_a - initial_arm_a[ci]);
        let shift_b = (state.positions[b] - initial_positions[b]) + (arm_b - initial_arm_b[ci]);
        let separation = base_separation[ci] + (shift_a - shift_b).dot(c.normal);

        c.anchor_a = arm_a;
        c.anchor_b = arm_b;
        c.penetration = -separation;
    }
}

/// Runs one Gauss-Seidel sweep over the coloured contacts, solving each
/// contact's normal impulse (soft when `use_bias`, bias-free otherwise) then its
/// friction impulses.
fn solve_sweep(
    state: &mut RigidBodyState,
    work: &mut [RigidContact],
    colouring: &RigidContactColouring,
    soft: &SoftParams,
    inv_h: f32,
    slop: f32,
    use_bias: bool,
) {
    for &(start, end) in colouring.ranges() {
        for &ci in &colouring.order()[start as usize..end as usize] {
            solve_one(state, &mut work[ci as usize], soft, inv_h, slop, use_bias);
        }
    }
}

/// Solves one contact's soft normal impulse and clamped friction impulses for a
/// single sweep, mutating the body velocities and the contact's accumulated
/// impulses in place.
fn solve_one(
    state: &mut RigidBodyState,
    c: &mut RigidContact,
    soft: &SoftParams,
    inv_h: f32,
    slop: f32,
    use_bias: bool,
) {
    let n = c.normal;
    let (t1, t2) = contact_tangents(n);

    // --- Soft normal impulse ---
    let k_n = effective_mass(state, c, n);
    if k_n > EPSILON {
        let normal_mass = 1.0 / k_n;
        let separation = -c.penetration;
        let (bias, mass_scale, impulse_scale) = if separation > 0.0 {
            // Speculative: the bodies are not yet touching, so allow approach
            // only up to closing the gap within this substep.
            (separation * inv_h, 1.0, 0.0)
        } else if use_bias {
            // Penetrating: pull the overlap out as a damped spring, leaving
            // `slop` of penetration uncorrected and clamping the recovery
            // speed. `(separation + slop) <= 0` guards against a push inside
            // the slop band.
            let soft_bias = (soft.bias_rate * (separation + slop)).clamp(-MAX_RECOVERY_SPEED, 0.0);
            (soft_bias, soft.mass_scale, soft.impulse_scale)
        } else {
            // Relaxation: no bias, full mass, no impulse decay.
            (0.0, 1.0, 0.0)
        };
        let vn = relative_velocity(state, c).dot(n);
        let impulse = -normal_mass * mass_scale * (vn + bias) - impulse_scale * c.normal_impulse;
        let new_impulse = (c.normal_impulse + impulse).max(0.0);
        let applied = new_impulse - c.normal_impulse;
        c.normal_impulse = new_impulse;
        apply_impulse(state, c, n * applied);
    }

    // --- Friction impulses (2-D cone, no bias) ---
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

/// Advances every translatable body's position and every rotatable body's
/// orientation by its current velocity over `h`.
fn integrate_positions(state: &mut RigidBodyState, movable: &[bool], h: f32) {
    for (i, &is_movable) in movable.iter().enumerate() {
        if !is_movable {
            continue;
        }
        if state.inverse_masses[i] > 0.0 {
            state.positions[i] += state.linear_velocities[i] * h;
        }
        let inv_inertia = state.inverse_inertias[i];
        let can_rotate = inv_inertia.x > 0.0 || inv_inertia.y > 0.0 || inv_inertia.z > 0.0;
        if can_rotate {
            state.orientations[i] =
                integrate_orientation(state.orientations[i], state.angular_velocities[i], h);
        }
    }
}

/// Advances the orientation `q` by angular velocity `omega` over `h` via the
/// quaternion kinematic equation `q' = normalize(q + 0.5 h (omega_q * q))`,
/// matching [`cpu_integrate`](super::cpu_integrate)'s orientation update.
fn integrate_orientation(q: Quat, omega: Vec3, h: f32) -> Quat {
    let omega_quat = Quat::from_xyzw(omega.x, omega.y, omega.z, 0.0);
    let dq = omega_quat * q;
    let half_h = 0.5 * h;
    let integrated = Quat::from_xyzw(
        q.x + dq.x * half_h,
        q.y + dq.y * half_h,
        q.z + dq.z * half_h,
        q.w + dq.w * half_h,
    );
    let length = integrated.length();
    if length > EPSILON {
        integrated / length
    } else {
        Quat::IDENTITY
    }
}

/// Restores the elastic bounce from the pre-step approach speed for every
/// contact whose approach exceeded the restitution threshold and which carried
/// a positive normal impulse, in a single pass after the substep loop.
fn apply_restitution(
    state: &mut RigidBodyState,
    work: &mut [RigidContact],
    approach_speed: &[f32],
    restitution_threshold: f32,
) {
    for (ci, c) in work.iter_mut().enumerate() {
        let vn0 = approach_speed[ci];
        if vn0 >= -restitution_threshold || c.normal_impulse <= 0.0 {
            continue;
        }
        let n = c.normal;
        let k_n = effective_mass(state, c, n);
        if k_n <= EPSILON {
            continue;
        }
        let normal_mass = 1.0 / k_n;
        let vn = relative_velocity(state, c).dot(n);
        let impulse = -normal_mass * (vn + c.restitution * vn0);
        let new_impulse = (c.normal_impulse + impulse).max(0.0);
        let applied = new_impulse - c.normal_impulse;
        c.normal_impulse = new_impulse;
        apply_impulse(state, c, n * applied);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A free body at `position` with the given inverse mass and isotropic
    /// inverse inertia, at identity orientation and rest.
    fn push_body(state: &mut RigidBodyState, position: Vec3, inv_mass: f32, inv_inertia: Vec3) {
        state.push(position, Quat::IDENTITY, inv_mass, inv_inertia);
    }

    /// A resting box of unit mass on a static floor, with the contact point
    /// directly beneath the box's centre so the normal impulse exerts no
    /// torque. Body 0 is the static floor, body 1 the box; the normal points
    /// up, from the floor to the box.
    fn resting_box() -> (RigidBodyState, Vec<RigidContact>, IntegratorConfig) {
        let mut state = RigidBodyState::new();
        push_body(&mut state, Vec3::ZERO, 0.0, Vec3::ZERO); // static floor
        push_body(&mut state, Vec3::new(0.0, 0.5, 0.0), 1.0, Vec3::splat(6.0)); // box
        let normal = Vec3::new(0.0, 1.0, 0.0);
        // Contact point at the box's base, directly below its centre.
        let contact = RigidContact::new(
            1,
            0,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            normal,
            0.0,
        )
        .with_friction(0.5);
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 0.0, 0.0);
        (state, vec![contact], integrator)
    }

    #[test]
    fn free_body_falls_under_gravity_without_contacts() {
        // With no contacts the stepper still integrates: a free body must gain
        // exactly `gravity * dt` of downward velocity over the frame, since the
        // substepped symplectic velocity update telescopes to the full step.
        let mut state = RigidBodyState::new();
        push_body(&mut state, Vec3::new(0.0, 10.0, 0.0), 1.0, Vec3::ZERO);
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 0.0, 0.0);
        let dt = 1.0 / 60.0;
        cpu_solve_contacts_tgs(
            &mut state,
            &mut [],
            &integrator,
            &TgsContactConfig::DEFAULT,
            dt,
        )
        .expect("step");

        let expected_vy = -9.81 * dt;
        assert!(
            (state.linear_velocities[0].y - expected_vy).abs() < 1e-5,
            "free fall velocity {} != {}",
            state.linear_velocities[0].y,
            expected_vy
        );
        assert!(state.positions[0].y < 10.0, "body did not fall");
    }

    #[test]
    fn resting_box_settles_without_gaining_energy() {
        // A box dropped onto a floor must settle: after many frames its speed
        // decays toward zero and its penetration stabilises near the slop band
        // rather than growing or oscillating (the Baumgarte energy leak would
        // leave a persistent jitter).
        let (mut state, mut contacts, integrator) = resting_box();
        let tgs = TgsContactConfig::DEFAULT;
        let dt = 1.0 / 60.0;

        for _ in 0..240 {
            cpu_solve_contacts_tgs(&mut state, &mut contacts, &integrator, &tgs, dt).expect("step");
        }

        let speed = state.linear_velocities[1].length();
        assert!(speed < 0.05, "box did not settle, residual speed {speed}");
        // Penetration is held small and non-negative-ish: within a few slops.
        let penetration = contacts[0].penetration;
        assert!(
            penetration < 4.0 * tgs.slop && penetration > -0.05,
            "penetration not controlled: {penetration}"
        );
        // The box must not have tunnelled through the floor.
        assert!(
            state.positions[1].y > 0.4,
            "box sank through the floor: {}",
            state.positions[1].y
        );
    }

    #[test]
    fn resting_box_does_not_accumulate_speed_over_time() {
        // Stronger anti-leak assertion: the speed late in the simulation is no
        // larger than it was once initial settling finished — energy is not
        // being manufactured frame over frame.
        let (mut state, mut contacts, integrator) = resting_box();
        let tgs = TgsContactConfig::DEFAULT;
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_contacts_tgs(&mut state, &mut contacts, &integrator, &tgs, dt).expect("step");
        }
        let early_speed = state.linear_velocities[1].length();
        for _ in 0..240 {
            cpu_solve_contacts_tgs(&mut state, &mut contacts, &integrator, &tgs, dt).expect("step");
        }
        let late_speed = state.linear_velocities[1].length();
        assert!(
            late_speed <= early_speed + 1e-3,
            "speed grew over time: early {early_speed}, late {late_speed}"
        );
    }

    #[test]
    fn head_on_elastic_collision_restores_separation_speed() {
        // Two equal point masses approach head-on with restitution 1. After the
        // step their relative separation speed should match their approach
        // speed (elastic), and linear momentum is conserved.
        let mut state = RigidBodyState::new();
        push_body(&mut state, Vec3::new(-0.6, 0.0, 0.0), 1.0, Vec3::ZERO);
        push_body(&mut state, Vec3::new(0.6, 0.0, 0.0), 1.0, Vec3::ZERO);
        state.linear_velocities[0] = Vec3::new(2.0, 0.0, 0.0);
        state.linear_velocities[1] = Vec3::new(-2.0, 0.0, 0.0);
        // Normal points from b (right) to a (left). Bodies already touching.
        let normal = Vec3::new(-1.0, 0.0, 0.0);
        let mut contacts =
            [RigidContact::new(0, 1, Vec3::ZERO, Vec3::ZERO, normal, 0.0).with_restitution(1.0)];
        // Zero gravity to isolate the collision response.
        let integrator = IntegratorConfig::new(Vec3::ZERO, 1, 0.0, 0.0);
        cpu_solve_contacts_tgs(
            &mut state,
            &mut contacts,
            &integrator,
            &TgsContactConfig::DEFAULT,
            1.0 / 60.0,
        )
        .expect("step");

        let approach = 2.0 - (-2.0); // 4 m/s closing
        let separation = state.linear_velocities[1].x - state.linear_velocities[0].x;
        assert!(
            (separation - approach).abs() < 0.1,
            "separation speed {separation} != approach {approach}"
        );
        let momentum = state.linear_velocities[0].x + state.linear_velocities[1].x;
        assert!(momentum.abs() < 1e-3, "momentum not conserved: {momentum}");
    }

    #[test]
    fn off_centre_hit_induces_spin() {
        // A body struck off its centre must acquire angular velocity: the
        // contact's angular arm converts part of the normal impulse into torque.
        let mut state = RigidBodyState::new();
        push_body(&mut state, Vec3::ZERO, 0.0, Vec3::ZERO); // static wall
        push_body(&mut state, Vec3::new(1.0, 0.0, 0.0), 1.0, Vec3::splat(6.0));
        state.linear_velocities[1] = Vec3::new(-2.0, 0.0, 0.0);
        // Contact offset in +Y from the body's centre; normal points +X (from
        // wall to body).
        let normal = Vec3::new(1.0, 0.0, 0.0);
        let mut contacts =
            [
                RigidContact::new(1, 0, Vec3::new(-0.5, 0.5, 0.0), Vec3::ZERO, normal, 0.0)
                    .with_restitution(1.0),
            ];
        let integrator = IntegratorConfig::new(Vec3::ZERO, 1, 0.0, 0.0);
        cpu_solve_contacts_tgs(
            &mut state,
            &mut contacts,
            &integrator,
            &TgsContactConfig::DEFAULT,
            1.0 / 60.0,
        )
        .expect("step");

        let spin = state.angular_velocities[1].length();
        assert!(spin > 1e-2, "off-centre hit produced no spin: {spin}");
    }

    #[test]
    fn three_body_stack_settles() {
        // A floor and two stacked boxes under gravity must converge to near
        // rest once the load propagates through the stack.
        let mut state = RigidBodyState::new();
        push_body(&mut state, Vec3::ZERO, 0.0, Vec3::ZERO); // static floor
        push_body(&mut state, Vec3::new(0.0, 0.5, 0.0), 1.0, Vec3::splat(6.0)); // box 1
        push_body(&mut state, Vec3::new(0.0, 1.5, 0.0), 1.0, Vec3::splat(6.0)); // box 2
        let up = Vec3::new(0.0, 1.0, 0.0);
        let mut contacts = vec![
            RigidContact::new(
                1,
                0,
                Vec3::new(0.0, -0.5, 0.0),
                Vec3::new(0.0, 0.5, 0.0),
                up,
                0.0,
            ),
            RigidContact::new(
                2,
                1,
                Vec3::new(0.0, -0.5, 0.0),
                Vec3::new(0.0, 0.5, 0.0),
                up,
                0.0,
            ),
        ];
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 0.0, 0.0);
        let tgs = TgsContactConfig::DEFAULT;
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            cpu_solve_contacts_tgs(&mut state, &mut contacts, &integrator, &tgs, dt).expect("step");
        }
        let s1 = state.linear_velocities[1].length();
        let s2 = state.linear_velocities[2].length();
        assert!(s1 < 0.1 && s2 < 0.1, "stack did not settle: {s1} {s2}");
    }

    #[test]
    fn rigid_limit_hertz_zero_settles_without_exploding() {
        // A non-positive contact frequency selects the rigid Baumgarte limit;
        // the resting box must still settle rather than diverge.
        let (mut state, mut contacts, integrator) = resting_box();
        let tgs = TgsContactConfig::new(0.0, 10.0, 0.005, 0.5, 1);
        let dt = 1.0 / 60.0;
        for _ in 0..240 {
            cpu_solve_contacts_tgs(&mut state, &mut contacts, &integrator, &tgs, dt).expect("step");
        }
        let speed = state.linear_velocities[1].length();
        assert!(
            speed.is_finite() && speed < 0.2,
            "rigid limit unstable: {speed}"
        );
        assert!(
            state.positions[1].y > 0.4 && state.positions[1].y.is_finite(),
            "rigid limit box misplaced: {}",
            state.positions[1].y
        );
    }

    #[test]
    fn empty_state_is_a_no_op() {
        let mut state = RigidBodyState::new();
        cpu_solve_contacts_tgs(
            &mut state,
            &mut [],
            &IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 0.0, 0.0),
            &TgsContactConfig::DEFAULT,
            1.0 / 60.0,
        )
        .expect("empty");
        assert!(state.is_empty());
    }

    #[test]
    fn non_positive_dt_is_a_no_op() {
        let (mut state, mut contacts, integrator) = resting_box();
        let before = state.positions[1];
        cpu_solve_contacts_tgs(
            &mut state,
            &mut contacts,
            &integrator,
            &TgsContactConfig::DEFAULT,
            0.0,
        )
        .expect("zero dt");
        assert_eq!(state.positions[1], before);
    }
}
