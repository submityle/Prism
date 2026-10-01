//! The `CPU` golden reference for the 6-DOF rigid-body integrator.
//!
//! [`cpu_integrate`] advances a [`RigidBodyState`] under per-body external
//! forces and torques for one frame, splitting the frame time step into equal
//! substeps and integrating each body independently. It is the authoritative
//! twin the device kernel in [`gpu`](super::gpu) must reproduce frame for
//! frame; the parity test (`tests/rigid_integrate_parity.rs`) bounds their
//! divergence to floating-point reassociation noise.
//!
//! # The integration scheme
//!
//! Each substep of size `h` advances a body in two decoupled halves:
//!
//! * **Linear.** Semi-implicit (symplectic) Euler: the acceleration
//!   `gravity + force * inverse_mass` updates the velocity, the (damped)
//!   velocity then updates the position. Integrating velocity before position
//!   is what makes the scheme symplectic and keeps a free body's energy from
//!   drifting.
//! * **Angular.** The explicit-gyroscopic form of Euler's rigid-body equations.
//!   The world-space angular velocity and torque are rotated into the body
//!   frame (where the inertia is the stored diagonal), the diagonal inverse
//!   inertia turns torque into angular acceleration, and the gyroscopic term
//!   `omega x (I * omega)` — the coupling that makes an asymmetric body tumble —
//!   is subtracted explicitly. The updated body-frame angular velocity is
//!   rotated back to world space and the orientation is advanced by the
//!   quaternion kinematic equation `q' = normalize(q + 0.5 * (omega_q * q) * h)`.
//!
//! # Honest scope
//!
//! This is an *integrator*, not a constraint or contact solver: it moves free
//! bodies under their applied loads and nothing else. The gyroscopic term is
//! integrated **explicitly**, which is exact in the parity sense (the device
//! does the identical arithmetic) and correct for the moderate spin rates of
//! typical gameplay, but an explicit treatment can gain energy for a body spun
//! fast about its intermediate axis; an implicit-gyroscopic variant and the
//! contact / joint solvers that consume this integrator are later slices.
//!
//! # Quaternion arithmetic
//!
//! The orientation dynamics are written in scalar form over `[f32; 4]`
//! `(x, y, z, w)` arrays rather than through [`glam::Quat`]'s methods, so the
//! exact sequence of multiplies and adds can be mirrored byte-for-byte-intent
//! in the device shader. `glam::Quat` is used only to store the result.
//!
//! Provenance: Euler's rigid-body equations with explicit gyroscopic coupling
//! and the quaternion kinematic equation (Baraff & Witkin; standard rigid-body
//! dynamics). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};

use super::body::RigidBodyState;
use super::config::{IntegratorConfig, RigidError};
use super::gyroscopic::{implicit_gyroscopic_body, GyroscopicConfig, GyroscopicMode};

/// Guard below which a quaternion is treated as degenerate and reset to
/// identity. Matches `EPSILON` in `shaders/rigid_integrate.wgsl` (`f32::EPSILON`).
const EPSILON: f32 = 1.192_092_9e-7;

/// Advances `state` by `dt` seconds under the per-body external `forces` and
/// `torques`, integrating in place.
///
/// `forces` and `torques` are world-space loads applied at each body's centre
/// of mass; either may be empty, which is treated as all-zero. When non-empty,
/// each must hold exactly one entry per body.
///
/// # Errors
///
/// Returns [`RigidError::InvalidConfig`] when `config` fails validation, or
/// [`RigidError::InconsistentState`] when the state arrays disagree in length
/// or a non-empty `forces`/`torques` slice does not match the body count. Does
/// nothing (returns `Ok`) when there are no bodies or `dt` is non-positive.
pub fn cpu_integrate(
    state: &mut RigidBodyState,
    forces: &[Vec3],
    torques: &[Vec3],
    config: &IntegratorConfig,
    dt: f32,
) -> Result<(), RigidError> {
    integrate_impl(
        state,
        forces,
        torques,
        config,
        &GyroscopicConfig::explicit(),
        dt,
    )
}

/// Advances `state` by `dt` seconds exactly like [`cpu_integrate`], but with the
/// gyroscopic coupling of the angular update integrated according to `gyro`.
///
/// With [`GyroscopicConfig::explicit`] this is identical to [`cpu_integrate`].
/// With [`GyroscopicConfig::implicit`] the angular update solves the
/// backward-Euler gyroscopic equation (see [`gyroscopic`](super::gyroscopic))
/// per substep for every body whose body-frame inertia is strictly positive on
/// all three axes; a body with any locked axis falls back to the explicit path
/// so the behaviour never becomes singular.
///
/// # Errors
///
/// Returns the same errors as [`cpu_integrate`]: [`RigidError::InvalidConfig`]
/// when `config` fails validation, or [`RigidError::InconsistentState`] when the
/// state arrays disagree in length or a non-empty `forces`/`torques` slice does
/// not match the body count. Does nothing (returns `Ok`) when there are no
/// bodies or `dt` is non-positive.
pub fn cpu_integrate_gyro(
    state: &mut RigidBodyState,
    forces: &[Vec3],
    torques: &[Vec3],
    config: &IntegratorConfig,
    gyro: &GyroscopicConfig,
    dt: f32,
) -> Result<(), RigidError> {
    integrate_impl(state, forces, torques, config, gyro, dt)
}

/// Shared integration driver backing both [`cpu_integrate`] and
/// [`cpu_integrate_gyro`]. The two public entry points differ only in the
/// [`GyroscopicConfig`] they pass; every other step — validation, substep
/// splitting, the semi-implicit linear update, and the quaternion kinematic
/// orientation advance — is identical.
fn integrate_impl(
    state: &mut RigidBodyState,
    forces: &[Vec3],
    torques: &[Vec3],
    config: &IntegratorConfig,
    gyro: &GyroscopicConfig,
    dt: f32,
) -> Result<(), RigidError> {
    config.validate()?;
    if !state.is_consistent() {
        return Err(RigidError::InconsistentState {
            reason: "per-body arrays must have equal length",
        });
    }
    let n = state.len();
    if !forces.is_empty() && forces.len() != n {
        return Err(RigidError::InconsistentState {
            reason: "forces length must match body count or be empty",
        });
    }
    if !torques.is_empty() && torques.len() != n {
        return Err(RigidError::InconsistentState {
            reason: "torques length must match body count or be empty",
        });
    }
    if state.is_empty() || dt <= 0.0 {
        return Ok(());
    }

    let substeps = config.effective_substeps();
    let h = dt / substeps as f32;
    if h <= 0.0 {
        return Ok(());
    }
    let half_h = 0.5 * h;
    let linear_damping_scale = (1.0 - config.linear_damping * h).max(0.0);
    let angular_damping_scale = (1.0 - config.angular_damping * h).max(0.0);
    let implicit_gyro = matches!(gyro.mode, GyroscopicMode::Implicit);
    let gyro_iterations = gyro.effective_iterations();

    for i in 0..n {
        let force = if forces.is_empty() {
            Vec3::ZERO
        } else {
            forces[i]
        };
        let torque = if torques.is_empty() {
            Vec3::ZERO
        } else {
            torques[i]
        };
        let inverse_mass = state.inverse_masses[i];
        let inverse_inertia = state.inverse_inertias[i];
        let can_translate = inverse_mass > 0.0;
        let can_rotate =
            inverse_inertia.x > 0.0 || inverse_inertia.y > 0.0 || inverse_inertia.z > 0.0;
        let inertia = principal_inertia(inverse_inertia);
        // The implicit solve needs a strictly positive inertia on every axis so
        // its Newton Jacobian is non-singular; a body with any locked axis falls
        // back to the explicit path. The same guard runs on the device.
        let use_implicit = implicit_gyro && inertia.min_element() > 0.0;

        let mut position = state.positions[i];
        let mut linear_velocity = state.linear_velocities[i];
        let orientation = state.orientations[i];
        let mut q = [orientation.x, orientation.y, orientation.z, orientation.w];
        let mut angular_velocity = state.angular_velocities[i];

        for _ in 0..substeps {
            if can_translate {
                let acceleration = config.gravity + force * inverse_mass;
                linear_velocity = (linear_velocity + acceleration * h) * linear_damping_scale;
                position += linear_velocity * h;
            }
            if can_rotate {
                let conjugate = quat_conj(q);
                // Pull torque and angular velocity into the body frame, where
                // the inertia is the stored diagonal.
                let torque_body = quat_rotate(conjugate, torque);
                let angular_acceleration_body = inverse_inertia * torque_body;
                let mut omega_body = quat_rotate(conjugate, angular_velocity);
                omega_body += angular_acceleration_body * h;
                if use_implicit {
                    // Implicit (backward-Euler) gyroscopic coupling: solve for
                    // the end-of-substep body-frame angular velocity.
                    omega_body =
                        implicit_gyroscopic_body(omega_body, inertia, h, gyro_iterations);
                } else {
                    // Explicit gyroscopic coupling: subtract omega x (I * omega),
                    // expressed as an angular acceleration via the inverse inertia.
                    let angular_momentum_body = inertia * omega_body;
                    let gyroscopic = omega_body.cross(angular_momentum_body);
                    omega_body -= inverse_inertia * gyroscopic * h;
                }
                // Back to world space and apply angular damping.
                angular_velocity = quat_rotate(q, omega_body) * angular_damping_scale;
                // Advance the orientation by the quaternion kinematic equation.
                let omega_quat = [
                    angular_velocity.x,
                    angular_velocity.y,
                    angular_velocity.z,
                    0.0,
                ];
                let dq = quat_mul(omega_quat, q);
                let integrated = [
                    q[0] + dq[0] * half_h,
                    q[1] + dq[1] * half_h,
                    q[2] + dq[2] * half_h,
                    q[3] + dq[3] * half_h,
                ];
                q = quat_normalize(integrated);
            }
        }

        state.positions[i] = position;
        state.linear_velocities[i] = linear_velocity;
        state.orientations[i] = Quat::from_xyzw(q[0], q[1], q[2], q[3]);
        state.angular_velocities[i] = angular_velocity;
    }
    Ok(())
}

/// Recovers the body-frame principal inertia diagonal from its inverse,
/// mapping a zero (locked) axis to zero inertia so it contributes no
/// gyroscopic coupling.
fn principal_inertia(inverse_inertia: Vec3) -> Vec3 {
    Vec3::new(
        if inverse_inertia.x > 0.0 {
            1.0 / inverse_inertia.x
        } else {
            0.0
        },
        if inverse_inertia.y > 0.0 {
            1.0 / inverse_inertia.y
        } else {
            0.0
        },
        if inverse_inertia.z > 0.0 {
            1.0 / inverse_inertia.z
        } else {
            0.0
        },
    )
}

/// Hamilton product `a * b` of two quaternions stored as `(x, y, z, w)`.
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Conjugate (inverse rotation) of a unit quaternion stored as `(x, y, z, w)`.
fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Rotates `v` by the unit quaternion `q`, using the expanded sandwich product
/// `2 (u . v) u + (s^2 - u . u) v + 2 s (u x v)` with `u = q.xyz`, `s = q.w`.
fn quat_rotate(q: [f32; 4], v: Vec3) -> Vec3 {
    let u = Vec3::new(q[0], q[1], q[2]);
    let s = q[3];
    u * (2.0 * u.dot(v)) + v * (s * s - u.dot(u)) + u.cross(v) * (2.0 * s)
}

/// Normalises a quaternion stored as `(x, y, z, w)`, falling back to identity
/// when its length is below [`EPSILON`].
fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let length_squared = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    let length = length_squared.sqrt();
    if length > EPSILON {
        [q[0] / length, q[1] / length, q[2] / length, q[3] / length]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// World-space angular momentum `L = R (I (R^T omega))` of body `i`.
    fn world_angular_momentum(state: &RigidBodyState, i: usize) -> Vec3 {
        let o = state.orientations[i];
        let q = [o.x, o.y, o.z, o.w];
        let inertia = principal_inertia(state.inverse_inertias[i]);
        let omega_body = quat_rotate(quat_conj(q), state.angular_velocities[i]);
        quat_rotate(q, inertia * omega_body)
    }

    #[test]
    fn free_fall_matches_closed_form() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::ZERO);
        let g = Vec3::new(0.0, -9.81, 0.0);
        let substeps = 100u32;
        let config = IntegratorConfig::new(g, substeps, 0.0, 0.0);
        let dt = 1.0;
        cpu_integrate(&mut state, &[], &[], &config, dt).expect("integrate");

        // Semi-implicit Euler is exact in velocity: v = g * t.
        let v = state.linear_velocities[0];
        assert!((v.y - g.y * dt).abs() < 1e-3, "velocity {v:?}");

        // Discrete position closed form: x = 0.5 * g * h^2 * S (S + 1).
        let h = dt / substeps as f32;
        let s = substeps as f32;
        let expected_y = 0.5 * g.y * h * h * s * (s + 1.0);
        assert!(
            (state.positions[0].y - expected_y).abs() < 1e-2,
            "position {} vs expected {expected_y}",
            state.positions[0].y
        );
    }

    #[test]
    fn static_body_ignores_gravity() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, 0.0, Vec3::ONE);
        let config = IntegratorConfig::default();
        cpu_integrate(&mut state, &[], &[], &config, 1.0).expect("integrate");
        assert_eq!(state.positions[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(state.linear_velocities[0], Vec3::ZERO);
    }

    #[test]
    fn locked_body_does_not_rotate() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::ZERO);
        state.angular_velocities[0] = Vec3::new(5.0, 0.0, 0.0);
        let config = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        cpu_integrate(&mut state, &[], &[Vec3::new(10.0, 0.0, 0.0)], &config, 1.0)
            .expect("integrate");
        assert_eq!(state.orientations[0], Quat::IDENTITY);
        assert_eq!(state.angular_velocities[0], Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn isotropic_sphere_preserves_spin_rate() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(2.0));
        let omega0 = Vec3::new(0.3, 1.7, -0.9);
        state.angular_velocities[0] = omega0;
        let config = IntegratorConfig::new(Vec3::ZERO, 500, 0.0, 0.0);
        cpu_integrate(&mut state, &[], &[], &config, 1.0).expect("integrate");

        // No torque and an isotropic inertia means zero gyroscopic coupling, so
        // the world-space angular velocity magnitude is conserved.
        let speed0 = omega0.length();
        let speed = state.angular_velocities[0].length();
        assert!(
            (speed - speed0).abs() < 1e-3,
            "spin rate {speed} vs {speed0}"
        );

        // The orientation stays a unit quaternion.
        let o = state.orientations[0];
        let len = (o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w).sqrt();
        assert!((len - 1.0).abs() < 1e-4, "quaternion length {len}");
    }

    #[test]
    fn torque_free_asymmetric_body_conserves_angular_momentum() {
        let mut state = RigidBodyState::new();
        // Asymmetric inertia (1, 2, 4): a classic tumbling top.
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::new(1.0, 0.5, 0.25));
        state.angular_velocities[0] = Vec3::new(1.0, 1.0, 1.0);
        let config = IntegratorConfig::new(Vec3::ZERO, 4000, 0.0, 0.0);

        let l0 = world_angular_momentum(&state, 0);
        cpu_integrate(&mut state, &[], &[], &config, 1.0).expect("integrate");
        let l1 = world_angular_momentum(&state, 0);

        // The explicit gyroscopic term conserves world-space angular momentum
        // for a torque-free body to within integration error.
        let drift = (l1 - l0).length();
        assert!(
            drift < 0.01 * l0.length(),
            "angular momentum drifted by {drift} (|L0| = {})",
            l0.length()
        );
    }

    #[test]
    fn constant_torque_about_principal_axis_accelerates_linearly() {
        let mut state = RigidBodyState::new();
        let inv_inertia = Vec3::new(0.5, 0.25, 0.125);
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, inv_inertia);
        let torque = Vec3::new(2.0, 0.0, 0.0);
        let config = IntegratorConfig::new(Vec3::ZERO, 1000, 0.0, 0.0);
        let dt = 1.0;
        cpu_integrate(&mut state, &[], &[torque], &config, dt).expect("integrate");

        // Torque about a single principal axis keeps omega parallel to that
        // axis (no gyroscopic coupling), so omega.x = inv_inertia.x * tau.x * t.
        let omega = state.angular_velocities[0];
        let expected_x = inv_inertia.x * torque.x * dt;
        assert!((omega.x - expected_x).abs() < 1e-2, "omega.x {}", omega.x);
        assert!(
            omega.y.abs() < 1e-3 && omega.z.abs() < 1e-3,
            "omega {omega:?}"
        );
    }

    #[test]
    fn orientation_stays_unit_over_many_frames() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::new(1.0, 2.0, 3.0));
        state.angular_velocities[0] = Vec3::new(2.0, -1.0, 0.5);
        let config = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        for _ in 0..600 {
            cpu_integrate(&mut state, &[], &[], &config, 1.0 / 60.0).expect("integrate");
        }
        let o = state.orientations[0];
        let len = (o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w).sqrt();
        assert!((len - 1.0).abs() < 1e-4, "quaternion length {len}");
    }

    #[test]
    fn empty_state_and_nonpositive_dt_are_noops() {
        let mut empty = RigidBodyState::new();
        let config = IntegratorConfig::default();
        cpu_integrate(&mut empty, &[], &[], &config, 1.0).expect("empty integrate");
        assert!(empty.is_empty());

        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::ONE);
        cpu_integrate(&mut state, &[], &[], &config, 0.0).expect("zero dt");
        assert_eq!(state.positions[0], Vec3::ZERO);
    }

    #[test]
    fn mismatched_force_length_errors() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::ONE);
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::ONE);
        let config = IntegratorConfig::default();
        let err = cpu_integrate(&mut state, &[Vec3::ZERO], &[], &config, 1.0).unwrap_err();
        assert!(matches!(err, RigidError::InconsistentState { .. }));
    }
}
