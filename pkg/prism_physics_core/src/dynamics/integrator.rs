//! Semi-implicit (symplectic) Euler integrator.
//!
//! The integrator advances each dynamic body one step by first updating
//! velocity, then using the new velocity to update position and orientation.
//! This ordering (velocity-before-position) is the semi-implicit Euler scheme,
//! which is stable for the gravity + damping systems used here.
//!
//! # Math
//!
//! For a dynamic body with gravity `g` and time step `dt`:
//!
//! - Linear: `v += g * dt`, then apply damping `v *= 1 / (1 + c_lin * dt)`,
//!   then `x += v * dt`.
//! - Angular: apply damping `w *= 1 / (1 + c_ang * dt)`, then integrate the
//!   orientation quaternion `q` using `q += 0.5 * (w_quat * q) * dt` followed
//!   by renormalization, where `w_quat = (w.x, w.y, w.z, 0)`.
//!
//! Static and kinematic bodies are not integrated. None of this is derived from
//! Unreal Engine source; it is the standard textbook rigid-body integration.

use crate::state::storage::BodyStorage;
use glam::{Quat, Vec3};

/// A stateless semi-implicit Euler integrator over [`BodyStorage`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Integrator;

impl Integrator {
    /// Advances all dynamic bodies in `bodies` by `dt` seconds under `gravity`.
    ///
    /// Static and kinematic bodies are left untouched. This is the free method
    /// entry point; [`Integrator::integrate`] on an instance forwards here.
    pub fn integrate(bodies: &mut BodyStorage, gravity: Vec3, dt: f32) {
        bodies.for_each_dynamic_mut(|pos, ori, linv, angv, _mass, lin_damp, ang_damp| {
            // Semi-implicit Euler: update velocities first.
            *linv += gravity * dt;

            // Exponential-style damping that is unconditionally stable.
            let lin_factor = 1.0 / (1.0 + lin_damp * dt);
            let ang_factor = 1.0 / (1.0 + ang_damp * dt);
            *linv *= lin_factor;
            *angv *= ang_factor;

            // Integrate position with the updated linear velocity.
            *pos += *linv * dt;

            // Integrate orientation: q' = normalize(q + 0.5 * (w_quat * q) * dt).
            let omega = Quat::from_xyzw(angv.x, angv.y, angv.z, 0.0);
            let dq = omega.mul_quat(*ori);
            let q = ori.to_array();
            let d = dq.to_array();
            let half_dt = 0.5 * dt;
            let integrated = Quat::from_xyzw(
                q[0] + d[0] * half_dt,
                q[1] + d[1] * half_dt,
                q[2] + d[2] * half_dt,
                q[3] + d[3] * half_dt,
            );
            *ori = normalize_or_identity(integrated);
        });
    }
}

/// Normalizes `q`, falling back to the identity if the quaternion has
/// collapsed to (near) zero length (which cannot happen for valid inputs but
/// keeps the integrator total).
fn normalize_or_identity(q: Quat) -> Quat {
    let len_sq = q.length_squared();
    if len_sq > 1e-20 {
        q.normalize()
    } else {
        Quat::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::body::BodyDesc;

    #[test]
    fn dynamic_body_falls_under_gravity() {
        let mut bodies = BodyStorage::new();
        let h = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let g = Vec3::new(0.0, -10.0, 0.0);
        Integrator::integrate(&mut bodies, g, 0.5);
        // Semi-implicit Euler: v = -5, x = v*dt = -2.5 after one 0.5s step.
        let v = bodies.linear_velocity(h).unwrap();
        let p = bodies.position(h).unwrap();
        assert!((v.y - -5.0).abs() < 1e-5);
        assert!((p.y - -2.5).abs() < 1e-5);
    }

    #[test]
    fn angular_velocity_rotates_orientation() {
        let mut bodies = BodyStorage::new();
        let desc = BodyDesc::dynamic_at(Vec3::ZERO).with_angular_velocity(Vec3::new(0.0, 1.0, 0.0));
        let h = bodies.insert(desc);
        for _ in 0..100 {
            Integrator::integrate(&mut bodies, Vec3::ZERO, 0.01);
        }
        let q = bodies.orientation(h).unwrap();
        // Orientation must remain a unit quaternion and have actually rotated.
        assert!((q.length() - 1.0).abs() < 1e-4);
        assert!(q.angle_between(Quat::IDENTITY) > 0.5);
    }
}
