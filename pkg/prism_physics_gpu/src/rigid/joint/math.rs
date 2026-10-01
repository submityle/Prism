//! Shared quaternion and inertia math for the `XPBD` joint steppers.
//!
//! Every joint `CPU` golden (`spherical_cpu`, `revolute_cpu`, ...) projects its
//! constraints with the *same* expanded quaternion sandwich product, Hamilton
//! product, world-space inverse-inertia transform, and `XPBD` rotation-delta
//! application. Keeping that arithmetic in one module guarantees the joint
//! family shares a single notion of "numerically zero" and one rotation formula,
//! which is what lets every joint `GPU` twin reproduce its golden bit-for-bit and
//! lets the joint, contact, and integrator stages compose without drift.
//!
//! The helpers operate on quaternions stored as `(x, y, z, w)` arrays so the
//! `CPU` reference and the `WGSL` shaders apply the identical component order.
//!
//! Provenance: standard quaternion kinematics and world-space inverse inertia
//! (Baraff & Witkin), with the direct `XPBD` rotation-delta application of
//! Müller et al., "Detailed Rigid Body Simulation with XPBD". No Unreal Engine
//! source or derived code.

use glam::{Quat, Vec3};

/// Length below which a separation vector, an alignment error, or an effective
/// mass is treated as degenerate and the correction is skipped. Matches the
/// `EPSILON` the rigid integrator and contact solver use, so every stage shares
/// one notion of "numerically zero".
pub(super) const EPSILON: f32 = 1.192_092_9e-7;

/// Whether the solver may rotate a body with the given body-frame inverse
/// inertia (any non-zero principal axis).
pub(super) fn can_rotate(inverse_inertia: Vec3) -> bool {
    inverse_inertia.x > 0.0 || inverse_inertia.y > 0.0 || inverse_inertia.z > 0.0
}

/// Advances an orientation by its world-space angular velocity over `h` via the
/// quaternion kinematic equation `q' = normalize(q + 0.5 h [omega, 0] q)`,
/// matching the rigid integrator. Falls back to the input orientation if the
/// integrated quaternion is degenerate.
pub(super) fn integrate_orientation(orientation: Quat, omega: Vec3, h: f32) -> Quat {
    let q = quat_array(orientation);
    let dq = quat_mul([omega.x, omega.y, omega.z, 0.0], q);
    let half_h = 0.5 * h;
    let integrated = [
        q[0] + dq[0] * half_h,
        q[1] + dq[1] * half_h,
        q[2] + dq[2] * half_h,
        q[3] + dq[3] * half_h,
    ];
    normalize_or(integrated, orientation)
}

/// Applies a direct `XPBD` rotation delta `omega` (a rotation vector, not scaled
/// by any time step) to an orientation: `q' = normalize(q + 0.5 [omega, 0] q)`
/// (Müller et al.). Falls back to the input orientation if the result is
/// degenerate.
pub(super) fn apply_rotation_delta(orientation: Quat, omega: Vec3) -> Quat {
    let q = quat_array(orientation);
    let dq = quat_mul([omega.x, omega.y, omega.z, 0.0], q);
    let integrated = [
        q[0] + 0.5 * dq[0],
        q[1] + 0.5 * dq[1],
        q[2] + 0.5 * dq[2],
        q[3] + 0.5 * dq[3],
    ];
    normalize_or(integrated, orientation)
}

/// Applies the world-space inverse inertia `R diag(inv_inertia) R^T` to `v`:
/// rotate `v` into the body frame, scale by the diagonal inverse inertia, and
/// rotate back. Mirrors the contact solver's identically named helper.
pub(super) fn world_inv_inertia_apply(orientation: Quat, inv_inertia: Vec3, v: Vec3) -> Vec3 {
    let q = quat_array(orientation);
    let body = quat_rotate(quat_conj(q), v);
    quat_rotate(q, inv_inertia * body)
}

/// Rotates `v` by the unit quaternion `orientation`, routing through the
/// array-based [`quat_rotate`] so the `CPU` reference and the shaders share one
/// rotation formula.
pub(super) fn rotate(orientation: Quat, v: Vec3) -> Vec3 {
    quat_rotate(quat_array(orientation), v)
}

/// Normalises a quaternion stored as `(x, y, z, w)`, returning `fallback` when
/// the length is below [`EPSILON`].
pub(super) fn normalize_or(q: [f32; 4], fallback: Quat) -> Quat {
    let length = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if length > EPSILON {
        Quat::from_xyzw(q[0] / length, q[1] / length, q[2] / length, q[3] / length)
    } else {
        fallback
    }
}

/// Unpacks a [`Quat`] into the `(x, y, z, w)` array the quaternion helpers
/// operate on.
pub(super) fn quat_array(q: Quat) -> [f32; 4] {
    [q.x, q.y, q.z, q.w]
}

/// Hamilton product `a * b` of two quaternions stored as `(x, y, z, w)`.
pub(super) fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
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
pub(super) fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Rotates `v` by the unit quaternion `q` via the expanded sandwich product
/// `2 (u . v) u + (s^2 - u . u) v + 2 s (u x v)` with `u = q.xyz`, `s = q.w`.
pub(super) fn quat_rotate(q: [f32; 4], v: Vec3) -> Vec3 {
    let u = Vec3::new(q[0], q[1], q[2]);
    let s = q[3];
    u * (2.0 * u.dot(v)) + v * (s * s - u.dot(u)) + u.cross(v) * (2.0 * s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn quat_rotate_matches_glam() {
        let q = Quat::from_axis_angle(Vec3::new(0.3, -0.7, 0.5).normalize(), 1.2);
        let v = Vec3::new(1.0, -2.0, 3.0);
        let ours = quat_rotate(quat_array(q), v);
        let glam = q * v;
        assert!((ours - glam).length() < 1.0e-5);
    }

    #[test]
    fn quat_mul_matches_glam() {
        let a = Quat::from_axis_angle(Vec3::X, 0.4);
        let b = Quat::from_axis_angle(Vec3::Y, -0.9);
        let ours = quat_mul(quat_array(a), quat_array(b));
        let glam = quat_array(a * b);
        for i in 0..4 {
            assert!((ours[i] - glam[i]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn world_inv_inertia_identity_orientation_scales_axiswise() {
        let out = world_inv_inertia_apply(Quat::IDENTITY, Vec3::new(2.0, 3.0, 4.0), Vec3::ONE);
        assert!((out - Vec3::new(2.0, 3.0, 4.0)).length() < 1.0e-6);
    }

    #[test]
    fn can_rotate_rejects_fully_locked_inertia() {
        assert!(!can_rotate(Vec3::ZERO));
        assert!(can_rotate(Vec3::new(0.0, 0.0, 1.0e-3)));
    }
}
