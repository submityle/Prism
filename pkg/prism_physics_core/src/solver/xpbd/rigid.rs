//! Rigid-body helpers shared by the XPBD position and velocity passes.
//!
//! These implement the standard rigid-body impulse algebra used by
//! position-based dynamics: mapping local inverse inertia into world space,
//! computing the generalized inverse mass along a direction at a lever arm, and
//! applying positional / velocity impulses at a contact offset.
//!
//! # Provenance
//!
//! The formulas are the textbook rigid-body results used by Müller et al.,
//! "Detailed Rigid Body Simulation with Extended Position Based Dynamics"
//! (2020), and contain no Unreal Engine source or derived code.

use glam::{Mat3, Quat, Vec3};

/// Builds the world-space inverse inertia tensor for a body.
///
/// The stored `inv_inertia` is the diagonal inverse inertia in the body's
/// principal frame; rotating it by the orientation `q` gives the world-space
/// tensor `R * diag(inv_inertia) * R^T`.
#[must_use]
pub fn inv_inertia_world(inv_inertia: Vec3, q: Quat) -> Mat3 {
    let r = Mat3::from_quat(q);
    let scaled = Mat3::from_cols(
        r.x_axis * inv_inertia.x,
        r.y_axis * inv_inertia.y,
        r.z_axis * inv_inertia.z,
    );
    scaled * r.transpose()
}

/// Returns the generalized inverse mass of a body for a unit `direction`
/// applied at world lever arm `r` (offset from the center of mass).
///
/// This is `inv_mass + (r x direction) . I_inv_world (r x direction)`, the
/// effective inverse mass seen by a constraint acting along `direction`.
#[must_use]
pub fn generalized_inverse_mass(
    inv_mass: f32,
    inv_inertia_world: Mat3,
    r: Vec3,
    direction: Vec3,
) -> f32 {
    let rn = r.cross(direction);
    inv_mass + rn.dot(inv_inertia_world * rn)
}

/// Applies a positional impulse `p` (an unnormalized correction vector) to a
/// body at world lever arm `r`, updating its position and orientation in place.
///
/// The linear part translates the center of mass by `inv_mass * p`; the angular
/// part rotates the orientation by the half-angle quaternion built from
/// `I_inv_world (r x p)`, then renormalizes.
pub fn apply_position_impulse(
    position: &mut Vec3,
    orientation: &mut Quat,
    inv_mass: f32,
    inv_inertia_world: Mat3,
    r: Vec3,
    p: Vec3,
) {
    *position += inv_mass * p;
    let dphi = inv_inertia_world * r.cross(p);
    apply_angular_delta(orientation, dphi);
}

/// Applies a velocity impulse `p` to a body at world lever arm `r`, updating its
/// linear and angular velocities in place.
pub fn apply_velocity_impulse(
    linear_velocity: &mut Vec3,
    angular_velocity: &mut Vec3,
    inv_mass: f32,
    inv_inertia_world: Mat3,
    r: Vec3,
    p: Vec3,
) {
    *linear_velocity += inv_mass * p;
    *angular_velocity += inv_inertia_world * r.cross(p);
}

/// Rotates `orientation` by the small rotation vector `dphi` (axis scaled by
/// angle) and renormalizes.
///
/// Uses the first-order quaternion update `q' = normalize(q + 0.5 * [dphi,0] q)`,
/// which is the same integration the free-body integrator uses for angular
/// velocity.
pub fn apply_angular_delta(orientation: &mut Quat, dphi: Vec3) {
    let omega = Quat::from_xyzw(dphi.x, dphi.y, dphi.z, 0.0);
    let dq = omega.mul_quat(*orientation);
    let q = orientation.to_array();
    let d = dq.to_array();
    let updated = Quat::from_xyzw(
        q[0] + 0.5 * d[0],
        q[1] + 0.5 * d[1],
        q[2] + 0.5 * d[2],
        q[3] + 0.5 * d[3],
    );
    let len_sq = updated.length_squared();
    *orientation = if len_sq > 1e-20 {
        updated.normalize()
    } else {
        *orientation
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_inertia_world_is_identity_for_unit_sphere_at_rest() {
        let m = inv_inertia_world(Vec3::ONE, Quat::IDENTITY);
        let diff = m - Mat3::IDENTITY;
        assert!(diff.to_cols_array().iter().all(|c| c.abs() < 1e-6));
    }

    #[test]
    fn generalized_inverse_mass_reduces_to_inv_mass_at_zero_arm() {
        let m = inv_inertia_world(Vec3::ONE, Quat::IDENTITY);
        let w = generalized_inverse_mass(0.5, m, Vec3::ZERO, Vec3::Y);
        assert!((w - 0.5).abs() < 1e-6);
    }

    #[test]
    fn position_impulse_translates_point_mass() {
        let mut x = Vec3::ZERO;
        let mut q = Quat::IDENTITY;
        let m = inv_inertia_world(Vec3::ZERO, q); // infinite inertia: no rotation
        apply_position_impulse(&mut x, &mut q, 2.0, m, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        assert!((x - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-6);
        assert!((q.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn velocity_impulse_spins_body_with_offset() {
        let mut v = Vec3::ZERO;
        let mut w = Vec3::ZERO;
        let m = inv_inertia_world(Vec3::ONE, Quat::IDENTITY);
        // Impulse along +x at lever arm +z produces angular velocity about y.
        apply_velocity_impulse(&mut v, &mut w, 1.0, m, Vec3::Z, Vec3::X);
        assert!(w.y.abs() > 1e-6);
    }
}
