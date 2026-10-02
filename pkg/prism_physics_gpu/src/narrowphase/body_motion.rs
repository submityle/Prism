//! Constant-twist rigid motion of a convex body over a substep: a linear
//! velocity and an angular velocity, sampled to a [`ConvexPose`] at any time
//! `t` in the substep.
//!
//! Continuous collision detection sweeps a body from its pose at the start of a
//! substep to its pose at the end. A [`BodyMotion`] records the two velocities
//! that drive that sweep and [`BodyMotion::pose_at`] reconstructs the
//! intermediate pose, so conservative advancement can probe the gap between two
//! moving hulls at successive times without materialising a dense set of
//! sampled poses.
//!
//! The angular term integrates as a constant-rate rotation about the body's
//! translating origin: over an elapsed time `t` the body turns by
//! `|angular| * t` radians about `angular`'s axis, composed onto the start
//! rotation, while the translation advances linearly by `linear * t`. This is
//! the screw motion conservative advancement assumes between two broad-phase
//! samples, matching the constant-velocity integrator the rigid solver uses
//! within one substep.
//!
//! Provenance: textbook constant-velocity rigid integration (screw motion) as
//! used by van den Bergen's ray-casting CCD and Mirtich's conservative
//! advancement. No Unreal Engine source or derived code.

use glam::{Quat, Vec3};

use super::convex_pose::ConvexPose;

/// Below this rotation angle (radians) over the sampled interval the rotation
/// is treated as the identity, avoiding a normalise of a near-zero axis.
const ANGLE_EPS: f32 = 1.0e-8;

/// A constant linear and angular velocity applied to a convex body over a
/// substep.
///
/// Velocities are world-space: `linear` in distance per substep-second and
/// `angular` as an axis whose length is the turn rate in radians per
/// substep-second. Sampling at time `t` yields the body's pose after `t`
/// seconds of this motion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyMotion {
    /// World-space linear velocity (distance per unit time).
    pub linear: Vec3,
    /// World-space angular velocity: axis scaled by the turn rate (radians per
    /// unit time).
    pub angular: Vec3,
}

impl BodyMotion {
    /// Builds a motion from an explicit linear and angular velocity.
    #[must_use]
    pub fn new(linear: Vec3, angular: Vec3) -> BodyMotion {
        BodyMotion { linear, angular }
    }

    /// A body at rest: no linear or angular velocity.
    #[must_use]
    pub fn still() -> BodyMotion {
        BodyMotion {
            linear: Vec3::ZERO,
            angular: Vec3::ZERO,
        }
    }

    /// The pose reached after `t` units of time, starting from `start`.
    ///
    /// The translation advances linearly; the rotation composes a constant-rate
    /// turn of `|angular| * t` radians about `angular`'s axis onto the start
    /// rotation. The result is renormalised so repeated sampling cannot drift
    /// off the unit quaternion.
    #[must_use]
    pub fn pose_at(&self, start: &ConvexPose, t: f32) -> ConvexPose {
        let translation = start.translation + self.linear * t;
        let angle = self.angular.length() * t;
        let rotation = if angle > ANGLE_EPS {
            let axis = self.angular.normalize();
            (Quat::from_axis_angle(axis, angle) * start.rotation).normalize()
        } else {
            start.rotation
        };
        ConvexPose {
            translation,
            rotation,
        }
    }
}

impl Default for BodyMotion {
    fn default() -> BodyMotion {
        BodyMotion::still()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn still_motion_leaves_the_pose_unchanged() {
        let start = ConvexPose::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_x(0.5));
        let m = BodyMotion::still();
        let p = m.pose_at(&start, 1.0);
        assert!((p.translation - start.translation).length() < 1.0e-7);
        assert!(p.rotation.angle_between(start.rotation) < 1.0e-6);
    }

    #[test]
    fn linear_velocity_advances_the_translation() {
        let start = ConvexPose::identity();
        let m = BodyMotion::new(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO);
        let p = m.pose_at(&start, 0.5);
        assert!((p.translation - Vec3::new(1.0, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn angular_velocity_turns_about_its_axis() {
        let start = ConvexPose::identity();
        // One radian per unit time about +z; sampled at t = FRAC_PI_2 units.
        let rate = 1.0;
        let t = core::f32::consts::FRAC_PI_2;
        let m = BodyMotion::new(Vec3::ZERO, Vec3::new(0.0, 0.0, rate));
        let p = m.pose_at(&start, t);
        // +x should rotate to +y after a quarter turn.
        let turned = p.rotation * Vec3::X;
        assert!((turned - Vec3::Y).length() < 1.0e-5, "turned {turned:?}");
    }

    #[test]
    fn sampling_is_consistent_with_the_screw_endpoints() {
        let start = ConvexPose::new(Vec3::new(-1.0, 0.5, 0.0), Quat::from_rotation_y(0.2));
        let m = BodyMotion::new(Vec3::new(1.0, -2.0, 3.0), Vec3::new(0.0, 0.3, 0.0));
        // Sampling at t = 0 recovers the start pose exactly.
        let p0 = m.pose_at(&start, 0.0);
        assert!((p0.translation - start.translation).length() < 1.0e-7);
        assert!(p0.rotation.angle_between(start.rotation) < 1.0e-6);
    }
}
