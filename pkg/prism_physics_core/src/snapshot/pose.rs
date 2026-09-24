//! The minimal render pose and its interpolation.
//!
//! A [`BodyPose`] is the subset of body state the renderer needs to draw a
//! body: its world-space position and orientation. Keeping it separate from the
//! full solver state keeps snapshots small and makes interpolation a pure,
//! allocation-free operation.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Linear
//! position interpolation and quaternion spherical interpolation are standard,
//! publicly documented operations provided directly by `glam`.

use glam::{Quat, Vec3};

/// The renderable pose of a single body: world position and orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BodyPose {
    /// World-space position of the body's origin.
    pub position: Vec3,
    /// World-space orientation of the body.
    pub orientation: Quat,
}

impl BodyPose {
    /// The identity pose: origin position and identity orientation.
    pub const IDENTITY: BodyPose = BodyPose {
        position: Vec3::ZERO,
        orientation: Quat::IDENTITY,
    };

    /// Creates a pose from a `position` and `orientation`.
    #[must_use]
    pub const fn new(position: Vec3, orientation: Quat) -> BodyPose {
        BodyPose {
            position,
            orientation,
        }
    }
}

impl Default for BodyPose {
    fn default() -> Self {
        BodyPose::IDENTITY
    }
}

/// Interpolates from `a` toward `b` by `alpha`, returning the in-between pose.
///
/// `alpha` is clamped to `[0, 1]` so the result never overshoots either
/// endpoint, which is what guarantees the render pose stays bracketed by the
/// two physics steps and cannot jitter past them. The position is linearly
/// interpolated and the orientation is spherically interpolated (`slerp`) for
/// constant angular velocity between the endpoints.
#[must_use]
pub fn lerp_pose(a: BodyPose, b: BodyPose, alpha: f32) -> BodyPose {
    let t = alpha.clamp(0.0, 1.0);
    BodyPose {
        position: a.position.lerp(b.position, t),
        orientation: a.orientation.slerp(b.orientation, t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lerp_endpoints_are_exact() {
        let a = BodyPose::new(Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY);
        let b = BodyPose::new(Vec3::new(10.0, -4.0, 2.0), Quat::IDENTITY);
        assert_eq!(lerp_pose(a, b, 0.0).position, a.position);
        assert_eq!(lerp_pose(a, b, 1.0).position, b.position);
    }

    #[test]
    fn lerp_midpoint_is_average_for_position() {
        let a = BodyPose::new(Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY);
        let b = BodyPose::new(Vec3::new(4.0, 8.0, -2.0), Quat::IDENTITY);
        let mid = lerp_pose(a, b, 0.5).position;
        assert_eq!(mid, Vec3::new(2.0, 4.0, -1.0));
    }

    #[test]
    fn alpha_is_clamped_to_unit_range() {
        let a = BodyPose::new(Vec3::ZERO, Quat::IDENTITY);
        let b = BodyPose::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY);
        // Over- and under-shoot are clamped back to the endpoints.
        assert_eq!(lerp_pose(a, b, 2.0).position, b.position);
        assert_eq!(lerp_pose(a, b, -1.0).position, a.position);
    }
}
