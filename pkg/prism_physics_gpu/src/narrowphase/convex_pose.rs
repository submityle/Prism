//! Rigid pose of a convex hull: a world translation and a rotation.
//!
//! A [`ConvexPose`] places a [`ConvexHull`](super::convex_hull::ConvexHull)'s
//! local geometry into the world as `world = translation + rotation * local`.
//! It is the per-body transform the convex-versus-convex narrow phase pairs with
//! shared cooked geometry, so one hull can be instanced under many poses without
//! duplicating its vertex, face, and edge tables.
//!
//! Provenance: textbook rigid transform; no Unreal Engine source or derived code.

use glam::{Quat, Vec3};

/// A rigid placement of a convex hull: world translation plus rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvexPose {
    /// World-space translation applied after the rotation.
    pub translation: Vec3,
    /// World-space rotation of the hull's local frame.
    pub rotation: Quat,
}

impl ConvexPose {
    /// Builds a pose from an explicit translation and rotation.
    #[must_use]
    pub fn new(translation: Vec3, rotation: Quat) -> ConvexPose {
        ConvexPose {
            translation,
            rotation,
        }
    }

    /// The identity pose: no translation, no rotation.
    #[must_use]
    pub fn identity() -> ConvexPose {
        ConvexPose {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }

    /// Maps a local-space point into the world: `translation + rotation * local`.
    #[must_use]
    pub fn transform_point(&self, local: Vec3) -> Vec3 {
        self.translation + self.rotation * local
    }

    /// Rotates a world-space direction into the hull's local frame, i.e.
    /// `inverse(rotation) * dir`. Translation does not affect directions.
    #[must_use]
    pub fn inverse_rotate(&self, dir: Vec3) -> Vec3 {
        self.rotation.inverse() * dir
    }
}

impl Default for ConvexPose {
    fn default() -> ConvexPose {
        ConvexPose::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_a_no_op() {
        let p = ConvexPose::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!((p.transform_point(v) - v).length() < 1.0e-7);
        assert!((p.inverse_rotate(v) - v).length() < 1.0e-7);
    }

    #[test]
    fn transform_then_inverse_rotate_round_trip_a_direction() {
        let p = ConvexPose::new(
            Vec3::new(5.0, -2.0, 1.0),
            Quat::from_rotation_y(core::f32::consts::FRAC_PI_3),
        );
        let dir = Vec3::new(0.0, 1.0, 1.0).normalize();
        // Rotating a direction out then back in recovers it; translation is
        // irrelevant to a direction.
        let out = p.rotation * dir;
        assert!((p.inverse_rotate(out) - dir).length() < 1.0e-6);
    }
}
