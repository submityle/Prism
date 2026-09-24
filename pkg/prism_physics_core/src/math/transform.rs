//! Rigid-body transforms.
//!
//! An [`Isometry`] is a rigid (distance-preserving) transform composed of a
//! translation and a rotation, i.e. a member of the special Euclidean group
//! SE(3). It is used to position bodies and colliders without introducing
//! scale or shear.

use glam::{Quat, Vec3};

/// A rigid transform: a rotation followed by a translation.
///
/// Applying the transform to a point `p` yields `rotation * p + translation`.
/// Composition follows the usual convention `(a.mul(b))` applies `b` first and
/// then `a`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Isometry {
    /// The translation component, applied after the rotation.
    pub translation: Vec3,
    /// The rotation component, applied before the translation.
    pub rotation: Quat,
}

impl Isometry {
    /// The identity transform (no rotation, no translation).
    pub const IDENTITY: Isometry = Isometry {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// Creates a transform from a translation and rotation.
    #[must_use]
    pub const fn new(translation: Vec3, rotation: Quat) -> Isometry {
        Isometry {
            translation,
            rotation,
        }
    }

    /// Creates a pure translation transform (identity rotation).
    #[must_use]
    pub const fn from_translation(translation: Vec3) -> Isometry {
        Isometry {
            translation,
            rotation: Quat::IDENTITY,
        }
    }

    /// Creates a pure rotation transform (zero translation).
    #[must_use]
    pub const fn from_rotation(rotation: Quat) -> Isometry {
        Isometry {
            translation: Vec3::ZERO,
            rotation,
        }
    }

    /// Composes two transforms: `self.mul(other)` applies `other` first, then
    /// `self`.
    #[must_use]
    pub fn mul(&self, other: &Isometry) -> Isometry {
        Isometry {
            translation: self.translation + self.rotation * other.translation,
            rotation: self.rotation * other.rotation,
        }
    }

    /// Returns the inverse transform such that `self.mul(&self.inverse())` is
    /// the identity (up to floating-point error).
    #[must_use]
    pub fn inverse(&self) -> Isometry {
        let inv_rot = self.rotation.inverse();
        Isometry {
            translation: inv_rot * (-self.translation),
            rotation: inv_rot,
        }
    }

    /// Transforms a point, applying both rotation and translation.
    #[must_use]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.rotation * point + self.translation
    }

    /// Transforms a direction vector, applying only the rotation.
    #[must_use]
    pub fn transform_vector(&self, vector: Vec3) -> Vec3 {
        self.rotation * vector
    }
}

impl Default for Isometry {
    fn default() -> Self {
        Isometry::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() <= 1e-5
    }

    #[test]
    fn identity_is_a_no_op() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert!(close(Isometry::IDENTITY.transform_point(p), p));
    }

    #[test]
    fn inverse_round_trips_points() {
        let iso = Isometry::new(
            Vec3::new(1.0, -2.0, 0.5),
            Quat::from_axis_angle(Vec3::Y, 0.9),
        );
        let p = Vec3::new(0.3, 4.0, -1.2);
        let mapped = iso.transform_point(p);
        let back = iso.inverse().transform_point(mapped);
        assert!(close(back, p));
    }

    #[test]
    fn mul_matches_sequential_application() {
        let a = Isometry::new(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::from_axis_angle(Vec3::Z, 0.5),
        );
        let b = Isometry::new(
            Vec3::new(2.0, 0.0, -1.0),
            Quat::from_axis_angle(Vec3::X, -0.7),
        );
        let p = Vec3::new(1.0, 1.0, 1.0);
        let composed = a.mul(&b).transform_point(p);
        let sequential = a.transform_point(b.transform_point(p));
        assert!(close(composed, sequential));
    }

    #[test]
    fn compose_with_inverse_yields_identity() {
        let iso = Isometry::new(
            Vec3::new(3.0, 1.5, -2.0),
            Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), 1.2),
        );
        let round = iso.mul(&iso.inverse());
        let p = Vec3::new(-1.0, 2.0, 3.0);
        assert!(close(round.transform_point(p), p));
    }
}
