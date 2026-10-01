//! Oriented plane with an inward-facing normal, used for frustum culling.

use glam::{Vec3, Vec4};

use super::aabb::Aabb;

/// A plane in the form `dot(normal, p) + d = 0`.
///
/// The [`Plane::normal`] is treated as pointing toward the *inside* half-space,
/// so a point `p` is on the inside when [`Plane::signed_distance`] is
/// non-negative. This is the convention used by [`crate::bounding::Frustum`].
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Plane {
    /// Plane normal, pointing toward the inside half-space.
    pub normal: Vec3,
    /// Signed offset: the plane is the locus of `dot(normal, p) + d = 0`.
    pub d: f32,
}

impl Plane {
    /// Creates a plane from an explicit normal and offset.
    ///
    /// The caller is responsible for the normal's length; use
    /// [`Plane::normalized`] when a true (metric) signed distance is required.
    #[inline]
    pub fn new(normal: Vec3, d: f32) -> Plane {
        Plane { normal, d }
    }

    /// Builds a plane from the homogeneous coefficients `(a, b, c, d)` stored in
    /// a [`Vec4`], where `normal = (a, b, c)` and the offset is `w`.
    #[inline]
    pub fn from_vec4(v: Vec4) -> Plane {
        Plane {
            normal: v.truncate(),
            d: v.w,
        }
    }

    /// Returns the signed distance from `p` to the plane.
    ///
    /// The value is positive on the inside half-space, negative on the outside,
    /// and metric only when the normal is unit length (see [`Plane::normalized`]).
    #[inline]
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        self.normal.dot(p) + self.d
    }

    /// Returns an equivalent plane whose normal is unit length, so that
    /// [`Plane::signed_distance`] yields a true Euclidean distance.
    #[inline]
    pub fn normalized(&self) -> Plane {
        let inv_len = 1.0 / self.normal.length();
        Plane {
            normal: self.normal * inv_len,
            d: self.d * inv_len,
        }
    }

    /// Returns `true` if `aabb` lies entirely on the outside half-space.
    ///
    /// Uses the positive-vertex test: the box corner farthest along `+normal`
    /// is the last point to leave the inside half-space, so if even that corner
    /// has a negative signed distance the whole box is outside. This never
    /// reports a box as outside when it still touches the inside half-space, so
    /// it is conservative in the direction frustum culling needs.
    #[inline]
    pub fn aabb_is_outside(&self, aabb: &Aabb) -> bool {
        let positive = Vec3::select(self.normal.cmpge(Vec3::ZERO), aabb.max, aabb.min);
        self.signed_distance(positive) < 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::Plane;
    use crate::bounding::Aabb;
    use approx::assert_relative_eq;
    use glam::{Vec3, Vec4};

    #[test]
    fn signed_distance_and_normalize() {
        // Plane x = 2 with inward normal +X, un-normalized (length 2).
        let p = Plane::new(Vec3::new(2.0, 0.0, 0.0), -4.0);
        // Raw distance is scaled by the normal length.
        assert_relative_eq!(p.signed_distance(Vec3::new(3.0, 0.0, 0.0)), 2.0, epsilon = 1e-6);
        let n = p.normalized();
        assert_relative_eq!(n.normal.length(), 1.0, epsilon = 1e-6);
        // After normalizing, the distance from x=3 to the plane x=2 is 1.
        assert_relative_eq!(n.signed_distance(Vec3::new(3.0, 0.0, 0.0)), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn from_vec4_round_trips() {
        let p = Plane::from_vec4(Vec4::new(0.0, 1.0, 0.0, -5.0));
        assert_eq!(p.normal, Vec3::Y);
        assert_eq!(p.d, -5.0);
    }

    #[test]
    fn aabb_outside_detection() {
        // Inward normal +X, plane at x = 0.
        let p = Plane::new(Vec3::X, 0.0);
        // Box fully behind the plane (x in [-2, -1]) is outside.
        assert!(p.aabb_is_outside(&Aabb::new(Vec3::new(-2.0, -1.0, -1.0), Vec3::new(-1.0, 1.0, 1.0))));
        // Box straddling the plane is not outside.
        assert!(!p.aabb_is_outside(&Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0))));
        // Box fully in front is not outside.
        assert!(!p.aabb_is_outside(&Aabb::new(Vec3::splat(1.0), Vec3::splat(2.0))));
    }
}
