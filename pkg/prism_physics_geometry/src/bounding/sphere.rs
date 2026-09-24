//! Bounding sphere type and conversions to/from [`Aabb`].

use glam::Vec3;

use super::aabb::Aabb;

/// A bounding sphere defined by a center and a radius.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundingSphere {
    /// Center point of the sphere.
    pub center: Vec3,
    /// Radius of the sphere (assumed non-negative).
    pub radius: f32,
}

impl BoundingSphere {
    /// Creates a sphere from a center and a radius.
    #[inline]
    pub fn new(center: Vec3, radius: f32) -> BoundingSphere {
        BoundingSphere { center, radius }
    }

    /// Returns the smallest sphere enclosing `aabb`.
    ///
    /// The center is the box center and the radius is the distance to a corner.
    #[inline]
    pub fn from_aabb(aabb: &Aabb) -> BoundingSphere {
        BoundingSphere {
            center: aabb.center(),
            radius: aabb.half_extents().length(),
        }
    }

    /// Returns `true` if `p` lies inside or on the sphere.
    #[inline]
    pub fn contains_point(&self, p: Vec3) -> bool {
        self.center.distance_squared(p) <= self.radius * self.radius
    }

    /// Returns `true` if `self` and `other` overlap (touching counts).
    #[inline]
    pub fn intersects_sphere(&self, other: &BoundingSphere) -> bool {
        let r = self.radius + other.radius;
        self.center.distance_squared(other.center) <= r * r
    }

    /// Returns `true` if the sphere overlaps `aabb`.
    ///
    /// Uses the squared distance from the sphere center to the closest point
    /// on the box.
    #[inline]
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        let closest = self.center.clamp(aabb.min, aabb.max);
        self.center.distance_squared(closest) <= self.radius * self.radius
    }

    /// Returns the axis-aligned box that tightly encloses the sphere.
    #[inline]
    pub fn to_aabb(&self) -> Aabb {
        Aabb::from_center_half_extents(self.center, Vec3::splat(self.radius))
    }
}

#[cfg(test)]
mod tests {
    use super::BoundingSphere;
    use crate::bounding::Aabb;
    use approx::assert_relative_eq;
    use glam::Vec3;

    #[test]
    fn from_aabb_covers_corners() {
        let a = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let s = BoundingSphere::from_aabb(&a);
        assert_eq!(s.center, Vec3::ZERO);
        assert_relative_eq!(s.radius, 3.0_f32.sqrt(), epsilon = 1e-6);
        assert!(s.contains_point(a.max));
        assert!(s.contains_point(a.min));
    }

    #[test]
    fn contains_and_sphere_intersection() {
        let s = BoundingSphere::new(Vec3::ZERO, 1.0);
        assert!(s.contains_point(Vec3::new(0.5, 0.5, 0.0)));
        assert!(!s.contains_point(Vec3::new(1.5, 0.0, 0.0)));
        assert!(s.intersects_sphere(&BoundingSphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0)));
        assert!(!s.intersects_sphere(&BoundingSphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0)));
    }

    #[test]
    fn sphere_aabb_intersection() {
        let s = BoundingSphere::new(Vec3::ZERO, 1.0);
        assert!(s.intersects_aabb(&Aabb::new(
            Vec3::new(0.5, -1.0, -1.0),
            Vec3::new(2.0, 1.0, 1.0)
        )));
        assert!(!s.intersects_aabb(&Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0))));
        let b = s.to_aabb();
        assert_eq!(b.min, Vec3::splat(-1.0));
        assert_eq!(b.max, Vec3::splat(1.0));
    }
}
