//! Capsule bounding volume: all points within a radius of a line segment.

use glam::Vec3;

use crate::bounding::{Aabb, BoundingSphere};
use crate::narrow::{closest_point_on_segment, closest_points_segment_segment};

/// A capsule defined by a core segment `[a, b]` and a surrounding `radius`.
///
/// The capsule is the Minkowski sum of the segment and a sphere of `radius`,
/// i.e. every point whose distance to the segment is at most `radius`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Capsule {
    /// First endpoint of the core segment.
    pub a: Vec3,
    /// Second endpoint of the core segment.
    pub b: Vec3,
    /// Radius around the core segment.
    pub radius: f32,
}

impl Capsule {
    /// Creates a capsule from its segment endpoints and radius.
    pub fn new(a: Vec3, b: Vec3, radius: f32) -> Capsule {
        Capsule { a, b, radius }
    }

    /// Returns the squared distance from `point` to the capsule's core segment.
    pub fn distance_squared_to_point(&self, point: Vec3) -> f32 {
        let closest = closest_point_on_segment(point, self.a, self.b);
        (point - closest).length_squared()
    }

    /// Returns `true` when `point` lies on or inside the capsule surface.
    pub fn contains_point(&self, point: Vec3) -> bool {
        self.distance_squared_to_point(point) <= self.radius * self.radius
    }

    /// Returns `true` when the capsule overlaps `sphere`.
    pub fn intersects_sphere(&self, sphere: &BoundingSphere) -> bool {
        let sum = self.radius + sphere.radius;
        self.distance_squared_to_point(sphere.center) <= sum * sum
    }

    /// Returns `true` when the two capsules overlap.
    pub fn intersects_capsule(&self, other: &Capsule) -> bool {
        let sum = self.radius + other.radius;
        let closest = closest_points_segment_segment(self.a, self.b, other.a, other.b);
        closest.distance_squared <= sum * sum
    }

    /// Returns a tight axis-aligned bounding box enclosing the capsule.
    ///
    /// This is the box of the core segment expanded by `radius` on each axis,
    /// which lets capsules be inserted into the broad-phase BVH.
    pub fn aabb(&self) -> Aabb {
        let r = Vec3::splat(self.radius);
        Aabb::new(self.a.min(self.b) - r, self.a.max(self.b) + r)
    }
}

#[cfg(test)]
mod tests {
    use super::Capsule;
    use crate::bounding::{Aabb, BoundingSphere};
    use approx::assert_relative_eq;
    use glam::Vec3;

    fn x_capsule() -> Capsule {
        Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5)
    }

    #[test]
    fn point_distance_and_containment() {
        let c = x_capsule();
        // Alongside the segment, 0.4 away radially -> inside.
        assert!(c.contains_point(Vec3::new(0.0, 0.4, 0.0)));
        // Just outside the radius.
        assert!(!c.contains_point(Vec3::new(0.0, 0.6, 0.0)));
        // Beyond the hemispherical cap at b.
        assert_relative_eq!(
            c.distance_squared_to_point(Vec3::new(2.0, 0.0, 0.0)),
            1.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn sphere_overlap() {
        let c = x_capsule();
        // Sphere grazing the cap region.
        let hit = BoundingSphere::new(Vec3::new(0.0, 0.8, 0.0), 0.4);
        assert!(c.intersects_sphere(&hit));
        let miss = BoundingSphere::new(Vec3::new(0.0, 1.2, 0.0), 0.4);
        assert!(!c.intersects_sphere(&miss));
    }

    #[test]
    fn capsule_overlap() {
        let c = x_capsule();
        // Parallel capsule 0.8 above -> gap 0.8 < 0.5 + 0.4.
        let near = Capsule::new(Vec3::new(-1.0, 0.8, 0.0), Vec3::new(1.0, 0.8, 0.0), 0.4);
        assert!(c.intersects_capsule(&near));
        // Pushed to 1.2 above -> gap exceeds the radius sum.
        let far = Capsule::new(Vec3::new(-1.0, 1.2, 0.0), Vec3::new(1.0, 1.2, 0.0), 0.4);
        assert!(!c.intersects_capsule(&far));
    }

    #[test]
    fn bounding_box_expands_by_radius() {
        let c = x_capsule();
        let expected = Aabb::new(Vec3::new(-1.5, -0.5, -0.5), Vec3::new(1.5, 0.5, 0.5));
        let got = c.aabb();
        assert!(got.min.abs_diff_eq(expected.min, 1e-6));
        assert!(got.max.abs_diff_eq(expected.max, 1e-6));
    }
}
