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

    /// Resolves this capsule against `other`, returning `(point, normal, depth)`
    /// when their surfaces overlap or [`None`] otherwise.
    ///
    /// `normal` is a unit vector pointing from this capsule toward `other` (the
    /// direction `other` must travel to separate), `point` is the midpoint of
    /// the overlapping surface span, and `depth >= 0` is the penetration. When
    /// the core segments touch or cross the gap direction is ill-defined, so a
    /// stable axis is taken from the cross product of the two segment
    /// directions, falling back to the world up axis for parallel cores.
    pub fn contact(&self, other: &Capsule) -> Option<(Vec3, Vec3, f32)> {
        let sum = self.radius + other.radius;
        let near = closest_points_segment_segment(self.a, self.b, other.a, other.b);
        if near.distance_squared > sum * sum {
            return None;
        }
        let dist = near.distance_squared.sqrt();
        let normal = if dist > 1.0e-6 {
            (near.c2 - near.c1) * dist.recip()
        } else {
            let cross = (self.b - self.a).cross(other.b - other.a);
            let n = cross.normalize_or_zero();
            if n == Vec3::ZERO { Vec3::Y } else { n }
        };
        let depth = sum - dist;
        let surface_self = near.c1 + normal * self.radius;
        let surface_other = near.c2 - normal * other.radius;
        let point = (surface_self + surface_other) * 0.5;
        Some((point, normal, depth))
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

    #[test]
    fn contact_parallel_overlap() {
        // Two parallel capsules 0.8 apart on y with radii 0.5 + 0.4 = 0.9 sum,
        // overlapping by 0.1 with an upward normal toward `other`.
        let a = x_capsule();
        let b = Capsule::new(Vec3::new(-1.0, 0.8, 0.0), Vec3::new(1.0, 0.8, 0.0), 0.4);
        let (_, normal, depth) = a.contact(&b).expect("overlap");
        assert!(normal.y > 0.99, "normal toward +y: {normal:?}");
        assert_relative_eq!(depth, 0.1, epsilon = 1e-5);
    }

    #[test]
    fn contact_too_far_is_none() {
        let a = x_capsule();
        let b = Capsule::new(Vec3::new(-1.0, 1.2, 0.0), Vec3::new(1.0, 1.2, 0.0), 0.4);
        assert!(a.contact(&b).is_none());
    }

    #[test]
    fn contact_crossing_cores() {
        // Perpendicular capsules whose cores meet at the origin: the gap is
        // zero, so the normal comes from the cross of the two axes and the
        // depth is the full radius sum.
        let a = x_capsule();
        let b = Capsule::new(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0), 0.3);
        let (_, normal, depth) = a.contact(&b).expect("overlap");
        assert_relative_eq!(depth, 0.8, epsilon = 1e-5);
        assert_relative_eq!(normal.length(), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn contact_degenerate_points_are_spheres() {
        // Point capsules behave like spheres: centres 0.8 apart, radii 0.5 each.
        let a = Capsule::new(Vec3::ZERO, Vec3::ZERO, 0.5);
        let b = Capsule::new(Vec3::new(0.8, 0.0, 0.0), Vec3::new(0.8, 0.0, 0.0), 0.5);
        let (point, normal, depth) = a.contact(&b).expect("overlap");
        assert!(normal.x > 0.99, "normal toward +x: {normal:?}");
        assert_relative_eq!(depth, 0.2, epsilon = 1e-5);
        assert_relative_eq!(point.x, 0.4, epsilon = 1e-5);
    }
}
