//! Support mappings for convex shapes, used by GJK/EPA style algorithms.
//!
//! A support mapping returns the farthest point of a convex shape along a
//! query direction. Together with the Minkowski difference it is the only
//! shape-specific primitive the GJK intersection test needs, which keeps the
//! core algorithm shape-agnostic.

use glam::Vec3;

use crate::bounding::{Aabb, BoundingSphere, Capsule, Obb};

/// A convex shape that can report its farthest point along a direction.
///
/// The returned point must lie on the shape's surface (or inside it) and be
/// the maximizer of `point.dot(dir)` over the shape. `dir` need not be unit
/// length and may be zero, in which case any point of the shape is valid.
pub trait SupportMap {
    /// Returns the farthest point of the shape in direction `dir`.
    fn support_point(&self, dir: Vec3) -> Vec3;
}

impl SupportMap for BoundingSphere {
    fn support_point(&self, dir: Vec3) -> Vec3 {
        self.center + dir.normalize_or_zero() * self.radius
    }
}

impl SupportMap for Aabb {
    fn support_point(&self, dir: Vec3) -> Vec3 {
        // Pick the max corner on each axis where the direction is non-negative,
        // otherwise the min corner.
        Vec3::select(dir.cmpge(Vec3::ZERO), self.max, self.min)
    }
}

impl SupportMap for Obb {
    fn support_point(&self, dir: Vec3) -> Vec3 {
        let axes = self.axes();
        let e = [self.half_extents.x, self.half_extents.y, self.half_extents.z];
        let mut point = self.center;
        for i in 0..3 {
            let sign = if dir.dot(axes[i]) >= 0.0 { e[i] } else { -e[i] };
            point += axes[i] * sign;
        }
        point
    }
}

impl SupportMap for Capsule {
    fn support_point(&self, dir: Vec3) -> Vec3 {
        // Farthest segment endpoint plus the radius offset along `dir`.
        let endpoint = if dir.dot(self.b - self.a) >= 0.0 {
            self.b
        } else {
            self.a
        };
        endpoint + dir.normalize_or_zero() * self.radius
    }
}

#[cfg(test)]
mod tests {
    use super::SupportMap;
    use crate::bounding::{Aabb, BoundingSphere, Capsule, Obb};
    use glam::{Quat, Vec3};

    #[test]
    fn sphere_support_is_on_surface() {
        let s = BoundingSphere::new(Vec3::new(1.0, 2.0, 3.0), 2.0);
        let p = s.support_point(Vec3::X);
        assert!(p.abs_diff_eq(Vec3::new(3.0, 2.0, 3.0), 1e-6));
    }

    #[test]
    fn aabb_support_picks_corner() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let p = b.support_point(Vec3::new(1.0, -1.0, 1.0));
        assert!(p.abs_diff_eq(Vec3::new(1.0, -1.0, 1.0), 1e-6));
    }

    #[test]
    fn obb_support_matches_axis_aligned_when_identity() {
        let b = Obb::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        let p = b.support_point(Vec3::new(-1.0, 1.0, -1.0));
        assert!(p.abs_diff_eq(Vec3::new(-1.0, 2.0, -3.0), 1e-6));
    }

    #[test]
    fn capsule_support_extends_endpoint() {
        let c = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5);
        let p = c.support_point(Vec3::X);
        // Picks the +x endpoint, then pushes out by the radius along +x.
        assert!(p.abs_diff_eq(Vec3::new(1.5, 0.0, 0.0), 1e-6));
    }
}
