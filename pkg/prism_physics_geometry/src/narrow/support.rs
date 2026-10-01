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

/// A support map that rigidly translates another shape by a fixed `offset`.
///
/// This lets GJK-based queries evaluate a convex shape at a shifted position
/// (for example successive samples along a motion sweep) without rebuilding the
/// underlying geometry. Translation commutes with the support map, so the
/// farthest point of the shifted shape is simply the base support plus the
/// offset.
#[derive(Clone, Copy, Debug)]
pub struct Translated<'a, S> {
    /// The underlying convex shape.
    pub shape: &'a S,
    /// World-space translation applied to the shape.
    pub offset: Vec3,
}

impl<'a, S> Translated<'a, S> {
    /// Wraps `shape`, offsetting it by `offset`.
    pub fn new(shape: &'a S, offset: Vec3) -> Self {
        Self { shape, offset }
    }
}

impl<S: SupportMap> SupportMap for Translated<'_, S> {
    fn support_point(&self, dir: Vec3) -> Vec3 {
        self.shape.support_point(dir) + self.offset
    }
}

/// A support map that inflates another shape by a uniform `margin`, forming its
/// Minkowski sum with a sphere of that radius (a rounded shape).
///
/// Inflating both operands of a GJK distance query by their collision margins
/// yields speculative-contact separation: when the rounded shells touch, the
/// cores are within `margin_a + margin_b` of each other.
#[derive(Clone, Copy, Debug)]
pub struct Inflated<'a, S> {
    /// The underlying convex shape.
    pub shape: &'a S,
    /// Non-negative inflation radius.
    pub margin: f32,
}

impl<'a, S> Inflated<'a, S> {
    /// Wraps `shape`, inflating it by `margin`.
    pub fn new(shape: &'a S, margin: f32) -> Self {
        Self { shape, margin }
    }
}

impl<S: SupportMap> SupportMap for Inflated<'_, S> {
    fn support_point(&self, dir: Vec3) -> Vec3 {
        self.shape.support_point(dir) + dir.normalize_or_zero() * self.margin
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
