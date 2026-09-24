//! Ray type with precomputed inverse direction for fast slab tests.

use glam::Vec3;

/// A ray with a normalized direction and a precomputed component-wise inverse
/// direction, plus a maximum parametric distance.
///
/// The inverse direction is cached so that repeated slab intersection tests
/// (see [`Aabb::ray_hit`]) avoid per-test divisions. For axis-aligned
/// directions a zero component yields an infinite inverse, which the slab test
/// handles correctly.
///
/// [`Aabb::ray_hit`]: crate::bounding::Aabb::ray_hit
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    /// Origin point of the ray.
    pub origin: Vec3,
    /// Normalized direction of the ray.
    pub dir: Vec3,
    /// Component-wise reciprocal of [`Ray::dir`] (`1.0 / dir`).
    pub inv_dir: Vec3,
    /// Maximum parametric distance along the ray considered a hit.
    pub tmax: f32,
}

impl Ray {
    /// Creates a ray from an origin and a direction.
    ///
    /// The direction is normalized, the inverse direction is computed
    /// component-wise, and `tmax` is set to [`f32::INFINITY`].
    #[inline]
    pub fn new(origin: Vec3, dir: Vec3) -> Ray {
        Ray::with_tmax(origin, dir, f32::INFINITY)
    }

    /// Creates a ray from an origin, a direction, and a maximum distance.
    ///
    /// The direction is normalized and the inverse direction is computed
    /// component-wise.
    #[inline]
    pub fn with_tmax(origin: Vec3, dir: Vec3, tmax: f32) -> Ray {
        let dir = dir.normalize();
        Ray {
            origin,
            dir,
            inv_dir: dir.recip(),
            tmax,
        }
    }

    /// Returns the point at parametric distance `t` along the ray.
    #[inline]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

#[cfg(test)]
mod tests {
    use super::Ray;
    use approx::assert_relative_eq;
    use glam::Vec3;

    #[test]
    fn new_normalizes_direction() {
        let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 5.0));
        assert_relative_eq!(ray.dir.length(), 1.0, epsilon = 1e-6);
        assert!(ray.dir.abs_diff_eq(Vec3::Z, 1e-6));
        assert_eq!(ray.tmax, f32::INFINITY);
    }

    #[test]
    fn inv_dir_is_reciprocal() {
        let ray = Ray::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0));
        assert_relative_eq!(ray.inv_dir.x, 1.0, epsilon = 1e-6);
        assert!(ray.inv_dir.y.is_infinite());
    }

    #[test]
    fn at_walks_along_direction() {
        let ray = Ray::with_tmax(Vec3::new(1.0, 2.0, 3.0), Vec3::X, 10.0);
        assert!(ray.at(4.0).abs_diff_eq(Vec3::new(5.0, 2.0, 3.0), 1e-6));
        assert_eq!(ray.tmax, 10.0);
    }
}
