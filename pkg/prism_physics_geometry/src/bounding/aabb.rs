//! Axis-aligned bounding box (`Aabb`) and its geometric queries.

use glam::Vec3;

use super::ray::Ray;

/// An axis-aligned bounding box defined by its minimum and maximum corners.
///
/// A box is *valid* when `min <= max` on every axis. The [`Aabb::EMPTY`] value
/// is intentionally invalid (min at `+inf`, max at `-inf`) so it acts as the
/// identity for [`Aabb::merged`]: merging anything with an empty box yields the
/// other box.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb {
    /// Minimum corner (smallest coordinate on each axis).
    pub min: Vec3,
    /// Maximum corner (largest coordinate on each axis).
    pub max: Vec3,
}

impl Aabb {
    /// The empty (absorbing-identity) box: `min = +inf`, `max = -inf`.
    ///
    /// Merging any box with this value returns the other box unchanged.
    pub const EMPTY: Aabb = Aabb {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    /// Creates a box from explicit corners.
    ///
    /// The caller is responsible for passing `min <= max` if a valid box is
    /// desired; see [`Aabb::is_valid`].
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Aabb {
        Aabb { min, max }
    }

    /// Creates a box from a center point and non-negative half extents.
    #[inline]
    pub fn from_center_half_extents(center: Vec3, half: Vec3) -> Aabb {
        Aabb {
            min: center - half,
            max: center + half,
        }
    }

    /// Creates the tightest box containing all `points`.
    ///
    /// Returns [`None`] if `points` is empty.
    pub fn from_points(points: &[Vec3]) -> Option<Aabb> {
        let mut iter = points.iter();
        let first = *iter.next()?;
        let mut aabb = Aabb::new(first, first);
        for &p in iter {
            aabb.grow_to_include(p);
        }
        Some(aabb)
    }

    /// Returns the center point of the box.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Returns half of the box's size on each axis.
    #[inline]
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Returns the full size of the box on each axis (`max - min`).
    #[inline]
    pub fn extents(&self) -> Vec3 {
        self.max - self.min
    }

    /// Returns the total surface area of the box.
    ///
    /// This is the cost metric used by the surface-area heuristic in the tree.
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let d = self.extents();
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Returns the volume of the box.
    #[inline]
    pub fn volume(&self) -> f32 {
        let d = self.extents();
        d.x * d.y * d.z
    }

    /// Returns the smallest box containing both `self` and `other`.
    #[inline]
    pub fn merged(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Expands `self` in place to also contain `other`.
    #[inline]
    pub fn merge(&mut self, other: &Aabb) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Returns a copy grown outward by `margin` on every axis.
    #[inline]
    pub fn expanded_by(&self, margin: f32) -> Aabb {
        let m = Vec3::splat(margin);
        Aabb {
            min: self.min - m,
            max: self.max + m,
        }
    }

    /// Expands `self` in place to include the point `p`.
    #[inline]
    pub fn grow_to_include(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    /// Returns `true` if `p` lies inside or on the boundary of the box.
    #[inline]
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }

    /// Returns `true` if `other` is fully contained within `self`.
    #[inline]
    pub fn contains_aabb(&self, other: &Aabb) -> bool {
        self.min.cmple(other.min).all() && self.max.cmpge(other.max).all()
    }

    /// Returns `true` if `self` and `other` overlap (touching counts).
    #[inline]
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.cmple(other.max).all() && self.max.cmpge(other.min).all()
    }

    /// Returns `true` if the box is valid, i.e. `min <= max` on every axis.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.min.cmple(self.max).all()
    }

    /// Returns the squared Euclidean distance from `p` to the nearest point on
    /// the box, or `0.0` when `p` lies inside the box.
    ///
    /// This clamps `p` into `[min, max]` per axis and measures to the clamp,
    /// which is the standard point-to-AABB squared distance.
    #[inline]
    pub fn distance_squared_to_point(&self, p: Vec3) -> f32 {
        let clamped = p.clamp(self.min, self.max);
        clamped.distance_squared(p)
    }

    /// Intersects `ray` against the box using the slab method.
    ///
    /// Returns the entry parameter `t` (clamped to be non-negative) when the
    /// ray enters the box within `[0, ray.tmax]`, or [`None`] on a miss. When
    /// the origin is already inside the box the returned `t` is `0.0`.
    pub fn ray_hit(&self, ray: &Ray) -> Option<f32> {
        let t1 = (self.min - ray.origin) * ray.inv_dir;
        let t2 = (self.max - ray.origin) * ray.inv_dir;
        let t_near = t1.min(t2).max_element().max(0.0);
        let t_far = t1.max(t2).min_element().min(ray.tmax);
        if t_far >= t_near {
            Some(t_near)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Aabb;
    use crate::bounding::Ray;
    use approx::assert_relative_eq;
    use glam::Vec3;

    fn unit() -> Aabb {
        Aabb::new(Vec3::ZERO, Vec3::ONE)
    }

    #[test]
    fn empty_is_merge_identity() {
        let a = unit();
        assert_eq!(Aabb::EMPTY.merged(&a), a);
        assert!(!Aabb::EMPTY.is_valid());
        assert!(a.is_valid());
    }

    #[test]
    fn from_points_and_center() {
        let pts = [
            Vec3::new(-1.0, 0.0, 2.0),
            Vec3::new(3.0, -2.0, 1.0),
            Vec3::new(0.0, 4.0, -1.0),
        ];
        let a = Aabb::from_points(&pts).unwrap();
        assert_eq!(a.min, Vec3::new(-1.0, -2.0, -1.0));
        assert_eq!(a.max, Vec3::new(3.0, 4.0, 2.0));
        assert!(Aabb::from_points(&[]).is_none());
        let b = Aabb::from_center_half_extents(Vec3::splat(1.0), Vec3::splat(0.5));
        assert_eq!(b.center(), Vec3::splat(1.0));
        assert_eq!(b.half_extents(), Vec3::splat(0.5));
    }

    #[test]
    fn area_and_volume() {
        let a = Aabb::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        // 2*(1*2 + 2*3 + 3*1) = 2*(2+6+3) = 22
        assert_relative_eq!(a.surface_area(), 22.0, epsilon = 1e-6);
        assert_relative_eq!(a.volume(), 6.0, epsilon = 1e-6);
        assert_eq!(a.extents(), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn merge_grow_and_expand() {
        let mut a = unit();
        a.merge(&Aabb::new(Vec3::splat(-1.0), Vec3::splat(0.5)));
        assert_eq!(a.min, Vec3::splat(-1.0));
        assert_eq!(a.max, Vec3::ONE);
        a.grow_to_include(Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(a.max.x, 5.0);
        let e = unit().expanded_by(0.5);
        assert_eq!(e.min, Vec3::splat(-0.5));
        assert_eq!(e.max, Vec3::splat(1.5));
    }

    #[test]
    fn contains_and_intersects() {
        let a = unit();
        assert!(a.contains_point(Vec3::splat(0.5)));
        assert!(a.contains_point(Vec3::ZERO));
        assert!(!a.contains_point(Vec3::splat(1.5)));
        assert!(a.contains_aabb(&Aabb::from_center_half_extents(
            Vec3::splat(0.5),
            Vec3::splat(0.25)
        )));
        assert!(!a.contains_aabb(&Aabb::new(Vec3::splat(-1.0), Vec3::splat(0.5))));
        assert!(a.intersects(&Aabb::new(Vec3::splat(0.5), Vec3::splat(2.0))));
        // touching faces count as intersecting
        assert!(a.intersects(&Aabb::new(Vec3::new(1.0, 0.0, 0.0), Vec3::splat(2.0))));
        assert!(!a.intersects(&Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0))));
    }

    #[test]
    fn ray_hit_front_face() {
        let a = unit();
        let ray = Ray::new(Vec3::new(0.5, 0.5, -5.0), Vec3::Z);
        let t = a.ray_hit(&ray).unwrap();
        assert_relative_eq!(t, 5.0, epsilon = 1e-5);
        assert_relative_eq!(ray.at(t).z, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn ray_hit_from_inside_is_zero() {
        let a = unit();
        let ray = Ray::new(Vec3::splat(0.5), Vec3::X);
        assert_relative_eq!(a.ray_hit(&ray).unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn ray_miss_and_tmax() {
        let a = unit();
        // parallel and offset -> miss
        let miss = Ray::new(Vec3::new(5.0, 5.0, -5.0), Vec3::Z);
        assert!(a.ray_hit(&miss).is_none());
        // pointing away -> miss
        let away = Ray::new(Vec3::new(0.5, 0.5, -5.0), Vec3::NEG_Z);
        assert!(a.ray_hit(&away).is_none());
        // hit is beyond tmax -> miss
        let short = Ray::with_tmax(Vec3::new(0.5, 0.5, -5.0), Vec3::Z, 1.0);
        assert!(a.ray_hit(&short).is_none());
    }

    #[test]
    fn distance_squared_inside_is_zero() {
        let a = unit();
        assert_relative_eq!(
            a.distance_squared_to_point(Vec3::splat(0.5)),
            0.0,
            epsilon = 1e-6
        );
        // On a face still counts as inside -> zero.
        assert_relative_eq!(
            a.distance_squared_to_point(Vec3::new(0.0, 0.5, 0.5)),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn distance_squared_axis_aligned() {
        let a = unit();
        // Two units left of the near x face at x = 0.
        assert_relative_eq!(
            a.distance_squared_to_point(Vec3::new(-2.0, 0.5, 0.5)),
            4.0,
            epsilon = 1e-5
        );
    }

    #[test]
    fn distance_squared_corner_diagonal() {
        let a = unit();
        // Offset by (-1, -1, -1) from the min corner -> squared distance 3.
        assert_relative_eq!(
            a.distance_squared_to_point(Vec3::splat(-1.0)),
            3.0,
            epsilon = 1e-5
        );
    }
}
