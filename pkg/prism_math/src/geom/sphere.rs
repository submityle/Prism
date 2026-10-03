//! Bounding sphere [`BoundingSphere`].

use crate::float::f32 as mf;
use crate::geom::aabb::Aabb3;
use crate::vec::Vec3;

/// A sphere defined by a `center` and a (non-negative) `radius`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BoundingSphere {
    /// The sphere center.
    pub center: Vec3,
    /// The sphere radius (expected to be non-negative).
    pub radius: f32,
}

impl BoundingSphere {
    /// Create a sphere from a `center` and `radius`.
    #[inline]
    pub const fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Build the sphere whose center is the point average and whose radius
    /// reaches the farthest point. Returns [`None`] when `points` is empty.
    ///
    /// This is a fast, non-minimal fit (not Welzl's exact minimal sphere) but
    /// is guaranteed to bound every input point.
    pub fn from_points(points: &[Vec3]) -> Option<Self> {
        let (first, rest) = points.split_first()?;
        let mut center = *first;
        for &p in rest {
            center += p;
        }
        center *= 1.0 / points.len() as f32;
        let mut r2 = 0.0f32;
        for &p in points {
            r2 = r2.max((p - center).length_squared());
        }
        Some(Self { center, radius: mf::sqrt(r2) })
    }

    /// The tight bounding sphere of an [`Aabb3`].
    #[inline]
    pub fn from_aabb(aabb: Aabb3) -> Self {
        let center = aabb.center();
        Self { center, radius: aabb.half_extents().length() }
    }

    /// True if `p` lies inside or on the sphere.
    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        (p - self.center).length_squared() <= self.radius * self.radius
    }

    /// True if `other` is fully contained within `self`.
    #[inline]
    pub fn contains_sphere(self, other: Self) -> bool {
        if other.radius > self.radius {
            return false;
        }
        let d = (other.center - self.center).length();
        d + other.radius <= self.radius
    }

    /// The smallest sphere containing both `self` and `other`.
    pub fn merge(self, other: Self) -> Self {
        let offset = other.center - self.center;
        let dist = offset.length();
        // One sphere already swallows the other.
        if dist + other.radius <= self.radius {
            return self;
        }
        if dist + self.radius <= other.radius {
            return other;
        }
        let new_radius = (dist + self.radius + other.radius) * 0.5;
        let dir = if dist > 1.0e-20 { offset * (1.0 / dist) } else { Vec3::ZERO };
        let center = self.center + dir * (new_radius - self.radius);
        Self { center, radius: new_radius }
    }

    /// The axis-aligned box that tightly bounds this sphere.
    #[inline]
    pub fn to_aabb(self) -> Aabb3 {
        Aabb3::from_center_half_extents(self.center, Vec3::splat(self.radius))
    }

    /// True if the center is finite and the radius is finite and non-negative.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.center.is_finite() && self.radius.is_finite() && self.radius >= 0.0
    }
}
