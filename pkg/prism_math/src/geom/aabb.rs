//! Axis-aligned bounding box [`Aabb3`].

use crate::float::f32 as mf;
use crate::vec::Vec3;

/// An axis-aligned bounding box stored as `min`/`max` corners.
///
/// An AABB is valid when `min <= max` on every axis. Constructors that take
/// arbitrary corners sort them, so callers never need to pre-order inputs.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Aabb3 {
    /// The minimum corner (smallest coordinate on each axis).
    pub min: Vec3,
    /// The maximum corner (largest coordinate on each axis).
    pub max: Vec3,
}

impl Aabb3 {
    /// Create a box from two corners, sorting them component-wise so the
    /// result is always valid regardless of input order.
    #[inline]
    pub fn new(a: Vec3, b: Vec3) -> Self {
        Self { min: a.min(b), max: a.max(b) }
    }

    /// Create a box from a center and (non-negative) half-extents.
    #[inline]
    pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self { min: center - half_extents, max: center + half_extents }
    }

    /// Build the tightest box that contains all `points`.
    ///
    /// Returns [`None`] when `points` is empty.
    pub fn from_points(points: &[Vec3]) -> Option<Self> {
        let (first, rest) = points.split_first()?;
        let mut bb = Self { min: *first, max: *first };
        for &p in rest {
            bb = bb.expand_to_include(p);
        }
        Some(bb)
    }

    /// The geometric center.
    #[inline]
    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// The full size along each axis (`max - min`).
    #[inline]
    pub fn extents(self) -> Vec3 {
        self.max - self.min
    }

    /// Half of [`Aabb3::extents`].
    #[inline]
    pub fn half_extents(self) -> Vec3 {
        self.extents() * 0.5
    }

    /// Total surface area of the box.
    #[inline]
    pub fn surface_area(self) -> f32 {
        let e = self.extents();
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }

    /// Volume of the box.
    #[inline]
    pub fn volume(self) -> f32 {
        let e = self.extents();
        e.x * e.y * e.z
    }

    /// True if `p` lies inside the box (inclusive of the boundary).
    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// True if `other` is fully contained within `self`.
    #[inline]
    pub fn contains_aabb(self, other: Self) -> bool {
        self.contains_point(other.min) && self.contains_point(other.max)
    }

    /// The point of the box closest to `p` (equal to `p` when inside).
    #[inline]
    pub fn closest_point(self, p: Vec3) -> Vec3 {
        p.clamp(self.min, self.max)
    }

    /// Squared Euclidean distance from `p` to the box (`0` when inside).
    #[inline]
    pub fn distance_squared(self, p: Vec3) -> f32 {
        (p - self.closest_point(p)).length_squared()
    }

    /// The smallest box containing both `self` and `other`.
    #[inline]
    pub fn merge(self, other: Self) -> Self {
        Self { min: self.min.min(other.min), max: self.max.max(other.max) }
    }

    /// The smallest box containing both `self` and the point `p`.
    #[inline]
    pub fn expand_to_include(self, p: Vec3) -> Self {
        Self { min: self.min.min(p), max: self.max.max(p) }
    }

    /// Grow the box outward by `amount` on every axis (negative shrinks it).
    #[inline]
    pub fn expand(self, amount: f32) -> Self {
        let a = Vec3::splat(amount);
        Self { min: self.min - a, max: self.max + a }
    }

    /// The overlap box of `self` and `other`, or [`None`] when disjoint.
    #[inline]
    pub fn intersection(self, other: Self) -> Option<Self> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        if min.x <= max.x && min.y <= max.y && min.z <= max.z {
            Some(Self { min, max })
        } else {
            None
        }
    }

    /// The eight corner vertices, ordered by the bit pattern of
    /// `(x, y, z)` picking `min` (bit clear) or `max` (bit set).
    #[inline]
    pub fn corners(self) -> [Vec3; 8] {
        [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ]
    }

    /// True if every corner coordinate is finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.min.is_finite() && self.max.is_finite()
    }

    /// The longest edge length, useful as a conservative size metric.
    #[inline]
    pub fn largest_extent(self) -> f32 {
        let e = self.extents();
        mf::abs(e.x).max(mf::abs(e.y)).max(mf::abs(e.z))
    }
}
