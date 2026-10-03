//! Oriented plane [`Plane`] in Hessian normal form.

use crate::float::f32 as mf;
use crate::vec::Vec3;

/// A plane described in Hessian normal form: the set of points `p` with
/// `normal.dot(p) + d == 0`.
///
/// The `normal` points toward the plane's positive half-space. For a plane
/// built from a point and normal, [`Plane::signed_distance`] returns the
/// Euclidean signed distance when `normal` is unit length.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Plane {
    /// The plane normal (unit length for Euclidean distances).
    pub normal: Vec3,
    /// The signed offset such that `normal.dot(p) + d == 0` on the plane.
    pub d: f32,
}

impl Plane {
    /// Create a plane from raw coefficients `normal` and `d` without
    /// normalizing. Prefer [`Plane::from_point_normal`] for geometric input.
    #[inline]
    pub const fn new(normal: Vec3, d: f32) -> Self {
        Self { normal, d }
    }

    /// Create a plane passing through `point` with the given `normal`.
    ///
    /// The `normal` is normalized so distances are Euclidean.
    #[inline]
    pub fn from_point_normal(point: Vec3, normal: Vec3) -> Self {
        let n = normal.normalize();
        Self { normal: n, d: -n.dot(point) }
    }

    /// Create a plane through three points, wound counter-clockwise so the
    /// `normal` follows the right-hand rule for `(b - a) x (c - a)`.
    ///
    /// Returns [`None`] when the points are collinear (degenerate normal).
    pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Option<Self> {
        let n = (b - a).cross(c - a);
        let len = n.length();
        if len <= 1.0e-20 {
            return None;
        }
        let normal = n * (1.0 / len);
        Some(Self { normal, d: -normal.dot(a) })
    }

    /// Return a copy with a unit-length `normal` (and `d` scaled to match).
    #[inline]
    pub fn normalized(self) -> Self {
        let len = self.normal.length();
        let inv = 1.0 / len;
        Self { normal: self.normal * inv, d: self.d * inv }
    }

    /// Signed distance from `p` to the plane. Positive values lie in the
    /// direction the `normal` points; the magnitude is Euclidean when the
    /// `normal` is unit length.
    #[inline]
    pub fn signed_distance(self, p: Vec3) -> f32 {
        self.normal.dot(p) + self.d
    }

    /// Project `p` orthogonally onto the plane.
    #[inline]
    pub fn project_point(self, p: Vec3) -> Vec3 {
        p - self.normal * self.signed_distance(p)
    }

    /// Flip the plane to face the opposite half-space.
    #[inline]
    pub fn flipped(self) -> Self {
        Self { normal: -self.normal, d: -self.d }
    }

    /// True if the normal and offset are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.normal.is_finite() && self.d.is_finite()
    }

    /// The absolute (unsigned) distance from `p` to the plane, assuming a
    /// unit-length `normal`.
    #[inline]
    pub fn distance(self, p: Vec3) -> f32 {
        mf::abs(self.signed_distance(p))
    }
}
