//! [`Frustum`]: the six clip planes of a view-projection, extracted and queried.

use crate::planetest::{classify_aabb, classify_sphere, Containment};
use prism_math::geom::Plane;
use prism_math::{Mat4, Vec3};

/// A camera view frustum: the six clip planes extracted from a view-projection
/// matrix, with sphere/AABB classification for culling.
///
/// Planes are stored in `left, right, bottom, top, near, far` order, each with
/// a normalized normal pointing *inward*, so a point is inside exactly when it
/// has non-negative signed distance to every plane.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    /// Extract the six clip planes from a view-projection matrix using the
    /// Gribb–Hartmann method.
    ///
    /// This targets the `[0, 1]` clip-space depth convention produced by
    /// [`crate::Projection`] (and used by Vulkan / D3D / Metal / WebGPU): the
    /// near plane is the matrix's third row and the far plane is
    /// `row3 - row2`. For the `[-1, 1]` OpenGL convention use
    /// [`Frustum::from_view_proj_gl`].
    #[must_use]
    pub fn from_view_proj(view_proj: Mat4) -> Self {
        Self {
            planes: prism_math::geom::frustum::Frustum::from_view_proj(view_proj).planes,
        }
    }

    /// Like [`Frustum::from_view_proj`], but for the `[-1, 1]` clip-space depth
    /// convention (OpenGL), where the near plane is `row3 + row2`.
    #[must_use]
    pub fn from_view_proj_gl(view_proj: Mat4) -> Self {
        Self {
            planes: prism_math::geom::frustum::Frustum::from_view_proj_gl(view_proj).planes,
        }
    }

    /// Wrap six already-inward-facing planes in `left, right, bottom, top,
    /// near, far` order.
    #[inline]
    #[must_use]
    pub const fn from_planes(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// The six planes in `left, right, bottom, top, near, far` order.
    #[inline]
    #[must_use]
    pub const fn planes(self) -> [Plane; 6] {
        self.planes
    }

    /// True when `p` is inside or on the boundary of all six planes.
    #[inline]
    #[must_use]
    pub fn contains_point(self, p: Vec3) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.signed_distance(p) >= 0.0)
    }

    /// Classify a sphere against the frustum.
    #[inline]
    #[must_use]
    pub fn contains_sphere(self, center: Vec3, radius: f32) -> Containment {
        classify_sphere(&self.planes, center, radius)
    }

    /// Classify an axis-aligned box, given by its `min`/`max` corners, against
    /// the frustum.
    #[inline]
    #[must_use]
    pub fn contains_aabb(self, min: Vec3, max: Vec3) -> Containment {
        classify_aabb(&self.planes, min, max)
    }

    /// True when a sphere is at least partially visible.
    #[inline]
    #[must_use]
    pub fn intersects_sphere(self, center: Vec3, radius: f32) -> bool {
        self.contains_sphere(center, radius).is_visible()
    }

    /// True when an axis-aligned box is at least partially visible.
    #[inline]
    #[must_use]
    pub fn intersects_aabb(self, min: Vec3, max: Vec3) -> bool {
        self.contains_aabb(min, max).is_visible()
    }
}
