//! [`Camera`]: a [`View`] paired with a [`Projection`].

use crate::{Frustum, Projection, View};
use prism_math::Mat4;

/// A camera: a view (where it is) and a projection (how it sees).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Camera {
    /// Where the camera is and how it is oriented.
    pub view: View,
    /// How the camera maps view space to clip space.
    pub projection: Projection,
}

impl Camera {
    /// Compose a camera from a view and a projection.
    #[inline]
    #[must_use]
    pub const fn new(view: View, projection: Projection) -> Self {
        Self { view, projection }
    }

    /// The world-to-camera (view) matrix.
    #[inline]
    #[must_use]
    pub fn view_matrix(self) -> Mat4 {
        self.view.matrix()
    }

    /// The camera-to-clip (projection) matrix.
    #[inline]
    #[must_use]
    pub fn projection_matrix(self) -> Mat4 {
        self.projection.matrix()
    }

    /// The world-to-clip matrix, `projection * view`.
    #[inline]
    #[must_use]
    pub fn view_projection(self) -> Mat4 {
        self.projection.matrix() * self.view.matrix()
    }

    /// The view frustum for this camera, extracted from
    /// [`Camera::view_projection`] with the Gribb–Hartmann method.
    #[inline]
    #[must_use]
    pub fn frustum(self) -> Frustum {
        Frustum::from_view_proj(self.view_projection())
    }
}
