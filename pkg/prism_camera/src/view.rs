//! The view half of a camera: a world-space pose inverted into a view matrix.

use prism_math::{look_at_rh, Affine3, Mat4, Quat, Vec3};

/// A camera's world-space pose, stored as an [`Affine3`] (translation plus a
/// rotation/scale linear part).
///
/// A [`View`] keeps the pose in *world* space — the natural form to author and
/// animate — and inverts it to the world-to-camera transform the renderer
/// wants. Keeping the world pose (rather than only the derived matrix) avoids
/// re-decomposing a matrix when the camera is moved.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct View {
    /// The camera's world transform.
    pub transform: Affine3,
}

impl View {
    /// Wrap an existing world transform.
    #[inline]
    #[must_use]
    pub const fn from_transform(transform: Affine3) -> Self {
        Self { transform }
    }

    /// Build a view from a world translation and rotation.
    #[inline]
    #[must_use]
    pub fn from_translation_rotation(translation: Vec3, rotation: Quat) -> Self {
        Self {
            transform: Affine3::from_scale_rotation_translation(Vec3::ONE, rotation, translation),
        }
    }

    /// Build a view that looks from `eye` toward `target`, using `up` as the
    /// vertical hint (right-handed; the camera's forward is `-Z`).
    #[inline]
    #[must_use]
    pub fn looking_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        Self {
            transform: Affine3::from_mat4(look_at_rh(eye, target, up)).inverse(),
        }
    }

    /// The camera's world-space position.
    #[inline]
    #[must_use]
    pub fn position(self) -> Vec3 {
        self.transform.translation
    }

    /// The camera-to-world matrix (this view's world transform).
    #[inline]
    #[must_use]
    pub fn transform_matrix(self) -> Mat4 {
        self.transform.to_mat4()
    }

    /// The world-to-camera (view) matrix: the inverse of the world transform.
    #[inline]
    #[must_use]
    pub fn matrix(self) -> Mat4 {
        self.transform.inverse().to_mat4()
    }
}
