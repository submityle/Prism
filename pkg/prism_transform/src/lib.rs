//! # prism_transform
//!
//! The transform algebra at the base of Prism's spatial hierarchy.
//!
//! - [`Transform`] is the **local** TRS (translation / rotation / scale) a user
//!   writes. Storing TRS (not a matrix) keeps rotation independently
//!   interpolable (`slerp`) and avoids per-frame matrix decomposition.
//! - [`GlobalTransform`] is the **world-space** result, stored as an
//!   [`Affine3`] because hierarchical accumulation of non-uniform scale can
//!   introduce shear that a pure TRS cannot represent.
//!
//! ## Milestone status (per the design-doc roadmap)
//! - **M0 (this crate, done):** the standalone algebra — `Transform`,
//!   `GlobalTransform`, composition, inverse, and direction/helper methods,
//!   with round-trip and associativity tests.
//! - **M1+ (planned):** ECS-relation hierarchy propagation, dirty-subtree
//!   incremental updates, parallel propagation via `prism_tasks`, fixed-step
//!   interpolation, big-world/deterministic paths, and GPU upload. Those
//!   require `prism_ecs`/`prism_tasks` and are intentionally not part of M0.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

use prism_math::{Affine3, Mat4, Quat, Vec3};

/// A node's local transform: translation, rotation, and scale relative to its
/// parent (or to the world when it has no parent).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Transform {
    /// Position relative to the parent.
    pub translation: Vec3,
    /// Orientation relative to the parent.
    pub rotation: Quat,
    /// Non-uniform scale relative to the parent.
    pub scale: Vec3,
}

impl Default for Transform {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// The identity transform (no translation, no rotation, unit scale).
    pub const IDENTITY: Self =
        Self { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE };

    /// Translation-only transform from components.
    #[inline]
    pub const fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self { translation: Vec3::new(x, y, z), rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }
    /// Translation-only transform.
    #[inline]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self { translation, rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }
    /// Rotation-only transform.
    #[inline]
    pub const fn from_rotation(rotation: Quat) -> Self {
        Self { translation: Vec3::ZERO, rotation, scale: Vec3::ONE }
    }
    /// Scale-only transform.
    #[inline]
    pub const fn from_scale(scale: Vec3) -> Self {
        Self { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale }
    }

    /// Builder: set translation.
    #[inline]
    pub const fn with_translation(mut self, t: Vec3) -> Self {
        self.translation = t;
        self
    }
    /// Builder: set rotation.
    #[inline]
    pub const fn with_rotation(mut self, r: Quat) -> Self {
        self.rotation = r;
        self
    }
    /// Builder: set scale.
    #[inline]
    pub const fn with_scale(mut self, s: Vec3) -> Self {
        self.scale = s;
        self
    }

    /// Local +X ("right") axis in parent space.
    #[inline]
    pub fn local_x(&self) -> Vec3 {
        self.rotation * Vec3::X
    }
    /// Local +Y ("up") axis in parent space.
    #[inline]
    pub fn local_y(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }
    /// Local +Z axis in parent space.
    #[inline]
    pub fn local_z(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }
    /// Forward direction (-Z) in parent space.
    #[inline]
    pub fn forward(&self) -> Vec3 {
        -self.local_z()
    }

    /// Rotate this transform in place by `delta` (applied after current).
    #[inline]
    pub fn rotate(&mut self, delta: Quat) {
        self.rotation = (delta * self.rotation).normalize();
    }
    /// Translate this transform in place in parent space.
    #[inline]
    pub fn translate(&mut self, delta: Vec3) {
        self.translation += delta;
    }

    /// Convert this TRS to an [`Affine3`].
    #[inline]
    pub fn to_affine(&self) -> Affine3 {
        Affine3::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
    /// Convert this TRS to a [`Mat4`].
    #[inline]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Transform a point from local space into parent space.
    #[inline]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.translation + self.rotation * (self.scale * point)
    }

    /// Compose with a child transform: `self` is the parent, `child` is local
    /// to `self`. The result is expressed in `self`'s parent space.
    ///
    /// Note: when both parent and child carry non-uniform scale the exact
    /// result can contain shear, which a pure TRS cannot represent. This
    /// method composes the TRS components directly (translation/rotation exact,
    /// scale multiplied component-wise); use [`Transform::mul_affine`] when a
    /// lossless result is required.
    #[inline]
    pub fn mul_transform(&self, child: &Transform) -> Transform {
        Transform {
            translation: self.transform_point(child.translation),
            rotation: (self.rotation * child.rotation).normalize(),
            scale: self.scale * child.scale,
        }
    }

    /// Lossless composition via affine math (may contain shear).
    #[inline]
    pub fn mul_affine(&self, child: &Transform) -> Affine3 {
        self.to_affine() * child.to_affine()
    }
}

impl core::ops::Mul for Transform {
    type Output = Transform;
    #[inline]
    fn mul(self, child: Transform) -> Transform {
        self.mul_transform(&child)
    }
}

/// World-space transform produced by hierarchy propagation. Stored as an
/// [`Affine3`] so accumulated non-uniform scale/shear is representable.
///
/// In M0 this is a value type with full algebra; the systems that actually
/// write it from an ECS hierarchy arrive in M1.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GlobalTransform(pub Affine3);

impl Default for GlobalTransform {
    #[inline]
    fn default() -> Self {
        Self(Affine3::IDENTITY)
    }
}

impl GlobalTransform {
    /// The identity world transform.
    pub const IDENTITY: Self = Self(Affine3::IDENTITY);

    /// Build from a local [`Transform`] (as if it had no parent).
    #[inline]
    pub fn from_transform(t: &Transform) -> Self {
        Self(t.to_affine())
    }
    /// The underlying affine.
    #[inline]
    pub fn affine(&self) -> Affine3 {
        self.0
    }
    /// World-space translation.
    #[inline]
    pub fn translation(&self) -> Vec3 {
        self.0.translation
    }
    /// As a [`Mat4`] for GPU upload or legacy interop.
    #[inline]
    pub fn to_matrix(&self) -> Mat4 {
        self.0.to_mat4()
    }
    /// Transform a point from local to world space.
    #[inline]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.0.transform_point3(point)
    }
    /// Inverse world transform.
    #[inline]
    pub fn inverse(&self) -> Self {
        Self(self.0.inverse())
    }

    /// Propagation step: combine this parent world transform with a child's
    /// local transform to produce the child's world transform.
    #[inline]
    pub fn mul_transform(&self, local: &Transform) -> GlobalTransform {
        GlobalTransform(self.0 * local.to_affine())
    }

    /// Recover an approximate local [`Transform`] (assumes no shear).
    #[inline]
    pub fn compute_transform(&self) -> Transform {
        let (scale, rotation, translation) = self.0.to_scale_rotation_translation();
        Transform { translation, rotation, scale }
    }
}

impl From<Transform> for GlobalTransform {
    #[inline]
    fn from(t: Transform) -> Self {
        Self::from_transform(&t)
    }
}

/// Common imports.
pub mod prelude {
    pub use crate::{GlobalTransform, Transform};
}

#[cfg(test)]
mod tests;
