//! Bevy-compatible prelude: glam-shaped aliases and compatibility helpers
//! (design doc §19, migration strategy), behind the `compat-bevy` feature.
//!
//! Prism is a standalone engine and depends on no `bevy_*` crate, but its
//! transform API is deliberately Bevy-shaped so a port is mostly mechanical.
//! This module gathers the names a Bevy codebase reaches for — the glam-named
//! math types and the `Transform`/`GlobalTransform` pair — behind one import,
//! plus a thin extension trait that supplies Bevy's method spellings on top of
//! Prism's.
//!
//! The math re-exports are the same [`prism_math`] types, simply surfaced under
//! their glam names (including [`Affine3A`], glam's `SIMD` affine, aliased to
//! Prism's [`Affine3`](prism_math::Affine3)). No conversion or wrapper is
//! involved, so there is zero cost to using this prelude over the native one.

pub use prism_math::{Mat2, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

/// glam's name for a 3D affine transform. Prism stores affines in
/// [`Affine3`](prism_math::Affine3); this alias lets Bevy-shaped code keep its
/// `Affine3A` spelling.
pub type Affine3A = prism_math::Affine3;

pub use crate::transform_2d::{GlobalTransform2d, Transform2d};
pub use crate::{GlobalTransform, Transform, TransformGraph};

/// Bevy's method spellings for [`GlobalTransform`], layered on top of Prism's.
///
/// Bevy code calls `global.compute_matrix()` and `global.compute_affine()`;
/// Prism spells these [`GlobalTransform::to_matrix`] and
/// [`GlobalTransform::affine`]. This trait provides the Bevy names so ported
/// call sites compile unchanged.
pub trait BevyGlobalTransformExt {
    /// The world matrix as a [`Mat4`] (Bevy's `compute_matrix`).
    fn compute_matrix(&self) -> Mat4;
    /// The world transform as an [`Affine3A`] (Bevy's `compute_affine`).
    fn compute_affine(&self) -> Affine3A;
    /// The world transform decomposed back to a local-style [`Transform`]
    /// (Bevy's `compute_transform`).
    fn compute_transform(&self) -> Transform;
}

impl BevyGlobalTransformExt for GlobalTransform {
    #[inline]
    fn compute_matrix(&self) -> Mat4 {
        self.to_matrix()
    }
    #[inline]
    fn compute_affine(&self) -> Affine3A {
        self.affine()
    }
    #[inline]
    fn compute_transform(&self) -> Transform {
        GlobalTransform::compute_transform(self)
    }
}

/// Bevy-shaped prelude: `use prism_transform::compat_bevy::prelude::*;` to get
/// the glam math names plus the transform types and compatibility helpers in
/// one import.
pub mod prelude {
    pub use super::{
        Affine3A, BevyGlobalTransformExt, GlobalTransform, GlobalTransform2d, Mat2, Mat3, Mat4,
        Quat, Transform, Transform2d, TransformGraph, Vec2, Vec3, Vec4,
    };
}
