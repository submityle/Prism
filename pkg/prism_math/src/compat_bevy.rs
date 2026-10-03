//! glam/`bevy_math`-shaped type aliases (`compat-bevy` feature).
//!
//! Prism's facade is already glam-shaped, but a handful of names differ from
//! glam: most notably glam spells the aligned affine transform `Affine3A`,
//! whereas Prism uses [`Affine3`](crate::Affine3). This module provides thin
//! aliases under glam's spelling plus a glam-style [`prelude`], so code ported
//! from a glam/`bevy_math` codebase keeps compiling with a single
//! `use prism_math::compat_bevy::prelude::*;`.
//!
//! These are zero-cost `type` aliases: they name the exact same Prism types and
//! do not introduce any new behavior or layout. The feature is additive and
//! never changes the primary API (per the design doc migration contract).

use crate as pm;

/// glam spells the aligned affine transform `Affine3A`; Prism's
/// [`Affine3`](crate::Affine3) is the same 3x4 affine transform.
pub type Affine3A = pm::Affine3;

/// Alias matching glam's `Vec2`.
pub type Vec2 = pm::Vec2;
/// Alias matching glam's `Vec3`.
pub type Vec3 = pm::Vec3;
/// Alias matching glam's `Vec3A`.
pub type Vec3A = pm::Vec3A;
/// Alias matching glam's `Vec4`.
pub type Vec4 = pm::Vec4;

/// Alias matching glam's `Mat2`.
pub type Mat2 = pm::Mat2;
/// Alias matching glam's `Mat3`.
pub type Mat3 = pm::Mat3;
/// Alias matching glam's `Mat4`.
pub type Mat4 = pm::Mat4;

/// Alias matching glam's `Quat`.
pub type Quat = pm::Quat;

/// Alias matching glam's double-precision `DVec2`.
pub type DVec2 = pm::DVec2;
/// Alias matching glam's double-precision `DVec3`.
pub type DVec3 = pm::DVec3;
/// Alias matching glam's double-precision `DVec4`.
pub type DVec4 = pm::DVec4;
/// Alias matching glam's double-precision `DMat2`.
pub type DMat2 = pm::DMat2;
/// Alias matching glam's double-precision `DMat3`.
pub type DMat3 = pm::DMat3;
/// Alias matching glam's double-precision `DMat4`.
pub type DMat4 = pm::DMat4;
/// Alias matching glam's double-precision `DQuat`.
pub type DQuat = pm::DQuat;
/// glam spells the double-precision aligned affine transform `DAffine3`.
pub type DAffine3 = pm::DAffine3;

/// A glam-shaped prelude re-exporting the aliases plus the shared constructor
/// free functions, so ported code can `use ...::prelude::*;`.
pub mod prelude {
    pub use super::{
        Affine3A, DAffine3, DMat2, DMat3, DMat4, DQuat, DVec2, DVec3, DVec4, Mat2, Mat3, Mat4,
        Quat, Vec2, Vec3, Vec3A, Vec4,
    };
    pub use crate::{dvec2, dvec3, dvec4, vec2, vec3, vec3a, vec4};
}
