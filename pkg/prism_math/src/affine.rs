//! [`Affine3`]: a 3x4 affine transform (a `Mat3` linear part plus a `Vec3`
//! translation). Cheaper to store and propagate than a full `Mat4` while still
//! representing rotation, non-uniform scale, shear, and translation.

use crate::mat::{Mat3, Mat4};
use crate::quat::Quat;
use crate::vec::Vec3;
use core::ops::Mul;

/// A 3D affine transform stored as a linear 3x3 matrix and a translation.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Affine3 {
    /// Linear part (rotation * scale * shear).
    pub matrix3: Mat3,
    /// Translation part.
    pub translation: Vec3,
}

impl Default for Affine3 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine3 {
    /// The identity transform.
    pub const IDENTITY: Self = Self { matrix3: Mat3::IDENTITY, translation: Vec3::ZERO };

    /// Build from a linear part and translation.
    #[inline]
    pub const fn from_mat3_translation(matrix3: Mat3, translation: Vec3) -> Self {
        Self { matrix3, translation }
    }
    /// Pure translation.
    #[inline]
    pub const fn from_translation(t: Vec3) -> Self {
        Self { matrix3: Mat3::IDENTITY, translation: t }
    }
    /// Pure rotation.
    #[inline]
    pub fn from_quat(q: Quat) -> Self {
        Self { matrix3: Mat3::from_quat(q), translation: Vec3::ZERO }
    }
    /// Pure non-uniform scale.
    #[inline]
    pub fn from_scale(s: Vec3) -> Self {
        Self { matrix3: Mat3::from_scale(s), translation: Vec3::ZERO }
    }
    /// Compose scale, then rotation, then translation.
    #[inline]
    pub fn from_scale_rotation_translation(scale: Vec3, rotation: Quat, translation: Vec3) -> Self {
        let r = Mat3::from_quat(rotation);
        Self {
            matrix3: Mat3::from_cols(r.x_axis * scale.x, r.y_axis * scale.y, r.z_axis * scale.z),
            translation,
        }
    }
    /// Transform a point (applies the linear part then the translation).
    #[inline]
    pub fn transform_point3(self, p: Vec3) -> Vec3 {
        self.matrix3.mul_vec3(p) + self.translation
    }
    /// Transform a direction (ignores translation).
    #[inline]
    pub fn transform_vector3(self, v: Vec3) -> Vec3 {
        self.matrix3.mul_vec3(v)
    }
    /// Inverse transform.
    #[inline]
    pub fn inverse(self) -> Self {
        let m = self.matrix3.inverse();
        Self { matrix3: m, translation: -m.mul_vec3(self.translation) }
    }
    /// Convert to an equivalent [`Mat4`].
    #[inline]
    pub fn to_mat4(self) -> Mat4 {
        Mat4::from_cols(
            self.matrix3.x_axis.extend(0.0),
            self.matrix3.y_axis.extend(0.0),
            self.matrix3.z_axis.extend(0.0),
            self.translation.extend(1.0),
        )
    }
    /// Build from a [`Mat4`], dropping the (assumed affine) bottom row.
    #[inline]
    pub fn from_mat4(m: Mat4) -> Self {
        Self {
            matrix3: m.to_mat3(),
            translation: Vec3::new(m.w_axis.x, m.w_axis.y, m.w_axis.z),
        }
    }
    /// Recover `(scale, rotation, translation)` from this transform.
    /// Assumes the linear part is scale * rotation (no shear).
    #[inline]
    pub fn to_scale_rotation_translation(self) -> (Vec3, Quat, Vec3) {
        let m = self.matrix3;
        let det = m.determinant();
        let sign = if det < 0.0 { -1.0 } else { 1.0 };
        let scale = Vec3::new(
            m.x_axis.length() * sign,
            m.y_axis.length(),
            m.z_axis.length(),
        );
        let inv = Vec3::new(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z);
        let rot = Mat3::from_cols(m.x_axis * inv.x, m.y_axis * inv.y, m.z_axis * inv.z);
        (scale, Quat::from_mat3(rot), self.translation)
    }
    /// True if every component is finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.matrix3.x_axis.is_finite()
            && self.matrix3.y_axis.is_finite()
            && self.matrix3.z_axis.is_finite()
            && self.translation.is_finite()
    }
}

impl Mul for Affine3 {
    type Output = Affine3;
    /// Compose two transforms. `a * b` applies `b` first, then `a`.
    #[inline]
    fn mul(self, rhs: Affine3) -> Affine3 {
        Affine3 {
            matrix3: self.matrix3 * rhs.matrix3,
            translation: self.matrix3.mul_vec3(rhs.translation) + self.translation,
        }
    }
}
impl Mul<Vec3> for Affine3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, p: Vec3) -> Vec3 {
        self.transform_point3(p)
    }
}

impl core::fmt::Debug for Affine3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Affine3 {{ matrix3: {:?}, translation: {:?} }}", self.matrix3, self.translation)
    }
}
