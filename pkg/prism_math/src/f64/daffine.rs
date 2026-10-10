//! [`DAffine3`]: the `f64` analogue of [`Affine3`](crate::Affine3).
//!
//! A 3x4 affine transform (a [`DMat3`] linear part plus a [`DVec3`]
//! translation). Cheaper to store and propagate than a full [`DMat4`] while
//! still representing rotation, non-uniform scale, shear, and translation. This
//! is the natural transform type for M3 big-world positions: the translation
//! carries full `f64` world coordinates before being rebased into small `f32`
//! offsets (see [`crate::bigworld`]).

use crate::f64::dmat::{DMat3, DMat4};
use crate::f64::dquat::DQuat;
use crate::f64::dvec::DVec3;
use crate::Affine3;
use core::ops::Mul;

/// A 3D affine transform stored as a linear 3x3 matrix and a translation.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct DAffine3 {
    /// Linear part (rotation * scale * shear).
    pub matrix3: DMat3,
    /// Translation part.
    pub translation: DVec3,
}

impl Default for DAffine3 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl DAffine3 {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        matrix3: DMat3::IDENTITY,
        translation: DVec3::ZERO,
    };

    /// Build from a linear part and translation.
    #[inline]
    pub const fn from_mat3_translation(matrix3: DMat3, translation: DVec3) -> Self {
        Self {
            matrix3,
            translation,
        }
    }
    /// Pure translation.
    #[inline]
    pub const fn from_translation(t: DVec3) -> Self {
        Self {
            matrix3: DMat3::IDENTITY,
            translation: t,
        }
    }
    /// Pure rotation.
    #[inline]
    pub fn from_quat(q: DQuat) -> Self {
        Self {
            matrix3: DMat3::from_quat(q),
            translation: DVec3::ZERO,
        }
    }
    /// Pure non-uniform scale.
    #[inline]
    pub fn from_scale(s: DVec3) -> Self {
        Self {
            matrix3: DMat3::from_scale(s),
            translation: DVec3::ZERO,
        }
    }
    /// Compose scale, then rotation, then translation.
    #[inline]
    pub fn from_scale_rotation_translation(
        scale: DVec3,
        rotation: DQuat,
        translation: DVec3,
    ) -> Self {
        let r = DMat3::from_quat(rotation);
        Self {
            matrix3: DMat3::from_cols(r.x_axis * scale.x, r.y_axis * scale.y, r.z_axis * scale.z),
            translation,
        }
    }
    /// Transform a point (applies the linear part then the translation).
    #[inline]
    pub fn transform_point3(self, p: DVec3) -> DVec3 {
        self.matrix3.mul_vec3(p) + self.translation
    }
    /// Transform a direction (ignores translation).
    #[inline]
    pub fn transform_vector3(self, v: DVec3) -> DVec3 {
        self.matrix3.mul_vec3(v)
    }
    /// Inverse transform.
    #[inline]
    pub fn inverse(self) -> Self {
        let m = self.matrix3.inverse();
        Self {
            matrix3: m,
            translation: -m.mul_vec3(self.translation),
        }
    }
    /// Convert to an equivalent [`DMat4`].
    #[inline]
    pub fn to_mat4(self) -> DMat4 {
        DMat4::from_cols(
            self.matrix3.x_axis.extend(0.0),
            self.matrix3.y_axis.extend(0.0),
            self.matrix3.z_axis.extend(0.0),
            self.translation.extend(1.0),
        )
    }
    /// Build from a [`DMat4`], dropping the (assumed affine) bottom row.
    #[inline]
    pub fn from_mat4(m: DMat4) -> Self {
        Self {
            matrix3: m.to_mat3(),
            translation: DVec3::new(m.w_axis.x, m.w_axis.y, m.w_axis.z),
        }
    }
    /// Recover `(scale, rotation, translation)` from this transform.
    /// Assumes the linear part is scale * rotation (no shear).
    #[inline]
    pub fn to_scale_rotation_translation(self) -> (DVec3, DQuat, DVec3) {
        let m = self.matrix3;
        let det = m.determinant();
        let sign = if det < 0.0 { -1.0 } else { 1.0 };
        let scale = DVec3::new(
            m.x_axis.length() * sign,
            m.y_axis.length(),
            m.z_axis.length(),
        );
        let inv = DVec3::new(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z);
        let rot = DMat3::from_cols(m.x_axis * inv.x, m.y_axis * inv.y, m.z_axis * inv.z);
        (scale, DQuat::from_mat3(rot), self.translation)
    }
    /// True if every component is finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.matrix3.x_axis.is_finite()
            && self.matrix3.y_axis.is_finite()
            && self.matrix3.z_axis.is_finite()
            && self.translation.is_finite()
    }
    /// Lossy conversion to the `f32` [`Affine3`].
    #[inline]
    pub fn as_affine3(self) -> Affine3 {
        Affine3::from_mat3_translation(self.matrix3.as_mat3(), self.translation.as_vec3())
    }
}

impl Affine3 {
    /// Widen to the `f64` [`DAffine3`].
    #[inline]
    pub fn as_daffine3(self) -> DAffine3 {
        DAffine3::from_mat3_translation(self.matrix3.as_dmat3(), self.translation.as_dvec3())
    }
}

impl Mul for DAffine3 {
    type Output = DAffine3;
    /// Compose two transforms. `a * b` applies `b` first, then `a`.
    #[inline]
    fn mul(self, rhs: DAffine3) -> DAffine3 {
        DAffine3 {
            matrix3: self.matrix3 * rhs.matrix3,
            translation: self.matrix3.mul_vec3(rhs.translation) + self.translation,
        }
    }
}
impl Mul<DVec3> for DAffine3 {
    type Output = DVec3;
    #[inline]
    fn mul(self, p: DVec3) -> DVec3 {
        self.transform_point3(p)
    }
}

impl core::fmt::Debug for DAffine3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "DAffine3 {{ matrix3: {:?}, translation: {:?} }}",
            self.matrix3, self.translation
        )
    }
}
