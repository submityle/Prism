//! Double-precision column-major matrices: [`DMat2`], [`DMat3`], [`DMat4`].
//!
//! These mirror the `f32` [`Mat2`](crate::Mat2), [`Mat3`](crate::Mat3), and
//! [`Mat4`](crate::Mat4) facade types, exposing the same method set at `f64`
//! precision for the M3 big-world path. Columns are stored as vectors and
//! vectors are columns, so a point is transformed with `m * v` and composition
//! reads right-to-left (`a * b` applies `b` first).

use crate::f64::dquat::DQuat;
use crate::f64::dvec::{DVec2, DVec3, DVec4};
use crate::{Mat2, Mat3, Mat4};
use core::ops::{Add, Mul, Sub};

/// A 2x2 column-major `f64` matrix.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct DMat2 {
    /// First column.
    pub x_axis: DVec2,
    /// Second column.
    pub y_axis: DVec2,
}

/// A 3x3 column-major `f64` matrix.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct DMat3 {
    /// First column.
    pub x_axis: DVec3,
    /// Second column.
    pub y_axis: DVec3,
    /// Third column.
    pub z_axis: DVec3,
}

/// A 4x4 column-major `f64` matrix.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct DMat4 {
    /// First column.
    pub x_axis: DVec4,
    /// Second column.
    pub y_axis: DVec4,
    /// Third column.
    pub z_axis: DVec4,
    /// Fourth column.
    pub w_axis: DVec4,
}

impl Default for DMat2 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}
impl Default for DMat3 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}
impl Default for DMat4 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl DMat2 {
    /// The zero matrix.
    pub const ZERO: Self = Self { x_axis: DVec2::ZERO, y_axis: DVec2::ZERO };
    /// The identity matrix.
    pub const IDENTITY: Self = Self { x_axis: DVec2::X, y_axis: DVec2::Y };

    /// Build from column vectors.
    #[inline]
    pub const fn from_cols(x_axis: DVec2, y_axis: DVec2) -> Self {
        Self { x_axis, y_axis }
    }
    /// Determinant.
    #[inline]
    pub fn determinant(self) -> f64 {
        self.x_axis.x * self.y_axis.y - self.y_axis.x * self.x_axis.y
    }
    /// Transpose.
    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            x_axis: DVec2::new(self.x_axis.x, self.y_axis.x),
            y_axis: DVec2::new(self.x_axis.y, self.y_axis.y),
        }
    }
    /// Transform a 2D vector.
    #[inline]
    pub fn mul_vec2(self, v: DVec2) -> DVec2 {
        self.x_axis * v.x + self.y_axis * v.y
    }
    /// Inverse (panics in debug if singular).
    #[inline]
    pub fn inverse(self) -> Self {
        let det = self.determinant();
        debug_assert!(det != 0.0, "DMat2::inverse of singular matrix");
        let inv = 1.0 / det;
        Self {
            x_axis: DVec2::new(self.y_axis.y * inv, -self.x_axis.y * inv),
            y_axis: DVec2::new(-self.y_axis.x * inv, self.x_axis.x * inv),
        }
    }
    /// Lossy conversion to the `f32` [`Mat2`].
    #[inline]
    pub fn as_mat2(self) -> Mat2 {
        Mat2::from_cols(self.x_axis.as_vec2(), self.y_axis.as_vec2())
    }
}

impl DMat3 {
    /// The zero matrix.
    pub const ZERO: Self = Self { x_axis: DVec3::ZERO, y_axis: DVec3::ZERO, z_axis: DVec3::ZERO };
    /// The identity matrix.
    pub const IDENTITY: Self = Self { x_axis: DVec3::X, y_axis: DVec3::Y, z_axis: DVec3::Z };

    /// Build from column vectors.
    #[inline]
    pub const fn from_cols(x_axis: DVec3, y_axis: DVec3, z_axis: DVec3) -> Self {
        Self { x_axis, y_axis, z_axis }
    }
    /// Diagonal matrix from a scale vector.
    #[inline]
    pub const fn from_diagonal(d: DVec3) -> Self {
        Self::from_cols(
            DVec3::new(d.x, 0.0, 0.0),
            DVec3::new(0.0, d.y, 0.0),
            DVec3::new(0.0, 0.0, d.z),
        )
    }
    /// Scale matrix (non-uniform).
    #[inline]
    pub const fn from_scale(s: DVec3) -> Self {
        Self::from_diagonal(s)
    }
    /// Rotation matrix from a unit quaternion.
    #[inline]
    pub fn from_quat(q: DQuat) -> Self {
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        Self::from_cols(
            DVec3::new(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)),
            DVec3::new(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
            DVec3::new(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)),
        )
    }
    /// Transpose.
    #[inline]
    pub fn transpose(self) -> Self {
        Self::from_cols(
            DVec3::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
            DVec3::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
            DVec3::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
        )
    }
    /// Determinant.
    #[inline]
    pub fn determinant(self) -> f64 {
        self.z_axis.dot(self.x_axis.cross(self.y_axis))
    }
    /// Transform a 3D vector.
    #[inline]
    pub fn mul_vec3(self, v: DVec3) -> DVec3 {
        self.x_axis * v.x + self.y_axis * v.y + self.z_axis * v.z
    }
    /// Inverse (panics in debug if singular).
    #[inline]
    pub fn inverse(self) -> Self {
        let a = self.x_axis;
        let b = self.y_axis;
        let c = self.z_axis;
        // Rows of the adjugate are the cross products of the column pairs.
        let r0 = b.cross(c);
        let r1 = c.cross(a);
        let r2 = a.cross(b);
        let det = a.dot(r0);
        debug_assert!(det != 0.0, "DMat3::inverse of singular matrix");
        let inv = 1.0 / det;
        // `[r0; r1; r2]` are the adjugate rows; transpose into columns.
        Self::from_cols(
            DVec3::new(r0.x, r1.x, r2.x) * inv,
            DVec3::new(r0.y, r1.y, r2.y) * inv,
            DVec3::new(r0.z, r1.z, r2.z) * inv,
        )
    }
    /// Lossy conversion to the `f32` [`Mat3`].
    #[inline]
    pub fn as_mat3(self) -> Mat3 {
        Mat3::from_cols(self.x_axis.as_vec3(), self.y_axis.as_vec3(), self.z_axis.as_vec3())
    }
}

impl DMat4 {
    /// The zero matrix.
    pub const ZERO: Self =
        Self { x_axis: DVec4::ZERO, y_axis: DVec4::ZERO, z_axis: DVec4::ZERO, w_axis: DVec4::ZERO };
    /// The identity matrix.
    pub const IDENTITY: Self =
        Self { x_axis: DVec4::X, y_axis: DVec4::Y, z_axis: DVec4::Z, w_axis: DVec4::W };

    /// Build from column vectors.
    #[inline]
    pub const fn from_cols(x_axis: DVec4, y_axis: DVec4, z_axis: DVec4, w_axis: DVec4) -> Self {
        Self { x_axis, y_axis, z_axis, w_axis }
    }
    /// Translation matrix.
    #[inline]
    pub const fn from_translation(t: DVec3) -> Self {
        Self::from_cols(DVec4::X, DVec4::Y, DVec4::Z, DVec4::new(t.x, t.y, t.z, 1.0))
    }
    /// Non-uniform scale matrix.
    #[inline]
    pub const fn from_scale(s: DVec3) -> Self {
        Self::from_cols(
            DVec4::new(s.x, 0.0, 0.0, 0.0),
            DVec4::new(0.0, s.y, 0.0, 0.0),
            DVec4::new(0.0, 0.0, s.z, 0.0),
            DVec4::W,
        )
    }
    /// Build from the upper-left 3x3 rotation and a translation.
    #[inline]
    pub fn from_mat3_translation(m: DMat3, t: DVec3) -> Self {
        Self::from_cols(
            m.x_axis.extend(0.0),
            m.y_axis.extend(0.0),
            m.z_axis.extend(0.0),
            t.extend(1.0),
        )
    }
    /// Rotation matrix from a unit quaternion.
    #[inline]
    pub fn from_quat(q: DQuat) -> Self {
        Self::from_mat3_translation(DMat3::from_quat(q), DVec3::ZERO)
    }
    /// Compose scale, then rotation, then translation.
    #[inline]
    pub fn from_scale_rotation_translation(scale: DVec3, rotation: DQuat, translation: DVec3) -> Self {
        let r = DMat3::from_quat(rotation);
        Self::from_cols(
            (r.x_axis * scale.x).extend(0.0),
            (r.y_axis * scale.y).extend(0.0),
            (r.z_axis * scale.z).extend(0.0),
            translation.extend(1.0),
        )
    }
    /// Extract the upper-left 3x3 submatrix.
    #[inline]
    pub fn to_mat3(self) -> DMat3 {
        DMat3::from_cols(self.x_axis.truncate(), self.y_axis.truncate(), self.z_axis.truncate())
    }
    /// Transpose.
    #[inline]
    pub fn transpose(self) -> Self {
        Self::from_cols(
            DVec4::new(self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x),
            DVec4::new(self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y),
            DVec4::new(self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z),
            DVec4::new(self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w),
        )
    }
    /// Transform a homogeneous 4D vector.
    #[inline]
    pub fn mul_vec4(self, v: DVec4) -> DVec4 {
        self.x_axis * v.x + self.y_axis * v.y + self.z_axis * v.z + self.w_axis * v.w
    }
    /// Transform a point (implicit `w = 1`, perspective divide applied).
    #[inline]
    pub fn transform_point3(self, p: DVec3) -> DVec3 {
        let r = self.x_axis * p.x + self.y_axis * p.y + self.z_axis * p.z + self.w_axis;
        let inv_w = 1.0 / r.w;
        DVec3::new(r.x * inv_w, r.y * inv_w, r.z * inv_w)
    }
    /// Transform a direction (implicit `w = 0`).
    #[inline]
    pub fn transform_vector3(self, v: DVec3) -> DVec3 {
        (self.x_axis * v.x + self.y_axis * v.y + self.z_axis * v.z).truncate()
    }
    /// Determinant.
    #[inline]
    pub fn determinant(self) -> f64 {
        let m = self;
        let (m00, m01, m02, m03) = (m.x_axis.x, m.x_axis.y, m.x_axis.z, m.x_axis.w);
        let (m10, m11, m12, m13) = (m.y_axis.x, m.y_axis.y, m.y_axis.z, m.y_axis.w);
        let (m20, m21, m22, m23) = (m.z_axis.x, m.z_axis.y, m.z_axis.z, m.z_axis.w);
        let (m30, m31, m32, m33) = (m.w_axis.x, m.w_axis.y, m.w_axis.z, m.w_axis.w);

        let a2323 = m22 * m33 - m23 * m32;
        let a1323 = m21 * m33 - m23 * m31;
        let a1223 = m21 * m32 - m22 * m31;
        let a0323 = m20 * m33 - m23 * m30;
        let a0223 = m20 * m32 - m22 * m30;
        let a0123 = m20 * m31 - m21 * m30;

        m00 * (m11 * a2323 - m12 * a1323 + m13 * a1223)
            - m01 * (m10 * a2323 - m12 * a0323 + m13 * a0223)
            + m02 * (m10 * a1323 - m11 * a0323 + m13 * a0123)
            - m03 * (m10 * a1223 - m11 * a0223 + m12 * a0123)
    }
    /// Inverse (panics in debug if singular). Full 4x4 cofactor inverse.
    #[inline]
    pub fn inverse(self) -> Self {
        let m = self;
        let (m00, m01, m02, m03) = (m.x_axis.x, m.x_axis.y, m.x_axis.z, m.x_axis.w);
        let (m10, m11, m12, m13) = (m.y_axis.x, m.y_axis.y, m.y_axis.z, m.y_axis.w);
        let (m20, m21, m22, m23) = (m.z_axis.x, m.z_axis.y, m.z_axis.z, m.z_axis.w);
        let (m30, m31, m32, m33) = (m.w_axis.x, m.w_axis.y, m.w_axis.z, m.w_axis.w);

        let coef00 = m22 * m33 - m32 * m23;
        let coef02 = m12 * m33 - m32 * m13;
        let coef03 = m12 * m23 - m22 * m13;
        let coef04 = m21 * m33 - m31 * m23;
        let coef06 = m11 * m33 - m31 * m13;
        let coef07 = m11 * m23 - m21 * m13;
        let coef08 = m21 * m32 - m31 * m22;
        let coef10 = m11 * m32 - m31 * m12;
        let coef11 = m11 * m22 - m21 * m12;
        let coef12 = m20 * m33 - m30 * m23;
        let coef14 = m10 * m33 - m30 * m13;
        let coef15 = m10 * m23 - m20 * m13;
        let coef16 = m20 * m32 - m30 * m22;
        let coef18 = m10 * m32 - m30 * m12;
        let coef19 = m10 * m22 - m20 * m12;
        let coef20 = m20 * m31 - m30 * m21;
        let coef22 = m10 * m31 - m30 * m11;
        let coef23 = m10 * m21 - m20 * m11;

        let fac0 = DVec4::new(coef00, coef00, coef02, coef03);
        let fac1 = DVec4::new(coef04, coef04, coef06, coef07);
        let fac2 = DVec4::new(coef08, coef08, coef10, coef11);
        let fac3 = DVec4::new(coef12, coef12, coef14, coef15);
        let fac4 = DVec4::new(coef16, coef16, coef18, coef19);
        let fac5 = DVec4::new(coef20, coef20, coef22, coef23);

        let vec0 = DVec4::new(m10, m00, m00, m00);
        let vec1 = DVec4::new(m11, m01, m01, m01);
        let vec2 = DVec4::new(m12, m02, m02, m02);
        let vec3 = DVec4::new(m13, m03, m03, m03);

        let inv0 = vec1 * fac0 - vec2 * fac1 + vec3 * fac2;
        let inv1 = vec0 * fac0 - vec2 * fac3 + vec3 * fac4;
        let inv2 = vec0 * fac1 - vec1 * fac3 + vec3 * fac5;
        let inv3 = vec0 * fac2 - vec1 * fac4 + vec2 * fac5;

        let sign_a = DVec4::new(1.0, -1.0, 1.0, -1.0);
        let sign_b = DVec4::new(-1.0, 1.0, -1.0, 1.0);

        let inverse = DMat4::from_cols(
            inv0 * sign_a,
            inv1 * sign_b,
            inv2 * sign_a,
            inv3 * sign_b,
        );

        let col0 = DVec4::new(
            inverse.x_axis.x,
            inverse.y_axis.x,
            inverse.z_axis.x,
            inverse.w_axis.x,
        );
        let det = m.x_axis.dot(col0);
        debug_assert!(det != 0.0, "DMat4::inverse of singular matrix");
        let inv_det = 1.0 / det;
        DMat4::from_cols(
            inverse.x_axis * inv_det,
            inverse.y_axis * inv_det,
            inverse.z_axis * inv_det,
            inverse.w_axis * inv_det,
        )
    }
    /// Lossy conversion to the `f32` [`Mat4`].
    #[inline]
    pub fn as_mat4(self) -> Mat4 {
        Mat4::from_cols(
            self.x_axis.as_vec4(),
            self.y_axis.as_vec4(),
            self.z_axis.as_vec4(),
            self.w_axis.as_vec4(),
        )
    }
}

// ---- f32 -> f64 widening conversions --------------------------------------

impl Mat2 {
    /// Widen to the `f64` [`DMat2`].
    #[inline]
    pub fn as_dmat2(self) -> DMat2 {
        DMat2::from_cols(self.x_axis.as_dvec2(), self.y_axis.as_dvec2())
    }
}
impl Mat3 {
    /// Widen to the `f64` [`DMat3`].
    #[inline]
    pub fn as_dmat3(self) -> DMat3 {
        DMat3::from_cols(self.x_axis.as_dvec3(), self.y_axis.as_dvec3(), self.z_axis.as_dvec3())
    }
}
impl Mat4 {
    /// Widen to the `f64` [`DMat4`].
    #[inline]
    pub fn as_dmat4(self) -> DMat4 {
        DMat4::from_cols(
            self.x_axis.as_dvec4(),
            self.y_axis.as_dvec4(),
            self.z_axis.as_dvec4(),
            self.w_axis.as_dvec4(),
        )
    }
}

// ---- operator impls -------------------------------------------------------

impl Mul for DMat2 {
    type Output = DMat2;
    #[inline]
    fn mul(self, r: DMat2) -> DMat2 {
        DMat2::from_cols(self.mul_vec2(r.x_axis), self.mul_vec2(r.y_axis))
    }
}
impl Mul<DVec2> for DMat2 {
    type Output = DVec2;
    #[inline]
    fn mul(self, v: DVec2) -> DVec2 {
        self.mul_vec2(v)
    }
}

impl Mul for DMat3 {
    type Output = DMat3;
    #[inline]
    fn mul(self, r: DMat3) -> DMat3 {
        DMat3::from_cols(
            self.mul_vec3(r.x_axis),
            self.mul_vec3(r.y_axis),
            self.mul_vec3(r.z_axis),
        )
    }
}
impl Mul<DVec3> for DMat3 {
    type Output = DVec3;
    #[inline]
    fn mul(self, v: DVec3) -> DVec3 {
        self.mul_vec3(v)
    }
}

impl Mul for DMat4 {
    type Output = DMat4;
    #[inline]
    fn mul(self, r: DMat4) -> DMat4 {
        DMat4::from_cols(
            self.mul_vec4(r.x_axis),
            self.mul_vec4(r.y_axis),
            self.mul_vec4(r.z_axis),
            self.mul_vec4(r.w_axis),
        )
    }
}
impl Mul<DVec4> for DMat4 {
    type Output = DVec4;
    #[inline]
    fn mul(self, v: DVec4) -> DVec4 {
        self.mul_vec4(v)
    }
}

impl Add for DMat4 {
    type Output = DMat4;
    #[inline]
    fn add(self, r: DMat4) -> DMat4 {
        DMat4::from_cols(
            self.x_axis + r.x_axis,
            self.y_axis + r.y_axis,
            self.z_axis + r.z_axis,
            self.w_axis + r.w_axis,
        )
    }
}
impl Sub for DMat4 {
    type Output = DMat4;
    #[inline]
    fn sub(self, r: DMat4) -> DMat4 {
        DMat4::from_cols(
            self.x_axis - r.x_axis,
            self.y_axis - r.y_axis,
            self.z_axis - r.z_axis,
            self.w_axis - r.w_axis,
        )
    }
}

impl core::fmt::Debug for DMat2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DMat2[{:?}, {:?}]", self.x_axis, self.y_axis)
    }
}
impl core::fmt::Debug for DMat3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DMat3[{:?}, {:?}, {:?}]", self.x_axis, self.y_axis, self.z_axis)
    }
}
impl core::fmt::Debug for DMat4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "DMat4[{:?}, {:?}, {:?}, {:?}]",
            self.x_axis, self.y_axis, self.z_axis, self.w_axis
        )
    }
}
