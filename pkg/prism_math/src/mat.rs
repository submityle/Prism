//! Column-major floating-point matrices: [`Mat2`], [`Mat3`], [`Mat4`].
//!
//! Columns are stored as vectors and vectors are columns, so a point is
//! transformed with `m * v` and composition reads right-to-left (`a * b`
//! applies `b` first).

use crate::backend;
use crate::quat::Quat;
use crate::vec::{Vec2, Vec3, Vec4};
use core::ops::{Add, Mul, Sub};

/// A 2x2 column-major matrix.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat2 {
    /// First column.
    pub x_axis: Vec2,
    /// Second column.
    pub y_axis: Vec2,
}

/// A 3x3 column-major matrix.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat3 {
    /// First column.
    pub x_axis: Vec3,
    /// Second column.
    pub y_axis: Vec3,
    /// Third column.
    pub z_axis: Vec3,
}

/// A 4x4 column-major matrix.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat4 {
    /// First column.
    pub x_axis: Vec4,
    /// Second column.
    pub y_axis: Vec4,
    /// Third column.
    pub z_axis: Vec4,
    /// Fourth column.
    pub w_axis: Vec4,
}

impl Default for Mat2 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}
impl Default for Mat3 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}
impl Default for Mat4 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat2 {
    /// The zero matrix.
    pub const ZERO: Self = Self { x_axis: Vec2::ZERO, y_axis: Vec2::ZERO };
    /// The identity matrix.
    pub const IDENTITY: Self = Self { x_axis: Vec2::X, y_axis: Vec2::Y };

    /// Build from column vectors.
    #[inline]
    pub const fn from_cols(x_axis: Vec2, y_axis: Vec2) -> Self {
        Self { x_axis, y_axis }
    }
    /// Determinant.
    #[inline]
    pub fn determinant(self) -> f32 {
        self.x_axis.x * self.y_axis.y - self.y_axis.x * self.x_axis.y
    }
    /// Transpose.
    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            x_axis: Vec2::new(self.x_axis.x, self.y_axis.x),
            y_axis: Vec2::new(self.x_axis.y, self.y_axis.y),
        }
    }
    /// Transform a 2D vector.
    #[inline]
    pub fn mul_vec2(self, v: Vec2) -> Vec2 {
        self.x_axis * v.x + self.y_axis * v.y
    }
    /// Inverse (panics in debug if singular).
    #[inline]
    pub fn inverse(self) -> Self {
        let det = self.determinant();
        debug_assert!(det != 0.0, "Mat2::inverse of singular matrix");
        let inv = 1.0 / det;
        Self {
            x_axis: Vec2::new(self.y_axis.y * inv, -self.x_axis.y * inv),
            y_axis: Vec2::new(-self.y_axis.x * inv, self.x_axis.x * inv),
        }
    }
}

impl Mat3 {
    /// The zero matrix.
    pub const ZERO: Self = Self { x_axis: Vec3::ZERO, y_axis: Vec3::ZERO, z_axis: Vec3::ZERO };
    /// The identity matrix.
    pub const IDENTITY: Self = Self { x_axis: Vec3::X, y_axis: Vec3::Y, z_axis: Vec3::Z };

    /// Build from column vectors.
    #[inline]
    pub const fn from_cols(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Self {
        Self { x_axis, y_axis, z_axis }
    }
    /// Diagonal matrix from a scale vector.
    #[inline]
    pub const fn from_diagonal(d: Vec3) -> Self {
        Self::from_cols(
            Vec3::new(d.x, 0.0, 0.0),
            Vec3::new(0.0, d.y, 0.0),
            Vec3::new(0.0, 0.0, d.z),
        )
    }
    /// Scale matrix (non-uniform).
    #[inline]
    pub const fn from_scale(s: Vec3) -> Self {
        Self::from_diagonal(s)
    }
    /// Rotation matrix from a unit quaternion.
    #[inline]
    pub fn from_quat(q: Quat) -> Self {
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        Self::from_cols(
            Vec3::new(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)),
            Vec3::new(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
            Vec3::new(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)),
        )
    }
    /// Transpose.
    #[inline]
    pub fn transpose(self) -> Self {
        Self::from_cols(
            Vec3::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
            Vec3::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
            Vec3::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
        )
    }
    /// Determinant.
    #[inline]
    pub fn determinant(self) -> f32 {
        self.z_axis.dot(self.x_axis.cross(self.y_axis))
    }
    /// Transform a 3D vector.
    #[inline]
    pub fn mul_vec3(self, v: Vec3) -> Vec3 {
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
        debug_assert!(det != 0.0, "Mat3::inverse of singular matrix");
        let inv_det = 1.0 / det;
        // Inverse = (1/det) * adjugate; adjugate rows become inverse columns.
        Self::from_cols(
            Vec3::new(r0.x, r1.x, r2.x) * inv_det,
            Vec3::new(r0.y, r1.y, r2.y) * inv_det,
            Vec3::new(r0.z, r1.z, r2.z) * inv_det,
        )
    }
}

impl Mat4 {
    /// The zero matrix.
    pub const ZERO: Self =
        Self { x_axis: Vec4::ZERO, y_axis: Vec4::ZERO, z_axis: Vec4::ZERO, w_axis: Vec4::ZERO };
    /// The identity matrix.
    pub const IDENTITY: Self =
        Self { x_axis: Vec4::X, y_axis: Vec4::Y, z_axis: Vec4::Z, w_axis: Vec4::W };

    /// Build from column vectors.
    #[inline]
    pub const fn from_cols(x_axis: Vec4, y_axis: Vec4, z_axis: Vec4, w_axis: Vec4) -> Self {
        Self { x_axis, y_axis, z_axis, w_axis }
    }
    /// Translation matrix.
    #[inline]
    pub const fn from_translation(t: Vec3) -> Self {
        Self::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::new(t.x, t.y, t.z, 1.0))
    }
    /// Non-uniform scale matrix.
    #[inline]
    pub const fn from_scale(s: Vec3) -> Self {
        Self::from_cols(
            Vec4::new(s.x, 0.0, 0.0, 0.0),
            Vec4::new(0.0, s.y, 0.0, 0.0),
            Vec4::new(0.0, 0.0, s.z, 0.0),
            Vec4::W,
        )
    }
    /// Build from the upper-left 3x3 rotation and a translation.
    #[inline]
    pub fn from_mat3_translation(m: Mat3, t: Vec3) -> Self {
        Self::from_cols(
            m.x_axis.extend(0.0),
            m.y_axis.extend(0.0),
            m.z_axis.extend(0.0),
            t.extend(1.0),
        )
    }
    /// Rotation matrix from a unit quaternion.
    #[inline]
    pub fn from_quat(q: Quat) -> Self {
        Self::from_mat3_translation(Mat3::from_quat(q), Vec3::ZERO)
    }
    /// Compose scale, then rotation, then translation.
    #[inline]
    pub fn from_scale_rotation_translation(scale: Vec3, rotation: Quat, translation: Vec3) -> Self {
        let r = Mat3::from_quat(rotation);
        Self::from_cols(
            (r.x_axis * scale.x).extend(0.0),
            (r.y_axis * scale.y).extend(0.0),
            (r.z_axis * scale.z).extend(0.0),
            translation.extend(1.0),
        )
    }
    /// Extract the upper-left 3x3 submatrix.
    #[inline]
    pub fn to_mat3(self) -> Mat3 {
        Mat3::from_cols(self.x_axis.truncate(), self.y_axis.truncate(), self.z_axis.truncate())
    }
    /// Transpose.
    #[inline]
    pub fn transpose(self) -> Self {
        Self::from_cols(
            Vec4::new(self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x),
            Vec4::new(self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y),
            Vec4::new(self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z),
            Vec4::new(self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w),
        )
    }
    /// Transform a homogeneous 4D vector.
    #[inline]
    pub fn mul_vec4(self, v: Vec4) -> Vec4 {
        let cols = [
            self.x_axis.to_array(),
            self.y_axis.to_array(),
            self.z_axis.to_array(),
            self.w_axis.to_array(),
        ];
        Vec4::from_array(backend::mat4_mul_vec4(&cols, v.to_array()))
    }
    /// Transform a point (implicit `w = 1`, perspective divide applied).
    #[inline]
    pub fn transform_point3(self, p: Vec3) -> Vec3 {
        let r = self.x_axis * p.x + self.y_axis * p.y + self.z_axis * p.z + self.w_axis;
        let inv_w = 1.0 / r.w;
        Vec3::new(r.x * inv_w, r.y * inv_w, r.z * inv_w)
    }
    /// Transform a direction (implicit `w = 0`).
    #[inline]
    pub fn transform_vector3(self, v: Vec3) -> Vec3 {
        (self.x_axis * v.x + self.y_axis * v.y + self.z_axis * v.z).truncate()
    }

    /// Determinant.
    #[inline]
    pub fn determinant(self) -> f32 {
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

        let fac0 = Vec4::new(coef00, coef00, coef02, coef03);
        let fac1 = Vec4::new(coef04, coef04, coef06, coef07);
        let fac2 = Vec4::new(coef08, coef08, coef10, coef11);
        let fac3 = Vec4::new(coef12, coef12, coef14, coef15);
        let fac4 = Vec4::new(coef16, coef16, coef18, coef19);
        let fac5 = Vec4::new(coef20, coef20, coef22, coef23);

        let vec0 = Vec4::new(m10, m00, m00, m00);
        let vec1 = Vec4::new(m11, m01, m01, m01);
        let vec2 = Vec4::new(m12, m02, m02, m02);
        let vec3 = Vec4::new(m13, m03, m03, m03);

        let inv0 = vec1 * fac0 - vec2 * fac1 + vec3 * fac2;
        let inv1 = vec0 * fac0 - vec2 * fac3 + vec3 * fac4;
        let inv2 = vec0 * fac1 - vec1 * fac3 + vec3 * fac5;
        let inv3 = vec0 * fac2 - vec1 * fac4 + vec2 * fac5;

        let sign_a = Vec4::new(1.0, -1.0, 1.0, -1.0);
        let sign_b = Vec4::new(-1.0, 1.0, -1.0, 1.0);

        let inverse = Mat4::from_cols(
            inv0 * sign_a,
            inv1 * sign_b,
            inv2 * sign_a,
            inv3 * sign_b,
        );

        let col0 = Vec4::new(
            inverse.x_axis.x,
            inverse.y_axis.x,
            inverse.z_axis.x,
            inverse.w_axis.x,
        );
        let det = m.x_axis.dot(col0);
        debug_assert!(det != 0.0, "Mat4::inverse of singular matrix");
        let inv_det = 1.0 / det;
        Mat4::from_cols(
            inverse.x_axis * inv_det,
            inverse.y_axis * inv_det,
            inverse.z_axis * inv_det,
            inverse.w_axis * inv_det,
        )
    }
}

impl Mul for Mat2 {
    type Output = Mat2;
    #[inline]
    fn mul(self, r: Mat2) -> Mat2 {
        Mat2::from_cols(self.mul_vec2(r.x_axis), self.mul_vec2(r.y_axis))
    }
}
impl Mul<Vec2> for Mat2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, v: Vec2) -> Vec2 {
        self.mul_vec2(v)
    }
}

impl Mul for Mat3 {
    type Output = Mat3;
    #[inline]
    fn mul(self, r: Mat3) -> Mat3 {
        Mat3::from_cols(
            self.mul_vec3(r.x_axis),
            self.mul_vec3(r.y_axis),
            self.mul_vec3(r.z_axis),
        )
    }
}
impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        self.mul_vec3(v)
    }
}

impl Mul for Mat4 {
    type Output = Mat4;
    #[inline]
    fn mul(self, r: Mat4) -> Mat4 {
        let a = [
            self.x_axis.to_array(),
            self.y_axis.to_array(),
            self.z_axis.to_array(),
            self.w_axis.to_array(),
        ];
        let b = [
            r.x_axis.to_array(),
            r.y_axis.to_array(),
            r.z_axis.to_array(),
            r.w_axis.to_array(),
        ];
        let m = backend::mat4_mul(&a, &b);
        Mat4::from_cols(
            Vec4::from_array(m[0]),
            Vec4::from_array(m[1]),
            Vec4::from_array(m[2]),
            Vec4::from_array(m[3]),
        )
    }
}
impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, v: Vec4) -> Vec4 {
        self.mul_vec4(v)
    }
}

impl Add for Mat4 {
    type Output = Mat4;
    #[inline]
    fn add(self, r: Mat4) -> Mat4 {
        Mat4::from_cols(
            self.x_axis + r.x_axis,
            self.y_axis + r.y_axis,
            self.z_axis + r.z_axis,
            self.w_axis + r.w_axis,
        )
    }
}
impl Sub for Mat4 {
    type Output = Mat4;
    #[inline]
    fn sub(self, r: Mat4) -> Mat4 {
        Mat4::from_cols(
            self.x_axis - r.x_axis,
            self.y_axis - r.y_axis,
            self.z_axis - r.z_axis,
            self.w_axis - r.w_axis,
        )
    }
}

impl core::fmt::Debug for Mat3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Mat3[{:?}, {:?}, {:?}]", self.x_axis, self.y_axis, self.z_axis)
    }
}
impl core::fmt::Debug for Mat4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Mat4[{:?}, {:?}, {:?}, {:?}]",
            self.x_axis, self.y_axis, self.z_axis, self.w_axis
        )
    }
}
