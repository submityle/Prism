//! Unit quaternions for 3D rotation.
//!
//! Stored as `(x, y, z, w)` with `w` the scalar part. Rotations compose with
//! `a * b` meaning "apply `b` then `a`", matching the matrix convention.

use crate::float::f32 as mf;
use crate::vec::Vec3;
use core::ops::{Mul, MulAssign, Neg};

/// A quaternion, normally kept unit-length to represent a rotation.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Quat {
    /// Imaginary X.
    pub x: f32,
    /// Imaginary Y.
    pub y: f32,
    /// Imaginary Z.
    pub z: f32,
    /// Scalar W.
    pub w: f32,
}

impl Default for Quat {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    /// The identity rotation.
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    /// Raw constructor from components.
    #[inline]
    pub const fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Build a rotation of `angle` radians about a unit `axis`.
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let (s, c) = mf::sin_cos(angle * 0.5);
        let a = axis * s;
        Self { x: a.x, y: a.y, z: a.z, w: c }
    }

    /// Rotation about the X axis.
    #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (s, c) = mf::sin_cos(angle * 0.5);
        Self { x: s, y: 0.0, z: 0.0, w: c }
    }
    /// Rotation about the Y axis.
    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (s, c) = mf::sin_cos(angle * 0.5);
        Self { x: 0.0, y: s, z: 0.0, w: c }
    }
    /// Rotation about the Z axis.
    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (s, c) = mf::sin_cos(angle * 0.5);
        Self { x: 0.0, y: 0.0, z: s, w: c }
    }

    /// Dot product (treating the quaternion as a 4-vector).
    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
    /// Length.
    #[inline]
    pub fn length(self) -> f32 {
        mf::sqrt(self.dot(self))
    }
    /// Squared length.
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }
    /// Normalize to a unit quaternion.
    #[inline]
    pub fn normalize(self) -> Self {
        let inv = 1.0 / self.length();
        Self { x: self.x * inv, y: self.y * inv, z: self.z * inv, w: self.w * inv }
    }
    /// Conjugate (inverse for a unit quaternion).
    #[inline]
    pub fn conjugate(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z, w: self.w }
    }
    /// Inverse. For unit quaternions this equals [`Quat::conjugate`].
    #[inline]
    pub fn inverse(self) -> Self {
        self.conjugate()
    }
    /// True if all components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
    /// True if close to another rotation within `max_abs_diff` per component,
    /// accounting for double-cover (`q` and `-q` are the same rotation).
    #[inline]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        let d = self.dot(rhs);
        // Pick the closer representative of the double cover.
        let rhs = if d < 0.0 { -rhs } else { rhs };
        mf::abs(self.x - rhs.x) <= max_abs_diff
            && mf::abs(self.y - rhs.y) <= max_abs_diff
            && mf::abs(self.z - rhs.z) <= max_abs_diff
            && mf::abs(self.w - rhs.w) <= max_abs_diff
    }

    /// Rotate a vector by this quaternion.
    #[inline]
    pub fn mul_vec3(self, v: Vec3) -> Vec3 {
        // v + 2w(q x v) + 2 q x (q x v)
        let u = Vec3::new(self.x, self.y, self.z);
        let t = u.cross(v) * 2.0;
        v + t * self.w + u.cross(t)
    }

    /// Normalized linear interpolation (cheap, approximate).
    #[inline]
    pub fn nlerp(self, mut rhs: Self, t: f32) -> Self {
        if self.dot(rhs) < 0.0 {
            rhs = -rhs;
        }
        Self {
            x: self.x + (rhs.x - self.x) * t,
            y: self.y + (rhs.y - self.y) * t,
            z: self.z + (rhs.z - self.z) * t,
            w: self.w + (rhs.w - self.w) * t,
        }
        .normalize()
    }

    /// Spherical linear interpolation along the shortest arc.
    #[inline]
    pub fn slerp(self, mut rhs: Self, t: f32) -> Self {
        let mut dot = self.dot(rhs);
        if dot < 0.0 {
            rhs = -rhs;
            dot = -dot;
        }
        const DOT_THRESHOLD: f32 = 0.9995;
        if dot > DOT_THRESHOLD {
            // Nearly colinear: fall back to normalized lerp to avoid div-by-zero.
            return self.nlerp(rhs, t);
        }
        let theta = mf::acos(dot.clamp(-1.0, 1.0));
        let sin_theta = mf::sin(theta);
        let s0 = mf::sin((1.0 - t) * theta) / sin_theta;
        let s1 = mf::sin(t * theta) / sin_theta;
        Self {
            x: self.x * s0 + rhs.x * s1,
            y: self.y * s0 + rhs.y * s1,
            z: self.z * s0 + rhs.z * s1,
            w: self.w * s0 + rhs.w * s1,
        }
    }

    /// As an array `[x, y, z, w]`.
    #[inline]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
}

impl Mul for Quat {
    type Output = Quat;
    /// Hamilton product. `a * b` = apply `b` then `a`.
    #[inline]
    fn mul(self, r: Quat) -> Quat {
        Quat {
            w: self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            x: self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            y: self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            z: self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        }
    }
}
impl MulAssign for Quat {
    #[inline]
    fn mul_assign(&mut self, r: Quat) {
        *self = *self * r;
    }
}
impl Mul<Vec3> for Quat {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        self.mul_vec3(v)
    }
}
impl Neg for Quat {
    type Output = Quat;
    #[inline]
    fn neg(self) -> Quat {
        Quat { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
    }
}

impl core::fmt::Debug for Quat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Quat({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl Quat {
    /// Build a quaternion from a pure-rotation 3x3 matrix (Shepperd's method).
    #[inline]
    pub fn from_mat3(m: crate::mat::Mat3) -> Self {
        // Column-major accessors: m{row}{col}.
        let m00 = m.x_axis.x;
        let m01 = m.y_axis.x;
        let m02 = m.z_axis.x;
        let m10 = m.x_axis.y;
        let m11 = m.y_axis.y;
        let m12 = m.z_axis.y;
        let m20 = m.x_axis.z;
        let m21 = m.y_axis.z;
        let m22 = m.z_axis.z;

        let trace = m00 + m11 + m22;
        if trace > 0.0 {
            let s = mf::sqrt(trace + 1.0) * 2.0; // s = 4w
            Self {
                w: 0.25 * s,
                x: (m21 - m12) / s,
                y: (m02 - m20) / s,
                z: (m10 - m01) / s,
            }
            .normalize()
        } else if m00 > m11 && m00 > m22 {
            let s = mf::sqrt(1.0 + m00 - m11 - m22) * 2.0; // s = 4x
            Self {
                w: (m21 - m12) / s,
                x: 0.25 * s,
                y: (m01 + m10) / s,
                z: (m02 + m20) / s,
            }
            .normalize()
        } else if m11 > m22 {
            let s = mf::sqrt(1.0 + m11 - m00 - m22) * 2.0; // s = 4y
            Self {
                w: (m02 - m20) / s,
                x: (m01 + m10) / s,
                y: 0.25 * s,
                z: (m12 + m21) / s,
            }
            .normalize()
        } else {
            let s = mf::sqrt(1.0 + m22 - m00 - m11) * 2.0; // s = 4z
            Self {
                w: (m10 - m01) / s,
                x: (m02 + m20) / s,
                y: (m12 + m21) / s,
                z: 0.25 * s,
            }
            .normalize()
        }
    }
}
