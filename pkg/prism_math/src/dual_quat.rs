//! Dual quaternions for rigid (rotation + translation) transforms and skinning.
//!
//! A unit dual quaternion `q = real + e*dual` (where `e^2 = 0`) encodes a rigid
//! motion with the same algebra used for quaternions: the real part is a unit
//! rotation quaternion and the dual part carries the translation. Compared with
//! a matrix or a separate quaternion-plus-vector, dual quaternions blend
//! smoothly without the candy-wrapper collapse of linear-blend skinning, which
//! is why they are the standard choice for skeletal skinning.
//!
//! The unit invariants are `|real| = 1` and `dot(real, dual) = 0`.
//! [`DualQuat::from_rotation_translation`] produces a unit dual quaternion, and
//! [`DualQuat::normalize`] restores the invariants after accumulation.
//!
//! Two blends are provided: [`DualQuat::nlerp`] (dual-quaternion linear
//! blending, the cheap per-vertex skinning blend) and [`DualQuat::sclerp`]
//! (screw-linear interpolation, the constant-speed shortest screw motion).

use crate::float::f32 as mf;
use crate::quat::Quat;
use crate::vec::Vec3;
use core::ops::Mul;

/// Component-wise quaternion addition (quaternions are not kept unit here).
#[inline]
fn q_add(a: Quat, b: Quat) -> Quat {
    Quat::from_xyzw(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w)
}

/// Component-wise quaternion subtraction.
#[inline]
fn q_sub(a: Quat, b: Quat) -> Quat {
    Quat::from_xyzw(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w)
}

/// Scale every component of a quaternion.
#[inline]
fn q_scale(a: Quat, s: f32) -> Quat {
    Quat::from_xyzw(a.x * s, a.y * s, a.z * s, a.w * s)
}

/// A dual quaternion `real + e*dual`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct DualQuat {
    /// Real (rotation) part.
    pub real: Quat,
    /// Dual (translation-carrying) part.
    pub dual: Quat,
}

impl Default for DualQuat {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl DualQuat {
    /// The identity transform (identity rotation, zero translation).
    pub const IDENTITY: Self = Self {
        real: Quat::IDENTITY,
        dual: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
    };

    /// Build directly from real and dual parts (no normalization).
    #[inline]
    #[must_use]
    pub const fn from_real_dual(real: Quat, dual: Quat) -> Self {
        Self { real, dual }
    }

    /// Build a unit dual quaternion from a rotation and a translation.
    ///
    /// The convention is "rotate, then translate": the resulting transform maps
    /// `p` to `rotation * p + translation`.
    #[inline]
    #[must_use]
    pub fn from_rotation_translation(rotation: Quat, translation: Vec3) -> Self {
        let real = rotation;
        // dual = 0.5 * t_quat * real, with t_quat = (t, 0).
        let t = Quat::from_xyzw(translation.x, translation.y, translation.z, 0.0);
        let dual = q_scale(t * real, 0.5);
        Self { real, dual }
    }

    /// Build a pure-rotation dual quaternion.
    #[inline]
    #[must_use]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self::from_rotation_translation(rotation, Vec3::ZERO)
    }

    /// Build a pure-translation dual quaternion.
    #[inline]
    #[must_use]
    pub fn from_translation(translation: Vec3) -> Self {
        Self::from_rotation_translation(Quat::IDENTITY, translation)
    }

    /// Extract the rotation part.
    #[inline]
    #[must_use]
    pub fn rotation(self) -> Quat {
        self.real
    }

    /// Extract the translation part.
    #[inline]
    #[must_use]
    pub fn translation(self) -> Vec3 {
        // t = 2 * (dual * conjugate(real)), taking the vector part.
        let t = q_scale(self.dual * self.real.conjugate(), 2.0);
        Vec3::new(t.x, t.y, t.z)
    }

    /// Decompose into `(rotation, translation)`.
    #[inline]
    #[must_use]
    pub fn to_rotation_translation(self) -> (Quat, Vec3) {
        (self.rotation(), self.translation())
    }

    /// The dot product of the real parts, used to pick the closer hemisphere
    /// of the double cover when blending.
    #[inline]
    #[must_use]
    pub fn real_dot(self, rhs: Self) -> f32 {
        self.real.dot(rhs.real)
    }

    /// Dual-quaternion conjugate (quaternion-conjugate both parts). For a unit
    /// dual quaternion this is the inverse rigid transform.
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self {
            real: self.real.conjugate(),
            dual: self.dual.conjugate(),
        }
    }

    /// The inverse rigid transform (identical to [`DualQuat::conjugate`] for a
    /// unit dual quaternion).
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        self.conjugate()
    }

    /// Normalize to restore the unit invariants `|real| = 1` and
    /// `dot(real, dual) = 0`.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        let mag = self.real.length();
        let inv = 1.0 / mag;
        let real = q_scale(self.real, inv);
        let mut dual = q_scale(self.dual, inv);
        // Remove any residual component of `dual` along `real` so the two parts
        // stay orthogonal (keeps the transform a pure rigid motion).
        let d = real.dot(dual);
        dual = q_sub(dual, q_scale(real, d));
        Self { real, dual }
    }

    /// Negate both parts. `q` and `-q` represent the same rigid transform.
    #[inline]
    #[must_use]
    pub fn negated(self) -> Self {
        Self {
            real: q_scale(self.real, -1.0),
            dual: q_scale(self.dual, -1.0),
        }
    }

    /// Transform a point by this (assumed unit) dual quaternion.
    #[inline]
    #[must_use]
    pub fn transform_point3(self, p: Vec3) -> Vec3 {
        let (r, t) = self.to_rotation_translation();
        r.mul_vec3(p) + t
    }

    /// Transform a direction (rotation only; translation is ignored).
    #[inline]
    #[must_use]
    pub fn transform_vector3(self, v: Vec3) -> Vec3 {
        self.real.mul_vec3(v)
    }

    /// Dual-quaternion linear blend (`DLB`): the cheap skinning blend.
    ///
    /// Blends the real and dual parts linearly (after aligning hemispheres) and
    /// renormalizes. This is fast and avoids linear-blend-skinning collapse,
    /// but is not constant-speed; use [`DualQuat::sclerp`] when that matters.
    #[inline]
    #[must_use]
    pub fn nlerp(self, rhs: Self, t: f32) -> Self {
        let mut other = rhs;
        if self.real_dot(rhs) < 0.0 {
            other = rhs.negated();
        }
        let real = q_add(q_scale(self.real, 1.0 - t), q_scale(other.real, t));
        let dual = q_add(q_scale(self.dual, 1.0 - t), q_scale(other.dual, t));
        Self { real, dual }.normalize()
    }

    /// Weighted blend of many dual quaternions (`DLB` for skinning).
    ///
    /// Accumulates `sum(w_i * q_i)` with each `q_i` flipped into the hemisphere
    /// of the first non-zero-weight input, then normalizes. Returns identity if
    /// the inputs are empty or the weights sum to zero.
    #[inline]
    #[must_use]
    pub fn blend_weighted(items: &[(Self, f32)]) -> Self {
        let mut acc_real = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
        let mut acc_dual = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
        let mut pivot: Option<Quat> = None;
        let mut any = false;
        for &(q, w) in items {
            if w == 0.0 {
                continue;
            }
            let aligned = match pivot {
                Some(p) => {
                    if p.dot(q.real) < 0.0 {
                        q.negated()
                    } else {
                        q
                    }
                }
                None => {
                    pivot = Some(q.real);
                    q
                }
            };
            acc_real = q_add(acc_real, q_scale(aligned.real, w));
            acc_dual = q_add(acc_dual, q_scale(aligned.dual, w));
            any = true;
        }
        if !any {
            return Self::IDENTITY;
        }
        let mag = acc_real.length();
        if mag <= 0.0 {
            return Self::IDENTITY;
        }
        Self {
            real: acc_real,
            dual: acc_dual,
        }
        .normalize()
    }

    /// Screw-linear interpolation (`ScLERP`): the constant-speed shortest screw
    /// motion between two unit dual quaternions.
    ///
    /// Computed as `self * (self^-1 * rhs)^t`, which rotates about and
    /// translates along a single screw axis at uniform rate.
    #[inline]
    #[must_use]
    pub fn sclerp(self, rhs: Self, t: f32) -> Self {
        let mut other = rhs;
        if self.real_dot(rhs) < 0.0 {
            other = rhs.negated();
        }
        let diff = self.inverse() * other;
        self * diff.powf(t)
    }

    /// Raise a unit dual quaternion to a real power via its screw parameters.
    ///
    /// `q^t` scales the screw's rotation angle and translation pitch by `t`
    /// while keeping the same screw axis. Pure translations (no rotation) are
    /// handled by scaling the translation linearly.
    #[inline]
    #[must_use]
    pub fn powf(self, t: f32) -> Self {
        let vr = Vec3::new(self.real.x, self.real.y, self.real.z);
        let wr = self.real.w;
        let vd = Vec3::new(self.dual.x, self.dual.y, self.dual.z);
        let wd = self.dual.w;

        let len_sq = vr.dot(vr);
        // Near-zero rotation: pure translation. real stays identity (sign
        // matched to wr), translation scales linearly.
        if len_sq < 1.0e-12 {
            let sign = if wr < 0.0 { -1.0 } else { 1.0 };
            return Self {
                real: Quat::from_xyzw(0.0, 0.0, 0.0, sign),
                dual: q_scale(self.dual, t),
            };
        }

        let inv_r = 1.0 / mf::sqrt(len_sq);
        // Rotation angle and screw pitch (translation along the axis).
        let angle = 2.0 * mf::acos(wr.clamp(-1.0, 1.0));
        let pitch = -2.0 * wd * inv_r;
        // Screw axis and its moment.
        let direction = vr * inv_r;
        let moment = (vd - direction * (pitch * wr * 0.5)) * inv_r;

        // Scale the dual angle by the exponent.
        let angle = angle * t;
        let pitch = pitch * t;
        let (sin_half, cos_half) = mf::sin_cos(angle * 0.5);

        let real = Quat::from_xyzw(
            direction.x * sin_half,
            direction.y * sin_half,
            direction.z * sin_half,
            cos_half,
        );
        let dual_v = moment * sin_half + direction * (pitch * 0.5 * cos_half);
        let dual = Quat::from_xyzw(dual_v.x, dual_v.y, dual_v.z, -pitch * 0.5 * sin_half);
        Self { real, dual }
    }

    /// True if every component of both parts is finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.real.is_finite() && self.dual.is_finite()
    }
}

impl Mul for DualQuat {
    type Output = DualQuat;
    /// Compose two rigid transforms. `a * b` applies `b` first, then `a`.
    #[inline]
    fn mul(self, rhs: DualQuat) -> DualQuat {
        let real = self.real * rhs.real;
        let dual = q_add(self.real * rhs.dual, self.dual * rhs.real);
        DualQuat { real, dual }
    }
}
