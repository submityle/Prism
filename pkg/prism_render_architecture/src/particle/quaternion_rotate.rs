//! Unit-`quaternion` algebra for per-particle orientation (design §16).
//!
//! A particle's world orientation is stored as a `quaternion` rather than a
//! matrix or Euler triple: it is four floats, it composes with a single
//! `Hamilton` product, it never gimbal-locks, and it interpolates cheaply. This
//! module owns the `CPU`-verifiable algebra contract for that representation —
//! construction, composition, inversion, vector rotation, matrix conversion,
//! and blending.
//!
//! # Strict scope
//! This module is *pure* `quaternion` algebra. It deliberately does **not**
//! pick a facing mode or build a `right`/`up` billboard frame — that is the
//! orientation-basis module's job — and it imports no other particle module
//! for its vector math, which is hand-written here.
//!
//! # No transcendental math
//! Every routine is a polynomial plus at most one `sqrt`:
//! * Construction takes a caller-supplied half-angle cosine/sine pair
//!   (`cos_half`, `sin_half`) instead of computing `sin`/`cos` internally, so
//!   there is no trigonometry in [`Quat::from_components`].
//! * Blending uses `nlerp` (normalized linear interpolation) rather than
//!   `slerp`, so there is no `acos`/`sin` on the interpolation path — only
//!   multiply/add followed by one normalization `sqrt`.
//! * [`Quat::to_matrix`] and [`Quat::from_matrix`] are exact polynomial /
//!   `sqrt` identities (the trace / `Shepperd` method).
//!
//! Degenerate inputs never produce a `NaN`: normalization of a near-zero
//! `quaternion` falls back to the identity, and axis normalization of a
//! near-zero axis falls back to the pure-scalar rotation.

/// Length below which a `quaternion` or axis is treated as degenerate and its
/// normalization falls back to a stable value instead of dividing by a
/// near-zero magnitude.
const MIN_LENGTH: f32 = 1.0e-6;

/// A rotation stored as the `quaternion` `x*i + y*j + z*k + w`, with `w` the
/// scalar part. Unit quaternions represent rotations; the algebra here also
/// works for non-unit quaternions where noted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    /// Coefficient of the `i` basis (rotation-axis `x`, scaled by `sin_half`).
    pub x: f32,
    /// Coefficient of the `j` basis (rotation-axis `y`, scaled by `sin_half`).
    pub y: f32,
    /// Coefficient of the `k` basis (rotation-axis `z`, scaled by `sin_half`).
    pub z: f32,
    /// Scalar part (the half-angle cosine `cos_half` for a unit rotation).
    pub w: f32,
}

impl Quat {
    /// The identity rotation `(0, 0, 0, 1)`: rotating by it is a no-op.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    /// Builds a `quaternion` directly from its four components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// The `Hamilton` product `self * other` composing two rotations (apply
    /// `other` first, then `self`). Named `hamilton` rather than `mul` because
    /// `quaternion` multiplication is non-commutative and is not the
    /// component-wise product a `Mul` impl would suggest.
    #[must_use]
    pub fn hamilton(&self, other: &Quat) -> Quat {
        let (x1, y1, z1, w1) = (self.x, self.y, self.z, self.w);
        let (x2, y2, z2, w2) = (other.x, other.y, other.z, other.w);
        Quat {
            x: w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            y: w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            z: w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w: w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        }
    }

    /// The conjugate `(-x, -y, -z, w)`. For a unit `quaternion` this is the
    /// inverse rotation.
    #[must_use]
    pub fn conjugate(&self) -> Quat {
        Quat {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// The multiplicative inverse `conjugate / length_sq`, valid for any
    /// non-degenerate `quaternion`. A near-zero `quaternion` has no inverse, so
    /// the identity is returned as a stable fallback.
    #[must_use]
    pub fn inverse(&self) -> Quat {
        let len_sq = self.length_sq();
        if len_sq < MIN_LENGTH {
            return Quat::identity();
        }
        let inv = 1.0 / len_sq;
        Quat {
            x: -self.x * inv,
            y: -self.y * inv,
            z: -self.z * inv,
            w: self.w * inv,
        }
    }

    /// The four-component dot product, equal to `cos` of the angle between the
    /// two quaternions on the unit hypersphere.
    #[must_use]
    pub fn dot(&self, other: &Quat) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// The squared magnitude `x*x + y*y + z*z + w*w` (no `sqrt`).
    #[must_use]
    pub fn length_sq(&self) -> f32 {
        self.dot(self)
    }

    /// The Euclidean magnitude (`length_sq` under one `sqrt`).
    #[must_use]
    pub fn length(&self) -> f32 {
        self.length_sq().sqrt()
    }

    /// Returns the unit `quaternion` in the same direction. A near-zero
    /// `quaternion` (whose direction is undefined) clamps to the identity so
    /// the result is never a `NaN`.
    #[must_use]
    pub fn normalize(&self) -> Quat {
        let len = self.length();
        if len < MIN_LENGTH {
            return Quat::identity();
        }
        let inv = 1.0 / len;
        Quat {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
            w: self.w * inv,
        }
    }

    /// Rotates `v` by this `quaternion`, computing `q * v * q^-1` via the
    /// conjugate (assuming a unit `quaternion`). Implemented with the
    /// multiply/add identity `v + 2*w*(u x v) + 2*(u x (u x v))` where `u` is
    /// the vector part, so it preserves length for a unit `quaternion`.
    #[must_use]
    pub fn rotate_vec3(&self, v: [f32; 3]) -> [f32; 3] {
        let u = [self.x, self.y, self.z];
        let t = v3_scale(v3_cross(u, v), 2.0);
        v3_add(v3_add(v, v3_scale(t, self.w)), v3_cross(u, t))
    }

    /// Builds a rotation from a caller-supplied half-angle cosine/sine pair and
    /// a rotation `axis`. The `axis` is normalized here; the trigonometry
    /// (`cos_half`, `sin_half`) is the caller's responsibility so this module
    /// stays free of transcendental functions. A near-zero `axis` yields the
    /// pure-scalar `quaternion` `(0, 0, 0, cos_half)`.
    #[must_use]
    pub fn from_components(cos_half: f32, sin_half: f32, axis: [f32; 3]) -> Quat {
        let len = v3_length(axis);
        if len < MIN_LENGTH {
            return Quat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: cos_half,
            };
        }
        let n = v3_scale(axis, 1.0 / len);
        Quat {
            x: n[0] * sin_half,
            y: n[1] * sin_half,
            z: n[2] * sin_half,
            w: cos_half,
        }
    }

    /// Converts this `quaternion` to a row-major 3x3 rotation matrix using the
    /// standard polynomial identity (exact for a unit `quaternion`).
    #[must_use]
    pub fn to_matrix(&self) -> [[f32; 3]; 3] {
        let (x, y, z, w) = (self.x, self.y, self.z, self.w);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    }

    /// Recovers a `quaternion` from a row-major 3x3 rotation matrix using the
    /// trace / `Shepperd` method: the branch with the largest pivot is chosen
    /// so the `sqrt` argument stays well away from zero. The sign of the result
    /// is one of the two equivalent double-cover representations.
    #[must_use]
    pub fn from_matrix(m: [[f32; 3]; 3]) -> Quat {
        let (m00, m11, m22) = (m[0][0], m[1][1], m[2][2]);
        let trace = m00 + m11 + m22;
        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            let inv = 1.0 / s;
            Quat {
                x: (m[2][1] - m[1][2]) * inv,
                y: (m[0][2] - m[2][0]) * inv,
                z: (m[1][0] - m[0][1]) * inv,
                w: 0.25 * s,
            }
        } else if m00 > m11 && m00 > m22 {
            let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
            let inv = 1.0 / s;
            Quat {
                x: 0.25 * s,
                y: (m[0][1] + m[1][0]) * inv,
                z: (m[0][2] + m[2][0]) * inv,
                w: (m[2][1] - m[1][2]) * inv,
            }
        } else if m11 > m22 {
            let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
            let inv = 1.0 / s;
            Quat {
                x: (m[0][1] + m[1][0]) * inv,
                y: 0.25 * s,
                z: (m[1][2] + m[2][1]) * inv,
                w: (m[0][2] - m[2][0]) * inv,
            }
        } else {
            let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
            let inv = 1.0 / s;
            Quat {
                x: (m[0][2] + m[2][0]) * inv,
                y: (m[1][2] + m[2][1]) * inv,
                z: 0.25 * s,
                w: (m[1][0] - m[0][1]) * inv,
            }
        }
    }

    /// Normalized linear interpolation between `a` and `b` at parameter `t`.
    /// When `a.dot(b)` is negative, `b` is negated first so the blend follows
    /// the shorter arc (the two double-cover representations are equivalent
    /// rotations). The linear blend is renormalized with a single `sqrt`, so
    /// there is no `slerp`-style `acos`/`sin`.
    #[must_use]
    pub fn nlerp(a: &Quat, b: &Quat, t: f32) -> Quat {
        let sign = if a.dot(b) < 0.0 { -1.0 } else { 1.0 };
        let blended = Quat {
            x: a.x + t * (sign * b.x - a.x),
            y: a.y + t * (sign * b.y - a.y),
            z: a.z + t * (sign * b.z - a.z),
            w: a.w + t * (sign * b.w - a.w),
        };
        blended.normalize()
    }
}

/// The three-component dot product of two vectors.
fn v3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The right-handed cross product `a x b`.
fn v3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Component-wise vector addition.
fn v3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales a vector by a scalar.
fn v3_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// The Euclidean length of a vector (one `sqrt`).
fn v3_length(a: [f32; 3]) -> f32 {
    v3_dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for floating-point comparisons (f32 avoids `==`).
    const CMP_EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn quat_approx(a: &Quat, b: &Quat) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z) && approx(a.w, b.w)
    }

    /// Compares two rotations up to the double-cover sign ambiguity.
    fn quat_approx_signed(a: &Quat, b: &Quat) -> bool {
        let neg = Quat::new(-b.x, -b.y, -b.z, -b.w);
        quat_approx(a, b) || quat_approx(a, &neg)
    }

    fn vec_approx(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    /// Half-angle cosine/sine for a 90-degree rotation (45-degree half-angle),
    /// i.e. `1/sqrt(2)`, taken from the core constant to satisfy Clippy.
    const HALF_90: f32 = core::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn identity_is_unit_and_neutral_layout() {
        let q = Quat::identity();
        assert!(approx(q.x, 0.0));
        assert!(approx(q.y, 0.0));
        assert!(approx(q.z, 0.0));
        assert!(approx(q.w, 1.0));
        assert!(approx(q.length(), 1.0));
    }

    #[test]
    fn hamilton_identity_is_neutral() {
        let q = Quat::new(0.1, 0.2, 0.3, 0.9).normalize();
        let id = Quat::identity();
        assert!(quat_approx(&q.hamilton(&id), &q));
        assert!(quat_approx(&id.hamilton(&q), &q));
    }

    #[test]
    fn hamilton_is_associative() {
        let a = Quat::new(0.2, -0.3, 0.5, 0.7).normalize();
        let b = Quat::new(-0.1, 0.4, 0.2, 0.8).normalize();
        let c = Quat::new(0.6, 0.1, -0.2, 0.5).normalize();
        let left = a.hamilton(&b).hamilton(&c);
        let right = a.hamilton(&b.hamilton(&c));
        assert!(quat_approx(&left, &right));
    }

    #[test]
    fn hamilton_is_not_commutative() {
        let a = Quat::from_components(HALF_90, HALF_90, [1.0, 0.0, 0.0]);
        let b = Quat::from_components(HALF_90, HALF_90, [0.0, 1.0, 0.0]);
        assert!(!quat_approx(&a.hamilton(&b), &b.hamilton(&a)));
    }

    #[test]
    fn conjugate_times_self_is_identity_for_unit() {
        let q = Quat::new(0.3, -0.4, 0.5, 0.6).normalize();
        let prod = q.hamilton(&q.conjugate());
        assert!(quat_approx(&prod, &Quat::identity()));
    }

    #[test]
    fn inverse_times_self_is_identity_general() {
        let q = Quat::new(1.0, 2.0, 3.0, 4.0);
        let prod = q.hamilton(&q.inverse());
        assert!(quat_approx(&prod, &Quat::identity()));
        let prod2 = q.inverse().hamilton(&q);
        assert!(quat_approx(&prod2, &Quat::identity()));
    }

    #[test]
    fn inverse_of_degenerate_is_identity() {
        let q = Quat::new(0.0, 0.0, 0.0, 0.0);
        assert!(quat_approx(&q.inverse(), &Quat::identity()));
    }

    #[test]
    fn dot_matches_manual_sum() {
        let a = Quat::new(1.0, 2.0, 3.0, 4.0);
        let b = Quat::new(5.0, 6.0, 7.0, 8.0);
        assert!(approx(a.dot(&b), 5.0 + 12.0 + 21.0 + 32.0));
    }

    #[test]
    fn length_and_length_sq_agree() {
        let q = Quat::new(1.0, 2.0, 2.0, 0.0);
        assert!(approx(q.length_sq(), 9.0));
        assert!(approx(q.length(), 3.0));
    }

    #[test]
    fn normalize_gives_unit() {
        let q = Quat::new(3.0, 0.0, 4.0, 12.0).normalize();
        assert!(approx(q.length(), 1.0));
    }

    #[test]
    fn normalize_near_zero_clamps_to_identity() {
        let q = Quat::new(1.0e-9, 0.0, -1.0e-9, 0.0);
        assert!(quat_approx(&q.normalize(), &Quat::identity()));
    }

    #[test]
    fn rotate_vec3_preserves_length() {
        let q = Quat::new(0.2, 0.5, -0.3, 0.6).normalize();
        let v = [1.0, -2.0, 3.0];
        let r = q.rotate_vec3(v);
        let len_in = v3_length(v);
        let len_out = v3_length(r);
        assert!(approx(len_in, len_out));
    }

    #[test]
    fn rotate_identity_is_noop() {
        let v = [1.0, -2.0, 3.0];
        assert!(vec_approx(Quat::identity().rotate_vec3(v), v));
    }

    #[test]
    fn rotate_vec3_90_about_z() {
        let q = Quat::from_components(HALF_90, HALF_90, [0.0, 0.0, 1.0]);
        let r = q.rotate_vec3([1.0, 0.0, 0.0]);
        assert!(vec_approx(r, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn rotate_vec3_180_about_z() {
        // Half-angle of 180 degrees is 90 degrees: cos_half = 0, sin_half = 1.
        let q = Quat::from_components(0.0, 1.0, [0.0, 0.0, 1.0]);
        let r = q.rotate_vec3([1.0, 0.0, 0.0]);
        assert!(vec_approx(r, [-1.0, 0.0, 0.0]));
    }

    #[test]
    fn rotate_vec3_90_about_x_moves_y_to_z() {
        let q = Quat::from_components(HALF_90, HALF_90, [1.0, 0.0, 0.0]);
        let r = q.rotate_vec3([0.0, 1.0, 0.0]);
        assert!(vec_approx(r, [0.0, 0.0, 1.0]));
    }

    #[test]
    fn from_components_normalizes_axis() {
        let unnormalized = Quat::from_components(HALF_90, HALF_90, [0.0, 0.0, 5.0]);
        let normalized = Quat::from_components(HALF_90, HALF_90, [0.0, 0.0, 1.0]);
        assert!(quat_approx(&unnormalized, &normalized));
        assert!(approx(unnormalized.length(), 1.0));
    }

    #[test]
    fn from_components_zero_axis_is_pure_scalar() {
        let q = Quat::from_components(0.75, 0.5, [0.0, 0.0, 0.0]);
        assert!(quat_approx(&q, &Quat::new(0.0, 0.0, 0.0, 0.75)));
    }

    #[test]
    fn to_matrix_identity_is_identity_matrix() {
        let m = Quat::identity().to_matrix();
        let expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(m
            .iter()
            .zip(expected.iter())
            .all(|(row, exp)| vec_approx(*row, *exp)));
    }

    #[test]
    fn to_matrix_rows_are_orthonormal() {
        let q = Quat::new(0.2, -0.5, 0.4, 0.7).normalize();
        let m = q.to_matrix();
        assert!(approx(v3_length(m[0]), 1.0));
        assert!(approx(v3_length(m[1]), 1.0));
        assert!(approx(v3_length(m[2]), 1.0));
        assert!(approx(v3_dot(m[0], m[1]), 0.0));
        assert!(approx(v3_dot(m[0], m[2]), 0.0));
        assert!(approx(v3_dot(m[1], m[2]), 0.0));
    }

    #[test]
    fn to_matrix_matches_rotate_vec3() {
        let q = Quat::from_components(HALF_90, HALF_90, [0.0, 0.0, 1.0]);
        let m = q.to_matrix();
        // Matrix-times-column-vector on the basis vectors matches rotate_vec3.
        let ex = [m[0][0], m[1][0], m[2][0]];
        assert!(vec_approx(ex, q.rotate_vec3([1.0, 0.0, 0.0])));
    }

    #[test]
    fn from_matrix_identity_is_identity() {
        let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(quat_approx_signed(&Quat::from_matrix(m), &Quat::identity()));
    }

    #[test]
    fn to_from_matrix_roundtrip_trace_branch() {
        let q = Quat::new(0.1, 0.2, 0.15, 0.96).normalize();
        let back = Quat::from_matrix(q.to_matrix());
        assert!(quat_approx_signed(&back, &q));
    }

    #[test]
    fn to_from_matrix_roundtrip_x_branch() {
        // A 180-degree turn about X drives a negative trace, x-major pivot.
        let q = Quat::from_components(0.0, 1.0, [1.0, 0.0, 0.0]);
        let back = Quat::from_matrix(q.to_matrix());
        assert!(quat_approx_signed(&back, &q));
    }

    #[test]
    fn to_from_matrix_roundtrip_y_branch() {
        let q = Quat::from_components(0.0, 1.0, [0.0, 1.0, 0.0]);
        let back = Quat::from_matrix(q.to_matrix());
        assert!(quat_approx_signed(&back, &q));
    }

    #[test]
    fn to_from_matrix_roundtrip_z_branch() {
        let q = Quat::from_components(0.0, 1.0, [0.0, 0.0, 1.0]);
        let back = Quat::from_matrix(q.to_matrix());
        assert!(quat_approx_signed(&back, &q));
    }

    #[test]
    fn nlerp_endpoints_return_inputs() {
        let a = Quat::from_components(HALF_90, HALF_90, [0.0, 0.0, 1.0]);
        let b = Quat::from_components(HALF_90, HALF_90, [1.0, 0.0, 0.0]);
        assert!(quat_approx(&Quat::nlerp(&a, &b, 0.0), &a));
        assert!(quat_approx(&Quat::nlerp(&a, &b, 1.0), &b));
    }

    #[test]
    fn nlerp_result_is_unit() {
        let a = Quat::from_components(HALF_90, HALF_90, [0.0, 1.0, 0.0]);
        let b = Quat::from_components(HALF_90, HALF_90, [0.0, 0.0, 1.0]);
        let mid = Quat::nlerp(&a, &b, 0.5);
        assert!(approx(mid.length(), 1.0));
    }

    #[test]
    fn nlerp_takes_shortest_path() {
        let a = Quat::identity();
        // The antipode of the identity is the same rotation with dot < 0.
        let b = Quat::new(0.0, 0.0, 0.0, -1.0);
        let mid = Quat::nlerp(&a, &b, 0.5);
        // Shortest-path blend stays at the identity rather than collapsing.
        assert!(quat_approx_signed(&mid, &Quat::identity()));
    }

    #[test]
    fn nlerp_midpoint_is_between_inputs() {
        let a = Quat::identity();
        let b = Quat::from_components(HALF_90, HALF_90, [0.0, 0.0, 1.0]);
        let mid = Quat::nlerp(&a, &b, 0.5);
        assert!(approx(mid.length(), 1.0));
        // The midpoint's scalar part lies between the two endpoints' scalars.
        assert!(mid.w < a.w && mid.w > b.w);
    }
}
