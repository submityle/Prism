//! Affine-transform `TRS` decomposition and `3x3` polar decomposition
//! (design §16, §26).
//!
//! A particle emitter's world placement arrives as a column-major affine
//! `4x4` matrix, but the simulation and shading contracts consume it as a
//! *translate / rotate / scale* triple: three floats of translation, a unit
//! `quaternion` orientation, and three floats of per-axis scale. This module
//! owns the `CPU`-verifiable contract that converts between the two forms and,
//! for the general (possibly sheared) case, extracts the closest rotation via
//! polar decomposition.
//!
//! # Strict scope
//! This module is deliberately self-contained. It does **not** reuse the
//! `quaternion_rotate` `Quat` type nor the `orientation_basis` frame builder:
//! it defines its own [`Mat3`], [`Mat4`], and [`Trs`] contract types plus a
//! bare `[f32; 4]` `xyzw` `quaternion`, so the decomposition `ABI` stays
//! independent of those sibling modules. All linear algebra here is hand
//! written against these local types.
//!
//! # No transcendental math
//! Every routine is polynomial plus at most one `sqrt` per branch, or a fixed
//! count of divide/multiply/add iterations:
//! * [`quat_from_mat3`] uses the trace / `Shepperd` method: one `sqrt` on the
//!   branch with the largest pivot.
//! * [`polar_decompose`] runs the averaging Newton iteration
//!   `R_next = 0.5 * (R + inverse_transpose(R))` for a fixed
//!   [`POLAR_ITERATIONS`] steps, using only the adjugate `3x3` inverse (pure
//!   arithmetic).
//! * The `3x3` inverse is computed from cofactors (`Cramer`'s rule), never a
//!   transcendental.
//!
//! Degenerate inputs never produce a `NaN`: a near-zero basis column falls
//! back to the corresponding identity axis, and a singular matrix's inverse
//! falls back to the identity.

/// Epsilon for the few ordering / magnitude guards; production code never
/// compares `f32` values with `==` or `!=`.
const CMP_EPS: f32 = 1.0e-6;

/// Fixed number of averaging-Newton steps [`polar_decompose`] runs. The
/// iteration converges quadratically once near the orthogonal factor, so this
/// count is comfortably beyond `f32` precision for well-conditioned inputs.
pub const POLAR_ITERATIONS: usize = 24;

/// Serialized `std430` byte size of a [`Trs`]: a translation `vec4`, a rotation
/// `vec4`, and a scale `vec4` (three `vec4` slots).
pub const MATRIX_DECOMPOSE_STD430_SIZE: usize = 48;

/// A column-major `3x3` matrix. `cols[c]` is the `c`-th column; element row `r`
/// column `c` is `cols[c][r]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    /// The three columns, each a `3`-vector, in column-major order.
    pub cols: [[f32; 3]; 3],
}

impl Mat3 {
    /// The identity `3x3` matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            cols: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Builds a matrix directly from its three columns.
    #[must_use]
    pub const fn from_cols(c0: [f32; 3], c1: [f32; 3], c2: [f32; 3]) -> Self {
        Self { cols: [c0, c1, c2] }
    }

    /// The matrix-times-vector product `self * v`.
    #[must_use]
    pub fn mul_vec3(&self, v: [f32; 3]) -> [f32; 3] {
        [
            self.cols[0][0] * v[0] + self.cols[1][0] * v[1] + self.cols[2][0] * v[2],
            self.cols[0][1] * v[0] + self.cols[1][1] * v[1] + self.cols[2][1] * v[2],
            self.cols[0][2] * v[0] + self.cols[1][2] * v[1] + self.cols[2][2] * v[2],
        ]
    }

    /// The matrix-times-matrix product `self * rhs`. Column `k` of the result
    /// is `self` applied to column `k` of `rhs`.
    #[must_use]
    pub fn mul_mat3(&self, rhs: &Mat3) -> Mat3 {
        Mat3 {
            cols: [
                self.mul_vec3(rhs.cols[0]),
                self.mul_vec3(rhs.cols[1]),
                self.mul_vec3(rhs.cols[2]),
            ],
        }
    }

    /// The transpose: rows become columns.
    #[must_use]
    pub fn transpose(&self) -> Mat3 {
        Mat3 {
            cols: [
                [self.cols[0][0], self.cols[1][0], self.cols[2][0]],
                [self.cols[0][1], self.cols[1][1], self.cols[2][1]],
                [self.cols[0][2], self.cols[1][2], self.cols[2][2]],
            ],
        }
    }

    /// The scalar determinant (Sarrus / cofactor expansion along the first
    /// row).
    #[must_use]
    pub fn determinant(&self) -> f32 {
        let m00 = self.cols[0][0];
        let m01 = self.cols[1][0];
        let m02 = self.cols[2][0];
        let m10 = self.cols[0][1];
        let m11 = self.cols[1][1];
        let m12 = self.cols[2][1];
        let m20 = self.cols[0][2];
        let m21 = self.cols[1][2];
        let m22 = self.cols[2][2];
        m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20)
    }
}

/// A column-major affine `4x4` matrix. `cols[3]` holds the translation; the
/// upper-left `3x3` is the rotation-scale-shear block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// The four columns, each a `4`-vector, in column-major order.
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    /// The identity `4x4` matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            cols: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Transforms an affine point `p` (implicit `w = 1`), applying the
    /// upper-left `3x3` block and then the translation column, and returns the
    /// resulting `3`-vector.
    #[must_use]
    pub fn mul_point(&self, p: [f32; 3]) -> [f32; 3] {
        [
            self.cols[0][0] * p[0]
                + self.cols[1][0] * p[1]
                + self.cols[2][0] * p[2]
                + self.cols[3][0],
            self.cols[0][1] * p[0]
                + self.cols[1][1] * p[1]
                + self.cols[2][1] * p[2]
                + self.cols[3][1],
            self.cols[0][2] * p[0]
                + self.cols[1][2] * p[1]
                + self.cols[2][2] * p[2]
                + self.cols[3][2],
        ]
    }
}

/// A decomposed affine transform: translation, a unit `quaternion` rotation
/// (`xyzw`), and per-axis scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trs {
    /// World-space translation.
    pub translation: [f32; 3],
    /// Unit `quaternion` orientation stored as `[x, y, z, w]`.
    pub rotation: [f32; 4],
    /// Per-axis scale; one component is negative when the source matrix
    /// encodes a reflection (negative determinant).
    pub scale: [f32; 3],
}

impl Trs {
    /// The identity transform: no translation, identity rotation, unit scale.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }

    /// Serializes this transform into its `std430` byte layout: translation
    /// `vec4`, rotation `vec4`, then scale `vec4`, little-endian, with the
    /// two `vec3` slots padded to `vec4` boundaries.
    #[must_use]
    pub fn to_std430(&self) -> [u8; MATRIX_DECOMPOSE_STD430_SIZE] {
        let mut bytes = [0u8; MATRIX_DECOMPOSE_STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.translation[0].to_le_bytes());
        bytes[4..8].copy_from_slice(&self.translation[1].to_le_bytes());
        bytes[8..12].copy_from_slice(&self.translation[2].to_le_bytes());
        bytes[16..20].copy_from_slice(&self.rotation[0].to_le_bytes());
        bytes[20..24].copy_from_slice(&self.rotation[1].to_le_bytes());
        bytes[24..28].copy_from_slice(&self.rotation[2].to_le_bytes());
        bytes[28..32].copy_from_slice(&self.rotation[3].to_le_bytes());
        bytes[32..36].copy_from_slice(&self.scale[0].to_le_bytes());
        bytes[36..40].copy_from_slice(&self.scale[1].to_le_bytes());
        bytes[40..44].copy_from_slice(&self.scale[2].to_le_bytes());
        bytes
    }
}

/// The squared length of a `3`-vector.
fn length_squared3(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// The Euclidean length of a `3`-vector.
fn length3(v: [f32; 3]) -> f32 {
    length_squared3(v).sqrt()
}

/// Divides a `3`-vector by `len`, falling back to `axis` when `len` is
/// degenerate so the result stays a valid unit vector.
fn normalized_or_axis(v: [f32; 3], len: f32, axis: [f32; 3]) -> [f32; 3] {
    if len.abs() < CMP_EPS {
        axis
    } else {
        let inv = 1.0 / len;
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }
}

/// Builds a column-major rotation [`Mat3`] from a unit `quaternion` `[x, y, z,
/// w]` using the standard polynomial identity (exact for a unit input).
#[must_use]
pub fn mat3_from_quat(q: [f32; 4]) -> Mat3 {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    Mat3 {
        cols: [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)],
            [2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)],
            [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)],
        ],
    }
}

/// Recovers a unit `quaternion` `[x, y, z, w]` from a column-major rotation
/// [`Mat3`] using the trace / `Shepperd` method: the branch with the largest
/// pivot is chosen so the `sqrt` argument stays away from zero. The result is
/// one of the two equivalent double-cover representations.
#[must_use]
pub fn quat_from_mat3(m: &Mat3) -> [f32; 4] {
    let m00 = m.cols[0][0];
    let m01 = m.cols[1][0];
    let m02 = m.cols[2][0];
    let m10 = m.cols[0][1];
    let m11 = m.cols[1][1];
    let m12 = m.cols[2][1];
    let m20 = m.cols[0][2];
    let m21 = m.cols[1][2];
    let m22 = m.cols[2][2];
    let trace = m00 + m11 + m22;
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let inv = 1.0 / s;
        [
            (m21 - m12) * inv,
            (m02 - m20) * inv,
            (m10 - m01) * inv,
            0.25 * s,
        ]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        let inv = 1.0 / s;
        [
            0.25 * s,
            (m01 + m10) * inv,
            (m02 + m20) * inv,
            (m21 - m12) * inv,
        ]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        let inv = 1.0 / s;
        [
            (m01 + m10) * inv,
            0.25 * s,
            (m12 + m21) * inv,
            (m02 - m20) * inv,
        ]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        let inv = 1.0 / s;
        [
            (m02 + m20) * inv,
            (m12 + m21) * inv,
            0.25 * s,
            (m10 - m01) * inv,
        ]
    }
}

/// The inverse-transpose of a `3x3` matrix, computed from cofactors
/// (`Cramer`'s rule). A near-singular matrix falls back to the identity so the
/// polar iteration can never divide by zero.
fn inverse_transpose(m: &Mat3) -> Mat3 {
    let m00 = m.cols[0][0];
    let m01 = m.cols[1][0];
    let m02 = m.cols[2][0];
    let m10 = m.cols[0][1];
    let m11 = m.cols[1][1];
    let m12 = m.cols[2][1];
    let m20 = m.cols[0][2];
    let m21 = m.cols[1][2];
    let m22 = m.cols[2][2];

    let c00 = m11 * m22 - m12 * m21;
    let c01 = -(m10 * m22 - m12 * m20);
    let c02 = m10 * m21 - m11 * m20;
    let c10 = -(m01 * m22 - m02 * m21);
    let c11 = m00 * m22 - m02 * m20;
    let c12 = -(m00 * m21 - m01 * m20);
    let c20 = m01 * m12 - m02 * m11;
    let c21 = -(m00 * m12 - m02 * m10);
    let c22 = m00 * m11 - m01 * m10;

    let det = m00 * c00 + m01 * c01 + m02 * c02;
    if det.abs() < CMP_EPS {
        return Mat3::identity();
    }
    let inv = 1.0 / det;
    // The inverse-transpose equals the cofactor matrix divided by the
    // determinant; stored column-major, `cols[c][r]` holds cofactor `(r, c)`.
    Mat3 {
        cols: [
            [c00 * inv, c10 * inv, c20 * inv],
            [c01 * inv, c11 * inv, c21 * inv],
            [c02 * inv, c12 * inv, c22 * inv],
        ],
    }
}

/// Polar decomposition of a `3x3` matrix `m` into `(R, S)` with `m = R * S`,
/// where `R` is orthogonal (the closest rotation, or an improper rotation when
/// `m` reflects) and `S` is symmetric.
///
/// `R` is found by the averaging-Newton iteration
/// `R_next = 0.5 * (R + inverse_transpose(R))`, which drives the matrix toward
/// the nearest orthogonal one; `S = transpose(R) * m` is then symmetric
/// positive definite for a positive-determinant input. The iteration runs a
/// fixed [`POLAR_ITERATIONS`] steps and uses only multiply / add / divide.
#[must_use]
pub fn polar_decompose(m: &Mat3) -> (Mat3, Mat3) {
    let mut r = *m;
    for _ in 0..POLAR_ITERATIONS {
        let it = inverse_transpose(&r);
        r = Mat3 {
            cols: [
                [
                    0.5 * (r.cols[0][0] + it.cols[0][0]),
                    0.5 * (r.cols[0][1] + it.cols[0][1]),
                    0.5 * (r.cols[0][2] + it.cols[0][2]),
                ],
                [
                    0.5 * (r.cols[1][0] + it.cols[1][0]),
                    0.5 * (r.cols[1][1] + it.cols[1][1]),
                    0.5 * (r.cols[1][2] + it.cols[1][2]),
                ],
                [
                    0.5 * (r.cols[2][0] + it.cols[2][0]),
                    0.5 * (r.cols[2][1] + it.cols[2][1]),
                    0.5 * (r.cols[2][2] + it.cols[2][2]),
                ],
            ],
        };
    }
    let s = r.transpose().mul_mat3(m);
    (r, s)
}

/// Decomposes an affine [`Mat4`] into a [`Trs`]: the translation is the fourth
/// column, the scale is the length of each basis column (with one axis negated
/// to preserve a right-handed frame when the determinant is negative), and the
/// rotation is the `quaternion` of the de-scaled `3x3` block via the
/// `Shepperd` method.
#[must_use]
pub fn decompose_affine(m: &Mat4) -> Trs {
    let translation = [m.cols[3][0], m.cols[3][1], m.cols[3][2]];

    let c0 = [m.cols[0][0], m.cols[0][1], m.cols[0][2]];
    let c1 = [m.cols[1][0], m.cols[1][1], m.cols[1][2]];
    let c2 = [m.cols[2][0], m.cols[2][1], m.cols[2][2]];

    let mut sx = length3(c0);
    let sy = length3(c1);
    let sz = length3(c2);

    // A negative determinant is a reflection; fold it into the first axis so
    // the recovered rotation stays a proper (right-handed) rotation.
    let basis = Mat3::from_cols(c0, c1, c2);
    if basis.determinant() < 0.0 {
        sx = -sx;
    }

    let r = Mat3::from_cols(
        normalized_or_axis(c0, sx, [1.0, 0.0, 0.0]),
        normalized_or_axis(c1, sy, [0.0, 1.0, 0.0]),
        normalized_or_axis(c2, sz, [0.0, 0.0, 1.0]),
    );
    let rotation = quat_from_mat3(&r);

    Trs {
        translation,
        rotation,
        scale: [sx, sy, sz],
    }
}

/// Composes a [`Trs`] back into an affine [`Mat4`], the inverse of
/// [`decompose_affine`] (up to the `quaternion` double cover). The upper-left
/// block is `rotation * diag(scale)` and the fourth column is the translation.
#[must_use]
pub fn compose_trs(trs: &Trs) -> Mat4 {
    let r = mat3_from_quat(trs.rotation);
    let (sx, sy, sz) = (trs.scale[0], trs.scale[1], trs.scale[2]);
    Mat4 {
        cols: [
            [r.cols[0][0] * sx, r.cols[0][1] * sx, r.cols[0][2] * sx, 0.0],
            [r.cols[1][0] * sy, r.cols[1][1] * sy, r.cols[1][2] * sy, 0.0],
            [r.cols[2][0] * sz, r.cols[2][1] * sz, r.cols[2][2] * sz, 0.0],
            [
                trs.translation[0],
                trs.translation[1],
                trs.translation[2],
                1.0,
            ],
        ],
    }
}

/// Total `GPU` storage size in bytes for `count` serialized [`Trs`] records,
/// clamped up to a single element so an empty batch still yields a valid
/// storage binding.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    crate::particle::gpu_layout::storage_bytes(MATRIX_DECOMPOSE_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

    const TEST_EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPS
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    fn approx_mat3(a: &Mat3, b: &Mat3) -> bool {
        (0..3).all(|c| approx3(a.cols[c], b.cols[c]))
    }

    fn approx_mat4(a: &Mat4, b: &Mat4) -> bool {
        (0..4).all(|c| (0..4).all(|r| approx(a.cols[c][r], b.cols[c][r])))
    }

    /// Equivalent up to the `quaternion` double cover (`q` and `-q` rotate
    /// identically).
    fn quat_equiv(a: [f32; 4], b: [f32; 4]) -> bool {
        let same =
            approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2]) && approx(a[3], b[3]);
        let flipped = approx(a[0], -b[0])
            && approx(a[1], -b[1])
            && approx(a[2], -b[2])
            && approx(a[3], -b[3]);
        same || flipped
    }

    /// A normalized `quaternion` for a rotation of `2 * atan2(s, c)` about a
    /// unit axis, built from a caller-supplied half-angle sine/cosine pair so
    /// no trigonometry runs in the test.
    fn quat_axis(axis: [f32; 3], sin_half: f32, cos_half: f32) -> [f32; 4] {
        [
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            cos_half,
        ]
    }

    #[test]
    fn mat3_identity_is_the_identity() {
        let i = Mat3::identity();
        assert!(approx3(i.mul_vec3([3.0, -2.0, 5.0]), [3.0, -2.0, 5.0]));
    }

    #[test]
    fn mat3_mul_vec3_uses_columns() {
        let m = Mat3::from_cols([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]);
        assert!(approx3(m.mul_vec3([1.0, 1.0, 1.0]), [2.0, 3.0, 4.0]));
    }

    #[test]
    fn mat3_mul_mat3_identity_is_neutral() {
        let m = Mat3::from_cols([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]);
        assert!(approx_mat3(&m.mul_mat3(&Mat3::identity()), &m));
        assert!(approx_mat3(&Mat3::identity().mul_mat3(&m), &m));
    }

    #[test]
    fn mat3_transpose_swaps_rows_and_columns() {
        let m = Mat3::from_cols([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
        let t = m.transpose();
        assert!(approx3(t.cols[0], [1.0, 4.0, 7.0]));
        assert!(approx3(t.cols[1], [2.0, 5.0, 8.0]));
        assert!(approx3(t.cols[2], [3.0, 6.0, 9.0]));
        assert!(approx_mat3(&t.transpose(), &m));
    }

    #[test]
    fn mat3_determinant_of_diagonal_is_product() {
        let m = Mat3::from_cols([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]);
        assert!(approx(m.determinant(), 24.0));
        assert!(approx(Mat3::identity().determinant(), 1.0));
    }

    #[test]
    fn mat3_determinant_sign_flips_under_reflection() {
        let m = Mat3::from_cols([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(m.determinant() < 0.0);
    }

    #[test]
    fn mat4_identity_leaves_points_unchanged() {
        let p = [4.0, -7.0, 2.0];
        assert!(approx3(Mat4::identity().mul_point(p), p));
    }

    #[test]
    fn mat4_mul_point_applies_translation() {
        let mut m = Mat4::identity();
        m.cols[3] = [10.0, 20.0, 30.0, 1.0];
        assert!(approx3(m.mul_point([1.0, 2.0, 3.0]), [11.0, 22.0, 33.0]));
    }

    #[test]
    fn decompose_identity_is_unit_trs() {
        let trs = decompose_affine(&Mat4::identity());
        assert!(approx3(trs.translation, [0.0, 0.0, 0.0]));
        assert!(approx3(trs.scale, [1.0, 1.0, 1.0]));
        assert!(quat_equiv(trs.rotation, [0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn decompose_pure_translation() {
        let mut m = Mat4::identity();
        m.cols[3] = [5.0, -3.0, 8.0, 1.0];
        let trs = decompose_affine(&m);
        assert!(approx3(trs.translation, [5.0, -3.0, 8.0]));
        assert!(approx3(trs.scale, [1.0, 1.0, 1.0]));
        assert!(quat_equiv(trs.rotation, [0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn decompose_pure_scale() {
        let trs = compose_trs(&Trs {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [2.0, 3.0, 4.0],
        });
        let back = decompose_affine(&trs);
        assert!(approx3(back.scale, [2.0, 3.0, 4.0]));
        assert!(quat_equiv(back.rotation, [0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn decompose_pure_rotation_recovers_quaternion() {
        // 90 degrees about +Z: sin_half = cos_half = sqrt(0.5).
        let h = 0.5_f32.sqrt();
        let q = quat_axis([0.0, 0.0, 1.0], h, h);
        let m = compose_trs(&Trs {
            translation: [0.0, 0.0, 0.0],
            rotation: q,
            scale: [1.0, 1.0, 1.0],
        });
        let trs = decompose_affine(&m);
        assert!(approx3(trs.scale, [1.0, 1.0, 1.0]));
        assert!(quat_equiv(trs.rotation, q));
    }

    #[test]
    fn compose_identity_trs_is_identity_matrix() {
        assert!(approx_mat4(
            &compose_trs(&Trs::identity()),
            &Mat4::identity()
        ));
    }

    #[test]
    fn trs_round_trip_matrix_is_identity() {
        let h = 0.5_f32.sqrt();
        let source = Trs {
            translation: [1.5, -2.0, 0.25],
            rotation: quat_axis([0.0, 1.0, 0.0], h, h),
            scale: [2.0, 0.5, 1.5],
        };
        let m = compose_trs(&source);
        let back = decompose_affine(&m);
        let m2 = compose_trs(&back);
        assert!(approx_mat4(&m, &m2));
    }

    #[test]
    fn trs_round_trip_preserves_fields() {
        let h = 0.5_f32.sqrt();
        let source = Trs {
            translation: [3.0, 4.0, 5.0],
            rotation: quat_axis([1.0, 0.0, 0.0], h, h),
            scale: [1.25, 2.0, 3.0],
        };
        let back = decompose_affine(&compose_trs(&source));
        assert!(approx3(back.translation, source.translation));
        assert!(approx3(back.scale, source.scale));
        assert!(quat_equiv(back.rotation, source.rotation));
    }

    #[test]
    fn decompose_negative_determinant_yields_one_negative_scale() {
        // Mirror the X axis: determinant is negative, so exactly one scale
        // component is negative and the recovered rotation stays proper.
        let mut m = Mat4::identity();
        m.cols[0] = [-1.0, 0.0, 0.0, 0.0];
        let trs = decompose_affine(&m);
        let negatives = trs.scale.iter().filter(|s| **s < 0.0).count();
        assert_eq!(negatives, 1);
        // Recomposing reproduces the reflection matrix.
        assert!(approx_mat4(&compose_trs(&trs), &m));
    }

    #[test]
    fn decompose_reflection_recomposes_matrix() {
        // Non-uniform scale plus reflection on Y.
        let mut m = Mat4::identity();
        m.cols[0] = [2.0, 0.0, 0.0, 0.0];
        m.cols[1] = [0.0, -3.0, 0.0, 0.0];
        m.cols[2] = [0.0, 0.0, 4.0, 0.0];
        let trs = decompose_affine(&m);
        assert!(approx_mat4(&compose_trs(&trs), &m));
    }

    #[test]
    fn polar_decompose_r_is_orthogonal() {
        let m = Mat3::from_cols([2.0, 0.3, 0.0], [0.1, 3.0, 0.2], [0.0, 0.4, 1.5]);
        let (r, _s) = polar_decompose(&m);
        let rrt = r.mul_mat3(&r.transpose());
        assert!(approx_mat3(&rrt, &Mat3::identity()));
    }

    #[test]
    fn polar_decompose_s_is_symmetric() {
        let m = Mat3::from_cols([2.0, 0.3, 0.1], [0.1, 3.0, 0.2], [0.05, 0.4, 1.5]);
        let (_r, s) = polar_decompose(&m);
        assert!(approx(s.cols[0][1], s.cols[1][0]));
        assert!(approx(s.cols[0][2], s.cols[2][0]));
        assert!(approx(s.cols[1][2], s.cols[2][1]));
    }

    #[test]
    fn polar_decompose_reconstructs_input() {
        let m = Mat3::from_cols([2.0, 0.3, 0.1], [0.1, 3.0, 0.2], [0.05, 0.4, 1.5]);
        let (r, s) = polar_decompose(&m);
        assert!(approx_mat3(&r.mul_mat3(&s), &m));
    }

    #[test]
    fn polar_decompose_of_orthogonal_is_identity_stretch() {
        let h = 0.5_f32.sqrt();
        let rot = mat3_from_quat(quat_axis([0.0, 0.0, 1.0], h, h));
        let (r, s) = polar_decompose(&rot);
        assert!(approx_mat3(&r, &rot));
        assert!(approx_mat3(&s, &Mat3::identity()));
    }

    #[test]
    fn polar_decompose_of_pure_scale_is_identity_rotation() {
        let scale = Mat3::from_cols([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]);
        let (r, s) = polar_decompose(&scale);
        assert!(approx_mat3(&r, &Mat3::identity()));
        assert!(approx_mat3(&s, &scale));
    }

    #[test]
    fn quat_from_identity_matrix_is_identity() {
        assert!(quat_equiv(
            quat_from_mat3(&Mat3::identity()),
            [0.0, 0.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn mat3_from_identity_quat_is_identity() {
        assert!(approx_mat3(
            &mat3_from_quat([0.0, 0.0, 0.0, 1.0]),
            &Mat3::identity()
        ));
    }

    #[test]
    fn quat_mat3_round_trip() {
        let h = 0.5_f32.sqrt();
        let axis = normalized_or_axis([1.0, 2.0, 2.0], length3([1.0, 2.0, 2.0]), [1.0, 0.0, 0.0]);
        let q = quat_axis(axis, h, h);
        let recovered = quat_from_mat3(&mat3_from_quat(q));
        assert!(quat_equiv(recovered, q));
    }

    #[test]
    fn mat3_from_quat_rotates_like_expected() {
        // 90 degrees about +Z sends +X to +Y.
        let h = 0.5_f32.sqrt();
        let r = mat3_from_quat(quat_axis([0.0, 0.0, 1.0], h, h));
        assert!(approx3(r.mul_vec3([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
    }

    #[test]
    fn std430_size_is_three_vec4_slots() {
        assert_eq!(MATRIX_DECOMPOSE_STD430_SIZE, 3 * VEC4_STRIDE);
        assert_eq!(MATRIX_DECOMPOSE_STD430_SIZE % VEC4_STRIDE, 0);
        assert_eq!(MATRIX_DECOMPOSE_STD430_SIZE, storage_bytes(VEC4_STRIDE, 3));
    }

    #[test]
    fn std430_round_trips_the_fields() {
        let trs = Trs {
            translation: [1.0, 2.0, 3.0],
            rotation: [0.1, 0.2, 0.3, 0.9],
            scale: [4.0, 5.0, 6.0],
        };
        let bytes = trs.to_std430();
        assert_eq!(bytes.len(), MATRIX_DECOMPOSE_STD430_SIZE);
        let read = |i: usize| {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[i..i + 4]);
            f32::from_le_bytes(b)
        };
        assert!(approx(read(0), 1.0));
        assert!(approx(read(4), 2.0));
        assert!(approx(read(8), 3.0));
        assert!(approx(read(12), 0.0));
        assert!(approx(read(16), 0.1));
        assert!(approx(read(20), 0.2));
        assert!(approx(read(24), 0.3));
        assert!(approx(read(28), 0.9));
        assert!(approx(read(32), 4.0));
        assert!(approx(read(36), 5.0));
        assert!(approx(read(40), 6.0));
        assert!(approx(read(44), 0.0));
    }

    #[test]
    fn gpu_storage_bytes_scales_and_reserves_one() {
        assert_eq!(gpu_storage_bytes(0), MATRIX_DECOMPOSE_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), MATRIX_DECOMPOSE_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(5), 5 * MATRIX_DECOMPOSE_STD430_SIZE);
    }
}
