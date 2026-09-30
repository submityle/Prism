//! `CPU` gold-standard for rotating already-projected spherical-harmonic (`SH`)
//! coefficients, for particle probe relighting and blending (design §16, §17).
//!
//! When a particle system carries baked `SH` irradiance (a dominant-light
//! direction plus ambient term) and the emitter or the scene rotates, the
//! stored coefficients must be re-expressed in the new frame *without*
//! re-projecting any radiance samples. This module owns that recombination and
//! nothing else. It is strictly disjoint from [`super::gi_probe`], which owns
//! `SH` basis evaluation, radiance projection, and irradiance convolution
//! (`sh_basis_l1` / `ShColorL1` and friends). This file never evaluates a basis
//! function, never projects a sample, and never reuses those probe types; it
//! only takes existing band-`L1` / band-`L2` coefficient vectors and applies a
//! rotation to them.
//!
//! Conventions (identical to [`super::gi_probe`] so the two agree, but the types
//! are independent):
//!
//! * band `L1` coefficients are ordered `(y, z, x)`, i.e.
//!   `(Y(1,-1), Y(1,0), Y(1,1))`.
//! * band `L2` coefficients are ordered
//!   `(Y(2,-2), Y(2,-1), Y(2,0), Y(2,1), Y(2,2))`.
//!
//! Rotation input is supplied by the caller, never derived here: either a 3x3
//! row-major rotation matrix, or the four components of a rotation quaternion
//! `(x, y, z, w)`. This module calls **no** trigonometric or transcendental
//! function; band-`L1` rotation is a permutation-plus-linear-combination of the
//! matrix rows, band-`L2` rotation is built from the band-`L1` rotation with the
//! `Ivanic`-`Ruedenberg` recurrence, and the quaternion-to-matrix conversion is
//! a pure polynomial. Only `f32` `sqrt` and rational arithmetic appear, so the
//! result is deterministic and matches a future `GPU` kernel bit for bit.
//!
//! `GPU` packing follows the shared `std430` `vec4` alignment from
//! [`super::gpu_layout`]: the band-`L0`/`L1` `RGB` payload lands in four `vec4`
//! slots (see [`ShL1Rgb::to_std430`] and [`SH_ROTATE_STD430_SIZE`]).

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Squared-length floor below which a hand-rolled band-`L1` direction is treated
/// as degenerate during [`nlerp_l1`], so the energy-preserving renormalization
/// never divides by zero or yields a `NaN`.
const MIN_LEN_SQ: f32 = 1e-12;

/// Byte size of one [`ShL1Rgb`] record in a `std430` storage buffer.
///
/// The band-`L0` `RGB` term and the three band-`L1` `RGB` coefficients each
/// occupy one padded `vec4` slot, so a record is four `vec4`s wide.
pub const SH_ROTATE_STD430_SIZE: usize = 4 * VEC4_STRIDE;

/// Builds a 3x3 row-major rotation matrix from the four components of a rotation
/// quaternion `(x, y, z, w)`.
///
/// The formula is the robust normalized form: it scales by `2 / |q|^2`, so a
/// non-unit quaternion still yields an orthonormal matrix, and the zero
/// quaternion returns the identity. Only multiplies, adds, and one division are
/// used — no `sqrt` and no trigonometry.
#[must_use]
pub fn rotation_matrix_from_quat(x: f32, y: f32, z: f32, w: f32) -> [[f32; 3]; 3] {
    let norm = x * x + y * y + z * z + w * w;
    let s = if norm > 0.0 { 2.0 / norm } else { 0.0 };
    let xx = x * x * s;
    let yy = y * y * s;
    let zz = z * z * s;
    let xy = x * y * s;
    let xz = x * z * s;
    let yz = y * z * s;
    let wx = w * x * s;
    let wy = w * y * s;
    let wz = w * z * s;
    [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]
}

/// Applies a 3x3 row-major matrix to a column vector.
fn apply_mat3(rot: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        rot[0][0] * v[0] + rot[0][1] * v[1] + rot[0][2] * v[2],
        rot[1][0] * v[0] + rot[1][1] * v[1] + rot[1][2] * v[2],
        rot[2][0] * v[0] + rot[2][1] * v[1] + rot[2][2] * v[2],
    ]
}

/// Rotates a single band-`L1` coefficient triple ordered `(y, z, x)` by a 3x3
/// row-major rotation matrix.
///
/// Band `L1` transforms exactly like a direction vector, so the coefficients are
/// shuffled from `SH` order `(y, z, x)` into Cartesian order `(x, y, z)`,
/// multiplied by the rotation, and shuffled back. The band-`L0` (`DC`) term is
/// separate and is not touched here.
#[must_use]
pub fn rotate_l1(coeffs: [f32; 3], rot: [[f32; 3]; 3]) -> [f32; 3] {
    // SH order (y, z, x) -> Cartesian (x, y, z).
    let cartesian = [coeffs[2], coeffs[0], coeffs[1]];
    let rotated = apply_mat3(&rot, cartesian);
    // Cartesian (x, y, z) -> SH order (y, z, x).
    [rotated[1], rotated[2], rotated[0]]
}

/// A per-channel colored band-`L0`/`L1` `SH` payload: one `DC` `RGB` term plus
/// three band-`L1` `RGB` coefficients.
///
/// `l1[k]` is the `RGB` triple of the `k`-th band-`L1` coefficient in `(y, z, x)`
/// order; each color channel is an independent `SH` vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShL1Rgb {
    /// The band-`L0` (`DC`) `RGB` term. Invariant under rotation.
    pub l0: [f32; 3],
    /// The three band-`L1` `RGB` coefficients in `(y, z, x)` order.
    pub l1: [[f32; 3]; 3],
}

impl ShL1Rgb {
    /// Rotates the band-`L1` coefficients of every color channel by `rot`,
    /// leaving the band-`L0` (`DC`) term unchanged.
    #[must_use]
    pub fn rotate(&self, rot: [[f32; 3]; 3]) -> Self {
        let r = rotate_l1([self.l1[0][0], self.l1[1][0], self.l1[2][0]], rot);
        let g = rotate_l1([self.l1[0][1], self.l1[1][1], self.l1[2][1]], rot);
        let b = rotate_l1([self.l1[0][2], self.l1[1][2], self.l1[2][2]], rot);
        let l1 = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
        Self { l0: self.l0, l1 }
    }

    /// Packs the record into its `std430` `vec4`-aligned word layout.
    ///
    /// The four `vec4` slots are `l0.rgb`, `l1[0].rgb`, `l1[1].rgb`,
    /// `l1[2].rgb`, each followed by a zero pad word, matching
    /// [`SH_ROTATE_STD430_SIZE`].
    #[must_use]
    pub fn to_std430(&self) -> [u32; 16] {
        [
            self.l0[0].to_bits(),
            self.l0[1].to_bits(),
            self.l0[2].to_bits(),
            0,
            self.l1[0][0].to_bits(),
            self.l1[0][1].to_bits(),
            self.l1[0][2].to_bits(),
            0,
            self.l1[1][0].to_bits(),
            self.l1[1][1].to_bits(),
            self.l1[1][2].to_bits(),
            0,
            self.l1[2][0].to_bits(),
            self.l1[2][1].to_bits(),
            self.l1[2][2].to_bits(),
            0,
        ]
    }
}

/// Reads the band-`L1` rotation matrix at signed indices in `-1..=1`.
///
/// Out-of-range indices return zero. The `Ivanic`-`Ruedenberg` `U` term samples
/// the band-`L1` matrix at `|index| = 2` when `m = +/-2`, but its scalar
/// coefficient is exactly zero there, so a zero sample keeps the product well
/// defined instead of indexing past the matrix.
fn band1(r1: &[[f32; 3]; 3], row: i32, col: i32) -> f32 {
    if !(-1..=1).contains(&row) || !(-1..=1).contains(&col) {
        return 0.0;
    }
    let r = usize::try_from(row + 1).unwrap_or_default();
    let c = usize::try_from(col + 1).unwrap_or_default();
    r1[r][c]
}

/// Converts a small integer to `f32` losslessly (values used here fit in `i16`).
fn small_f32(v: i32) -> f32 {
    f32::from(i16::try_from(v).unwrap_or_default())
}

/// The `Ivanic`-`Ruedenberg` `P` term for band `L2` (previous band is `L1`).
fn p2(r1: &[[f32; 3]; 3], i: i32, mu: i32, n: i32) -> f32 {
    if n == 2 {
        band1(r1, i, 1) * band1(r1, mu, 1) - band1(r1, i, -1) * band1(r1, mu, -1)
    } else if n == -2 {
        band1(r1, i, 1) * band1(r1, mu, -1) + band1(r1, i, -1) * band1(r1, mu, 1)
    } else {
        band1(r1, i, 0) * band1(r1, mu, n)
    }
}

/// The `Ivanic`-`Ruedenberg` `U` term for band `L2`.
fn u_term(r1: &[[f32; 3]; 3], m: i32, n: i32) -> f32 {
    p2(r1, 0, m, n)
}

/// The `Ivanic`-`Ruedenberg` `V` term for band `L2`.
fn v_term(r1: &[[f32; 3]; 3], m: i32, n: i32) -> f32 {
    if m > 0 {
        let s1 = if m == 1 {
            core::f32::consts::SQRT_2
        } else {
            1.0
        };
        let s2 = if m == 1 { 0.0 } else { 1.0 };
        p2(r1, 1, m - 1, n) * s1 - p2(r1, -1, -m + 1, n) * s2
    } else if m < 0 {
        let s1 = if m == -1 { 0.0 } else { 1.0 };
        let s2 = if m == -1 {
            core::f32::consts::SQRT_2
        } else {
            1.0
        };
        p2(r1, 1, m + 1, n) * s1 + p2(r1, -1, -m - 1, n) * s2
    } else {
        p2(r1, 1, 1, n) + p2(r1, -1, -1, n)
    }
}

/// The `Ivanic`-`Ruedenberg` `u` and `v` scalar coefficients for band `L2`.
///
/// The `w` coefficient is identically zero for `l = 2`, so it is omitted.
fn uv_coeff(m: i32, n: i32) -> (f32, f32) {
    let denom = if n == 2 || n == -2 {
        12
    } else {
        (2 + n) * (2 - n)
    };
    let denom_f = small_f32(denom);
    let u = (small_f32((2 + m) * (2 - m)) / denom_f).sqrt();
    let abs_m = m.abs();
    let delta0 = i32::from(m == 0);
    let v_num = small_f32((1 + delta0) * (1 + abs_m) * (2 + abs_m));
    let v_sign = if m == 0 { -1.0 } else { 1.0 };
    let v = 0.5 * (v_num / denom_f).sqrt() * v_sign;
    (u, v)
}

/// Builds the 5x5 band-`L2` `SH` rotation matrix from a 3x3 row-major rotation
/// matrix using the `Ivanic`-`Ruedenberg` recurrence.
///
/// Rows and columns are indexed by `m = -2..=2` (offset by two), matching the
/// `(Y(2,-2)..Y(2,2))` coefficient order. The result is orthonormal whenever the
/// input matrix is a proper rotation.
#[must_use]
pub fn l2_rotation_matrix(rot: [[f32; 3]; 3]) -> [[f32; 5]; 5] {
    let mut out = [[0.0_f32; 5]; 5];
    for m in -2..=2_i32 {
        for n in -2..=2_i32 {
            let (u, v) = uv_coeff(m, n);
            let value = u * u_term(&rot, m, n) + v * v_term(&rot, m, n);
            let mi = usize::try_from(m + 2).unwrap_or_default();
            let ni = usize::try_from(n + 2).unwrap_or_default();
            out[mi][ni] = value;
        }
    }
    out
}

/// Rotates a band-`L2` coefficient vector (order `(Y(2,-2)..Y(2,2))`) by a 3x3
/// row-major rotation matrix.
///
/// Because the band-`L2` `SH` rotation matrix is orthonormal, the coefficient
/// "energy" (sum of squares) is preserved up to floating-point rounding.
#[must_use]
pub fn rotate_l2(coeffs: [f32; 5], rot: [[f32; 3]; 3]) -> [f32; 5] {
    let matrix = l2_rotation_matrix(rot);
    let mut out = [0.0_f32; 5];
    for row in 0..5 {
        let r = matrix[row];
        out[row] = r[0] * coeffs[0]
            + r[1] * coeffs[1]
            + r[2] * coeffs[2]
            + r[3] * coeffs[3]
            + r[4] * coeffs[4];
    }
    out
}

/// Normalized linear interpolation between two colored band-`L0`/`L1` payloads.
///
/// The band-`L0` term is a plain lerp. Each channel's band-`L1` direction vector
/// is lerped and then rescaled so its magnitude equals the lerp of the endpoint
/// magnitudes — an energy-preserving approximation that avoids the shrink a raw
/// vector average would introduce. `t` is clamped to `0..=1`. A degenerate
/// (near-zero) blended direction collapses to zero instead of dividing by zero.
/// Energy-preserving blend of one color channel's band-`L1` direction vector.
fn nlerp_channel(va: [f32; 3], vb: [f32; 3], t: f32) -> [f32; 3] {
    let inv = 1.0 - t;
    let la = (va[0] * va[0] + va[1] * va[1] + va[2] * va[2]).sqrt();
    let lb = (vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2]).sqrt();
    let mut v = [
        inv * va[0] + t * vb[0],
        inv * va[1] + t * vb[1],
        inv * va[2] + t * vb[2],
    ];
    let target = inv * la + t * lb;
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq > MIN_LEN_SQ {
        let scale = target / len_sq.sqrt();
        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
    } else {
        v = [0.0, 0.0, 0.0];
    }
    v
}

#[must_use]
pub fn nlerp_l1(a: &ShL1Rgb, b: &ShL1Rgb, t: f32) -> ShL1Rgb {
    let t = t.clamp(0.0, 1.0);
    let inv = 1.0 - t;
    let l0 = [
        inv * a.l0[0] + t * b.l0[0],
        inv * a.l0[1] + t * b.l0[1],
        inv * a.l0[2] + t * b.l0[2],
    ];
    let r = nlerp_channel(
        [a.l1[0][0], a.l1[1][0], a.l1[2][0]],
        [b.l1[0][0], b.l1[1][0], b.l1[2][0]],
        t,
    );
    let g = nlerp_channel(
        [a.l1[0][1], a.l1[1][1], a.l1[2][1]],
        [b.l1[0][1], b.l1[1][1], b.l1[2][1]],
        t,
    );
    let bl = nlerp_channel(
        [a.l1[0][2], a.l1[1][2], a.l1[2][2]],
        [b.l1[0][2], b.l1[1][2], b.l1[2][2]],
        t,
    );
    let l1 = [
        [r[0], g[0], bl[0]],
        [r[1], g[1], bl[1]],
        [r[2], g[2], bl[2]],
    ];
    ShL1Rgb { l0, l1 }
}

/// The `std430` byte size of a storage buffer holding `count` [`ShL1Rgb`]
/// records, clamped up to a single element (see
/// [`super::gpu_layout::storage_bytes`]).
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(SH_ROTATE_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the float assertions in this module's tests.
    const CMP_EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    /// Row-major 3x3 product `a * b`.
    fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
        let mut out = [[0.0_f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        out
    }

    #[test]
    fn identity_matrix_leaves_l1_unchanged() {
        let c = [0.3, -0.7, 1.1];
        assert_eq!(rotate_l1(c, IDENTITY), c);
    }

    #[test]
    fn rotate_l1_90z_maps_x_axis_to_y_axis() {
        // 90 degrees about z: x -> y, y -> -x.
        let rot = rotation_matrix_from_quat(
            0.0,
            0.0,
            core::f32::consts::FRAC_1_SQRT_2,
            core::f32::consts::FRAC_1_SQRT_2,
        );
        // Pure +x band-L1 coefficient: (y, z, x) = (0, 0, 1).
        let out = rotate_l1([0.0, 0.0, 1.0], rot);
        // Expect pure +y: (y, z, x) = (1, 0, 0).
        assert!(approx(out[0], 1.0));
        assert!(approx(out[1], 0.0));
        assert!(approx(out[2], 0.0));
    }

    #[test]
    fn rotate_l1_90z_maps_y_axis_to_negative_x() {
        let rot = rotation_matrix_from_quat(
            0.0,
            0.0,
            core::f32::consts::FRAC_1_SQRT_2,
            core::f32::consts::FRAC_1_SQRT_2,
        );
        // Pure +y: (y, z, x) = (1, 0, 0).
        let out = rotate_l1([1.0, 0.0, 0.0], rot);
        // Expect pure -x: (y, z, x) = (0, 0, -1).
        assert!(approx(out[0], 0.0));
        assert!(approx(out[1], 0.0));
        assert!(approx(out[2], -1.0));
    }

    #[test]
    fn rotate_l1_preserves_length() {
        let rot = rotation_matrix_from_quat(0.2, 0.5, -0.3, 0.8);
        let c = [0.4, -1.2, 0.9];
        let out = rotate_l1(c, rot);
        let before = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        let after = out[0] * out[0] + out[1] * out[1] + out[2] * out[2];
        assert!(approx(before, after));
    }

    #[test]
    fn rotate_l1_compose_equals_product() {
        let r1 = rotation_matrix_from_quat(0.1, 0.2, 0.3, 0.9);
        let r2 = rotation_matrix_from_quat(-0.4, 0.1, 0.2, 0.85);
        let c = [0.6, -0.5, 0.7];
        let stepwise = rotate_l1(rotate_l1(c, r1), r2);
        let composed = rotate_l1(c, mat3_mul(r2, r1));
        for k in 0..3 {
            assert!(approx(stepwise[k], composed[k]));
        }
    }

    #[test]
    fn sh_l1_rgb_dc_invariant_under_rotation() {
        let sh = ShL1Rgb {
            l0: [0.2, 0.4, 0.6],
            l1: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        };
        let rot = rotation_matrix_from_quat(0.3, -0.2, 0.5, 0.78);
        let out = sh.rotate(rot);
        assert_eq!(out.l0, sh.l0);
    }

    #[test]
    fn sh_l1_rgb_rotates_each_channel_independently() {
        let sh = ShL1Rgb {
            l0: [0.0, 0.0, 0.0],
            // Channel r is pure +x, channel g is pure +y, channel b is pure +z.
            l1: [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        };
        let rot = rotation_matrix_from_quat(
            0.0,
            0.0,
            core::f32::consts::FRAC_1_SQRT_2,
            core::f32::consts::FRAC_1_SQRT_2,
        );
        let out = sh.rotate(rot);
        // Per channel, run rotate_l1 on the gathered triple and compare.
        for channel in 0..3 {
            let band = [sh.l1[0][channel], sh.l1[1][channel], sh.l1[2][channel]];
            let expect = rotate_l1(band, rot);
            assert!(approx(out.l1[0][channel], expect[0]));
            assert!(approx(out.l1[1][channel], expect[1]));
            assert!(approx(out.l1[2][channel], expect[2]));
        }
    }

    #[test]
    fn identity_matrix_leaves_l2_unchanged() {
        let c = [0.3, -0.7, 1.1, 0.2, -0.5];
        let out = rotate_l2(c, IDENTITY);
        for k in 0..5 {
            assert!(approx(out[k], c[k]));
        }
    }

    #[test]
    fn l2_rotation_matrix_is_identity_for_identity() {
        let m = l2_rotation_matrix(IDENTITY);
        for (i, row) in m.iter().enumerate() {
            for (j, &cell) in row.iter().enumerate() {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(approx(cell, expect));
            }
        }
    }

    #[test]
    fn l2_rotation_matrix_is_orthonormal() {
        let rot = rotation_matrix_from_quat(0.2, 0.5, -0.3, 0.8);
        let m = l2_rotation_matrix(rot);
        // M * M^T should be the identity.
        for (i, row_i) in m.iter().enumerate() {
            for (j, row_j) in m.iter().enumerate() {
                let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(x, y)| x * y).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn rotate_l2_preserves_energy() {
        let rot = rotation_matrix_from_quat(0.4, -0.1, 0.6, 0.7);
        let c = [0.3, -0.7, 1.1, 0.2, -0.5];
        let out = rotate_l2(c, rot);
        let before: f32 = c.iter().map(|v| v * v).sum();
        let after: f32 = out.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-4);
    }

    #[test]
    fn rotate_l2_compose_equals_product() {
        let r1 = rotation_matrix_from_quat(0.1, 0.2, 0.3, 0.9);
        let r2 = rotation_matrix_from_quat(-0.4, 0.1, 0.2, 0.85);
        let c = [0.6, -0.5, 0.7, -0.2, 0.9];
        let stepwise = rotate_l2(rotate_l2(c, r1), r2);
        let composed = rotate_l2(c, mat3_mul(r2, r1));
        for k in 0..5 {
            assert!((stepwise[k] - composed[k]).abs() < 1e-4);
        }
    }

    #[test]
    fn rotate_l2_90z_is_orthonormal_and_energy_safe() {
        let rot = rotation_matrix_from_quat(
            0.0,
            0.0,
            core::f32::consts::FRAC_1_SQRT_2,
            core::f32::consts::FRAC_1_SQRT_2,
        );
        let c = [1.0, 0.0, 0.0, 0.0, 0.0];
        let out = rotate_l2(c, rot);
        let after: f32 = out.iter().map(|v| v * v).sum();
        assert!((after - 1.0).abs() < 1e-4);
    }

    #[test]
    fn quat_identity_is_identity_matrix() {
        let m = rotation_matrix_from_quat(0.0, 0.0, 0.0, 1.0);
        for (i, row) in m.iter().enumerate() {
            for (j, &cell) in row.iter().enumerate() {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(approx(cell, expect));
            }
        }
    }

    #[test]
    fn quat_zero_returns_identity() {
        let m = rotation_matrix_from_quat(0.0, 0.0, 0.0, 0.0);
        assert_eq!(m, IDENTITY);
    }

    #[test]
    fn quat_matrix_is_orthonormal() {
        let m = rotation_matrix_from_quat(1.0, 2.0, 3.0, 4.0);
        for i in 0..3 {
            for j in 0..3 {
                let dot = m[i][0] * m[j][0] + m[i][1] * m[j][1] + m[i][2] * m[j][2];
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(approx(dot, expect));
            }
        }
    }

    #[test]
    fn quat_nonunit_still_orthonormal() {
        // Deliberately non-unit quaternion; robust form must still normalize.
        let m = rotation_matrix_from_quat(0.0, 0.0, 3.0, 3.0);
        for row in &m {
            let len = row[0] * row[0] + row[1] * row[1] + row[2] * row[2];
            assert!(approx(len, 1.0));
        }
    }

    #[test]
    fn quat_90z_matches_expected_matrix() {
        let m = rotation_matrix_from_quat(
            0.0,
            0.0,
            core::f32::consts::FRAC_1_SQRT_2,
            core::f32::consts::FRAC_1_SQRT_2,
        );
        let expect = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx(m[i][j], expect[i][j]));
            }
        }
    }

    #[test]
    fn rotate_l1_via_quat_90_maps_axis() {
        let rot = rotation_matrix_from_quat(
            0.0,
            0.0,
            core::f32::consts::FRAC_1_SQRT_2,
            core::f32::consts::FRAC_1_SQRT_2,
        );
        let out = rotate_l1([0.0, 0.0, 1.0], rot);
        assert!(approx(out[0], 1.0));
    }

    #[test]
    fn nlerp_endpoint_t0_returns_a() {
        let a = ShL1Rgb {
            l0: [0.2, 0.4, 0.6],
            l1: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        };
        let b = ShL1Rgb {
            l0: [1.0, 1.0, 1.0],
            l1: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        assert_eq!(nlerp_l1(&a, &b, 0.0), a);
    }

    #[test]
    fn nlerp_endpoint_t1_returns_b() {
        let a = ShL1Rgb {
            l0: [0.2, 0.4, 0.6],
            l1: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        };
        let b = ShL1Rgb {
            l0: [1.0, 1.0, 1.0],
            l1: [[1.0, 0.2, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]],
        };
        assert_eq!(nlerp_l1(&a, &b, 1.0), b);
    }

    #[test]
    fn nlerp_midpoint_l0_is_average() {
        let a = ShL1Rgb {
            l0: [0.0, 0.2, 1.0],
            l1: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let b = ShL1Rgb {
            l0: [1.0, 0.6, 0.0],
            l1: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let out = nlerp_l1(&a, &b, 0.5);
        assert!(approx(out.l0[0], 0.5));
        assert!(approx(out.l0[1], 0.4));
        assert!(approx(out.l0[2], 0.5));
    }

    #[test]
    fn nlerp_preserves_l1_energy() {
        // Channel r: a is +x length 1, b is +y length 1; midpoint should keep length 1.
        let a = ShL1Rgb {
            l0: [0.0, 0.0, 0.0],
            l1: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        };
        let b = ShL1Rgb {
            l0: [0.0, 0.0, 0.0],
            l1: [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        };
        let out = nlerp_l1(&a, &b, 0.5);
        let len =
            out.l1[0][0] * out.l1[0][0] + out.l1[1][0] * out.l1[1][0] + out.l1[2][0] * out.l1[2][0];
        assert!(approx(len, 1.0));
    }

    #[test]
    fn nlerp_clamps_t_above_one() {
        let a = ShL1Rgb {
            l0: [0.2, 0.4, 0.6],
            l1: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        };
        let b = ShL1Rgb {
            l0: [1.0, 1.0, 1.0],
            l1: [[1.0, 0.2, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]],
        };
        // t > 1 clamps to 1, so the result equals b.
        assert_eq!(nlerp_l1(&a, &b, 2.5), b);
    }

    #[test]
    fn std430_size_is_four_vec4() {
        assert_eq!(SH_ROTATE_STD430_SIZE, 64);
    }

    #[test]
    fn std430_padding_words_are_zero() {
        let sh = ShL1Rgb {
            l0: [0.2, 0.4, 0.6],
            l1: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        };
        let words = sh.to_std430();
        assert_eq!(words[3], 0);
        assert_eq!(words[7], 0);
        assert_eq!(words[11], 0);
        assert_eq!(words[15], 0);
    }

    #[test]
    fn std430_encodes_scalar_bits() {
        let sh = ShL1Rgb {
            l0: [0.2, 0.4, 0.6],
            l1: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        };
        let words = sh.to_std430();
        assert_eq!(words[0], 0.2_f32.to_bits());
        assert_eq!(words[4], 0.1_f32.to_bits());
        assert_eq!(words[8], 0.4_f32.to_bits());
        assert_eq!(words[12], 0.7_f32.to_bits());
    }

    #[test]
    fn gpu_storage_bytes_matches_layout() {
        assert_eq!(gpu_storage_bytes(0), SH_ROTATE_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), SH_ROTATE_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(10), SH_ROTATE_STD430_SIZE * 10);
    }
}
