//! `CPU` reference for 3D Gaussian splatting (`3DGS` / `EWA` splatting) used by
//! the point-primitive particle renderer (design §16, §22).
//!
//! A Gaussian splat represents a particle as an anisotropic 3D Gaussian: a mean
//! position, a per-axis scale, an orientation quaternion, and an opacity. The
//! renderer projects each 3D Gaussian to a 2D screen-space Gaussian (the `EWA`
//! resampling filter), inverts its 2D covariance into a *conic* form, and
//! composites the resulting elliptical footprint front-to-back. This module
//! owns the device-free reference so a future `GPU` draw kernel matches it bit
//! for bit; the `GPU` packing follows the shared `std430` `vec4` alignment from
//! [`super::gpu_layout`].
//!
//! The pipeline mirrors the production `3DGS` math without reusing any of its
//! code:
//!
//! 1. [`Gaussian3d::cov3_from_scale_quat`] builds the 3D covariance
//!    `Sigma = R S Sᵀ Rᵀ`. The rotation `R` is expanded from the quaternion
//!    with a pure multiply-add polynomial (no trigonometry), and the result is
//!    returned as the six unique entries of the symmetric 3x3 matrix.
//! 2. [`project_to_2d`] applies the perspective camera Jacobian `J` (built from
//!    focal length over depth, a pure division) to obtain the 2D covariance
//!    `Sigma' = J Sigma Jᵀ`.
//! 3. [`conic_from_cov2`] inverts the 2x2 covariance into a [`Conic2d`],
//!    returning `None` for a degenerate (near-singular) footprint.
//! 4. [`Conic2d::eval_weight`] evaluates the splat's alpha at a pixel offset. It
//!    replaces the Gaussian `exp(-power)` with a *rational* approximation
//!    (`1 / (1 + power + power² / 2)`) so no transcendental function is used and
//!    the falloff stays deterministic and portable.
//! 5. [`Conic2d::bounding_aabb`] and [`Conic2d::bounding_tiles`] give the 3σ
//!    `AABB` and the covered `tile` range for the binning stage.
//!
//! Only `sqrt`, `floor`, division, and integer / polynomial arithmetic appear
//! here — no `sin`, `cos`, `exp`, `powf`, or other transcendental call — so the
//! contract is reproducible across the `CPU` reference and the eventual `GPU`
//! kernel.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Minimum squared length below which a quaternion is treated as degenerate and
/// falls back to the identity rotation instead of dividing by zero.
const MIN_LEN_SQ: f32 = 1e-12;

/// Minimum absolute view-space depth used when projecting, so a Gaussian on or
/// behind the camera plane cannot divide by zero in the Jacobian.
const MIN_DEPTH: f32 = 1e-6;

/// Determinant magnitude below which a 2x2 matrix is treated as singular
/// (degenerate splat) rather than inverted into a `NaN`-laden conic.
const DET_EPS: f32 = 1e-12;

/// Gaussian footprint radius, in standard deviations, used for the bounding
/// `AABB`. Three sigmas capture ~99.7% of a 1D Gaussian's mass.
const SIGMA_RADIUS: f32 = 3.0;

/// Byte stride of one [`Gaussian3d`] record in a `std430` storage buffer.
///
/// The eleven meaningful scalars pack into three `vec4` slots:
/// `vec4(mean, opacity)`, `vec4(scale, pad)`, and `vec4(quat)`.
pub const GAUSSIAN_STRIDE: usize = 3 * VEC4_STRIDE;

/// A single anisotropic 3D Gaussian primitive (a `3DGS` splat).
///
/// `quat` is stored in `[x, y, z, w]` order and need not be pre-normalized: the
/// covariance builder normalizes it internally.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gaussian3d {
    /// World- (or view-) space center of the Gaussian.
    pub mean: [f32; 3],
    /// Per-axis standard deviations before rotation.
    pub scale: [f32; 3],
    /// Orientation quaternion in `[x, y, z, w]` order.
    pub quat: [f32; 4],
    /// Scalar opacity multiplier in `0..=1`.
    pub opacity: f32,
}

impl Gaussian3d {
    /// Creates a new Gaussian splat from its raw fields.
    #[must_use]
    pub fn new(mean: [f32; 3], scale: [f32; 3], quat: [f32; 4], opacity: f32) -> Self {
        Self {
            mean,
            scale,
            quat,
            opacity,
        }
    }

    /// Builds the 3D covariance `Sigma = R S Sᵀ Rᵀ` and returns its six unique
    /// symmetric entries `[xx, xy, xz, yy, yz, zz]`.
    ///
    /// The rotation `R` comes from [`quat_to_rotation`] (a pure multiply-add
    /// quaternion-to-matrix expansion). Because `S` is diagonal,
    /// `Sigma_ij = Σ_k R_ik R_jk s_k²`, which is evaluated directly without
    /// materializing the intermediate `R S` product.
    #[must_use]
    pub fn cov3_from_scale_quat(&self) -> [f32; 6] {
        let r = quat_to_rotation(self.quat);
        let s2 = [
            self.scale[0] * self.scale[0],
            self.scale[1] * self.scale[1],
            self.scale[2] * self.scale[2],
        ];
        let sigma = |i: usize, j: usize| -> f32 {
            r[i][0] * r[j][0] * s2[0] + r[i][1] * r[j][1] * s2[1] + r[i][2] * r[j][2] * s2[2]
        };
        [
            sigma(0, 0),
            sigma(0, 1),
            sigma(0, 2),
            sigma(1, 1),
            sigma(1, 2),
            sigma(2, 2),
        ]
    }

    /// Packs the splat into three `std430` `vec4` slots as raw bit patterns.
    ///
    /// Layout: `[mean.x, mean.y, mean.z, opacity]`,
    /// `[scale.x, scale.y, scale.z, 0]`, `[quat.x, quat.y, quat.z, quat.w]`.
    #[must_use]
    pub fn to_std430(&self) -> [u32; 12] {
        [
            self.mean[0].to_bits(),
            self.mean[1].to_bits(),
            self.mean[2].to_bits(),
            self.opacity.to_bits(),
            self.scale[0].to_bits(),
            self.scale[1].to_bits(),
            self.scale[2].to_bits(),
            0.0f32.to_bits(),
            self.quat[0].to_bits(),
            self.quat[1].to_bits(),
            self.quat[2].to_bits(),
            self.quat[3].to_bits(),
        ]
    }
}

/// Expands a quaternion (in `[x, y, z, w]` order) into a 3x3 rotation matrix
/// using the standard multiply-add polynomial identity.
///
/// The quaternion is normalized first; a degenerate (near-zero) quaternion
/// falls back to the identity rotation. No trigonometric function is used: the
/// rotation matrix is a pure polynomial in the quaternion components.
#[must_use]
pub fn quat_to_rotation(quat: [f32; 4]) -> [[f32; 3]; 3] {
    let len_sq = quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3];
    let (x, y, z, w) = if len_sq < MIN_LEN_SQ {
        (0.0, 0.0, 0.0, 1.0)
    } else {
        let inv_len = 1.0 / len_sq.sqrt();
        (
            quat[0] * inv_len,
            quat[1] * inv_len,
            quat[2] * inv_len,
            quat[3] * inv_len,
        )
    };
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]
}

/// Projects a 3D covariance to its 2D screen-space form via the `EWA` camera
/// Jacobian and returns the three unique entries `[a, b, c]` of the symmetric
/// 2x2 covariance `[[a, b], [b, c]]`.
///
/// `cov3` is the six-entry symmetric 3D covariance, `mean_view` is the splat
/// center in view space, and `focal` holds the `x`/`y` focal lengths in pixels.
/// The Jacobian `J` of the perspective divide is
/// `[[fx/z, 0, -fx·x/z²], [0, fy/z, -fy·y/z²]]`; this assumes the view rotation
/// has already been folded into `cov3` (so `W` is the identity here) and forms
/// `Sigma' = J Sigma Jᵀ`. The depth is clamped away from zero by [`MIN_DEPTH`].
#[must_use]
pub fn project_to_2d(cov3: [f32; 6], mean_view: [f32; 3], focal: [f32; 2]) -> [f32; 3] {
    let fx = focal[0];
    let fy = focal[1];
    let depth = mean_view[2];
    let z = if depth.abs() < MIN_DEPTH {
        if depth < 0.0 {
            -MIN_DEPTH
        } else {
            MIN_DEPTH
        }
    } else {
        depth
    };
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;

    // Jacobian rows: J = [[j00, 0, j02], [0, j11, j12]].
    let j00 = fx * inv_z;
    let j02 = -fx * mean_view[0] * inv_z2;
    let j11 = fy * inv_z;
    let j12 = -fy * mean_view[1] * inv_z2;

    let c00 = cov3[0];
    let c01 = cov3[1];
    let c02 = cov3[2];
    let c11 = cov3[3];
    let c12 = cov3[4];
    let c22 = cov3[5];

    // A = J · Sigma (2x3).
    let a00 = j00 * c00 + j02 * c02;
    let a01 = j00 * c01 + j02 * c12;
    let a02 = j00 * c02 + j02 * c22;
    let a11 = j11 * c11 + j12 * c12;
    let a12 = j11 * c12 + j12 * c22;

    // Sigma' = A · Jᵀ (2x2), symmetric.
    let out_a = a00 * j00 + a02 * j02;
    let out_b = a01 * j11 + a02 * j12;
    let out_c = a11 * j11 + a12 * j12;
    [out_a, out_b, out_c]
}

/// The inverse-covariance (conic) form of a projected 2D Gaussian footprint.
///
/// The conic coefficients `[a, b, c]` are the entries of `Sigma'⁻¹`, and
/// `center` is the pixel-space center of the splat. The evaluated power is the
/// Mahalanobis form `0.5·(a·dx² + c·dy²) + b·dx·dy`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conic2d {
    /// Top-left inverse-covariance coefficient.
    pub a: f32,
    /// Off-diagonal inverse-covariance coefficient.
    pub b: f32,
    /// Bottom-right inverse-covariance coefficient.
    pub c: f32,
    /// Pixel-space center of the footprint.
    pub center: [f32; 2],
}

/// Axis-aligned bounding box of a splat footprint in pixel space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplatAabb {
    /// Minimum corner `[x, y]`.
    pub min: [f32; 2],
    /// Maximum corner `[x, y]`.
    pub max: [f32; 2],
}

/// Inclusive range of `tile` indices a footprint overlaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRange {
    /// First overlapped column tile.
    pub min_x: u32,
    /// First overlapped row tile.
    pub min_y: u32,
    /// Last overlapped column tile (inclusive).
    pub max_x: u32,
    /// Last overlapped row tile (inclusive).
    pub max_y: u32,
}

impl TileRange {
    /// Number of tiles covered by this range (inclusive on both ends).
    #[must_use]
    pub fn tile_count(&self) -> u32 {
        let cols = self.max_x - self.min_x + 1;
        let rows = self.max_y - self.min_y + 1;
        cols * rows
    }
}

/// Inverts the symmetric 2x2 covariance `[[a, b], [b, c]]` into its [`Conic2d`]
/// form, returning `None` when the determinant `a·c - b²` is near zero (a
/// degenerate, un-invertible footprint).
///
/// The returned conic is centered at the origin; callers set [`Conic2d::center`]
/// to the projected pixel position separately.
#[must_use]
pub fn conic_from_cov2(cov2: [f32; 3]) -> Option<Conic2d> {
    let a = cov2[0];
    let b = cov2[1];
    let c = cov2[2];
    let det = a * c - b * b;
    if det < DET_EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    Some(Conic2d {
        a: c * inv_det,
        b: -b * inv_det,
        c: a * inv_det,
        center: [0.0, 0.0],
    })
}

impl Conic2d {
    /// Recovers the 2D covariance `[a, b, c]` this conic was inverted from, or
    /// zeros for a degenerate conic. Useful for bounds and round-trip checks.
    #[must_use]
    pub fn to_cov2(&self) -> [f32; 3] {
        let det = self.a * self.c - self.b * self.b;
        if det.abs() < DET_EPS {
            return [0.0, 0.0, 0.0];
        }
        let inv_det = 1.0 / det;
        [self.c * inv_det, -self.b * inv_det, self.a * inv_det]
    }

    /// Evaluates the splat alpha at a pixel offset `(dx, dy)` from the center.
    ///
    /// The Mahalanobis power is `0.5·(a·dx² + c·dy²) + b·dx·dy`, clamped to be
    /// non-negative. The Gaussian falloff `exp(-power)` is replaced by the
    /// rational approximation `1 / (1 + power + power² / 2)` (the second-order
    /// Padé-style denominator of `exp`), which is `1` at the center, strictly
    /// decreasing in `power`, and always positive — no transcendental call. The
    /// result is scaled by `opacity`.
    #[must_use]
    pub fn eval_weight(&self, dx: f32, dy: f32, opacity: f32) -> f32 {
        let power = 0.5 * (self.a * dx * dx + self.c * dy * dy) + self.b * dx * dy;
        let p = power.max(0.0);
        let falloff = 1.0 / (1.0 + p + 0.5 * p * p);
        opacity * falloff
    }

    /// Returns the 3σ axis-aligned bounding box of the footprint in pixel space.
    ///
    /// The half-extents are `SIGMA_RADIUS · sqrt(Sigma_xx)` and
    /// `SIGMA_RADIUS · sqrt(Sigma_yy)`, where the covariance is recovered from
    /// the conic via [`Conic2d::to_cov2`]. A degenerate conic yields a
    /// zero-size box at the center.
    #[must_use]
    pub fn bounding_aabb(&self) -> SplatAabb {
        let cov = self.to_cov2();
        let rx = SIGMA_RADIUS * cov[0].max(0.0).sqrt();
        let ry = SIGMA_RADIUS * cov[2].max(0.0).sqrt();
        SplatAabb {
            min: [self.center[0] - rx, self.center[1] - ry],
            max: [self.center[0] + rx, self.center[1] + ry],
        }
    }

    /// Returns the inclusive range of `tile` indices the 3σ `AABB` overlaps for
    /// a square `tile` of `tile_size` pixels.
    ///
    /// Tile indices are floored from the box corners and clamped to be
    /// non-negative, so a footprint straddling the screen origin still yields a
    /// valid, in-bounds range. A `tile_size` of zero is treated as one.
    #[must_use]
    pub fn bounding_tiles(&self, tile_size: u32) -> TileRange {
        let ts = tile_size.max(1);
        let ts_f = ts as f32;
        let aabb = self.bounding_aabb();
        TileRange {
            min_x: floor_to_u32(aabb.min[0] / ts_f),
            min_y: floor_to_u32(aabb.min[1] / ts_f),
            max_x: floor_to_u32(aabb.max[0] / ts_f),
            max_y: floor_to_u32(aabb.max[1] / ts_f),
        }
    }
}

/// Floors a scalar and clamps it to a non-negative `u32` `tile` index.
fn floor_to_u32(x: f32) -> u32 {
    let f = x.floor().max(0.0);
    f as u32
}

/// Total byte size of a `std430` storage buffer holding `count` Gaussian splats,
/// clamped up to a single element so an empty set still yields a valid `GPU`
/// binding.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(GAUSSIAN_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Comparison epsilon for the test assertions only.
    const CMP_EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    /// Determinant of a symmetric 3x3 given as `[xx, xy, xz, yy, yz, zz]`.
    fn det3(m: [f32; 6]) -> f32 {
        let (xx, xy, xz, yy, yz, zz) = (m[0], m[1], m[2], m[3], m[4], m[5]);
        xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
    }

    #[test]
    fn identity_quat_is_identity_matrix() {
        let r = quat_to_rotation([0.0, 0.0, 0.0, 1.0]);
        for (i, row) in r.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(val, expected));
            }
        }
    }

    #[test]
    fn degenerate_quat_falls_back_to_identity() {
        let r = quat_to_rotation([0.0, 0.0, 0.0, 0.0]);
        assert!(approx_eq(r[0][0], 1.0));
        assert!(approx_eq(r[1][1], 1.0));
        assert!(approx_eq(r[2][2], 1.0));
    }

    #[test]
    fn rotation_matrix_is_orthonormal() {
        // 90° about x expressed as a non-normalized quaternion.
        let r = quat_to_rotation([1.0, 0.0, 0.0, 1.0]);
        // Each row has unit length.
        for row in &r {
            let len_sq = row[0] * row[0] + row[1] * row[1] + row[2] * row[2];
            assert!(approx_eq(len_sq, 1.0));
        }
        // Rows are mutually orthogonal.
        let dot01 = r[0][0] * r[1][0] + r[0][1] * r[1][1] + r[0][2] * r[1][2];
        let dot02 = r[0][0] * r[2][0] + r[0][1] * r[2][1] + r[0][2] * r[2][2];
        assert!(approx_eq(dot01, 0.0));
        assert!(approx_eq(dot02, 0.0));
    }

    #[test]
    fn cov3_unit_quat_is_symmetric_and_positive_definite() {
        let g = Gaussian3d::new([0.0, 0.0, 0.0], [2.0, 1.0, 0.5], [1.0, 0.0, 0.0, 1.0], 1.0);
        let cov = g.cov3_from_scale_quat();
        // Diagonal entries are strictly positive.
        assert!(cov[0] > 0.0);
        assert!(cov[3] > 0.0);
        assert!(cov[5] > 0.0);
        // Determinant is positive (positive-definite).
        assert!(det3(cov) > 0.0);
    }

    #[test]
    fn cov3_isotropic_scale_is_scaled_identity() {
        let s = 1.5;
        let g = Gaussian3d::new([0.0, 0.0, 0.0], [s, s, s], [0.3, -0.6, 0.2, 0.9], 1.0);
        let cov = g.cov3_from_scale_quat();
        let s2 = s * s;
        // A rotated isotropic Gaussian stays isotropic: diagonal = s², off = 0.
        assert!(approx_eq(cov[0], s2));
        assert!(approx_eq(cov[3], s2));
        assert!(approx_eq(cov[5], s2));
        assert!(approx_eq(cov[1], 0.0));
        assert!(approx_eq(cov[2], 0.0));
        assert!(approx_eq(cov[4], 0.0));
    }

    #[test]
    fn cov3_identity_rotation_is_scale_squared() {
        let g = Gaussian3d::new([0.0, 0.0, 0.0], [2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0], 1.0);
        let cov = g.cov3_from_scale_quat();
        assert!(approx_eq(cov[0], 4.0));
        assert!(approx_eq(cov[3], 9.0));
        assert!(approx_eq(cov[5], 16.0));
        assert!(approx_eq(cov[1], 0.0));
        assert!(approx_eq(cov[2], 0.0));
        assert!(approx_eq(cov[4], 0.0));
    }

    #[test]
    fn cov3_is_deterministic() {
        let g = Gaussian3d::new([1.0, 2.0, 3.0], [0.7, 1.3, 0.9], [0.1, 0.2, 0.3, 0.9], 0.5);
        assert_eq!(g.cov3_from_scale_quat(), g.cov3_from_scale_quat());
    }

    #[test]
    fn project_isotropic_cov_is_symmetric_positive() {
        // Isotropic 3D covariance projected head-on with equal focal lengths.
        let cov3 = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let cov2 = project_to_2d(cov3, [0.0, 0.0, 5.0], [500.0, 500.0]);
        assert!(cov2[0] > 0.0);
        assert!(cov2[2] > 0.0);
        // On-axis, equal focal lengths -> circular projection (a == c, b == 0).
        assert!(approx_eq(cov2[0], cov2[2]));
        assert!(approx_eq(cov2[1], 0.0));
    }

    #[test]
    fn project_closer_splat_is_larger() {
        let cov3 = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let near = project_to_2d(cov3, [0.0, 0.0, 2.0], [500.0, 500.0]);
        let far = project_to_2d(cov3, [0.0, 0.0, 8.0], [500.0, 500.0]);
        // The perspective Jacobian scales as 1/z², so the near splat is wider.
        assert!(near[0] > far[0]);
    }

    #[test]
    fn project_clamps_zero_depth_without_nan() {
        let cov3 = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let cov2 = project_to_2d(cov3, [0.0, 0.0, 0.0], [500.0, 500.0]);
        assert!(cov2[0].is_finite());
        assert!(cov2[1].is_finite());
        assert!(cov2[2].is_finite());
    }

    #[test]
    fn conic_isotropic_cov_is_circular() {
        let conic = conic_from_cov2([4.0, 0.0, 4.0]).expect("invertible");
        assert!(approx_eq(conic.a, conic.c));
        assert!(approx_eq(conic.b, 0.0));
        // Inverse of diag(4, 4) is diag(0.25, 0.25).
        assert!(approx_eq(conic.a, 0.25));
    }

    #[test]
    fn conic_from_cov2_round_trips() {
        let cov2 = [4.0, 1.0, 3.0];
        let conic = conic_from_cov2(cov2).expect("invertible");
        let recovered = conic.to_cov2();
        assert!(approx_eq(recovered[0], cov2[0]));
        assert!(approx_eq(recovered[1], cov2[1]));
        assert!(approx_eq(recovered[2], cov2[2]));
    }

    #[test]
    fn conic_from_cov2_degenerate_returns_none() {
        // Rank-1 covariance: det = 1*1 - 1*1 = 0.
        assert!(conic_from_cov2([1.0, 1.0, 1.0]).is_none());
        // Exactly zero covariance is degenerate too.
        assert!(conic_from_cov2([0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn eval_weight_is_maximal_at_center() {
        let conic = conic_from_cov2([4.0, 0.0, 4.0]).expect("invertible");
        let opacity = 0.8;
        let center = conic.eval_weight(0.0, 0.0, opacity);
        assert!(approx_eq(center, opacity));
        // Any offset is strictly smaller than the center weight.
        assert!(conic.eval_weight(1.0, 0.0, opacity) < center);
        assert!(conic.eval_weight(0.0, 1.0, opacity) < center);
    }

    #[test]
    fn eval_weight_decreases_monotonically_with_distance() {
        let conic = conic_from_cov2([4.0, 0.0, 4.0]).expect("invertible");
        let w0 = conic.eval_weight(0.0, 0.0, 1.0);
        let w1 = conic.eval_weight(1.0, 0.0, 1.0);
        let w2 = conic.eval_weight(2.0, 0.0, 1.0);
        let w3 = conic.eval_weight(4.0, 0.0, 1.0);
        assert!(w0 > w1);
        assert!(w1 > w2);
        assert!(w2 > w3);
        assert!(w3 > 0.0);
    }

    #[test]
    fn eval_weight_scales_linearly_with_opacity() {
        let conic = conic_from_cov2([4.0, 0.0, 4.0]).expect("invertible");
        let full = conic.eval_weight(1.0, 0.5, 1.0);
        let half = conic.eval_weight(1.0, 0.5, 0.5);
        assert!(approx_eq(half, 0.5 * full));
    }

    #[test]
    fn eval_weight_is_isotropic_for_circular_conic() {
        let conic = conic_from_cov2([4.0, 0.0, 4.0]).expect("invertible");
        // A circular footprint gives equal weight along x and y at equal radius.
        let along_x = conic.eval_weight(1.5, 0.0, 1.0);
        let along_y = conic.eval_weight(0.0, 1.5, 1.0);
        assert!(approx_eq(along_x, along_y));
    }

    #[test]
    fn bounding_aabb_isotropic_is_square_three_sigma() {
        // cov = diag(4, 4) -> sigma = 2 -> 3-sigma radius = 6.
        let conic = conic_from_cov2([4.0, 0.0, 4.0]).expect("invertible");
        let aabb = conic.bounding_aabb();
        assert!(approx_eq(aabb.min[0], -6.0));
        assert!(approx_eq(aabb.min[1], -6.0));
        assert!(approx_eq(aabb.max[0], 6.0));
        assert!(approx_eq(aabb.max[1], 6.0));
    }

    #[test]
    fn bounding_aabb_is_centered() {
        let mut conic = conic_from_cov2([4.0, 0.0, 9.0]).expect("invertible");
        conic.center = [50.0, 70.0];
        let aabb = conic.bounding_aabb();
        // sigma_x = 2 -> rx = 6; sigma_y = 3 -> ry = 9.
        assert!(approx_eq(aabb.min[0], 44.0));
        assert!(approx_eq(aabb.max[0], 56.0));
        assert!(approx_eq(aabb.min[1], 61.0));
        assert!(approx_eq(aabb.max[1], 79.0));
    }

    #[test]
    fn bounding_tiles_covers_expected_range() {
        let mut conic = conic_from_cov2([4.0, 0.0, 4.0]).expect("invertible");
        conic.center = [50.0, 70.0];
        // AABB = [44, 64]..[56, 76]; tile 16 -> x: 2..3, y: 4..4.
        let tiles = conic.bounding_tiles(16);
        assert_eq!(tiles.min_x, 2);
        assert_eq!(tiles.max_x, 3);
        assert_eq!(tiles.min_y, 4);
        assert_eq!(tiles.max_y, 4);
        assert_eq!(tiles.tile_count(), 2);
    }

    #[test]
    fn bounding_tiles_clamps_negative_to_zero() {
        // Centered at origin -> AABB spans negative pixels, which clamp to 0.
        let conic = conic_from_cov2([4.0, 0.0, 4.0]).expect("invertible");
        let tiles = conic.bounding_tiles(16);
        assert_eq!(tiles.min_x, 0);
        assert_eq!(tiles.min_y, 0);
    }

    #[test]
    fn bounding_tiles_treats_zero_tile_size_as_one() {
        let conic = conic_from_cov2([4.0, 0.0, 4.0]).expect("invertible");
        let tiles = conic.bounding_tiles(0);
        // Does not panic and yields a sane inclusive range.
        assert!(tiles.max_x >= tiles.min_x);
        assert!(tiles.max_y >= tiles.min_y);
    }

    #[test]
    fn std430_stride_and_storage_bytes() {
        assert_eq!(GAUSSIAN_STRIDE, 48);
        // Empty set still reserves a single element.
        assert_eq!(gpu_storage_bytes(0), GAUSSIAN_STRIDE);
        assert_eq!(gpu_storage_bytes(1), GAUSSIAN_STRIDE);
        assert_eq!(gpu_storage_bytes(4), 4 * GAUSSIAN_STRIDE);
    }

    #[test]
    fn std430_packing_layout_round_trips() {
        let g = Gaussian3d::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.1, 0.2, 0.3, 0.9], 0.7);
        let packed = g.to_std430();
        assert_eq!(packed.len(), 12);
        // Slot 0: mean, opacity.
        assert!(approx_eq(f32::from_bits(packed[0]), 1.0));
        assert!(approx_eq(f32::from_bits(packed[1]), 2.0));
        assert!(approx_eq(f32::from_bits(packed[2]), 3.0));
        assert!(approx_eq(f32::from_bits(packed[3]), 0.7));
        // Slot 1: scale, pad.
        assert!(approx_eq(f32::from_bits(packed[4]), 4.0));
        assert!(approx_eq(f32::from_bits(packed[5]), 5.0));
        assert!(approx_eq(f32::from_bits(packed[6]), 6.0));
        assert!(approx_eq(f32::from_bits(packed[7]), 0.0));
        // Slot 2: quat.
        assert!(approx_eq(f32::from_bits(packed[8]), 0.1));
        assert!(approx_eq(f32::from_bits(packed[9]), 0.2));
        assert!(approx_eq(f32::from_bits(packed[10]), 0.3));
        assert!(approx_eq(f32::from_bits(packed[11]), 0.9));
    }

    #[test]
    fn full_pipeline_produces_invertible_footprint() {
        let g = Gaussian3d::new([0.5, -0.5, 6.0], [0.2, 0.3, 0.1], [0.2, 0.1, 0.4, 0.9], 0.9);
        let cov3 = g.cov3_from_scale_quat();
        let cov2 = project_to_2d(cov3, g.mean, [600.0, 600.0]);
        let conic = conic_from_cov2(cov2).expect("well-formed splat inverts");
        // Center weight equals opacity and a distant sample is dimmer.
        let peak = conic.eval_weight(0.0, 0.0, g.opacity);
        assert!(approx_eq(peak, g.opacity));
        assert!(conic.eval_weight(3.0, 3.0, g.opacity) < peak);
        // Collect a small monotone falloff profile to exercise Vec usage.
        let profile: Vec<f32> = (0..4)
            .map(|i| conic.eval_weight(i as f32, 0.0, g.opacity))
            .collect();
        for w in profile.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }
}
