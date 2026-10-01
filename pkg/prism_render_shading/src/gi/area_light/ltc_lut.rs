//! Linearly Transformed Cosines (LTC) matrix look-up table — CPU golden.
//!
//! This module is the backend-neutral numerical reference for the LTC
//! representation of the GGX specular lobe used by the engine's area-light
//! passes.  It stores, per `(n·v, roughness)` grid texel, the five non-zero
//! coefficients of the *inverse* transform `M⁻¹` that maps a world/shading-frame
//! light direction back into the canonical clamped-cosine space where the
//! polygon integration of [`super::polygon`] is exact.
//!
//! Because the upstream UE / Heitz baked LUT data cannot be fetched locally,
//! the table is produced by a **purely numerical fitter**: for a given
//! `(n·v, roughness)` the GGX lobe of [`crate::gi::spec_gi::ggx_lobe`] is
//! integrated over the hemisphere and its first moment (mean reflected
//! direction) and tangential second moments are matched by an LTC matrix.  No
//! AI / ML / neural components are involved — only classical moment matching.
//!
//! # Conventions
//! * All directions live in a *local shading frame* with the surface normal at
//!   `+Z`; a direction's `z` is its cosine with the normal.  The view direction
//!   is taken in the `+X` half of the `XZ` incidence plane, so the isotropic
//!   LTC matrix has the block-sparse structure
//!
//!   ```text
//!   M = [ m00  0   m02 ]        M⁻¹ = [ a00  0   a02 ]
//!       [ 0    m11 0   ]              [ 0    a11 0   ]
//!       [ m20  0   m22 ]              [ a20  0   a22 ]
//!   ```
//!
//!   and the LUT stores the five coefficients `(a00, a02, a11, a20, a22)` of
//!   `M⁻¹` plus the lobe amplitude (directional albedo).
//! * The LTC distribution `D` is invariant to an overall scale of `M`, so the
//!   forward matrix is normalised to a unit mean-direction column before being
//!   inverted; a degenerate (near-singular) fit falls back to the identity.
//! * Transcendental math goes through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent `f32::sqrt`.  Everything is `f32` to mirror the GPU twin.
//! * Every function is a deterministic pure function (no RNG / IO / GPU /
//!   `unsafe`) and defends against degeneracy — denominators are floored,
//!   determinants are checked, and no result is ever `NaN`.
//!
//! # References
//! * Heitz, Dupuy, Hill, Neubelt 2016, *Real-Time Polygonal-Light Shading with
//!   Linearly Transformed Cosines* (SIGGRAPH).
//! * Hill, Heitz 2016, *LTC Fitting* reference implementation (BRDF-LTC fit).

use alloc::vec::Vec;
use bevy_math::{Mat3, Vec3, ops};
use core::f32::consts::PI;

use crate::gi::spec_gi::ggx_lobe::{ggx_brdf_scalar, roughness_to_alpha};

/// Smallest `|det|` of the fitted forward matrix treated as invertible.  Below
/// this the lobe is a near-mirror and the fit collapses to the identity.
pub const MIN_DET: f32 = 1.0e-5;

/// Minimum tangential standard deviation used when building the forward matrix,
/// keeping the lobe a finite (if very sharp) distribution.
pub const MIN_SPREAD: f32 = 1.0e-4;

/// Perceptual roughness at or below which the GGX lobe is treated as a perfect
/// mirror; the LTC fit then collapses to [`LtcCoeffs::IDENTITY`] by convention,
/// since a Dirac reflection cannot be resolved by a clamped-cosine transform.
pub const MIRROR_ROUGHNESS: f32 = 1.0e-3;

/// Canonical clamped-cosine tangential standard deviation.
///
/// For the clamped-cosine distribution `D(ω) = cosθ/π` on the upper hemisphere,
/// `⟨(sinθ cosφ)²⟩ = 1/4`, so its projected standard deviation is `1/2`.  The
/// fitter scales measured spreads relative to this reference.
pub const COSINE_STD: f32 = 0.5;

/// Default LUT resolution per axis.
pub const DEFAULT_SIZE: u32 = 32;

/// Default hemisphere grid resolution per axis used by the fitter.
pub const DEFAULT_FIT_GRID: u32 = 32;

/// The five non-zero coefficients of an isotropic LTC inverse matrix `M⁻¹`
/// together with the lobe amplitude (directional albedo).
///
/// The coefficients follow the block-sparse layout documented at module level;
/// [`LtcCoeffs::IDENTITY`] is the clamped-cosine passthrough.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LtcCoeffs {
    /// `M⁻¹[0][0]`.
    pub a00: f32,
    /// `M⁻¹[0][2]`.
    pub a02: f32,
    /// `M⁻¹[1][1]`.
    pub a11: f32,
    /// `M⁻¹[2][0]`.
    pub a20: f32,
    /// `M⁻¹[2][2]`.
    pub a22: f32,
    /// Lobe amplitude / directional albedo in `[0, 1]`.
    pub amplitude: f32,
}

impl LtcCoeffs {
    /// Identity transform (clamped-cosine passthrough) with unit amplitude.
    pub const IDENTITY: Self = Self {
        a00: 1.0,
        a02: 0.0,
        a11: 1.0,
        a20: 0.0,
        a22: 1.0,
        amplitude: 1.0,
    };

    /// Returns the coefficients clamped to finite values, falling back to the
    /// identity whenever any entry is non-finite.
    #[inline]
    pub fn sanitized(self) -> Self {
        let fields = [self.a00, self.a02, self.a11, self.a20, self.a22, self.amplitude];
        if fields.iter().any(|f| !f.is_finite()) {
            return Self::IDENTITY;
        }
        Self {
            amplitude: self.amplitude.clamp(0.0, 1.0),
            ..self
        }
    }

    /// Determinant of the stored inverse matrix `M⁻¹`.
    ///
    /// For the block-sparse structure this is `(a00·a22 − a02·a20)·a11`.
    #[inline]
    pub fn determinant(&self) -> f32 {
        (self.a00 * self.a22 - self.a02 * self.a20) * self.a11
    }

    /// Applies the inverse transform `M⁻¹` to a direction, returning
    /// `M⁻¹ · dir` (not renormalised).
    #[inline]
    pub fn apply(&self, dir: Vec3) -> Vec3 {
        Vec3::new(
            self.a00 * dir.x + self.a02 * dir.z,
            self.a11 * dir.y,
            self.a20 * dir.x + self.a22 * dir.z,
        )
    }

    /// Applies `M⁻¹` and renormalises to the unit sphere, returning the zero
    /// vector for a degenerate (zero-length) result.
    #[inline]
    pub fn apply_normalized(&self, dir: Vec3) -> Vec3 {
        self.apply(dir).normalize_or_zero()
    }

    /// Full `Mat3` form of the stored inverse matrix `M⁻¹`.
    #[inline]
    pub fn to_mat3(&self) -> Mat3 {
        // `Mat3::from_cols` takes column vectors; column `j` holds `M⁻¹[·][j]`.
        Mat3::from_cols(
            Vec3::new(self.a00, 0.0, self.a20),
            Vec3::new(0.0, self.a11, 0.0),
            Vec3::new(self.a02, 0.0, self.a22),
        )
    }
}

/// Clamped-cosine distribution value `max(z, 0) / π` for a unit direction.
///
/// This is the canonical LTC base distribution evaluated after a direction has
/// been transformed by `M⁻¹`.  Returns `0` for below-horizon or non-finite
/// input.
#[inline]
pub fn clamped_cosine_pdf(dir: Vec3) -> f32 {
    if !dir.is_finite() {
        return 0.0;
    }
    (dir.z.max(0.0)) / PI
}

/// Evaluates the LTC density for an incident direction `wi` under the inverse
/// transform `coeffs`.
///
/// The density is `D_cos(normalize(M⁻¹·wi)) · |det M⁻¹| / |M⁻¹·wi|³`, the exact
/// change-of-variables of the clamped cosine through the linear transform.
/// Returns `0` for degenerate configurations so the reference never emits
/// `NaN`.
#[inline]
pub fn ltc_pdf(coeffs: &LtcCoeffs, wi: Vec3) -> f32 {
    let transformed = coeffs.apply(wi);
    let len = transformed.length();
    if !(len > 0.0) || !len.is_finite() {
        return 0.0;
    }
    let cos_dist = clamped_cosine_pdf(transformed / len);
    let jacobian = coeffs.determinant().abs() / (len * len * len).max(1.0e-20);
    let d = cos_dist * jacobian;
    if d.is_finite() { d.max(0.0) } else { 0.0 }
}

/// A baked `size × size` table of LTC inverse-matrix coefficients.
///
/// Stored row-major: `roughness` indexes rows, `n·v` indexes columns.
#[derive(Clone, Debug, PartialEq)]
pub struct LtcLut {
    size: u32,
    texels: Vec<LtcCoeffs>,
}

impl LtcLut {
    /// Grid resolution per axis.
    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Immutable row-major view of the stored coefficients.
    #[inline]
    pub fn texels(&self) -> &[LtcCoeffs] {
        &self.texels
    }

    /// Clamp-to-edge row-major index for integer texel `(col, row)`.
    #[inline]
    fn index(&self, col: u32, row: u32) -> usize {
        if self.size == 0 {
            return 0;
        }
        let c = col.min(self.size - 1);
        let r = row.min(self.size - 1);
        (r * self.size + c) as usize
    }

    /// Clamp-to-edge nearest fetch, defaulting to the identity for an empty
    /// table.
    #[inline]
    fn fetch(&self, col: u32, row: u32) -> LtcCoeffs {
        if self.texels.is_empty() {
            return LtcCoeffs::IDENTITY;
        }
        self.texels[self.index(col, row)]
    }

    /// Clamp-to-edge bilinear sample of the LTC coefficients for a view cosine
    /// `n_dot_v` and perceptual `roughness`, both in `[0, 1]`.
    ///
    /// Returns [`LtcCoeffs::IDENTITY`] for an empty table so callers always
    /// receive a finite, invertible transform.
    pub fn sample(&self, n_dot_v: f32, roughness: f32) -> LtcCoeffs {
        if self.size == 0 || self.texels.is_empty() {
            return LtcCoeffs::IDENTITY;
        }
        let s = self.size as f32;
        let fx = (n_dot_v.clamp(0.0, 1.0) * s - 0.5).clamp(0.0, s - 1.0);
        let fy = (roughness.clamp(0.0, 1.0) * s - 0.5).clamp(0.0, s - 1.0);
        let x0 = floor_u32(fx);
        let y0 = floor_u32(fy);
        let x1 = (x0 + 1).min(self.size - 1);
        let y1 = (y0 + 1).min(self.size - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let c00 = self.fetch(x0, y0);
        let c10 = self.fetch(x1, y0);
        let c01 = self.fetch(x0, y1);
        let c11 = self.fetch(x1, y1);

        lerp_coeffs(
            lerp_coeffs(c00, c10, tx),
            lerp_coeffs(c01, c11, tx),
            ty,
        )
        .sanitized()
    }
}

/// Component-wise linear interpolation of two coefficient sets.
#[inline]
fn lerp_coeffs(a: LtcCoeffs, b: LtcCoeffs, t: f32) -> LtcCoeffs {
    let t = t.clamp(0.0, 1.0);
    let l = |x: f32, y: f32| x + (y - x) * t;
    LtcCoeffs {
        a00: l(a.a00, b.a00),
        a02: l(a.a02, b.a02),
        a11: l(a.a11, b.a11),
        a20: l(a.a20, b.a20),
        a22: l(a.a22, b.a22),
        amplitude: l(a.amplitude, b.amplitude),
    }
}

/// Floor of a non-negative `f32` to `u32` via [`bevy_math::ops::floor`].
#[inline]
fn floor_u32(x: f32) -> u32 {
    let f = ops::floor(x.max(0.0));
    if f.is_finite() { f as u32 } else { 0 }
}

/// Fits the LTC inverse-matrix coefficients for a `(n_dot_v, roughness)` pair.
///
/// The GGX lobe is integrated on a `grid × grid` hemisphere lattice; its mean
/// reflected direction fixes the incidence-plane shear while the tangential
/// standard deviations fix the lobe widths.  The forward matrix is normalised
/// to a unit mean-direction column and inverted analytically.  A near-mirror
/// (near-singular) lobe falls back to [`LtcCoeffs::IDENTITY`].
///
/// `grid` is floored at `2`; `n_dot_v` and `roughness` are clamped to `[0, 1]`.
pub fn fit_ltc(n_dot_v: f32, roughness: f32, grid: u32) -> LtcCoeffs {
    let grid = grid.max(2);
    let ndv = n_dot_v.clamp(1.0e-4, 1.0);
    // A near-mirror lobe is a Dirac reflection the clamped cosine cannot model;
    // fall back to the identity transform by convention.
    if roughness.clamp(0.0, 1.0) <= MIRROR_ROUGHNESS {
        return LtcCoeffs::IDENTITY;
    }
    let alpha = roughness_to_alpha(roughness);

    // View direction in the `+X` half of the `XZ` incidence plane.
    let sin_v = (1.0 - ndv * ndv).max(0.0).sqrt();
    let wo = Vec3::new(sin_v, 0.0, ndv);

    // Pass 1: amplitude (directional albedo) and the mean reflected direction.
    let mut amplitude = 0.0f32;
    let mut mean = Vec3::ZERO;
    let inv = 1.0 / grid as f32;
    for i in 0..grid {
        // Stratified `cosθ ∈ (0, 1]`.
        let cos_theta = (i as f32 + 0.5) * inv;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        for j in 0..grid {
            let phi = 2.0 * PI * (j as f32 + 0.5) * inv;
            let (sp, cp) = ops::sin_cos(phi);
            let wi = Vec3::new(sin_theta * cp, sin_theta * sp, cos_theta);
            // Uniform-hemisphere solid-angle weight `dω = (dcosθ)(dφ) = inv·2π·inv`.
            let weight = ggx_brdf_scalar(wo, wi, alpha, alpha, 1.0) * cos_theta;
            amplitude += weight;
            mean += weight * wi;
        }
    }
    // Normalise the Monte-Carlo-style sum by the sample measure.
    let measure = 2.0 * PI * inv * inv;
    amplitude *= measure;
    mean *= measure;

    if !(amplitude > 0.0) || !amplitude.is_finite() || !mean.is_finite() {
        return LtcCoeffs::IDENTITY;
    }

    let avg_dir = mean.normalize_or_zero();
    if avg_dir.length_squared() < 0.5 {
        return LtcCoeffs::IDENTITY;
    }

    // Lobe frame: `avg_dir` plus an in-plane tangent `t1` and out-of-plane `t2`.
    // By isotropy with `wo` in the `XZ` plane the mean lies in `XZ`, so `t2`
    // is `±Y` and `t1` is the in-plane perpendicular.
    let t2 = Vec3::Y;
    let t1 = avg_dir.cross(t2).normalize_or_zero();
    let t1 = if t1.length_squared() < 0.5 { Vec3::X } else { t1 };

    // Pass 2: tangential second moments around the mean direction.
    let mut var1 = 0.0f32;
    let mut var2 = 0.0f32;
    for i in 0..grid {
        let cos_theta = (i as f32 + 0.5) * inv;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        for j in 0..grid {
            let phi = 2.0 * PI * (j as f32 + 0.5) * inv;
            let (sp, cp) = ops::sin_cos(phi);
            let wi = Vec3::new(sin_theta * cp, sin_theta * sp, cos_theta);
            let weight = ggx_brdf_scalar(wo, wi, alpha, alpha, 1.0) * cos_theta;
            let p1 = wi.dot(t1);
            let p2 = wi.dot(t2);
            var1 += weight * p1 * p1;
            var2 += weight * p2 * p2;
        }
    }
    // Normalise by the (already weighted) amplitude to get projected variances.
    let norm = (amplitude / measure).max(1.0e-20);
    let std1 = (var1 / norm).max(0.0).sqrt().max(MIN_SPREAD);
    let std2 = (var2 / norm).max(0.0).sqrt().max(MIN_SPREAD);

    // Scale factors relative to the canonical clamped-cosine spread.
    let scale1 = (std1 / COSINE_STD).max(MIN_SPREAD);
    let scale2 = (std2 / COSINE_STD).max(MIN_SPREAD);

    // Forward matrix columns (unit mean-direction column => scale-normalised):
    //   colX = t1 · scale1,  colY = t2 · scale2,  colZ = avg_dir.
    // With `avg_dir = (ax, 0, az)` and `t1 = (az, 0, -ax)` the forward block is
    //   [ az·s1   0    ax ]
    //   [ 0       s2   0  ]
    //   [ -ax·s1  0    az ]
    let ax = avg_dir.x;
    let az = avg_dir.z;
    let det_block = scale1 * (ax * ax + az * az); // == scale1 (unit avg_dir)
    let det = det_block * scale2;
    if !(det.abs() > MIN_DET) || !det.is_finite() {
        return LtcCoeffs::IDENTITY;
    }

    // Analytic inverse of the block-sparse forward matrix.
    //   XZ block A = [[az·s1, ax], [-ax·s1, az]],  detA = s1·(ax²+az²) = s1.
    //   A⁻¹ = (1/detA) · [[az, -ax], [ax·s1, az·s1]].
    let inv_det_a = 1.0 / det_block;
    let a00 = az * inv_det_a;
    let a02 = -ax * inv_det_a;
    let a20 = ax * scale1 * inv_det_a;
    let a22 = az * scale1 * inv_det_a;
    let a11 = 1.0 / scale2;

    LtcCoeffs {
        a00,
        a02,
        a11,
        a20,
        a22,
        amplitude,
    }
    .sanitized()
}

/// Fits the LTC coefficients at the default hemisphere grid resolution.
#[inline]
pub fn fit_ltc_default(n_dot_v: f32, roughness: f32) -> LtcCoeffs {
    fit_ltc(n_dot_v, roughness, DEFAULT_FIT_GRID)
}

/// Bakes a `size × size` LTC LUT using `grid × grid` hemisphere samples per
/// texel.
///
/// Texel centres map to `(i + 0.5) / size` on both axes so neither `n·v` nor
/// `roughness` is sampled at the exact `0` edge; `size` is floored at `1`.
pub fn bake_ltc_lut(size: u32, grid: u32) -> LtcLut {
    let size = size.max(1);
    let mut texels = Vec::with_capacity((size * size) as usize);
    let inv = 1.0 / size as f32;
    for row in 0..size {
        let roughness = (row as f32 + 0.5) * inv;
        for col in 0..size {
            let n_dot_v = (col as f32 + 0.5) * inv;
            texels.push(fit_ltc(n_dot_v, roughness, grid));
        }
    }
    LtcLut { size, texels }
}

/// Bakes an LTC LUT at [`DEFAULT_SIZE`] / [`DEFAULT_FIT_GRID`].
#[inline]
pub fn bake_ltc_lut_default() -> LtcLut {
    bake_ltc_lut(DEFAULT_SIZE, DEFAULT_FIT_GRID)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_invertible_passthrough() {
        let id = LtcCoeffs::IDENTITY;
        assert!((id.determinant() - 1.0).abs() < 1e-6);
        let d = Vec3::new(0.2, -0.3, 0.9);
        assert!((id.apply(d) - d).length() < 1e-6);
        let m = id.to_mat3();
        assert!((m - Mat3::IDENTITY).abs().to_cols_array().iter().all(|x| *x < 1e-6));
    }

    #[test]
    fn apply_matches_mat3() {
        let c = fit_ltc_default(0.6, 0.4);
        let d = Vec3::new(0.3, 0.5, 0.8);
        let via_apply = c.apply(d);
        let via_mat = c.to_mat3() * d;
        assert!((via_apply - via_mat).length() < 1e-5, "{via_apply:?} {via_mat:?}");
    }

    #[test]
    fn mirror_roughness_falls_back_to_identity() {
        // A near-perfect mirror collapses the fit to the identity.
        let c = fit_ltc_default(1.0, 0.0);
        assert_eq!(c, LtcCoeffs::IDENTITY);
        let c2 = fit_ltc_default(0.8, 0.0);
        assert_eq!(c2, LtcCoeffs::IDENTITY);
    }

    #[test]
    fn fit_is_invertible_and_positive_det() {
        for &ndv in &[0.1f32, 0.3, 0.6, 0.9, 1.0] {
            for &r in &[0.3f32, 0.5, 0.7, 1.0] {
                let c = fit_ltc_default(ndv, r);
                let det = c.determinant();
                assert!(det.is_finite(), "ndv={ndv} r={r} det={det}");
                assert!(det.abs() > 0.0, "ndv={ndv} r={r} det={det}");
                // All coefficients finite.
                for v in [c.a00, c.a02, c.a11, c.a20, c.a22, c.amplitude] {
                    assert!(v.is_finite(), "non-finite coeff ndv={ndv} r={r}");
                }
                assert!((0.0..=1.0).contains(&c.amplitude));
            }
        }
    }

    #[test]
    fn normal_incidence_is_symmetric() {
        // Looking straight down the lobe is symmetric: no incidence-plane shear.
        let c = fit_ltc_default(1.0, 0.6);
        assert!(c.a02.abs() < 1e-3, "a02={}", c.a02);
        assert!(c.a20.abs() < 1e-3, "a20={}", c.a20);
        // X and Y scale identically for an isotropic lobe seen head-on.
        assert!((c.a00 - c.a11).abs() < 5e-2, "a00={} a11={}", c.a00, c.a11);
    }

    #[test]
    fn grazing_introduces_shear() {
        // At grazing angles the mean reflected direction tilts off the normal,
        // introducing non-zero off-diagonal shear.
        let c = fit_ltc_default(0.2, 0.6);
        assert!(c.a02.abs() + c.a20.abs() > 1e-3, "expected shear, got {c:?}");
    }

    #[test]
    fn ltc_pdf_is_finite_and_nonnegative() {
        let c = fit_ltc_default(0.5, 0.5);
        for wi in [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.4, 0.1, 0.9).normalize(),
            Vec3::new(-0.6, 0.2, 0.77).normalize(),
            Vec3::NEG_Z,
        ] {
            let p = ltc_pdf(&c, wi);
            assert!(p.is_finite() && p >= 0.0, "pdf={p} wi={wi:?}");
        }
    }

    #[test]
    fn lut_sampling_is_bilinear_and_clamped() {
        let lut = bake_ltc_lut(8, 12);
        assert_eq!(lut.size(), 8);
        assert_eq!(lut.texels().len(), 64);
        // Out-of-range queries clamp to edge rather than panic / NaN.
        let edge = lut.sample(-1.0, 2.0);
        assert!(edge.determinant().is_finite());
        let mid = lut.sample(0.5, 0.5);
        for v in [mid.a00, mid.a02, mid.a11, mid.a20, mid.a22, mid.amplitude] {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn empty_lut_returns_identity() {
        let lut = LtcLut { size: 0, texels: Vec::new() };
        assert_eq!(lut.sample(0.5, 0.5), LtcCoeffs::IDENTITY);
    }

    #[test]
    fn sanitized_rejects_nan() {
        let bad = LtcCoeffs {
            a00: f32::NAN,
            ..LtcCoeffs::IDENTITY
        };
        assert_eq!(bad.sanitized(), LtcCoeffs::IDENTITY);
    }

    #[test]
    fn amplitude_is_bounded_directional_albedo() {
        // The fitted amplitude is a directional albedo: finite and within
        // `[0, 1]` across the view/roughness domain, and substantial for a
        // rough reflector.
        for &ndv in &[0.15f32, 0.5, 0.95] {
            for &r in &[0.4f32, 0.8, 1.0] {
                let a = fit_ltc_default(ndv, r).amplitude;
                assert!(a.is_finite() && (0.0..=1.0).contains(&a), "ndv={ndv} r={r} a={a}");
            }
        }
        assert!(fit_ltc_default(0.5, 0.8).amplitude > 0.1);
    }
}
