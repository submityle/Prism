//! Split-sum environment-BRDF LUT (the "DFG" / scale-bias table).
//!
//! Karis' split-sum approximation factors the pre-integrated specular
//! environment response into a *lighting* term (the pre-filtered radiance) and
//! a *BRDF* term that depends only on the view cosine `µ = n·v` and the surface
//! `roughness`. This module bakes and samples the BRDF term.
//!
//! Writing the Schlick Fresnel as `F(c) = F0·(1 - (1-c)^5) + (1-c)^5` and
//! integrating the single-scattering GGX reflectance over the hemisphere splits
//! the environment BRDF into two scalars:
//!
//! ```text
//! ∫ f_spec·F·cosθ dω = F0 · scale + bias
//! scale = ∫ (f_spec/F)·(1 - (1 - v·h)^5)·cosθ dω
//! bias  = ∫ (f_spec/F)·     (1 - v·h)^5 ·cosθ dω
//! ```
//!
//! Both integrals are estimated by importance sampling the GGX *visible*
//! normal distribution (VNDF). With VNDF sampling the single-sample Monte-Carlo
//! weight of the Fresnel-free specular lobe collapses to the masking ratio
//! `G2 / G1(v)` (Heitz 2018), so each sample only needs to split that weight by
//! the Fresnel polynomial. The GGX primitives (`roughness_to_alpha`,
//! `smith_g1`, `smith_g2`, `sample_ggx_vndf`, …) are reused from
//! [`crate::gi::spec_gi::ggx_lobe`]; this module never re-derives GGX.
//!
//! # Conventions
//! * The view direction lives in a local frame with the surface normal at `+Z`.
//!   For a given `µ = n·v ∈ (0, 1]` the view is placed in the `x–z` plane as
//!   `wo = (sqrt(1-µ²), 0, µ)`.
//! * `scale` and `bias` are both in `[0, 1]`; `scale + bias` (= the white-Fresnel
//!   directional albedo `E(µ)`) never exceeds `1`.
//! * The LUT is a square grid stored row-major with `roughness` on the outer
//!   (row) axis and `µ` on the inner (column) axis. Texel centres map to
//!   `(i + 0.5) / size`; sampling is clamp-to-edge bilinear.
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent method. Every routine is a deterministic pure function (no RNG,
//!   I/O, GPU, globals or `unsafe`) and defends against degeneracy so a result
//!   is always finite and never `NaN`.
//!
//! # References
//! * Karis 2013, *Real Shading in Unreal Engine 4* — the split-sum DFG LUT.
//! * Heitz 2018, *Sampling the GGX Distribution of Visible Normals* — the VNDF
//!   weight `G2 / G1`.

use alloc::vec::Vec;
use bevy_math::{Vec2, Vec3};

use crate::gi::spec_gi::ggx_lobe::{roughness_to_alpha, sample_ggx_vndf, smith_g1, smith_g2};

/// Smallest view cosine used when baking so the local view never degenerates to
/// the exact horizon (where the frame and the pdf become ill-defined).
pub const MIN_COS: f32 = 1.0e-3;

/// Default sample count per texel used by [`bake_dfg_lut_default`].
pub const DEFAULT_SAMPLES: u32 = 1024;

/// Default LUT resolution per axis used by [`bake_dfg_lut_default`].
pub const DEFAULT_SIZE: u32 = 64;

/// Builds the local view direction for a view cosine `µ = n·v`.
///
/// Places the view in the `x–z` plane as `(sqrt(1-µ²), 0, µ)` with `µ` clamped
/// to `[MIN_COS, 1]`, matching the LUT parameterization.
#[inline]
fn view_from_cos(n_dot_v: f32) -> Vec3 {
    let mu = n_dot_v.clamp(MIN_COS, 1.0);
    let sin_v = (1.0 - mu * mu).max(0.0).sqrt();
    Vec3::new(sin_v, 0.0, mu)
}

/// Reflects the local view `wo` about a half vector `h` (both unit, `+Z` up).
#[inline]
fn reflect_about(wo: Vec3, h: Vec3) -> Vec3 {
    (2.0 * wo.dot(h) * h - wo).normalize_or_zero()
}

/// Integrates the split-sum environment BRDF `(scale, bias)` at one
/// `(n_dot_v, roughness)` cell using `samples` VNDF draws.
///
/// Returns a [`Vec2`] whose `x` is the `scale` (the `F0` coefficient) and whose
/// `y` is the `bias` (the additive term). Both components are clamped to
/// `[0, 1]`. `samples` is floored at `1`. A low-discrepancy stratified set is
/// generated internally so the result is deterministic for fixed inputs.
///
/// With VNDF sampling the Fresnel-free weight of a reflected sample is
/// `G2(v,l) / G1(v)`; this routine splits that weight by the Schlick Fresnel
/// polynomial `(1 - v·h)^5` into the `scale` and `bias` accumulators.
pub fn integrate_dfg(n_dot_v: f32, roughness: f32, samples: u32) -> Vec2 {
    let n = samples.max(1);
    let alpha = roughness_to_alpha(roughness);
    let wo = view_from_cos(n_dot_v);
    let g1_v = smith_g1(wo.z, alpha).max(1.0e-6);

    // Correlated multi-jittered grid over the unit square for low variance.
    let strata = isqrt_u32(n).max(1);
    let mut scale = 0.0f64;
    let mut bias = 0.0f64;
    let mut taken = 0u32;

    for s in 0..n {
        let (u1, u2) = stratified_unit(s, strata);
        let h = sample_ggx_vndf(wo, alpha, alpha, u1, u2);
        let wi = reflect_about(wo, h);
        if wi.z <= 0.0 || h.length_squared() <= 0.0 {
            continue;
        }
        let v_dot_h = wo.dot(h).clamp(0.0, 1.0);
        let g2 = smith_g2(wo.z, wi.z, alpha);
        // Fresnel-free VNDF weight: f·cos / pdf = G2 / G1(v).
        let weight = (g2 / g1_v) as f64;
        if !weight.is_finite() {
            continue;
        }
        let fc = pow5(1.0 - v_dot_h) as f64;
        scale += (1.0 - fc) * weight;
        bias += fc * weight;
        taken += 1;
    }

    if taken == 0 {
        // Degenerate configuration (e.g. grazing view): behave like a mirror
        // seen head-on so the table stays finite and energy-bounded.
        return Vec2::new(1.0, 0.0);
    }

    let inv = 1.0 / n as f64;
    let scale = (scale * inv) as f32;
    let bias = (bias * inv) as f32;
    sanitize_scale_bias(scale, bias)
}

/// A baked square `(roughness, µ)` split-sum DFG table.
///
/// Stored row-major: `roughness` indexes rows, `µ = n·v` indexes columns. Each
/// texel holds `(scale, bias)` as a [`Vec2`].
#[derive(Clone, Debug, PartialEq)]
pub struct DfgLut {
    size: u32,
    texels: Vec<Vec2>,
}

impl DfgLut {
    /// Grid resolution per axis.
    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Immutable view of the row-major `(scale, bias)` texels.
    #[inline]
    pub fn texels(&self) -> &[Vec2] {
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

    /// Clamp-to-edge nearest fetch of integer texel `(col, row)`.
    #[inline]
    fn fetch(&self, col: u32, row: u32) -> Vec2 {
        if self.texels.is_empty() {
            return Vec2::new(1.0, 0.0);
        }
        self.texels[self.index(col, row)]
    }

    /// Clamp-to-edge bilinear sample of the `(scale, bias)` pair for a view
    /// cosine `n_dot_v` and perceptual `roughness`, both in `[0, 1]`.
    ///
    /// Returns `(1, 0)` (mirror, lossless) for an empty table so callers always
    /// receive a finite, energy-bounded result.
    pub fn sample(&self, n_dot_v: f32, roughness: f32) -> Vec2 {
        if self.size == 0 || self.texels.is_empty() {
            return Vec2::new(1.0, 0.0);
        }
        let s = self.size as f32;
        let fx = (n_dot_v.clamp(0.0, 1.0) * s - 0.5).clamp(0.0, s - 1.0);
        let fy = (roughness.clamp(0.0, 1.0) * s - 0.5).clamp(0.0, s - 1.0);
        let x0 = ops_floor_u32(fx);
        let y0 = ops_floor_u32(fy);
        let x1 = (x0 + 1).min(self.size - 1);
        let y1 = (y0 + 1).min(self.size - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let c00 = self.fetch(x0, y0);
        let c10 = self.fetch(x1, y0);
        let c01 = self.fetch(x0, y1);
        let c11 = self.fetch(x1, y1);
        let top = c00.lerp(c10, tx);
        let bot = c01.lerp(c11, tx);
        let v = top.lerp(bot, ty);
        sanitize_scale_bias(v.x, v.y)
    }

    /// Evaluates the pre-integrated environment BRDF `F0·scale + bias` for an
    /// RGB `f0` at the given `(n_dot_v, roughness)`.
    ///
    /// `f0` is clamped per channel to `[0, 1]`; the result is non-negative and
    /// finite.
    #[inline]
    pub fn evaluate(&self, f0: Vec3, n_dot_v: f32, roughness: f32) -> Vec3 {
        let sb = self.sample(n_dot_v, roughness);
        env_brdf(f0, sb)
    }
}

/// Bakes a `size × size` DFG LUT with `samples` VNDF draws per texel.
///
/// `size` is floored at `1`. Texel centres map to `(i + 0.5) / size` on both
/// axes, so neither `µ` nor `roughness` is ever sampled at the exact `0` edge.
pub fn bake_dfg_lut(size: u32, samples: u32) -> DfgLut {
    let size = size.max(1);
    let mut texels = Vec::with_capacity((size * size) as usize);
    let inv = 1.0 / size as f32;
    for row in 0..size {
        let roughness = (row as f32 + 0.5) * inv;
        for col in 0..size {
            let n_dot_v = (col as f32 + 0.5) * inv;
            texels.push(integrate_dfg(n_dot_v, roughness, samples));
        }
    }
    DfgLut { size, texels }
}

/// Bakes a DFG LUT at [`DEFAULT_SIZE`] / [`DEFAULT_SAMPLES`].
#[inline]
pub fn bake_dfg_lut_default() -> DfgLut {
    bake_dfg_lut(DEFAULT_SIZE, DEFAULT_SAMPLES)
}

/// Combines a split-sum `(scale, bias)` pair with an RGB `f0` into the
/// pre-integrated environment BRDF `F0·scale + bias`.
///
/// `f0` is clamped per channel to `[0, 1]`; `scale`/`bias` are clamped to
/// `[0, 1]`. The result is non-negative and finite.
#[inline]
pub fn env_brdf(f0: Vec3, scale_bias: Vec2) -> Vec3 {
    let sb = sanitize_scale_bias(scale_bias.x, scale_bias.y);
    let f0 = f0.clamp(Vec3::ZERO, Vec3::ONE);
    let v = f0 * sb.x + Vec3::splat(sb.y);
    if v.is_finite() { v.max(Vec3::ZERO) } else { Vec3::ZERO }
}

/// Clamps a `(scale, bias)` pair so both lie in `[0, 1]`, `scale + bias ≤ 1`
/// (energy conservation of the white-Fresnel directional albedo), and neither
/// is `NaN`/infinite.
#[inline]
fn sanitize_scale_bias(scale: f32, bias: f32) -> Vec2 {
    let s = if scale.is_finite() { scale.clamp(0.0, 1.0) } else { 0.0 };
    let b = if bias.is_finite() { bias.clamp(0.0, 1.0) } else { 0.0 };
    // The white-Fresnel directional albedo scale + bias cannot exceed 1; trim
    // the bias first so a lossless mirror (scale → 1) is preserved.
    let b = b.min((1.0 - s).max(0.0));
    Vec2::new(s, b)
}

/// `x^5` via three multiplies (avoids `powf` for the Schlick polynomial).
#[inline]
fn pow5(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let x2 = x * x;
    x2 * x2 * x
}

/// Integer floor of a non-negative `f32` as `u32` (saturating).
#[inline]
fn ops_floor_u32(x: f32) -> u32 {
    let f = bevy_math::ops::floor(x.max(0.0));
    if f.is_finite() { f as u32 } else { 0 }
}

/// Integer square root of `n` (largest `r` with `r² ≤ n`).
#[inline]
fn isqrt_u32(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut r = (n as f32).sqrt() as u32;
    while r.saturating_mul(r) > n {
        r -= 1;
    }
    while (r + 1).saturating_mul(r + 1) <= n {
        r += 1;
    }
    r
}

/// Correlated multi-jittered style unit-square point for sample `s` on a
/// `strata × strata` grid; falls back to a Hammersley-like pair past the grid.
#[inline]
fn stratified_unit(s: u32, strata: u32) -> (f32, f32) {
    let cells = strata.saturating_mul(strata).max(1);
    let i = s % cells;
    let sx = i % strata;
    let sy = i / strata;
    // Deterministic sub-cell jitter from a cheap integer hash of the sample.
    let (jx, jy) = hash_jitter(s);
    let inv = 1.0 / strata as f32;
    let u1 = (sx as f32 + jx) * inv;
    let u2 = (sy as f32 + jy) * inv;
    (u1.clamp(0.0, 1.0 - 1.0e-6), u2.clamp(0.0, 1.0 - 1.0e-6))
}

/// Two decorrelated jitters in `[0, 1)` from a 32-bit integer hash of `s`.
#[inline]
fn hash_jitter(s: u32) -> (f32, f32) {
    let mut a = s.wrapping_mul(0x9e37_79b9).wrapping_add(0x85eb_ca6b);
    a ^= a >> 15;
    a = a.wrapping_mul(0x2545_f491);
    a ^= a >> 13;
    let b = a.wrapping_mul(0x27d4_eb2f) ^ 0xd3a2_646c;
    let jx = (a >> 8) as f32 * (1.0 / (1u32 << 24) as f32);
    let jy = (b >> 8) as f32 * (1.0 / (1u32 << 24) as f32);
    (jx, jy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_bias_in_unit_range_and_energy_bounded() {
        for &r in &[0.05f32, 0.2, 0.5, 0.8, 1.0] {
            for &mu in &[0.08f32, 0.25, 0.5, 0.75, 1.0] {
                let sb = integrate_dfg(mu, r, 2048);
                assert!((0.0..=1.0).contains(&sb.x), "scale={} r={r} mu={mu}", sb.x);
                assert!((0.0..=1.0).contains(&sb.y), "bias={} r={r} mu={mu}", sb.y);
                assert!(
                    sb.x + sb.y <= 1.0 + 1.0e-3,
                    "scale+bias={} must stay <=1 (r={r} mu={mu})",
                    sb.x + sb.y
                );
            }
        }
    }

    #[test]
    fn mirror_limit_scale_to_one_bias_to_zero() {
        // At near-zero roughness and head-on view the lobe is a sharp mirror:
        // scale -> 1, bias -> 0.
        let sb = integrate_dfg(1.0, 0.0, 4096);
        assert!(sb.x > 0.98, "scale={} should approach 1", sb.x);
        assert!(sb.y < 0.02, "bias={} should approach 0", sb.y);
    }

    #[test]
    fn white_fresnel_albedo_decreases_with_roughness() {
        // E(µ) = scale + bias for F0 = 1 is highest for a smooth surface and
        // loses energy (to the uncomputed multiscatter) as roughness grows.
        let mu = 0.6f32;
        let smooth = integrate_dfg(mu, 0.1, 4096);
        let rough = integrate_dfg(mu, 0.9, 4096);
        let e_smooth = smooth.x + smooth.y;
        let e_rough = rough.x + rough.y;
        assert!(
            e_rough <= e_smooth + 1.0e-2,
            "E(rough)={e_rough} should not exceed E(smooth)={e_smooth}"
        );
        assert!(e_rough < 1.0, "rough albedo={e_rough} should lose energy");
    }

    #[test]
    fn baked_lut_matches_point_integration() {
        let lut = bake_dfg_lut(32, 1024);
        // Sample at a texel centre so bilinear filtering returns that texel.
        let col = 20u32;
        let row = 12u32;
        let mu = (col as f32 + 0.5) / 32.0;
        let r = (row as f32 + 0.5) / 32.0;
        let sampled = lut.sample(mu, r);
        let reference = integrate_dfg(mu, r, 1024);
        assert!((sampled.x - reference.x).abs() < 1.0e-6, "scale mismatch");
        assert!((sampled.y - reference.y).abs() < 1.0e-6, "bias mismatch");
    }

    #[test]
    fn bilinear_sampling_is_bounded_and_monotone_edges() {
        let lut = bake_dfg_lut(32, 512);
        for i in 0..=10u32 {
            for j in 0..=10u32 {
                let mu = i as f32 / 10.0;
                let r = j as f32 / 10.0;
                let sb = lut.sample(mu, r);
                assert!(sb.x.is_finite() && sb.y.is_finite());
                assert!((0.0..=1.0).contains(&sb.x));
                assert!((0.0..=1.0).contains(&sb.y));
                assert!(sb.x + sb.y <= 1.0 + 1.0e-3);
            }
        }
    }

    #[test]
    fn evaluate_blends_f0_between_scale_and_bias() {
        let lut = bake_dfg_lut(32, 512);
        let mu = 0.5f32;
        let r = 0.3f32;
        let sb = lut.sample(mu, r);
        let white = lut.evaluate(Vec3::ONE, mu, r);
        let expected = Vec3::splat(sb.x + sb.y);
        assert!((white - expected).length() < 1.0e-6);
        // A black conductor (F0 = 0) keeps only the bias term.
        let black = lut.evaluate(Vec3::ZERO, mu, r);
        assert!((black - Vec3::splat(sb.y)).length() < 1.0e-6);
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        let sb = integrate_dfg(0.0, 0.0, 0);
        assert!(sb.x.is_finite() && sb.y.is_finite());
        let empty = DfgLut { size: 0, texels: Vec::new() };
        let sb = empty.sample(0.5, 0.5);
        assert_eq!(sb, Vec2::new(1.0, 0.0));
        let v = env_brdf(Vec3::splat(f32::NAN), Vec2::new(f32::INFINITY, -1.0));
        assert!(v.is_finite());
    }

    #[test]
    fn env_brdf_clamps_scale_bias_sum() {
        // Even with an out-of-range table entry the directional albedo is kept
        // energy-bounded.
        let v = env_brdf(Vec3::ONE, Vec2::new(0.9, 0.9));
        assert!(v.x <= 1.0 + 1.0e-6, "white albedo={} must be <=1", v.x);
    }
}
