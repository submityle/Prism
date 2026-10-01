//! Sheen (cloth) and clearcoat environment-BRDF references.
//!
//! Two secondary specular lobes layered on top of the base GGX response:
//!
//! * **Sheen** — the retro-reflective rim seen on cloth and velvet. It uses the
//!   Estevez-Kulla "Charlie" sheen distribution (Sony Imageworks 2017) together
//!   with the fitted soft-shadowing visibility term, and a pre-integrated
//!   directional-albedo LUT `E_sheen(µ, roughness)` so the lobe can be scaled to
//!   conserve energy (the glTF `KHR_materials_sheen` "sheen env BRDF").
//! * **Clearcoat** — a thin, smooth dielectric layer with a fixed IOR of `1.5`
//!   (so `F0 = 0.04`). Its environment response is the ordinary GGX split-sum
//!   `F0·scale + bias` evaluated at the clearcoat roughness; the DFG integration
//!   is reused from [`crate::gi::env_brdf::dfg_lut`].
//!
//! # Conventions
//! * Directions live in a local frame with the normal at `+Z`; a cosine is a
//!   direction's `z` component. `µ = n·v`, `µ_l = n·l`.
//! * Sheen `roughness ∈ [0, 1]` maps directly to the Charlie width `α`
//!   (perceptually linear, *not* squared) following Imageworks/glTF, floored at
//!   [`MIN_SHEEN_ROUGHNESS`].
//! * `E_sheen` and the clearcoat env term lie in `[0, 1]`.
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent method. Every routine is a deterministic pure function (no RNG,
//!   I/O, GPU, globals or `unsafe`) and never emits `NaN`.
//!
//! # References
//! * Estevez & Kulla 2017, *Production Friendly Microfacet Sheen BRDF*.
//! * Karis 2013, *Real Shading in Unreal Engine 4* — the split-sum clearcoat.
//! * Khronos `KHR_materials_sheen` / `KHR_materials_clearcoat`.

use alloc::vec::Vec;
use bevy_math::{Vec3, ops};
use core::f32::consts::{PI, TAU};

use crate::gi::env_brdf::dfg_lut::integrate_dfg;

/// Fixed clearcoat Fresnel reflectance at normal incidence (IOR 1.5).
pub const CLEARCOAT_F0: f32 = 0.04;

/// Smallest sheen roughness so the Charlie exponent `1/α` stays finite.
pub const MIN_SHEEN_ROUGHNESS: f32 = 1.0e-3;

/// Default sheen-LUT resolution per axis.
pub const DEFAULT_SHEEN_SIZE: u32 = 32;

/// Default hemisphere sample count per sheen-LUT texel.
pub const DEFAULT_SHEEN_SAMPLES: u32 = 2048;

/// Estevez-Kulla "Charlie" sheen normal-distribution function.
///
/// `D(h) = (2 + 1/α) · sinθ_h^{1/α} / (2π)` with `cosθ_h = n·h`. Returns `0` for
/// a back-facing half vector; `roughness` (→ `α`) is floored at
/// [`MIN_SHEEN_ROUGHNESS`]. The distribution is normalised so
/// `∫ D(h) cosθ_h dω = 1`.
#[inline]
pub fn charlie_ndf(n_dot_h: f32, roughness: f32) -> f32 {
    let cos_h = n_dot_h.clamp(-1.0, 1.0);
    if cos_h <= 0.0 {
        return 0.0;
    }
    let alpha = roughness.clamp(MIN_SHEEN_ROUGHNESS, 1.0);
    let inv_alpha = 1.0 / alpha;
    let sin2 = (1.0 - cos_h * cos_h).max(0.0);
    let sin_h = sin2.sqrt();
    // sinθ^{1/α}; guard the base so `powf(0, …)` stays 0 (not NaN).
    let pow = if sin_h <= 0.0 { 0.0 } else { ops::powf(sin_h, inv_alpha) };
    let d = (2.0 + inv_alpha) * pow / TAU;
    if d.is_finite() { d.max(0.0) } else { 0.0 }
}

/// Fitted helper for the Estevez-Kulla Charlie soft-shadowing `Λ` term.
///
/// Interpolates the published coefficients between the `α = 1` and `α = 0`
/// fits by `(1 - α)²` and evaluates `a / (1 + b·x^c) + d·x + e`.
#[inline]
fn lambda_sheen_helper(x: f32, alpha: f32) -> f32 {
    let one_minus = (1.0 - alpha).clamp(0.0, 1.0);
    let t = one_minus * one_minus;
    let a = lerp(21.5473, 25.3245, t);
    let b = lerp(3.829_87, 3.324_35, t);
    let c = lerp(0.198_23, 0.168_01, t);
    let d = lerp(-1.977_60, -1.273_93, t);
    let e = lerp(-4.320_54, -4.859_67, t);
    let xc = ops::powf(x.max(0.0), c);
    let v = a / (1.0 + b * xc) + d * x + e;
    if v.is_finite() { v } else { 0.0 }
}

/// Estevez-Kulla Charlie soft-shadowing `Λ(cosθ)` with the published
/// symmetric split at `cosθ = 0.5`.
#[inline]
fn lambda_sheen(cos_theta: f32, alpha: f32) -> f32 {
    let c = cos_theta.clamp(0.0, 1.0);
    let l = if c < 0.5 {
        ops::exp(lambda_sheen_helper(c, alpha))
    } else {
        ops::exp(2.0 * lambda_sheen_helper(0.5, alpha) - lambda_sheen_helper(1.0 - c, alpha))
    };
    if l.is_finite() { l.max(0.0) } else { 0.0 }
}

/// Estevez-Kulla Charlie visibility term
/// `V = 1 / ((1 + Λ(µ) + Λ(µ_l)) · 4·µ·µ_l)` (soft-shadowing variant).
///
/// Combines the fitted shadowing `Λ` with the `1/(4 µ µ_l)` BRDF denominator so
/// the caller multiplies this directly by [`charlie_ndf`]. Returns `0` for a
/// below-horizon configuration. Both cosines are floored away from `0` to keep
/// the denominator finite.
#[inline]
pub fn charlie_visibility(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
        return 0.0;
    }
    let alpha = roughness.clamp(MIN_SHEEN_ROUGHNESS, 1.0);
    let mu = n_dot_v.clamp(1.0e-4, 1.0);
    let mu_l = n_dot_l.clamp(1.0e-4, 1.0);
    let shadow = 1.0 + lambda_sheen(mu, alpha) + lambda_sheen(mu_l, alpha);
    let denom = shadow * 4.0 * mu * mu_l;
    if denom <= 1.0e-8 {
        return 0.0;
    }
    let v = 1.0 / denom;
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

/// Fresnel-free Charlie sheen BRDF value `D·V` for local `wo`, `wi`.
///
/// This excludes the sheen colour (applied by the caller) and the `n·l` cosine.
/// Returns `0` for below-horizon directions.
#[inline]
pub fn sheen_brdf(wo: Vec3, wi: Vec3, roughness: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let h = (wo + wi).normalize_or_zero();
    if h.length_squared() <= 0.0 {
        return 0.0;
    }
    let d = charlie_ndf(h.z, roughness);
    let v = charlie_visibility(wo.z, wi.z, roughness);
    let f = d * v;
    if f.is_finite() { f.max(0.0) } else { 0.0 }
}

/// Single-scattering sheen directional albedo
/// `E_sheen(µ) = ∫ D·V·cosθ_l dω_l` at view cosine `n_dot_v` and `roughness`.
///
/// Estimated with a uniform-hemisphere Fibonacci lattice (pdf `1/2π`), so the
/// estimator is `(2π/N) Σ sheen_brdf·cosθ_l`. The sheen lobe is not
/// energy-conserving by construction; the result is clamped to `[0, 1]` and is
/// used to scale the lobe so it never adds net energy. `samples` is floored at
/// `1`.
pub fn sheen_directional_albedo(n_dot_v: f32, roughness: f32, samples: u32) -> f32 {
    let n = samples.max(1);
    let mu = n_dot_v.clamp(1.0e-3, 1.0);
    let sin_v = (1.0 - mu * mu).max(0.0).sqrt();
    let wo = Vec3::new(sin_v, 0.0, mu);

    // Fibonacci lattice on the upper hemisphere: uniform in solid angle.
    let golden = PI * (3.0 - (5.0f32).sqrt()); // golden angle
    let mut acc = 0.0f64;
    for i in 0..n {
        // cosθ uniform in (0, 1] → uniform-hemisphere direction.
        let cos_t = (i as f32 + 0.5) / n as f32;
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        let phi = golden * i as f32;
        let (sp, cp) = ops::sin_cos(phi);
        let wi = Vec3::new(sin_t * cp, sin_t * sp, cos_t);
        let f = sheen_brdf(wo, wi, roughness);
        acc += (f * wi.z) as f64;
    }
    // Uniform-hemisphere Monte-Carlo weight: domain measure 2π over N samples.
    let e = (acc / n as f64 * TAU as f64) as f32;
    e.clamp(0.0, 1.0)
}

/// A baked square `(roughness, µ)` sheen directional-albedo table.
///
/// Stored row-major: `roughness` indexes rows, `µ = n·v` indexes columns. Each
/// texel is the scalar `E_sheen` used to energy-scale the sheen lobe.
#[derive(Clone, Debug, PartialEq)]
pub struct SheenLut {
    size: u32,
    texels: Vec<f32>,
}

impl SheenLut {
    /// Grid resolution per axis.
    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Immutable view of the row-major `E_sheen` texels.
    #[inline]
    pub fn texels(&self) -> &[f32] {
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
    fn fetch(&self, col: u32, row: u32) -> f32 {
        if self.texels.is_empty() {
            return 0.0;
        }
        self.texels[self.index(col, row)]
    }

    /// Clamp-to-edge bilinear sample of `E_sheen` for `n_dot_v` and `roughness`
    /// (both in `[0, 1]`). Returns `0` for an empty table.
    pub fn sample(&self, n_dot_v: f32, roughness: f32) -> f32 {
        if self.size == 0 || self.texels.is_empty() {
            return 0.0;
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
        let top = lerp(self.fetch(x0, y0), self.fetch(x1, y0), tx);
        let bot = lerp(self.fetch(x0, y1), self.fetch(x1, y1), tx);
        let v = lerp(top, bot, ty);
        v.clamp(0.0, 1.0)
    }
}

/// Bakes a `size × size` sheen directional-albedo LUT with `samples`
/// hemisphere draws per texel. Texel centres map to `(i + 0.5) / size`.
pub fn bake_sheen_lut(size: u32, samples: u32) -> SheenLut {
    let size = size.max(1);
    let mut texels = Vec::with_capacity((size * size) as usize);
    let inv = 1.0 / size as f32;
    for row in 0..size {
        let roughness = (row as f32 + 0.5) * inv;
        for col in 0..size {
            let n_dot_v = (col as f32 + 0.5) * inv;
            texels.push(sheen_directional_albedo(n_dot_v, roughness, samples));
        }
    }
    SheenLut { size, texels }
}

/// Bakes a sheen LUT at the default resolution / sample count.
#[inline]
pub fn bake_sheen_lut_default() -> SheenLut {
    bake_sheen_lut(DEFAULT_SHEEN_SIZE, DEFAULT_SHEEN_SAMPLES)
}

/// Schlick clearcoat Fresnel reflectance at view cosine `n_dot_v` using the
/// fixed [`CLEARCOAT_F0`].
///
/// `F = F0 + (1 - F0)(1 - µ)^5`, with `µ` clamped to `[0, 1]`; result in
/// `[F0, 1]`.
#[inline]
pub fn clearcoat_fresnel(n_dot_v: f32) -> f32 {
    let mu = n_dot_v.clamp(0.0, 1.0);
    let f = CLEARCOAT_F0 + (1.0 - CLEARCOAT_F0) * pow5(1.0 - mu);
    f.clamp(CLEARCOAT_F0, 1.0)
}

/// Pre-integrated clearcoat environment reflectance `F0·scale + bias` for the
/// fixed IOR-1.5 layer at view cosine `n_dot_v` and clearcoat `roughness`.
///
/// Reuses the GGX split-sum DFG integration with scalar `F0 = 0.04`. The result
/// is in `[0, 1]`. `samples` is floored at `1` inside the DFG integrator.
#[inline]
pub fn clearcoat_env_brdf(n_dot_v: f32, roughness: f32, samples: u32) -> f32 {
    let sb = integrate_dfg(n_dot_v, roughness, samples);
    let v = CLEARCOAT_F0 * sb.x + sb.y;
    v.clamp(0.0, 1.0)
}

/// Linear interpolation `a + (b - a)·t` with `t` unclamped (callers pass
/// in-range weights).
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// `x^5` via three multiplies (Schlick polynomial), with the base clamped to
/// `[0, 1]`.
#[inline]
fn pow5(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let x2 = x * x;
    x2 * x2 * x
}

/// Integer floor of a non-negative `f32` as `u32` (saturating).
#[inline]
fn floor_u32(x: f32) -> u32 {
    let f = ops::floor(x.max(0.0));
    if f.is_finite() { f as u32 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charlie_ndf_normalises_over_hemisphere() {
        // ∫ D(h) cosθ dω = 1 over the hemisphere (Fibonacci lattice, uniform).
        for &r in &[0.3f32, 0.6, 1.0] {
            let n = 60_000usize;
            let golden = PI * (3.0 - (5.0f32).sqrt());
            let mut acc = 0.0f64;
            for i in 0..n {
                let cos_t = (i as f32 + 0.5) / n as f32;
                let d = charlie_ndf(cos_t, r);
                acc += (d * cos_t) as f64;
                let _ = golden; // azimuth is irrelevant for an isotropic D.
            }
            let integral = (acc / n as f64 * TAU as f64) as f32;
            assert!(
                (integral - 1.0).abs() < 0.03,
                "charlie D integral={integral} r={r}"
            );
        }
    }

    #[test]
    fn charlie_ndf_backface_is_zero() {
        assert_eq!(charlie_ndf(0.0, 0.5), 0.0);
        assert_eq!(charlie_ndf(-0.4, 0.5), 0.0);
    }

    #[test]
    fn sheen_visibility_positive_and_finite() {
        for &r in &[0.2f32, 0.5, 1.0] {
            for &c in &[0.05f32, 0.3, 0.7, 1.0] {
                let v = charlie_visibility(c, 0.6, r);
                assert!(v.is_finite() && v >= 0.0, "V={v} c={c} r={r}");
            }
        }
        assert_eq!(charlie_visibility(-0.1, 0.5, 0.5), 0.0);
    }

    #[test]
    fn sheen_brdf_reciprocal() {
        let a = Vec3::new(0.2, 0.1, 0.974).normalize();
        let b = Vec3::new(-0.3, 0.25, 0.92).normalize();
        let fab = sheen_brdf(a, b, 0.4);
        let fba = sheen_brdf(b, a, 0.4);
        assert!((fab - fba).abs() < 1.0e-6, "fab={fab} fba={fba}");
    }

    #[test]
    fn sheen_albedo_in_unit_range() {
        for &r in &[0.1f32, 0.5, 1.0] {
            for &mu in &[0.1f32, 0.5, 0.9] {
                let e = sheen_directional_albedo(mu, r, 4096);
                assert!((0.0..=1.0).contains(&e), "E_sheen={e} r={r} mu={mu}");
            }
        }
    }

    #[test]
    fn sheen_lut_matches_point_integration() {
        let lut = bake_sheen_lut(16, 4096);
        let col = 9u32;
        let row = 6u32;
        let mu = (col as f32 + 0.5) / 16.0;
        let r = (row as f32 + 0.5) / 16.0;
        let sampled = lut.sample(mu, r);
        let reference = sheen_directional_albedo(mu, r, 4096);
        assert!(
            (sampled - reference).abs() < 1.0e-6,
            "sampled={sampled} reference={reference}"
        );
    }

    #[test]
    fn clearcoat_fresnel_endpoints() {
        assert!((clearcoat_fresnel(1.0) - CLEARCOAT_F0).abs() < 1.0e-6);
        assert!((clearcoat_fresnel(0.0) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn clearcoat_env_bounds_and_mirror_limit() {
        for &r in &[0.05f32, 0.3, 0.7, 1.0] {
            for &mu in &[0.1f32, 0.5, 1.0] {
                let e = clearcoat_env_brdf(mu, r, 2048);
                assert!((0.0..=1.0).contains(&e), "cc env={e} r={r} mu={mu}");
            }
        }
        // A near-mirror clearcoat at head-on reflects ≈ F0.
        let e = clearcoat_env_brdf(1.0, 0.02, 4096);
        assert!((e - CLEARCOAT_F0).abs() < 0.02, "cc mirror env={e}");
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        assert_eq!(charlie_ndf(f32::NAN, 0.5), 0.0);
        assert_eq!(sheen_brdf(Vec3::NEG_Z, Vec3::Z, 0.5), 0.0);
        let empty = SheenLut { size: 0, texels: Vec::new() };
        assert_eq!(empty.sample(0.5, 0.5), 0.0);
        assert!(sheen_directional_albedo(0.0, 0.0, 0).is_finite());
        assert!(clearcoat_env_brdf(0.0, 0.0, 0).is_finite());
    }
}
