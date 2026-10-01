//! Kulla-Conty multiple-scattering energy compensation for GGX specular.
//!
//! The single-scattering GGX BRDF loses energy as roughness grows: light that
//! bounces more than once between microfacets is never accounted for, which
//! darkens rough metals and dielectrics. Kulla & Conty (2017) model the missing
//! energy with an added, perfectly diffuse-like lobe parameterised entirely by
//! the single-scattering *directional albedo* `E(µ)` and its cosine-weighted
//! hemispherical average `E_avg`.
//!
//! Two complementary formulations are provided:
//!
//! * the Kulla-Conty achromatic multiple-scattering BRDF kernel
//!   `f_ms(µo, µi) = (1 - E(µo))(1 - E(µi)) / (π (1 - E_avg))`, whose directional
//!   albedo is exactly `1 - E(µ)` so single + multi scatter reflects all energy
//!   for a lossless (white) surface, and
//! * the Fernández-Agüera colored compensation used for image-based lighting,
//!   which scales the multiscatter lobe by the average Fresnel `F_avg` so a
//!   tinted conductor keeps the right hue after many bounces.
//!
//! The single-scattering albedo `E(µ)` is obtained by integrating the
//! Fresnel-free GGX reflectance (equivalently `scale + bias` from the split-sum
//! DFG LUT in [`crate::gi::env_brdf::dfg_lut`]). GGX primitives are reused from
//! [`crate::gi::spec_gi::ggx_lobe`]; this module never re-derives GGX.
//!
//! # Conventions
//! * `µ = n·v ∈ (0, 1]`; `E(µ)`, `E_avg ∈ [0, 1]`.
//! * `F_avg = F0 + (1 - F0) / 21` is the analytic hemispherical average of the
//!   Schlick Fresnel.
//! * Every total reflectance (single + multi) stays `≤ 1` per channel for
//!   `F0 ≤ 1` (the white-furnace / energy-conservation property).
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent method. All routines are deterministic pure functions (no RNG,
//!   I/O, GPU, globals or `unsafe`) and never emit `NaN`.
//!
//! # References
//! * Kulla & Conty 2017, *Revisiting Physically Based Shading at Imageworks*.
//! * Fernández-Agüera 2019, *A Multiple-Scattering Microfacet Model for
//!   Real-Time Image-Based Lighting*.

use bevy_math::Vec3;
use core::f32::consts::{FRAC_1_PI, PI};

use crate::gi::env_brdf::dfg_lut::integrate_dfg;

/// Smallest view cosine used when integrating the directional albedo.
pub const MIN_COS: f32 = 1.0e-3;

/// Default sample count used by [`directional_albedo`] / [`average_albedo`].
pub const DEFAULT_SAMPLES: u32 = 1024;

/// Reciprocal of the Schlick average-Fresnel denominator (`1 / 21`).
const INV_21: f32 = 1.0 / 21.0;

/// Single-scattering directional albedo `E(µ) = ∫ f_ss·cosθ dω` for the
/// Fresnel-free (white) GGX lobe at view cosine `n_dot_v` and `roughness`.
///
/// Equivalent to `scale + bias` from the split-sum DFG integration; it is the
/// fraction of incident energy a lossless single-scatter surface reflects, in
/// `[0, 1]`. `samples` is floored at `1`.
#[inline]
pub fn directional_albedo(n_dot_v: f32, roughness: f32, samples: u32) -> f32 {
    let sb = integrate_dfg(n_dot_v.clamp(MIN_COS, 1.0), roughness, samples);
    (sb.x + sb.y).clamp(0.0, 1.0)
}

/// Cosine-weighted hemispherical average of the directional albedo,
/// `E_avg = 2 ∫₀¹ E(µ) µ dµ`, estimated with `strata` cosine samples.
///
/// This is the mean fraction of energy a lossless single-scatter surface
/// reflects over all view directions, in `[0, 1]`. `strata` and `samples` are
/// floored at `1`.
pub fn average_albedo(roughness: f32, strata: u32, samples: u32) -> f32 {
    let m = strata.max(1);
    let mut acc = 0.0f64;
    // Midpoint rule in µ with the cosine weight 2µ (so ∫ 2µ dµ = 1).
    for i in 0..m {
        let mu = (i as f32 + 0.5) / m as f32;
        let e = directional_albedo(mu, roughness, samples);
        acc += (e * 2.0 * mu) as f64;
    }
    let avg = (acc / m as f64) as f32;
    avg.clamp(0.0, 1.0)
}

/// Analytic hemispherical average of the Schlick Fresnel for RGB `f0`:
/// `F_avg = F0 + (1 - F0) / 21`.
///
/// `f0` is clamped per channel to `[0, 1]`; the result lies in `[1/21, 1]`.
#[inline]
pub fn f_avg(f0: Vec3) -> Vec3 {
    let f0 = f0.clamp(Vec3::ZERO, Vec3::ONE);
    (f0 + (Vec3::ONE - f0) * INV_21).clamp(Vec3::ZERO, Vec3::ONE)
}

/// Scalar [`f_avg`] for a single-channel `f0` (e.g. a clearcoat's `0.04`).
#[inline]
pub fn f_avg_scalar(f0: f32) -> f32 {
    let f0 = f0.clamp(0.0, 1.0);
    (f0 + (1.0 - f0) * INV_21).clamp(0.0, 1.0)
}

/// Kulla-Conty achromatic multiple-scattering BRDF kernel.
///
/// `f_ms(µo, µi) = (1 - E(µo))(1 - E(µi)) / (π (1 - E_avg))`. This is a BRDF
/// value (units of `sr⁻¹`); multiply by `cosθ_i` and integrate to recover the
/// multiscatter directional albedo `1 - E(µo)`. Returns `0` when the surface is
/// already lossless (`E_avg → 1`). All inputs are clamped to `[0, 1]`.
#[inline]
pub fn kulla_conty_ms(e_o: f32, e_i: f32, e_avg: f32) -> f32 {
    let eo = e_o.clamp(0.0, 1.0);
    let ei = e_i.clamp(0.0, 1.0);
    let ea = e_avg.clamp(0.0, 1.0);
    let denom = PI * (1.0 - ea);
    if denom <= 1.0e-6 {
        return 0.0;
    }
    let v = (1.0 - eo) * (1.0 - ei) / denom;
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

/// Fernández-Agüera multiple-scattering reflectance `F_ms·E_ms` for RGB `f0`.
///
/// Given the split-sum single-scatter reflectance `FssEss = F0·scale + bias`,
/// the missing energy `E_ms = 1 - (scale + bias)` and the average Fresnel
/// `F_avg`, the added multiscatter energy is
/// `E_ms · FssEss · F_avg / (1 - F_avg · E_ms)` (per channel). Returns the
/// non-negative compensation term to *add* to the single-scatter response.
#[inline]
pub fn multiscatter_compensation(f0: Vec3, scale: f32, bias: f32) -> Vec3 {
    let f0 = f0.clamp(Vec3::ZERO, Vec3::ONE);
    let scale = scale.clamp(0.0, 1.0);
    let bias = bias.clamp(0.0, 1.0);
    let ess = (scale + bias).clamp(0.0, 1.0);
    let ems = (1.0 - ess).max(0.0);
    let fss_ess = (f0 * scale + Vec3::splat(bias)).max(Vec3::ZERO);
    let favg = f_avg(f0);
    let denom = Vec3::ONE - favg * ems;
    // Guard each channel's denominator (it is in [1/21·0 .. 1], never 0 for
    // F_avg ≥ 1/21 and E_ms ≤ 1, but stay defensive).
    let inv = Vec3::new(
        safe_recip(denom.x),
        safe_recip(denom.y),
        safe_recip(denom.z),
    );
    let comp = fss_ess * favg * ems * inv;
    if comp.is_finite() { comp.max(Vec3::ZERO) } else { Vec3::ZERO }
}

/// Full energy-compensated specular environment reflectance for RGB `f0`:
/// the single-scatter `FssEss` plus the Fernández-Agüera multiscatter term.
///
/// For a lossless white surface (`f0 = 1`) the total equals `1` regardless of
/// roughness (the white-furnace property). The result is clamped to `[0, 1]`
/// per channel so total reflectance never exceeds the incident energy.
#[inline]
pub fn multiscatter_specular(f0: Vec3, scale: f32, bias: f32) -> Vec3 {
    let f0 = f0.clamp(Vec3::ZERO, Vec3::ONE);
    let scale = scale.clamp(0.0, 1.0);
    let bias = bias.clamp(0.0, 1.0);
    let fss_ess = (f0 * scale + Vec3::splat(bias)).max(Vec3::ZERO);
    let total = fss_ess + multiscatter_compensation(f0, scale, bias);
    if total.is_finite() {
        total.clamp(Vec3::ZERO, Vec3::ONE)
    } else {
        Vec3::ZERO
    }
}

/// Reciprocal of `x` with a floored magnitude so a (near-)zero denominator can
/// never produce a non-finite result.
#[inline]
fn safe_recip(x: f32) -> f32 {
    let d = if x.abs() < 1.0e-6 { 1.0e-6 } else { x };
    let r = 1.0 / d;
    if r.is_finite() { r } else { 0.0 }
}

/// `1 / π`, re-exported for callers normalising a diffuse-like multiscatter
/// lobe without pulling in `core::f32::consts`.
pub const INV_PI: f32 = FRAC_1_PI;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::env_brdf::dfg_lut::bake_dfg_lut;

    #[test]
    fn directional_albedo_in_unit_range() {
        for &r in &[0.05f32, 0.3, 0.6, 1.0] {
            for &mu in &[0.1f32, 0.5, 1.0] {
                let e = directional_albedo(mu, r, 2048);
                assert!((0.0..=1.0).contains(&e), "E={e} r={r} mu={mu}");
            }
        }
    }

    #[test]
    fn directional_albedo_matches_dfg_sum() {
        let lut = bake_dfg_lut(32, 1024);
        let col = 18u32;
        let row = 14u32;
        let mu = (col as f32 + 0.5) / 32.0;
        let r = (row as f32 + 0.5) / 32.0;
        let sb = lut.sample(mu, r);
        let e = directional_albedo(mu, r, 1024);
        assert!((e - (sb.x + sb.y)).abs() < 1.0e-6, "E={e} vs {}", sb.x + sb.y);
    }

    #[test]
    fn average_albedo_in_unit_range_and_decreases_with_roughness() {
        let smooth = average_albedo(0.1, 32, 1024);
        let rough = average_albedo(0.95, 32, 1024);
        assert!((0.0..=1.0).contains(&smooth), "E_avg(smooth)={smooth}");
        assert!((0.0..=1.0).contains(&rough), "E_avg(rough)={rough}");
        assert!(rough < smooth, "rough E_avg={rough} should be < {smooth}");
    }

    #[test]
    fn f_avg_endpoints() {
        // White stays white; black floors at 1/21.
        assert!((f_avg(Vec3::ONE) - Vec3::ONE).length() < 1.0e-6);
        let black = f_avg(Vec3::ZERO);
        assert!((black - Vec3::splat(INV_21)).length() < 1.0e-6);
        assert!((f_avg_scalar(0.04) - (0.04 + 0.96 * INV_21)).abs() < 1.0e-6);
    }

    #[test]
    fn kulla_conty_albedo_recovers_missing_energy() {
        // The Kulla-Conty kernel integrates (over cosine-weighted directions)
        // to exactly 1 - E(µo); estimate that integral by Monte-Carlo.
        let roughness = 0.8f32;
        let e_avg = average_albedo(roughness, 48, 1024);
        let mu_o = 0.7f32;
        let e_o = directional_albedo(mu_o, roughness, 2048);

        let n = 4096usize;
        let mut acc = 0.0f64;
        for i in 0..n {
            // Cosine-weighted hemisphere sample: cosθ_i = sqrt(1 - u), pdf =
            // cosθ_i/π, so the estimator of ∫ f_ms cosθ dω is mean of π·f_ms.
            let u = (i as f32 + 0.5) / n as f32;
            let mu_i = (1.0 - u).max(0.0).sqrt();
            let e_i = directional_albedo(mu_i.max(MIN_COS), roughness, 1024);
            let f_ms = kulla_conty_ms(e_o, e_i, e_avg);
            acc += (PI * f_ms) as f64;
        }
        let ms_albedo = (acc / n as f64) as f32;
        let expected = 1.0 - e_o;
        assert!(
            (ms_albedo - expected).abs() < 0.05,
            "ms_albedo={ms_albedo} expected≈{expected}"
        );
    }

    #[test]
    fn white_furnace_total_energy_is_one() {
        // For a lossless white surface single + multi scatter reflects all
        // incident energy at every roughness and view angle.
        let lut = bake_dfg_lut(32, 2048);
        for &r in &[0.1f32, 0.4, 0.7, 1.0] {
            for &mu in &[0.15f32, 0.5, 0.85] {
                let sb = lut.sample(mu, r);
                let total = multiscatter_specular(Vec3::ONE, sb.x, sb.y);
                assert!(
                    (total.x - 1.0).abs() < 2.0e-2,
                    "white furnace total={} r={r} mu={mu}",
                    total.x
                );
            }
        }
    }

    #[test]
    fn colored_total_never_exceeds_one_and_adds_energy() {
        let lut = bake_dfg_lut(32, 1024);
        let f0 = Vec3::new(0.95, 0.64, 0.54); // copper-ish
        for &r in &[0.2f32, 0.5, 0.9] {
            let mu = 0.6f32;
            let sb = lut.sample(mu, r);
            let single = f0 * sb.x + Vec3::splat(sb.y);
            let total = multiscatter_specular(f0, sb.x, sb.y);
            // Multiscatter only adds energy and never overshoots.
            assert!(total.x >= single.x - 1.0e-4, "ms removed energy");
            assert!(total.x <= 1.0 + 1.0e-6, "ms overshoots: {}", total.x);
            assert!(total.is_finite());
        }
    }

    #[test]
    fn compensation_is_zero_for_lossless_single_scatter() {
        // When the single-scatter already captures all energy (scale+bias = 1)
        // there is nothing to compensate.
        let comp = multiscatter_compensation(Vec3::splat(0.5), 0.7, 0.3);
        assert!(comp.length() < 1.0e-5, "comp={comp:?} should vanish");
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        assert_eq!(kulla_conty_ms(0.5, 0.5, 1.0), 0.0);
        let c = multiscatter_compensation(Vec3::splat(f32::NAN), f32::INFINITY, -1.0);
        assert!(c.is_finite());
        let s = multiscatter_specular(Vec3::splat(2.0), 2.0, 2.0);
        assert!(s.is_finite() && s.x <= 1.0 + 1.0e-6);
        assert!(average_albedo(0.5, 0, 0).is_finite());
    }
}
