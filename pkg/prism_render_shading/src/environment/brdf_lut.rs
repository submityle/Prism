//! Split-sum environment-BRDF integration (the "DFG" half of Karis 2013).
//!
//! Real-time IBL splits the reflectance integral into a prefiltered radiance
//! term (see [`super::prefilter`]) and a scalar environment BRDF that depends
//! only on view angle and roughness.  This module integrates that BRDF with
//! GGX importance sampling and stores it as a 2D lookup table indexed by
//! `(n_dot_v, perceptual_roughness)`.  Each texel holds the split-sum
//! `(scale, bias)` so the shader reconstructs specular reflectance as
//! `F0 * scale + bias`, matching Unreal's `PreIntegratedGF` / `EnvBRDF`.
//!
//! The integration mirrors what the GPU `brdf_lut.wesl` compute pass produces,
//! so the CPU golden and the shader share one reference.  All transcendentals
//! route through `bevy_math::ops` for cross-platform determinism.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use super::sampling::{hammersley, importance_sample_ggx};

/// Smith height-correlated masking-shadowing for IBL, using Karis's
/// `k = alpha^2 / 2` remap (distinct from the analytic-light `k`).
fn geometry_smith_ibl(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    // IBL masking-shadowing uses Karis's k = roughness^2 / 2 remap on the
    // perceptual roughness (distinct from the (r+1)^2/8 analytic-light remap).
    let k = roughness * roughness * 0.5;
    let g1 = |cos_theta: f32| cos_theta / (cos_theta * (1.0 - k) + k);
    g1(n_dot_v) * g1(n_dot_l)
}

/// Integrates the split-sum environment BRDF for one `(n_dot_v, roughness)`
/// pair with `sample_count` GGX importance samples.
///
/// Returns `[scale, bias]` such that the specular reflectance for a Fresnel
/// `f0` is `f0 * scale + bias`.  At normal incidence on a mirror
/// (`roughness == 0`, `n_dot_v == 1`) this is `[1, 0]`; away from normal it
/// bakes the Schlick Fresnel rise (`bias` grows as `n_dot_v` drops) so
/// `f0 * scale + bias` reproduces `f0 + (1 - f0) * (1 - n_dot_v)^5`.
pub fn integrate_brdf(n_dot_v: f32, roughness: f32, sample_count: u32) -> [f32; 2] {
    let n_dot_v = n_dot_v.clamp(1.0e-4, 1.0);
    let roughness = roughness.clamp(0.0, 1.0);
    let samples = sample_count.max(1);

    // View vector in the tangent frame (normal = +Z), lying in the x-z plane.
    let view = [ops::sqrt((1.0 - n_dot_v * n_dot_v).max(0.0)), 0.0, n_dot_v];

    let mut scale = 0.0f32;
    let mut bias = 0.0f32;
    for i in 0..samples {
        let xi = hammersley(i, samples);
        let h = importance_sample_ggx(xi, roughness);
        let v_dot_h = view[0] * h[0] + view[1] * h[1] + view[2] * h[2];
        // Reflect the view about the sampled half vector to get the light dir.
        let light = [
            2.0 * v_dot_h * h[0] - view[0],
            2.0 * v_dot_h * h[1] - view[1],
            2.0 * v_dot_h * h[2] - view[2],
        ];
        let n_dot_l = light[2];
        if n_dot_l <= 0.0 {
            continue;
        }
        let n_dot_h = h[2].max(0.0);
        let v_dot_h = v_dot_h.max(0.0);
        let g = geometry_smith_ibl(n_dot_v, n_dot_l, roughness);
        // Importance-sampling weight: G * VoH / (NoH * NoV) folds the GGX pdf.
        let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v).max(1.0e-6);
        let one_minus_voh = 1.0 - v_dot_h;
        // Schlick Fresnel complement (1 - VoH)^5 without a transcendental.
        let fc = one_minus_voh * one_minus_voh * one_minus_voh * one_minus_voh * one_minus_voh;
        scale += (1.0 - fc) * g_vis;
        bias += fc * g_vis;
    }

    let inv = (samples as f32).recip();
    [scale * inv, bias * inv]
}

/// A precomputed split-sum environment BRDF table.
///
/// The table is square: the x axis samples `n_dot_v` and the y axis samples
/// perceptual `roughness`, both across `[0, 1]` at texel centres.  Each entry
/// stores the `[scale, bias]` returned by [`integrate_brdf`].
#[derive(Clone, Debug, PartialEq)]
pub struct DfgLut {
    /// Shared edge length of the square table in texels.
    pub resolution: u32,
    /// Row-major `[scale, bias]` texels (`index = y * resolution + x`).
    pub texels: Vec<[f32; 2]>,
}

impl DfgLut {
    /// Builds a `resolution x resolution` table using `sample_count` GGX
    /// samples per texel.
    ///
    /// Returns `None` when `resolution` is zero so callers can fall back to the
    /// analytic [`super::env_brdf_approx`] fit instead of an empty table.
    pub fn generate(resolution: u32, sample_count: u32) -> Option<Self> {
        if resolution == 0 {
            return None;
        }
        let inv = (resolution as f32).recip();
        let mut texels = vec![[0.0f32; 2]; (resolution as usize) * (resolution as usize)];
        for y in 0..resolution {
            // Texel centres so the table samples the open interval, avoiding the
            // degenerate n_dot_v = 0 column.
            let roughness = ((y as f32) + 0.5) * inv;
            for x in 0..resolution {
                let n_dot_v = ((x as f32) + 0.5) * inv;
                texels[(y * resolution + x) as usize] =
                    integrate_brdf(n_dot_v, roughness, sample_count);
            }
        }
        Some(Self { resolution, texels })
    }

    /// Bilinearly samples the table at `(n_dot_v, roughness)`.
    ///
    /// Both inputs are clamped to `[0, 1]` and mapped onto the texel-centre
    /// grid, so queries outside the sampled range clamp to the nearest edge.
    pub fn sample(&self, n_dot_v: f32, roughness: f32) -> [f32; 2] {
        let n = self.resolution;
        if n == 0 || self.texels.is_empty() {
            return [1.0, 0.0];
        }
        let max_index = (n - 1) as f32;
        // Undo the +0.5 texel-centre offset to land on integer sample lattice.
        let fx = (n_dot_v.clamp(0.0, 1.0) * (n as f32) - 0.5).clamp(0.0, max_index);
        let fy = (roughness.clamp(0.0, 1.0) * (n as f32) - 0.5).clamp(0.0, max_index);
        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(n - 1);
        let y1 = (y0 + 1).min(n - 1);
        let tx = fx - (x0 as f32);
        let ty = fy - (y0 as f32);
        let at = |x: u32, y: u32| self.texels[(y * n + x) as usize];
        let c00 = at(x0, y0);
        let c10 = at(x1, y0);
        let c01 = at(x0, y1);
        let c11 = at(x1, y1);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        [
            lerp(lerp(c00[0], c10[0], tx), lerp(c01[0], c11[0], tx), ty),
            lerp(lerp(c00[1], c10[1], tx), lerp(c01[1], c11[1], tx), ty),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::env_brdf_approx;

    #[test]
    fn mirror_reproduces_schlick_fresnel() {
        // roughness = 0 collapses the lobe to the mirror direction, so the
        // split sum degenerates to the Schlick Fresnel term:
        //   f0 * scale + bias == f0 + (1 - f0) * (1 - NoV)^5.
        // At normal incidence that means [scale, bias] == [1, 0]; at grazing
        // the bias carries the Fresnel rise.
        let [scale, bias] = integrate_brdf(1.0, 0.0, 1024);
        assert!((scale - 1.0).abs() < 1.0e-3, "normal-incidence scale {scale}");
        assert!(bias.abs() < 1.0e-3, "normal-incidence bias {bias}");

        for &n_dot_v in &[0.15_f32, 0.4, 0.7, 1.0] {
            let [scale, bias] = integrate_brdf(n_dot_v, 0.0, 1024);
            let one_minus = 1.0 - n_dot_v;
            let fc = one_minus * one_minus * one_minus * one_minus * one_minus;
            // Reconstruct reflectance for two arbitrary Fresnel values.
            for &f0 in &[0.04_f32, 0.9] {
                let reconstructed = f0 * scale + bias;
                let schlick = f0 + (1.0 - f0) * fc;
                assert!(
                    (reconstructed - schlick).abs() < 2.0e-3,
                    "NoV {n_dot_v} f0 {f0}: {reconstructed} vs {schlick}"
                );
            }
        }
    }

    #[test]
    fn energy_is_conserved_across_the_table() {
        for &roughness in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            for &n_dot_v in &[0.05_f32, 0.25, 0.5, 0.75, 1.0] {
                let [scale, bias] = integrate_brdf(n_dot_v, roughness, 512);
                assert!(scale.is_finite() && bias.is_finite());
                assert!(scale >= -1.0e-3 && bias >= -1.0e-3, "negative {scale}/{bias}");
                assert!(
                    scale + bias <= 1.0 + 1.0e-2,
                    "energy {scale}+{bias} at NoV {n_dot_v} r {roughness}"
                );
            }
        }
    }

    #[test]
    fn integration_converges_with_more_samples() {
        // A correct importance-sampling estimator is (near) unbiased, so the
        // Monte-Carlo estimate must stabilise as the sample budget grows.
        for &roughness in &[0.15_f32, 0.35, 0.6, 0.85] {
            for &n_dot_v in &[0.15_f32, 0.5, 0.8, 1.0] {
                let coarse = integrate_brdf(n_dot_v, roughness, 256);
                let fine = integrate_brdf(n_dot_v, roughness, 4096);
                assert!(
                    (coarse[0] - fine[0]).abs() < 0.03 && (coarse[1] - fine[1]).abs() < 0.03,
                    "not converged at NoV {n_dot_v} r {roughness}: {coarse:?} vs {fine:?}"
                );
            }
        }
    }

    #[test]
    fn stays_in_the_ballpark_of_the_analytic_fit() {
        // Karis's mobile EnvBRDFApprox is a cheap fit to this same integral, so
        // it should be in the same ballpark everywhere (a coarse sanity bound
        // that catches gross integrator errors), even though the integrated LUT
        // is the ground truth and the fit visibly diverges on smooth head-on
        // configurations.
        for &roughness in &[0.1_f32, 0.35, 0.6, 0.85, 1.0] {
            for &n_dot_v in &[0.2_f32, 0.4, 0.6] {
                let integrated = integrate_brdf(n_dot_v, roughness, 2048);
                let analytic = env_brdf_approx(n_dot_v, roughness);
                assert!(
                    (integrated[0] - analytic[0]).abs() < 0.12,
                    "scale {integrated:?} vs {analytic:?} at NoV {n_dot_v} r {roughness}"
                );
                assert!(
                    (integrated[1] - analytic[1]).abs() < 0.1,
                    "bias {integrated:?} vs {analytic:?} at NoV {n_dot_v} r {roughness}"
                );
            }
        }
    }

    #[test]
    fn scale_decreases_as_roughness_grows_at_fixed_view() {
        // More microfacet spread pushes energy out of the specular peak, so the
        // F0-weighted scale term drops monotonically with roughness.
        let n_dot_v = 0.7;
        let mut previous = f32::INFINITY;
        for step in 0..=8 {
            let roughness = (step as f32) / 8.0;
            let [scale, _] = integrate_brdf(n_dot_v, roughness, 2048);
            assert!(
                scale <= previous + 5.0e-3,
                "scale rose at r {roughness}: {scale} after {previous}"
            );
            previous = scale;
        }
    }

    #[test]
    fn lut_sample_reproduces_texel_centres() {
        let lut = DfgLut::generate(32, 256).expect("non-zero resolution");
        let inv = 1.0 / 32.0;
        for &(x, y) in &[(0u32, 0u32), (5, 12), (31, 31), (16, 3)] {
            let n_dot_v = ((x as f32) + 0.5) * inv;
            let roughness = ((y as f32) + 0.5) * inv;
            let sampled = lut.sample(n_dot_v, roughness);
            let stored = lut.texels[(y * 32 + x) as usize];
            assert!((sampled[0] - stored[0]).abs() < 1.0e-5, "{sampled:?} vs {stored:?}");
            assert!((sampled[1] - stored[1]).abs() < 1.0e-5, "{sampled:?} vs {stored:?}");
        }
    }

    #[test]
    fn zero_resolution_is_rejected() {
        assert!(DfgLut::generate(0, 64).is_none());
    }
}
