//! Glossy specular ReSTIR — directional GGX-lobe reservoir reuse (CPU golden).
//!
//! Diffuse ReSTIR GI (see [`crate::gi::screen_probe::restir`]) resamples a
//! luminance × geometry target that is *view independent*, so a neighbour's
//! radiance can be reused almost verbatim.  Glossy specular is different: a
//! stored incident radiance only contributes if it arrives along a direction
//! the destination's GGX lobe actually values, which depends on the view vector
//! and the roughness.  This module adapts the generic weighted-reservoir
//! machinery to that setting by swapping the scalar target for a *directional
//! GGX-lobe* target, so reuse re-targets a neighbour's incoming radiance onto
//! the current pixel's view/roughness instead of copying it blindly.
//!
//! It deliberately *reuses* the existing [`Reservoir`], [`GiSample`] and
//! [`balance_heuristic`] primitives (confidence weights, GRIS merge, M-cap,
//! unbiased contribution weight `W`), and only layers glossy-specific target
//! and reuse-weight functions on top.
//!
//! # Conventions
//! * A [`GlossyShadingPoint`] fixes the destination: world position, unit
//!   normal, unit view direction (towards the camera), perceptual roughness,
//!   signed anisotropy, and RGB `f0`.
//! * The reservoir payload stays [`GiSample`]: the stored secondary sample
//!   point and its outgoing radiance.  The incident direction is reconstructed
//!   as `normalize(sample_point - visible_point)`, so merging across pixels is
//!   a *reconnection* shift that re-evaluates the lobe at the new vertex.
//! * The resampling target is `p̂ = luminance(f_spec · cos · L)`: the GGX BRDF
//!   times the incident cosine times the stored radiance, reduced to luminance.
//!   This is exactly the quantity whose variance ReSTIR minimises.
//! * Reuse is gated by [`roughness_reuse_weight`] (lobes of very different
//!   width must not pollute each other) and the temporal M-cap is tightened for
//!   sharp surfaces by [`roughness_confidence_cap`] (mirror-like reflections
//!   are view-sensitive and must stay fresh).
//! * Every helper is a deterministic pure function with defensive clamps; no
//!   RNG, I/O, GPU, globals or `unsafe`, and no `NaN` is ever produced.

use crate::gi::sample::mapping::orthonormal_basis;
use crate::gi::screen_probe::restir::{GiSample, Reservoir, luminance};
use bevy_math::{Vec3, ops};

pub use crate::gi::screen_probe::restir::balance_heuristic;

use super::ggx_lobe::{ggx_brdf, roughness_to_alpha_anisotropic};

/// The destination shading point a glossy reservoir resamples towards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlossyShadingPoint {
    /// World-space shading position.
    pub position: Vec3,
    /// Unit surface normal.
    pub normal: Vec3,
    /// Unit view direction, pointing from the surface towards the camera.
    pub view: Vec3,
    /// Perceptual roughness in `[0, 1]`.
    pub roughness: f32,
    /// Signed anisotropy in `[-1, 1]` (0 = isotropic).
    pub anisotropy: f32,
    /// RGB Fresnel reflectance at normal incidence.
    pub f0: Vec3,
}

impl GlossyShadingPoint {
    /// Convenience constructor for an isotropic glossy point.
    #[inline]
    pub fn isotropic(position: Vec3, normal: Vec3, view: Vec3, roughness: f32, f0: Vec3) -> Self {
        Self {
            position,
            normal: normal.normalize_or_zero(),
            view: view.normalize_or_zero(),
            roughness,
            anisotropy: 0.0,
            f0,
        }
    }

    /// Projects a world-space direction into this point's local shading frame
    /// (`+Z` = normal).  Returns [`Vec3::ZERO`] for a degenerate normal.
    #[inline]
    pub fn local_dir(&self, world: Vec3) -> Vec3 {
        let n = self.normal;
        if n.length_squared() <= 0.0 {
            return Vec3::ZERO;
        }
        let (t, b) = orthonormal_basis([n.x, n.y, n.z]);
        let t = Vec3::new(t[0], t[1], t[2]);
        let b = Vec3::new(b[0], b[1], b[2]);
        Vec3::new(world.dot(t), world.dot(b), world.dot(n))
    }
}

/// The glossy *lobe throughput* `f_spec · (n·wi)` towards the sample point.
///
/// This is the destination-dependent part of the path throughput — the GGX
/// BRDF times the incident cosine — evaluated for the direction from the
/// shading point to `sample.sample_point`.  It excludes the stored radiance so
/// callers can multiply it by `L · W` for the final contribution.  Returns
/// [`Vec3::ZERO`] for a degenerate geometry or below-horizon direction.
#[inline]
pub fn glossy_lobe_throughput(point: &GlossyShadingPoint, sample: &GiSample) -> Vec3 {
    let to_sample = sample.sample_point - sample.visible_point;
    let dist2 = to_sample.length_squared();
    if dist2 <= 1.0e-12 {
        return Vec3::ZERO;
    }
    let wi_world = to_sample / dist2.sqrt();
    let wo = point.local_dir(point.view);
    let wi = point.local_dir(wi_world);
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return Vec3::ZERO;
    }
    let (ax, ay) = roughness_to_alpha_anisotropic(point.roughness, point.anisotropy);
    let f = ggx_brdf(wo, wi, ax, ay, point.f0);
    let v = f * wi.z;
    if v.is_finite() { v.max(Vec3::ZERO) } else { Vec3::ZERO }
}

/// The glossy lobe throughput `f_spec · (n·wi)` for an explicit unit world
/// incident direction `wi_world`.
///
/// This is the direction-space counterpart of [`glossy_lobe_throughput`], used
/// by the BRDF/light MIS estimator which samples directions directly rather
/// than reconnection vertices.  Returns [`Vec3::ZERO`] for a below-horizon
/// direction.
#[inline]
pub fn glossy_lobe_throughput_dir(point: &GlossyShadingPoint, wi_world: Vec3) -> Vec3 {
    let wo = point.local_dir(point.view);
    let wi = point.local_dir(wi_world);
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return Vec3::ZERO;
    }
    let (ax, ay) = roughness_to_alpha_anisotropic(point.roughness, point.anisotropy);
    let f = ggx_brdf(wo, wi, ax, ay, point.f0);
    let v = f * wi.z;
    if v.is_finite() { v.max(Vec3::ZERO) } else { Vec3::ZERO }
}

/// Directional glossy resampling target `p̂` for `sample` at `point`.
///
/// `p̂ = luminance(f_spec · cos · L)` — the GGX lobe throughput modulated by the
/// stored outgoing radiance, reduced to Rec. 709 luminance.  Always finite and
/// non-negative; degenerate geometry or a dark/back-facing sample yields `0`,
/// which the reservoir then discards.
#[inline]
pub fn glossy_target_function(point: &GlossyShadingPoint, sample: &GiSample) -> f32 {
    let throughput = glossy_lobe_throughput(point, sample);
    let contrib = throughput * sample.radiance;
    let t = luminance(contrib);
    if t.is_finite() { t.max(0.0) } else { 0.0 }
}

/// Similarity weight in `[0, 1]` for reusing a neighbour whose roughness and
/// normal differ from the destination's.
///
/// Combines a Gaussian on the roughness gap (lobes of very different width must
/// not mix) with a cosine power on the normal gap (surfaces facing away carry a
/// mismatched lobe).  `sigma_roughness` controls roughness tolerance; a larger
/// value reuses more aggressively.  Degenerate inputs collapse to `0`.
#[inline]
pub fn roughness_reuse_weight(
    dst_roughness: f32,
    src_roughness: f32,
    dst_normal: Vec3,
    src_normal: Vec3,
    sigma_roughness: f32,
) -> f32 {
    let sigma = sigma_roughness.max(1.0e-3);
    let dr = (dst_roughness.clamp(0.0, 1.0) - src_roughness.clamp(0.0, 1.0)) / sigma;
    let rough_w = ops::exp(-0.5 * dr * dr);

    let n_dst = dst_normal.normalize_or_zero();
    let n_src = src_normal.normalize_or_zero();
    if n_dst.length_squared() == 0.0 || n_src.length_squared() == 0.0 {
        return 0.0;
    }
    // Normal agreement raised to a power sharpens the rejection of mismatched
    // orientations; the exponent grows as the destination gets smoother.
    let cos_n = n_dst.dot(n_src).clamp(0.0, 1.0);
    let exponent = 8.0 + 56.0 * (1.0 - dst_roughness.clamp(0.0, 1.0));
    let normal_w = ops::powf(cos_n, exponent);

    let w = rough_w * normal_w;
    if w.is_finite() { w.clamp(0.0, 1.0) } else { 0.0 }
}

/// Roughness-dependent temporal confidence cap (`M`-cap).
///
/// Sharp (low-roughness) specular is strongly view dependent, so stale history
/// must not dominate; the cap therefore scales from a small floor at mirror
/// roughness up to `base_cap` as the surface becomes rough-diffuse.  Returns a
/// value in `[1, base_cap]`.
#[inline]
pub fn roughness_confidence_cap(roughness: f32, base_cap: f32) -> f32 {
    let base = base_cap.max(1.0);
    let r = roughness.clamp(0.0, 1.0);
    // Smooth ramp: mirror -> ~1 frame, rough -> full base_cap.
    let cap = 1.0 + (base - 1.0) * r;
    cap.clamp(1.0, base)
}

/// Streams one freshly traced candidate into a glossy reservoir using RIS.
///
/// `source_pdf` is the density the candidate direction/radiance was drawn from
/// (e.g. the BRDF-sampling pdf); the RIS resampling weight is
/// `p̂ / source_pdf`.  `rng_uniform ∈ [0, 1]` drives the streaming replacement
/// test.  A degenerate candidate (`source_pdf <= 0`, zero target) is ignored
/// without perturbing the reservoir.  Returns `true` when the candidate was
/// selected.
#[inline]
pub fn stream_glossy_candidate(
    reservoir: &mut Reservoir<GiSample>,
    point: &GlossyShadingPoint,
    sample: GiSample,
    source_pdf: f32,
    rng_uniform: f32,
) -> bool {
    if !source_pdf.is_finite() || source_pdf <= 0.0 {
        return false;
    }
    let p_hat = glossy_target_function(point, &sample);
    let weight = p_hat / source_pdf;
    reservoir.update(sample, weight, rng_uniform)
}

/// Merges a neighbour (spatial or temporal) into `canonical` with GRIS reuse,
/// re-targeting the neighbour's sample onto the destination `point`.
///
/// The neighbour's selected sample is re-evaluated through the *destination's*
/// glossy lobe to obtain the shift-mapped target `p̂`, then scaled by the
/// roughness/normal [`roughness_reuse_weight`] so dissimilar neighbours are
/// suppressed.  Confidence still accumulates even when the induced weight is
/// zero (keeping the balance heuristic correct).  Returns `true` when the
/// neighbour's sample was selected.
#[inline]
pub fn merge_glossy(
    canonical: &mut Reservoir<GiSample>,
    neighbour: &Reservoir<GiSample>,
    point: &GlossyShadingPoint,
    neighbour_roughness: f32,
    neighbour_normal: Vec3,
    sigma_roughness: f32,
    rng_uniform: f32,
) -> bool {
    let shifted_pdf = match neighbour.sample() {
        Some(s) => glossy_target_function(point, &s),
        None => 0.0,
    };
    let reuse = roughness_reuse_weight(
        point.roughness,
        neighbour_roughness,
        point.normal,
        neighbour_normal,
        sigma_roughness,
    );
    canonical.merge(neighbour, shifted_pdf * reuse, rng_uniform)
}

/// Finalises the unbiased contribution weight `W` of a glossy reservoir at
/// `point` and returns it.
///
/// Evaluates the selected sample's own destination target `p̂` and feeds it to
/// [`Reservoir::finalize_weight`], giving `W = (w_sum / m) / p̂`.  Returns `0`
/// for an empty or degenerate reservoir.
#[inline]
pub fn finalize_glossy(reservoir: &mut Reservoir<GiSample>, point: &GlossyShadingPoint) -> f32 {
    let p_hat = match reservoir.sample() {
        Some(s) => glossy_target_function(point, &s),
        None => 0.0,
    };
    reservoir.finalize_weight(p_hat);
    reservoir.contribution_weight()
}

/// Final glossy specular radiance for the selected sample: `f_spec · cos · L · W`.
///
/// Multiplies the destination lobe throughput by the stored radiance and the
/// reservoir's finalised contribution weight `W`.  Returns [`Vec3::ZERO`] for an
/// empty reservoir or a zero weight.
#[inline]
pub fn glossy_contribution(reservoir: &Reservoir<GiSample>, point: &GlossyShadingPoint) -> Vec3 {
    let sample = match reservoir.sample() {
        Some(s) => s,
        None => return Vec3::ZERO,
    };
    let w = reservoir.contribution_weight();
    if !w.is_finite() || w <= 0.0 {
        return Vec3::ZERO;
    }
    let throughput = glossy_lobe_throughput(point, &sample);
    let v = throughput * sample.radiance * w;
    if v.is_finite() { v.max(Vec3::ZERO) } else { Vec3::ZERO }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(roughness: f32) -> GlossyShadingPoint {
        GlossyShadingPoint::isotropic(
            Vec3::ZERO,
            Vec3::Z,
            Vec3::new(0.0, 0.3, 0.954).normalize(),
            roughness,
            Vec3::splat(0.5),
        )
    }

    fn mirror_sample(point: &GlossyShadingPoint, radiance: Vec3) -> GiSample {
        // Place the sample along the mirror-reflected view direction.
        let v = point.view;
        let n = point.normal;
        let refl = 2.0 * v.dot(n) * n - v;
        GiSample {
            visible_point: point.position,
            visible_normal: n,
            sample_point: point.position + refl * 3.0,
            sample_normal: -refl.normalize(),
            radiance,
        }
    }

    #[test]
    fn target_peaks_near_mirror_direction() {
        let point = make_point(0.1);
        let on_axis = mirror_sample(&point, Vec3::splat(2.0));
        // An off-axis sample along a grazing tangent direction.
        let off = GiSample {
            sample_point: point.position + Vec3::new(0.9, 0.0, 0.436) * 3.0,
            ..on_axis
        };
        let t_on = glossy_target_function(&point, &on_axis);
        let t_off = glossy_target_function(&point, &off);
        assert!(t_on > t_off, "on={t_on} off={t_off}");
        assert!(t_on.is_finite() && t_on >= 0.0);
    }

    #[test]
    fn target_is_zero_for_backfacing_or_dark_samples() {
        let point = make_point(0.3);
        // Dark radiance -> zero target.
        let dark = mirror_sample(&point, Vec3::ZERO);
        assert_eq!(glossy_target_function(&point, &dark), 0.0);
        // Sample behind the surface -> zero throughput.
        let behind = GiSample {
            sample_point: point.position + Vec3::new(0.0, 0.0, -2.0),
            ..mirror_sample(&point, Vec3::splat(1.0))
        };
        assert_eq!(glossy_target_function(&point, &behind), 0.0);
    }

    #[test]
    fn reuse_weight_rejects_mismatched_roughness_and_normal() {
        // Identical roughness and normal -> ~1.
        let w_same = roughness_reuse_weight(0.3, 0.3, Vec3::Z, Vec3::Z, 0.1);
        assert!((w_same - 1.0).abs() < 1e-4, "w_same={w_same}");
        // Large roughness gap -> small weight.
        let w_rough = roughness_reuse_weight(0.1, 0.9, Vec3::Z, Vec3::Z, 0.1);
        assert!(w_rough < 0.05, "w_rough={w_rough}");
        // Tilted normal -> suppressed.
        let tilted = Vec3::new(0.6, 0.0, 0.8).normalize();
        let w_norm = roughness_reuse_weight(0.1, 0.1, Vec3::Z, tilted, 0.1);
        assert!(w_norm < 0.5, "w_norm={w_norm}");
    }

    #[test]
    fn confidence_cap_tightens_for_sharp_surfaces() {
        let base = 32.0;
        let mirror = roughness_confidence_cap(0.0, base);
        let rough = roughness_confidence_cap(1.0, base);
        assert!(mirror < rough, "mirror={mirror} rough={rough}");
        assert!((rough - base).abs() < 1e-4);
        assert!(mirror >= 1.0);
    }

    #[test]
    fn streaming_then_finalize_gives_unbiased_weight() {
        let point = make_point(0.25);
        let mut r = Reservoir::<GiSample>::new();
        let s = mirror_sample(&point, Vec3::splat(3.0));
        // Draw with source pdf equal to the target so W should be ~1/p_hat * p_hat/pdf.
        let selected = stream_glossy_candidate(&mut r, &point, s, 0.5, 0.0);
        assert!(selected);
        let w = finalize_glossy(&mut r, &point);
        // W = (w_sum / m) / p_hat = (p_hat/pdf) / p_hat = 1/pdf = 2.
        assert!((w - 2.0).abs() < 1e-3, "w={w}");
        let c = glossy_contribution(&r, &point);
        assert!(c.is_finite() && c.length() > 0.0);
    }

    #[test]
    fn merge_retargets_neighbour_to_destination_lobe() {
        let dst = make_point(0.2);
        // A neighbour reservoir carrying a mirror sample for a *different* point.
        let src_point = make_point(0.2);
        let mut neighbour = Reservoir::<GiSample>::new();
        let s = mirror_sample(&src_point, Vec3::splat(4.0));
        stream_glossy_candidate(&mut neighbour, &src_point, s, 1.0, 0.0);
        finalize_glossy(&mut neighbour, &src_point);
        neighbour.cap_confidence(8.0);

        let mut canonical = Reservoir::<GiSample>::new();
        let own = mirror_sample(&dst, Vec3::splat(1.0));
        stream_glossy_candidate(&mut canonical, &dst, own, 1.0, 0.5);

        let before = canonical.confidence();
        let _ = merge_glossy(&mut canonical, &neighbour, &dst, 0.2, dst.normal, 0.1, 0.0);
        // Confidence accumulated from the neighbour regardless of selection.
        assert!(canonical.confidence() > before);
        let w = finalize_glossy(&mut canonical, &dst);
        assert!(w.is_finite() && w >= 0.0);
    }

    #[test]
    fn merge_mismatched_roughness_does_not_blow_up() {
        let dst = make_point(0.05); // near-mirror
        let src_point = make_point(0.95); // rough
        let mut neighbour = Reservoir::<GiSample>::new();
        let s = mirror_sample(&src_point, Vec3::splat(10.0));
        stream_glossy_candidate(&mut neighbour, &src_point, s, 1.0, 0.0);
        finalize_glossy(&mut neighbour, &src_point);

        let mut canonical = Reservoir::<GiSample>::new();
        let own = mirror_sample(&dst, Vec3::splat(1.0));
        stream_glossy_candidate(&mut canonical, &dst, own, 1.0, 0.5);
        // Reuse weight is tiny, so the rough neighbour should not be selected.
        let selected = merge_glossy(&mut canonical, &neighbour, &dst, 0.95, dst.normal, 0.1, 0.5);
        assert!(!selected);
        let c = glossy_contribution(&canonical, &dst);
        assert!(c.is_finite());
    }

    #[test]
    fn empty_reservoir_contributes_nothing() {
        let point = make_point(0.3);
        let r = Reservoir::<GiSample>::new();
        assert_eq!(glossy_contribution(&r, &point), Vec3::ZERO);
    }

    #[test]
    fn results_are_deterministic() {
        let point = make_point(0.3);
        let build = || {
            let mut r = Reservoir::<GiSample>::new();
            stream_glossy_candidate(&mut r, &point, mirror_sample(&point, Vec3::splat(2.0)), 1.0, 0.2);
            stream_glossy_candidate(&mut r, &point, mirror_sample(&point, Vec3::splat(3.0)), 1.5, 0.7);
            finalize_glossy(&mut r, &point);
            r
        };
        assert_eq!(build(), build());
    }
}
