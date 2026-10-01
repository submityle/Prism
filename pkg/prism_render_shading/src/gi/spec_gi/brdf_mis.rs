//! BRDF / light multiple-importance sampling for glossy reflection — CPU golden.
//!
//! A glossy reflection integral `∫ f_spec(wo, wi) L(wi) (n·wi) dω` is estimated
//! by combining two complementary samplers: *BRDF sampling* (good for sharp
//! lobes / smooth surfaces, poor for small bright lights) and *light sampling*
//! (good for small bright lights, poor for near-mirror lobes).  Multiple
//! importance sampling (MIS, Veach & Guibas 1995) blends the two with
//! per-sample weights so the combined estimator inherits the lower variance of
//! whichever technique dominates, while staying unbiased.
//!
//! This module provides:
//! * the [`balance_heuristic_pair`] and [`power_heuristic`] (`β = 2`) weights,
//!   with sample-count-aware variants for stratified multi-sample MIS,
//! * [`brdf_pdf_glossy`] — the solid-angle pdf of a BRDF-sampled direction,
//!   reusing the GGX VNDF reflection pdf from [`super::ggx_lobe`],
//! * [`area_to_solid_angle_pdf`] — the Jacobian that puts an area-light pdf in
//!   the same solid-angle measure as the BRDF pdf, and
//! * [`mis_estimator_glossy`] — the one-BRDF-sample + one-light-sample combined
//!   estimator that returns the MIS-weighted radiance.
//!
//! # Conventions
//! * All pdfs are expressed with respect to **solid angle** at the shading
//!   point; mixing measures is the classic MIS bug, so [`area_to_solid_angle_pdf`]
//!   is the only place the measure changes.
//! * Weights are deterministic pure functions and defend against degeneracy: a
//!   zero/non-finite denominator yields a `0` weight, and the two paired weights
//!   sum to `1` whenever at least one pdf is positive (partition of unity).
//! * `f32` arithmetic mirrors the WESL/GPU twin.  No RNG, I/O, GPU, globals or
//!   `unsafe`, and no `NaN` is ever produced.
//!
//! # References
//! * Veach & Guibas 1995, *Optimally Combining Sampling Techniques for Monte
//!   Carlo Rendering* — the balance and power heuristics.

use crate::gi::screen_probe::restir::luminance;
use bevy_math::Vec3;

use super::ggx_lobe::vndf_pdf_reflect;
use super::glossy_reservoir::{GlossyShadingPoint, glossy_lobe_throughput_dir};

/// Balance-heuristic MIS weight `w_a = pdf_a / (pdf_a + pdf_b)` for a single
/// sample from technique `a` paired against technique `b`.
///
/// Returns `0` for a non-positive `pdf_a` or a degenerate (zero / non-finite)
/// denominator; otherwise a value in `[0, 1]`.
#[inline]
pub fn balance_heuristic_pair(pdf_a: f32, pdf_b: f32) -> f32 {
    let a = pdf_a.max(0.0);
    let b = pdf_b.max(0.0);
    if a <= 0.0 {
        return 0.0;
    }
    let denom = a + b;
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    (a / denom).clamp(0.0, 1.0)
}

/// Power-heuristic (`β = 2`) MIS weight `w_a = pdf_a^2 / (pdf_a^2 + pdf_b^2)`.
///
/// The power heuristic sharpens the balance heuristic towards the lower-variance
/// technique and is the standard choice for BRDF/light MIS.  Returns `0` for a
/// non-positive `pdf_a` or a degenerate denominator.
#[inline]
pub fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a = pdf_a.max(0.0);
    let b = pdf_b.max(0.0);
    if a <= 0.0 {
        return 0.0;
    }
    let a2 = a * a;
    let b2 = b * b;
    let denom = a2 + b2;
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    (a2 / denom).clamp(0.0, 1.0)
}

/// Sample-count-aware power heuristic for `n_a` samples of technique `a` and
/// `n_b` samples of technique `b`.
///
/// Uses the weighted densities `n_a·pdf_a` and `n_b·pdf_b`:
/// `w_a = (n_a pdf_a)^2 / ((n_a pdf_a)^2 + (n_b pdf_b)^2)`.  With `n_a = n_b`
/// this reduces to [`power_heuristic`].  Returns `0` for degenerate inputs.
#[inline]
pub fn power_heuristic_counts(n_a: f32, pdf_a: f32, n_b: f32, pdf_b: f32) -> f32 {
    let a = n_a.max(0.0) * pdf_a.max(0.0);
    let b = n_b.max(0.0) * pdf_b.max(0.0);
    power_heuristic(a, b)
}

/// Sample-count-aware balance heuristic for `n_a`/`n_b` samples.
#[inline]
pub fn balance_heuristic_counts(n_a: f32, pdf_a: f32, n_b: f32, pdf_b: f32) -> f32 {
    let a = n_a.max(0.0) * pdf_a.max(0.0);
    let b = n_b.max(0.0) * pdf_b.max(0.0);
    balance_heuristic_pair(a, b)
}

/// Solid-angle pdf of a BRDF-sampled incident direction `wi_world` at `point`.
///
/// Transforms the view and incident directions into the shading frame and
/// evaluates the GGX VNDF reflection pdf (see
/// [`super::ggx_lobe::vndf_pdf_reflect`]).  Returns `0` for below-horizon
/// directions.
#[inline]
pub fn brdf_pdf_glossy(point: &GlossyShadingPoint, wi_world: Vec3) -> f32 {
    let wo = point.local_dir(point.view);
    let wi = point.local_dir(wi_world);
    let (ax, ay) = super::ggx_lobe::roughness_to_alpha_anisotropic(point.roughness, point.anisotropy);
    vndf_pdf_reflect(wo, wi, ax, ay)
}

/// Converts an **area**-measure light pdf to a **solid-angle** pdf at the
/// shading point.
///
/// `pdf_ω = pdf_A · dist² / |cos_light|`, where `dist` is the distance from the
/// shading point to the sampled light point and `cos_light` is the cosine
/// between the light's normal and the direction back to the shading point.
/// Returns `0` for a degenerate (grazing / coincident) configuration.
#[inline]
pub fn area_to_solid_angle_pdf(pdf_area: f32, dist: f32, cos_light: f32) -> f32 {
    let p = pdf_area.max(0.0);
    let d2 = dist * dist;
    let c = cos_light.abs();
    if p <= 0.0 || !d2.is_finite() || c <= 1.0e-6 {
        return 0.0;
    }
    let pdf = p * d2 / c;
    if pdf.is_finite() { pdf.max(0.0) } else { 0.0 }
}

/// A direct-lighting sample used by the MIS estimator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionSample {
    /// Unit incident direction at the shading point, pointing towards the light.
    pub wi: Vec3,
    /// Incident radiance `L(wi)` arriving along `wi`.
    pub radiance: Vec3,
    /// Solid-angle pdf of the technique that produced this direction.
    pub pdf: f32,
}

impl DirectionSample {
    /// A null sample carrying no energy and an invalid pdf (ignored by the
    /// estimator).
    pub const NONE: Self = Self {
        wi: Vec3::ZERO,
        radiance: Vec3::ZERO,
        pdf: 0.0,
    };
}

/// Single-sample contribution of a BRDF-sampled direction: `f·cos·L / pdf`.
///
/// This is the raw (unweighted) Monte-Carlo estimator term; the caller
/// multiplies it by the MIS weight.  Returns [`Vec3::ZERO`] for a degenerate
/// pdf or below-horizon direction.
#[inline]
pub fn brdf_sample_estimate(point: &GlossyShadingPoint, sample: &DirectionSample) -> Vec3 {
    if !sample.pdf.is_finite() || sample.pdf <= 0.0 {
        return Vec3::ZERO;
    }
    let throughput = glossy_lobe_throughput_dir(point, sample.wi.normalize_or_zero());
    let v = throughput * sample.radiance / sample.pdf;
    if v.is_finite() { v.max(Vec3::ZERO) } else { Vec3::ZERO }
}

/// Combined one-BRDF-sample + one-light-sample MIS estimator for glossy
/// reflection.
///
/// `brdf_sample` is a direction drawn from the GGX lobe (its `pdf` the BRDF
/// pdf); `light_sample` is a direction drawn from a light (its `pdf` the light's
/// solid-angle pdf).  Each term is weighted by the power heuristic using the
/// *other* technique's pdf evaluated for the same direction, then summed:
///
/// `L ≈ w_b · f·cos·L_b / pdf_b  +  w_l · f·cos·L_l / pdf_l`.
///
/// The light-sampled term's BRDF pdf and the BRDF-sampled term's light pdf are
/// supplied by the caller (`light_pdf_for_brdf_dir`, `brdf_pdf_for_light_dir`)
/// because evaluating a light's pdf for an arbitrary direction is light-shape
/// specific.  Any degenerate term contributes `0`.  The result is always finite
/// and non-negative.
#[inline]
pub fn mis_estimator_glossy(
    point: &GlossyShadingPoint,
    brdf_sample: &DirectionSample,
    light_pdf_for_brdf_dir: f32,
    light_sample: &DirectionSample,
    brdf_pdf_for_light_dir: f32,
) -> Vec3 {
    let mut out = Vec3::ZERO;

    // BRDF-sampled term, weighted against the light pdf for the same direction.
    if brdf_sample.pdf.is_finite() && brdf_sample.pdf > 0.0 {
        let w = power_heuristic(brdf_sample.pdf, light_pdf_for_brdf_dir);
        if w > 0.0 {
            out += brdf_sample_estimate(point, brdf_sample) * w;
        }
    }

    // Light-sampled term, weighted against the BRDF pdf for the same direction.
    if light_sample.pdf.is_finite() && light_sample.pdf > 0.0 {
        let w = power_heuristic(light_sample.pdf, brdf_pdf_for_light_dir);
        if w > 0.0 {
            out += brdf_sample_estimate(point, light_sample) * w;
        }
    }

    if out.is_finite() { out.max(Vec3::ZERO) } else { Vec3::ZERO }
}

/// Rec. 709 luminance of a glossy MIS estimate, re-exported for callers that
/// need a scalar magnitude (e.g. firefly clamping).
#[inline]
pub fn mis_estimate_luminance(estimate: Vec3) -> f32 {
    luminance(estimate)
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

    #[test]
    fn balance_pair_partitions_unity() {
        let a = 3.0f32;
        let b = 7.0f32;
        let wa = balance_heuristic_pair(a, b);
        let wb = balance_heuristic_pair(b, a);
        assert!((wa + wb - 1.0).abs() < 1e-6, "wa={wa} wb={wb}");
        assert!((wa - 0.3).abs() < 1e-6);
    }

    #[test]
    fn power_heuristic_partitions_unity_and_sharpens() {
        let a = 3.0f32;
        let b = 1.0f32;
        let wa = power_heuristic(a, b);
        let wb = power_heuristic(b, a);
        assert!((wa + wb - 1.0).abs() < 1e-6, "wa={wa} wb={wb}");
        // Power heuristic favours the stronger technique more than balance.
        let bal = balance_heuristic_pair(a, b);
        assert!(wa > bal, "power {wa} should exceed balance {bal}");
    }

    #[test]
    fn single_technique_gets_full_weight() {
        // When the other pdf is zero, the active technique owns the whole weight.
        assert!((power_heuristic(2.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((balance_heuristic_pair(2.0, 0.0) - 1.0).abs() < 1e-6);
        // A dead technique gets nothing.
        assert_eq!(power_heuristic(0.0, 5.0), 0.0);
    }

    #[test]
    fn count_aware_reduces_to_pairwise_when_equal() {
        let p = power_heuristic(4.0, 2.0);
        let pc = power_heuristic_counts(1.0, 4.0, 1.0, 2.0);
        assert!((p - pc).abs() < 1e-6);
        let b = balance_heuristic_pair(4.0, 2.0);
        let bc = balance_heuristic_counts(1.0, 4.0, 1.0, 2.0);
        assert!((b - bc).abs() < 1e-6);
    }

    #[test]
    fn degenerate_weights_are_zero() {
        assert_eq!(power_heuristic(f32::NAN, 1.0), 0.0);
        assert_eq!(balance_heuristic_pair(1.0, f32::NAN), 1.0); // NaN->0 denom term
        assert_eq!(power_heuristic(-1.0, -1.0), 0.0);
        assert_eq!(area_to_solid_angle_pdf(1.0, 2.0, 0.0), 0.0);
    }

    #[test]
    fn area_to_solid_angle_jacobian() {
        // pdf_A = 0.25, dist = 2 (dist^2 = 4), cos = 0.5 -> 0.25 * 4 / 0.5 = 2.
        let pdf = area_to_solid_angle_pdf(0.25, 2.0, 0.5);
        assert!((pdf - 2.0).abs() < 1e-6, "pdf={pdf}");
    }

    #[test]
    fn brdf_pdf_matches_lobe_peak() {
        let point = make_point(0.2);
        // Mirror direction should carry a high BRDF pdf.
        let n = point.normal;
        let v = point.view;
        let refl = (2.0 * v.dot(n) * n - v).normalize();
        let pdf_peak = brdf_pdf_glossy(&point, refl);
        // A grazing off-axis direction has lower pdf.
        let off = Vec3::new(0.9, 0.0, 0.436).normalize();
        let pdf_off = brdf_pdf_glossy(&point, off);
        assert!(pdf_peak > pdf_off, "peak={pdf_peak} off={pdf_off}");
        assert!(pdf_peak.is_finite() && pdf_peak > 0.0);
    }

    #[test]
    fn mis_estimator_combines_both_terms() {
        let point = make_point(0.3);
        let n = point.normal;
        let v = point.view;
        let refl = (2.0 * v.dot(n) * n - v).normalize();

        // BRDF-sampled direction near the lobe peak.
        let brdf_pdf = brdf_pdf_glossy(&point, refl);
        let brdf_sample = DirectionSample {
            wi: refl,
            radiance: Vec3::splat(2.0),
            pdf: brdf_pdf,
        };
        // Light-sampled direction (same bright light, modest light pdf).
        let light_pdf = 1.5f32;
        let light_sample = DirectionSample {
            wi: refl,
            radiance: Vec3::splat(2.0),
            pdf: light_pdf,
        };
        // Cross pdfs for the same (shared) direction.
        let light_pdf_for_brdf_dir = light_pdf;
        let brdf_pdf_for_light_dir = brdf_pdf;

        let est = mis_estimator_glossy(
            &point,
            &brdf_sample,
            light_pdf_for_brdf_dir,
            &light_sample,
            brdf_pdf_for_light_dir,
        );
        assert!(est.is_finite() && est.length() > 0.0, "est={est:?}");

        // Analytic check: for a shared direction with shared radiance, the MIS
        // sum equals a single evaluation of f*cos*L * (w_b/pdf_b + w_l/pdf_l).
        let w_b = power_heuristic(brdf_pdf, light_pdf);
        let w_l = power_heuristic(light_pdf, brdf_pdf);
        assert!((w_b + w_l - 1.0).abs() < 1e-6);
        let throughput = glossy_lobe_throughput_dir(&point, refl) * Vec3::splat(2.0);
        let expected = throughput * (w_b / brdf_pdf + w_l / light_pdf);
        assert!((est - expected).length() < 1e-4, "est={est:?} expected={expected:?}");
    }

    #[test]
    fn mis_single_technique_matches_plain_estimator() {
        // With the light technique dead (pdf 0), MIS collapses to the plain BRDF
        // estimator f*cos*L / pdf (because w_b -> 1).
        let point = make_point(0.4);
        let refl = Vec3::new(0.1, 0.2, 0.974).normalize();
        let pdf = brdf_pdf_glossy(&point, refl).max(0.5);
        let brdf_sample = DirectionSample {
            wi: refl,
            radiance: Vec3::splat(1.0),
            pdf,
        };
        let est = mis_estimator_glossy(&point, &brdf_sample, 0.0, &DirectionSample::NONE, 0.0);
        let plain = brdf_sample_estimate(&point, &brdf_sample);
        assert!((est - plain).length() < 1e-5, "est={est:?} plain={plain:?}");
    }

    #[test]
    fn degenerate_samples_contribute_nothing() {
        let point = make_point(0.3);
        let est = mis_estimator_glossy(
            &point,
            &DirectionSample::NONE,
            0.0,
            &DirectionSample::NONE,
            0.0,
        );
        assert_eq!(est, Vec3::ZERO);
    }
}
