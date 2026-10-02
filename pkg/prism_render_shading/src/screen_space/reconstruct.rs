//! Spatial reconstruction (bilateral resolve) of the noisy multi-ray SSR trace.
//!
//! The GGX-importance-sampled trace in [`super::sample`] shoots several rays per
//! pixel, but on rough surfaces even a handful of rays leaves visible noise.
//! Rather than pay for hundreds of rays, AAA screen-space reflection resolves
//! the trace with an edge-aware neighbourhood filter: every nearby pixel's
//! reflection is a valid extra sample of the *same* lobe, so a bilateral blur
//! that respects surface normals and depth (and each neighbour's own trace
//! confidence) collapses the noise without bleeding across silhouettes.
//!
//! This module is the CPU golden for that resolve.  The kernel is expressed as
//! pure weight functions so the `ssr_resolve.wesl` twin shares one reference:
//! the shader gathers a fixed pixel neighbourhood, computes the identical
//! weights, and averages.  The per-pixel roughness scaling of the spatial sigma
//! lives on the GPU (a smooth surface wants a tight kernel, a rough one a wide
//! one); the golden takes the already-scaled sigma so it stays a pure,
//! deterministic weight reference.  Every transcendental routes through
//! [`bevy_math::ops`] for cross-platform determinism.

use bevy_math::{ops, Vec3, Vec4};

/// One neighbourhood tap fed to the resolve: a neighbour pixel's reflected
/// radiance and trace confidence plus the surface attributes the bilateral
/// weights compare against the centre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrResolveSample {
    /// The neighbour's reflected radiance (`ssr_out.rgb`).
    pub reflected: Vec3,
    /// The neighbour's trace confidence (`ssr_out.a`, in `[0, 1]`).
    pub confidence: f32,
    /// The neighbour's view-space surface normal (unit length).
    pub normal: Vec3,
    /// The neighbour's linear/view depth used for the depth bilateral term.
    pub depth: f32,
    /// Squared pixel distance from the centre (`dx*dx + dy*dy`) for the spatial
    /// Gaussian.  The centre tap passes `0`.
    pub dist_sq: f32,
}

/// Edge-aware kernel tunables shared by the golden and the `ssr_resolve.wesl`
/// twin.  `spatial_sigma` is already roughness-scaled by the caller (the GPU
/// widens it for rough pixels); the golden treats it as a fixed Gaussian sigma.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrResolveParams {
    /// Gaussian sigma of the spatial term, in pixels (already roughness-scaled).
    pub spatial_sigma: f32,
    /// Exponent applied to `max(dot(n, n_centre), 0)`; larger rejects tilted
    /// neighbours harder, keeping reflections off curved silhouettes.
    pub normal_power: f32,
    /// Relative depth tolerance: the depth term is `exp(-|d - d0| / (sigma *
    /// max(|d0|, eps)))`, so it scales with distance from the camera.
    pub depth_sigma: f32,
}

impl Default for SsrResolveParams {
    fn default() -> Self {
        Self {
            spatial_sigma: 2.0,
            normal_power: 8.0,
            depth_sigma: 0.05,
        }
    }
}

/// Geometry-only bilateral weight for one tap (spatial x normal x depth), *not*
/// yet folded with the neighbour's confidence.
///
/// Kept separate so the resolve can accumulate the geometry weight (how much a
/// neighbour *should* contribute) apart from its confidence (how much valid
/// radiance it actually carries), letting low-confidence neighbourhoods lower
/// the resolved confidence instead of silently inventing radiance.
pub fn resolve_geometry_weight(
    params: &SsrResolveParams,
    center_normal: Vec3,
    center_depth: f32,
    sample: &SsrResolveSample,
) -> f32 {
    let sigma = params.spatial_sigma.max(1.0e-4);
    let spatial = ops::exp(-sample.dist_sq / (2.0 * sigma * sigma));

    let n_dot = center_normal.dot(sample.normal).max(0.0);
    let normal = ops::powf(n_dot, params.normal_power.max(0.0));

    let scale = params.depth_sigma.max(1.0e-4) * center_depth.abs().max(1.0e-4);
    let depth = ops::exp(-(sample.depth - center_depth).abs() / scale);

    (spatial * normal * depth).max(0.0)
}

/// Resolves the reflection at one pixel from its neighbourhood taps.
///
/// Returns `rgb` = the confidence-weighted mean reflected radiance and `a` = the
/// resolved confidence (the geometry-weighted mean of the taps' confidences),
/// so a neighbourhood of mostly-missed rays yields a low-confidence resolve the
/// composite then leans toward the IBL fallback for.  An empty or fully
/// zero-weight neighbourhood returns all zeros.
pub fn resolve_reflection<'a, I>(
    params: &SsrResolveParams,
    center_normal: Vec3,
    center_depth: f32,
    samples: I,
) -> Vec4
where
    I: IntoIterator<Item = &'a SsrResolveSample>,
{
    let mut color = Vec3::ZERO;
    // Sum of geometry weights (denominator for the resolved confidence).
    let mut geom_sum = 0.0;
    // Sum of geometry * confidence (denominator for the resolved radiance).
    let mut radiance_weight_sum = 0.0;

    for sample in samples {
        let geom = resolve_geometry_weight(params, center_normal, center_depth, sample);
        let radiance_weight = geom * sample.confidence.clamp(0.0, 1.0);
        color += sample.reflected * radiance_weight;
        geom_sum += geom;
        radiance_weight_sum += radiance_weight;
    }

    let reflected = if radiance_weight_sum > 1.0e-8 {
        color / radiance_weight_sum
    } else {
        Vec3::ZERO
    };
    let confidence = if geom_sum > 1.0e-8 {
        (radiance_weight_sum / geom_sum).clamp(0.0, 1.0)
    } else {
        0.0
    };
    reflected.extend(confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tap(
        reflected: Vec3,
        confidence: f32,
        normal: Vec3,
        depth: f32,
        dist_sq: f32,
    ) -> SsrResolveSample {
        SsrResolveSample {
            reflected,
            confidence,
            normal,
            depth,
            dist_sq,
        }
    }

    #[test]
    fn single_matching_center_tap_passes_through() {
        let n = Vec3::Z;
        let params = SsrResolveParams::default();
        let s = tap(Vec3::new(0.4, 0.6, 0.8), 1.0, n, 5.0, 0.0);
        let out = resolve_reflection(&params, n, 5.0, [&s]);
        assert!((out.truncate() - s.reflected).length() < 1.0e-5);
        assert!((out.w - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn a_tilted_neighbour_is_rejected_by_the_normal_term() {
        let n = Vec3::Z;
        let params = SsrResolveParams::default();
        let aligned = resolve_geometry_weight(&params, n, 1.0, &tap(Vec3::ONE, 1.0, n, 1.0, 1.0));
        let tilted = resolve_geometry_weight(
            &params,
            n,
            1.0,
            &tap(
                Vec3::ONE,
                1.0,
                Vec3::new(0.7, 0.0, 0.7).normalize(),
                1.0,
                1.0,
            ),
        );
        assert!(
            tilted < aligned * 0.25,
            "tilted {tilted} vs aligned {aligned}"
        );
    }

    #[test]
    fn a_depth_discontinuity_is_rejected() {
        let n = Vec3::Z;
        let params = SsrResolveParams::default();
        let near = resolve_geometry_weight(&params, n, 10.0, &tap(Vec3::ONE, 1.0, n, 10.05, 1.0));
        let far = resolve_geometry_weight(&params, n, 10.0, &tap(Vec3::ONE, 1.0, n, 40.0, 1.0));
        assert!(far < near * 0.1, "far {far} vs near {near}");
    }

    #[test]
    fn distant_taps_are_down_weighted_by_the_spatial_term() {
        let n = Vec3::Z;
        let params = SsrResolveParams::default();
        let close = resolve_geometry_weight(&params, n, 1.0, &tap(Vec3::ONE, 1.0, n, 1.0, 1.0));
        let distant = resolve_geometry_weight(&params, n, 1.0, &tap(Vec3::ONE, 1.0, n, 64.0, 1.0));
        assert!(distant < close, "distant {distant} vs close {close}");
    }

    #[test]
    fn zero_confidence_neighbourhood_resolves_to_zero_confidence() {
        let n = Vec3::Z;
        let params = SsrResolveParams::default();
        let taps = [
            tap(Vec3::ONE, 0.0, n, 1.0, 0.0),
            tap(Vec3::ONE, 0.0, n, 1.0, 1.0),
        ];
        let out = resolve_reflection(&params, n, 1.0, taps.iter());
        assert_eq!(out.truncate(), Vec3::ZERO);
        assert!(out.w.abs() < 1.0e-6);
    }

    #[test]
    fn confident_taps_dominate_the_resolved_radiance() {
        let n = Vec3::Z;
        let params = SsrResolveParams::default();
        // A confident red centre and an ignored (zero-confidence) green tap: the
        // resolve must return essentially the red radiance.
        let taps = [
            tap(Vec3::new(1.0, 0.0, 0.0), 1.0, n, 1.0, 0.0),
            tap(Vec3::new(0.0, 1.0, 0.0), 0.0, n, 1.0, 1.0),
        ];
        let out = resolve_reflection(&params, n, 1.0, taps.iter());
        assert!((out.truncate() - Vec3::new(1.0, 0.0, 0.0)).length() < 1.0e-5);
    }

    #[test]
    fn resolve_is_finite_and_bounded() {
        let n = Vec3::new(0.2, 0.3, 0.9).normalize();
        let params = SsrResolveParams::default();
        let taps = [
            tap(Vec3::new(2.0, 3.0, 4.0), 0.8, n, 7.0, 0.0),
            tap(Vec3::new(1.0, 1.0, 1.0), 0.5, Vec3::Z, 7.2, 2.0),
            tap(Vec3::new(0.0, 0.0, 0.0), 0.2, n, 6.9, 4.0),
        ];
        let out = resolve_reflection(&params, n, 7.0, taps.iter());
        assert!(out.is_finite());
        assert!((0.0..=1.0).contains(&out.w));
    }
}
