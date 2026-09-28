//! Backend-neutral CPU reference for screen-space global illumination (SSGI):
//! a single indirect *diffuse* bounce gathered from the on-screen radiance.
//!
//! Where [`super`] reflects one specular ray, SSGI integrates the cosine-
//! weighted Lambertian hemisphere.  Several rays are cast around the surface
//! normal and each is marched across the *same* Hi-Z depth pyramid the
//! reflection trace uses ([`super::march_hierarchical`] over
//! [`super::DepthPyramid`]).  A ray that intersects on-screen geometry picks up
//! that surface's already-shaded radiance (colour bleeding + one indirect
//! bounce); a ray that escapes the framebuffer contributes the distant/sky
//! radiance the caller supplies (the IBL/SH ambient fallback), so energy stays
//! consistent with the ambient term SSGI augments.
//!
//! Cosine-weighted importance sampling is what makes the estimator cheap: the
//! Monte-Carlo estimator of the Lambertian integral
//! `∫ (albedo/π) L_i (n·l) dω` with pdf `(n·l)/π` collapses to
//! `albedo · mean(L_i)`, so [`gather_indirect_diffuse`] is a plain average of
//! the per-ray radiance scaled by albedo — no per-sample cosine or pdf divide.
//!
//! Determinism mirrors the reflection sampler: directions come from the fixed
//! low-discrepancy [`super::hammersley`] set (not a per-frame RNG) and every
//! transcendental routes through [`bevy_math::ops`], so this golden and the
//! future WESL twin pick byte-identical directions for a given
//! `(sample_index, sample_count, normal)`.
//!
//! Stages, each independently testable and each mirrored by the WESL twin:
//! * hemisphere sampling — [`cosine_sample_direction`]
//! * per-ray screen projection — [`build_hemisphere_ray`]
//! * trace — [`trace_indirect_ray`] (reuses the reflection march)
//! * gather / estimator — [`gather_indirect_diffuse`]

use bevy_math::{ops, Vec2, Vec3};
use core::f32::consts::TAU;

use super::march::{march_hierarchical, DepthPyramid, SsrMarchConfig, SsrMarchResult};
use super::ray::{project_view_to_screen, ScreenRay, SsrCamera};
use super::sample::orthonormal_basis;

/// Numerical floor shared by the ray-clipping and normalisation guards.
const EPS: f32 = 1.0e-4;

/// Maps a low-discrepancy 2D sample `xi ∈ [0, 1)²` to a cosine-weighted
/// hemisphere direction around unit `normal` (Malley's method).
///
/// The radius of a concentric-disc sample is `sqrt(xi.x)` and its height is
/// `sqrt(1 - xi.x)`, which is exactly a `cos`-distributed elevation; the disc
/// is oriented in the tangent plane by [`orthonormal_basis`] so the returned
/// direction lies in the hemisphere about `normal` with pdf `(n·l)/π`.
///
/// Returns [`Vec3::ZERO`] for a degenerate `normal` so callers can skip the ray.
pub fn cosine_sample_direction(xi: Vec2, normal: Vec3) -> Vec3 {
    let n = normal.normalize_or_zero();
    if n == Vec3::ZERO {
        return Vec3::ZERO;
    }
    let u = xi.x.clamp(0.0, 1.0);
    let radius = ops::sqrt(u);
    let phi = TAU * xi.y;
    let x = radius * ops::cos(phi);
    let y = radius * ops::sin(phi);
    let z = ops::sqrt((1.0 - u).max(0.0));
    let (tangent, bitangent) = orthonormal_basis(n);
    (tangent * x + bitangent * y + n * z).normalize_or_zero()
}

/// Builds the screen-space ray for a surface at `position_view` shooting along
/// unit `direction_view` (a hemisphere sample from [`cosine_sample_direction`]),
/// marching up to `max_distance` view-space units.
///
/// This is the reflection [`super::build_screen_ray`] with the reflect step
/// removed: it takes the traced direction directly so the diffuse hemisphere
/// integrator can reuse the identical near-plane clip + reverse-Z projection.
/// Returns `None` when the surface is at/behind the camera or the ray is
/// entirely behind the near plane (nothing on-screen to gather).
pub fn build_hemisphere_ray(
    camera: SsrCamera,
    position_view: Vec3,
    direction_view: Vec3,
    max_distance: f32,
) -> Option<ScreenRay> {
    if position_view.z >= -camera.near {
        return None;
    }
    let dir = direction_view.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }

    let requested = max_distance.max(1.0e-3);
    let mut travel = requested;

    // Clip the far endpoint to the near plane so it never projects behind the
    // camera.  View space looks down -Z, so "in front" means z <= -near.
    let plane = -camera.near;
    let end_view = position_view + dir * travel;
    if end_view.z > plane {
        let denom = dir.z;
        if denom.abs() <= 1.0e-6 {
            return None;
        }
        let clipped = (plane - position_view.z) / denom;
        if clipped <= EPS {
            return None;
        }
        travel = clipped;
    }

    let clipped_fraction = (travel / requested).clamp(EPS, 1.0);
    let end_view = position_view + dir * travel;

    let start = project_view_to_screen(camera.clip_from_view, position_view);
    let end = project_view_to_screen(camera.clip_from_view, end_view);
    if start.clip_w <= 0.0 || end.clip_w <= 0.0 {
        return None;
    }

    Some(ScreenRay {
        start_uv: start.uv,
        start_depth: start.device_depth,
        end_uv: end.uv,
        end_depth: end.device_depth,
        clipped_fraction,
    })
}

/// Traces one hemisphere sample end to end: build the screen ray, then march it
/// across the depth `pyramid`.  Returns `None` when the ray degenerates before
/// the march (off-screen origin / behind the near plane), otherwise the raw
/// march result whose `uv` the caller samples the scene colour at.
pub fn trace_indirect_ray(
    camera: SsrCamera,
    pyramid: &DepthPyramid,
    position_view: Vec3,
    direction_view: Vec3,
    max_distance: f32,
    config: SsrMarchConfig,
) -> Option<SsrMarchResult> {
    let ray = build_hemisphere_ray(camera, position_view, direction_view, max_distance)?;
    Some(march_hierarchical(
        pyramid,
        ray.start_uv,
        ray.start_depth,
        ray.end_uv,
        ray.end_depth,
        config,
    ))
}

/// Tunable weights for the indirect-diffuse gather.
#[derive(Clone, Copy, Debug)]
pub struct SsgiParams {
    /// Multiplier applied to the gathered indirect radiance (artistic gain).
    pub intensity: f32,
    /// View-space distance a hit fades out over, relative to the ray's
    /// `max_distance`: hits at `travel >= distance_falloff` contribute nothing,
    /// which suppresses the bright halo a very distant hit would otherwise
    /// bleed onto a near surface.  In `(0, 1]`; `1.0` disables the falloff.
    pub distance_falloff: f32,
    /// How strongly near hits darken the sky term into an occlusion factor, in
    /// `[0, 1]`.  `0` reports full sky (no SSGI occlusion); `1` fully removes
    /// the sky where every ray was blocked.
    pub occlusion_strength: f32,
}

impl Default for SsgiParams {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            distance_falloff: 1.0,
            occlusion_strength: 1.0,
        }
    }
}

/// Per-ray outcome fed to [`gather_indirect_diffuse`].
///
/// The trace and the scene-colour sample are kept out of the estimator (exactly
/// as the reflection golden separates the march from the previous-frame colour
/// fetch) so the gather stays a pure, deterministic function of the radiances.
#[derive(Clone, Copy, Debug)]
pub struct SsgiRaySample {
    /// `true` when the ray could be built and marched (an off-screen or
    /// behind-camera origin yields `valid = false` and is ignored so it neither
    /// adds radiance nor dilutes the average).
    pub valid: bool,
    /// `true` when the march intersected on-screen geometry; `false` means the
    /// ray escaped the framebuffer and should take the `sky_radiance` fallback.
    pub hit: bool,
    /// Linear HDR radiance the ray gathered: the scene colour at the hit UV for
    /// a hit; ignored for a miss (the gather substitutes `sky_radiance`).
    pub radiance: Vec3,
    /// Normalized ray parameter `[0, 1]` at the hit, driving the distance
    /// falloff.  Unused on a miss.
    pub travel: f32,
}

/// Result of the cosine-weighted indirect-diffuse gather.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsgiGather {
    /// Indirect diffuse radiance to add to (or blend with) the surface's
    /// ambient term: `albedo · mean(L_i) · intensity`.
    pub indirect: Vec3,
    /// Fraction of valid rays that hit near geometry, in `[0, 1]` — a
    /// screen-space ambient-occlusion estimate the caller may reuse.
    pub occlusion: f32,
    /// Trust weight in `[0, 1]`: the fraction of the requested rays that were
    /// valid.  Drives the blend back toward pure IBL where SSGI is unreliable
    /// (grazing pixels, screen edges) just like the reflection confidence.
    pub confidence: f32,
}

/// Falloff weight for a hit at normalized `travel` under `distance_falloff`.
///
/// Full weight for near hits, ramping linearly to zero as `travel` approaches
/// `distance_falloff` so distant colour bleed cannot form a hard halo.
fn distance_weight(travel: f32, distance_falloff: f32) -> f32 {
    let cutoff = distance_falloff.clamp(EPS, 1.0);
    (1.0 - (travel / cutoff)).clamp(0.0, 1.0)
}

/// Cosine-weighted Monte-Carlo gather of the indirect diffuse hemisphere.
///
/// With cosine-weighted directions the Lambertian estimator is the plain mean
/// of per-ray radiance times `albedo`.  A hit contributes its (distance-faded)
/// scene colour; a miss contributes `sky_radiance` (the IBL/SH ambient the
/// caller already computes) so unshadowed directions keep the ambient look.
/// Invalid rays are dropped from the average entirely.
///
/// `samples` is the full requested ray set (including invalid entries) so
/// `confidence` can report how many survived.
pub fn gather_indirect_diffuse(
    albedo: Vec3,
    sky_radiance: Vec3,
    samples: &[SsgiRaySample],
    params: SsgiParams,
) -> SsgiGather {
    let requested = samples.len().max(1) as f32;
    let mut valid = 0.0f32;
    let mut hits = 0.0f32;
    let mut radiance_sum = Vec3::ZERO;

    for sample in samples {
        if !sample.valid {
            continue;
        }
        valid += 1.0;
        if sample.hit {
            hits += 1.0;
            let weight = distance_weight(sample.travel, params.distance_falloff);
            // A faded hit blends toward the sky term over the missing weight so
            // a distant hit degrades to the ambient fallback rather than to
            // black (which would punch a dark hole into the gather).
            radiance_sum += sample.radiance * weight + sky_radiance * (1.0 - weight);
        } else {
            radiance_sum += sky_radiance;
        }
    }

    if valid <= 0.0 {
        // No usable ray: fall back to the pure ambient term with zero trust so
        // the caller keeps its IBL/SH result untouched.
        return SsgiGather {
            indirect: albedo * sky_radiance * params.intensity,
            occlusion: 0.0,
            confidence: 0.0,
        };
    }

    let mean_radiance = radiance_sum / valid;
    let occlusion = (hits / valid) * params.occlusion_strength.clamp(0.0, 1.0);
    let confidence = (valid / requested).clamp(0.0, 1.0);

    SsgiGather {
        indirect: albedo * mean_radiance * params.intensity,
        occlusion,
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screen_space::hammersley;
    use crate::screen_space::reverse_z_perspective;
    use bevy_math::UVec2;

    fn test_camera() -> SsrCamera {
        SsrCamera {
            clip_from_view: reverse_z_perspective(core::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0),
            near: 0.1,
        }
    }

    #[test]
    fn cosine_samples_lie_in_the_hemisphere_about_the_normal() {
        let normal = Vec3::new(0.3, 0.7, 0.65).normalize();
        for i in 0..64u32 {
            let dir = cosine_sample_direction(hammersley(i, 64), normal);
            assert!((dir.length() - 1.0).abs() < 1.0e-4, "unit length");
            assert!(dir.dot(normal) >= -1.0e-4, "same hemisphere as normal");
        }
    }

    #[test]
    fn cosine_samples_average_toward_the_normal() {
        // A cosine-weighted set is biased toward the pole, so the mean sample
        // direction should align with the normal (not scatter uniformly).
        let normal = Vec3::Z;
        let n = 256u32;
        let mut mean = Vec3::ZERO;
        for i in 0..n {
            mean += cosine_sample_direction(hammersley(i, n), normal);
        }
        mean /= n as f32;
        assert!(mean.dot(normal) > 0.4, "mean tilts toward the normal: {mean:?}");
        assert!(mean.x.abs() < 0.05 && mean.y.abs() < 0.05, "azimuth cancels");
    }

    #[test]
    fn degenerate_normal_yields_zero_direction() {
        assert_eq!(cosine_sample_direction(Vec2::new(0.3, 0.6), Vec3::ZERO), Vec3::ZERO);
    }

    #[test]
    fn hemisphere_ray_rejects_behind_camera_origin() {
        let camera = test_camera();
        // z = +1 is behind the camera (view looks down -z).
        let ray = build_hemisphere_ray(camera, Vec3::new(0.0, 0.0, 1.0), Vec3::Y, 5.0);
        assert!(ray.is_none());
    }

    #[test]
    fn hemisphere_ray_projects_a_forward_sample() {
        let camera = test_camera();
        let position = Vec3::new(0.0, 0.0, -2.0);
        let direction = Vec3::new(0.2, 0.3, -0.5).normalize();
        let ray = build_hemisphere_ray(camera, position, direction, 4.0)
            .expect("forward ray should project");
        assert!(ray.clipped_fraction > 0.0 && ray.clipped_fraction <= 1.0);
        // The origin projects near screen centre for an on-axis point.
        assert!((ray.start_uv - Vec2::splat(0.5)).length() < 1.0e-3);
    }

    fn hit(radiance: Vec3, travel: f32) -> SsgiRaySample {
        SsgiRaySample { valid: true, hit: true, radiance, travel }
    }
    fn miss() -> SsgiRaySample {
        SsgiRaySample { valid: true, hit: false, radiance: Vec3::ZERO, travel: 1.0 }
    }

    #[test]
    fn full_hit_set_bleeds_colour_scaled_by_albedo() {
        let albedo = Vec3::new(0.5, 0.5, 0.5);
        let bounce = Vec3::new(1.0, 0.0, 0.0);
        let samples = [hit(bounce, 0.0), hit(bounce, 0.0), hit(bounce, 0.0), hit(bounce, 0.0)];
        let g = gather_indirect_diffuse(albedo, Vec3::ZERO, &samples, SsgiParams::default());
        // Near hits (travel 0) carry full weight: indirect = albedo * bounce.
        assert!((g.indirect - albedo * bounce).length() < 1.0e-5);
        assert!((g.occlusion - 1.0).abs() < 1.0e-5);
        assert!((g.confidence - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn all_miss_falls_back_to_sky_radiance() {
        let albedo = Vec3::splat(0.8);
        let sky = Vec3::new(0.2, 0.4, 0.6);
        let samples = [miss(), miss(), miss(), miss()];
        let g = gather_indirect_diffuse(albedo, sky, &samples, SsgiParams::default());
        assert!((g.indirect - albedo * sky).length() < 1.0e-5, "misses take the sky term");
        assert!(g.occlusion.abs() < 1.0e-5, "no occlusion when every ray escapes");
    }

    #[test]
    fn distant_hits_fade_toward_the_sky_not_to_black() {
        let albedo = Vec3::ONE;
        let sky = Vec3::splat(0.5);
        let params = SsgiParams { distance_falloff: 1.0, ..Default::default() };
        // A hit at the far end (travel ~1) should have (near) zero hit weight and
        // read the sky, never darker than the sky floor.
        let far = [hit(Vec3::ZERO, 1.0)];
        let g = gather_indirect_diffuse(albedo, sky, &far, params);
        assert!((g.indirect - sky).length() < 1.0e-4, "far hit degrades to sky: {:?}", g.indirect);
    }

    #[test]
    fn invalid_rays_are_dropped_and_lower_confidence() {
        let albedo = Vec3::ONE;
        let sky = Vec3::splat(0.3);
        let samples = [
            hit(Vec3::new(1.0, 1.0, 1.0), 0.0),
            SsgiRaySample { valid: false, hit: false, radiance: Vec3::ZERO, travel: 0.0 },
            SsgiRaySample { valid: false, hit: false, radiance: Vec3::ZERO, travel: 0.0 },
            SsgiRaySample { valid: false, hit: false, radiance: Vec3::ZERO, travel: 0.0 },
        ];
        let g = gather_indirect_diffuse(albedo, sky, &samples, SsgiParams::default());
        // Only one valid ray -> mean is that hit; confidence = 1/4.
        assert!((g.indirect - Vec3::ONE).length() < 1.0e-5);
        assert!((g.confidence - 0.25).abs() < 1.0e-5);
    }

    #[test]
    fn no_valid_rays_keeps_pure_ambient_with_zero_confidence() {
        let albedo = Vec3::splat(0.6);
        let sky = Vec3::new(0.1, 0.2, 0.3);
        let samples = [SsgiRaySample { valid: false, hit: false, radiance: Vec3::ZERO, travel: 0.0 }];
        let g = gather_indirect_diffuse(albedo, sky, &samples, SsgiParams::default());
        assert!((g.indirect - albedo * sky).length() < 1.0e-5);
        assert_eq!(g.confidence, 0.0);
    }

    #[test]
    fn trace_indirect_ray_marches_a_flat_wall() {
        // A flat wall one unit behind everything: a ray pointed into it should
        // return a hit within the framebuffer.
        let camera = test_camera();
        let size = UVec2::new(64, 64);
        // Constant device depth plane (reverse-Z): fill mip0 with a mid depth.
        let mip0 = vec![0.5f32; (size.x * size.y) as usize];
        let pyramid = DepthPyramid::from_nearest_reduction(&mip0, size);
        let position = Vec3::new(0.0, 0.0, -2.0);
        let direction = Vec3::new(0.1, 0.0, -1.0).normalize();
        let result = trace_indirect_ray(
            camera,
            &pyramid,
            position,
            direction,
            8.0,
            SsrMarchConfig::default(),
        );
        assert!(result.is_some(), "forward ray should trace");
    }

    #[test]
    fn intensity_scales_the_gathered_radiance() {
        let albedo = Vec3::ONE;
        let bounce = Vec3::new(0.4, 0.4, 0.4);
        let samples = [hit(bounce, 0.0)];
        let base = gather_indirect_diffuse(albedo, Vec3::ZERO, &samples, SsgiParams::default());
        let boosted = gather_indirect_diffuse(
            albedo,
            Vec3::ZERO,
            &samples,
            SsgiParams { intensity: 2.0, ..Default::default() },
        );
        assert!((boosted.indirect - base.indirect * 2.0).length() < 1.0e-5);
    }
}
