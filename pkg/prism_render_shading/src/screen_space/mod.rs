//! Backend-neutral CPU reference for screen-space reflections (SSR).
//!
//! SSR reflects the view ray off the shaded surface, marches the reflected ray
//! across the depth pyramid to find where it re-enters the framebuffer, then
//! reprojects that hit into the previous frame's colour to supply a sharp,
//! contact-accurate specular term the prefiltered environment map cannot.  The
//! result is only trustworthy under several conditions (on-screen, facing into
//! the scene, smooth enough, not run to the march limit), so a confidence
//! weight blends it back toward the IBL lobe wherever the trace is unreliable.
//!
//! The pipeline is split into cohesive stages, each independently testable and
//! each mirrored bit-for-bit by the `ssr.wesl` GPU twin:
//!
//! * [`ray`] — reflect the view ray and project it to a screen-space ray.
//! * [`march`] — hierarchical (Hi-Z) depth-pyramid ray march.
//! * [`sample`] — deterministic GGX importance sampling (Hammersley + NDF/
//!   visibility) for the multi-ray rough-reflection lobe.
//! * [`reconstruct`] — edge-aware spatial resolve that denoises the multi-ray
//!   trace with a normal/depth/confidence bilateral kernel.
//! * [`fade`] — edge/facing/roughness/distance confidence and the IBL blend.
//! * [`temporal`] — cross-frame reprojection + neighbourhood-clip accumulation
//!   that averages the resolve over time to kill the multi-ray boil.
//!
//! End to end, [`trace_screen_space_reflection`] runs the ray build and march
//! and hands the raw hit to the confidence stage the resolve pass consumes.

mod fade;
mod march;
mod ray;
mod reconstruct;
mod sample;
mod temporal;

pub use fade::{
    blend_specular, distance_fade, edge_fade, facing_fade, reflection_mip, roughness_fade,
    smoothstep, trace_confidence, SsrConfidenceParams, SsrTraceSample,
};
pub use march::{
    march_hierarchical, DepthPyramid, SsrMarchConfig, SsrMarchResult,
};
pub use ray::{
    build_screen_ray, project_view_to_screen, reflect, reverse_z_perspective, ScreenRay,
    ScreenSample, SsrCamera,
};
pub use reconstruct::{
    resolve_geometry_weight, resolve_reflection, SsrResolveParams, SsrResolveSample,
};
pub use sample::{
    ggx_ndf, hammersley, importance_sample_ggx, radical_inverse_vdc, smith_ggx_visibility,
};
pub use temporal::{
    accumulate_temporal, adaptive_history_weight, clip_history_to_aabb, clip_history_to_aabb_ex,
    expand_bounds, relax_box_for_confidence, reproject_prev_uv, variance_clip_box, ClipResult,
    SsrTemporalParams,
};

use bevy_math::Vec3;

/// A fully evaluated screen-space reflection trace: the raw march plus the
/// confidence weight that blends it over the IBL specular fallback.
#[derive(Clone, Copy, Debug)]
pub struct ScreenSpaceReflection {
    /// The hierarchical march result (hit flag, UV, depth, travel).
    pub march: SsrMarchResult,
    /// Combined `[0, 1]` blend confidence (`0` on a miss or fully faded trace).
    pub confidence: f32,
}

/// Builds and marches the reflection ray for one shaded surface, then folds the
/// hit through every fade into a blend-ready [`ScreenSpaceReflection`].
///
/// `position_view`/`normal_view` describe the surface in view space,
/// `roughness` is its perceptual roughness, and `max_distance` bounds the
/// view-space march length.  Returns a zero-confidence miss when the ray
/// cannot be built (surface behind the camera, degenerate reflection, etc.).
pub fn trace_screen_space_reflection(
    pyramid: &DepthPyramid,
    camera: SsrCamera,
    position_view: Vec3,
    normal_view: Vec3,
    roughness: f32,
    max_distance: f32,
    march_config: SsrMarchConfig,
    confidence_params: SsrConfidenceParams,
) -> ScreenSpaceReflection {
    let Some(ray) = build_screen_ray(camera, position_view, normal_view, max_distance) else {
        return ScreenSpaceReflection {
            march: SsrMarchResult {
                hit: false,
                uv: bevy_math::Vec2::new(-1.0, -1.0),
                depth: 0.0,
                travel: 0.0,
                iterations: 0,
            },
            confidence: 0.0,
        };
    };

    let march = march_hierarchical(
        pyramid,
        ray.start_uv,
        ray.start_depth,
        ray.end_uv,
        ray.end_depth,
        march_config,
    );

    let incident = position_view.normalize_or_zero();
    let reflection = reflect(incident, normal_view.normalize_or_zero()).normalize_or_zero();
    let sample = SsrTraceSample {
        hit: march.hit,
        hit_uv: march.uv,
        // Normalize the march travel by the clipped fraction so the distance
        // fade measures progress along the *requested* view-space length.
        travel: (march.travel * ray.clipped_fraction).clamp(0.0, 1.0),
        reflection,
        camera_to_surface: incident,
        roughness,
    };
    let confidence = trace_confidence(sample, confidence_params);

    ScreenSpaceReflection { march, confidence }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use bevy_math::{UVec2, Vec2};

    #[test]
    fn end_to_end_smooth_floor_reflects_and_trusts_the_hit() {
        // A smooth floor facing up; build the ray, then place a constant-depth
        // surface midway along its device-depth span so the march is
        // guaranteed to straddle (and therefore hit) it.
        let camera = SsrCamera {
            clip_from_view: reverse_z_perspective(
                core::f32::consts::FRAC_PI_2,
                1.0,
                0.5,
                100.0,
            ),
            near: 0.5,
        };
        let position = Vec3::new(0.0, -1.0, -4.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let ray = build_screen_ray(camera, position, normal, 10.0).expect("ray");
        // Reverse-Z: start_depth is the nearest, end_depth the farthest; a
        // surface between them is crossed by the descending ray.
        let mid = (ray.start_depth + ray.end_depth) * 0.5;
        let mip0 = vec![mid; 64 * 64];
        let pyramid = DepthPyramid::from_nearest_reduction(&mip0, UVec2::new(64, 64));

        let ssr = trace_screen_space_reflection(
            &pyramid,
            camera,
            position,
            normal,
            0.05,
            10.0,
            SsrMarchConfig::default(),
            SsrConfidenceParams::default(),
        );
        // Smooth surface + on-screen hit => usable confidence.
        assert!(ssr.march.hit);
        assert!(ssr.confidence > 0.0);
        assert!((ssr.march.depth - mid).abs() < 1.0e-2);
    }

    #[test]
    fn rough_surface_falls_back_to_ibl() {
        let camera = SsrCamera {
            clip_from_view: reverse_z_perspective(
                core::f32::consts::FRAC_PI_2,
                1.0,
                0.5,
                100.0,
            ),
            near: 0.5,
        };
        let mip0 = vec![0.95f32; 64 * 64];
        let pyramid = DepthPyramid::from_nearest_reduction(&mip0, UVec2::new(64, 64));
        let ssr = trace_screen_space_reflection(
            &pyramid,
            camera,
            Vec3::new(0.0, -1.0, -4.0),
            Vec3::new(0.0, 1.0, 0.0),
            0.9, // very rough: beyond max_roughness
            10.0,
            SsrMarchConfig::default(),
            SsrConfidenceParams::default(),
        );
        assert_eq!(ssr.confidence, 0.0, "rough surface must defer to IBL");
    }

    #[test]
    fn surface_behind_camera_is_a_zero_confidence_miss() {
        let camera = SsrCamera {
            clip_from_view: reverse_z_perspective(
                core::f32::consts::FRAC_PI_2,
                1.0,
                0.5,
                100.0,
            ),
            near: 0.5,
        };
        let mip0 = vec![0.5f32; 16 * 16];
        let pyramid = DepthPyramid::from_nearest_reduction(&mip0, UVec2::new(16, 16));
        let ssr = trace_screen_space_reflection(
            &pyramid,
            camera,
            Vec3::new(0.0, 0.0, 1.0), // behind the camera
            Vec3::Y,
            0.1,
            10.0,
            SsrMarchConfig::default(),
            SsrConfidenceParams::default(),
        );
        assert!(!ssr.march.hit);
        assert_eq!(ssr.confidence, 0.0);
        assert_eq!(ssr.march.uv, Vec2::new(-1.0, -1.0));
    }
}
