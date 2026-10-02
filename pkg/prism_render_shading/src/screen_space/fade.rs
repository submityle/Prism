//! Confidence and fade weights that gate a raw SSR hit before it is blended
//! over the image-based specular fallback.
//!
//! A screen-space trace only carries valid radiance under several conditions:
//! the hit must be inside the frame, the reflection ray must travel *into* the
//! scene rather than back toward the camera, the surface must be smooth enough
//! for a single mirror ray to approximate the GGX lobe, and the march must not
//! have run to its far limit.  Each of these is expressed as an independent
//! weight in `[0, 1]`; their product is the trace confidence used to `mix`
//! between the prefiltered IBL specular (confidence `0`) and the reprojected
//! screen colour (confidence `1`).
//!
//! Every weight is a pure function of scalars/vectors so the CPU golden and the
//! WESL twin (`ssr.wesl`) share one reference.  Transcendentals route through
//! [`bevy_math::ops`] for cross-platform determinism.

use bevy_math::{ops, Vec2, Vec3};

/// Clamps `value` to the unit range.
fn saturate(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

/// Hermite `smoothstep` matching the WGSL/HLSL intrinsic.
///
/// Returns `0` at or below `edge0`, `1` at or above `edge1`, and a smooth
/// `3t^2 - 2t^3` ramp in between.  `edge0 == edge1` degenerates to a step.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 == edge1 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = saturate((x - edge0) / (edge1 - edge0));
    t * t * (3.0 - 2.0 * t)
}

/// Fades the trace toward `0` as the hit approaches any screen border.
///
/// `fade_start` is the border margin in UV units (each axis) where the ramp
/// begins; a hit at the exact edge returns `0`, a hit deeper than `fade_start`
/// from every border returns `1`.  This suppresses the smeared reflections
/// that appear when screen-space data runs off the framebuffer.
pub fn edge_fade(uv: Vec2, fade_start: f32) -> f32 {
    let fade = fade_start.max(1.0e-4);
    // Distance to the nearest border on each axis, then the closest overall.
    let dx = uv.x.min(1.0 - uv.x);
    let dy = uv.y.min(1.0 - uv.y);
    let nearest = dx.min(dy);
    smoothstep(0.0, fade, nearest)
}

/// Fades rays that point back toward the camera.
///
/// `reflection` and `camera_to_surface` are unit view-space directions; the
/// dot product is positive when the reflected ray travels deeper into the
/// scene (reliable) and negative when it heads back at the viewer (the screen
/// holds no data for it).  The ramp keeps grazing rays partially trusted while
/// killing the fully back-facing ones.
pub fn facing_fade(reflection: Vec3, camera_to_surface: Vec3) -> f32 {
    let alignment = reflection.dot(camera_to_surface);
    smoothstep(-0.15, 0.35, alignment)
}

/// Fades SSR out as the surface roughens past what a single mirror ray can
/// represent.
///
/// Below `full_roughness` the trace is fully trusted; above `max_roughness`
/// it is abandoned to the prefiltered IBL lobe.  Between them the confidence
/// ramps down smoothly so the transition to IBL is not visible as a seam.
pub fn roughness_fade(roughness: f32, full_roughness: f32, max_roughness: f32) -> f32 {
    let lo = full_roughness.min(max_roughness);
    let hi = max_roughness.max(full_roughness + 1.0e-4);
    1.0 - smoothstep(lo, hi, roughness)
}

/// Fades the trace toward its far march limit.
///
/// `travel` is the normalized ray parameter in `[0, 1]` (`0` at the shaded
/// pixel, `1` at the maximum march distance).  Confidence stays flat until
/// `fade_start`, then ramps to `0` at the limit, hiding the hard cut where the
/// march budget is exhausted.
pub fn distance_fade(travel: f32, fade_start: f32) -> f32 {
    let start = fade_start.clamp(0.0, 1.0);
    1.0 - smoothstep(start, 1.0, saturate(travel))
}

/// Parameters controlling how the individual SSR fades are combined.
#[derive(Clone, Copy, Debug)]
pub struct SsrConfidenceParams {
    /// UV border margin where [`edge_fade`] begins ramping.
    pub edge_fade_start: f32,
    /// Perceptual roughness below which SSR is fully trusted.
    pub full_roughness: f32,
    /// Perceptual roughness at/above which SSR is fully replaced by IBL.
    pub max_roughness: f32,
    /// Normalized travel where [`distance_fade`] begins ramping.
    pub distance_fade_start: f32,
}

impl Default for SsrConfidenceParams {
    fn default() -> Self {
        Self {
            edge_fade_start: 0.1,
            full_roughness: 0.2,
            max_roughness: 0.6,
            distance_fade_start: 0.7,
        }
    }
}

/// Everything a resolved trace needs to compute its blend confidence.
#[derive(Clone, Copy, Debug)]
pub struct SsrTraceSample {
    /// Whether the hierarchical march reported an intersection.
    pub hit: bool,
    /// UV of the intersection (only meaningful when `hit`).
    pub hit_uv: Vec2,
    /// Normalized ray travel `[0, 1]` at the intersection.
    pub travel: f32,
    /// Unit view-space reflection direction of the traced ray.
    pub reflection: Vec3,
    /// Unit view-space direction from the camera to the shaded surface.
    pub camera_to_surface: Vec3,
    /// Perceptual roughness of the shaded surface.
    pub roughness: f32,
}

/// Combines every fade into the final `[0, 1]` confidence used to blend the
/// screen-space colour over the IBL specular fallback.
///
/// A miss returns `0` immediately.  Otherwise the edge, facing, roughness and
/// distance weights multiply together, so any single disqualifying condition
/// drives the whole trace back to the IBL lobe.
pub fn trace_confidence(sample: SsrTraceSample, params: SsrConfidenceParams) -> f32 {
    if !sample.hit {
        return 0.0;
    }
    let edge = edge_fade(sample.hit_uv, params.edge_fade_start);
    let facing = facing_fade(sample.reflection, sample.camera_to_surface);
    let roughness = roughness_fade(
        sample.roughness,
        params.full_roughness,
        params.max_roughness,
    );
    let distance = distance_fade(sample.travel, params.distance_fade_start);
    saturate(edge * facing * roughness * distance)
}

/// Blends the reprojected screen colour over the prefiltered IBL specular by
/// the trace `confidence`.
///
/// At confidence `0` the result is pure IBL (the SSR miss/fallback path); at
/// `1` it is the screen colour.  Both inputs are linear HDR radiance.
pub fn blend_specular(ibl_specular: Vec3, ssr_color: Vec3, confidence: f32) -> Vec3 {
    let c = saturate(confidence);
    ibl_specular.lerp(ssr_color, c)
}

/// Perceptual-roughness-driven mip bias for sampling a blurred copy of the
/// previous-frame colour, approximating the GGX lobe footprint the single
/// mirror ray cannot capture on its own.
///
/// Returns a fractional mip level in `[0, max_mip]`: mirror-smooth surfaces
/// read mip `0` (sharp) while rougher surfaces read progressively blurrier
/// mips, so the reflection softens with roughness the way the prefiltered
/// environment does.
pub fn reflection_mip(roughness: f32, max_mip: f32) -> f32 {
    // Match the perceptual mapping the prefiltered environment uses: mip grows
    // with the square root of linear roughness so the blur tracks the lobe.
    let linear = saturate(roughness) * saturate(roughness);
    let mip = ops::sqrt(linear) * max_mip.max(0.0);
    mip.clamp(0.0, max_mip.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothstep_matches_endpoints_and_midpoint() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1.0e-6);
        // Degenerate edges behave as a hard step.
        assert_eq!(smoothstep(0.5, 0.5, 0.4), 0.0);
        assert_eq!(smoothstep(0.5, 0.5, 0.6), 1.0);
    }

    #[test]
    fn edge_fade_kills_borders_and_trusts_centre() {
        assert_eq!(edge_fade(Vec2::new(0.0, 0.5), 0.1), 0.0);
        assert_eq!(edge_fade(Vec2::new(0.5, 1.0), 0.1), 0.0);
        assert_eq!(edge_fade(Vec2::new(0.5, 0.5), 0.1), 1.0);
        // Just inside the margin returns a partial weight.
        let partial = edge_fade(Vec2::new(0.05, 0.5), 0.1);
        assert!(partial > 0.0 && partial < 1.0);
    }

    #[test]
    fn facing_fade_rejects_rays_toward_camera() {
        // Camera at origin looking down -Z; surface is in front (negative z).
        let camera_to_surface = Vec3::new(0.0, 0.0, -1.0);
        // Reflection heading deeper into the scene is trusted.
        assert!(facing_fade(Vec3::new(0.0, 0.0, -1.0), camera_to_surface) > 0.99);
        // Reflection heading straight back at the camera is rejected.
        assert_eq!(
            facing_fade(Vec3::new(0.0, 0.0, 1.0), camera_to_surface),
            0.0
        );
    }

    #[test]
    fn roughness_fade_is_monotonic_between_thresholds() {
        assert_eq!(roughness_fade(0.1, 0.2, 0.6), 1.0);
        assert_eq!(roughness_fade(0.7, 0.2, 0.6), 0.0);
        let mid = roughness_fade(0.4, 0.2, 0.6);
        assert!(mid > 0.0 && mid < 1.0);
        // Rougher surfaces never gain confidence.
        assert!(roughness_fade(0.5, 0.2, 0.6) <= roughness_fade(0.3, 0.2, 0.6));
    }

    #[test]
    fn distance_fade_ramps_to_zero_at_limit() {
        assert_eq!(distance_fade(0.0, 0.7), 1.0);
        assert_eq!(distance_fade(0.5, 0.7), 1.0);
        assert_eq!(distance_fade(1.0, 0.7), 0.0);
        let mid = distance_fade(0.85, 0.7);
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn confidence_is_zero_on_miss_and_positive_on_clean_hit() {
        let params = SsrConfidenceParams::default();
        let miss = SsrTraceSample {
            hit: false,
            hit_uv: Vec2::new(0.5, 0.5),
            travel: 0.1,
            reflection: Vec3::new(0.0, 0.0, -1.0),
            camera_to_surface: Vec3::new(0.0, 0.0, -1.0),
            roughness: 0.1,
        };
        assert_eq!(trace_confidence(miss, params), 0.0);

        let hit = SsrTraceSample { hit: true, ..miss };
        assert!(trace_confidence(hit, params) > 0.5);
    }

    #[test]
    fn blend_selects_endpoints_by_confidence() {
        let ibl = Vec3::new(0.2, 0.2, 0.2);
        let ssr = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(blend_specular(ibl, ssr, 0.0), ibl);
        assert_eq!(blend_specular(ibl, ssr, 1.0), ssr);
        let mid = blend_specular(ibl, ssr, 0.5);
        assert!((mid.x - 0.6).abs() < 1.0e-6);
    }

    #[test]
    fn reflection_mip_grows_with_roughness() {
        assert_eq!(reflection_mip(0.0, 5.0), 0.0);
        assert!((reflection_mip(1.0, 5.0) - 5.0).abs() < 1.0e-6);
        assert!(reflection_mip(0.5, 5.0) < reflection_mip(0.8, 5.0));
    }
}
