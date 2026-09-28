//! Cross-frame temporal accumulation for screen-space reflections.
//!
//! Even after the spatial [`super::reconstruct`] resolve, a multi-ray SSR trace
//! still shimmers frame to frame: each frame draws fresh Hammersley samples, so
//! a rough reflection boils under any camera or surface motion.  AAA renderers
//! kill that boil by *accumulating over time* — the current frame's noisy
//! resolve is averaged with the same surface's reflection from previous frames,
//! turning N frames of few-ray traces into one many-ray estimate for free.
//!
//! The hard part is finding "the same surface" last frame and rejecting stale
//! history.  Prism is a visibility-buffer deferred renderer with no per-pixel
//! motion-vector G-buffer, so this reprojects purely from camera motion: it
//! reconstructs each pixel's world position from its reverse-Z device depth and
//! the inverse current view-projection, then reprojects that world point
//! through the *previous* frame's view-projection to recover the UV the surface
//! occupied last frame.  Static geometry under a moving camera reprojects
//! exactly; a disocclusion (or an off-screen reprojection) drops history and
//! falls back to the current frame.  To suppress ghosting on the surfaces that
//! do move, the reprojected history is clipped to the axis-aligned colour box
//! of the current pixel's neighbourhood before the exponential blend.
//!
//! This module is the CPU golden; the `ssr_temporal.wesl` twin shares the same
//! reprojection, neighbourhood-clip and accumulation math bit for bit.  Every
//! transcendental (there are none here) would route through [`bevy_math::ops`]
//! for cross-platform determinism.

use bevy_math::{Mat4, Vec2, Vec3, Vec4};

/// Exponential-accumulation tunables shared by the golden and the
/// `ssr_temporal.wesl` twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrTemporalParams {
    /// Fraction of the reprojected history kept each frame when the sample is
    /// valid.  Higher converges smoother but adds latency; `0.9` blends in
    /// roughly a tenth of the current frame per step (a ~10-frame window).
    pub history_weight: f32,
    /// Symmetric expansion (in colour units) applied to the neighbourhood AABB
    /// before clipping history.  A small slack lets the clamp tolerate residual
    /// trace noise without letting stale reflections leak back in.
    pub clamp_expand: f32,
}

impl Default for SsrTemporalParams {
    fn default() -> Self {
        Self {
            history_weight: 0.9,
            clamp_expand: 0.0,
        }
    }
}

/// Reconstruct a pixel's world position from its reverse-Z device depth and the
/// inverse current view-projection, then reproject it through the previous
/// frame's view-projection to find the UV the surface occupied last frame.
///
/// Returns [`None`] when there is nothing to reproject: a background/sky pixel
/// (reverse-Z `device_depth <= 0`), a point behind the previous camera, or a
/// reprojection that lands outside the previous frustum (an off-screen sample
/// or a disocclusion), all of which must fall back to the current frame.
///
/// `world_from_clip` is `inverse(current clip_from_world)`; `clip_from_world_prev`
/// is the previous frame's view-projection.  NDC follows the trace's
/// convention: `ndc = (uv.x*2-1, (1-uv.y)*2-1, device_depth)`.
pub fn reproject_prev_uv(
    world_from_clip: Mat4,
    clip_from_world_prev: Mat4,
    uv: Vec2,
    device_depth: f32,
) -> Option<Vec2> {
    // Reverse-Z: only depth > 0 is a real surface; <= 0 is background/sky.
    if device_depth <= 0.0 {
        return None;
    }
    let ndc = Vec3::new(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let world_h = world_from_clip * ndc.extend(1.0);
    if world_h.w == 0.0 {
        return None;
    }
    let world = world_h.truncate() / world_h.w;

    let clip_prev = clip_from_world_prev * world.extend(1.0);
    // Behind (or on) the previous near plane: no valid projection.
    if clip_prev.w <= 0.0 {
        return None;
    }
    let ndc_prev = clip_prev.truncate() / clip_prev.w;
    // Outside the previous frustum in XY -> off-screen / disocclusion.
    if ndc_prev.x < -1.0 || ndc_prev.x > 1.0 || ndc_prev.y < -1.0 || ndc_prev.y > 1.0 {
        return None;
    }

    Some(Vec2::new(
        ndc_prev.x * 0.5 + 0.5,
        1.0 - (ndc_prev.y * 0.5 + 0.5),
    ))
}

/// Clip `history` to the axis-aligned colour box `[box_min, box_max]` using the
/// AABB-clip (Karis) method: rather than clamp each channel independently
/// (which desaturates and clings to box faces), scale the ray from the box
/// centre toward `history` so it just reaches the box surface.  History already
/// inside the box is returned unchanged.
pub fn clip_history_to_aabb(history: Vec3, box_min: Vec3, box_max: Vec3) -> Vec3 {
    let center = 0.5 * (box_max + box_min);
    // Guard against a degenerate (flat) neighbourhood collapsing the box.
    let extent = (0.5 * (box_max - box_min)).max(Vec3::splat(1.0e-5));
    let dir = history - center;
    let units = dir / extent;
    let max_unit = units.x.abs().max(units.y.abs()).max(units.z.abs());
    if max_unit > 1.0 {
        center + dir / max_unit
    } else {
        history
    }
}

/// Exponentially accumulate the current frame's resolved reflection
/// (`current.rgb` radiance + `current.a` confidence) with the reprojected
/// `history`, clipping the history colour to the current neighbourhood box
/// `[box_min, box_max]` first.
///
/// When `valid` is false (a dropped reprojection: off-screen, disoccluded, or a
/// background pixel) the history is discarded and the current frame is returned
/// as-is, so temporal accumulation never ghosts across a surface change.  The
/// confidence channel is blended by the same weight but is *not* clipped to the
/// colour box (it is not a colour).
pub fn accumulate_temporal(
    params: &SsrTemporalParams,
    current: Vec4,
    history: Vec4,
    box_min: Vec3,
    box_max: Vec3,
    valid: bool,
) -> Vec4 {
    if !valid {
        return current;
    }
    let expand = Vec3::splat(params.clamp_expand.max(0.0));
    let clamped = clip_history_to_aabb(history.truncate(), box_min - expand, box_max + expand);
    let w = params.history_weight.clamp(0.0, 1.0);
    let rgb = current.truncate() * (1.0 - w) + clamped * w;
    let conf = current.w * (1.0 - w) + history.w * w;
    rgb.extend(conf)
}

/// Grow a running colour AABB to include `sample`.  The trace's neighbourhood
/// clamp seeds `min`/`max` from the current pixel then folds each 3x3 tap in;
/// the golden exposes it so tests build the same box the shader gathers.
pub fn expand_bounds(min: Vec3, max: Vec3, sample: Vec3) -> (Vec3, Vec3) {
    (min.min(sample), max.max(sample))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec4;

    fn approx(a: Vec2, b: Vec2) -> bool {
        (a - b).length() < 1.0e-4
    }

    // A reverse-Z perspective clip_from_world with the camera at the origin
    // looking down -Z, matching the trace/reconstruct convention.
    fn clip_from_world(z_translation: f32) -> Mat4 {
        // Simple reverse-Z-ish projection: near maps to depth 1, far to 0.
        let proj = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, 0.5, 0.0),
        );
        proj * Mat4::from_translation(Vec3::new(0.0, 0.0, z_translation))
    }

    #[test]
    fn static_camera_reprojects_to_the_same_uv() {
        let cur = clip_from_world(0.0);
        let world_from_clip = cur.inverse();
        let uv = Vec2::new(0.37, 0.62);
        // Any surface depth in (0,1]; reproject through the identical matrix.
        let prev = reproject_prev_uv(world_from_clip, cur, uv, 0.5).unwrap();
        assert!(approx(prev, uv), "identity reprojection must be a no-op: {prev:?}");
    }

    #[test]
    fn background_and_offscreen_drop_history() {
        let cur = clip_from_world(0.0);
        let inv = cur.inverse();
        // Reverse-Z background.
        assert!(reproject_prev_uv(inv, cur, Vec2::new(0.5, 0.5), 0.0).is_none());
        assert!(reproject_prev_uv(inv, cur, Vec2::new(0.5, 0.5), -0.1).is_none());
    }

    #[test]
    fn camera_translation_shifts_the_reprojected_uv() {
        let cur = clip_from_world(0.0);
        let inv = cur.inverse();
        // Previous frame camera sat further back along +Z (moved forward since).
        let prev_mat = clip_from_world(0.2);
        let uv = Vec2::new(0.6, 0.4);
        let prev = reproject_prev_uv(inv, prev_mat, uv, 0.5);
        // A parallax shift must move (or drop) the sample, never reproject to
        // the identical UV.
        if let Some(p) = prev {
            assert!(!approx(p, uv), "moving camera must not reproject in place");
        }
    }

    #[test]
    fn history_inside_the_box_is_untouched() {
        let h = Vec3::new(0.4, 0.5, 0.6);
        let out = clip_history_to_aabb(h, Vec3::splat(0.0), Vec3::splat(1.0));
        assert!((out - h).length() < 1.0e-6);
    }

    #[test]
    fn history_outside_the_box_is_clipped_onto_its_surface() {
        // Centre (0.5), history pushed far past the +x face.
        let h = Vec3::new(3.0, 0.5, 0.5);
        let out = clip_history_to_aabb(h, Vec3::splat(0.0), Vec3::splat(1.0));
        assert!((out.x - 1.0).abs() < 1.0e-5, "x must land on the box face: {out:?}");
        // y/z unchanged because the dominant axis drove the scale.
        assert!((out.y - 0.5).abs() < 1.0e-5 && (out.z - 0.5).abs() < 1.0e-5);
    }

    #[test]
    fn invalid_samples_return_the_current_frame() {
        let params = SsrTemporalParams::default();
        let cur = Vec4::new(0.2, 0.3, 0.4, 0.8);
        let hist = Vec4::new(0.9, 0.9, 0.9, 0.1);
        let out = accumulate_temporal(&params, cur, hist, Vec3::ZERO, Vec3::ONE, false);
        assert_eq!(out, cur);
    }

    #[test]
    fn valid_samples_blend_toward_clamped_history() {
        let params = SsrTemporalParams {
            history_weight: 0.9,
            clamp_expand: 0.0,
        };
        let cur = Vec4::new(0.2, 0.2, 0.2, 1.0);
        // History far outside the neighbourhood box gets clipped before blend.
        let hist = Vec4::new(5.0, 5.0, 5.0, 0.0);
        let box_min = Vec3::splat(0.1);
        let box_max = Vec3::splat(0.3);
        let out = accumulate_temporal(&params, cur, hist, box_min, box_max, true);
        // Clamped history is at most box_max (0.3); blended rgb must stay within
        // [min(cur,box), max(cur,box)], i.e. never near the raw 5.0.
        assert!(out.x <= 0.3 + 1.0e-4, "clamp must bound the blend: {out:?}");
        assert!(out.x >= 0.2 - 1.0e-4);
        // Confidence EMA: 0.1*1.0 + 0.9*0.0 = 0.1.
        assert!((out.w - 0.1).abs() < 1.0e-5, "confidence EMA: {out:?}");
    }

    #[test]
    fn expand_bounds_grows_symmetrically() {
        let (mn, mx) = expand_bounds(Vec3::splat(0.5), Vec3::splat(0.5), Vec3::new(0.2, 0.7, 0.5));
        assert_eq!(mn, Vec3::new(0.2, 0.5, 0.5));
        assert_eq!(mx, Vec3::new(0.5, 0.7, 0.5));
    }
}
