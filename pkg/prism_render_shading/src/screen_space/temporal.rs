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
//! history.  Prism's resolve pass writes a per-pixel **motion-vector G-buffer**
//! (see [`super::motion`]) that folds in *both* camera and per-object motion, so
//! the preferred reprojection just adds that vector back to the current UV
//! ([`reproject_prev_uv_motion`]) to recover where the surface sat last frame —
//! no depth reconstruction and, crucially, no ghosting on animated or skinned
//! geometry.  The older depth-only camera reprojection ([`reproject_prev_uv`],
//! reconstruct world from reverse-Z depth + inverse current view-projection,
//! then project through the previous view-projection) is kept as the
//! backend-neutral reference for static scenes.  Either way a disocclusion (or
//! an off-screen reprojection) drops history and falls back to the current
//! frame, and the reprojected history is clipped to the axis-aligned colour box
//! of the current pixel's neighbourhood before the exponential blend to suppress
//! any residual ghosting.
//!
//! This module is the CPU golden; the `ssr_temporal.wesl` twin shares the same
//! motion reprojection, neighbourhood-clip and accumulation math bit for bit.
//! Every transcendental (there are none here) would route through
//! [`bevy_math::ops`] for cross-platform determinism.

use bevy_math::{Mat4, Vec2, Vec3, Vec4};

/// Exponential-accumulation tunables shared by the golden and the
/// `ssr_temporal.wesl` twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrTemporalParams {
    /// Fraction of the reprojected history kept each frame when the sample is
    /// valid *and* the history agrees with the current neighbourhood.  Higher
    /// converges smoother but adds latency; `0.9` blends in roughly a tenth of
    /// the current frame per step (a ~10-frame window).  This is the *upper*
    /// bound: [`adaptive_history_weight`] pulls it down toward
    /// [`Self::min_history_weight`] as the reprojected history is clipped.
    pub history_weight: f32,
    /// Symmetric expansion (in colour units) applied to the neighbourhood AABB
    /// before clipping history.  A small slack lets the clamp tolerate residual
    /// trace noise without letting stale reflections leak back in.
    pub clamp_expand: f32,
    /// Standard-deviation multiplier for the variance clip box: the clamp box is
    /// `mean ± variance_gamma·σ` of the 3x3 neighbourhood rather than its raw
    /// min/max.  This is the AAA variance-clipping method (Salvi/Karis): the raw
    /// min/max hugs single outlier taps and flickers, while `mean ± γσ` tracks
    /// the neighbourhood distribution.  `1.25` is the usual TAA value.
    pub variance_gamma: f32,
    /// Floor the adaptive weight decays toward when the reprojected history sits
    /// far outside the clip box (a disocclusion or a moving surface).  `0.5`
    /// still smooths a little on a disocclusion while dropping most of the stale
    /// history, trading a touch of noise for no visible ghost.  Clamped to
    /// `[0, history_weight]`.
    pub min_history_weight: f32,
    /// How much to relax (scale up) the clip box when the *current* sample is a
    /// low-confidence miss.  SSR misses are often transient — a ray that ran out
    /// of budget or got skipped by the HZB one frame reappears the next — so a
    /// hard variance clip against a momentarily black pixel would discard the
    /// converged reflection and make it pop.  The box half-extent is scaled by
    /// `1 + (1 - confidence)·confidence_relax`, so a confident hit clips tightly
    /// (responsive) while a full miss widens the box and preserves history.
    /// `0` disables the relaxation.
    pub confidence_relax: f32,
}

impl Default for SsrTemporalParams {
    fn default() -> Self {
        Self {
            history_weight: 0.9,
            clamp_expand: 0.0,
            variance_gamma: 1.25,
            min_history_weight: 0.5,
            confidence_relax: 4.0,
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

/// Reproject a pixel's history UV directly from its **motion vector** — the
/// screen-space displacement `current_uv - previous_uv` the resolve pass writes
/// per covered pixel (see [`super::motion`]).  Because that vector already folds
/// in *both* camera and per-object motion, adding it back recovers exactly where
/// the surface sat last frame with no depth reconstruction and no ghosting on
/// animated or skinned geometry — the reason a motion-vector G-buffer supersedes
/// the depth-only camera reprojection in [`reproject_prev_uv`].
///
/// Returns [`None`] when the recovered UV lands outside `[0, 1]²` (an off-screen
/// reprojection or a disocclusion), which the accumulation treats as a dropped
/// sample and falls back to the current frame.
pub fn reproject_prev_uv_motion(uv: Vec2, motion: Vec2) -> Option<Vec2> {
    let prev = uv - motion;
    if prev.x < 0.0 || prev.x > 1.0 || prev.y < 0.0 || prev.y > 1.0 {
        return None;
    }
    Some(prev)
}

/// The result of clipping history to the neighbourhood box: the clamped colour
/// plus the *overshoot* — how far outside the box the raw history sat, in box
/// half-extents beyond the surface (`0` when already inside).  The overshoot is
/// the disocclusion signal [`adaptive_history_weight`] uses to decay the blend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipResult {
    /// History clamped onto (or left inside) the box surface.
    pub clipped: Vec3,
    /// `max_unit - 1` where `max_unit` is the largest per-axis distance from the
    /// box centre in half-extents; `0` when the history is already inside.
    pub overshoot: f32,
}

/// Clip `history` to the axis-aligned colour box `[box_min, box_max]` using the
/// AABB-clip (Karis) method — scale the ray from the box centre toward `history`
/// so it just reaches the box surface — and report how far outside the box it
/// began.  This is the primitive both [`clip_history_to_aabb`] and the adaptive
/// accumulation build on.
pub fn clip_history_to_aabb_ex(history: Vec3, box_min: Vec3, box_max: Vec3) -> ClipResult {
    let center = 0.5 * (box_max + box_min);
    // Guard against a degenerate (flat) neighbourhood collapsing the box.
    let extent = (0.5 * (box_max - box_min)).max(Vec3::splat(1.0e-5));
    let dir = history - center;
    let units = dir / extent;
    let max_unit = units.x.abs().max(units.y.abs()).max(units.z.abs());
    if max_unit > 1.0 {
        ClipResult {
            clipped: center + dir / max_unit,
            overshoot: max_unit - 1.0,
        }
    } else {
        ClipResult {
            clipped: history,
            overshoot: 0.0,
        }
    }
}

/// Clip `history` to the axis-aligned colour box `[box_min, box_max]` using the
/// AABB-clip (Karis) method: rather than clamp each channel independently
/// (which desaturates and clings to box faces), scale the ray from the box
/// centre toward `history` so it just reaches the box surface.  History already
/// inside the box is returned unchanged.
pub fn clip_history_to_aabb(history: Vec3, box_min: Vec3, box_max: Vec3) -> Vec3 {
    clip_history_to_aabb_ex(history, box_min, box_max).clipped
}

/// Build the variance clip box `mean ± variance_gamma·σ` from the first two
/// colour moments of the 3x3 neighbourhood (`mean` and `mean_sq`, the mean of
/// per-channel squares).  This is AAA variance clipping (Salvi/Karis): unlike a
/// raw min/max box that hugs a single outlier tap and flickers, the `mean ± γσ`
/// box tracks the neighbourhood distribution, so it rejects stale history
/// firmly in flat regions yet widens gracefully across noisy edges.
///
/// The variance is clamped to be non-negative before the square root to absorb
/// the catastrophic cancellation `mean_sq - mean²` can suffer in float.
pub fn variance_clip_box(mean: Vec3, mean_sq: Vec3, variance_gamma: f32) -> (Vec3, Vec3) {
    let variance = (mean_sq - mean * mean).max(Vec3::ZERO);
    let sigma = Vec3::new(variance.x.sqrt(), variance.y.sqrt(), variance.z.sqrt());
    let half = sigma * variance_gamma.max(0.0);
    (mean - half, mean + half)
}

/// Decay the blend weight from [`SsrTemporalParams::history_weight`] toward
/// [`SsrTemporalParams::min_history_weight`] as the reprojected history is
/// clipped further outside the neighbourhood box.
///
/// `overshoot` is [`ClipResult::overshoot`]: `0` when the history sat inside the
/// box (a static, well-matched surface — keep the full weight for maximum
/// denoising) rising toward and past `1` as the history disagrees (a
/// disocclusion or a moving surface — shed history to kill the ghost).  The
/// decay saturates at `overshoot == 1` (one full half-extent outside), a knee
/// that reaches the floor quickly without a hard cliff.
pub fn adaptive_history_weight(params: &SsrTemporalParams, overshoot: f32) -> f32 {
    let base = params.history_weight.clamp(0.0, 1.0);
    let floor = params.min_history_weight.clamp(0.0, base);
    let t = overshoot.clamp(0.0, 1.0);
    base + (floor - base) * t
}

/// Scale the neighbourhood clip box around its centre by
/// `1 + (1 - confidence)·relax`, widening it as the current sample loses
/// confidence.  A confident hit (`confidence == 1`) leaves the box untouched
/// for a responsive, tight clip; a full miss (`confidence == 0`) inflates it by
/// `1 + relax`, so a transient SSR miss keeps the converged history instead of
/// clipping it to a momentarily black pixel and popping.
pub fn relax_box_for_confidence(
    box_min: Vec3,
    box_max: Vec3,
    confidence: f32,
    relax: f32,
) -> (Vec3, Vec3) {
    let center = 0.5 * (box_max + box_min);
    let half = 0.5 * (box_max - box_min);
    let scale = 1.0 + (1.0 - confidence.clamp(0.0, 1.0)) * relax.max(0.0);
    (center - half * scale, center + half * scale)
}

/// Exponentially accumulate the current frame's resolved reflection
/// (`current.rgb` radiance + `current.a` confidence) with the reprojected
/// `history`, clipping the history colour to the current neighbourhood box
/// `[box_min, box_max]` first and adapting the blend weight to how far the
/// history had to be clipped.
///
/// The caller passes the variance clip box from [`variance_clip_box`] (the
/// shader gathers the same moments over its 3x3 neighbourhood).  The box is
/// first relaxed by [`relax_box_for_confidence`] so a low-confidence current
/// sample (a transient SSR miss) keeps its converged history rather than
/// clipping it to a momentarily black pixel.  History that lands inside the box
/// keeps the full [`SsrTemporalParams::history_weight`] for maximum denoising;
/// history dragged far outside decays toward
/// [`SsrTemporalParams::min_history_weight`] via [`adaptive_history_weight`], so
/// a disocclusion or a moving surface sheds its stale reflection instead of
/// ghosting.
///
/// When `valid` is false (a dropped reprojection: off-screen, disoccluded, or a
/// background pixel) the history is discarded and the current frame is returned
/// as-is.  The confidence channel is blended by the same adaptive weight but is
/// *not* clipped to the colour box (it is not a colour).
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
    // Widen the clip box for a low-confidence current sample so a transient miss
    // preserves the converged reflection instead of clipping it away.
    let (box_min, box_max) =
        relax_box_for_confidence(box_min, box_max, current.w, params.confidence_relax);
    let expand = Vec3::splat(params.clamp_expand.max(0.0));
    let clip = clip_history_to_aabb_ex(history.truncate(), box_min - expand, box_max + expand);
    let w = adaptive_history_weight(params, clip.overshoot);
    let rgb = current.truncate() * (1.0 - w) + clip.clipped * w;
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
    fn motion_reprojection_subtracts_the_vector() {
        // A zero motion vector (static surface under a static camera) keeps the
        // pixel in place; a non-zero vector shifts the history UV by exactly its
        // negative, following the surface back to last frame's position.
        let uv = Vec2::new(0.4, 0.55);
        assert!(approx(reproject_prev_uv_motion(uv, Vec2::ZERO).unwrap(), uv));
        let motion = Vec2::new(0.05, -0.1);
        let prev = reproject_prev_uv_motion(uv, motion).unwrap();
        assert!(approx(prev, uv - motion), "motion reproject must be uv - motion: {prev:?}");
    }

    #[test]
    fn motion_reprojection_drops_offscreen_history() {
        // A vector that carries the lookup outside [0,1]^2 (an off-screen
        // reprojection or a disocclusion) drops history to the current frame.
        let uv = Vec2::new(0.02, 0.5);
        assert!(reproject_prev_uv_motion(uv, Vec2::new(0.5, 0.0)).is_none());
        assert!(reproject_prev_uv_motion(uv, Vec2::new(0.0, -0.9)).is_none());
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
            variance_gamma: 1.25,
            min_history_weight: 0.5,
            confidence_relax: 4.0,
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
        // The history sat far outside the box, so the adaptive weight collapsed
        // to the floor (0.5): confidence EMA = 0.5*1.0 + 0.5*0.0 = 0.5.
        assert!((out.w - 0.5).abs() < 1.0e-5, "adaptive confidence EMA: {out:?}");
    }

    #[test]
    fn history_inside_the_box_keeps_the_full_weight() {
        let params = SsrTemporalParams::default();
        let cur = Vec4::new(0.2, 0.2, 0.2, 1.0);
        // History sits inside a wide box (overshoot 0) -> full history_weight.
        let hist = Vec4::new(0.25, 0.25, 0.25, 0.0);
        let out = accumulate_temporal(
            &params,
            cur,
            hist,
            Vec3::splat(0.0),
            Vec3::splat(1.0),
            true,
        );
        // Full weight 0.9: confidence EMA = 0.1*1.0 + 0.9*0.0 = 0.1.
        assert!((out.w - 0.1).abs() < 1.0e-5, "full-weight confidence EMA: {out:?}");
    }

    #[test]
    fn adaptive_weight_decays_from_base_to_floor_with_overshoot() {
        let params = SsrTemporalParams::default();
        // Inside the box -> base weight.
        assert!((adaptive_history_weight(&params, 0.0) - 0.9).abs() < 1.0e-6);
        // One half-extent outside -> saturates at the floor.
        assert!((adaptive_history_weight(&params, 1.0) - 0.5).abs() < 1.0e-6);
        // Beyond saturation stays at the floor (no undershoot).
        assert!((adaptive_history_weight(&params, 4.0) - 0.5).abs() < 1.0e-6);
        // Monotonic: partway between base and floor.
        let mid = adaptive_history_weight(&params, 0.5);
        assert!(mid < 0.9 && mid > 0.5, "midpoint must sit between: {mid}");
    }

    #[test]
    fn variance_box_tracks_the_distribution_not_outliers() {
        // Neighbourhood mean 0.5, per-channel variance 0.04 (sigma 0.2).
        let mean = Vec3::splat(0.5);
        let mean_sq = Vec3::splat(0.5 * 0.5 + 0.04);
        let (mn, mx) = variance_clip_box(mean, mean_sq, 1.25);
        // Box is mean ± 1.25*0.2 = 0.5 ± 0.25.
        assert!((mn.x - 0.25).abs() < 1.0e-5, "box_min: {mn:?}");
        assert!((mx.x - 0.75).abs() < 1.0e-5, "box_max: {mx:?}");
    }

    #[test]
    fn variance_box_absorbs_negative_variance_from_cancellation() {
        // mean_sq < mean² (float cancellation) must not NaN via sqrt of a
        // negative: the box collapses to a point at the mean.
        let mean = Vec3::splat(0.5);
        let mean_sq = Vec3::splat(0.24);
        let (mn, mx) = variance_clip_box(mean, mean_sq, 1.25);
        assert!((mn - mean).length() < 1.0e-6 && (mx - mean).length() < 1.0e-6);
    }

    #[test]
    fn confidence_relax_widens_the_box_for_a_miss_only() {
        // Confident (1.0): box unchanged.
        let (mn, mx) = relax_box_for_confidence(Vec3::splat(0.4), Vec3::splat(0.6), 1.0, 4.0);
        assert!((mn.x - 0.4).abs() < 1.0e-6 && (mx.x - 0.6).abs() < 1.0e-6);
        // Full miss (0.0) with relax 4.0: half-extent 0.1 scales x5 to 0.5.
        let (mn, mx) = relax_box_for_confidence(Vec3::splat(0.4), Vec3::splat(0.6), 0.0, 4.0);
        assert!((mn.x - 0.0).abs() < 1.0e-5, "widened box_min: {mn:?}");
        assert!((mx.x - 1.0).abs() < 1.0e-5, "widened box_max: {mx:?}");
    }

    #[test]
    fn a_low_confidence_miss_preserves_converged_history() {
        // Current frame is a near-black, zero-confidence miss with a tight,
        // near-black neighbourhood (all rays missed this frame).  A hard
        // variance clip would drag the converged history down to the black box
        // face; the confidence relaxation widens the box so more of the history
        // survives the blend.  The golden assertion is comparative: with the
        // relaxation on, strictly more history must come through than with it
        // disabled (`confidence_relax = 0`), and the result must stay bounded by
        // the relaxed box face (no unclipped ghost leaks in).
        let cur = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let hist = Vec4::new(0.6, 0.6, 0.6, 0.95);
        let box_min = Vec3::splat(0.0);
        let box_max = Vec3::splat(0.05);

        let relaxed = SsrTemporalParams::default();
        let rigid = SsrTemporalParams {
            confidence_relax: 0.0,
            ..SsrTemporalParams::default()
        };
        let out_relaxed = accumulate_temporal(&relaxed, cur, hist, box_min, box_max, true);
        let out_rigid = accumulate_temporal(&rigid, cur, hist, box_min, box_max, true);

        // The relaxed box preserves strictly more of the converged reflection.
        assert!(
            out_relaxed.x > out_rigid.x + 1.0e-3,
            "relaxation must preserve more history: relaxed {out_relaxed:?} vs rigid {out_rigid:?}"
        );
        // ...but the clip still bounds it: the relaxed box half-extent is
        // 0.025·5 = 0.125 around centre 0.025, so the clamped history tops out
        // at the 0.15 box face and the half-weight blend cannot exceed it.
        assert!(
            out_relaxed.x <= 0.15 + 1.0e-4,
            "relaxed clip must still bound the history: {out_relaxed:?}"
        );
    }

    #[test]
    fn a_confident_hit_still_clips_stale_history() {
        let params = SsrTemporalParams::default();
        // Confident current hit: no relaxation, so a wildly different history is
        // clipped to the tight neighbourhood box (no ghost leaks in).
        let cur = Vec4::new(0.2, 0.2, 0.2, 1.0);
        let hist = Vec4::new(5.0, 5.0, 5.0, 1.0);
        let out = accumulate_temporal(
            &params,
            cur,
            hist,
            Vec3::splat(0.1),
            Vec3::splat(0.3),
            true,
        );
        assert!(out.x <= 0.3 + 1.0e-4, "confident hit must still clamp: {out:?}");
    }
    #[test]
    fn clip_ex_reports_overshoot_outside_and_zero_inside() {
        let inside = clip_history_to_aabb_ex(Vec3::splat(0.5), Vec3::ZERO, Vec3::ONE);
        assert_eq!(inside.overshoot, 0.0);
        // Centre 0.5, half-extent 0.5; history at 1.5 is one extent past +x face
        // -> max_unit = (1.5-0.5)/0.5 = 2.0 -> overshoot 1.0.
        let outside = clip_history_to_aabb_ex(
            Vec3::new(1.5, 0.5, 0.5),
            Vec3::ZERO,
            Vec3::ONE,
        );
        assert!((outside.overshoot - 1.0).abs() < 1.0e-5, "{outside:?}");
        assert!((outside.clipped.x - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn expand_bounds_grows_symmetrically() {
        let (mn, mx) = expand_bounds(Vec3::splat(0.5), Vec3::splat(0.5), Vec3::new(0.2, 0.7, 0.5));
        assert_eq!(mn, Vec3::new(0.2, 0.5, 0.5));
        assert_eq!(mx, Vec3::new(0.5, 0.7, 0.5));
    }
}
