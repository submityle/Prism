//! Motion-vector reprojection and disocclusion detection for temporal
//! upscaling.
//!
//! Temporal upscaling reconstructs a high-resolution frame by accumulating a
//! low-resolution render target into a display-resolution history. Two grids
//! are therefore in play at once — the **render** grid the current frame is
//! drawn on and the **display** grid the history lives on — and this module
//! owns the arithmetic that bridges them plus the *disocclusion* test that
//! decides when the reprojected history may be trusted.
//!
//! * **Grid conversion** — [`display_to_render`] / [`render_to_display`] scale a
//!   pixel coordinate between the two grids by `render_scale`
//!   (render / display, in `(0, 1]`).
//! * **Reprojection** — [`reproject_history_uv`] follows the repo's
//!   motion-vector contract (`motion_uv = current_uv - previous_uv`, see
//!   [`crate::motion_vector`]) so `previous_uv = current_uv - motion` looks the
//!   surface up in last frame's history.
//! * **Disocclusion** — [`is_offscreen`] rejects history that reprojects off
//!   the previous frame, and [`depth_disocclusion`] rejects history whose depth
//!   disagrees with the reprojected current depth (a surface newly revealed
//!   behind a moving occluder). [`disocclusion_factor`] fuses both into a single
//!   `[0, 1]` invalidation signal the accumulation pass consumes.
//!
//! Everything is plain arithmetic (only `abs`/`max`), mirrored bit-for-bit by
//! the GPU reproject twin.

use bevy_math::Vec2;

/// Convert a **display**-grid pixel coordinate to the **render** grid by
/// scaling with `render_scale` (render / display). A display pixel at `(x, y)`
/// maps to `(x, y) * render_scale` in the smaller render target.
#[must_use]
pub fn display_to_render(display_px: Vec2, render_scale: f32) -> Vec2 {
    display_px * render_scale
}

/// Convert a **render**-grid pixel coordinate to the **display** grid, the
/// inverse of [`display_to_render`]. A non-positive scale is guarded so the
/// division cannot blow up, returning the input unchanged.
#[must_use]
pub fn render_to_display(render_px: Vec2, render_scale: f32) -> Vec2 {
    if render_scale > 0.0 {
        render_px / render_scale
    } else {
        render_px
    }
}

/// Reproject a current-frame UV into the previous frame using the motion
/// vector.
///
/// The repo stores `motion_uv = current_uv - previous_uv` (see
/// [`crate::motion_vector`]), so the previous-frame lookup is
/// `current_uv - motion`. The result is *not* clamped to `[0, 1]`; call
/// [`is_offscreen`] to decide how to treat a lookup that leaves the frame.
#[must_use]
pub fn reproject_history_uv(current_uv: Vec2, motion: Vec2) -> Vec2 {
    current_uv - motion
}

/// Whether a reprojected UV falls outside the `[0, 1]²` history frame, where no
/// valid history exists and the accumulation must fall back to the current
/// (reconstructed) sample.
#[must_use]
pub fn is_offscreen(uv: Vec2) -> bool {
    uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0
}

/// A per-pixel depth disocclusion signal in `[0, 1]` from linear (view-space,
/// positive-into-scene) depths.
///
/// `current_depth` is this frame's linear depth at the pixel; `history_depth`
/// is the linear depth sampled from the previous frame at the reprojected UV.
/// When a surface is continuous across the two frames the depths agree, so the
/// relative difference `|current - history| / max(current, history)` is tiny
/// and the factor is `0` (fully trusted). When a moving occluder reveals a
/// surface behind it the depths diverge, the relative difference exceeds
/// `tolerance`, and the factor ramps to `1` (fully disoccluded) at `2·tolerance`
/// — a soft knee rather than a hard cliff so a grazing surface does not flicker
/// between trusted and rejected.
///
/// `tolerance` is a relative fraction (e.g. `0.025` for 2.5%). A non-positive
/// or non-finite depth on either side is treated as a full disocclusion.
#[must_use]
pub fn depth_disocclusion(current_depth: f32, history_depth: f32, tolerance: f32) -> f32 {
    if current_depth <= 0.0
        || history_depth <= 0.0
        || !current_depth.is_finite()
        || !history_depth.is_finite()
    {
        return 1.0;
    }
    let tol = tolerance.max(1.0e-6);
    let denom = current_depth.max(history_depth);
    let relative = (current_depth - history_depth).abs() / denom;
    // Soft ramp from `tol` (still trusted) to `2·tol` (fully disoccluded).
    ((relative - tol) / tol).clamp(0.0, 1.0)
}

/// Fuse the off-screen and depth disocclusion tests into a single `[0, 1]`
/// invalidation signal: `1` when the reprojected UV left the frame, otherwise
/// the [`depth_disocclusion`] factor. `0` means the history is fully trusted.
#[must_use]
pub fn disocclusion_factor(
    reprojected_uv: Vec2,
    current_depth: f32,
    history_depth: f32,
    tolerance: f32,
) -> f32 {
    if is_offscreen(reprojected_uv) {
        return 1.0;
    }
    depth_disocclusion(current_depth, history_depth, tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Vec2, b: Vec2) -> bool {
        (a - b).abs().max_element() < 1.0e-6
    }

    #[test]
    fn grid_conversions_round_trip() {
        let display = Vec2::new(1920.0, 540.0);
        let render = display_to_render(display, 0.5);
        assert!(approx(render, Vec2::new(960.0, 270.0)));
        assert!(approx(render_to_display(render, 0.5), display));
    }

    #[test]
    fn render_to_display_guards_zero_scale() {
        let p = Vec2::new(3.0, 4.0);
        assert!(approx(render_to_display(p, 0.0), p));
    }

    #[test]
    fn reprojection_follows_the_motion_contract() {
        // motion = current - previous, so previous = current - motion.
        let current = Vec2::new(0.6, 0.4);
        let previous = Vec2::new(0.5, 0.45);
        let motion = current - previous;
        assert!(approx(reproject_history_uv(current, motion), previous));
    }

    #[test]
    fn zero_motion_keeps_the_same_uv() {
        let uv = Vec2::new(0.3, 0.7);
        assert!(approx(reproject_history_uv(uv, Vec2::ZERO), uv));
    }

    #[test]
    fn offscreen_detects_out_of_frame_lookups() {
        assert!(!is_offscreen(Vec2::new(0.0, 1.0)));
        assert!(!is_offscreen(Vec2::new(0.5, 0.5)));
        assert!(is_offscreen(Vec2::new(-0.01, 0.5)));
        assert!(is_offscreen(Vec2::new(0.5, 1.01)));
    }

    #[test]
    fn matching_depths_are_fully_trusted() {
        assert_eq!(depth_disocclusion(10.0, 10.0, 0.025), 0.0);
        // A sub-tolerance wobble is still trusted.
        assert_eq!(depth_disocclusion(10.0, 10.1, 0.025), 0.0);
    }

    #[test]
    fn diverging_depths_ramp_to_full_disocclusion() {
        // 20% relative difference is far past a 2.5% tolerance => saturated.
        assert_eq!(depth_disocclusion(10.0, 12.0, 0.025), 1.0);
        // A value inside the soft knee sits strictly between 0 and 1.
        let mid = depth_disocclusion(100.0, 103.75, 0.025);
        assert!(mid > 0.0 && mid < 1.0, "expected a partial factor, got {mid}");
    }

    #[test]
    fn invalid_depths_are_treated_as_disocclusion() {
        assert_eq!(depth_disocclusion(0.0, 10.0, 0.025), 1.0);
        assert_eq!(depth_disocclusion(10.0, -1.0, 0.025), 1.0);
        assert_eq!(depth_disocclusion(f32::INFINITY, 10.0, 0.025), 1.0);
    }

    #[test]
    fn fused_factor_prefers_offscreen_rejection() {
        // Depths agree, but the lookup left the frame => full invalidation.
        let f = disocclusion_factor(Vec2::new(1.5, 0.5), 10.0, 10.0, 0.025);
        assert_eq!(f, 1.0);
        // On-screen and depth-consistent => trusted.
        let g = disocclusion_factor(Vec2::new(0.5, 0.5), 10.0, 10.0, 0.025);
        assert_eq!(g, 0.0);
    }
}
