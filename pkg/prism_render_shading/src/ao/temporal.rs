//! Cross-frame temporal accumulation for the denoised GTAO buffer.
//!
//! The spatial [`super::denoise`] bilateral kills most of the per-pixel grain,
//! but a single-frame GTAO estimate still *boils* under motion: each frame the
//! horizon search lands on slightly different depths, so flat walls shimmer and
//! contact shadows crawl.  AAA GTAO (`XeGTAO`'s temporal filter) removes that
//! boil by accumulating the denoised visibility over time — reprojecting the
//! previous frame's converged AO onto the current pixel and blending a little
//! of the new estimate in each frame, turning N noisy frames into one stable
//! estimate.
//!
//! Because a screen-space AO pass runs *before* the shading resolve writes its
//! motion-vector G-buffer, there is no per-object motion here; instead the
//! reprojection reconstructs each pixel's world position from its linear view
//! depth ([`super::GtaoCamera`] + the current `world_from_view`) and projects it
//! through the *previous* frame's `clip_from_world`
//! ([`reproject_prev_uv_gtao`]) — exactly the depth-based camera reprojection `XeGTAO`
//! uses for its temporal denoiser.  A disocclusion, an off-screen reprojection,
//! or a background pixel drops history and falls back to the current frame, and
//! the reprojected history is variance-clipped to the current neighbourhood
//! before the exponential blend so a moving occluder cannot drag a stale
//! contact shadow across the screen (ghosting).
//!
//! This module is the CPU golden the `gtao_temporal.wesl` twin reproduces
//! bit-for-bit within tolerance.  Ambient visibility is a scalar, so the whole
//! filter is the scalar analogue of the RGB SSR temporal in
//! [`crate::screen_space::temporal`].

use bevy_math::{ops, Mat4, Vec2, Vec3};

use super::GtaoCamera;

/// Exponential-accumulation tunables shared by the golden and the
/// `gtao_temporal.wesl` twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GtaoTemporalParams {
    /// Fraction of the reprojected history kept each frame when the sample is
    /// valid *and* the history agrees with the current neighbourhood.  Higher
    /// converges smoother but adds latency; `0.9` blends in roughly a tenth of
    /// the current frame per step (a ~10-frame window).  This is the *upper*
    /// bound: [`gtao_adaptive_history_weight`] pulls it toward
    /// [`Self::min_history_weight`] as the history is clipped.
    pub history_weight: f32,
    /// Floor the adaptive weight decays toward when the reprojected history sits
    /// far outside the clip box (a disocclusion or a moving occluder).  `0`
    /// fully drops stale history on a hard disocclusion; a small positive value
    /// keeps a touch of smoothing through it.  Clamped to `[0, history_weight]`.
    pub min_history_weight: f32,
    /// Standard-deviation multiplier for the variance clip band: history is
    /// clamped to `mean ± variance_gamma·σ` of the current 3x3 neighbourhood
    /// rather than its raw min/max.  This is the AAA variance-clipping method
    /// (Salvi/Karis): the raw min/max hugs a single outlier tap and flickers,
    /// while `mean ± γσ` tracks the neighbourhood distribution.  `1.0` is a good
    /// value for a scalar occlusion signal.
    pub variance_gamma: f32,
}

impl Default for GtaoTemporalParams {
    fn default() -> Self {
        Self {
            history_weight: 0.9,
            min_history_weight: 0.0,
            variance_gamma: 1.0,
        }
    }
}

/// Guards the adaptive-weight knee and the variance square root against zero.
const WEIGHT_EPSILON: f32 = 1.0e-6;

/// Reconstruct a pixel's world position from its linear view depth (via
/// `camera` + the current `world_from_view`) and reproject it through the
/// *previous* frame's `clip_from_world` to find the UV the surface occupied last
/// frame.
///
/// Returns [`None`] when there is nothing to reproject: a background/sky pixel
/// (`linear_depth <= 0`), a point behind the previous camera, or a reprojection
/// that lands outside the previous frustum (an off-screen sample or a
/// disocclusion), all of which fall back to the current frame.
///
/// `uv` follows the prepass convention (origin top-left, `y` down); the returned
/// UV uses the same convention.
pub fn reproject_prev_uv_gtao(
    camera: GtaoCamera,
    world_from_view: Mat4,
    clip_from_world_prev: Mat4,
    uv: Vec2,
    linear_depth: f32,
) -> Option<Vec2> {
    if !(linear_depth.is_finite()) || linear_depth <= 0.0 {
        return None;
    }
    let view = camera.reconstruct([uv.x, uv.y], linear_depth);
    let world = world_from_view * Vec3::from(view).extend(1.0);
    if world.w == 0.0 {
        return None;
    }
    let world = world.truncate() / world.w;

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

/// The result of clipping scalar history to the neighbourhood band: the clamped
/// value plus the *overshoot* — how far outside the band the raw history sat, in
/// half-widths beyond the surface (`0` when already inside).  The overshoot is
/// the disocclusion signal [`gtao_adaptive_history_weight`] uses to decay the blend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GtaoClipResult {
    /// History clamped onto (or left inside) the band `[lo, hi]`.
    pub clipped: f32,
    /// `dist / half - 1` where `dist` is the distance from the band centre and
    /// `half` its half-width; `0` when the history is already inside the band.
    pub overshoot: f32,
}

/// Build the variance clip band `mean ± variance_gamma·σ` from the first two
/// moments of the current 3x3 neighbourhood (`mean` and `mean_sq`, the mean of
/// the squared visibilities).  Scalar analogue of
/// [`crate::screen_space::temporal::variance_clip_box`].
///
/// The variance is clamped non-negative before the square root to absorb the
/// catastrophic cancellation `mean_sq - mean²` can suffer in float.
pub fn variance_clip_band(mean: f32, mean_sq: f32, variance_gamma: f32) -> (f32, f32) {
    let variance = (mean_sq - mean * mean).max(0.0);
    let sigma = ops::sqrt(variance);
    let half = sigma * variance_gamma.max(0.0);
    (mean - half, mean + half)
}

/// Clip scalar `history` to the band `[lo, hi]` and report how far outside it
/// began, in band half-widths.  History inside the band is returned unchanged.
pub fn clip_history(history: f32, lo: f32, hi: f32) -> GtaoClipResult {
    let center = 0.5 * (lo + hi);
    let half = (0.5 * (hi - lo)).max(WEIGHT_EPSILON);
    let dist = history - center;
    let unit = ops::abs(dist) / half;
    if unit > 1.0 {
        GtaoClipResult {
            clipped: center + dist / unit,
            overshoot: unit - 1.0,
        }
    } else {
        GtaoClipResult {
            clipped: history,
            overshoot: 0.0,
        }
    }
}

/// Decay the blend weight from [`GtaoTemporalParams::history_weight`] toward
/// [`GtaoTemporalParams::min_history_weight`] as the reprojected history is
/// clipped further outside the neighbourhood band.
///
/// `overshoot` is [`GtaoClipResult::overshoot`]: `0` when the history sat inside the
/// band (a static, well-matched surface — keep the full weight for maximum
/// denoising) rising toward and past `1` as it disagrees (a disocclusion or a
/// moving occluder — shed history to kill the ghost).  The decay saturates at
/// `overshoot == 1` (one full half-width outside), a knee that reaches the floor
/// quickly without a hard cliff.
pub fn gtao_adaptive_history_weight(params: &GtaoTemporalParams, overshoot: f32) -> f32 {
    let base = params.history_weight.clamp(0.0, 1.0);
    let floor = params.min_history_weight.clamp(0.0, base);
    let t = overshoot.clamp(0.0, 1.0);
    base + (floor - base) * t
}

/// Exponentially accumulate the current frame's denoised visibility with the
/// reprojected `history`, variance-clipping the history to the current
/// neighbourhood band first and adapting the blend weight to how far the history
/// had to be clipped.
///
/// The caller passes the neighbourhood moments (`mean`, `mean_sq`, over the same
/// 3x3 window the shader gathers); [`variance_clip_band`] turns them into the
/// clip band.  History that lands inside the band keeps the full
/// [`GtaoTemporalParams::history_weight`] for maximum denoising; history dragged
/// far outside decays toward [`GtaoTemporalParams::min_history_weight`] via
/// [`gtao_adaptive_history_weight`], so a disocclusion or a moving occluder sheds its
/// stale occlusion instead of ghosting.
///
/// When `valid` is false (a dropped reprojection: off-screen, disoccluded, or a
/// background pixel) the history is discarded and the current frame is returned
/// as-is.  The result is clamped to `[0, 1]` (ambient visibility).
pub fn accumulate_ao(
    params: &GtaoTemporalParams,
    current: f32,
    history: f32,
    mean: f32,
    mean_sq: f32,
    valid: bool,
) -> f32 {
    if !valid {
        return current.clamp(0.0, 1.0);
    }
    let (lo, hi) = variance_clip_band(mean, mean_sq, params.variance_gamma);
    let clip = clip_history(history, lo, hi);
    let w = gtao_adaptive_history_weight(params, clip.overshoot);
    let blended = current * (1.0 - w) + clip.clipped * w;
    blended.clamp(0.0, 1.0)
}

/// Fold a neighbour visibility `sample` into the running first/second moments
/// (`sum`, `sum_sq`, `count`) the temporal filter averages over its 3x3 window.
/// The shader accumulates the identical moments so the two agree on the clip
/// band; the golden exposes it so tests build the same band.
pub fn accumulate_moment(sum: f32, sum_sq: f32, count: f32, sample: f32) -> (f32, f32, f32) {
    (sum + sample, sum_sq + sample * sample, count + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1.0e-4
    }

    // A camera looking down -Z at the world origin, identity world_from_view.
    fn camera() -> GtaoCamera {
        GtaoCamera {
            tan_half_fov_x: 1.0,
            tan_half_fov_y: 1.0,
        }
    }

    // Perspective-ish clip_from_world with the camera at the origin looking
    // down -Z: a point at view depth z projects to the frustum centre.
    fn clip_from_world(z_translation: f32) -> Mat4 {
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
        let prev = clip_from_world(0.0);
        let uv = Vec2::new(0.5, 0.5);
        let out = reproject_prev_uv_gtao(camera(), Mat4::IDENTITY, prev, uv, 4.0)
            .expect("centre pixel reprojects");
        assert!(approx(out.x, 0.5) && approx(out.y, 0.5));
    }

    #[test]
    fn background_and_offscreen_reprojection_drops_history() {
        // Background: non-positive linear depth.
        assert!(reproject_prev_uv_gtao(
            camera(),
            Mat4::IDENTITY,
            clip_from_world(0.0),
            Vec2::new(0.5, 0.5),
            0.0
        )
        .is_none());
        // A large sideways camera translation pushes the point off-screen.
        let prev = clip_from_world(0.0) * Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0));
        assert!(
            reproject_prev_uv_gtao(camera(), Mat4::IDENTITY, prev, Vec2::new(0.5, 0.5), 4.0)
                .is_none()
        );
    }

    #[test]
    fn invalid_sample_returns_current_frame() {
        let params = GtaoTemporalParams::default();
        // history far from current, but the sample is invalid -> ignore history.
        let out = accumulate_ao(&params, 0.3, 0.9, 0.3, 0.09, false);
        assert!(approx(out, 0.3));
    }

    #[test]
    fn matched_history_keeps_full_weight_and_smooths() {
        let params = GtaoTemporalParams::default();
        // Neighbourhood mean 0.5, no variance -> band collapses to [0.5, 0.5].
        // A history of 0.5 sits inside, so weight == history_weight (0.9).
        let out = accumulate_ao(&params, 0.4, 0.5, 0.5, 0.25, true);
        // 0.4*0.1 + 0.5*0.9 = 0.49
        assert!(approx(out, 0.49));
    }

    #[test]
    fn disagreeing_history_sheds_weight_toward_the_floor() {
        let params = GtaoTemporalParams {
            history_weight: 0.9,
            min_history_weight: 0.0,
            variance_gamma: 1.0,
        };
        // Flat neighbourhood (variance 0) at 0.2 with history at 1.0: the band
        // is a point, overshoot huge -> weight decays to the 0.0 floor, so the
        // clipped history (0.2) blends at weight 0 and the result is the current
        // frame.
        let out = accumulate_ao(&params, 0.2, 1.0, 0.2, 0.04, true);
        assert!(approx(out, 0.2));
    }

    #[test]
    fn variance_band_brackets_the_mean() {
        let (lo, hi) = variance_clip_band(0.5, 0.3, 1.0);
        // variance = 0.3 - 0.25 = 0.05, sigma ~= 0.2236
        assert!(lo < 0.5 && hi > 0.5);
        assert!(approx(hi - lo, 2.0 * (0.05_f32).sqrt()));
    }

    #[test]
    fn accumulated_output_stays_in_unit_range() {
        let params = GtaoTemporalParams::default();
        for &(cur, hist) in &[(0.0, 1.0), (1.0, 0.0), (0.5, 0.5), (0.9, 0.1)] {
            let out = accumulate_ao(&params, cur, hist, 0.5, 0.5, true);
            assert!((0.0..=1.0).contains(&out), "out {out} out of range");
        }
    }

    #[test]
    fn moments_accumulate_a_running_sum() {
        let (mut s, mut sq, mut n) = (0.0, 0.0, 0.0);
        for v in [0.2_f32, 0.4, 0.6] {
            (s, sq, n) = accumulate_moment(s, sq, n, v);
        }
        assert!(approx(s, 1.2) && approx(n, 3.0));
        assert!(approx(sq, 0.04 + 0.16 + 0.36));
    }
}
