//! Backend-neutral CPU golden for **temporal upscaling** (FSR2 / UE-TSR class).
//!
//! Where TAA ([`crate::taa`]) accumulates a *same-resolution* jittered history
//! to anti-alias, temporal upscaling reconstructs a **higher**-resolution image
//! from a stream of *lower*-resolution render targets: the camera is jittered
//! over a longer Halton sequence, each frame is drawn at `render_scale` of the
//! display resolution, and the accumulation resolves those samples onto the full
//! display grid. It is the modern alternative to brute-force supersampling — the
//! renderer shades far fewer pixels yet, integrated over time, resolves detail
//! close to native.
//!
//! The pipeline, one output (display) pixel at a time:
//!
//! 1. [`jitter`] — pick the sub-pixel camera offset for the frame, over a
//!    sequence whose length grows with the upscale ratio.
//! 2. [`history`] — reconstruct the low-resolution input onto the display grid
//!    with a Lanczos-2 kernel and reproject the display-resolution history.
//! 3. [`reproject`] — follow the motion vector to last frame's history and test
//!    for disocclusion (off-screen or depth mismatch).
//! 4. [`clip_history_neighbourhood`] — clip the history to the current
//!    neighbourhood's `YCoCg` colour box to kill ghosting.
//! 5. [`robust`] — lock thin features, grow the temporal accumulation, and shed
//!    history weight on disocclusion.
//! 6. [`sharpen`] — finish with RCAS, scaled by
//!    [`TemporalUpscaleSettings::sharpness`].
//!
//! [`upscale_pixel`] performs steps 2–5 for one pixel (the *accumulation* pass);
//! [`sharpen_upscaled`] is the separate neighbourhood sharpening pass (step 6)
//! that runs over the resolved image. The orchestrator is driven by the
//! architecture contract [`TemporalUpscaleSettings`] and honours
//! [`InvalidationMask`] history invalidation, so it stays aligned with the GPU
//! pipeline byte-for-byte.

pub mod history;
pub mod jitter;
pub mod reproject;
pub mod robust;
pub mod sharpen;

use bevy_math::{Vec2, Vec3};
use prism_render_architecture::history::InvalidationMask;
use prism_render_architecture::temporal_upscale::TemporalUpscaleSettings;

pub use history::{
    clip_history_neighbourhood, lanczos2_weight, lanczos2_weights, lock_strength,
    reconstruct_lanczos2, reconstruct_lanczos2_1d, sinc, update_lock, ClippedHistory, HistoryLock,
    MAX_LOCK_LIFETIME,
};
pub use jitter::{
    upscale_jitter, upscale_phase_count, BASE_UPSCALE_JITTER_LEN, MAX_UPSCALE_JITTER_LEN,
};
pub use reproject::{
    depth_disocclusion, disocclusion_factor, display_to_render, is_offscreen,
    render_to_display, reproject_history_uv,
};
pub use robust::{
    disocclusion_history_weight, history_blend_alpha, history_blend_alpha_clamped, is_thin_feature,
    luma, neighbourhood_luma_range, thin_feature_strength, update_accumulation,
    MAX_ACCUMULATION_FRAMES,
};
pub use sharpen::{apply_rcas, rcas, rcas_luma, rcas_noise, CrossTaps, RcasParams, RCAS_LIMIT};

/// Tunables for the temporal-upscale accumulation pass that are not carried by
/// the architecture [`TemporalUpscaleSettings`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpscaleConfig {
    /// Relative linear-depth tolerance for the disocclusion test (e.g. `0.025`
    /// for 2.5%). Larger values keep more history across depth discontinuities.
    pub depth_tolerance: f32,
    /// Half-width, in standard deviations, of the `YCoCg` neighbourhood clip
    /// box. `1.0` is the tight AAA default.
    pub variance_gamma: f32,
    /// Minimum neighbourhood luma range that counts as a lockable thin feature.
    pub thin_feature_contrast: f32,
    /// Floor on the current-sample blend weight so the accumulation never fully
    /// freezes (e.g. `1/32`).
    pub min_alpha: f32,
}

impl Default for UpscaleConfig {
    fn default() -> Self {
        Self {
            depth_tolerance: 0.025,
            variance_gamma: 1.0,
            thin_feature_contrast: 0.1,
            min_alpha: 1.0 / 32.0,
        }
    }
}

/// Per-pixel inputs to [`upscale_pixel`], all resolved on the display grid.
#[derive(Clone, Copy, Debug)]
pub struct UpscaleInput<'a> {
    /// Current-frame colour reconstructed onto this display pixel (RGB), e.g.
    /// via [`history::reconstruct_lanczos2`] of the low-resolution input.
    pub current_color: Vec3,
    /// The current-frame 3x3 neighbourhood (row-major RGB, centre at index 4)
    /// for the `YCoCg` clip and thin-feature detection.
    pub neighbourhood: &'a [Vec3; 9],
    /// Reprojected previous-frame history colour at this pixel (RGB).
    pub history_color: Vec3,
    /// Reprojected history UV in display space, for the off-screen test
    /// ([`reproject::reproject_history_uv`]).
    pub reprojected_uv: Vec2,
    /// Linear (view-space, positive) current depth at the pixel.
    pub current_depth: f32,
    /// Linear previous depth sampled at [`Self::reprojected_uv`].
    pub history_depth: f32,
    /// Last frame's temporal accumulation count carried for this pixel.
    pub prev_accumulation: f32,
    /// Last frame's thin-feature lock carried for this pixel.
    pub prev_lock: HistoryLock,
}

/// The result of the accumulation pass for one output pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpscaleOutput {
    /// The resolved (accumulated, pre-sharpen) display-resolution colour.
    pub color: Vec3,
    /// Updated accumulation count to carry into next frame.
    pub accumulation: f32,
    /// Updated thin-feature lock to carry into next frame.
    pub lock: HistoryLock,
    /// The `[0, 1]` disocclusion signal that drove the blend (`0` fully
    /// trusted, `1` fully invalidated); useful for debug overlays.
    pub disocclusion: f32,
}

/// Whether this frame's `invalidation` events intersect the feature's
/// [`TemporalUpscaleSettings::invalidation_dependencies`], forcing a full
/// history reset (a camera cut, resize, etc.).
#[must_use]
pub fn history_invalidated(
    settings: &TemporalUpscaleSettings,
    invalidation: InvalidationMask,
) -> bool {
    settings.invalidation_dependencies.intersects(invalidation)
}

/// Resolve one display-resolution output pixel: reproject and clip the history,
/// detect disocclusion and thin features, grow the temporal accumulation, and
/// blend the reconstructed current sample with the trusted history.
///
/// `settings` supplies `render_scale` context and the invalidation dependency
/// mask; `invalidation` is the set of events that occurred this frame. When they
/// intersect, or the reprojection is disoccluded, the accumulation collapses
/// toward the freshly reconstructed current sample instead of ghosting.
///
/// The returned [`UpscaleOutput::color`] is the accumulated colour *before*
/// sharpening; run [`sharpen_upscaled`] as a second pass over the resolved
/// image (RCAS needs a neighbourhood of resolved pixels).
#[must_use]
pub fn upscale_pixel(
    input: &UpscaleInput<'_>,
    settings: &TemporalUpscaleSettings,
    invalidation: InvalidationMask,
    config: &UpscaleConfig,
) -> UpscaleOutput {
    // Neighbourhood colour clip (anti-ghosting); its overshoot feeds the
    // disocclusion signal alongside the depth/off-screen test.
    let clipped = clip_history_neighbourhood(
        input.history_color,
        input.neighbourhood,
        config.variance_gamma,
    );

    let mut disocclusion = disocclusion_factor(
        input.reprojected_uv,
        input.current_depth,
        input.history_depth,
        config.depth_tolerance,
    );
    disocclusion = disocclusion.max(clipped.overshoot.clamp(0.0, 1.0));
    if history_invalidated(settings, invalidation) {
        disocclusion = 1.0;
    }

    // Thin-feature lock: hold sub-pixel detail through the reconstruction.
    let (lo, hi) = neighbourhood_luma_range(input.neighbourhood);
    let center_luma = luma(input.current_color);
    let requested = is_thin_feature(center_luma, lo, hi, config.thin_feature_contrast);
    let lock = update_lock(input.prev_lock, requested, disocclusion);
    let lock_amount = lock_strength(lock);

    // Temporal accumulation and the resulting blend weight.
    let accumulation =
        update_accumulation(input.prev_accumulation, disocclusion, lock_amount);
    let alpha = history_blend_alpha_clamped(accumulation, config.min_alpha);
    // The current sample must fully win on a disocclusion, regardless of how
    // much history had accumulated.
    let current_weight = alpha.max(disocclusion).clamp(0.0, 1.0);

    let color = clipped.color.lerp(input.current_color, current_weight);

    UpscaleOutput {
        color,
        accumulation,
        lock,
        disocclusion,
    }
}

/// The RCAS sharpening pass (step 6) over a five-tap cross of the resolved
/// (accumulated) image, scaled by [`TemporalUpscaleSettings::sharpness`].
///
/// Kept separate from [`upscale_pixel`] because RCAS is a neighbourhood
/// operation: it needs the resolved colours of the pixel and its four
/// axis-neighbours, which only exist once the accumulation pass has written the
/// whole frame. `sharpness == 0` (the default) is an exact identity.
#[must_use]
pub fn sharpen_upscaled(
    resolved_cross: &CrossTaps,
    settings: &TemporalUpscaleSettings,
    denoise: bool,
) -> Vec3 {
    rcas(
        resolved_cross,
        &RcasParams {
            sharpness: settings.sharpness,
            denoise,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(render_scale: f32, sharpness: f32) -> TemporalUpscaleSettings {
        TemporalUpscaleSettings {
            render_scale,
            sharpness,
            invalidation_dependencies: InvalidationMask::CAMERA_CUT,
        }
    }

    fn input<'a>(
        current: Vec3,
        neighbourhood: &'a [Vec3; 9],
        history: Vec3,
        prev_accumulation: f32,
    ) -> UpscaleInput<'a> {
        UpscaleInput {
            current_color: current,
            neighbourhood,
            history_color: history,
            reprojected_uv: Vec2::new(0.5, 0.5),
            current_depth: 10.0,
            history_depth: 10.0,
            prev_accumulation,
            prev_lock: HistoryLock::default(),
        }
    }

    #[test]
    fn static_signal_is_a_fixed_point() {
        // Current == history == neighbourhood, no motion, matched depth: the
        // resolve must be a no-op no matter how much history had accumulated.
        let colour = Vec3::new(0.4, 0.55, 0.2);
        let neigh = [colour; 9];
        let out = upscale_pixel(
            &input(colour, &neigh, colour, 12.0),
            &settings(0.5, 0.0),
            InvalidationMask::default(),
            &UpscaleConfig::default(),
        );
        assert!((out.color - colour).abs().max_element() < 1.0e-5, "{:?}", out.color);
        assert_eq!(out.disocclusion, 0.0);
    }

    #[test]
    fn first_frame_falls_back_to_the_current_sample() {
        // No accumulation yet => alpha 1 => the reconstructed current wins.
        let current = Vec3::new(0.3, 0.6, 0.9);
        let neigh = [current; 9];
        let out = upscale_pixel(
            &input(current, &neigh, Vec3::new(9.0, 9.0, 9.0), 0.0),
            &settings(0.5, 0.0),
            InvalidationMask::default(),
            &UpscaleConfig::default(),
        );
        assert!((out.color - current).abs().max_element() < 1.0e-4, "{:?}", out.color);
    }

    #[test]
    fn depth_disocclusion_collapses_to_current() {
        // History depth disagrees sharply => full disocclusion => current wins
        // and the accumulation resets to a single sample.
        let current = Vec3::new(0.2, 0.2, 0.2);
        let neigh = [current; 9];
        let mut inp = input(current, &neigh, Vec3::new(0.9, 0.1, 0.1), 15.0);
        inp.history_depth = 25.0; // 150% away from the 10.0 current depth
        let out = upscale_pixel(
            &inp,
            &settings(0.5, 0.0),
            InvalidationMask::default(),
            &UpscaleConfig::default(),
        );
        assert_eq!(out.disocclusion, 1.0);
        assert!((out.color - current).abs().max_element() < 1.0e-4, "{:?}", out.color);
        assert_eq!(out.accumulation, 1.0);
    }

    #[test]
    fn offscreen_reprojection_collapses_to_current() {
        let current = Vec3::new(0.5, 0.4, 0.3);
        let neigh = [current; 9];
        let mut inp = input(current, &neigh, Vec3::new(0.9, 0.1, 0.1), 15.0);
        inp.reprojected_uv = Vec2::new(-0.1, 0.5);
        let out = upscale_pixel(
            &inp,
            &settings(0.5, 0.0),
            InvalidationMask::default(),
            &UpscaleConfig::default(),
        );
        assert_eq!(out.disocclusion, 1.0);
        assert!((out.color - current).abs().max_element() < 1.0e-4, "{:?}", out.color);
    }

    #[test]
    fn matching_invalidation_dependency_resets_history() {
        // The camera-cut dependency intersects a camera-cut event => reset even
        // though depth and reprojection agree perfectly.
        let current = Vec3::new(0.3, 0.3, 0.3);
        let neigh = [current; 9];
        let out = upscale_pixel(
            &input(current, &neigh, Vec3::new(0.9, 0.0, 0.0), 15.0),
            &settings(0.5, 0.0),
            InvalidationMask::CAMERA_CUT,
            &UpscaleConfig::default(),
        );
        assert_eq!(out.disocclusion, 1.0);
        assert!((out.color - current).abs().max_element() < 1.0e-4, "{:?}", out.color);
    }

    #[test]
    fn unrelated_invalidation_does_not_reset() {
        // The feature only depends on CAMERA_CUT; an EXPOSURE-only event must
        // not force a reset, so a well-tracked history is preserved.
        let current = Vec3::new(0.3, 0.3, 0.3);
        let neigh = [current; 9];
        let out = upscale_pixel(
            &input(current, &neigh, current, 15.0),
            &settings(0.5, 0.0),
            InvalidationMask::EXPOSURE,
            &UpscaleConfig::default(),
        );
        assert_eq!(out.disocclusion, 0.0);
        assert!(out.accumulation > 15.0, "history should keep growing: {}", out.accumulation);
    }

    #[test]
    fn accumulated_history_dominates_a_tracked_surface() {
        // Well-tracked surface with lots of history: a small current change is
        // integrated slowly, so the output stays close to the history.
        let history = Vec3::new(0.5, 0.5, 0.5);
        let current = Vec3::new(0.6, 0.6, 0.6);
        // Neighbourhood spanning both so the clip does not reject the history.
        let neigh = [
            Vec3::splat(0.5),
            Vec3::splat(0.55),
            Vec3::splat(0.6),
            Vec3::splat(0.52),
            current,
            Vec3::splat(0.58),
            Vec3::splat(0.5),
            Vec3::splat(0.62),
            Vec3::splat(0.54),
        ];
        let out = upscale_pixel(
            &input(current, &neigh, history, MAX_ACCUMULATION_FRAMES),
            &settings(0.5, 0.0),
            InvalidationMask::default(),
            &UpscaleConfig::default(),
        );
        // Closer to history than to current (history weight dominates).
        assert!(
            (out.color.x - history.x).abs() < (out.color.x - current.x).abs(),
            "history should dominate a tracked surface: {:?}",
            out.color
        );
    }

    #[test]
    fn thin_feature_builds_a_lock() {
        // A bright centre on a dark ring is a thin feature => the lock lifetime
        // grows from zero.
        let current = Vec3::splat(1.0);
        let neigh = [
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            current,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        ];
        let out = upscale_pixel(
            &input(current, &neigh, current, 4.0),
            &settings(0.5, 0.0),
            InvalidationMask::default(),
            &UpscaleConfig::default(),
        );
        assert!(out.lock.lifetime > 0.0, "a thin feature must request a lock: {:?}", out.lock);
    }

    #[test]
    fn sharpen_pass_honours_settings_sharpness() {
        let cross = [
            Vec3::splat(0.4),
            Vec3::splat(0.4),
            Vec3::splat(0.6),
            Vec3::splat(0.4),
            Vec3::splat(0.4),
        ];
        // sharpness 0 => identity.
        let off = sharpen_upscaled(&cross, &settings(0.5, 0.0), false);
        assert!((off - Vec3::splat(0.6)).abs().max_element() < 1.0e-5, "{off:?}");
        // sharpness 1 => lifts the bright centre.
        let on = sharpen_upscaled(&cross, &settings(0.5, 1.0), false);
        assert!(on.x > 0.6, "sharpening must lift the centre: {on:?}");
    }

    #[test]
    fn jitter_sequence_lengthens_with_the_render_scale() {
        // The orchestrator's jitter source is the upscale-aware sequence.
        assert_eq!(upscale_phase_count(0.5), BASE_UPSCALE_JITTER_LEN * 4);
    }
}
