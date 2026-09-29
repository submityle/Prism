//! Immediate (push-constant) blocks shared with the temporal-upscale shaders.
//!
//! Kept byte-for-byte in sync with the WESL structs so the GPU twin reads the
//! same layout the CPU golden [`prism_render_shading::upscale`] was validated
//! against:
//!
//! * [`GpuUpscaleReconstructParams`] mirrors the `UpscaleReconstructParams`
//!   block in `shaders/upscale_reconstruct.wesl` — the two grid extents, the
//!   golden [`UpscaleConfig`] tunables and the accumulation/lock caps, plus the
//!   [`InvalidationMask`] dependency/event pair and the history-validity flag
//!   the reconstruction consults before it trusts the reprojected history.
//! * [`GpuUpscaleRcasParams`] mirrors the `UpscaleRcasParams` block in
//!   `shaders/upscale_rcas.wesl` — the output extent and the golden
//!   [`RcasParams`] sharpen tunables.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::history::InvalidationMask;
use prism_render_architecture::temporal_upscale::TemporalUpscaleSettings;
use prism_render_shading::upscale::{
    RcasParams, UpscaleConfig, MAX_ACCUMULATION_FRAMES, MAX_LOCK_LIFETIME,
};

/// 8x8 pixel tile per workgroup, matching both shaders' `@workgroup_size(8,8,1)`.
pub(crate) const UPSCALE_WORKGROUP_SIZE: u32 = 8;

/// Render/display extents + golden [`UpscaleConfig`] tunables + accumulation and
/// lock caps + history-invalidation state, uploaded as the
/// `upscale_reconstruct.wesl` immediate block.
///
/// All fourteen fields are 4-byte scalars laid out back to back, so the
/// `#[repr(C)]` record is 56 bytes with no padding and matches the WESL
/// `UpscaleReconstructParams` struct field-for-field, in order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuUpscaleReconstructParams {
    /// Low-resolution render-target width (this frame's draw size), in texels.
    pub render_width: u32,
    /// Low-resolution render-target height, in texels.
    pub render_height: u32,
    /// Display (output / history) width, in texels.
    pub display_width: u32,
    /// Display (output / history) height, in texels.
    pub display_height: u32,
    /// render / display in `(0, 1]`; the grid-conversion scale.
    pub render_scale: f32,
    /// Relative linear-depth tolerance for the disocclusion test (e.g. 0.025).
    pub depth_tolerance: f32,
    /// Half-width, in standard deviations, of the `YCoCg` neighbourhood clip box.
    pub variance_gamma: f32,
    /// Minimum neighbourhood luma range that counts as a lockable thin feature.
    pub thin_feature_contrast: f32,
    /// Floor on the current-sample blend weight (e.g. 1/32).
    pub min_alpha: f32,
    /// Upper bound on the temporal accumulation count, in frames.
    pub max_accumulation: f32,
    /// Upper bound on a history-lock lifetime, in frames.
    pub max_lock_lifetime: f32,
    /// The settings' history-invalidation dependency mask.
    pub invalidation_dependencies: u32,
    /// This frame's history-invalidation events; a non-empty intersection with
    /// `invalidation_dependencies` forces a full reset.
    pub invalidation_events: u32,
    /// `1` when a valid previous frame exists (no first frame / resize / cut).
    pub valid_history: u32,
}

impl GpuUpscaleReconstructParams {
    /// Builds the reconstruction params from the render/display extents, the
    /// architecture [`TemporalUpscaleSettings`] (its `render_scale` and the
    /// `invalidation_dependencies` mask), the golden accumulation-pass
    /// [`UpscaleConfig`] tunables, this frame's `invalidation_events`, and
    /// whether the ping-pong history is trustworthy this frame.
    ///
    /// The accumulation and lock caps are folded in from the golden
    /// [`MAX_ACCUMULATION_FRAMES`] and [`MAX_LOCK_LIFETIME`] so the GPU twin
    /// clamps identically to the CPU golden.
    pub(crate) fn new(
        render_extent: (u32, u32),
        display_extent: (u32, u32),
        settings: &TemporalUpscaleSettings,
        config: &UpscaleConfig,
        invalidation_events: InvalidationMask,
        valid_history: bool,
    ) -> Self {
        Self {
            render_width: render_extent.0,
            render_height: render_extent.1,
            display_width: display_extent.0,
            display_height: display_extent.1,
            render_scale: settings.render_scale,
            depth_tolerance: config.depth_tolerance,
            variance_gamma: config.variance_gamma,
            thin_feature_contrast: config.thin_feature_contrast,
            min_alpha: config.min_alpha,
            max_accumulation: MAX_ACCUMULATION_FRAMES,
            max_lock_lifetime: MAX_LOCK_LIFETIME,
            invalidation_dependencies: settings.invalidation_dependencies.0,
            invalidation_events: invalidation_events.0,
            valid_history: u32::from(valid_history),
        }
    }
}

/// Display extent + golden [`RcasParams`] sharpen tunables, uploaded as the
/// `upscale_rcas.wesl` immediate block.
///
/// Four 4-byte scalars laid out back to back, so the `#[repr(C)]` record is 16
/// bytes with no padding and matches the WESL `UpscaleRcasParams` struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuUpscaleRcasParams {
    /// Display (output) width, in texels.
    pub width: u32,
    /// Display (output) height, in texels.
    pub height: u32,
    /// Lobe scale in `[0, 1]` from [`TemporalUpscaleSettings::sharpness`]; `0`
    /// disables sharpening (exact identity), `1` is full-strength RCAS.
    pub sharpness: f32,
    /// `1` to attenuate the lobe by the noise estimate, else `0`.
    pub denoise: u32,
}

impl GpuUpscaleRcasParams {
    /// Builds the RCAS params from the display extent and the golden
    /// [`RcasParams`] sharpen tunables. The `sharpness` is carried straight
    /// through from [`TemporalUpscaleSettings::sharpness`] (see
    /// [`RcasParams::from`] callers), so `0` is an exact identity.
    pub(crate) fn new(display_extent: (u32, u32), params: &RcasParams) -> Self {
        Self {
            width: display_extent.0,
            height: display_extent.1,
            sharpness: params.sharpness,
            denoise: u32::from(params.denoise),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upscale_workgroup_size_matches_shaders() {
        assert_eq!(UPSCALE_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn reconstruct_params_layout_matches_the_wesl_immediate_block() {
        // Fourteen 4-byte scalars, no padding.
        assert_eq!(size_of::<GpuUpscaleReconstructParams>(), 56);
        assert_eq!(align_of::<GpuUpscaleReconstructParams>(), 4);
    }

    #[test]
    fn reconstruct_params_fold_in_the_golden_defaults() {
        let settings = TemporalUpscaleSettings::default();
        let config = UpscaleConfig::default();
        let params = GpuUpscaleReconstructParams::new(
            (960, 540),
            (1920, 1080),
            &settings,
            &config,
            InvalidationMask::default(),
            true,
        );

        assert_eq!(params.render_width, 960);
        assert_eq!(params.render_height, 540);
        assert_eq!(params.display_width, 1920);
        assert_eq!(params.display_height, 1080);
        // Straight from the architecture settings / golden config defaults.
        assert_eq!(params.render_scale, settings.render_scale);
        assert_eq!(params.depth_tolerance, config.depth_tolerance);
        assert_eq!(params.variance_gamma, config.variance_gamma);
        assert_eq!(params.thin_feature_contrast, config.thin_feature_contrast);
        assert_eq!(params.min_alpha, config.min_alpha);
        // Golden accumulation / lock caps.
        assert_eq!(params.max_accumulation, MAX_ACCUMULATION_FRAMES);
        assert_eq!(params.max_lock_lifetime, MAX_LOCK_LIFETIME);
        // The default dependency mask is "everything" (u32::MAX).
        assert_eq!(
            params.invalidation_dependencies,
            settings.invalidation_dependencies.0
        );
        assert_eq!(params.invalidation_events, 0);
        assert_eq!(params.valid_history, 1);
    }

    #[test]
    fn reconstruct_params_carry_the_invalidation_pair_and_history_flag() {
        let settings = TemporalUpscaleSettings {
            render_scale: 0.5,
            sharpness: 0.0,
            invalidation_dependencies: InvalidationMask::CAMERA_CUT,
        };
        let params = GpuUpscaleReconstructParams::new(
            (1, 1),
            (2, 2),
            &settings,
            &UpscaleConfig::default(),
            InvalidationMask::CAMERA_CUT,
            false,
        );
        assert_eq!(
            params.invalidation_dependencies,
            InvalidationMask::CAMERA_CUT.0
        );
        assert_eq!(params.invalidation_events, InvalidationMask::CAMERA_CUT.0);
        // A matching dependency/event intersection is what the shader treats as
        // a reset; here it also has no trusted history.
        assert!(settings
            .invalidation_dependencies
            .intersects(InvalidationMask::CAMERA_CUT));
        assert_eq!(params.valid_history, 0);
    }

    #[test]
    fn rcas_params_layout_matches_the_wesl_immediate_block() {
        // Four 4-byte scalars, no padding.
        assert_eq!(size_of::<GpuUpscaleRcasParams>(), 16);
        assert_eq!(align_of::<GpuUpscaleRcasParams>(), 4);
    }

    #[test]
    fn rcas_params_fold_in_the_golden_defaults() {
        // Golden `RcasParams::default()` is sharpness 0 (identity), denoise off.
        let params = GpuUpscaleRcasParams::new((1920, 1080), &RcasParams::default());
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
        assert_eq!(params.sharpness, 0.0);
        assert_eq!(params.denoise, 0);

        let sharp = GpuUpscaleRcasParams::new(
            (800, 600),
            &RcasParams {
                sharpness: 0.75,
                denoise: true,
            },
        );
        assert_eq!(sharp.width, 800);
        assert_eq!(sharp.height, 600);
        assert_eq!(sharp.sharpness, 0.75);
        assert_eq!(sharp.denoise, 1);
    }
}
