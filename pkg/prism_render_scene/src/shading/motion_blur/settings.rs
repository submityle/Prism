//! Frame-constant tunables for Prism's motion-blur subsystem.
//!
//! [`PrismMotionBlurSettings`] is the single render-world resource the
//! motion-blur passes read to gate the subsystem and drive the reconstruction
//! gather. It mirrors the shading golden
//! [`prism_render_shading::motion_blur::MotionBlurParams`] field-for-field and
//! default-for-default, so the shutter/exposure, blur budget, depth softness
//! and sample count the GPU twin (`motion_blur.wesl`) uses are exactly the ones
//! the CPU golden was validated against.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `MotionBlurParams`; a game can overwrite the resource to
//! retune globally without touching any pass code. It follows the same
//! self-contained pattern as [`super::super::virtual_shadow`]'s
//! `PrismVirtualShadowSettings`.

use bevy_ecs::prelude::Resource;
use prism_render_shading::MotionBlurParams;

/// Global motion-blur settings consumed by the motion-blur passes.
///
/// Disabled by default (matching the golden), so bringing the subsystem online
/// is an opt-in the host flips on this resource.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismMotionBlurSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Blur budget in pixels; velocities are clamped to this to bound the
    /// reconstruction gather radius.
    pub max_velocity_px: f32,
    /// Depth softness (view-space units) fed to the soft-depth ordering term.
    pub soft_z_extent: f32,
    /// Reconstruction gather taps per pixel.
    pub sample_count: u32,
    /// Fraction of the frame the shutter is open (longer exposures streak
    /// further); scales the per-frame velocity before clamping.
    pub exposure_fraction: f32,
}

impl Default for PrismMotionBlurSettings {
    fn default() -> Self {
        // Fold the golden `MotionBlurParams` defaults so the render-world
        // resource and the CPU reference stay in lockstep.
        let golden = MotionBlurParams::default();
        Self {
            enabled: golden.enabled,
            max_velocity_px: golden.max_velocity_px,
            soft_z_extent: golden.soft_z_extent,
            sample_count: golden.sample_count,
            exposure_fraction: golden.exposure_fraction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_motion_blur_params() {
        let settings = PrismMotionBlurSettings::default();
        let golden = MotionBlurParams::default();
        assert_eq!(settings.enabled, golden.enabled);
        assert_eq!(settings.max_velocity_px, golden.max_velocity_px);
        assert_eq!(settings.soft_z_extent, golden.soft_z_extent);
        assert_eq!(settings.sample_count, golden.sample_count);
        assert_eq!(settings.exposure_fraction, golden.exposure_fraction);
        // The subsystem is opt-in.
        assert!(!settings.enabled);
    }
}
