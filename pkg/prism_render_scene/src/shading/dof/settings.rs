//! Frame-constant tunables for Prism's depth-of-field subsystem.
//!
//! [`PrismDofSettings`] is the single render-world resource the `DoF` passes read
//! to gate the subsystem and drive the thin-lens optics, the bokeh gather and
//! the sharp/blurred blend. It folds the shading golden
//! [`prism_render_shading::DofParams`] (and its embedded
//! [`prism_render_shading::DofCamera`]) field-for-field and default-for-default,
//! so the optics, blur budget and effect scale the GPU twin (`dof.wesl`) uses
//! are exactly the ones the CPU golden was validated against.
//!
//! Two fields have no golden counterpart because the CPU reference models only
//! the optics and the blend, not the device sampling or the depth
//! reconstruction: `sample_count` (disk taps per field) and `scene_units_to_mm`
//! (the reconstructed view distance is in scene units while the golden optics
//! are in millimetres). `near` is carried for the reconstruction block's layout.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `DofParams`; a game can overwrite the resource to retune
//! globally without touching any pass code, mirroring
//! [`super::super::motion_blur`]'s `PrismMotionBlurSettings`.

use bevy_ecs::prelude::Resource;
use prism_render_shading::DofParams;

/// Global depth-of-field settings consumed by the `DoF` passes.
///
/// Disabled by default (matching the golden `enabled_scale == 0`), so bringing
/// the subsystem online is an opt-in the host flips on this resource — either by
/// toggling `enabled` or by raising `enabled_scale` above zero.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismDofSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Distance the lens is focused at, in millimetres (golden `focus_distance`).
    pub focus_distance_mm: f32,
    /// Lens focal length in millimetres (golden `focal_length`).
    pub focal_length_mm: f32,
    /// Aperture f-number / f-stop (golden `aperture_f_stop`); a wider aperture
    /// (smaller number) yields a larger circle of confusion.
    pub aperture_f_stop: f32,
    /// Sensor width in millimetres (golden `sensor_size`), used both to clamp
    /// the `CoC` and to rescale it from mm to pixels.
    pub sensor_width_mm: f32,
    /// Rendered image width in pixels for the mm->pixel `CoC` conversion.
    pub image_width_px: f32,
    /// Largest gather radius (pixels) the blend saturates at and the disk
    /// gather is bounded by (golden `max_coc_pixels`).
    pub max_coc_pixels: f32,
    /// Global effect scale in `[0, 1]`; 0 disables `DoF` (golden `enabled_scale`).
    pub enabled_scale: f32,
    /// Disk taps per field for the bokeh gather (device-only; the golden defines
    /// no sampling pattern). Floored to one before upload.
    pub sample_count: u32,
    /// Scene-unit -> millimetre scale (device-only; the reconstructed view
    /// distance is in scene units while the golden optics are in mm).
    pub scene_units_to_mm: f32,
    /// Positive near-plane distance (scene units) carried in the reconstruction
    /// block for layout parity with the shader struct.
    pub near: f32,
}

impl Default for PrismDofSettings {
    fn default() -> Self {
        // Fold the golden `DofParams` / `DofCamera` defaults so the render-world
        // resource and the CPU reference stay in lockstep.
        let golden = DofParams::default();
        Self {
            // Opt-in: the golden's identity blend (`enabled_scale == 0`) is the
            // disabled state, so the master enable defaults off to match.
            enabled: false,
            focus_distance_mm: golden.camera.focus_distance_mm,
            focal_length_mm: golden.camera.focal_length_mm,
            aperture_f_stop: golden.camera.camera.aperture,
            sensor_width_mm: golden.camera.sensor_width_mm,
            image_width_px: golden.image_width_px,
            max_coc_pixels: golden.max_coc_pixels,
            enabled_scale: golden.enabled_scale,
            // Device-only defaults: a 32-tap spiral per field is a solid AAA
            // baseline, and full-frame scene units default to millimetres * 1000.
            sample_count: 32,
            scene_units_to_mm: 1000.0,
            near: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_dof_params() {
        let settings = PrismDofSettings::default();
        let golden = DofParams::default();
        assert_eq!(settings.focus_distance_mm, golden.camera.focus_distance_mm);
        assert_eq!(settings.focal_length_mm, golden.camera.focal_length_mm);
        assert_eq!(settings.aperture_f_stop, golden.camera.camera.aperture);
        assert_eq!(settings.sensor_width_mm, golden.camera.sensor_width_mm);
        assert_eq!(settings.image_width_px, golden.image_width_px);
        assert_eq!(settings.max_coc_pixels, golden.max_coc_pixels);
        assert_eq!(settings.enabled_scale, golden.enabled_scale);
        // The subsystem is opt-in and the golden default blend is the identity.
        assert!(!settings.enabled);
        assert_eq!(settings.enabled_scale, 0.0);
    }

    #[test]
    fn device_only_defaults_are_sane() {
        let settings = PrismDofSettings::default();
        assert_eq!(settings.sample_count, 32);
        assert_eq!(settings.scene_units_to_mm, 1000.0);
        assert!(settings.near > 0.0);
    }
}
