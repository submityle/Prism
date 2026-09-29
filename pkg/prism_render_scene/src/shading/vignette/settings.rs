//! Render-world resource gating the vignette pass and feeding the golden
//! tunables into the [`GpuVignetteParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::VignetteParams`] defaults so the
//! render-world resource and the CPU reference stay in lockstep, with two
//! device-side additions the golden does not model: a master `enabled` gate and
//! a global `scale` that fades the whole effect toward identity.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `VignetteParams`; a game can overwrite the resource to
//! retune globally without touching any pass code, mirroring
//! [`super::super::dof`]'s `PrismDofSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::VignetteParams;

use super::abi::GpuVignetteParams;

/// Global vignette settings consumed by the vignette pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass allocates
/// nothing and dispatches nothing). Unlike the golden's neutral `intensity = 0`,
/// the render-world resource ships a visible artistic default so flipping
/// `enabled` on is immediately meaningful; set `intensity = 0` (or `scale = 0`)
/// for the golden identity.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismVignetteSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Fall-off model: `0` = natural `cos^4`, `1` = artistic `smoothstep`
    /// (golden `VignetteMode` discriminant).
    pub mode: u32,
    /// Darkening origin for the artistic model (golden `center`). Frame centre.
    pub center: [f32; 2],
    /// Artistic darkening strength (golden `intensity`).
    pub intensity: f32,
    /// Artistic transition width (golden `smoothness`).
    pub smoothness: f32,
    /// Extra artistic transition softness, added to `smoothness` (golden
    /// `feather`).
    pub feather: f32,
    /// Square (`0`) to circular (`1`) shape blend (golden `roundness`).
    pub roundness: f32,
    /// Horizontal aspect correction for the artistic model (golden
    /// `aspect_ratio`).
    pub aspect_ratio: f32,
    /// Focal ratio for the natural `cos^4` model (golden `focal_ratio`).
    pub focal_ratio: f32,
    /// Global effect scale in `[0, 1]` (device-only); fades the darkening toward
    /// identity. `1` reproduces the golden factor exactly.
    pub scale: f32,
}

impl Default for PrismVignetteSettings {
    fn default() -> Self {
        // Fold the golden `VignetteParams` defaults so the render-world resource
        // and the CPU reference stay in lockstep for every field the golden
        // models; `intensity` is the one deliberate override (see below).
        let golden = VignetteParams::default();
        Self {
            // Opt-in: the subsystem allocates and dispatches nothing until the
            // host flips this on.
            enabled: false,
            mode: golden.mode.as_u32(),
            center: golden.center,
            // Device-visible default: the golden's neutral intensity is 0
            // (identity), but a render-world default of 0 would make enabling the
            // pass a no-op, so ship a moderate artist vignette. Set to 0 for the
            // golden identity.
            intensity: 0.4,
            smoothness: golden.smoothness,
            feather: golden.feather,
            roundness: golden.roundness,
            aspect_ratio: golden.aspect_ratio,
            focal_ratio: golden.focal_ratio,
            // Full-strength effect by default; scale = 1 is the exact golden twin.
            scale: 1.0,
        }
    }
}

impl PrismVignetteSettings {
    /// Builds the [`GpuVignetteParams`] immediate block for a framebuffer of the
    /// given extent from these settings.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuVignetteParams {
        GpuVignetteParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_vignette_params() {
        let settings = PrismVignetteSettings::default();
        let golden = VignetteParams::default();
        assert_eq!(settings.mode, golden.mode.as_u32());
        assert_eq!(settings.center, golden.center);
        assert_eq!(settings.smoothness, golden.smoothness);
        assert_eq!(settings.feather, golden.feather);
        assert_eq!(settings.roundness, golden.roundness);
        assert_eq!(settings.aspect_ratio, golden.aspect_ratio);
        assert_eq!(settings.focal_ratio, golden.focal_ratio);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
    }

    #[test]
    fn device_defaults_are_visible_when_enabled() {
        let settings = PrismVignetteSettings::default();
        // A moderate artist vignette so enabling the pass is immediately visible.
        assert!(settings.intensity > 0.0);
        // Full strength; scale = 1 is the exact golden factor.
        assert_eq!(settings.scale, 1.0);
    }

    #[test]
    fn params_folds_the_extent_and_controls() {
        let settings = PrismVignetteSettings::default();
        let params = settings.params(UVec2::new(1280, 720));
        assert_eq!(params.screen_size, [1280, 720]);
        assert_eq!(params.intensity, settings.intensity);
        assert_eq!(params.mode, settings.mode);
        assert_eq!(params.scale, settings.scale);
    }
}
