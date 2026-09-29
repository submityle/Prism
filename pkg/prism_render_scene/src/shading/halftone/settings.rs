//! Render-world resource gating the halftone pass and feeding the golden
//! tunables into the [`GpuHalftoneParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::HalftoneParams`] defaults so the
//! render-world resource and the CPU reference stay in lockstep, with one
//! device-side change from the golden: the master `enabled` gate defaults off so
//! the subsystem is opt-in.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `HalftoneParams`; a game can overwrite the resource to
//! retune globally without touching any pass code, mirroring
//! [`super::super::color_grade`]'s `PrismColorGradeSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::HalftoneParams;

use super::abi::GpuHalftoneParams;

/// Global halftone settings consumed by the filter pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass allocates
/// nothing and dispatches nothing). The screen tunables ship the golden neutral
/// values, so once enabled the filter behaves exactly like the golden.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismHalftoneSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Side length of a screen cell, in pixels.
    pub cell_size: f32,
    /// Rotation of the dot lattice, in radians.
    pub angle: f32,
}

impl Default for PrismHalftoneSettings {
    fn default() -> Self {
        // Fold the golden `HalftoneParams` defaults so the render-world resource
        // and the CPU reference stay in lockstep. The screen tunables are the
        // golden neutrals; `enabled` is forced off so the subsystem is opt-in.
        let golden = HalftoneParams::default();
        Self {
            // Opt-in: the subsystem allocates and dispatches nothing until the
            // host flips this on.
            enabled: false,
            cell_size: golden.cell_size,
            angle: golden.angle,
        }
    }
}

impl PrismHalftoneSettings {
    /// Builds the [`GpuHalftoneParams`] immediate block for a framebuffer of the
    /// given extent from these settings.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuHalftoneParams {
        GpuHalftoneParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::{Vec2, Vec3};

    #[test]
    fn defaults_mirror_the_golden_halftone_params() {
        let settings = PrismHalftoneSettings::default();
        let golden = HalftoneParams::default();
        assert_eq!(settings.cell_size, golden.cell_size);
        assert_eq!(settings.angle, golden.angle);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
    }

    #[test]
    fn default_settings_are_a_golden_identity_filter() {
        // The device default is disabled, so halftoning any pixel through the
        // CPU golden with the default params returns the scene unchanged. This
        // guards the device default against drift from the golden's identity
        // contract.
        let settings = PrismHalftoneSettings::default();
        let golden = HalftoneParams {
            cell_size: settings.cell_size,
            angle: settings.angle,
            enabled: settings.enabled,
        };
        let scene = Vec3::new(0.2, 0.5, 0.9);
        let out = prism_render_shading::apply_halftone(
            scene,
            Vec2::new(8.0, 5.0),
            &golden,
            Vec3::ZERO,
            Vec3::ONE,
        );
        assert_eq!(out, scene);
    }

    #[test]
    fn params_packs_the_extent_and_controls() {
        let settings = PrismHalftoneSettings::default();
        let params = settings.params(UVec2::new(1280, 720));
        assert_eq!(params.screen_size, [1280, 720]);
        assert_eq!(params.cell_size, settings.cell_size);
        assert_eq!(params.angle, settings.angle);
    }
}
