//! Render-world resource gating the cross-hatching pass and feeding the golden
//! tunables into the [`GpuHatchingParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::HatchingParams`] defaults so the
//! render-world resource and the CPU reference stay in lockstep, with one
//! device-side change from the golden: the master `enabled` gate defaults off so
//! the subsystem is opt-in.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `HatchingParams`; a game can overwrite the resource to
//! retune globally without touching any pass code, mirroring
//! [`super::super::color_grade`]'s `PrismColorGradeSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::HatchingParams;

use super::abi::GpuHatchingParams;

/// Global cross-hatching settings consumed by the filter pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass allocates
/// nothing and dispatches nothing). The stroke tunables ship the golden neutral
/// values, so once enabled the filter behaves exactly like the golden.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismHatchingSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Stripe frequency (periods across the normalized screen).
    pub frequency: f32,
    /// Stroke half-width in fractional-period units.
    pub thickness: f32,
    /// The three stroke directions (radians), added darkest-last.
    pub angles: [f32; 3],
    /// Descending luminance tiers at which each stroke set switches on.
    pub thresholds: [f32; 3],
}

impl Default for PrismHatchingSettings {
    fn default() -> Self {
        // Fold the golden `HatchingParams` defaults so the render-world resource
        // and the CPU reference stay in lockstep. The stroke tunables are the
        // golden neutrals; `enabled` is forced off so the subsystem is opt-in.
        let golden = HatchingParams::default();
        Self {
            // Opt-in: the subsystem allocates and dispatches nothing until the
            // host flips this on.
            enabled: false,
            frequency: golden.frequency,
            thickness: golden.thickness,
            angles: golden.angles,
            thresholds: golden.thresholds,
        }
    }
}

impl PrismHatchingSettings {
    /// Builds the [`GpuHatchingParams`] immediate block for a framebuffer of the
    /// given extent from these settings.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuHatchingParams {
        GpuHatchingParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::{Vec2, Vec3};

    #[test]
    fn defaults_mirror_the_golden_hatching_params() {
        let settings = PrismHatchingSettings::default();
        let golden = HatchingParams::default();
        assert_eq!(settings.frequency, golden.frequency);
        assert_eq!(settings.thickness, golden.thickness);
        assert_eq!(settings.angles, golden.angles);
        assert_eq!(settings.thresholds, golden.thresholds);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
    }

    #[test]
    fn default_settings_are_a_golden_identity_filter() {
        // The device default is disabled, so hatching any pixel through the CPU
        // golden with the default params returns the scene unchanged. This
        // guards the device default against drift from the golden's identity
        // contract.
        let settings = PrismHatchingSettings::default();
        let golden = HatchingParams {
            enabled: settings.enabled,
            frequency: settings.frequency,
            thickness: settings.thickness,
            angles: settings.angles,
            thresholds: settings.thresholds,
        };
        let scene = Vec3::new(0.2, 0.5, 0.9);
        let out = prism_render_shading::apply_hatching(
            scene,
            Vec2::new(0.13, 0.27),
            &golden,
            Vec3::ZERO,
            Vec3::ONE,
        );
        assert_eq!(out, scene);
    }

    #[test]
    fn params_packs_the_extent_and_controls() {
        let settings = PrismHatchingSettings::default();
        let params = settings.params(UVec2::new(1280, 720));
        assert_eq!(params.screen_size, [1280, 720]);
        assert_eq!(params.frequency, settings.frequency);
        assert_eq!(params.thickness, settings.thickness);
    }
}
