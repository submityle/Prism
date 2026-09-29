//! Render-world resource gating the Kuwahara pass and feeding the golden
//! tunables into the [`GpuKuwaharaParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::KuwaharaParams`] defaults so the
//! render-world resource and the CPU reference stay in lockstep, with one
//! device-side change from the golden: the master `enabled` gate defaults off so
//! the subsystem is opt-in.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `KuwaharaParams`; a game can overwrite the resource to
//! retune globally without touching any pass code, mirroring
//! [`super::super::color_grade`]'s `PrismColorGradeSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::KuwaharaParams;

use super::abi::GpuKuwaharaParams;

/// Global Kuwahara settings consumed by the filter pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). The `radius` ships the golden
/// neutral value, so once enabled the filter behaves exactly like the golden.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrismKuwaharaSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Neighbourhood radius `r` (golden `radius`); each of the four overlapping
    /// quadrants is `(r + 1) x (r + 1)`.
    pub radius: u32,
}

impl Default for PrismKuwaharaSettings {
    fn default() -> Self {
        // Fold the golden `KuwaharaParams` defaults so the render-world resource
        // and the CPU reference stay in lockstep. `radius` is the golden neutral;
        // `enabled` is forced off so the subsystem is opt-in.
        let golden = KuwaharaParams::default();
        Self {
            // Opt-in: the subsystem allocates and dispatches nothing until the
            // host flips this on.
            enabled: false,
            radius: golden.radius,
        }
    }
}

impl PrismKuwaharaSettings {
    /// Builds the [`GpuKuwaharaParams`] immediate block for a framebuffer of the
    /// given extent from these settings.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuKuwaharaParams {
        GpuKuwaharaParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    #[test]
    fn defaults_mirror_the_golden_kuwahara_params() {
        let settings = PrismKuwaharaSettings::default();
        let golden = KuwaharaParams::default();
        assert_eq!(settings.radius, golden.radius);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
    }

    #[test]
    fn default_settings_are_a_golden_identity_filter() {
        // The device default is disabled, so filtering any pixel through the CPU
        // golden with the default params returns the centre pixel unchanged.
        // This guards the device default against drift from the golden's
        // identity contract.
        let settings = PrismKuwaharaSettings::default();
        let golden = KuwaharaParams {
            radius: settings.radius,
            enabled: settings.enabled,
        };
        let center = Vec3::new(0.3, 0.5, 0.8);
        let regions = [(Vec3::ZERO, 0.9), (Vec3::ONE, 0.05)];
        let out = prism_render_shading::apply_kuwahara(&regions, center, &golden);
        assert_eq!(out, center);
    }

    #[test]
    fn params_packs_the_extent_and_controls() {
        let settings = PrismKuwaharaSettings::default();
        let params = settings.params(UVec2::new(1280, 720));
        assert_eq!(params.screen_size, [1280, 720]);
        assert_eq!(params.radius, settings.radius);
    }
}
