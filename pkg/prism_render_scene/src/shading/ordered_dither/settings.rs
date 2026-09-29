//! Render-world resource gating the ordered-dither pass and feeding the golden
//! tunables into the [`GpuOrderedDitherParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::OrderedDitherParams`] defaults so
//! the render-world resource and the CPU reference stay in lockstep. The golden
//! already models the `enabled` master gate, so unlike the gamut map this
//! resource adds no device-only controls: it is a straight mirror of the golden
//! artist params plus the shared framebuffer extent packed at build time.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `OrderedDitherParams`; a game can overwrite the resource to
//! retune globally without touching any pass code, mirroring
//! [`super::super::gamut_map`]'s `PrismGamutMapSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::OrderedDitherParams;

use super::abi::GpuOrderedDitherParams;

/// Global ordered-dither settings consumed by the ordered-dither pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). Every artist control ships the
/// golden default, so with the gate raised the default is the golden dither; a
/// host raises `enabled` and dials in the palette and blend.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismOrderedDitherSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Number of discrete quantisation steps per channel, i.e. the palette /
    /// bit depth (golden `levels`). Clamped up to `2` on device.
    pub levels: u32,
    /// Blend weight of the dithered result over the input (golden `strength`,
    /// `a + (b - a) * t`); `0` is the identity.
    pub strength: f32,
}

impl Default for PrismOrderedDitherSettings {
    fn default() -> Self {
        // Fold the golden `OrderedDitherParams` defaults so the render-world
        // resource and the CPU reference stay in lockstep. `enabled` is off (the
        // subsystem is opt-in); the palette and blend ship the golden defaults.
        let golden = OrderedDitherParams::default();
        Self {
            // Opt-in: the subsystem allocates and dispatches nothing until the
            // host flips this on.
            enabled: false,
            levels: golden.levels,
            strength: golden.strength,
        }
    }
}

impl PrismOrderedDitherSettings {
    /// Builds the [`GpuOrderedDitherParams`] immediate block for a framebuffer
    /// of the given extent from these settings.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuOrderedDitherParams {
        GpuOrderedDitherParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_ordered_dither_params() {
        let settings = PrismOrderedDitherSettings::default();
        let golden = OrderedDitherParams::default();
        assert_eq!(settings.levels, golden.levels);
        assert_eq!(settings.strength, golden.strength);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
    }

    #[test]
    fn default_settings_are_a_golden_identity_dither() {
        // The device defaults fold the golden neutral values, and the golden's
        // own `enabled` flag is what the device folds into its `enabled` gate.
        // With the gate off the golden pass is a bit-exact identity for every
        // pixel regardless of the palette / blend, guarding the device defaults
        // against drift from the golden's identity contract.
        use prism_render_shading::apply_ordered_dither;
        let settings = PrismOrderedDitherSettings::default();
        let golden = OrderedDitherParams {
            levels: settings.levels,
            strength: settings.strength,
            enabled: settings.enabled,
        };
        for (px, py) in [(0u32, 0u32), (1, 2), (3, 3), (17, 42)] {
            for rgb in [[0.15_f32, 0.4, 0.85], [0.0, 0.5, 1.0], [0.33, 0.66, 0.99]] {
                assert_eq!(apply_ordered_dither(rgb, px, py, &golden), rgb);
            }
        }
    }

    #[test]
    fn params_packs_the_extent_and_controls() {
        let settings = PrismOrderedDitherSettings::default();
        let params = settings.params(UVec2::new(2560, 1440));
        assert_eq!(params.screen_size, [2560, 1440]);
        assert_eq!(params.levels, settings.levels);
        assert_eq!(params.strength, settings.strength);
    }
}
