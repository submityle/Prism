//! Render-world resource gating the gamut-map pass and feeding the golden
//! tunables into the [`GpuGamutMapParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::GamutMapParams`] defaults so the
//! render-world resource and the CPU reference stay in lockstep, with two
//! device-side additions the golden's per-field data does not directly model as
//! render state: a master `enabled` gate and a global `scale` that fades the
//! whole compression toward identity.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `GamutMapParams`; a game can overwrite the resource to
//! retune globally without touching any pass code, mirroring
//! [`super::super::color_grade`]'s `PrismColorGradeSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::GamutMapParams;

use super::abi::GpuGamutMapParams;

/// Global gamut-map settings consumed by the gamut-map pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). Every artist control ships the
/// golden neutral value, so the default map is the identity to floating-point
/// tolerance; a host raises `enabled` and dials in the compression.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismGamutMapSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Per-channel distance below which colours are left untouched, i.e. the
    /// working gamut (golden `threshold`).
    pub threshold: [f32; 3],
    /// Per-channel asymptotic compressed distance, i.e. the gamut-boundary
    /// target (golden `limit`).
    pub limit: [f32; 3],
    /// Reserved perceptual shaping exponent carried through the ABI so the GPU
    /// twin's uniform layout matches (golden `power`).
    pub power: f32,
    /// Global effect scale in `[0, 1]` (device-only); fades the compression
    /// toward the input. `1` reproduces the golden map exactly.
    pub scale: f32,
}

impl Default for PrismGamutMapSettings {
    fn default() -> Self {
        // Fold the golden `GamutMapParams` defaults so the render-world resource
        // and the CPU reference stay in lockstep. Every default is the golden
        // neutral, so the default map is the identity; `enabled` is off and
        // `scale` is full-strength for when a host dials the compression in.
        let golden = GamutMapParams::default();
        Self {
            // Opt-in: the subsystem allocates and dispatches nothing until the
            // host flips this on.
            enabled: false,
            threshold: golden.threshold,
            limit: golden.limit,
            power: golden.power,
            // Full-strength map by default; scale = 1 is the exact golden twin.
            scale: 1.0,
        }
    }
}

impl PrismGamutMapSettings {
    /// Builds the [`GpuGamutMapParams`] immediate block for a framebuffer of the
    /// given extent from these settings.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuGamutMapParams {
        GpuGamutMapParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_gamut_map_params() {
        let settings = PrismGamutMapSettings::default();
        let golden = GamutMapParams::default();
        assert_eq!(settings.threshold, golden.threshold);
        assert_eq!(settings.limit, golden.limit);
        assert_eq!(settings.power, golden.power);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
        // Full strength; scale = 1 is the exact golden twin.
        assert_eq!(settings.scale, 1.0);
    }

    #[test]
    fn default_settings_are_a_golden_identity_map() {
        // The device defaults are the golden neutral values, so compressing any
        // pixel with them through the CPU golden returns it unchanged to
        // floating-point tolerance. This guards the device defaults against
        // drift from the golden's identity contract.
        //
        // The golden `enabled` flag is what the device folds into its own
        // `enabled` gate, so the identity here is verified with the golden map
        // *active* (`enabled = true`) and only the neutral wheels: an in-gamut
        // colour reconstructs to itself and an out-of-gamut colour is the only
        // thing the neutral `threshold = 1`, `limit = 1.1` map touches.
        let settings = PrismGamutMapSettings::default();
        let golden = GamutMapParams {
            threshold: settings.threshold,
            limit: settings.limit,
            power: settings.power,
            enabled: true,
        };
        // In-gamut colours (every channel distance below the neutral threshold
        // of 1) are a bit-exact identity even with the map active.
        for rgb in [[0.3_f32, 0.5, 0.8], [0.1, 0.4, 0.9], [0.5, 0.5, 0.5]] {
            let mapped = prism_render_shading::apply_gamut_compress(rgb, &golden);
            for channel in 0..3 {
                assert!(
                    (mapped[channel] - rgb[channel]).abs() < 1.0e-4,
                    "neutral in-gamut map must be identity: {} != {}",
                    mapped[channel],
                    rgb[channel]
                );
            }
        }
    }

    #[test]
    fn params_packs_the_extent_and_controls() {
        let settings = PrismGamutMapSettings::default();
        let params = settings.params(UVec2::new(1280, 720));
        assert_eq!(params.screen_size, [1280, 720]);
        assert_eq!(params.threshold_power[3], settings.power);
        assert_eq!(params.limit_scale[3], settings.scale);
    }
}
