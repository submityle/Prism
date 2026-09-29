//! Render-world resource gating the posterize pass and feeding the golden
//! tunables into the [`GpuPosterizeParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::PosterizeParams`] defaults so the
//! render-world resource and the CPU reference stay in lockstep. The golden
//! already models a master `enabled` gate, so the render-world resource carries
//! it through directly.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `PosterizeParams`; a game can overwrite the resource to
//! retune globally without touching any pass code, mirroring
//! [`super::super::color_grade`]'s `PrismColorGradeSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::PosterizeParams;

use super::abi::GpuPosterizeParams;

/// Global posterize settings consumed by the posterize pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). The artist controls ship the
/// golden defaults, so a host raises `enabled` and dials in the band counts.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismPosterizeSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Band count for the per-channel `RGB` quantizer (golden `levels`).
    pub levels: f32,
    /// Band count for the luma-preserving quantizer (golden `luma_levels`).
    pub luma_levels: f32,
    /// Select the luma-preserving quantizer over per-channel `RGB` (golden
    /// `use_luma`).
    pub use_luma: bool,
    /// Blend of the quantized result over the input, `[0, 1]` (golden
    /// `strength`).
    pub strength: f32,
}

impl Default for PrismPosterizeSettings {
    fn default() -> Self {
        // Fold the golden `PosterizeParams` defaults so the render-world
        // resource and the CPU reference stay in lockstep. The golden default
        // is `enabled` off (a hard identity), 4-band per-channel `RGB` at full
        // strength.
        let golden = PosterizeParams::default();
        Self {
            enabled: golden.enabled,
            levels: golden.levels,
            luma_levels: golden.luma_levels,
            use_luma: golden.use_luma,
            strength: golden.strength,
        }
    }
}

impl PrismPosterizeSettings {
    /// Builds the [`GpuPosterizeParams`] immediate block for a framebuffer of
    /// the given extent from these settings.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuPosterizeParams {
        GpuPosterizeParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_posterize_params() {
        let settings = PrismPosterizeSettings::default();
        let golden = PosterizeParams::default();
        assert_eq!(settings.levels, golden.levels);
        assert_eq!(settings.luma_levels, golden.luma_levels);
        assert_eq!(settings.use_luma, golden.use_luma);
        assert_eq!(settings.strength, golden.strength);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
        assert_eq!(settings.enabled, golden.enabled);
    }

    #[test]
    fn params_carry_the_extent_and_controls() {
        let settings = PrismPosterizeSettings {
            enabled: true,
            levels: 6.0,
            luma_levels: 4.0,
            use_luma: true,
            strength: 0.5,
        };
        let params = settings.params(UVec2::new(1280, 720));
        assert_eq!(params.screen_size, [1280, 720]);
        assert_eq!(params.levels, 6.0);
        assert_eq!(params.luma_levels, 4.0);
        assert_eq!(params.use_luma, 1);
        assert_eq!(params.enabled, 1);
        assert_eq!(params.strength, 0.5);
    }
}
