//! Render-world resource gating the `CAS` pass and feeding the golden tunables
//! into the [`GpuCasParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::CasParams`] defaults so the
//! render-world resource and the CPU reference stay in lockstep, with one
//! device-side addition the golden does not model: a master `enabled` gate.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `CasParams`; a game can overwrite the resource to retune
//! globally without touching any pass code, mirroring
//! [`super::super::color_grade`]'s `PrismColorGradeSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::CasParams;

use super::abi::GpuCasParams;

/// Global `CAS` settings consumed by the sharpen pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). The artist controls ship the
/// golden neutral values (`strength = 0`, a bit-exact identity), so a host
/// raises `enabled` and dials in a `strength`.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismCasSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Blend of the sharpened result over the original centre, `[0, 1]`
    /// (golden `strength`).
    pub strength: f32,
    /// Adaptive sharpening peak selector, `[0, 1]` (golden `sharpness`).
    pub sharpness: f32,
}

impl Default for PrismCasSettings {
    fn default() -> Self {
        // Fold the golden `CasParams` defaults so the render-world resource and
        // the CPU reference stay in lockstep. The golden default is `strength`
        // 0 (a bit-exact identity), and `enabled` is off until a host opts in.
        let golden = CasParams::default();
        Self {
            enabled: false,
            strength: golden.strength,
            sharpness: golden.sharpness,
        }
    }
}

impl PrismCasSettings {
    /// Builds the [`GpuCasParams`] immediate block for a framebuffer of the
    /// given extent from these settings.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuCasParams {
        GpuCasParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_cas_params() {
        let settings = PrismCasSettings::default();
        let golden = CasParams::default();
        assert_eq!(settings.strength, golden.strength);
        assert_eq!(settings.sharpness, golden.sharpness);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
    }

    #[test]
    fn params_carry_the_extent_and_controls() {
        let settings = PrismCasSettings {
            enabled: true,
            strength: 0.5,
            sharpness: 0.7,
        };
        let params = settings.params(UVec2::new(1280, 720));
        assert_eq!(params.screen_size, [1280, 720]);
        assert_eq!(params.strength, 0.5);
        assert_eq!(params.sharpness, 0.7);
    }
}
