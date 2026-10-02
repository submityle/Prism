//! Render-world resource gating the tone-map pass and feeding the operator /
//! white-point selection into the [`GpuTonemapParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::TonemapParams`] and
//! [`prism_render_shading::TonemapOperator`] defaults so the render-world
//! resource and the CPU reference stay in lockstep, with one device-side
//! addition the golden's per-field data does not model as render state: a master
//! `enabled` gate.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `TonemapParams`; a game can overwrite the resource to pick
//! an operator globally without touching any pass code, mirroring
//! [`super::super::gamut_map`]'s `PrismGamutMapSettings`.
//!
//! Tone mapping is display-referred and this pass owns the curve when enabled.
//! Because Prism otherwise delegates tone mapping to the Bevy `PostProcess`
//! node (see `composite.wesl`), a host that flips `enabled` on must also pin its
//! camera to `Tonemapping::None` to avoid double-mapping; the pass is opt-in and
//! disabled by default precisely so the default pipeline keeps the existing
//! single-owner behaviour.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::{TonemapOperator, TonemapParams};

use super::abi::GpuTonemapParams;

/// Global tone-map settings consumed by the tone-map pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass allocates
/// nothing and dispatches nothing). The operator and white point ship the golden
/// defaults, so once enabled the device curve is the exact golden twin; a host
/// raises `enabled`, picks an operator and (for extended Reinhard) a white
/// point.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismTonemapSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// The tone-map operator applied to the pre-exposed linear HDR scene colour
    /// (golden [`TonemapOperator`]).
    pub operator: TonemapOperator,
    /// White point for [`TonemapOperator::ReinhardExtended`] (golden
    /// `reinhard_white_point`); ignored by the other operators.
    pub reinhard_white_point: f32,
}

impl Default for PrismTonemapSettings {
    fn default() -> Self {
        // Fold the golden defaults so the render-world resource and the CPU
        // reference stay in lockstep; `enabled` is off so the default pipeline
        // keeps delegating tone mapping to its existing owner.
        let golden = TonemapParams::default();
        Self {
            enabled: false,
            operator: TonemapOperator::default(),
            reinhard_white_point: golden.reinhard_white_point,
        }
    }
}

impl PrismTonemapSettings {
    /// The operator selector code carried in the immediate block, matching the
    /// CPU enum declaration order and the shader's `apply_tonemap` dispatch.
    pub(crate) fn operator_code(&self) -> u32 {
        match self.operator {
            TonemapOperator::Reinhard => 0,
            TonemapOperator::ReinhardExtended => 1,
            TonemapOperator::AcesNarkowicz => 2,
            TonemapOperator::AcesFitted => 3,
            TonemapOperator::AgX => 4,
        }
    }

    /// Builds the [`GpuTonemapParams`] immediate block for a framebuffer of the
    /// given extent from these settings.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuTonemapParams {
        GpuTonemapParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_are_disabled_and_golden_reinhard() {
        let settings = PrismTonemapSettings::default();
        assert!(!settings.enabled, "the tone-map pass is opt-in");
        assert_eq!(settings.operator, TonemapOperator::Reinhard);
        assert_eq!(
            settings.reinhard_white_point,
            TonemapParams::default().reinhard_white_point
        );
    }

    #[test]
    fn operator_code_matches_the_golden_declaration_order() {
        for (operator, code) in [
            (TonemapOperator::Reinhard, 0),
            (TonemapOperator::ReinhardExtended, 1),
            (TonemapOperator::AcesNarkowicz, 2),
            (TonemapOperator::AcesFitted, 3),
            (TonemapOperator::AgX, 4),
        ] {
            let settings = PrismTonemapSettings {
                operator,
                ..Default::default()
            };
            assert_eq!(settings.operator_code(), code);
        }
    }

    #[test]
    fn params_packs_the_extent_and_controls() {
        let settings = PrismTonemapSettings {
            operator: TonemapOperator::AgX,
            ..Default::default()
        };
        let params = settings.params(UVec2::new(1280, 720));
        assert_eq!(params.screen_size, [1280, 720]);
        assert_eq!(params.operator, 4);
        assert_eq!(params.reinhard_white_point, settings.reinhard_white_point);
    }
}
