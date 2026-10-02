//! Render-world resource gating the light-routing cull pass and feeding the
//! per-light routing records + [`GpuLightRoutingParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::LightRouting`] defaults so the
//! render-world resource and the CPU reference stay in lockstep, with one
//! device-side addition the golden decision math does not model: a master
//! `enabled` gate.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `LightRouting`; a game overwrites the resource to register
//! each light's channel/layer routing and the receiving primitive's channel
//! mask globally without touching any pass code, mirroring
//! [`super::super::color_grade`]'s `PrismColorGradeSettings`.

use bevy_ecs::prelude::Resource;
use prism_render_shading::{LightRouting, LightingChannelMask};

use super::abi::GpuLightRoutingParams;

/// Global light-routing settings consumed by the cull pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). Ships a single golden-default
/// routing record and the golden default primitive channel mask, so with the
/// gate on the default cull keeps exactly the one channel-0 / key-layer light
/// the CPU reference would.
///
/// Carries a `Vec` of records, so it is `Clone` but not `Copy`.
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrismLightRoutingSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// The receiving primitive's lighting-channel bitmask (golden
    /// `LightingChannelMask`): a light survives the cull iff its channel mask
    /// intersects this one.
    pub primitive_channels: LightingChannelMask,
    /// One routing record per light, indexed by the light's slot in the
    /// cluster light list. Uploaded verbatim into the per-view routing buffer.
    pub records: Vec<LightRouting>,
}

impl Default for PrismLightRoutingSettings {
    fn default() -> Self {
        // Fold the golden defaults: one channel-0 / key-layer light and a
        // channel-0 primitive, so the default cull keeps exactly that light,
        // matching the CPU reference. `enabled` is off (opt-in).
        Self {
            enabled: false,
            primitive_channels: LightingChannelMask::default(),
            records: vec![LightRouting::default()],
        }
    }
}

impl PrismLightRoutingSettings {
    /// Builds the [`GpuLightRoutingParams`] immediate block from these settings.
    pub(crate) fn params(&self) -> GpuLightRoutingParams {
        GpuLightRoutingParams::from_settings(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_light_routing() {
        let settings = PrismLightRoutingSettings::default();
        // Opt-in gate off by default.
        assert!(!settings.enabled);
        // A single golden-default routing record and the golden primitive mask.
        assert_eq!(settings.records, vec![LightRouting::default()]);
        assert_eq!(settings.primitive_channels, LightingChannelMask::default());
    }

    #[test]
    fn default_settings_keep_the_single_golden_light() {
        // The device defaults route the one default light onto the default
        // primitive exactly as the CPU golden would (channel 0 shared).
        let settings = PrismLightRoutingSettings::default();
        assert!(settings.records[0].lights_primitive(settings.primitive_channels));
    }

    #[test]
    fn params_forward_the_record_count_and_primitive_mask() {
        let settings = PrismLightRoutingSettings::default();
        let params = settings.params();
        assert_eq!(params.light_count, 1);
        assert_eq!(params.word_count, 1);
        assert_eq!(
            params.primitive_channels,
            settings.primitive_channels.bits()
        );
    }
}
