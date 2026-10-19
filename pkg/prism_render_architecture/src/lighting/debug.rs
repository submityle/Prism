//! Contract-driven lighting diagnostics metadata.
//!
//! The metadata is deliberately independent from UI code. A debug consumer can
//! register every recipe from [`RecipeId::ALL`] without maintaining a second
//! recipe list, while the renderer remains honest about what each channel
//! means, how it should be scaled, and which related views are useful.

use super::contract::RecipeId;

/// The physical or diagnostic meaning of a recipe output channel.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ChannelSemantics {
    /// Linear outgoing or accumulated light.
    Radiance,
    /// Irradiance before material response.
    Irradiance,
    /// A normalized visibility or confidence value.
    Visibility,
    /// A non-negative temporal age in frames.
    Age,
    /// A normalized diagnostic mask.
    Mask,
}

/// Display scaling for a debug channel.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Scale {
    /// Preserve the channel's linear numeric meaning.
    Linear,
    /// Compress high dynamic range while preserving zero.
    Log,
    /// Apply a perceptual display transform in the consumer.
    Perceptual,
}

/// A stable view identifier that can be combined with a recipe output.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DebugView {
    DirectRadiance,
    IndirectRadiance,
    Irradiance,
    Visibility,
    ReservoirAge,
    Variance,
    BiasBound,
    BudgetUsage,
    Rejection,
    FieldEpoch,
}

/// Display metadata emitted by a lighting recipe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebugMeta {
    /// The semantic meaning of the primary channel.
    pub channel: ChannelSemantics,
    /// Inclusive display-domain bounds before consumer-side clamping.
    pub range: (f32, f32),
    /// The display scale appropriate for the channel.
    pub scale: Scale,
    /// Number of frames used by temporal aggregation.
    pub temporal_window: u32,
    /// Related views that should be offered alongside the primary channel.
    pub combine_with: &'static [DebugView],
}

impl RecipeId {
    /// Stable display metadata for this recipe.
    #[must_use]
    pub const fn debug_meta(self) -> DebugMeta {
        const DIRECT: &[DebugView] = &[DebugView::Visibility, DebugView::BudgetUsage];
        const INDIRECT: &[DebugView] = &[
            DebugView::ReservoirAge,
            DebugView::Variance,
            DebugView::BiasBound,
        ];
        const REFLECTION: &[DebugView] = &[DebugView::Visibility, DebugView::Variance];
        const OFFLINE: &[DebugView] = &[DebugView::Variance, DebugView::BiasBound];

        match self {
            Self::DirectStochasticRt
            | Self::DirectStochasticVsm
            | Self::DirectDeterministicCsm
            | Self::SpecializedDeterministic => DebugMeta {
                channel: ChannelSemantics::Radiance,
                range: (0.0, 16.0),
                scale: Scale::Log,
                temporal_window: 1,
                combine_with: DIRECT,
            },
            Self::DdgiProbes
            | Self::ScreenProbeGi
            | Self::SurfaceCacheSurfel
            | Self::WorldRestir
            | Self::SparseSdfGi
            | Self::RadianceCascades => DebugMeta {
                channel: ChannelSemantics::Irradiance,
                range: (0.0, 8.0),
                scale: Scale::Log,
                temporal_window: 8,
                combine_with: INDIRECT,
            },
            Self::SsgiScreenSpace | Self::SpecularRestir => DebugMeta {
                channel: ChannelSemantics::Radiance,
                range: (0.0, 16.0),
                scale: Scale::Log,
                temporal_window: 4,
                combine_with: REFLECTION,
            },
            Self::PathTracedOffline => DebugMeta {
                channel: ChannelSemantics::Radiance,
                range: (0.0, 32.0),
                scale: Scale::Perceptual,
                temporal_window: 1,
                combine_with: OFFLINE,
            },
        }
    }

    /// Number of compiled shader variants reserved by this recipe's current
    /// contract. This is a planning bound, not a claim that every recipe has
    /// a live GPU backend.
    #[must_use]
    pub const fn variant_count(self) -> u32 {
        match self {
            Self::DirectStochasticRt | Self::SpecularRestir | Self::PathTracedOffline => 4,
            Self::DirectStochasticVsm
            | Self::DdgiProbes
            | Self::ScreenProbeGi
            | Self::SurfaceCacheSurfel
            | Self::WorldRestir
            | Self::SparseSdfGi
            | Self::RadianceCascades => 2,
            Self::DirectDeterministicCsm
            | Self::SpecializedDeterministic
            | Self::SsgiScreenSpace => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_recipe_has_stable_debug_metadata_and_variant_bound() {
        for recipe in RecipeId::ALL {
            let meta = recipe.debug_meta();
            assert!(meta.range.0.is_finite());
            assert!(meta.range.1 > meta.range.0);
            assert!(meta.temporal_window > 0);
            assert!(recipe.variant_count() > 0);
        }
    }

    #[test]
    fn gi_metadata_exposes_temporal_and_error_views() {
        let meta = RecipeId::WorldRestir.debug_meta();
        assert_eq!(meta.channel, ChannelSemantics::Irradiance);
        assert_eq!(meta.temporal_window, 8);
        assert!(meta.combine_with.contains(&DebugView::Variance));
        assert!(meta.combine_with.contains(&DebugView::ReservoirAge));
    }
}
