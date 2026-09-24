//! Frame-constant tunables for Prism's shadow subsystem.
//!
//! [`PrismShadowSettings`] is the single render-world resource the shadow
//! extraction pass reads to decide how many cascades to fit, where to place the
//! `PSSM` split boundaries, how far the directional shadow range reaches, and
//! which bias/filter parameters to bake into every emitted GPU record.  The
//! values are deliberate defaults tuned for a one-sun outdoor scene; a game can
//! overwrite the resource to retune globally without touching the extraction
//! code.
//!
//! The directional/point config blocks are the reference
//! [`DirectionalShadowConfig`] / [`PointShadowConfig`] types verbatim, so the
//! numbers extraction bakes into the GPU records are exactly the ones the CPU
//! golden reference ([`prism_render_shading::shadow`]) and its `shadow.wesl`
//! twin were validated against.

use bevy_ecs::prelude::Resource;
use prism_render_shading::{DirectionalShadowConfig, PointShadowConfig, ShadowFilter};

/// Global shadow-quality settings consumed by the shadow extraction pass.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismShadowSettings {
    /// Number of directional cascades to fit, clamped to the reference
    /// `MAX_CASCADE_COUNT` at extraction time.
    pub cascade_count: u32,
    /// `PSSM` blend factor between the uniform and logarithmic split schemes
    /// (`0` = fully uniform, `1` = fully logarithmic).
    pub split_lambda: f32,
    /// Far distance (world units) the directional cascade set covers.  Bevy's
    /// default perspective projection is an infinite reverse-`Z` frustum whose
    /// far plane is unusable for frustum fitting, so the cascade builder uses
    /// this finite bound instead of the camera's own far plane.
    pub max_distance: f32,
    /// Bias/blend/filter tunables shared by every directional cascade.
    pub directional: DirectionalShadowConfig,
    /// Bias/filter tunables shared by every point-light cube face.  Its
    /// `texel_uv_size` is a placeholder here and is overwritten with the live
    /// atlas texel size during extraction.
    pub point: PointShadowConfig,
}

impl Default for PrismShadowSettings {
    fn default() -> Self {
        Self {
            cascade_count: 4,
            split_lambda: 0.5,
            max_distance: 200.0,
            directional: DirectionalShadowConfig {
                normal_offset_scale: 2.0,
                const_depth_bias: 0.0005,
                slope_depth_bias: 0.002,
                max_depth_bias: 0.02,
                cascade_blend_fraction: 0.1,
                filter: ShadowFilter::Pcf { radius: 2 },
            },
            point: PointShadowConfig {
                const_bias: 0.001,
                slope_bias: 0.002,
                max_bias: 0.02,
                pcf_radius: 1,
                // Overwritten with `ShadowAtlasConfig::texel_uv_size()` during
                // extraction once the live atlas resolution is known.
                texel_uv_size: [0.0, 0.0],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_fit_the_reference_cascade_bounds() {
        let settings = PrismShadowSettings::default();
        assert_eq!(settings.cascade_count, 4);
        assert!(settings.split_lambda >= 0.0 && settings.split_lambda <= 1.0);
        assert!(settings.max_distance > 0.0);
        assert_eq!(settings.directional.filter, ShadowFilter::Pcf { radius: 2 });
        assert_eq!(settings.point.pcf_radius, 1);
    }
}
