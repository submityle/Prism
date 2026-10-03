//! Render-world resource gating the area-light `LTC` subsystem and feeding the
//! golden `LTC`-`LUT` bake tunables.
//!
//! Folds the golden
//! [`prism_render_shading::gi::area_light::ltc_lut::DEFAULT_SIZE`] and
//! [`prism_render_shading::gi::area_light::ltc_lut::DEFAULT_FIT_GRID`] so the
//! baked look-up table the render world uploads matches the `CPU` reference,
//! with a device-side `enabled` gate the golden geometry helpers do not model.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden types; a host overwrites it to re-bake the `LUT` or retune the
//! bake tunables without touching any pass code, mirroring
//! [`super::super::world_space_gi`]'s `PrismWorldSpaceGiSettings`.

use bevy_ecs::prelude::Resource;
use prism_render_shading::gi::area_light::ltc_lut::{DEFAULT_FIT_GRID, DEFAULT_SIZE};

/// Global area-light settings consumed by the `LTC`-`LUT` bake and (in block 2)
/// the clustered-lighting resolve path.
///
/// Disabled by default (opt-in; when `false` the subsystem bakes and uploads
/// nothing). The `lut_size` / `fit_grid` ship the golden defaults so the
/// uploaded `LUT` is bit-identical to the `CPU` reference baked by
/// [`prism_render_shading::gi::area_light::ltc_lut::bake_ltc_lut_default`].
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismAreaLightSettings {
    /// Master enable; when `false` the subsystem bakes and uploads nothing.
    pub enabled: bool,
    /// `LTC`-`LUT` resolution per axis (`n·v` x roughness). Mirrors the golden
    /// [`DEFAULT_SIZE`].
    pub lut_size: u32,
    /// Hemisphere fitter grid resolution per axis used while baking. Mirrors the
    /// golden [`DEFAULT_FIT_GRID`].
    pub fit_grid: u32,
}

impl Default for PrismAreaLightSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            lut_size: DEFAULT_SIZE,
            fit_grid: DEFAULT_FIT_GRID,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_lut_dimensions() {
        let settings = PrismAreaLightSettings::default();
        assert_eq!(settings.lut_size, DEFAULT_SIZE);
        assert_eq!(settings.fit_grid, DEFAULT_FIT_GRID);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
        // Non-degenerate bake dimensions.
        assert!(settings.lut_size >= 1);
        assert!(settings.fit_grid >= 2);
    }
}
