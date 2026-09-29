//! Frame-constant tunables for Prism's virtual-shadow-map (VSM) subsystem.
//!
//! [`PrismVirtualShadowSettings`] is the single render-world resource the VSM
//! passes read to size the clipmap, budget the physical page atlas and choose
//! the soft-shadow filter width.  It maps directly onto the two golden
//! contracts:
//!
//! * the architecture boundary
//!   [`prism_render_architecture::virtual_shadow::VirtualShadowSettings`]
//!   (physical page budget, page size, clip level count), reproduced by
//!   [`PrismVirtualShadowSettings::contract`); and
//! * the shading golden [`prism_render_shading::ClipmapConfig`], reproduced by
//!   [`PrismVirtualShadowSettings::clipmap`), so the world-scale addressing the
//!   GPU twins (`vsm_page_mark.wesl` / `vsm_sample.wesl`) use is exactly the one
//!   the CPU golden [`prism_render_shading::shadow`] virtual-shadow-map
//!   reference was validated against.
//!
//! The name is prefixed `Prism` to avoid clashing with the architecture
//! contract's own `VirtualShadowSettings`; a game can overwrite the resource to
//! retune globally without touching any pass code.

use bevy_ecs::prelude::Resource;
use prism_render_architecture::virtual_shadow::VirtualShadowSettings;
use prism_render_shading::ClipmapConfig;

/// Global virtual-shadow-map settings consumed by the VSM passes.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismVirtualShadowSettings {
    /// Number of physical pages the atlas can back at once (the working-set
    /// budget the allocator evicts against).
    pub physical_pages: u32,
    /// Edge length of a page in texels.
    pub page_size: u16,
    /// Number of clipmap levels (level `0` is finest).
    pub max_clip_levels: u8,
    /// Pages along one edge of a level's camera-centred resident window.
    pub pages_per_level_edge: u16,
    /// World-space edge length of one texel at the finest clip level.
    pub level0_texel_world_size: f32,
    /// View distance at or below which the finest level is selected; each
    /// doubling of distance advances one level.
    pub level0_max_distance: f32,
    /// World page coordinate mapped to key coordinate `0`; a positive bias lets
    /// negative world pages fit the unsigned page-key fields.
    pub page_coord_bias: i32,
    /// Soft-shadow PCF box half-width in physical texels applied at sample time.
    pub pcf_radius: i32,
}

impl Default for PrismVirtualShadowSettings {
    fn default() -> Self {
        // Defaults match the golden virtual_sm test driver's ClipmapConfig
        // (page_size = 128, 4 levels, 8 pages per edge, 0.1 texel world size,
        // 10.0 level-0 distance, 32768 coord bias) with a 4096-page atlas.
        Self {
            physical_pages: 4096,
            page_size: 128,
            max_clip_levels: 4,
            pages_per_level_edge: 8,
            level0_texel_world_size: 0.1,
            level0_max_distance: 10.0,
            page_coord_bias: 32_768,
            pcf_radius: 1,
        }
    }
}

impl PrismVirtualShadowSettings {
    /// Projects these settings onto the architecture
    /// [`VirtualShadowSettings`] contract (physical budget, page size, level
    /// count).
    pub(crate) fn contract(&self) -> VirtualShadowSettings {
        VirtualShadowSettings {
            physical_pages: self.physical_pages,
            page_size: self.page_size,
            max_clip_levels: self.max_clip_levels,
        }
    }

    /// Builds the golden [`ClipmapConfig`] these settings describe, folding the
    /// architecture contract together with the world-scale tunables the shaders
    /// address pages with.  Delegates to [`ClipmapConfig::from_settings`], which
    /// clamps every field to its valid range.
    pub(crate) fn clipmap(&self) -> ClipmapConfig {
        ClipmapConfig::from_settings(
            &self.contract(),
            self.pages_per_level_edge,
            self.level0_texel_world_size,
            self.level0_max_distance,
            self.page_coord_bias,
        )
    }

    /// Physical pages along one edge of the square atlas grid the sample pass
    /// tiles the physical pages into: `ceil(sqrt(physical_pages))`, always at
    /// least `1` so the atlas is never zero-sized.
    pub(crate) fn physical_pages_per_edge(&self) -> u32 {
        if self.physical_pages == 0 {
            return 1;
        }
        let edge = (f64::from(self.physical_pages)).sqrt().ceil() as u32;
        edge.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_golden_clipmap_config() {
        let s = PrismVirtualShadowSettings::default();
        let c = s.clipmap();
        assert_eq!(c.levels, 4);
        assert_eq!(c.page_size, 128);
        assert_eq!(c.pages_per_level_edge, 8);
        assert!((c.level0_texel_world_size - 0.1).abs() < 1.0e-6);
        assert!((c.level0_max_distance - 10.0).abs() < 1.0e-6);
        assert_eq!(c.page_coord_bias, 32_768);
    }

    #[test]
    fn contract_mirrors_the_architecture_boundary() {
        let s = PrismVirtualShadowSettings::default();
        let contract = s.contract();
        assert_eq!(contract.physical_pages, 4096);
        assert_eq!(contract.page_size, 128);
        assert_eq!(contract.max_clip_levels, 4);
    }

    fn with_physical_pages(physical_pages: u32) -> PrismVirtualShadowSettings {
        PrismVirtualShadowSettings {
            physical_pages,
            ..PrismVirtualShadowSettings::default()
        }
    }

    #[test]
    fn physical_pages_per_edge_is_ceil_sqrt() {
        assert_eq!(with_physical_pages(4096).physical_pages_per_edge(), 64); // exact square
        assert_eq!(with_physical_pages(4097).physical_pages_per_edge(), 65); // rounds up
        assert_eq!(with_physical_pages(1).physical_pages_per_edge(), 1);
        assert_eq!(with_physical_pages(0).physical_pages_per_edge(), 1); // never zero-sized
    }

    #[test]
    fn clipmap_clamps_degenerate_settings() {
        let s = PrismVirtualShadowSettings {
            max_clip_levels: 0,
            page_size: 0,
            pages_per_level_edge: 0,
            ..PrismVirtualShadowSettings::default()
        };
        let c = s.clipmap();
        assert_eq!(c.levels, 1);
        assert_eq!(c.page_size, 1);
        assert_eq!(c.pages_per_level_edge, 1);
    }
}
