//! Resolves the golden VSM driver's per-frame residency into the flat list of
//! physical atlas pages the raster fill pass repaints this frame.
//!
//! [`request_keys`](super::decode::request_keys) decoded the marked resident-window
//! slots into the [`ShadowPageKey`]s the driver made resident; this module pairs
//! each of those keys with the physical atlas page the golden
//! [`VirtualPageTable`] assigned it and precomputes the world-space light-plane
//! rectangle plus the atlas tile the raster pass renders caster depth into. It
//! is pure and CPU-testable, sharing the same [`atlas_tile_origin`] tiling geometry
//! the sampler (`vsm_sample.wesl`) reads a resident page's depth back with.

use bevy_math::{IVec2, UVec2, Vec2};
use prism_render_shading::{ClipmapConfig, Residency, ShadowPageKey, VirtualPageTable};

use super::super::atlas::atlas_tile_origin;

/// One resident clipmap page the raster fill pass renders caster depth into,
/// resolved to its backing physical atlas tile and world-space light-plane
/// footprint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VsmRenderPage {
    /// Physical atlas page index backing this clipmap page.
    pub physical_page: u32,
    /// Top-left texel of the page's tile inside the physical atlas texture.
    pub atlas_tile_origin: UVec2,
    /// World-space light-plane origin of the page's lower-left corner.
    pub world_origin: Vec2,
    /// World-space edge length of the page (`clipmap.page_world_size(level)`).
    pub world_size: f32,
    /// Clipmap level the page belongs to (0 = finest).
    pub level: u16,
}

/// Pairs every this-frame render key with its backing physical atlas page and
/// precomputes the raster fill footprint.
///
/// Keys whose page is not resident in `table` (evicted before the driver promoted
/// it) are dropped: the raster pass only repaints pages the page table actually
/// points the sampler at. `clipmap` and `table` must come from the same driver
/// frame that produced `keys`.
pub(crate) fn build_render_pages(
    clipmap: &ClipmapConfig,
    table: &VirtualPageTable,
    keys: &[ShadowPageKey],
    physical_pages_per_edge: u32,
    page_size: u32,
) -> Vec<VsmRenderPage> {
    keys.iter()
        .filter_map(|key| {
            let physical_page = table.get(key).and_then(Residency::physical_page)?;
            let world_size = clipmap.page_world_size(key.level);
            let page = IVec2::new(
                i32::from(key.x) - clipmap.page_coord_bias,
                i32::from(key.y) - clipmap.page_coord_bias,
            );
            Some(VsmRenderPage {
                physical_page,
                atlas_tile_origin: atlas_tile_origin(
                    physical_page,
                    physical_pages_per_edge,
                    page_size,
                ),
                world_origin: page.as_vec2() * world_size,
                world_size,
                level: key.level,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_shading::{VirtualShadowMap, VirtualShadowSettings};

    fn driven() -> (VirtualShadowMap, ClipmapConfig, ShadowPageKey) {
        let settings = VirtualShadowSettings {
            physical_pages: 64,
            ..Default::default()
        };
        let mut vsm = VirtualShadowMap::from_settings(&settings, 8, 0.1, 10.0, 32_768);
        let clipmap = *vsm.clipmap();
        let key = clipmap.page_key(0, 0, IVec2::new(0, 0)).unwrap();
        vsm.drive_frame_with_requests(0, Vec2::ZERO, false, &[key], &[]);
        (vsm, clipmap, key)
    }

    #[test]
    fn render_page_geometry_matches_the_resident_page() {
        let (vsm, clipmap, key) = driven();
        let page_size = u32::from(clipmap.page_size);
        let pages = build_render_pages(&clipmap, vsm.table(), &[key], 8, page_size);
        assert_eq!(pages.len(), 1);
        let page = pages[0];
        assert_eq!(page.level, 0);
        assert_eq!(page.world_origin, Vec2::ZERO);
        assert_eq!(page.world_size, clipmap.page_world_size(0));
        assert_eq!(
            page.atlas_tile_origin,
            atlas_tile_origin(page.physical_page, 8, page_size)
        );
    }

    #[test]
    fn unbacked_keys_are_dropped() {
        let (vsm, clipmap, _key) = driven();
        let page_size = u32::from(clipmap.page_size);
        let far = clipmap.page_key(0, 3, IVec2::new(1000, 1000)).unwrap();
        let pages = build_render_pages(&clipmap, vsm.table(), &[far], 8, page_size);
        assert!(pages.is_empty());
    }
}
