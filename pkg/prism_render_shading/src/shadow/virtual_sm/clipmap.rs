//! Directional-light clipmap levels: level selection, world-space texel
//! snapping and page-grid addressing.
//!
//! A directional (sun) light has no natural centre, so its virtual shadow map
//! is organised as a *clipmap*: a stack of concentric levels centred on the
//! camera, each level covering twice the world extent of the finer one at half
//! the angular resolution.  A surface point is shadowed from the finest level
//! whose texels are still dense enough for its on-screen size, exactly like the
//! clipmap Unreal's VSM builds for directional lights.
//!
//! Two properties make the level stable frame to frame:
//!
//! * **Absolute page addressing.**  A page's identity is `floor(world / page_
//!   world_size)`, a function of the *world* position only, so a static
//!   receiver keeps the same [`ShadowPageKey`] no matter where the camera is —
//!   the residency cache never churns just because the camera moved.
//! * **World-space texel snapping.**  The finite window of pages a level keeps
//!   resident is centred on the camera but snapped to whole-page boundaries
//!   (`snapped_origin`), so panning the camera slides the window in integer
//!   page steps instead of shimmering the texel grid — the same snapping idea
//!   `csm.rs` applies to cascaded shadow maps.
//!
//! Everything here is pure `bevy_math` vector arithmetic with a CPU golden test
//! so it stays a byte-for-byte twin of the GPU clipmap addressing.

use bevy_math::{ops, IVec2, UVec2, Vec2};
use prism_render_architecture::virtual_shadow::{ShadowPageKey, VirtualShadowSettings};

/// Static description of a directional clipmap: how many levels it has, how big
/// a page is, the finest level's world scale and where the world origin maps in
/// the unsigned [`ShadowPageKey`] coordinate space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipmapConfig {
    /// Number of clipmap levels (`>= 1`); level `0` is the finest.
    pub levels: u16,
    /// Edge length of a page in texels (`>= 1`).
    pub page_size: u16,
    /// Number of pages along one edge of a level's resident window (`>= 1`).
    pub pages_per_level_edge: u16,
    /// World-space edge length of one texel at the finest level (`> 0`).
    pub level0_texel_world_size: f32,
    /// View distance at or below which level `0` is selected (`> 0`); each
    /// doubling of distance advances one level.
    pub level0_max_distance: f32,
    /// World page coordinate mapped to key coordinate `0`; a positive bias lets
    /// negative world pages fit the unsigned [`ShadowPageKey`] `x`/`y` fields.
    pub page_coord_bias: i32,
}

/// One resolved clipmap level for a given camera position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipmapLevel {
    /// Clipmap level index (`0` is finest).
    pub level: u16,
    /// World-space edge length of one texel at this level.
    pub texel_world_size: f32,
    /// World-space edge length of one page at this level.
    pub page_world_size: f32,
    /// Bottom-left page of the resident window, in absolute world page coords.
    pub origin_page: IVec2,
    /// World position of the bottom-left corner of `origin_page` (snapped).
    pub snapped_origin: Vec2,
    /// Number of pages along one edge of the resident window.
    pub pages_per_edge: u16,
}

impl ClipmapLevel {
    /// Whether the absolute world `page` falls inside this level's window.
    pub fn contains(&self, page: IVec2) -> bool {
        let edge = i32::from(self.pages_per_edge);
        let local = page - self.origin_page;
        local.x >= 0 && local.y >= 0 && local.x < edge && local.y < edge
    }

    /// Window-local coordinates of an absolute world `page`, or `None` when the
    /// page lies outside the window.
    pub fn local_page(&self, page: IVec2) -> Option<UVec2> {
        if self.contains(page) {
            let local = page - self.origin_page;
            Some(UVec2::new(local.x as u32, local.y as u32))
        } else {
            None
        }
    }
}

impl ClipmapConfig {
    /// Builds a config from the architecture [`VirtualShadowSettings`] contract,
    /// pulling `levels`/`page_size` from it and taking the remaining world-scale
    /// tunables explicitly.  Clamps every field to its valid range.
    pub fn from_settings(
        settings: &VirtualShadowSettings,
        pages_per_level_edge: u16,
        level0_texel_world_size: f32,
        level0_max_distance: f32,
        page_coord_bias: i32,
    ) -> Self {
        Self {
            levels: u16::from(settings.max_clip_levels).max(1),
            page_size: settings.page_size.max(1),
            pages_per_level_edge: pages_per_level_edge.max(1),
            level0_texel_world_size: level0_texel_world_size.max(1.0e-6),
            level0_max_distance: level0_max_distance.max(1.0e-6),
            page_coord_bias,
        }
    }

    /// Number of clipmap levels (always `>= 1`).
    pub fn level_count(&self) -> u16 {
        self.levels.max(1)
    }

    /// World-space edge length of one texel at `level` (doubles each level).
    pub fn texel_world_size(&self, level: u16) -> f32 {
        let level = level.min(self.level_count() - 1);
        self.level0_texel_world_size.max(1.0e-6) * ops::exp2(f32::from(level))
    }

    /// World-space edge length of one page at `level`.
    pub fn page_world_size(&self, level: u16) -> f32 {
        self.texel_world_size(level) * f32::from(self.page_size.max(1))
    }

    /// Selects the finest clipmap level whose texels are still dense enough for
    /// a receiver at `view_distance`: level `0` up to `level0_max_distance`,
    /// then one coarser level per doubling of distance, clamped to the stack.
    pub fn select_level(&self, view_distance: f32) -> u16 {
        let distance = view_distance.max(0.0);
        let reference = self.level0_max_distance.max(1.0e-6);
        let ratio = distance / reference;
        let level = if ratio <= 1.0 {
            0
        } else {
            ops::ceil(ops::log2(ratio)) as i32
        };
        let max_level = i32::from(self.level_count()) - 1;
        level.clamp(0, max_level) as u16
    }

    /// Absolute world page coordinate of `world_xy` at `level`
    /// (`floor(world / page_world_size)`), independent of the camera.
    pub fn world_page_coords(&self, level: u16, world_xy: Vec2) -> IVec2 {
        let pws = self.page_world_size(level);
        (world_xy / pws).floor().as_ivec2()
    }

    /// Resolves the resident window for `level`, snapping its centre to whole
    /// pages around `camera_xy` so the window slides in integer page steps.
    pub fn build_level(&self, level: u16, camera_xy: Vec2) -> ClipmapLevel {
        let level = level.min(self.level_count() - 1);
        let pws = self.page_world_size(level);
        let edge = i32::from(self.pages_per_level_edge.max(1));
        let center_page = (camera_xy / pws).floor().as_ivec2();
        let origin_page = center_page - IVec2::splat(edge / 2);
        let snapped_origin = origin_page.as_vec2() * pws;
        ClipmapLevel {
            level,
            texel_world_size: self.texel_world_size(level),
            page_world_size: pws,
            origin_page,
            snapped_origin,
            pages_per_edge: self.pages_per_level_edge.max(1),
        }
    }

    /// Encodes an absolute world `page` at `level` for `light` into a
    /// [`ShadowPageKey`], biasing by `page_coord_bias` so nearby-origin pages
    /// fit the unsigned key fields.  Returns `None` when the biased coordinate
    /// falls outside the `u16` range the key can represent.
    pub fn page_key(&self, light: u32, level: u16, page: IVec2) -> Option<ShadowPageKey> {
        let bx = page.x.checked_add(self.page_coord_bias)?;
        let by = page.y.checked_add(self.page_coord_bias)?;
        if bx < 0 || by < 0 || bx > i32::from(u16::MAX) || by > i32::from(u16::MAX) {
            return None;
        }
        Some(ShadowPageKey {
            light,
            level,
            x: bx as u16,
            y: by as u16,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ClipmapConfig {
        ClipmapConfig {
            levels: 4,
            page_size: 128,
            pages_per_level_edge: 8,
            level0_texel_world_size: 0.1,
            level0_max_distance: 10.0,
            page_coord_bias: 32_768,
        }
    }

    /// Each level doubles the texel and page world size of the finer one.
    #[test]
    fn levels_double_in_world_size() {
        let c = config();
        assert!((c.texel_world_size(0) - 0.1).abs() < 1.0e-6);
        assert!((c.texel_world_size(1) - 0.2).abs() < 1.0e-6);
        assert!((c.texel_world_size(3) - 0.8).abs() < 1.0e-6);
        // page = texel * page_size.
        assert!((c.page_world_size(0) - 0.1 * 128.0).abs() < 1.0e-4);
    }

    /// Level selection is level 0 up to the reference distance, then one level
    /// per doubling, clamped to the coarsest level.
    #[test]
    fn level_selection_advances_per_doubling() {
        let c = config(); // level0_max_distance = 10, 4 levels.
        assert_eq!(c.select_level(0.0), 0);
        assert_eq!(c.select_level(10.0), 0);
        assert_eq!(c.select_level(15.0), 1); // ratio 1.5 -> ceil(log2)=1
        assert_eq!(c.select_level(25.0), 2); // ratio 2.5 -> ceil(log2 ~1.32)=2
        assert_eq!(c.select_level(1000.0), 3); // clamped to coarsest
    }

    /// A static world point keeps its absolute page coordinate no matter where
    /// the camera is: the cache-stability guarantee.
    #[test]
    fn world_page_coords_are_camera_independent() {
        let c = config();
        let point = Vec2::new(37.0, -12.5);
        let a = c.world_page_coords(1, point);
        let b = c.world_page_coords(1, point);
        assert_eq!(a, b);
        // Directly checks floor(world / page_world_size).
        let pws = c.page_world_size(1);
        assert_eq!(a.x, (37.0f32 / pws).floor() as i32);
        assert_eq!(a.y, (-12.5f32 / pws).floor() as i32);
    }

    /// The resident window snaps to whole pages: a sub-page camera nudge leaves
    /// the window origin unchanged, so nothing shimmers.
    #[test]
    fn window_origin_snaps_to_whole_pages() {
        let c = config();
        let pws = c.page_world_size(0);
        let base = Vec2::new(5.0, 5.0);
        let level_a = c.build_level(0, base);
        // Nudge the camera by a fraction of a page.
        let nudged = base + Vec2::splat(pws * 0.3);
        let level_b = c.build_level(0, nudged);
        assert_eq!(level_a.origin_page, level_b.origin_page);
        // Snapped origin is an exact multiple of the page world size.
        let ratio = level_a.snapped_origin.x / pws;
        assert!((ratio - ops::round(ratio)).abs() < 1.0e-4);
    }

    /// Moving the camera a full page shifts the window by exactly one page.
    #[test]
    fn window_shifts_one_page_per_page_move() {
        let c = config();
        let pws = c.page_world_size(2);
        let level_a = c.build_level(2, Vec2::new(0.0, 0.0));
        let level_b = c.build_level(2, Vec2::new(pws, 0.0));
        assert_eq!(level_b.origin_page - level_a.origin_page, IVec2::new(1, 0));
    }

    /// The window membership and local coordinate helpers agree.
    #[test]
    fn window_membership_and_local_coords() {
        let c = config();
        let level = c.build_level(0, Vec2::new(0.0, 0.0));
        let inside = level.origin_page + IVec2::new(1, 2);
        assert!(level.contains(inside));
        assert_eq!(level.local_page(inside), Some(UVec2::new(1, 2)));
        let outside = level.origin_page - IVec2::new(1, 0);
        assert!(!level.contains(outside));
        assert_eq!(level.local_page(outside), None);
    }

    /// Page-key encoding biases world pages into the unsigned key space and
    /// rejects coordinates that cannot be represented.
    #[test]
    fn page_key_biases_and_bounds_check() {
        let c = config();
        let key = c.page_key(2, 1, IVec2::new(-3, 4)).expect("in range");
        assert_eq!(key.light, 2);
        assert_eq!(key.level, 1);
        assert_eq!(i32::from(key.x), -3 + c.page_coord_bias);
        assert_eq!(i32::from(key.y), 4 + c.page_coord_bias);
        // A page far outside the representable range is rejected.
        assert!(c.page_key(0, 0, IVec2::new(i32::MAX, 0)).is_none());
        assert!(c
            .page_key(0, 0, IVec2::new(-c.page_coord_bias - 1, 0))
            .is_none());
    }
}
