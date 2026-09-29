//! Directional-light shadow clipmap: level and page selection.
//!
//! A directional light cannot use a single shadow map at Nanite quality: the
//! world region it must cover is enormous while the texel density needed at the
//! camera is very high. A clipmap resolves this with a stack of concentric
//! levels centred on the camera. Level 0 is the finest — a small world region
//! at the highest texel density — and each successive level doubles the world
//! extent it covers, halving density, so distant receivers fall back to coarse
//! levels while near receivers get crisp shadows. Every level is the same
//! `resolution x resolution` grid of fixed-texel pages; only the world size a
//! page spans changes per level.
//!
//! This module is the GPU-independent decision layer for that clipmap. It
//! answers three questions deterministically so they can be unit-tested:
//! which level a receiver needs given its required shadow-texel size, where a
//! level's page grid sits in light space (snapped to the page grid so shadows
//! stay stable as the camera creeps), and which [`ShadowPageKey`] a light-space
//! position maps to. It operates in 2D light space: the render layer projects
//! world positions onto the plane perpendicular to the light and feeds the
//! resulting `(x, y)`, keeping all trigonometry out of this layer.

use super::ShadowPageKey;

/// Largest clip level count the fixed-point page math stays exact for.
///
/// Page size scales by `2^level`; capping the exponent keeps `1 << level`
/// inside `u32` and the derived `f32` sizes exact.
pub const MAX_CLIP_LEVELS: u8 = 24;

/// Static description of one directional light's shadow clipmap.
///
/// `resolution` is the page count per side of every level's square grid;
/// `page_texel_dim` is the texel count per side of a single page; and
/// `level0_page_size` is the world-space edge length a page spans at the finest
/// level. Level `L` scales the page (and thus texel) size by `2^L`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipmapConfig {
    /// Light this clipmap belongs to; stamped into every [`ShadowPageKey`].
    pub light: u32,
    /// Number of clip levels, finest (0) to coarsest.
    pub level_count: u8,
    /// Pages per side of each level's square grid.
    pub resolution: u16,
    /// Texels per side of a single page.
    pub page_texel_dim: u16,
    /// World-space edge length of a page at the finest level (level 0).
    pub level0_page_size: f32,
}

impl ClipmapConfig {
    /// Clamps a level request to the configured, representable range.
    #[must_use]
    fn clamp_level(&self, level: u8) -> u8 {
        let last = self.level_count.saturating_sub(1);
        level.min(last).min(MAX_CLIP_LEVELS)
    }

    /// World-space edge length a page spans at `level`.
    ///
    /// Doubles each level; `level` is clamped to the representable range so the
    /// `1 << level` shift never overflows.
    #[must_use]
    pub fn page_size(&self, level: u8) -> f32 {
        let level = self.clamp_level(level).min(MAX_CLIP_LEVELS);
        let scale = (1u32 << level) as f32;
        self.level0_page_size * scale
    }

    /// World-space edge length a single shadow texel spans at `level`.
    #[must_use]
    pub fn texel_size(&self, level: u8) -> f32 {
        let dim = f32::from(self.page_texel_dim.max(1));
        self.page_size(level) / dim
    }

    /// World-space edge length the whole level grid spans.
    #[must_use]
    pub fn level_span(&self, level: u8) -> f32 {
        self.page_size(level) * f32::from(self.resolution)
    }

    /// Selects the finest level whose texel is coarse enough for a receiver
    /// that needs a world-space shadow texel of at least `required_texel_size`.
    ///
    /// Texel size doubles with level, so the search walks from the finest level
    /// up and returns the first level whose texel meets the requirement. A
    /// requirement finer than level 0 can offer clamps to level 0 (the finest
    /// available); a requirement coarser than the top level clamps to the
    /// coarsest. An empty clipmap resolves to level 0.
    #[must_use]
    pub fn select_level(&self, required_texel_size: f32) -> u8 {
        if self.level_count == 0 {
            return 0;
        }
        let required = required_texel_size.max(0.0);
        let mut level = 0u8;
        while level < self.level_count.saturating_sub(1) {
            if self.texel_size(level) >= required {
                return level;
            }
            level += 1;
        }
        self.clamp_level(level)
    }

    /// Light-space minimum corner of `level`'s page grid for a camera at
    /// `camera`.
    ///
    /// The grid is centred on the camera and snapped to the level's page size
    /// so sub-page camera motion never shifts which world region a page covers,
    /// which is what keeps clipmap shadows from crawling. Snapping floors the
    /// camera onto the page lattice, then backs off by half the grid.
    #[must_use]
    pub fn level_origin(&self, camera: [f32; 2], level: u8) -> [f32; 2] {
        let ps = self.page_size(level);
        let half = f32::from(self.resolution / 2) * ps;
        [
            snap_to_grid(camera[0], ps) - half,
            snap_to_grid(camera[1], ps) - half,
        ]
    }

    /// Maps a light-space position to its page within `level`'s grid.
    ///
    /// Returns `None` when the position falls outside the level's coverage or
    /// when the clipmap is empty, so a receiver beyond the shadowed region marks
    /// no page rather than aliasing onto an edge page.
    #[must_use]
    pub fn page_of(
        &self,
        camera: [f32; 2],
        position: [f32; 2],
        level: u8,
    ) -> Option<ShadowPageKey> {
        if self.level_count == 0 || self.resolution == 0 {
            return None;
        }
        let level = self.clamp_level(level);
        let ps = self.page_size(level);
        let origin = self.level_origin(camera, level);
        let fx = ((position[0] - origin[0]) / ps).floor();
        let fy = ((position[1] - origin[1]) / ps).floor();
        let last = f32::from(self.resolution);
        if fx < 0.0 || fy < 0.0 || fx >= last || fy >= last {
            return None;
        }
        Some(ShadowPageKey {
            light: self.light,
            level: u16::from(level),
            x: fx as u16,
            y: fy as u16,
        })
    }

    /// Selects both the level and the page a receiver needs in one call.
    ///
    /// Combines [`select_level`](Self::select_level) with
    /// [`page_of`](Self::page_of): the receiver's `required_texel_size` picks the
    /// level, then its light-space `position` picks the page within that level.
    #[must_use]
    pub fn select_page(
        &self,
        camera: [f32; 2],
        position: [f32; 2],
        required_texel_size: f32,
    ) -> Option<ShadowPageKey> {
        let level = self.select_level(required_texel_size);
        self.page_of(camera, position, level)
    }
}

/// Floors `coord` onto the lattice of step `step`, returning the lattice point.
///
/// A non-positive step degenerates to the coordinate itself so a misconfigured
/// clipmap cannot divide by zero or invert the lattice.
#[must_use]
fn snap_to_grid(coord: f32, step: f32) -> f32 {
    if step <= 0.0 {
        return coord;
    }
    (coord / step).floor() * step
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ClipmapConfig {
        ClipmapConfig {
            light: 3,
            level_count: 4,
            resolution: 8,
            page_texel_dim: 128,
            level0_page_size: 4.0,
        }
    }

    #[test]
    fn page_and_texel_sizes_double_each_level() {
        let c = config();
        assert_eq!(c.page_size(0), 4.0);
        assert_eq!(c.page_size(1), 8.0);
        assert_eq!(c.page_size(2), 16.0);
        // texel = page_size / page_texel_dim.
        assert_eq!(c.texel_size(0), 4.0 / 128.0);
        assert_eq!(c.level_span(0), 4.0 * 8.0);
    }

    #[test]
    fn level_clamps_beyond_the_stack() {
        let c = config();
        // Level 9 clamps to the coarsest configured level (3).
        assert_eq!(c.page_size(9), c.page_size(3));
    }

    #[test]
    fn select_level_picks_finest_that_meets_requirement() {
        let c = config();
        // texel(0)=0.03125, texel(1)=0.0625, texel(2)=0.125, texel(3)=0.25.
        // A tiny requirement can only get the finest level.
        assert_eq!(c.select_level(0.001), 0);
        // Just above texel(0) needs level 1.
        assert_eq!(c.select_level(0.05), 1);
        // Above texel(2) needs level 3.
        assert_eq!(c.select_level(0.2), 3);
        // Coarser than everything clamps to the coarsest.
        assert_eq!(c.select_level(100.0), 3);
    }

    #[test]
    fn origin_is_stable_under_subpage_motion() {
        let c = config();
        // Two cameras within the same level-0 page snap to the same origin.
        let a = c.level_origin([1.0, 1.0], 0);
        let b = c.level_origin([2.9, 2.9], 0);
        assert_eq!(a, b);
        // Crossing a page boundary shifts the origin by exactly one page.
        let d = c.level_origin([4.1, 1.0], 0);
        assert_eq!(d[0] - a[0], c.page_size(0));
    }

    #[test]
    fn page_of_maps_center_and_rejects_outside() {
        let c = config();
        let camera = [0.0, 0.0];
        // A position at the camera lands inside the grid.
        let key = c.page_of(camera, [0.0, 0.0], 0).expect("in range");
        assert_eq!(key.light, 3);
        assert_eq!(key.level, 0);
        // Far outside the level-0 coverage returns None.
        assert!(c.page_of(camera, [10_000.0, 0.0], 0).is_none());
    }

    #[test]
    fn adjacent_positions_map_to_adjacent_pages() {
        let c = config();
        let camera = [0.0, 0.0];
        let ps = c.page_size(0);
        let origin = c.level_origin(camera, 0);
        // Sample the centers of two horizontally adjacent pages.
        let p0 = c.page_of(camera, [origin[0] + ps * 0.5, origin[1] + ps * 0.5], 0);
        let p1 = c.page_of(camera, [origin[0] + ps * 1.5, origin[1] + ps * 0.5], 0);
        let p0 = p0.expect("page 0");
        let p1 = p1.expect("page 1");
        assert_eq!(p1.x, p0.x + 1);
        assert_eq!(p1.y, p0.y);
    }

    #[test]
    fn empty_clipmap_yields_no_page() {
        let mut c = config();
        c.level_count = 0;
        assert!(c.page_of([0.0, 0.0], [0.0, 0.0], 0).is_none());
        assert_eq!(c.select_level(1.0), 0);
    }

    #[test]
    fn select_page_combines_level_and_page() {
        let c = config();
        let camera = [0.0, 0.0];
        // A coarse requirement drives a coarse level; the origin position stays
        // in range because coarse levels cover more world.
        let key = c.select_page(camera, [0.0, 0.0], 100.0).expect("in range");
        assert_eq!(key.level, 3);
    }
}
