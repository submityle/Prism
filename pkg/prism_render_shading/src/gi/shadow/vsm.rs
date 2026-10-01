//! Virtual Shadow Maps — clipmap addressing & level selection (CPU golden).
//!
//! Virtual Shadow Maps (VSM, UE5-style) replace a monolithic shadow atlas with
//! a *sparse* virtual address space that is tiled into fixed-size **pages**.
//! For a directional light the virtual space is organised as a **clipmap**: a
//! stack of concentric square levels centred on the camera, where each level
//! covers four times the world-space area of the one below it (its per-texel
//! world footprint doubles every level).  A shading point selects the coarsest
//! level whose texel is still finer than the point's on-screen footprint, so
//! near geometry draws from high-resolution levels and distant geometry from
//! cheap coarse ones — all out of one unified virtual resolution.
//!
//! This module is the backend-neutral reference for the *addressing* layer:
//!
//! * [`VsmClipmap`] holds the immutable clipmap geometry (texel size, level
//!   count, page size, pages-per-side) and derives per-level world extents,
//!   texel footprints and snapped level origins.
//! * [`VsmClipmap::select_level_by_footprint`] and
//!   [`VsmClipmap::select_level_by_distance`] implement the two standard level
//!   heuristics (projected-footprint and radial-distance).
//! * [`VsmClipmap::world_to_virtual_page`] / [`VsmClipmap::virtual_page_min`]
//!   round-trip a light-plane position to the virtual page that stores it.
//! * [`VsmPageTable`] maps virtual pages to physical pool slots and tracks the
//!   requested / resident marking bits used by the residency manager.
//!
//! It is strictly the addressing & level-selection math; GPU residency,
//! eviction policy and allocation live in the real-time runtime, not here.
//!
//! # Conventions
//! * The clipmap plane is the light's view XY plane; positions on it are
//!   [`Vec2`] in world units, so the GPU twin shares the same metric.
//! * Level `0` is the finest; `texel_world_size(level) = base * 2^level`, i.e.
//!   each coarser level doubles the per-texel footprint (quadruples the area).
//! * A level is a square grid of `pages_per_side` pages, each `page_size`
//!   texels on a side, giving `pages_per_side * page_size` texels per side.
//! * Levels are *snapped* to a whole-page world grid so camera motion shifts
//!   the clipmap in page-sized steps (no sub-page shimmering), matching the
//!   residency scroll the GPU performs.
//! * Every helper is a deterministic pure function; the only allocation is the
//!   page-table buffer owned by [`VsmPageTable`].  All queries defend against
//!   degenerate inputs (zero sizes, non-finite coordinates) and never return
//!   `NaN` or out-of-range indices.

use alloc::vec::Vec;
use bevy_math::{ops, IVec2, Vec2};

/// Sentinel physical-page id meaning "no physical page is mapped".
pub const VSM_PAGE_UNMAPPED: u32 = u32::MAX;

/// Bit set on a page-table entry that has been requested this frame.
const PAGE_FLAG_REQUESTED: u32 = 0b01;
/// Bit set on a page-table entry that currently has a resident physical page.
const PAGE_FLAG_RESIDENT: u32 = 0b10;

/// Immutable geometry of a directional-light VSM clipmap.
///
/// All derived quantities (extents, footprints, origins) are computed on demand
/// from these fields so the struct stays `Copy` and trivially shareable with the
/// GPU twin's uniform block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VsmClipmap {
    /// World-space size of one texel at level `0` (the finest level).
    base_texel_size: f32,
    /// Number of clipmap levels, always `>= 1`.
    level_count: u32,
    /// Texels per page along one side, always `>= 1`.
    page_size: u32,
    /// Pages per level along one side, always `>= 1`.
    pages_per_side: u32,
}

impl VsmClipmap {
    /// Builds a clipmap, clamping every parameter into a valid range.
    ///
    /// `base_texel_size` is forced to a tiny positive floor (so inversion never
    /// divides by zero and a non-finite input degrades to the floor);
    /// `level_count`, `page_size` and `pages_per_side` are each clamped to at
    /// least `1`.
    #[inline]
    pub fn new(
        base_texel_size: f32,
        level_count: u32,
        page_size: u32,
        pages_per_side: u32,
    ) -> Self {
        let base_texel_size = if base_texel_size.is_finite() {
            base_texel_size.max(1.0e-6)
        } else {
            1.0e-6
        };
        Self {
            base_texel_size,
            level_count: level_count.max(1),
            page_size: page_size.max(1),
            pages_per_side: pages_per_side.max(1),
        }
    }

    /// World-space texel size at level `0`.
    #[inline]
    pub fn base_texel_size(&self) -> f32 {
        self.base_texel_size
    }

    /// Number of clipmap levels (`>= 1`).
    #[inline]
    pub fn level_count(&self) -> u32 {
        self.level_count
    }

    /// Index of the coarsest level (`level_count - 1`).
    #[inline]
    pub fn max_level(&self) -> u32 {
        self.level_count - 1
    }

    /// Texels per page side (`>= 1`).
    #[inline]
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    /// Pages per level side (`>= 1`).
    #[inline]
    pub fn pages_per_side(&self) -> u32 {
        self.pages_per_side
    }

    /// Texels per level side (`pages_per_side * page_size`).
    #[inline]
    pub fn texels_per_side(&self) -> u32 {
        self.pages_per_side.saturating_mul(self.page_size)
    }

    /// Clamps an arbitrary level into `[0, max_level]`.
    #[inline]
    pub fn clamp_level(&self, level: u32) -> u32 {
        level.min(self.max_level())
    }

    /// World-space size of one texel at `level`.
    ///
    /// Equals `base_texel_size * 2^level`; the level is clamped first so the
    /// result is always finite and positive.
    #[inline]
    pub fn texel_world_size(&self, level: u32) -> f32 {
        let level = self.clamp_level(level);
        self.base_texel_size * ops::exp2(level as f32)
    }

    /// World-space side length covered by one whole page at `level`.
    #[inline]
    pub fn page_world_size(&self, level: u32) -> f32 {
        self.texel_world_size(level) * self.page_size as f32
    }

    /// Full world-space side length covered by `level`.
    #[inline]
    pub fn level_world_extent(&self, level: u32) -> f32 {
        self.texel_world_size(level) * self.texels_per_side() as f32
    }

    /// Snaps a desired clipmap centre to the whole-page grid of `level`.
    ///
    /// Snapping to the page granularity (rather than the texel) keeps the
    /// per-page virtual addressing stable as the camera moves: the level scrolls
    /// in page-sized increments, exactly matching the residency pool's scroll.
    /// Non-finite components fall back to `0.0`.
    #[inline]
    pub fn snapped_center(&self, level: u32, center: Vec2) -> Vec2 {
        let page = self.page_world_size(level);
        let snap = |v: f32| -> f32 {
            if v.is_finite() {
                (v / page).floor() * page
            } else {
                0.0
            }
        };
        Vec2::new(snap(center.x), snap(center.y))
    }

    /// World-space minimum (lower-left) corner of `level` for a snapped centre.
    ///
    /// `center` is snapped with [`snapped_center`](Self::snapped_center) and the
    /// level is positioned so the snapped centre sits at its middle.
    #[inline]
    pub fn level_origin(&self, level: u32, center: Vec2) -> Vec2 {
        let snapped = self.snapped_center(level, center);
        let half = self.level_world_extent(level) * 0.5;
        snapped - Vec2::splat(half)
    }

    /// Selects the clipmap level from a shading point's world-space footprint.
    ///
    /// `footprint_world` is the world-space size covered by one screen pixel at
    /// the shading point.  The chosen level is the coarsest whose texel is still
    /// no larger than that footprint — i.e. the finest level that is not wasteful
    /// — computed as `ceil(log2(footprint / base))` and clamped to
    /// `[0, max_level]`.  A non-positive or non-finite footprint selects the
    /// finest level `0`.
    #[inline]
    pub fn select_level_by_footprint(&self, footprint_world: f32) -> u32 {
        if !(footprint_world > 0.0) || !footprint_world.is_finite() {
            return 0;
        }
        let ratio = footprint_world / self.base_texel_size;
        if ratio <= 1.0 {
            return 0;
        }
        let level = ops::log2(ratio).ceil();
        if !level.is_finite() || level <= 0.0 {
            0
        } else {
            self.clamp_level(level as u32)
        }
    }

    /// Selects the clipmap level that radially contains a point at `distance`.
    ///
    /// Picks the smallest level whose *half extent* reaches `distance` from the
    /// clipmap centre, i.e. the first level that geometrically covers the point.
    /// A non-positive or non-finite distance selects level `0`.
    #[inline]
    pub fn select_level_by_distance(&self, distance: f32) -> u32 {
        if !(distance > 0.0) || !distance.is_finite() {
            return 0;
        }
        // half_extent(level) = 0.5 * texels_per_side * base * 2^level >= distance
        //   => 2^level >= 2*distance / (texels_per_side * base)
        let denom = self.texels_per_side() as f32 * self.base_texel_size;
        let ratio = (2.0 * distance) / denom;
        if ratio <= 1.0 {
            return 0;
        }
        let level = ops::log2(ratio).ceil();
        if !level.is_finite() || level <= 0.0 {
            0
        } else {
            self.clamp_level(level as u32)
        }
    }

    /// Integer virtual page coordinate storing a light-plane position.
    ///
    /// The position is expressed relative to the level's snapped origin and
    /// floored to a page cell.  The returned coordinate is **not** clamped to the
    /// level grid: callers test it against [`page_in_bounds`](Self::page_in_bounds)
    /// to detect points outside the level's coverage.  Non-finite components map
    /// to `0`.
    #[inline]
    pub fn world_to_virtual_page(&self, level: u32, center: Vec2, world: Vec2) -> IVec2 {
        let origin = self.level_origin(level, center);
        let page = self.page_world_size(level);
        let local = world - origin;
        let coord = |v: f32| -> i32 {
            if v.is_finite() {
                (v / page).floor() as i32
            } else {
                0
            }
        };
        IVec2::new(coord(local.x), coord(local.y))
    }

    /// World-space minimum corner of a virtual page.
    #[inline]
    pub fn virtual_page_min(&self, level: u32, center: Vec2, page: IVec2) -> Vec2 {
        let origin = self.level_origin(level, center);
        let size = self.page_world_size(level);
        origin + Vec2::new(page.x as f32 * size, page.y as f32 * size)
    }

    /// Returns whether a virtual page coordinate lies inside the level grid.
    #[inline]
    pub fn page_in_bounds(&self, page: IVec2) -> bool {
        let n = self.pages_per_side as i32;
        page.x >= 0 && page.y >= 0 && page.x < n && page.y < n
    }

    /// Linear page index within one level (row-major), or `None` if out of
    /// bounds.
    #[inline]
    pub fn page_linear_index(&self, page: IVec2) -> Option<u32> {
        if self.page_in_bounds(page) {
            Some(page.y as u32 * self.pages_per_side + page.x as u32)
        } else {
            None
        }
    }

    /// Number of pages in one level (`pages_per_side^2`).
    #[inline]
    pub fn pages_per_level(&self) -> u32 {
        self.pages_per_side.saturating_mul(self.pages_per_side)
    }
}

/// Virtual→physical page map plus per-page request/residency marking bits.
///
/// Stores one entry per `(level, page)` across the whole clipmap, laid out as
/// `level_count` contiguous blocks of `pages_per_level` row-major entries.  The
/// residency manager marks pages [`request`](Self::request)ed during traversal,
/// then [`map`](Self::map)s the ones it allocates to physical pool slots.
#[derive(Clone, Debug, PartialEq)]
pub struct VsmPageTable {
    level_count: u32,
    pages_per_level: u32,
    pages_per_side: u32,
    physical: Vec<u32>,
    flags: Vec<u32>,
}

impl VsmPageTable {
    /// Allocates a fully-unmapped page table sized for `clipmap`.
    #[inline]
    pub fn new(clipmap: &VsmClipmap) -> Self {
        let per_level = clipmap.pages_per_level();
        let total = (clipmap.level_count() as usize) * (per_level as usize);
        let mut physical = Vec::new();
        physical.resize(total, VSM_PAGE_UNMAPPED);
        let mut flags = Vec::new();
        flags.resize(total, 0);
        Self {
            level_count: clipmap.level_count(),
            pages_per_level: per_level,
            pages_per_side: clipmap.pages_per_side(),
            physical,
            flags,
        }
    }

    /// Flat buffer index for `(level, page)`, or `None` if either is out of
    /// range.
    #[inline]
    fn index(&self, level: u32, page: IVec2) -> Option<usize> {
        if level >= self.level_count {
            return None;
        }
        let n = self.pages_per_side as i32;
        if page.x < 0 || page.y < 0 || page.x >= n || page.y >= n {
            return None;
        }
        let within = page.y as u32 * self.pages_per_side + page.x as u32;
        Some(level as usize * self.pages_per_level as usize + within as usize)
    }

    /// Total number of entries (`level_count * pages_per_level`).
    #[inline]
    pub fn len(&self) -> usize {
        self.physical.len()
    }

    /// Returns whether the table holds no entries (never true after `new`, which
    /// clamps all dimensions to `>= 1`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.physical.is_empty()
    }

    /// Marks a page as requested this frame; returns `false` if out of range.
    #[inline]
    pub fn request(&mut self, level: u32, page: IVec2) -> bool {
        match self.index(level, page) {
            Some(i) => {
                self.flags[i] |= PAGE_FLAG_REQUESTED;
                true
            }
            None => false,
        }
    }

    /// Returns whether a page is marked requested.
    #[inline]
    pub fn is_requested(&self, level: u32, page: IVec2) -> bool {
        self.index(level, page)
            .is_some_and(|i| self.flags[i] & PAGE_FLAG_REQUESTED != 0)
    }

    /// Binds a physical pool slot to a page and marks it resident.
    ///
    /// Returns `false` if `(level, page)` is out of range.  Mapping
    /// [`VSM_PAGE_UNMAPPED`] clears the resident bit (an explicit un-map).
    #[inline]
    pub fn map(&mut self, level: u32, page: IVec2, physical: u32) -> bool {
        match self.index(level, page) {
            Some(i) => {
                self.physical[i] = physical;
                if physical == VSM_PAGE_UNMAPPED {
                    self.flags[i] &= !PAGE_FLAG_RESIDENT;
                } else {
                    self.flags[i] |= PAGE_FLAG_RESIDENT;
                }
                true
            }
            None => false,
        }
    }

    /// Physical pool slot bound to a page, or `None` when unmapped/out of range.
    #[inline]
    pub fn physical_of(&self, level: u32, page: IVec2) -> Option<u32> {
        self.index(level, page).and_then(|i| {
            let p = self.physical[i];
            if p == VSM_PAGE_UNMAPPED {
                None
            } else {
                Some(p)
            }
        })
    }

    /// Returns whether a page currently has a resident physical slot.
    #[inline]
    pub fn is_resident(&self, level: u32, page: IVec2) -> bool {
        self.index(level, page)
            .is_some_and(|i| self.flags[i] & PAGE_FLAG_RESIDENT != 0)
    }

    /// Clears all request bits (called at the start of a residency pass).
    #[inline]
    pub fn clear_requests(&mut self) {
        for f in &mut self.flags {
            *f &= !PAGE_FLAG_REQUESTED;
        }
    }

    /// Counts entries currently marked requested.
    #[inline]
    pub fn requested_count(&self) -> usize {
        self.flags
            .iter()
            .filter(|f| *f & PAGE_FLAG_REQUESTED != 0)
            .count()
    }

    /// Counts entries currently marked resident.
    #[inline]
    pub fn resident_count(&self) -> usize {
        self.flags
            .iter()
            .filter(|f| *f & PAGE_FLAG_RESIDENT != 0)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip() -> VsmClipmap {
        // base texel 0.1 world units, 4 levels, 8-texel pages, 4 pages/side
        VsmClipmap::new(0.1, 4, 8, 4)
    }

    #[test]
    fn constructor_clamps_degenerate_inputs() {
        let c = VsmClipmap::new(f32::NAN, 0, 0, 0);
        assert!(c.base_texel_size() > 0.0 && c.base_texel_size().is_finite());
        assert_eq!(c.level_count(), 1);
        assert_eq!(c.page_size(), 1);
        assert_eq!(c.pages_per_side(), 1);
    }

    #[test]
    fn texel_size_doubles_each_level() {
        let c = clip();
        for l in 0..c.level_count() {
            let expected = 0.1 * ops::exp2(l as f32);
            assert!((c.texel_world_size(l) - expected).abs() < 1e-6);
        }
        // Beyond max level clamps to the coarsest.
        assert_eq!(c.texel_world_size(99), c.texel_world_size(c.max_level()));
    }

    #[test]
    fn level_extent_quadruples_area_each_level() {
        let c = clip();
        let e0 = c.level_world_extent(0);
        let e1 = c.level_world_extent(1);
        assert!((e1 - 2.0 * e0).abs() < 1e-5);
    }

    #[test]
    fn footprint_level_selection_is_monotone_and_clamped() {
        let c = clip();
        // Footprint smaller than base texel -> finest level.
        assert_eq!(c.select_level_by_footprint(0.05), 0);
        assert_eq!(c.select_level_by_footprint(0.1), 0);
        // Footprint equal to level-1 texel (0.2) -> level 1.
        assert_eq!(c.select_level_by_footprint(0.2), 1);
        assert_eq!(c.select_level_by_footprint(0.4), 2);
        // Monotonic non-decreasing in footprint.
        let mut prev = 0;
        let mut f = 0.01;
        while f < 100.0 {
            let l = c.select_level_by_footprint(f);
            assert!(l >= prev);
            prev = l;
            f *= 1.3;
        }
        // Huge footprint clamps to max level.
        assert_eq!(c.select_level_by_footprint(1.0e9), c.max_level());
    }

    #[test]
    fn degenerate_footprint_selects_finest() {
        let c = clip();
        assert_eq!(c.select_level_by_footprint(0.0), 0);
        assert_eq!(c.select_level_by_footprint(-1.0), 0);
        assert_eq!(c.select_level_by_footprint(f32::NAN), 0);
        assert_eq!(c.select_level_by_footprint(f32::INFINITY), 0);
    }

    #[test]
    fn distance_level_selection_covers_point() {
        let c = clip();
        // A point within the finest half-extent stays at level 0.
        let half0 = c.level_world_extent(0) * 0.5;
        assert_eq!(c.select_level_by_distance(half0 * 0.5), 0);
        // A point just beyond level 0 picks a coarser level that reaches it.
        let l = c.select_level_by_distance(half0 * 1.5);
        assert!(l >= 1);
        assert!(c.level_world_extent(l) * 0.5 >= half0 * 1.5 - 1e-4);
        // Very far clamps to max level.
        assert_eq!(c.select_level_by_distance(1.0e9), c.max_level());
    }

    #[test]
    fn snapping_quantises_to_page_grid() {
        let c = clip();
        let level = 1;
        let page = c.page_world_size(level);
        let s = c.snapped_center(level, Vec2::new(page * 2.3, -page * 0.4));
        // Snapped values are integer multiples of the page size.
        assert!((s.x / page - (s.x / page).round()).abs() < 1e-4);
        assert!((s.y / page - (s.y / page).round()).abs() < 1e-4);
        // Idempotent.
        assert_eq!(c.snapped_center(level, s), s);
    }

    #[test]
    fn world_page_roundtrip_recovers_cell() {
        let c = clip();
        let level = 2;
        let center = Vec2::new(3.0, -2.0);
        // Pick a world point near the level centre and recover its page, then
        // confirm the page's min corner is within one page of the point.
        let world = center + Vec2::new(0.37, -0.21);
        let page = c.world_to_virtual_page(level, center, world);
        assert!(c.page_in_bounds(page));
        let min = c.virtual_page_min(level, center, page);
        let size = c.page_world_size(level);
        assert!(world.x >= min.x - 1e-4 && world.x <= min.x + size + 1e-4);
        assert!(world.y >= min.y - 1e-4 && world.y <= min.y + size + 1e-4);
    }

    #[test]
    fn page_linear_index_bounds() {
        let c = clip();
        assert_eq!(c.page_linear_index(IVec2::new(0, 0)), Some(0));
        let n = c.pages_per_side() as i32;
        assert_eq!(
            c.page_linear_index(IVec2::new(n - 1, n - 1)),
            Some(c.pages_per_level() - 1)
        );
        assert_eq!(c.page_linear_index(IVec2::new(-1, 0)), None);
        assert_eq!(c.page_linear_index(IVec2::new(n, 0)), None);
    }

    #[test]
    fn page_table_request_map_residency() {
        let c = clip();
        let mut table = VsmPageTable::new(&c);
        assert_eq!(table.len(), (c.level_count() * c.pages_per_level()) as usize);
        assert!(!table.is_empty());

        let p = IVec2::new(1, 2);
        assert!(table.request(0, p));
        assert!(table.is_requested(0, p));
        assert_eq!(table.requested_count(), 1);
        assert!(!table.is_resident(0, p));

        assert!(table.map(0, p, 42));
        assert_eq!(table.physical_of(0, p), Some(42));
        assert!(table.is_resident(0, p));
        assert_eq!(table.resident_count(), 1);

        // Un-map clears residency.
        assert!(table.map(0, p, VSM_PAGE_UNMAPPED));
        assert_eq!(table.physical_of(0, p), None);
        assert!(!table.is_resident(0, p));

        table.clear_requests();
        assert_eq!(table.requested_count(), 0);
    }

    #[test]
    fn page_table_rejects_out_of_range() {
        let c = clip();
        let mut table = VsmPageTable::new(&c);
        let n = c.pages_per_side() as i32;
        assert!(!table.request(c.level_count(), IVec2::ZERO));
        assert!(!table.request(0, IVec2::new(-1, 0)));
        assert!(!table.map(0, IVec2::new(n, 0), 1));
        assert_eq!(table.physical_of(0, IVec2::new(0, n)), None);
    }

    #[test]
    fn distinct_pages_map_to_distinct_indices() {
        let c = clip();
        let table = VsmPageTable::new(&c);
        // Spot-check that the index mapping is injective across a level block.
        let n = c.pages_per_side() as i32;
        let mut seen = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let idx = table.index(1, IVec2::new(x, y)).unwrap();
                assert!(!seen.contains(&idx));
                seen.push(idx);
            }
        }
        assert_eq!(seen.len(), c.pages_per_level() as usize);
    }
}
