//! Decode a `GPU` sampler-feedback readback into per-page streaming demand.
//!
//! The residency table and scheduler consume [`PageDemand`] records, but those
//! are an abstract per-page request; a real pipeline produces them from a
//! compact buffer the `GPU` writes while shading. The standard sampler-feedback
//! layout is a dense *min-mip* grid: one byte per base-level page cell of a
//! streamable texture holding the finest absolute mip any shading sample wanted
//! for that cell this frame, or [`NOT_REQUESTED`] when nothing sampled it. This
//! module turns that grid into deduplicated [`PageDemand`]s addressed at the mip
//! each cell actually wants.
//!
//! The decode is pure integer work and `no_std`-friendly: a desired mip `d`
//! coarser than the base level maps a base cell `(x, y)` to the coarser mip's
//! page `(x >> (d - base_mip), y >> (d - base_mip))`, so several fine cells can
//! collapse onto one coarse page. Because the desired mip is part of the page
//! key, fine cells that collapse onto one coarse page produce identical demands,
//! so a [`BTreeMap`] keyed on [`TexturePageKey`] deduplicates them and yields the
//! list in ascending key order regardless of grid traversal.

use super::feedback::PageDemand;
use super::{TexturePageKey, TextureSemantic};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// Sentinel min-mip value marking a base-page cell no view sampled this frame.
pub const NOT_REQUESTED: u8 = 0xFF;

/// Immutable description of one streamable texture's feedback grid.
///
/// The grid is `pages_x * pages_y` bytes in row-major order, one entry per
/// base-level page cell. `base_mip` is the finest streamable mip (the grid's
/// own level) and `mip_count` bounds how coarse a request may be, so a decoded
/// page always addresses a mip in `base_mip..base_mip + mip_count`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FeedbackTextureDesc {
    /// Source texture identifier shared with [`TexturePageKey::texture`].
    pub texture: u32,
    /// Array layer or cube face this grid describes.
    pub layer: u16,
    /// Perceptual channel, forwarded to each emitted [`PageDemand`].
    pub semantic: TextureSemantic,
    /// Finest streamable mip level; the grid is sampled at this level.
    pub base_mip: u8,
    /// Number of streamable mip levels at and coarser than `base_mip`.
    pub mip_count: u8,
    /// Base-level page-grid width in page cells.
    pub pages_x: u16,
    /// Base-level page-grid height in page cells.
    pub pages_y: u16,
    /// Physical byte cost of one resident page tile.
    pub page_byte_cost: u64,
    /// Fixed-point screen importance applied to every page this texture emits.
    pub screen_importance: u16,
}

impl FeedbackTextureDesc {
    /// Coarsest mip a request may address (inclusive).
    #[must_use]
    pub const fn max_mip(&self) -> u8 {
        self.base_mip
            .saturating_add(self.mip_count.saturating_sub(1))
    }

    /// Number of bytes a correctly-sized grid for this texture occupies.
    #[must_use]
    pub const fn grid_len(&self) -> usize {
        self.pages_x as usize * self.pages_y as usize
    }
}

/// Decodes a min-mip feedback grid into deduplicated per-page demand.
///
/// `grid` is `desc.`[`grid_len`](FeedbackTextureDesc::grid_len) bytes, row-major
/// at the base mip; cells equal to [`NOT_REQUESTED`] are skipped. `resident_mip`
/// reports the finest mip currently backed for a page key (or `None`), filling
/// [`PageDemand::resident_mip`] so the scheduler can score the shortfall. `frame`
/// stamps every demand for `LRU` tie-breaks.
///
/// A desired mip is clamped into `desc.base_mip..=desc.max_mip()`. Returns demand
/// in ascending [`TexturePageKey`] order; a grid of the wrong length yields no
/// demand rather than panicking, mirroring the scheduler's skip-not-abort policy.
#[must_use]
pub fn decode_feedback(
    desc: &FeedbackTextureDesc,
    grid: &[u8],
    mut resident_mip: impl FnMut(TexturePageKey) -> Option<u8>,
    frame: u64,
) -> Vec<PageDemand> {
    if grid.len() != desc.grid_len() || desc.mip_count == 0 {
        return Vec::new();
    }

    let base = desc.base_mip;
    let max_mip = desc.max_mip();
    let mut merged: BTreeMap<TexturePageKey, PageDemand> = BTreeMap::new();

    for y in 0..desc.pages_y {
        for x in 0..desc.pages_x {
            let cell = grid[y as usize * desc.pages_x as usize + x as usize];
            if cell == NOT_REQUESTED {
                continue;
            }
            // Clamp the requested absolute mip into the streamable range.
            let desired = cell.clamp(base, max_mip);
            let shift = desired - base;
            let key = TexturePageKey {
                texture: desc.texture,
                mip: desired,
                layer: desc.layer,
                x: x >> shift,
                y: y >> shift,
            };
            let demand = PageDemand {
                key,
                semantic: desc.semantic,
                desired_mip: desired,
                resident_mip: resident_mip(key),
                screen_importance: desc.screen_importance,
                byte_cost: desc.page_byte_cost,
                frame,
            };
            // Several fine cells can collapse onto one coarse page; because the
            // desired mip is part of the key, every colliding demand is
            // identical, so a first-writer-wins insert deduplicates exactly.
            merged.entry(key).or_insert(demand);
        }
    }

    merged.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc() -> FeedbackTextureDesc {
        FeedbackTextureDesc {
            texture: 7,
            layer: 0,
            semantic: TextureSemantic::Color,
            base_mip: 0,
            mip_count: 4,
            pages_x: 4,
            pages_y: 4,
            page_byte_cost: 65_536,
            screen_importance: 500,
        }
    }

    /// Row-major index into the 4-wide feedback grid for fine cell `(x, y)`.
    fn cell(x: usize, y: usize) -> usize {
        y * 4 + x
    }

    #[test]
    fn wrong_length_grid_yields_no_demand() {
        let d = desc();
        assert!(decode_feedback(&d, &[0u8; 3], |_| None, 1).is_empty());
    }

    #[test]
    fn not_requested_cells_are_skipped() {
        let d = desc();
        let grid = [NOT_REQUESTED; 16];
        assert!(decode_feedback(&d, &grid, |_| None, 1).is_empty());
    }

    #[test]
    fn single_finest_request_maps_to_its_cell() {
        let d = desc();
        let mut grid = [NOT_REQUESTED; 16];
        // Cell (2, 1) wants mip 0.
        grid[cell(2, 1)] = 0;
        let demands = decode_feedback(&d, &grid, |_| None, 9);
        assert_eq!(demands.len(), 1);
        let dem = demands[0];
        assert_eq!(dem.key.texture, 7);
        assert_eq!(dem.key.mip, 0);
        assert_eq!(dem.key.x, 2);
        assert_eq!(dem.key.y, 1);
        assert_eq!(dem.desired_mip, 0);
        assert_eq!(dem.resident_mip, None);
        assert_eq!(dem.screen_importance, 500);
        assert_eq!(dem.byte_cost, 65_536);
        assert_eq!(dem.frame, 9);
    }

    #[test]
    fn coarse_request_collapses_cells_and_shifts_coords() {
        let d = desc();
        let mut grid = [NOT_REQUESTED; 16];
        // Four fine cells of the top-left 2x2 all want mip 1, which has
        // half-resolution pages, so they collapse onto page (0, 0) at mip 1.
        grid[cell(0, 0)] = 1;
        grid[cell(1, 0)] = 1;
        grid[cell(0, 1)] = 1;
        grid[cell(1, 1)] = 1;
        let demands = decode_feedback(&d, &grid, |_| None, 1);
        assert_eq!(demands.len(), 1);
        assert_eq!(demands[0].key.mip, 1);
        assert_eq!(demands[0].key.x, 0);
        assert_eq!(demands[0].key.y, 0);
    }

    #[test]
    fn same_mip_cells_collapsing_to_one_page_deduplicate() {
        let d = desc();
        let mut grid = [NOT_REQUESTED; 16];
        // The whole top-left 2x2 asks for mip 1; all four fine cells reduce to
        // the single half-resolution page (0, 0) at mip 1 and must dedup to one
        // demand rather than four.
        grid[cell(0, 0)] = 1;
        grid[cell(1, 0)] = 1;
        grid[cell(0, 1)] = 1;
        grid[cell(1, 1)] = 1;
        let demands = decode_feedback(&d, &grid, |_| None, 1);
        assert_eq!(demands.len(), 1);
        assert_eq!(demands[0].key.mip, 1);
        assert_eq!(demands[0].key.x, 0);
        assert_eq!(demands[0].key.y, 0);
        assert_eq!(demands[0].desired_mip, 1);
    }

    #[test]
    fn request_coarser_than_streamable_is_clamped() {
        let d = desc(); // mip_count 4 -> max mip 3.
        let mut grid = [NOT_REQUESTED; 16];
        grid[0] = 200; // absurdly coarse request.
        let demands = decode_feedback(&d, &grid, |_| None, 1);
        assert_eq!(demands.len(), 1);
        assert_eq!(demands[0].key.mip, 3, "clamped to max streamable mip");
        assert_eq!(demands[0].desired_mip, 3);
    }

    #[test]
    fn resident_mip_closure_fills_the_record() {
        let d = desc();
        let mut grid = [NOT_REQUESTED; 16];
        grid[0] = 0;
        let demands = decode_feedback(&d, &grid, |key| Some(key.mip + 2), 1);
        assert_eq!(demands[0].resident_mip, Some(2));
    }

    #[test]
    fn output_is_ascending_key_order() {
        let d = desc();
        let mut grid = [NOT_REQUESTED; 16];
        // Fill every cell at mip 0 so each is its own page in scrambled order.
        for (i, cell) in grid.iter_mut().enumerate() {
            *cell = 0;
            let _ = i;
        }
        let demands = decode_feedback(&d, &grid, |_| None, 1);
        assert_eq!(demands.len(), 16);
        let keys: Vec<TexturePageKey> = demands.iter().map(|dmd| dmd.key).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn base_mip_offset_addresses_absolute_levels() {
        let mut d = desc();
        d.base_mip = 2;
        d.mip_count = 3; // streamable mips 2..=4.
        let mut grid = [NOT_REQUESTED; 16];
        grid[0] = 0; // below base; clamps up to base_mip 2.
        let demands = decode_feedback(&d, &grid, |_| None, 1);
        assert_eq!(demands[0].key.mip, 2);
        assert_eq!(demands[0].desired_mip, 2);
    }
}
