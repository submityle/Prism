//! World-position → cell interest bridge (design §13.1 + §13.3).
//!
//! [`StreamDriver::stream`](crate::partition::driver::StreamDriver::stream)
//! loads the neighbourhood around a set of **interest sources** expressed as
//! integer [`CellCoord`]s, but the game only ever knows an interest's
//! *continuous* position — a camera, a player, a mission anchor — as a
//! double-precision [`WorldPos`] (design §13.3). [`InterestGrid`] is the
//! missing map: it floor-divides a world point onto the streaming grid so the
//! owner can turn "where the camera is" into "which cell wants its
//! surroundings resident".
//!
//! # Mirroring the floating-origin grid
//!
//! `CellCoord`'s documentation notes that the world→cell mapping is defined by
//! the owning scene and *typically mirrors the floating-origin
//! [`GridCell`](crate::partition::floating_origin::GridCell) grid*. Building an
//! interest grid with [`InterestGrid::matching`] guarantees exactly that: it
//! borrows the [`FloatingOrigin`]'s cell size, so a world point lands in the
//! same cell index the floating origin would quantize it to (down to the
//! `i64`→`i32` range of a streaming coordinate). Keeping the two grids aligned
//! means the cell an entity is *rebased* against and the cell that drives its
//! *streaming* are the same lattice — no half-cell skew between LOD and
//! residency.
//!
//! # Determinism
//!
//! [`interests`](InterestGrid::interests) sorts its output by `(x, y, z)` and
//! removes duplicates, so two interest sources sharing a cell collapse to one
//! entry and the resulting slice is frame-stable regardless of source order
//! (design §14) — matching the ordering guarantees the streamer itself makes.
//!
//! ```
//! use prism_ecs::partition::floating_origin::{FloatingOrigin, WorldPos};
//! use prism_ecs::partition::interest::InterestGrid;
//!
//! let origin = FloatingOrigin::new(1000.0); // 1 km cells
//! let grid = InterestGrid::matching(&origin);
//!
//! // Camera and a nearby ally share a cell; a distant anchor is its own.
//! let camera = WorldPos::new(1500.0, 20.0, -500.0);
//! let ally = WorldPos::new(1800.0, 5.0, -200.0);
//! let anchor = WorldPos::new(-10.0, 0.0, 9000.0);
//!
//! let interests = grid.interests(&[camera, ally, anchor]);
//! // (-1, 0, 9) and (1, 0, -1); the duplicate (1, 0, -1) collapsed.
//! assert_eq!(interests.len(), 2);
//! // driver.stream(world, &interests);
//! ```

use alloc::vec::Vec;

use crate::partition::cell::CellCoord;
use crate::partition::floating_origin::{FloatingOrigin, GridCell, WorldPos};

/// Maps continuous [`WorldPos`] interest sources onto integer [`CellCoord`]s
/// for the streaming scheduler (design §13.1 + §13.3).
///
/// The grid is a single `cell_size` (metres): the edge length of a cubic
/// streaming cell. Construct it with [`matching`](Self::matching) to mirror a
/// [`FloatingOrigin`] exactly, or [`new`](Self::new) for a standalone grid.
///
/// The grid holds no state beyond its cell size and never mutates anything —
/// [`cell_of`](Self::cell_of) and [`interests`](Self::interests) are pure
/// coordinate transforms.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InterestGrid {
    cell_size: f64,
}

impl InterestGrid {
    /// Creates an interest grid with the given cell edge length in metres.
    ///
    /// # Panics
    /// Panics if `cell_size` is not strictly positive and finite, matching
    /// [`FloatingOrigin::new`].
    pub fn new(cell_size: f64) -> Self {
        assert!(
            cell_size.is_finite() && cell_size > 0.0,
            "cell_size must be finite and > 0"
        );
        Self { cell_size }
    }

    /// Builds an interest grid that mirrors `origin`'s lattice by borrowing its
    /// [`cell_size`](FloatingOrigin::cell_size).
    ///
    /// Use this so the cell an entity is rebased against (§13.3) and the cell
    /// that drives its streaming (§13.1) are the same grid: a world point then
    /// maps to the index [`FloatingOrigin::quantize`] would assign it, within
    /// the `i32` range of a streaming coordinate.
    pub fn matching(origin: &FloatingOrigin) -> Self {
        // `origin` already guarantees a finite, positive cell size.
        Self {
            cell_size: origin.cell_size(),
        }
    }

    /// The cell edge length in metres.
    #[inline]
    pub const fn cell_size(&self) -> f64 {
        self.cell_size
    }

    /// Floor-divides an absolute [`WorldPos`] onto the streaming grid.
    ///
    /// Each axis is `floor(pos / cell_size)`, so a point exactly on a cell
    /// boundary belongs to the *upper* cell and negative coordinates round
    /// toward negative infinity (not toward zero) — the behaviour streaming
    /// needs for a seamless lattice across the origin. Extreme inputs saturate
    /// into the `i32` range rather than wrapping.
    #[inline]
    pub fn cell_of(&self, pos: WorldPos) -> CellCoord {
        CellCoord::new(
            floor_div(pos.x, self.cell_size),
            floor_div(pos.y, self.cell_size),
            floor_div(pos.z, self.cell_size),
        )
    }

    /// Projects a floating-origin [`GridCell`] straight onto a streaming
    /// [`CellCoord`] by narrowing each `i64` axis to `i32` (saturating).
    ///
    /// When the interest's position has already been quantized by the
    /// floating origin, this skips re-dividing and reuses that split directly,
    /// which is exactly what [`matching`](Self::matching) keeps consistent with
    /// [`cell_of`](Self::cell_of).
    #[inline]
    pub fn cell_of_grid(&self, cell: GridCell) -> CellCoord {
        CellCoord::new(narrow(cell.x), narrow(cell.y), narrow(cell.z))
    }

    /// Maps every world-space interest source to its cell, then sorts by
    /// `(x, y, z)` and removes duplicates.
    ///
    /// The result is ready to hand to
    /// [`StreamDriver::stream`](crate::partition::driver::StreamDriver::stream):
    /// collapsing sources that share a cell avoids loading the same
    /// neighbourhood twice, and the stable order keeps the schedule
    /// deterministic (design §14).
    pub fn interests(&self, sources: &[WorldPos]) -> Vec<CellCoord> {
        let mut cells: Vec<CellCoord> = sources.iter().map(|&p| self.cell_of(p)).collect();
        cells.sort_unstable_by(|a, b| {
            a.x.cmp(&b.x)
                .then(a.y.cmp(&b.y))
                .then(a.z.cmp(&b.z))
        });
        cells.dedup();
        cells
    }
}

/// Floor-divides `value` by `size` into an `i32` cell index.
///
/// Mirrors `FloatingOrigin::split_axis`: a truncating float→int cast rounds
/// toward zero, so negatives are nudged down one to realise a true floor. The
/// intermediate `i64` plus a final clamp keeps extreme or non-finite inputs
/// well-defined (saturating) instead of producing an unspecified `i32`.
#[inline]
fn floor_div(value: f64, size: f64) -> i32 {
    let q = value / size;
    // Saturating float→int cast (Rust >= 1.45) pins ±inf / overflow to the
    // i64 bounds; the clamp below then narrows safely to i32.
    let mut cell = q as i64;
    if (cell as f64) > q {
        // `saturating_sub` keeps `q == f64::NEG_INFINITY` (cell == i64::MIN)
        // well-defined instead of overflowing.
        cell = cell.saturating_sub(1);
    }
    narrow(cell)
}

/// Narrows an `i64` cell index to `i32`, saturating out-of-range values to the
/// nearest bound rather than wrapping.
#[inline]
fn narrow(value: i64) -> i32 {
    if value > i32::MAX as i64 {
        i32::MAX
    } else if value < i32::MIN as i64 {
        i32::MIN
    } else {
        value as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_non_positive_cell_size() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                std::panic::catch_unwind(|| InterestGrid::new(bad)).is_err(),
                "cell_size {bad} should panic"
            );
        }
    }

    #[test]
    fn cell_of_floors_positive_coords() {
        let grid = InterestGrid::new(100.0);
        // 0..100 -> 0, 100..200 -> 1, boundary belongs to the upper cell.
        assert_eq!(grid.cell_of(WorldPos::new(0.0, 50.0, 99.9)), CellCoord::new(0, 0, 0));
        assert_eq!(grid.cell_of(WorldPos::new(100.0, 250.0, 100.0)), CellCoord::new(1, 2, 1));
    }

    #[test]
    fn cell_of_floors_negative_coords_toward_neg_infinity() {
        let grid = InterestGrid::new(100.0);
        // -0.1 is in cell -1, not 0; -100 is exactly the start of cell -1.
        assert_eq!(grid.cell_of(WorldPos::new(-0.1, -100.0, -100.1)), CellCoord::new(-1, -1, -2));
        assert_eq!(grid.cell_of(WorldPos::new(-250.0, -1.0, -0.0)), CellCoord::new(-3, -1, 0));
    }

    #[test]
    fn cell_of_grid_narrows_directly() {
        let grid = InterestGrid::new(100.0);
        assert_eq!(grid.cell_of_grid(GridCell::new(5, -3, 42)), CellCoord::new(5, -3, 42));
    }

    #[test]
    fn matching_mirrors_floating_origin_quantization() {
        let origin = FloatingOrigin::new(1000.0);
        let grid = InterestGrid::matching(&origin);
        assert_eq!(grid.cell_size(), origin.cell_size());
        for pos in [
            WorldPos::new(1500.0, 20.0, -500.0),
            WorldPos::new(-10.0, 0.0, 9000.0),
            WorldPos::new(-999.9, -1000.0, 1000.0),
        ] {
            let (gcell, _) = origin.quantize(pos);
            // The interest grid and the floating origin agree cell-for-cell.
            assert_eq!(grid.cell_of(pos), grid.cell_of_grid(gcell));
        }
    }

    #[test]
    fn interests_sort_and_dedup() {
        let grid = InterestGrid::new(100.0);
        let camera = WorldPos::new(150.0, 20.0, -50.0); // (1, 0, -1)
        let ally = WorldPos::new(180.0, 5.0, -20.0); //    (1, 0, -1) duplicate
        let anchor = WorldPos::new(-10.0, 0.0, 900.0); //  (-1, 0, 9)
        let interests = grid.interests(&[camera, ally, anchor]);
        assert_eq!(interests, [CellCoord::new(-1, 0, 9), CellCoord::new(1, 0, -1)]);
    }

    #[test]
    fn interests_on_empty_sources_is_empty() {
        let grid = InterestGrid::new(100.0);
        assert!(grid.interests(&[]).is_empty());
    }

    #[test]
    fn floor_div_saturates_extreme_inputs() {
        let grid = InterestGrid::new(1.0);
        assert_eq!(grid.cell_of(WorldPos::new(f64::INFINITY, f64::NEG_INFINITY, 0.0)),
            CellCoord::new(i32::MAX, i32::MIN, 0));
        // A magnitude beyond i32 range clamps to the bound.
        assert_eq!(grid.cell_of(WorldPos::new(5e18, -5e18, 0.0)),
            CellCoord::new(i32::MAX, i32::MIN, 0));
    }
}
