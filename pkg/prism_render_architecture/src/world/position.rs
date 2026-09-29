//! Absolute world positions, the camera-relative origin, and rebasing.
//!
//! A [`WorldPosition`] is an integer cell plus a small local offset. Rendering,
//! however, wants coordinates near zero so that single-precision (`f32`)
//! transforms stay accurate; a position 1000 km from the numeric origin loses
//! centimetre precision once squeezed into `f32`. The renderer therefore keeps a
//! [`WorldOrigin`] near the camera and expresses everything as an offset from
//! it. When the camera travels far enough the origin is *rebased* to a new cell
//! and every offset is recomputed, bounding the magnitude that ever reaches
//! `f32`. An `epoch` stamps each origin so stale, pre-rebase offsets can be
//! detected and invalidated.

use super::cell::{WorldCell, WorldGrid};

/// An absolute position: integer cell plus local offset within the cell.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldPosition {
    pub cell: WorldCell,
    pub local: [f64; 3],
}

/// The camera-relative origin all render-space offsets are measured from.
///
/// Integer-only, so it derives full equality/ordering. The `epoch` increments on
/// every rebase; offsets captured under an older epoch are stale.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct WorldOrigin {
    pub cell: WorldCell,
    pub epoch: u64,
}

impl WorldOrigin {
    /// Builds an origin at `cell` with the given `epoch`.
    #[must_use]
    pub const fn new(cell: WorldCell, epoch: u64) -> Self {
        Self { cell, epoch }
    }

    /// Returns a copy rebased to `cell` with the epoch advanced by one.
    ///
    /// Advancing the epoch is what invalidates offsets captured against the
    /// previous origin; callers must rebase those offsets with
    /// [`rebase_offset`].
    #[must_use]
    pub const fn rebased_to(self, cell: WorldCell) -> Self {
        Self {
            cell,
            epoch: self.epoch.wrapping_add(1),
        }
    }

    /// True when `epoch` matches this origin's epoch (offset is still valid).
    #[must_use]
    pub const fn is_current(self, epoch: u64) -> bool {
        self.epoch == epoch
    }

    /// True when `epoch` predates this origin's epoch (offset is stale).
    #[must_use]
    pub const fn is_stale(self, epoch: u64) -> bool {
        epoch != self.epoch
    }
}

impl WorldPosition {
    /// Builds a position from a cell and local offset without normalization.
    #[must_use]
    pub const fn new(cell: WorldCell, local: [f64; 3]) -> Self {
        Self { cell, local }
    }

    /// Render-space offset of this position relative to `origin`.
    ///
    /// `offset = (cell - origin.cell) * cell_size + local`. When the position is
    /// near the origin this is small and safe to demote to `f32`.
    #[must_use]
    pub fn render_offset(self, origin: WorldOrigin, grid: WorldGrid) -> [f64; 3] {
        let delta = self.cell.offset(origin.cell);
        let size = grid.cell_size();
        [
            delta[0] as f64 * size + self.local[0],
            delta[1] as f64 * size + self.local[1],
            delta[2] as f64 * size + self.local[2],
        ]
    }

    /// Reconstructs an absolute position from a render-space `offset` measured
    /// against `origin`, re-normalizing onto the grid.
    #[must_use]
    pub fn from_render_offset(offset: [f64; 3], origin: WorldOrigin, grid: WorldGrid) -> Self {
        let size = grid.cell_size();
        let world = [
            f64::from(origin.cell.x) * size + offset[0],
            f64::from(origin.cell.y) * size + offset[1],
            f64::from(origin.cell.z) * size + offset[2],
        ];
        grid.quantize(world)
    }
}

/// Rebases a single render-space offset from `old_origin` to `new_origin`.
///
/// `new_offset = old_offset - (new_origin.cell - old_origin.cell) * cell_size`.
/// The epochs are not inspected here; the caller advances the origin epoch and
/// applies this to every live offset so none exceed the new origin's range.
#[must_use]
pub fn rebase_offset(
    offset: [f64; 3],
    old_origin: WorldOrigin,
    new_origin: WorldOrigin,
    grid: WorldGrid,
) -> [f64; 3] {
    let delta = new_origin.cell.offset(old_origin.cell);
    let size = grid.cell_size();
    [
        offset[0] - delta[0] as f64 * size,
        offset[1] - delta[1] as f64 * size,
        offset[2] - delta[2] as f64 * size,
    ]
}

/// Rebases a batch of render-space offsets in place.
pub fn rebase_all(
    offsets: &mut [[f64; 3]],
    old_origin: WorldOrigin,
    new_origin: WorldOrigin,
    grid: WorldGrid,
) {
    for offset in offsets.iter_mut() {
        *offset = rebase_offset(*offset, old_origin, new_origin, grid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1.0e-6
    }

    fn magnitude(v: [f64; 3]) -> f64 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    #[test]
    fn render_offset_is_zero_at_origin() {
        let grid = WorldGrid::new(100.0).unwrap();
        let origin = WorldOrigin::new(WorldCell::new(5, 5, 5), 0);
        let pos = WorldPosition::new(WorldCell::new(5, 5, 5), [1.0, 2.0, 3.0]);
        let offset = pos.render_offset(origin, grid);
        assert!(close(offset[0], 1.0));
        assert!(close(offset[1], 2.0));
        assert!(close(offset[2], 3.0));
    }

    #[test]
    fn render_offset_roundtrips_through_absolute() {
        let grid = WorldGrid::new(64.0).unwrap();
        let origin = WorldOrigin::new(WorldCell::new(-3, 7, 12), 4);
        let pos = WorldPosition::new(WorldCell::new(10, -2, 40), [1.5, 63.0, 0.25]);
        let offset = pos.render_offset(origin, grid);
        let back = WorldPosition::from_render_offset(offset, origin, grid);
        assert_eq!(back.cell, pos.cell);
        assert!(close(back.local[0], pos.local[0]));
        assert!(close(back.local[1], pos.local[1]));
        assert!(close(back.local[2], pos.local[2]));
    }

    #[test]
    fn rebasing_shrinks_offset_magnitude_for_far_positions() {
        // A position far from the numeric origin: ~1000 km at 100 m cells.
        let grid = WorldGrid::new(100.0).unwrap();
        let far = WorldPosition::new(WorldCell::new(10_000, 0, 0), [12.5, 0.0, 0.0]);

        let old_origin = WorldOrigin::new(WorldCell::new(0, 0, 0), 0);
        let far_offset = far.render_offset(old_origin, grid);
        assert!(magnitude(far_offset) > 1.0e6);

        // Rebase the origin next to the position.
        let new_origin = old_origin.rebased_to(far.cell);
        let near_offset = rebase_offset(far_offset, old_origin, new_origin, grid);
        assert!(magnitude(near_offset) < 100.0);

        // The rebased offset matches computing it fresh against the new origin.
        let fresh = far.render_offset(new_origin, grid);
        assert!(close(near_offset[0], fresh[0]));
        assert!(close(near_offset[1], fresh[1]));
        assert!(close(near_offset[2], fresh[2]));
    }

    #[test]
    fn rebasing_preserves_f32_precision() {
        // Demote a far offset to f32 (lossy) vs. a rebased near offset (crisp).
        let grid = WorldGrid::new(1.0).unwrap();
        let pos = WorldPosition::new(WorldCell::new(8_000_000, 0, 0), [0.125, 0.0, 0.0]);
        let old_origin = WorldOrigin::new(WorldCell::new(0, 0, 0), 0);

        let far = pos.render_offset(old_origin, grid);
        let far_f32 = far[0] as f32 as f64;
        let far_err = (far_f32 - far[0]).abs();

        let new_origin = old_origin.rebased_to(pos.cell);
        let near = rebase_offset(far, old_origin, new_origin, grid);
        let near_f32 = near[0] as f32 as f64;
        let near_err = (near_f32 - near[0]).abs();

        // Far-from-origin f32 loses the sub-metre detail; rebased keeps it.
        assert!(far_err > 0.1);
        assert!(near_err < 1.0e-3);
    }

    #[test]
    fn batch_rebase_matches_scalar() {
        let grid = WorldGrid::new(50.0).unwrap();
        let old_origin = WorldOrigin::new(WorldCell::new(1, 2, 3), 0);
        let new_origin = old_origin.rebased_to(WorldCell::new(4, 2, 3));
        let mut offsets = [[100.0, 0.0, 0.0], [0.0, 50.0, -25.0]];
        let expected: [[f64; 3]; 2] = [
            rebase_offset(offsets[0], old_origin, new_origin, grid),
            rebase_offset(offsets[1], old_origin, new_origin, grid),
        ];
        rebase_all(&mut offsets, old_origin, new_origin, grid);
        for (got, want) in offsets.iter().zip(expected.iter()) {
            assert!(close(got[0], want[0]));
            assert!(close(got[1], want[1]));
            assert!(close(got[2], want[2]));
        }
    }

    #[test]
    fn epoch_advances_and_flags_staleness() {
        let origin = WorldOrigin::new(WorldCell::new(0, 0, 0), 7);
        let rebased = origin.rebased_to(WorldCell::new(1, 0, 0));
        assert_eq!(rebased.epoch, 8);
        assert!(rebased.is_current(8));
        assert!(!rebased.is_current(7));
        assert!(rebased.is_stale(7));
        assert!(!rebased.is_stale(8));
    }

    #[test]
    fn epoch_wraps_without_panicking() {
        let origin = WorldOrigin::new(WorldCell::new(0, 0, 0), u64::MAX);
        let rebased = origin.rebased_to(WorldCell::new(1, 0, 0));
        assert_eq!(rebased.epoch, 0);
    }
}
