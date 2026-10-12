//! Area-of-interest management (design §8 "AOI 兴趣管理").
//!
//! Replicating the whole world to every connection does not scale. AOI answers
//! "which entities is this connection interested in right now?" so the transport
//! sends only those — and, frame to frame, only the ones that **entered** or
//! **left** view.
//!
//! [`AoiGrid`] is a uniform 2D interest grid over the horizontal plane (`x`, `z`):
//! entities bucket into fixed cells, and a radius query visits only the cells the
//! circle overlaps, then filters by exact distance. Everything is deterministic —
//! a `BTreeMap`/`BTreeSet` and an ascending output — so every server replica
//! derives the same interest set from the same positions. [`interest_delta`] then
//! turns two consecutive views into the enter/leave edges a replicator ships.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

/// The most cells a radius query will walk before falling back to scanning every
/// entity. A cell walk is only an acceleration, so capping it trades a little
/// work for a hard upper bound on a single query.
const MAX_CELLS: i64 = 4096;

/// A rejected AOI operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AoiError {
    /// The cell size must be finite and greater than zero.
    BadCellSize,
    /// Positions (and query centres) must be finite.
    NonFinite,
}

/// A uniform 2D interest grid over `(x, z)`.
#[derive(Clone, Debug)]
pub struct AoiGrid {
    cell_size: f32,
    cells: BTreeMap<(i32, i32), BTreeSet<u32>>,
    positions: BTreeMap<u32, (f32, f32)>,
}

impl AoiGrid {
    /// A grid with the given square cell size.
    ///
    /// # Errors
    /// [`AoiError::BadCellSize`] when `cell_size` is not finite and positive.
    pub fn new(cell_size: f32) -> Result<Self, AoiError> {
        if !cell_size.is_finite() || cell_size <= 0.0 {
            return Err(AoiError::BadCellSize);
        }
        Ok(Self {
            cell_size,
            cells: BTreeMap::new(),
            positions: BTreeMap::new(),
        })
    }

    /// The cell edge length.
    #[must_use]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// The number of tracked entities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether no entity is tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The last position recorded for `id`.
    #[must_use]
    pub fn position(&self, id: u32) -> Option<(f32, f32)> {
        self.positions.get(&id).copied()
    }

    /// Insert or move `id` to `(x, z)`.
    ///
    /// Re-inserting an existing id moves it (its old cell is vacated).
    ///
    /// # Errors
    /// [`AoiError::NonFinite`] when `x`/`z` is not finite.
    pub fn insert(&mut self, id: u32, x: f32, z: f32) -> Result<(), AoiError> {
        if !x.is_finite() || !z.is_finite() {
            return Err(AoiError::NonFinite);
        }
        if let Some((old_x, old_z)) = self.positions.get(&id).copied() {
            let old_cell = self.cell_of(old_x, old_z);
            if let Some(bucket) = self.cells.get_mut(&old_cell) {
                bucket.remove(&id);
                if bucket.is_empty() {
                    self.cells.remove(&old_cell);
                }
            }
        }
        let cell = self.cell_of(x, z);
        self.cells.entry(cell).or_default().insert(id);
        self.positions.insert(id, (x, z));
        Ok(())
    }

    /// Remove `id`, returning whether it was tracked.
    pub fn remove(&mut self, id: u32) -> bool {
        let Some((x, z)) = self.positions.remove(&id) else {
            return false;
        };
        let cell = self.cell_of(x, z);
        if let Some(bucket) = self.cells.get_mut(&cell) {
            bucket.remove(&id);
            if bucket.is_empty() {
                self.cells.remove(&cell);
            }
        }
        true
    }

    /// Every tracked id within `radius` of `(x, z)`, ascending.
    ///
    /// A non-finite centre or a non-positive radius yields an empty view. The
    /// result is exact: a cell is a coarse candidate set, and each candidate is
    /// checked against the true (squared) distance, so an entity sharing the
    /// centre's cell but outside the radius is excluded.
    ///
    /// The cell walk is bounded: a radius (or cell size) so extreme that the
    /// circle spans more than [`MAX_CELLS`] cells falls back to scanning every
    /// tracked entity, so a finite input can never make the query iterate over
    /// billions of empty cells.
    #[must_use]
    pub fn query_circle(&self, x: f32, z: f32, radius: f32) -> Vec<u32> {
        if !x.is_finite() || !z.is_finite() || !radius.is_finite() || radius <= 0.0 {
            return Vec::new();
        }
        let radius_sq = radius * radius;
        let (min_x, min_z) = self.cell_of(x - radius, z - radius);
        let (max_x, max_z) = self.cell_of(x + radius, z + radius);

        // Bound the cell scan (see the doc above).
        let cells_x = i64::from(max_x) - i64::from(min_x) + 1;
        let cells_z = i64::from(max_z) - i64::from(min_z) + 1;
        if cells_x.saturating_mul(cells_z) > MAX_CELLS {
            return self.scan_all(x, z, radius, radius_sq);
        }

        // A `BTreeSet` keeps the view ascending and deduplicated regardless of
        // cell iteration order.
        let mut view = BTreeSet::new();
        for cx in min_x..=max_x {
            for cz in min_z..=max_z {
                let Some(bucket) = self.cells.get(&(cx, cz)) else {
                    continue;
                };
                for &id in bucket {
                    if let Some(&(ex, ez)) = self.positions.get(&id)
                        && within(ex, ez, x, z, radius, radius_sq)
                    {
                        view.insert(id);
                    }
                }
            }
        }
        view.into_iter().collect()
    }

    /// The exact-distance scan used when the cell range is too large to walk.
    /// `positions` is a `BTreeMap`, so the output is ascending.
    fn scan_all(&self, x: f32, z: f32, radius: f32, radius_sq: f32) -> Vec<u32> {
        self.positions
            .iter()
            .filter(|&(_, &(ex, ez))| within(ex, ez, x, z, radius, radius_sq))
            .map(|(&id, _)| id)
            .collect()
    }

    /// The grid cell a position falls in.
    fn cell_of(&self, x: f32, z: f32) -> (i32, i32) {
        (
            floor_to_i32(x / self.cell_size),
            floor_to_i32(z / self.cell_size),
        )
    }
}

/// The change in a connection's interest set between two consecutive views.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InterestDelta {
    /// Ids newly in view, ascending.
    pub entered: Vec<u32>,
    /// Ids no longer in view, ascending.
    pub left: Vec<u32>,
}

impl InterestDelta {
    /// Whether nothing changed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entered.is_empty() && self.left.is_empty()
    }
}

/// The enter/leave edges between an old and a new interest set.
///
/// Both inputs are treated as **sets**: order does not matter and duplicates are
/// ignored, so a caller passing a view that is not strictly ascending (or a
/// hand-built list with repeats) still gets the correct edges rather than a bogus
/// enter+leave pair for one id. The result is ascending and deterministic.
#[must_use]
pub fn interest_delta(previous: &[u32], current: &[u32]) -> InterestDelta {
    let before: BTreeSet<u32> = previous.iter().copied().collect();
    let after: BTreeSet<u32> = current.iter().copied().collect();
    let entered = after.difference(&before).copied().collect();
    let left = before.difference(&after).copied().collect();
    InterestDelta { entered, left }
}

/// Floor a finite `f32` to an `i32` without `f32::floor` (std-only).
///
/// Truncates toward zero, then steps down for a negative value with a fractional
/// part. The cast saturates for inputs past `i32`, so far-apart positions degrade
/// to one extreme bucket — still correct, since the exact circle test filters
/// candidates anyway.
fn floor_to_i32(value: f32) -> i32 {
    let truncated = value as i32;
    if value < 0.0 && truncated as f32 != value {
        truncated.saturating_sub(1)
    } else {
        truncated
    }
}

/// Whether the finite position `(ex, ez)` lies within `radius` of `(x, z)`.
///
/// A box test (which cannot overflow) rejects the obvious misses. The exact
/// circle test then uses the squared form normally — safe because the box test
/// bounds `|dx|, |dz| <= radius`, so neither square overflows — and the
/// radius-normalised form when `radius * radius` saturates to infinity, which
/// keeps an extreme radius from admitting entities in the box corners.
fn within(ex: f32, ez: f32, x: f32, z: f32, radius: f32, radius_sq: f32) -> bool {
    let (dx, dz) = (ex - x, ez - z);
    if dx.abs() > radius || dz.abs() > radius {
        return false;
    }
    if radius_sq.is_finite() {
        dx * dx + dz * dz <= radius_sq
    } else {
        let (nx, nz) = (dx / radius, dz / radius);
        nx * nx + nz * nz <= 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn grid() -> AoiGrid {
        AoiGrid::new(10.0).expect("valid cell size")
    }

    #[test]
    fn a_radius_query_returns_only_entities_in_range_ascending() {
        let mut aoi = grid();
        // Insertion order is deliberately not id order.
        aoi.insert(30, 1.0, 1.0).unwrap();
        aoi.insert(10, 2.0, 2.0).unwrap();
        aoi.insert(20, 100.0, 100.0).unwrap();

        assert_eq!(aoi.query_circle(0.0, 0.0, 5.0), vec![10, 30]);
        assert!(aoi.query_circle(0.0, 0.0, 1.0).is_empty());
        assert_eq!(aoi.len(), 3);
    }

    #[test]
    fn a_same_cell_entity_outside_the_radius_is_filtered() {
        // Both entities share cell (0,0), but only one is within the radius: the
        // exact distance test, not the cell bucket, decides membership.
        let mut aoi = grid();
        aoi.insert(1, 0.0, 0.0).unwrap();
        aoi.insert(2, 9.5, 9.5).unwrap();
        assert_eq!(aoi.query_circle(0.0, 0.0, 1.0), vec![1]);
        assert_eq!(aoi.query_circle(0.0, 0.0, 14.0), vec![1, 2]);
    }

    #[test]
    fn moving_and_removing_update_the_view() {
        let mut aoi = grid();
        aoi.insert(7, 0.0, 0.0).unwrap();
        assert_eq!(aoi.query_circle(0.0, 0.0, 2.0), vec![7]);

        aoi.insert(7, 50.0, 50.0).unwrap(); // move out of range
        assert!(aoi.query_circle(0.0, 0.0, 2.0).is_empty());
        assert_eq!(aoi.position(7), Some((50.0, 50.0)));
        assert_eq!(aoi.query_circle(50.0, 50.0, 1.0), vec![7]);

        assert!(aoi.remove(7));
        assert!(!aoi.remove(7), "removing twice is a no-op");
        assert!(aoi.is_empty());
    }

    #[test]
    fn queries_reject_a_degenerate_input() {
        let mut aoi = grid();
        aoi.insert(1, 0.0, 0.0).unwrap();
        assert!(aoi.query_circle(0.0, 0.0, 0.0).is_empty());
        assert!(aoi.query_circle(0.0, 0.0, -1.0).is_empty());
        assert!(aoi.query_circle(f32::NAN, 0.0, 1.0).is_empty());
        assert_eq!(aoi.insert(2, f32::INFINITY, 0.0), Err(AoiError::NonFinite));
        assert_eq!(AoiGrid::new(0.0).err(), Some(AoiError::BadCellSize));
    }

    #[test]
    fn floor_handles_signs_and_extremes() {
        assert_eq!(floor_to_i32(2.7), 2);
        assert_eq!(floor_to_i32(-0.5), -1);
        assert_eq!(floor_to_i32(-3.0), -3);
        assert_eq!(floor_to_i32(i32::MIN as f32), i32::MIN);
        assert_eq!(floor_to_i32(f32::MAX), i32::MAX);
    }

    #[test]
    fn interest_delta_reports_enter_and_leave() {
        let previous = [1, 3, 5, 7];
        let current = [1, 2, 5, 8];
        let delta = interest_delta(&previous, &current);
        assert_eq!(delta.entered, vec![2, 8]);
        assert_eq!(delta.left, vec![3, 7]);
        assert!(!delta.is_empty());

        let same = interest_delta(&[1, 2], &[1, 2]);
        assert_eq!(same, InterestDelta::default());
        assert!(same.is_empty());
        assert_eq!(interest_delta(&[], &[4]).entered, vec![4]);
        assert_eq!(interest_delta(&[4], &[]).left, vec![4]);
    }

    /// A radius spanning more cells than the walk cap falls back to scanning
    /// every entity, so a finite input cannot hang and the view stays exact.
    #[test]
    fn a_huge_radius_falls_back_to_a_full_scan() {
        let mut aoi = grid();
        aoi.insert(1, 0.0, 0.0).unwrap();
        aoi.insert(2, 3.0, 4.0).unwrap();
        aoi.insert(3, 1.0e9, 0.0).unwrap();
        assert_eq!(aoi.query_circle(0.0, 0.0, 1.0e9), vec![1, 2, 3]);
        assert_eq!(aoi.query_circle(0.0, 0.0, 10.0), vec![1, 2]);
        assert_eq!(aoi.query_circle(0.0, 0.0, f32::MAX), vec![1, 2, 3]);
    }

    /// A subnormal cell size makes every coordinate overflow the cell index; the
    /// fallback keeps the query bounded and correct rather than walking the whole
    /// i32 cell space.
    #[test]
    fn a_subnormal_cell_size_does_not_hang() {
        let mut aoi = AoiGrid::new(1.0e-45).expect("finite positive cell size");
        aoi.insert(1, 0.0, 0.0).unwrap();
        aoi.insert(2, 100.0, 100.0).unwrap();
        assert_eq!(aoi.query_circle(0.0, 0.0, 1.0), vec![1]);
        assert_eq!(aoi.query_circle(0.0, 0.0, 200.0), vec![1, 2]);
    }

    /// An extreme radius must not admit a box-corner entity whose true distance
    /// exceeds it (regression: `radius * radius` overflowed to infinity).
    #[test]
    fn an_extreme_radius_excludes_box_corners() {
        let mut aoi = AoiGrid::new(1.0).expect("valid cell size");
        let radius = 1.0e20_f32;
        aoi.insert(1, 0.0, 0.0).unwrap();
        aoi.insert(2, radius, radius).unwrap(); // distance r*sqrt(2) > r
        assert_eq!(aoi.query_circle(0.0, 0.0, radius), vec![1]);
    }

    /// `interest_delta` treats the views as sets, so duplicates or a non-ascending
    /// order cannot fabricate an enter+leave pair for one id.
    #[test]
    fn interest_delta_is_robust_to_duplicates_and_order() {
        assert_eq!(interest_delta(&[1, 1], &[1]), InterestDelta::default());
        assert_eq!(interest_delta(&[1], &[1, 1]), InterestDelta::default());
        let unsorted = interest_delta(&[5, 1], &[1, 9]);
        assert_eq!(unsorted.entered, vec![9]);
        assert_eq!(unsorted.left, vec![5]);
    }
}
