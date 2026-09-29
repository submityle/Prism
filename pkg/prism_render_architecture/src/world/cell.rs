//! Integer world cells and the grid that quantizes `f64` world coordinates.
//!
//! A large world is partitioned into a uniform grid of cubic cells addressed by
//! signed integer coordinates. Splitting an absolute position into an integer
//! cell plus a small local offset is what keeps rendering math in `f32` range
//! even when the world spans kilometres: the integer part carries the magnitude
//! and the local part stays near the origin. This module owns the cell type, the
//! configurable grid, and the integer distance/neighbourhood queries.

use super::position::WorldPosition;

/// Integer address of a cell in the uniform world grid.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct WorldCell {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl WorldCell {
    /// Builds a cell from explicit integer coordinates.
    #[must_use]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Per-axis signed offset `self - other`, widened to `i64` so it never
    /// overflows for any pair of `i32` cells.
    #[must_use]
    pub const fn offset(self, other: Self) -> [i64; 3] {
        [
            self.x as i64 - other.x as i64,
            self.y as i64 - other.y as i64,
            self.z as i64 - other.z as i64,
        ]
    }

    /// Manhattan (L1) distance in whole cells.
    #[must_use]
    pub const fn manhattan(self, other: Self) -> u64 {
        let d = self.offset(other);
        d[0].unsigned_abs() + d[1].unsigned_abs() + d[2].unsigned_abs()
    }

    /// Chebyshev (L-infinity) distance in whole cells.
    #[must_use]
    pub const fn chebyshev(self, other: Self) -> u64 {
        let d = self.offset(other);
        let ax = d[0].unsigned_abs();
        let ay = d[1].unsigned_abs();
        let az = d[2].unsigned_abs();
        let xy = if ax > ay { ax } else { ay };
        if xy > az {
            xy
        } else {
            az
        }
    }

    /// Squared Euclidean distance in whole cells (exact, integer).
    #[must_use]
    pub const fn euclidean_squared(self, other: Self) -> u64 {
        let d = self.offset(other);
        let ax = d[0].unsigned_abs();
        let ay = d[1].unsigned_abs();
        let az = d[2].unsigned_abs();
        ax.saturating_mul(ax)
            .saturating_add(ay.saturating_mul(ay))
            .saturating_add(az.saturating_mul(az))
    }

    /// Euclidean distance in whole cells, using `sqrt` (no transcendental).
    #[must_use]
    pub fn euclidean(self, other: Self) -> f64 {
        let squared = self.euclidean_squared(other) as f64;
        squared.sqrt()
    }

    /// True when `other` is a face/edge/corner neighbour (Chebyshev 1).
    #[must_use]
    pub const fn is_neighbor(self, other: Self) -> bool {
        self.chebyshev(other) == 1
    }

    /// The six face-adjacent (6-connected) neighbours.
    #[must_use]
    pub const fn face_neighbors(self) -> [Self; 6] {
        [
            Self::new(self.x + 1, self.y, self.z),
            Self::new(self.x - 1, self.y, self.z),
            Self::new(self.x, self.y + 1, self.z),
            Self::new(self.x, self.y - 1, self.z),
            Self::new(self.x, self.y, self.z + 1),
            Self::new(self.x, self.y, self.z - 1),
        ]
    }
}

/// Uniform grid that maps `f64` world coordinates to cells and back.
///
/// The cell size is configurable but fixed for the grid's lifetime; a valid grid
/// always has a finite, strictly positive cell size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldGrid {
    cell_size: f64,
}

impl WorldGrid {
    /// Builds a grid with the given cell size, or `None` when `cell_size` is not
    /// finite and strictly positive.
    #[must_use]
    pub fn new(cell_size: f64) -> Option<Self> {
        if !cell_size.is_finite() || cell_size <= 0.0 {
            return None;
        }
        Some(Self { cell_size })
    }

    /// The grid's cell size in world units.
    #[must_use]
    pub const fn cell_size(self) -> f64 {
        self.cell_size
    }

    /// Quantizes one axis into an integer cell index plus a local offset in
    /// `[0, cell_size)`.
    fn quantize_axis(self, value: f64) -> (i32, f64) {
        let quotient = (value / self.cell_size).floor();
        let cell = quotient as i32;
        let local = value - quotient * self.cell_size;
        (cell, local)
    }

    /// Splits an absolute `f64` world position into a cell plus local offset.
    ///
    /// The returned local offset always lies in `[0, cell_size)` per axis, so
    /// [`Self::world_of`] reconstructs the input within floating-point rounding.
    #[must_use]
    pub fn quantize(self, world: [f64; 3]) -> WorldPosition {
        let (cx, lx) = self.quantize_axis(world[0]);
        let (cy, ly) = self.quantize_axis(world[1]);
        let (cz, lz) = self.quantize_axis(world[2]);
        WorldPosition {
            cell: WorldCell::new(cx, cy, cz),
            local: [lx, ly, lz],
        }
    }

    /// Reconstructs the absolute `f64` world position from a cell + local offset.
    #[must_use]
    pub fn world_of(self, position: WorldPosition) -> [f64; 3] {
        [
            f64::from(position.cell.x) * self.cell_size + position.local[0],
            f64::from(position.cell.y) * self.cell_size + position.local[1],
            f64::from(position.cell.z) * self.cell_size + position.local[2],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1.0e-9
    }

    #[test]
    fn offset_widens_without_overflow() {
        let a = WorldCell::new(i32::MAX, 0, i32::MIN);
        let b = WorldCell::new(i32::MIN, 0, i32::MAX);
        let d = a.offset(b);
        assert_eq!(d[0], i64::from(i32::MAX) - i64::from(i32::MIN));
        assert_eq!(d[2], i64::from(i32::MIN) - i64::from(i32::MAX));
    }

    #[test]
    fn distance_metrics_agree_on_known_case() {
        let a = WorldCell::new(0, 0, 0);
        let b = WorldCell::new(3, 4, 0);
        assert_eq!(a.manhattan(b), 7);
        assert_eq!(a.chebyshev(b), 4);
        assert_eq!(a.euclidean_squared(b), 25);
        assert!(close(a.euclidean(b), 5.0));
    }

    #[test]
    fn neighbor_predicate_and_list() {
        let c = WorldCell::new(2, 2, 2);
        for n in c.face_neighbors() {
            assert!(c.is_neighbor(n));
            assert_eq!(c.manhattan(n), 1);
        }
        assert!(!c.is_neighbor(c));
        assert!(!c.is_neighbor(WorldCell::new(4, 2, 2)));
    }

    #[test]
    fn grid_rejects_invalid_cell_sizes() {
        assert!(WorldGrid::new(0.0).is_none());
        assert!(WorldGrid::new(-1.0).is_none());
        assert!(WorldGrid::new(f64::NAN).is_none());
        assert!(WorldGrid::new(f64::INFINITY).is_none());
        assert!(WorldGrid::new(64.0).is_some());
    }

    #[test]
    fn quantize_produces_local_in_range() {
        let grid = WorldGrid::new(100.0).unwrap();
        let pos = grid.quantize([250.0, -30.0, 99.999]);
        assert_eq!(pos.cell, WorldCell::new(2, -1, 0));
        for axis in pos.local {
            assert!((0.0..100.0).contains(&axis));
        }
        assert!(close(pos.local[0], 50.0));
        assert!(close(pos.local[1], 70.0));
    }

    #[test]
    fn quantize_world_of_roundtrips() {
        let grid = WorldGrid::new(128.0).unwrap();
        let samples = [
            [0.0, 0.0, 0.0],
            [127.5, -256.25, 4096.0],
            [-0.001, 1_000_000.0, -1_000_000.0],
        ];
        for world in samples {
            let back = grid.world_of(grid.quantize(world));
            assert!(close(back[0], world[0]));
            assert!(close(back[1], world[1]));
            assert!(close(back[2], world[2]));
        }
    }
}
