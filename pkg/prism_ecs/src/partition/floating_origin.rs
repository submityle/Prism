//! 64-bit floating-origin rebasing (design §13.3).
//!
//! Large worlds (Star Citizen scale, UE5 World Partition) cannot store entity
//! positions in a single `f32` world space: at tens of kilometres from the
//! origin, an `f32`'s ~7 significant decimal digits leave gaps of several
//! centimetres, so geometry visibly *jitters* as the camera moves. The fix,
//! used by Star Citizen and others, is a **floating origin**:
//!
//! * The authoritative world position of an entity is kept in **double
//!   precision** as a *coarse* integer [`GridCell`] plus a *fine* single-
//!   precision [`LocalPos`] offset inside that cell. The cell index absorbs the
//!   large magnitude; the local offset stays small, so its `f32` keeps full
//!   sub-millimetre precision regardless of how far the cell is from `(0,0,0)`.
//! * Rendering and physics never work in absolute world space. Instead every
//!   frame they pick an **origin cell** (normally the camera's cell) and
//!   *rebase* each entity into that cell's local `f32` space via
//!   [`FloatingOrigin::rebase`]. Near the camera the rebased coordinates are
//!   tiny, so the `f32` pipeline is jitter-free; far-away entities lose
//!   precision but are culled or LOD'd (design §13.1/§13.2) before it matters.
//! * Only the cell origins are ever stored/compared in `f64`
//!   ([`FloatingOrigin::world_position`]); the hot per-entity data stays `f32`.
//!
//! This module is pure, `World`-independent math plus the two components the
//! transform-propagation adapter attaches to entities
//! (`prism_transform` consumes them, design §13.3). Everything here is fully
//! CPU-testable and deterministic: the quantisation uses an exact
//! floor-division (no `std` float intrinsics, keeping the kernel `no_std`), so
//! `quantize` followed by `world_position` round-trips bit-for-bit within the
//! chosen cell size.
//!
//! # Example
//!
//! ```ignore
//! let grid = FloatingOrigin::new(1024.0);                 // 1 km cells
//! let (cell, local) = grid.quantize(WorldPos::new(9_000_000.0, 0.0, 12.5));
//! // Camera sits in `cell`, so rebasing the entity there yields a small f32:
//! let g = grid.with_origin(cell);
//! let r = g.rebase(cell, local);
//! assert!(r.x.abs() < 1024.0);
//! ```

use crate::component::Component;

/// A double-precision world position in metres — the authoritative truth for
/// an entity in a large world (design §13.3).
///
/// Game logic that needs an absolute coordinate works through this type;
/// render/physics use the rebased [`LocalPos`] instead.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct WorldPos {
    /// World X in metres.
    pub x: f64,
    /// World Y in metres.
    pub y: f64,
    /// World Z in metres.
    pub z: f64,
}

impl WorldPos {
    /// Creates a world position from its three `f64` components.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Integer grid-cell coordinate: the *coarse* half of a floating-origin
/// position (design §13.3).
///
/// `i64` axes give a practically unbounded world: with a 1 km cell the
/// addressable range spans far beyond any real play space while every cell's
/// local offset keeps full `f32` precision. Attached to entities as a
/// [`Component`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct GridCell {
    /// Cell index along X.
    pub x: i64,
    /// Cell index along Y.
    pub y: i64,
    /// Cell index along Z.
    pub z: i64,
}

impl GridCell {
    /// The origin cell `(0, 0, 0)`.
    pub const ORIGIN: Self = Self { x: 0, y: 0, z: 0 };

    /// Creates a cell coordinate from its three integer axes.
    pub const fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }

    /// The signed cell delta `self - other`, per axis.
    ///
    /// Used when rebasing: the delta is small near the active origin, so
    /// multiplying it by the cell size stays representable in `f32`.
    pub const fn offset_from(self, other: Self) -> (i64, i64, i64) {
        (self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Component for GridCell {}

/// Cell-relative single-precision offset: the *fine* half of a floating-origin
/// position (design §13.3).
///
/// Always kept inside `[0, cell_size)` on each axis by
/// [`FloatingOrigin::normalize`], so its `f32` precision is independent of the
/// cell's distance from the world origin. This is the coordinate render and
/// physics actually consume. Attached to entities as a [`Component`].
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct LocalPos {
    /// Local X offset within the cell, in metres.
    pub x: f32,
    /// Local Y offset within the cell, in metres.
    pub y: f32,
    /// Local Z offset within the cell, in metres.
    pub z: f32,
}

impl LocalPos {
    /// Creates a local offset from its three `f32` components.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Creates a local offset from an `[x, y, z]` array.
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            z: a[2],
        }
    }

    /// Returns this offset as an `[x, y, z]` array — the `[f32; 3]` shape the
    /// LOD/stream processors consume (design §13.2).
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

impl From<LocalPos> for [f32; 3] {
    #[inline]
    fn from(p: LocalPos) -> Self {
        p.to_array()
    }
}

impl From<[f32; 3]> for LocalPos {
    #[inline]
    fn from(a: [f32; 3]) -> Self {
        LocalPos::from_array(a)
    }
}

/// The floating-origin grid (design §13.3): a uniform cell size plus the
/// currently active **origin cell** that render/physics rebase against.
///
/// Typically stored as a resource and advanced each frame via
/// [`recenter`](Self::recenter) to follow the camera. The cell size is the edge
/// length of a cubic cell in metres and must be strictly positive and finite.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FloatingOrigin {
    cell_size: f64,
    origin: GridCell,
}

impl Component for FloatingOrigin {}

impl FloatingOrigin {
    /// Creates a grid with the given cell size (metres) and the origin at
    /// [`GridCell::ORIGIN`].
    ///
    /// # Panics
    /// Panics if `cell_size` is not strictly positive and finite.
    pub fn new(cell_size: f64) -> Self {
        assert!(
            cell_size.is_finite() && cell_size > 0.0,
            "cell_size must be finite and > 0"
        );
        Self {
            cell_size,
            origin: GridCell::ORIGIN,
        }
    }

    /// Returns a copy of this grid with its active origin set to `origin`.
    pub const fn with_origin(mut self, origin: GridCell) -> Self {
        self.origin = origin;
        self
    }

    /// The cell edge length in metres.
    pub const fn cell_size(&self) -> f64 {
        self.cell_size
    }

    /// The currently active origin cell that [`rebase`](Self::rebase) is
    /// relative to.
    pub const fn origin(&self) -> GridCell {
        self.origin
    }

    /// Moves the active origin to `origin` (e.g. follow the camera's cell).
    pub fn recenter(&mut self, origin: GridCell) {
        self.origin = origin;
    }

    /// Moves the active origin onto the cell containing `pos` and returns the
    /// camera's resulting `(cell, local)` split (design §13.3).
    ///
    /// Call this each frame with the camera's absolute [`WorldPos`] so the
    /// origin tracks the camera; because the origin only moves when the camera
    /// crosses a cell boundary, the cell size itself provides the recenter
    /// hysteresis and avoids per-frame thrash.
    pub fn recenter_to(&mut self, pos: WorldPos) -> (GridCell, LocalPos) {
        let (cell, local) = self.quantize(pos);
        self.origin = cell;
        (cell, local)
    }

    /// Splits an absolute [`WorldPos`] into its coarse [`GridCell`] and fine
    /// [`LocalPos`] on this grid.
    ///
    /// The returned local offset is guaranteed to lie in `[0, cell_size)` on
    /// each axis. Pairs with [`world_position`](Self::world_position) as an
    /// exact round-trip within the cell size.
    pub fn quantize(&self, pos: WorldPos) -> (GridCell, LocalPos) {
        let (cx, lx) = self.split_axis(pos.x);
        let (cy, ly) = self.split_axis(pos.y);
        let (cz, lz) = self.split_axis(pos.z);
        (
            GridCell::new(cx, cy, cz),
            LocalPos::new(lx as f32, ly as f32, lz as f32),
        )
    }

    /// Reconstructs the absolute double-precision [`WorldPos`] from a cell and
    /// its local offset. Inverse of [`quantize`](Self::quantize).
    pub fn world_position(&self, cell: GridCell, local: LocalPos) -> WorldPos {
        WorldPos::new(
            cell.x as f64 * self.cell_size + local.x as f64,
            cell.y as f64 * self.cell_size + local.y as f64,
            cell.z as f64 * self.cell_size + local.z as f64,
        )
    }

    /// Rebases an entity at `(cell, local)` into the active origin's local
    /// `f32` space (design §13.3): the coordinate render/physics consume.
    ///
    /// The result is `(cell - origin) * cell_size + local`. Near the active
    /// origin the cell delta is tiny, so the `f32` output keeps full precision
    /// — this is exactly what eliminates far-from-origin jitter.
    pub fn rebase(&self, cell: GridCell, local: LocalPos) -> LocalPos {
        let (dx, dy, dz) = cell.offset_from(self.origin);
        LocalPos::new(
            (dx as f64 * self.cell_size + local.x as f64) as f32,
            (dy as f64 * self.cell_size + local.y as f64) as f32,
            (dz as f64 * self.cell_size + local.z as f64) as f32,
        )
    }

    /// Rebases `(cell, local)` into the active origin's local space and returns
    /// it as an `[x, y, z]` array — the direct bridge from a big-world position
    /// to the `[f32; 3]` the LOD/stream processors measure distance in
    /// (design §13.2 + §13.3).
    pub fn rebase_array(&self, cell: GridCell, local: LocalPos) -> [f32; 3] {
        self.rebase(cell, local).to_array()
    }

    /// Quantizes an absolute [`WorldPos`] and rebases it into the active
    /// origin's local `[f32; 3]` space in one step (design §13.3).
    ///
    /// The direct path from a double-precision world point (camera, interest
    /// source, ad-hoc query) to the jitter-free local coordinate the LOD/stream
    /// processors measure in.
    pub fn rebase_world(&self, pos: WorldPos) -> [f32; 3] {
        let (cell, local) = self.quantize(pos);
        self.rebase_array(cell, local)
    }

    /// Re-canonicalises a possibly-drifted `(cell, local)` so the local offset
    /// is back inside `[0, cell_size)` on every axis, carrying any overflow
    /// into the cell index.
    ///
    /// Physics integrates in local space, so a fast entity's `local` can grow
    /// past a cell boundary; calling this keeps the `f32` offset small and its
    /// precision high without changing the entity's absolute world position.
    pub fn normalize(&self, cell: GridCell, local: LocalPos) -> (GridCell, LocalPos) {
        let (cx, lx) = self.carry_axis(cell.x, local.x);
        let (cy, ly) = self.carry_axis(cell.y, local.y);
        let (cz, lz) = self.carry_axis(cell.z, local.z);
        (GridCell::new(cx, cy, cz), LocalPos::new(lx, ly, lz))
    }

    /// Floor-divides `value` by the cell size into `(cell_index, local)` with
    /// `local` in `[0, cell_size)`, using exact integer floor (no `std` float
    /// intrinsics) so the split is deterministic and `no_std`-clean.
    fn split_axis(&self, value: f64) -> (i64, f64) {
        let q = value / self.cell_size;
        // Truncating cast rounds toward zero; adjust downward for negatives so
        // this is a true floor. Saturating-cast semantics keep extreme inputs
        // well-defined.
        let mut cell = q as i64;
        if (cell as f64) > q {
            cell -= 1;
        }
        let local = value - cell as f64 * self.cell_size;
        (cell, local)
    }

    /// Normalises one axis: folds a local offset back into `[0, cell_size)` and
    /// carries the whole-cell part into the integer index.
    fn carry_axis(&self, cell: i64, local: f32) -> (i64, f32) {
        let (dcell, rem) = self.split_axis(local as f64);
        (cell + dcell, rem as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_world_position_round_trips() {
        let grid = FloatingOrigin::new(1024.0);
        let pos = WorldPos::new(9_000_000.0, -2_500.5, 12.5);
        let (cell, local) = grid.quantize(pos);
        let back = grid.world_position(cell, local);
        // Round-trip is exact for the cell origin; local is f32 so allow its
        // rounding on the fractional metre only.
        assert!((back.x - pos.x).abs() < 1.0e-3);
        assert!((back.y - pos.y).abs() < 1.0e-3);
        assert!((back.z - pos.z).abs() < 1.0e-3);
    }

    #[test]
    fn local_offset_is_within_cell() {
        let grid = FloatingOrigin::new(100.0);
        for &v in &[0.0, 50.0, 99.999, 100.0, 250.0, -1.0, -100.0, -150.0] {
            let (_cell, local) = grid.quantize(WorldPos::new(v, v, v));
            assert!(
                (0.0..100.0).contains(&local.x),
                "local {} out of range for input {}",
                local.x,
                v
            );
        }
    }

    #[test]
    fn negative_coordinates_floor_correctly() {
        let grid = FloatingOrigin::new(100.0);
        // -1 metre lands in cell -1 at local 99.
        let (cell, local) = grid.quantize(WorldPos::new(-1.0, 0.0, 0.0));
        assert_eq!(cell.x, -1);
        assert!((local.x - 99.0).abs() < 1.0e-3);
        // Exact multiple sits at the start of its cell.
        let (cell2, local2) = grid.quantize(WorldPos::new(-200.0, 0.0, 0.0));
        assert_eq!(cell2.x, -2);
        assert!(local2.x.abs() < 1.0e-3);
    }

    #[test]
    fn rebase_at_own_cell_is_just_local() {
        let grid = FloatingOrigin::new(1024.0);
        let cell = GridCell::new(8790, 0, 0);
        let local = LocalPos::new(12.5, -3.0, 7.0);
        let g = grid.with_origin(cell);
        let r = g.rebase(cell, local);
        assert_eq!(r, local);
    }

    #[test]
    fn rebase_neighbour_cell_adds_one_cell_size() {
        let grid = FloatingOrigin::new(1024.0).with_origin(GridCell::new(100, 0, 0));
        let r = grid.rebase(GridCell::new(101, 0, 0), LocalPos::new(5.0, 0.0, 0.0));
        assert!((r.x - 1029.0).abs() < 1.0e-3);
    }

    #[test]
    fn rebase_far_entity_stays_small_near_origin() {
        // Two entities a few cells apart, viewed from a far-away origin cell:
        // their rebased f32 difference must be exact to the metre even though
        // their absolute f64 coordinates are in the millions.
        let grid = FloatingOrigin::new(1024.0).with_origin(GridCell::new(10_000, 0, 0));
        let a = grid.rebase(GridCell::new(10_000, 0, 0), LocalPos::new(1.0, 0.0, 0.0));
        let b = grid.rebase(GridCell::new(10_002, 0, 0), LocalPos::new(1.0, 0.0, 0.0));
        assert!((a.x - 1.0).abs() < 1.0e-4);
        assert!((b.x - (1.0 + 2.0 * 1024.0)).abs() < 1.0e-3);
    }

    #[test]
    fn normalize_carries_overflow_into_cell() {
        let grid = FloatingOrigin::new(100.0);
        // Local drifted past two cells in +X and below zero in -Y.
        let (cell, local) =
            grid.normalize(GridCell::new(5, 5, 5), LocalPos::new(250.0, -30.0, 0.0));
        assert_eq!(cell.x, 7);
        assert!((local.x - 50.0).abs() < 1.0e-3);
        assert_eq!(cell.y, 4);
        assert!((local.y - 70.0).abs() < 1.0e-3);
        assert_eq!(cell.z, 5);
    }

    #[test]
    fn normalize_preserves_world_position() {
        let grid = FloatingOrigin::new(128.0);
        let cell = GridCell::new(3, -2, 10);
        let local = LocalPos::new(400.0, -50.0, 5.0);
        let before = grid.world_position(cell, local);
        let (ncell, nlocal) = grid.normalize(cell, local);
        let after = grid.world_position(ncell, nlocal);
        assert!((before.x - after.x).abs() < 1.0e-2);
        assert!((before.y - after.y).abs() < 1.0e-2);
        assert!((before.z - after.z).abs() < 1.0e-2);
        assert!((0.0..128.0).contains(&nlocal.x));
        assert!((0.0..128.0).contains(&nlocal.y));
    }

    #[test]
    fn recenter_moves_origin() {
        let mut grid = FloatingOrigin::new(256.0);
        assert_eq!(grid.origin(), GridCell::ORIGIN);
        grid.recenter(GridCell::new(1, 2, 3));
        assert_eq!(grid.origin(), GridCell::new(1, 2, 3));
    }

    #[test]
    fn offset_from_is_signed_delta() {
        let a = GridCell::new(10, 5, -3);
        let b = GridCell::new(4, 7, -3);
        assert_eq!(a.offset_from(b), (6, -2, 0));
    }

    #[test]
    fn recenter_to_tracks_the_camera_cell() {
        let mut grid = FloatingOrigin::new(1000.0);
        // Camera at absolute 3025 m lands in cell 3; origin follows it and the
        // returned split is the camera's own (cell, local).
        let (cell, local) = grid.recenter_to(WorldPos::new(3025.0, 0.0, 0.0));
        assert_eq!(cell, GridCell::new(3, 0, 0));
        assert!((local.x - 25.0).abs() < 1.0e-3);
        assert_eq!(grid.origin(), GridCell::new(3, 0, 0));
        // After recentering, the camera itself rebases near the local origin.
        assert!(grid.rebase_world(WorldPos::new(3025.0, 0.0, 0.0))[0].abs() < 26.0);
    }

    #[test]
    fn rebase_world_matches_quantize_then_rebase() {
        let grid = FloatingOrigin::new(1000.0).with_origin(GridCell::new(10, 0, 0));
        let pos = WorldPos::new(11_025.0, 0.0, 0.0); // cell 11, local 25
        let direct = grid.rebase_world(pos);
        let (cell, local) = grid.quantize(pos);
        let staged = grid.rebase_array(cell, local);
        assert_eq!(direct, staged);
        // One cell east of the origin (1000 m) + 25 m local.
        assert!((direct[0] - 1025.0).abs() < 1.0e-2);
    }

    #[test]
    #[should_panic]
    fn zero_cell_size_panics() {
        let _ = FloatingOrigin::new(0.0);
    }
}
