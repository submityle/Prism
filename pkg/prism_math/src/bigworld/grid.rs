//! Grid-cell origin rebasing for jitter-free large worlds.
//!
//! See the [module docs](crate::bigworld) for the precision contract. This file
//! defines the integer [`GridCell`] lattice and the [`GridPosition`] split of a
//! world point into `(cell, local f32 offset)`, plus the rebasing math that
//! expresses a far-away point as a small camera-relative `f32` offset.

use crate::f64::{DAffine3, DVec3};
use crate::Vec3;
use core::ops::{Add, Sub};

/// An integer cell in the world-space grid lattice.
///
/// A cell spans [`GridCell::CELL_SIZE`] metres along each axis. The cell's
/// world-space origin (its minimum corner) is `cell * CELL_SIZE` and is
/// representable *exactly* in `f64` for any `i32` cell within
/// `|cell| * CELL_SIZE < 2^53`, i.e. the entire `i32` range. Cell coordinates
/// therefore never lose precision, no matter how far from the world origin.
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(C)]
pub struct GridCell {
    /// Cell index along X.
    pub x: i32,
    /// Cell index along Y.
    pub y: i32,
    /// Cell index along Z.
    pub z: i32,
}

impl GridCell {
    /// Edge length of a cell in metres.
    ///
    /// A power of two so that `cell * CELL_SIZE` is an exact `f64` product and
    /// the local-offset split is free of rounding. At 1024 m a single-precision
    /// local offset resolves to `1024 * 2^-23 ≈ 0.12 mm`, comfortably below the
    /// 1 mm jitter target.
    pub const CELL_SIZE: f64 = 1024.0;

    /// The cell at the world origin.
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    /// Create a cell from explicit indices.
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
    /// The cell that contains `world` (component-wise floor division).
    #[inline]
    pub fn from_dvec3(world: DVec3) -> Self {
        Self {
            x: floor_div(world.x),
            y: floor_div(world.y),
            z: floor_div(world.z),
        }
    }
    /// The world-space origin (minimum corner) of this cell.
    #[inline]
    pub fn origin(self) -> DVec3 {
        DVec3::new(
            self.x as f64 * Self::CELL_SIZE,
            self.y as f64 * Self::CELL_SIZE,
            self.z as f64 * Self::CELL_SIZE,
        )
    }
    /// The exact world-space translation from the `origin` cell to this cell,
    /// `(self - origin) * CELL_SIZE`.
    ///
    /// This is the heart of origin rebasing: the difference of two cell indices
    /// is computed in `i64` (never overflowing for `i32` inputs) and scaled by
    /// the exact `CELL_SIZE`, so the result carries no rounding error.
    #[inline]
    pub fn relative_translation(self, origin: GridCell) -> DVec3 {
        DVec3::new(
            (self.x as i64 - origin.x as i64) as f64 * Self::CELL_SIZE,
            (self.y as i64 - origin.y as i64) as f64 * Self::CELL_SIZE,
            (self.z as i64 - origin.z as i64) as f64 * Self::CELL_SIZE,
        )
    }
    /// A pure-translation [`DAffine3`] mapping this cell's local space into the
    /// `origin` cell's local space (see [`Self::relative_translation`]).
    #[inline]
    pub fn relative_transform(self, origin: GridCell) -> DAffine3 {
        DAffine3::from_translation(self.relative_translation(origin))
    }
}

impl Add for GridCell {
    type Output = GridCell;
    #[inline]
    fn add(self, r: GridCell) -> GridCell {
        GridCell::new(self.x + r.x, self.y + r.y, self.z + r.z)
    }
}
impl Sub for GridCell {
    type Output = GridCell;
    #[inline]
    fn sub(self, r: GridCell) -> GridCell {
        GridCell::new(self.x - r.x, self.y - r.y, self.z - r.z)
    }
}

impl core::fmt::Debug for GridCell {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "GridCell({}, {}, {})", self.x, self.y, self.z)
    }
}

/// A world position split into a [`GridCell`] and a small local offset.
///
/// The `offset` is stored in single precision and nominally lies in
/// `[0, CELL_SIZE)` along each axis, so it always keeps sub-millimetre
/// precision regardless of how far the cell is from the world origin. Reassemble
/// the exact `f64` world position with [`Self::to_dvec3`], or rebase into a
/// camera-relative `f32` offset with [`Self::rebased_offset`].
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct GridPosition {
    /// The containing grid cell.
    pub cell: GridCell,
    /// Local offset from the cell origin, in metres (`f32`).
    pub offset: Vec3,
}

impl GridPosition {
    /// Build a position from a cell and an explicit local offset.
    #[inline]
    pub const fn new(cell: GridCell, offset: Vec3) -> Self {
        Self { cell, offset }
    }
    /// Split an exact `f64` world position into `(cell, f32 offset)`.
    ///
    /// The cell is `floor(world / CELL_SIZE)` and the offset is the `f64`
    /// remainder narrowed to `f32`; because the remainder is in
    /// `[0, CELL_SIZE)` the narrowing keeps ~0.12 mm precision.
    #[inline]
    pub fn from_dvec3(world: DVec3) -> Self {
        let cell = GridCell::from_dvec3(world);
        let offset = (world - cell.origin()).as_vec3();
        Self { cell, offset }
    }
    /// Reassemble the exact `f64` world position (`cell.origin() + offset`).
    #[inline]
    pub fn to_dvec3(self) -> DVec3 {
        self.cell.origin() + self.offset.as_dvec3()
    }
    /// Re-split so the local `offset` is canonicalised back into
    /// `[0, CELL_SIZE)`, folding any accumulated drift into the cell index.
    ///
    /// Use this after integrating motion in local space to stop the offset from
    /// growing without bound (which would reintroduce `f32` jitter).
    #[inline]
    pub fn recenter(self) -> Self {
        Self::from_dvec3(self.to_dvec3())
    }
    /// Rebase this position into an `f32` offset relative to the `origin`
    /// cell's origin.
    ///
    /// For a point `d` metres from the `origin` cell the result has an absolute
    /// `f32` error of about `d * 2^-23`: ≤ 0.5 mm for `d ≤ 4 km` and ≤ 12 mm at
    /// 100 km. Rebasing to the camera's cell therefore keeps nearby geometry
    /// (where jitter is visible) sub-millimetre accurate even when the whole
    /// scene sits 100 km from the world origin.
    #[inline]
    pub fn rebased_offset(self, origin: GridCell) -> Vec3 {
        (self.cell.relative_translation(origin) + self.offset.as_dvec3()).as_vec3()
    }
}

impl core::fmt::Debug for GridPosition {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "GridPosition {{ cell: {:?}, offset: {:?} }}", self.cell, self.offset)
    }
}

/// Component-wise floor division of `v` by [`GridCell::CELL_SIZE`], returning an
/// `i32` cell index. Values outside the `i32` range saturate.
#[inline]
fn floor_div(v: f64) -> i32 {
    let c = crate::float::f64::floor(v / GridCell::CELL_SIZE);
    if c >= i32::MAX as f64 {
        i32::MAX
    } else if c <= i32::MIN as f64 {
        i32::MIN
    } else {
        c as i32
    }
}
