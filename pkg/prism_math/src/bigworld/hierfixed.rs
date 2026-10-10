//! Hierarchical fixed-point coordinates for deterministic big worlds.
//!
//! The floating-origin [`GridPosition`](super::GridPosition) kills rendering
//! jitter but still reconstructs through `f32`/`f64`, so it is *not*
//! bit-reproducible across machines. Networked open worlds that need both a
//! large extent *and* lockstep determinism cannot use it for the authoritative
//! simulation state.
//!
//! This module provides the deterministic counterpart: a world point is a
//! coarse integer `cell: [i32; 3]` plus an in-cell `local: FxVec3` offset in
//! metres (signed Q32.32 fixed point from [`crate::fixed`]). Every operation
//! here is pure integer arithmetic — no floating point anywhere — so results
//! are identical on every platform, which is the requirement for lockstep
//! netcode and replay.
//!
//! # Model
//! Cells are [`FixedGridPosition::CELL_SIZE`]-metre cubes (a power of two, the
//! same convention as [`super::GridCell`]). A canonical position keeps its
//! `local` offset in `[0, CELL_SIZE)` on each axis; [`FixedGridPosition::canonical`]
//! re-centres by carrying whole-cell overflow into the integer `cell` using
//! Euclidean division on the raw fixed-point bits (exact, branch-free of
//! floats, and correct for negative offsets).
//!
//! # Cross-cell arithmetic
//! Differences are taken by *rebasing*: the integer cell delta is multiplied by
//! the exact cell size and added to the local delta, yielding a single
//! [`FxVec3`] offset ([`FixedGridPosition::rebased_offset`]). Intermediate
//! products use `i128` and saturate into the Q32.32 range, so the offset
//! between any two points is exact whenever it fits in the fixed-point type
//! (comfortably the case for the local neighbourhoods that fixed-point physics
//! actually operates on).

use crate::fixed::{Fixed, FxVec3};

/// Fractional bits of the Q32.32 [`Fixed`] representation.
const FRAC_BITS: u32 = 32;

/// A deterministic big-world position: integer cell plus fixed-point local.
///
/// Pure integer/fixed-point storage with no floating-point round trips, so two
/// machines that apply the same operations reach bit-identical coordinates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FixedGridPosition {
    /// Coarse integer cell index along each axis.
    pub cell: [i32; 3],
    /// In-cell offset in metres; canonical positions keep this in `[0, CELL_SIZE)`.
    pub local: FxVec3,
}

impl FixedGridPosition {
    /// Edge length of a cell in metres. A power of two so that
    /// `cell * CELL_SIZE` is an exact fixed-point product.
    pub const CELL_SIZE: i32 = 1024;

    /// Raw Q32.32 bit width of one cell edge (`CELL_SIZE << FRAC_BITS`).
    const CELL_BITS: i64 = (Self::CELL_SIZE as i64) << FRAC_BITS;

    /// A position at the world origin.
    pub const ORIGIN: Self = Self {
        cell: [0, 0, 0],
        local: FxVec3 {
            x: Fixed::ZERO,
            y: Fixed::ZERO,
            z: Fixed::ZERO,
        },
    };

    /// Build from an explicit cell and local offset (not re-centred).
    #[inline]
    #[must_use]
    pub const fn new(cell: [i32; 3], local: FxVec3) -> Self {
        Self { cell, local }
    }

    /// Re-centre so every `local` axis lies in `[0, CELL_SIZE)`, carrying any
    /// whole-cell overflow or underflow into the integer `cell`.
    ///
    /// Uses Euclidean division on the raw bits, so a negative offset borrows a
    /// cell exactly as expected (`local` is always non-negative afterwards).
    #[inline]
    #[must_use]
    pub fn canonical(self) -> Self {
        let (cx, lx) = Self::carry(self.cell[0], self.local.x);
        let (cy, ly) = Self::carry(self.cell[1], self.local.y);
        let (cz, lz) = Self::carry(self.cell[2], self.local.z);
        Self {
            cell: [cx, cy, cz],
            local: FxVec3::new(lx, ly, lz),
        }
    }

    /// Carry one axis: fold whole cells out of `local` into `cell`.
    #[inline]
    fn carry(cell: i32, local: Fixed) -> (i32, Fixed) {
        let bits = local.to_bits();
        let q = bits.div_euclid(Self::CELL_BITS);
        let r = bits.rem_euclid(Self::CELL_BITS);
        // Saturate the cell index; worlds never approach the i32 cell limit.
        let new_cell = (cell as i64 + q).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        (new_cell, Fixed::from_bits(r))
    }

    /// Translate by a fixed-point `delta` (metres) and re-centre.
    #[inline]
    #[must_use]
    pub fn translated(self, delta: FxVec3) -> Self {
        Self {
            cell: self.cell,
            local: FxVec3::new(
                Fixed::from_bits(self.local.x.to_bits().wrapping_add(delta.x.to_bits())),
                Fixed::from_bits(self.local.y.to_bits().wrapping_add(delta.y.to_bits())),
                Fixed::from_bits(self.local.z.to_bits().wrapping_add(delta.z.to_bits())),
            ),
        }
        .canonical()
    }

    /// Offset of `self` from `origin` as a single fixed-point vector (metres).
    ///
    /// Exact whenever the true offset fits in the Q32.32 range; it saturates
    /// otherwise. This is the deterministic analogue of expressing a point
    /// relative to the camera cell for local physics or rendering.
    #[inline]
    #[must_use]
    pub fn rebased_offset(self, origin: Self) -> FxVec3 {
        FxVec3::new(
            Self::axis_offset(self.cell[0], self.local.x, origin.cell[0], origin.local.x),
            Self::axis_offset(self.cell[1], self.local.y, origin.cell[1], origin.local.y),
            Self::axis_offset(self.cell[2], self.local.z, origin.cell[2], origin.local.z),
        )
    }

    /// One axis of [`Self::rebased_offset`], computed in `i128` then saturated.
    #[inline]
    fn axis_offset(cell: i32, local: Fixed, ocell: i32, olocal: Fixed) -> Fixed {
        let cell_delta = i64::from(cell) - i64::from(ocell);
        let bits = i128::from(cell_delta) * i128::from(Self::CELL_BITS)
            + i128::from(local.to_bits())
            - i128::from(olocal.to_bits());
        let clamped = bits.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
        Fixed::from_bits(clamped)
    }

    /// Squared distance to `other` in metres² (via the rebased offset).
    #[inline]
    #[must_use]
    pub fn distance_squared(self, other: Self) -> Fixed {
        let d = other.rebased_offset(self);
        d.dot(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fx(m: f64) -> Fixed {
        Fixed::from_f64(m)
    }

    #[test]
    fn canonical_keeps_local_in_cell() {
        // Local offset of 1500 m overflows the 1024 m cell by one cell + 476 m.
        let p = FixedGridPosition::new([3, -2, 0], FxVec3::new(fx(1500.0), fx(-100.0), fx(0.0)))
            .canonical();
        assert_eq!(p.cell[0], 4);
        assert!((p.local.x.to_f64() - 476.0).abs() < 1e-6);
        // Negative local borrows a cell: -100 m -> cell-1, local 924 m.
        assert_eq!(p.cell[1], -3);
        assert!((p.local.y.to_f64() - 924.0).abs() < 1e-6);
    }

    #[test]
    fn canonical_is_idempotent() {
        let p = FixedGridPosition::new([0, 0, 0], FxVec3::new(fx(5000.0), fx(-3333.0), fx(777.0)))
            .canonical();
        let q = p.canonical();
        assert_eq!(p, q);
        for axis in [p.local.x, p.local.y, p.local.z] {
            let m = axis.to_f64();
            assert!((0.0..1024.0).contains(&m), "local {m} out of cell");
        }
    }

    #[test]
    fn rebased_offset_crosses_cells_exactly() {
        // Two points three cells apart with known local offsets.
        let a = FixedGridPosition::new([10, 0, 0], FxVec3::new(fx(100.0), fx(0.0), fx(0.0)));
        let b = FixedGridPosition::new([13, 0, 0], FxVec3::new(fx(300.0), fx(0.0), fx(0.0)));
        // b - a = 3*1024 + (300 - 100) = 3272 m.
        let off = b.rebased_offset(a);
        assert!((off.x.to_f64() - 3272.0).abs() < 1e-6, "{}", off.x.to_f64());
    }

    #[test]
    fn translate_then_rebase_round_trips() {
        let origin = FixedGridPosition::ORIGIN;
        let delta = FxVec3::new(fx(2500.5), fx(-1300.25), fx(42.0));
        let moved = origin.translated(delta);
        // Canonical form moved across cells, but the offset back to the origin
        // must reproduce the applied delta exactly.
        let back = moved.rebased_offset(origin);
        assert!((back.x.to_f64() - 2500.5).abs() < 1e-6);
        assert!((back.y.to_f64() - (-1300.25)).abs() < 1e-6);
        assert!((back.z.to_f64() - 42.0).abs() < 1e-6);
    }

    #[test]
    fn offset_is_antisymmetric_and_deterministic() {
        let a = FixedGridPosition::new([-5, 7, 2], FxVec3::new(fx(10.0), fx(900.0), fx(512.0)))
            .canonical();
        let b = FixedGridPosition::new([3, -1, 2], FxVec3::new(fx(700.0), fx(50.0), fx(1.0)))
            .canonical();
        let ab = a.rebased_offset(b);
        let ba = b.rebased_offset(a);
        // Offsets are exact integers of raw bits, so negation is bit-exact.
        assert_eq!(ab.x.to_bits(), -ba.x.to_bits());
        assert_eq!(ab.y.to_bits(), -ba.y.to_bits());
        assert_eq!(ab.z.to_bits(), -ba.z.to_bits());
    }

    #[test]
    fn distance_squared_matches_manual() {
        let a = FixedGridPosition::new([0, 0, 0], FxVec3::new(fx(0.0), fx(0.0), fx(0.0)));
        let b =
            FixedGridPosition::new([1, 0, 0], FxVec3::new(fx(76.0), fx(3.0), fx(0.0))).canonical();
        // offset = (1024 + 76, 3, 0) = (1100, 3, 0); |.|^2 = 1210009.
        let d2 = a.distance_squared(b).to_f64();
        assert!((d2 - 1_210_009.0).abs() < 1.0, "d2 ={d2}");
    }
}
