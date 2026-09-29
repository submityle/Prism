//! Staggered `MAC` grid geometry and cell classification for the `GPU` fluid
//! solver.
//!
//! [`GridDims`] captures the immutable geometry of a `nx * ny * nz` cell grid:
//! the cell size, the world-space origin, and the flat-index arithmetic for the
//! three staggered face-velocity fields and the cell-centred classification.
//! Velocities live on cell faces (`u` on the `x`-normal faces, `v` on the
//! `y`-normal faces, `w` on the `z`-normal faces) so the discrete divergence
//! and pressure-gradient operators are compact, exactly as in the `CPU`
//! reference [`prism_physics_core`](prism_physics_core::fluid::mac_grid).
//!
//! The geometry is deliberately split from the mutable field data (which lives
//! in `GPU` buffers on the device path, and in plain `Vec`s on the `CPU` golden
//! path) so both engines share one source of truth for indexing.
//!
//! # Provenance
//!
//! The `MAC` staggered layout is a standard, publicly documented `CFD` construct
//! (Harlow and Welch 1965; Bridson). This module contains no Unreal Engine
//! source or derived code.

use glam::Vec3;

/// Classification of a `MAC` cell.
///
/// The discriminants are the wire encoding shared with the `WGSL` kernels: the
/// classification is uploaded as a `u32` per cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CellType {
    /// An empty (gas) cell; pressure is pinned to zero here.
    Air = 0,
    /// A cell containing liquid.
    Fluid = 1,
    /// A solid obstacle / wall cell; velocity normal to it is constrained.
    Solid = 2,
}

impl CellType {
    /// The `u32` wire encoding uploaded to the device.
    #[must_use]
    pub fn code(self) -> u32 {
        self as u32
    }

    /// Decodes a `u32` wire value, treating unknown codes as [`CellType::Air`].
    #[must_use]
    pub fn from_code(code: u32) -> CellType {
        match code {
            1 => CellType::Fluid,
            2 => CellType::Solid,
            _ => CellType::Air,
        }
    }
}

/// Immutable geometry of a staggered `MAC` grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridDims {
    /// Number of cells along `x`.
    pub nx: u32,
    /// Number of cells along `y`.
    pub ny: u32,
    /// Number of cells along `z`.
    pub nz: u32,
    /// Cell spacing (uniform on every axis).
    pub dx: f32,
    /// World-space position of the corner of cell `(0, 0, 0)`.
    pub origin: Vec3,
}

impl GridDims {
    /// Creates a grid geometry.
    #[must_use]
    pub fn new(nx: u32, ny: u32, nz: u32, dx: f32, origin: Vec3) -> GridDims {
        GridDims {
            nx,
            ny,
            nz,
            dx,
            origin,
        }
    }

    /// Whether every axis has at least one cell.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.nx > 0 && self.ny > 0 && self.nz > 0 && self.dx > 0.0 && self.dx.is_finite()
    }

    /// Number of cells.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        (self.nx * self.ny * self.nz) as usize
    }

    /// Number of `u` (`x`-face) samples, dims `(nx + 1, ny, nz)`.
    #[must_use]
    pub fn u_count(&self) -> usize {
        ((self.nx + 1) * self.ny * self.nz) as usize
    }

    /// Number of `v` (`y`-face) samples, dims `(nx, ny + 1, nz)`.
    #[must_use]
    pub fn v_count(&self) -> usize {
        (self.nx * (self.ny + 1) * self.nz) as usize
    }

    /// Number of `w` (`z`-face) samples, dims `(nx, ny, nz + 1)`.
    #[must_use]
    pub fn w_count(&self) -> usize {
        (self.nx * self.ny * (self.nz + 1)) as usize
    }

    /// Flat index of the `u` face `(i, j, k)`.
    #[must_use]
    pub fn u_idx(&self, i: u32, j: u32, k: u32) -> usize {
        (i + (self.nx + 1) * (j + self.ny * k)) as usize
    }

    /// Flat index of the `v` face `(i, j, k)`.
    #[must_use]
    pub fn v_idx(&self, i: u32, j: u32, k: u32) -> usize {
        (i + self.nx * (j + (self.ny + 1) * k)) as usize
    }

    /// Flat index of the `w` face `(i, j, k)`.
    #[must_use]
    pub fn w_idx(&self, i: u32, j: u32, k: u32) -> usize {
        (i + self.nx * (j + self.ny * k)) as usize
    }

    /// Offset of the `u` field inside a concatenated `[u | v | w]` face array.
    #[must_use]
    pub fn u_offset(&self) -> usize {
        0
    }

    /// Offset of the `v` field inside a concatenated `[u | v | w]` face array.
    #[must_use]
    pub fn v_offset(&self) -> usize {
        self.u_count()
    }

    /// Offset of the `w` field inside a concatenated `[u | v | w]` face array.
    #[must_use]
    pub fn w_offset(&self) -> usize {
        self.u_count() + self.v_count()
    }

    /// Total length of a concatenated `[u | v | w]` face array.
    #[must_use]
    pub fn face_total(&self) -> usize {
        self.u_count() + self.v_count() + self.w_count()
    }

    /// Flat index of cell `(i, j, k)`.
    #[must_use]
    pub fn cell_idx(&self, i: u32, j: u32, k: u32) -> usize {
        (i + self.nx * (j + self.ny * k)) as usize
    }

    /// Cell-space coordinate of world position `p` (in units of `dx`, relative
    /// to the origin).
    #[must_use]
    pub fn cell_space(&self, p: Vec3) -> Vec3 {
        (p - self.origin) / self.dx
    }

    /// Returns the cell indices containing `p`, if inside the grid.
    #[must_use]
    pub fn cell_of(&self, p: Vec3) -> Option<(u32, u32, u32)> {
        let c = self.cell_space(p);
        if c.x < 0.0 || c.y < 0.0 || c.z < 0.0 {
            return None;
        }
        let (i, j, k) = (c.x as u32, c.y as u32, c.z as u32);
        if i >= self.nx || j >= self.ny || k >= self.nz {
            return None;
        }
        Some((i, j, k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_match_staggered_dims() {
        let g = GridDims::new(4, 5, 6, 0.1, Vec3::ZERO);
        assert_eq!(g.cell_count(), 4 * 5 * 6);
        assert_eq!(g.u_count(), 5 * 5 * 6);
        assert_eq!(g.v_count(), 4 * 6 * 6);
        assert_eq!(g.w_count(), 4 * 5 * 7);
    }

    #[test]
    fn cell_of_maps_positions() {
        let g = GridDims::new(10, 10, 10, 0.1, Vec3::ZERO);
        assert_eq!(g.cell_of(Vec3::new(0.05, 0.05, 0.05)), Some((0, 0, 0)));
        assert_eq!(g.cell_of(Vec3::new(0.95, 0.35, 0.15)), Some((9, 3, 1)));
        assert_eq!(g.cell_of(Vec3::new(-0.1, 0.0, 0.0)), None);
        assert_eq!(g.cell_of(Vec3::new(2.0, 0.0, 0.0)), None);
    }

    #[test]
    fn cell_type_roundtrips() {
        for t in [CellType::Air, CellType::Fluid, CellType::Solid] {
            assert_eq!(CellType::from_code(t.code()), t);
        }
        assert_eq!(CellType::from_code(99), CellType::Air);
    }
}
