//! Bounded uniform-grid configuration and error type.
//!
//! [`GridConfig`] pins a finite axis-aligned lattice: an `origin`, a uniform
//! `cell_size`, and an integer resolution `dims` = `(nx, ny, nz)`. Unlike the
//! spatial-hash broad phase, cells are addressed by a *dense linear index*
//! `x + y * nx + z * nx * ny` with no hashing, so there are no collisions and a
//! particle maps to exactly one cell. Points that fall outside the lattice are
//! clamped onto the boundary cell along each axis.
//!
//! The linear index doubles as the 32-bit sort key consumed by the radix sort,
//! so [`GridConfig::validate`] rejects a resolution whose cell count would not
//! fit in a `u32`.
//!
//! # Provenance
//!
//! Standard bounded uniform-grid parameters (Green, "Particle Simulation using
//! CUDA", NVIDIA 2008). No Unreal Engine source or derived code.

use glam::Vec3;

/// A finite, dense, axis-aligned uniform grid over a bounded region of space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridConfig {
    /// World-space position of the grid's minimum corner (cell `(0, 0, 0)`).
    pub origin: Vec3,
    /// Edge length of every cubic cell. Must be strictly positive and finite.
    pub cell_size: f32,
    /// Cell counts along `x`, `y`, and `z`. Each must be strictly positive.
    pub dims: [u32; 3],
}

impl GridConfig {
    /// Creates a configuration from an `origin`, a `cell_size`, and a per-axis
    /// resolution `dims` = `(nx, ny, nz)`.
    #[must_use]
    pub fn new(origin: Vec3, cell_size: f32, dims: [u32; 3]) -> GridConfig {
        GridConfig {
            origin,
            cell_size,
            dims,
        }
    }

    /// Total number of cells, `nx * ny * nz`, computed in `u64` so the product
    /// cannot overflow before [`GridConfig::validate`] range-checks it.
    #[must_use]
    pub fn num_cells(&self) -> u64 {
        let [nx, ny, nz] = self.dims;
        u64::from(nx) * u64::from(ny) * u64::from(nz)
    }

    /// Validates the configuration, returning the first violated invariant.
    ///
    /// # Errors
    ///
    /// Returns [`GridError::InvalidConfig`] when `cell_size` is not strictly
    /// positive and finite, when any axis resolution is zero, or when the total
    /// cell count does not fit in a `u32` (it is used as the radix sort key).
    pub fn validate(&self) -> Result<(), GridError> {
        if !self.cell_size.is_finite() || self.cell_size <= 0.0 {
            return Err(GridError::InvalidConfig(
                "cell_size must be positive and finite",
            ));
        }
        let [nx, ny, nz] = self.dims;
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(GridError::InvalidConfig(
                "every axis resolution must be non-zero",
            ));
        }
        if self.num_cells() > u64::from(u32::MAX) {
            return Err(GridError::InvalidConfig(
                "cell count must fit in a u32 to serve as the sort key",
            ));
        }
        Ok(())
    }
}

/// Errors the uniform-grid builder can report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridError {
    /// A configuration invariant was violated; carries a static reason.
    InvalidConfig(&'static str),
}

impl core::fmt::Display for GridError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GridError::InvalidConfig(reason) => {
                write!(f, "invalid uniform-grid config: {reason}")
            }
        }
    }
}

impl core::error::Error for GridError {}

#[cfg(test)]
mod tests {
    use super::{GridConfig, GridError};
    use glam::Vec3;

    #[test]
    fn num_cells_multiplies_the_axes() {
        let config = GridConfig::new(Vec3::ZERO, 1.0, [4, 5, 6]);
        assert_eq!(config.num_cells(), 120);
    }

    #[test]
    fn valid_config_passes() {
        let config = GridConfig::new(Vec3::new(-1.0, -2.0, -3.0), 0.5, [8, 8, 8]);
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn rejects_non_positive_cell_size() {
        let config = GridConfig::new(Vec3::ZERO, 0.0, [2, 2, 2]);
        assert!(matches!(
            config.validate(),
            Err(GridError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_zero_axis() {
        let config = GridConfig::new(Vec3::ZERO, 1.0, [4, 0, 4]);
        assert!(matches!(
            config.validate(),
            Err(GridError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_cell_count_exceeding_u32() {
        let config = GridConfig::new(Vec3::ZERO, 1.0, [2 << 15, 2 << 15, 4]);
        assert!(matches!(
            config.validate(),
            Err(GridError::InvalidConfig(_))
        ));
    }
}
