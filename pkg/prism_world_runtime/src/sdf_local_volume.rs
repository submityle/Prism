//! Owned, engine-neutral quantized SDF samples on a regular local-space grid.
//!
//! Positions and grid spacing are integer millimeters. Sample distances keep
//! their host-defined quantization units: interpolation never rescales them to
//! millimeters. A far-valued boundary encloses a finite narrow-band influence
//! region; samples outside that region are explicitly far, not unknown.

use alloc::vec::Vec;
use core::fmt;

/// Invalid local SDF geometry, storage, resource budget, or far boundary.
///
/// Axis indices are X = 0, Y = 1, Z = 2. Construction consumes the supplied
/// sample vector even on failure, but never allocates additional sample storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldSdfLocalVolumeError {
    /// An axis has fewer than two grid samples.
    InvalidDimension {
        /// Axis containing the invalid dimension.
        axis: usize,
        /// Rejected number of grid samples on that axis.
        dimension: u32,
    },
    /// Grid spacing must be strictly positive.
    InvalidVoxelSize,
    /// The sample count cannot be represented by this platform's `usize`.
    SampleCountOverflow,
    /// An axis's `(dimension - 1) * voxel_size_mm` exceeds `i64::MAX`.
    ExtentOverflow {
        /// Axis whose maximum local millimeter coordinate cannot be represented.
        axis: usize,
    },
    /// The required sample count exceeds the caller's resource limit.
    SampleBudgetExceeded {
        /// Exact sample count required by the grid dimensions.
        required: usize,
        /// Caller-provided maximum sample count, not a byte count.
        max_samples: usize,
    },
    /// Storage must contain exactly the checked product of all dimensions.
    SampleCountMismatch {
        /// Exact sample count required by the grid dimensions.
        expected: usize,
        /// Length of the supplied sample vector.
        actual: usize,
    },
    /// Every sample on each of the six boundary faces must be `i16::MAX`.
    InvalidBoundarySample {
        /// Grid-sample coordinate of a rejected boundary sample, in X/Y/Z order.
        coordinate: [u32; 3],
        /// Rejected quantized distance.
        value: i16,
    },
}

impl fmt::Display for WorldSdfLocalVolumeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { axis, dimension } => write!(
                f,
                "local SDF dimension on axis {axis} must be at least 2, got {dimension}"
            ),
            Self::InvalidVoxelSize => f.write_str("local SDF voxel spacing must be positive"),
            Self::SampleCountOverflow => {
                f.write_str("local SDF sample count exceeds the platform usize range")
            }
            Self::ExtentOverflow { axis } => {
                write!(f, "local SDF extent on axis {axis} exceeds i64::MAX")
            }
            Self::SampleBudgetExceeded {
                required,
                max_samples,
            } => write!(
                f,
                "local SDF requires {required} samples, exceeding budget {max_samples}"
            ),
            Self::SampleCountMismatch { expected, actual } => write!(
                f,
                "local SDF requires exactly {expected} samples, got {actual}"
            ),
            Self::InvalidBoundarySample { coordinate, value } => write!(
                f,
                "local SDF boundary sample at {coordinate:?} must be i16::MAX, got {value}"
            ),
        }
    }
}

impl core::error::Error for WorldSdfLocalVolumeError {}

/// An immutable, owned local quantized signed-distance volume.
///
/// Samples are X-fastest: `x + dimensions[0] * (y + dimensions[1] * z)`.
/// Grid vertex zero is at local `[0, 0, 0]`; the last vertex is at
/// [`max_local_mm`](Self::max_local_mm). All six boundary faces are explicitly
/// far (`i16::MAX`), and every query outside the grid is also far. Far samples
/// participate numerically in interpolation; they are not missing-data markers.
///
/// The caller defines distance quantization independently of millimeter grid
/// spacing. No distance rescaling, normalization, analytic SDF validation, or
/// Lipschitz validation is performed. In particular, the enforced far shell
/// alone does not guarantee a conservative sphere-tracing distance bound.
///
/// Construction takes ownership without copying or allocating sample storage.
/// It validates only the boundary after constant-space geometry/storage checks,
/// taking O(boundary samples) time. Queries take O(1) time and allocate nothing.
#[derive(Debug)]
pub struct WorldSdfLocalVolume {
    dimensions: [u32; 3],
    voxel_size_mm: u32,
    max_local_mm: [i64; 3],
    strides: [usize; 3],
    samples: Vec<i16>,
}

impl WorldSdfLocalVolume {
    /// Validates and owns a complete X-fastest quantized SDF grid.
    ///
    /// Each dimension must be at least two and spacing must be nonzero. The
    /// checked dimension product must fit `usize`; each extent must fit `i64`.
    /// `max_samples` limits the required number of samples, not bytes or vector
    /// capacity. Storage must have exactly that length. All six faces, including
    /// their edges and corners, must contain only `i16::MAX`.
    ///
    /// Validation order is dimensions, spacing, checked sample count, extents,
    /// sample budget, exact storage length, then boundary contents. Thus budget
    /// rejection happens before reading any samples, even for incomplete input.
    /// The caller still owns responsibility for budgeting the vector's initial
    /// allocation and any excess capacity; this constructor never allocates.
    pub fn new(
        dimensions: [u32; 3],
        voxel_size_mm: u32,
        samples: Vec<i16>,
        max_samples: usize,
    ) -> Result<Self, WorldSdfLocalVolumeError> {
        for (axis, dimension) in dimensions.into_iter().enumerate() {
            if dimension < 2 {
                return Err(WorldSdfLocalVolumeError::InvalidDimension { axis, dimension });
            }
        }
        if voxel_size_mm == 0 {
            return Err(WorldSdfLocalVolumeError::InvalidVoxelSize);
        }

        let mut sizes = [0_usize; 3];
        let mut required = 1_usize;
        for (axis, dimension) in dimensions.into_iter().enumerate() {
            sizes[axis] = usize::try_from(dimension)
                .map_err(|_| WorldSdfLocalVolumeError::SampleCountOverflow)?;
            required = required
                .checked_mul(sizes[axis])
                .ok_or(WorldSdfLocalVolumeError::SampleCountOverflow)?;
        }

        let mut max_local_mm = [0_i64; 3];
        for (axis, dimension) in dimensions.into_iter().enumerate() {
            // Both factors are u32-derived, so their product fits u64.
            let extent = u64::from(dimension - 1) * u64::from(voxel_size_mm);
            max_local_mm[axis] = i64::try_from(extent)
                .map_err(|_| WorldSdfLocalVolumeError::ExtentOverflow { axis })?;
        }
        if required > max_samples {
            return Err(WorldSdfLocalVolumeError::SampleBudgetExceeded {
                required,
                max_samples,
            });
        }
        if samples.len() != required {
            return Err(WorldSdfLocalVolumeError::SampleCountMismatch {
                expected: required,
                actual: samples.len(),
            });
        }

        let volume = Self {
            dimensions,
            voxel_size_mm,
            max_local_mm,
            // These partial products fit because all dimensions are positive
            // and their full product was checked above.
            strides: [1, sizes[0], sizes[0] * sizes[1]],
            samples,
        };
        volume.validate_boundary()?;
        Ok(volume)
    }

    /// Returns the number of samples on each axis, in X/Y/Z order.
    pub fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    /// Returns the strictly positive grid spacing in integer millimeters.
    /// This is not a multiplier for the stored distance quantization units.
    pub fn voxel_size_mm(&self) -> u32 {
        self.voxel_size_mm
    }

    /// Returns the inclusive position of the last grid vertex on each axis.
    /// Each component is `(dimension - 1) * voxel_size_mm`, checked to fit i64.
    pub fn max_local_mm(&self) -> [i64; 3] {
        self.max_local_mm
    }

    /// Samples local integer millimeters using exact trilinear interpolation.
    ///
    /// Positions outside the inclusive `[0, max_local_mm]` box return
    /// `i16::MAX`, including arbitrary `i128` extremes. Boundary queries also
    /// return far, as required by the validated shell. Interior interpolation
    /// treats all eight corner values, including far, as numeric distances in
    /// their original host-defined units.
    ///
    /// Products and sums use i128 without floating point or intermediate
    /// rounding. Only the final result is rounded to the nearest integer, with
    /// exact half values rounded away from zero. The convex combination stays
    /// in the i16 range without clamping. Queries never allocate or mutate.
    pub fn sample(&self, local_position_mm: [i128; 3]) -> i16 {
        for (axis, position) in local_position_mm.into_iter().enumerate() {
            // Check before arithmetic or narrowing, including i128::MIN/MAX.
            // Every point on a face interpolates only validated far samples.
            if position <= 0 || position >= i128::from(self.max_local_mm[axis]) {
                return i16::MAX;
            }
        }

        let spacing = i128::from(self.voxel_size_mm);
        let mut lower = [0_usize; 3];
        let mut weights = [[0_i128; 2]; 3];
        for (axis, position) in local_position_mm.into_iter().enumerate() {
            // Strictly interior coordinates guarantee 0 <= lower < dim - 1.
            lower[axis] = (position / spacing) as usize;
            let remainder = position % spacing;
            weights[axis] = [spacing - remainder, remainder];
        }
        let base = lower[0] + lower[1] * self.strides[1] + lower[2] * self.strides[2];
        let mut numerator = 0_i128;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let index = base + dx + dy * self.strides[1] + dz * self.strides[2];
                    numerator += i128::from(self.samples[index])
                        * weights[0][dx]
                        * weights[1][dy]
                        * weights[2][dz];
                }
            }
        }

        // spacing < 2^32; denominator < 2^96. Nonnegative corner weights
        // sum to denominator, so |numerator| <= 32768*denominator < 2^111.
        // Partial sums and twice the absolute remainder also fit i128.
        let denominator = spacing * spacing * spacing;
        let quotient = numerator / denominator;
        let remainder = numerator % denominator;
        let rounded = if remainder.abs() * 2 >= denominator {
            quotient + numerator.signum()
        } else {
            quotient
        };
        // A rounded convex combination of i16 values remains an i16 value.
        rounded as i16
    }

    fn validate_boundary(&self) -> Result<(), WorldSdfLocalVolumeError> {
        let [nx, ny, nz] = self.dimensions;
        // Visit each boundary vertex exactly once, without scanning the interior.
        for z in [0, nz - 1] {
            for y in 0..ny {
                for x in 0..nx {
                    self.validate_boundary_sample([x, y, z])?;
                }
            }
        }
        for z in 1..nz - 1 {
            for y in [0, ny - 1] {
                for x in 0..nx {
                    self.validate_boundary_sample([x, y, z])?;
                }
            }
            for y in 1..ny - 1 {
                for x in [0, nx - 1] {
                    self.validate_boundary_sample([x, y, z])?;
                }
            }
        }
        Ok(())
    }

    fn validate_boundary_sample(
        &self,
        coordinate: [u32; 3],
    ) -> Result<(), WorldSdfLocalVolumeError> {
        let index = coordinate[0] as usize
            + coordinate[1] as usize * self.strides[1]
            + coordinate[2] as usize * self.strides[2];
        let value = self.samples[index];
        if value != i16::MAX {
            return Err(WorldSdfLocalVolumeError::InvalidBoundarySample { coordinate, value });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
