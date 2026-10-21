//! Integer world-space focus tracking for the derived SDF clipmap.

use core::fmt;
use prism_world_residency::{BrickCoord, ClipmapAxis};

/// Fixed mapping from global millimeters to mip-zero brick coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfFocusConfig {
    /// Global millimeter position of voxel/brick zero.
    pub origin_mm: [i64; 3],
    /// Positive size of one mip-zero voxel in millimeters.
    pub mip0_voxel_size_mm: u32,
    /// Samples per brick axis, in 1..=32, matching the source grid and ring.
    pub brick_size: u8,
    /// Per-axis deadband extending both ends of the current brick interval.
    /// Each component must be strictly less than one brick's extent.
    pub hysteresis_mm: [u64; 3],
}

impl WorldSdfFocusConfig {
    /// Validates the entire mapping without allocating or creating a planner.
    pub fn new(
        origin_mm: [i64; 3],
        mip0_voxel_size_mm: u32,
        brick_size: u8,
        hysteresis_mm: [u64; 3],
    ) -> Result<Self, WorldSdfFocusError> {
        let config = Self {
            origin_mm,
            mip0_voxel_size_mm,
            brick_size,
            hysteresis_mm,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(self) -> Result<(), WorldSdfFocusError> {
        if self.mip0_voxel_size_mm == 0 || !(1..=32).contains(&self.brick_size) {
            return Err(WorldSdfFocusError::InvalidGeometry);
        }
        let extent = self.brick_extent_mm();
        for (axis, margin) in AXES.into_iter().zip(self.hysteresis_mm) {
            if margin >= extent {
                return Err(WorldSdfFocusError::InvalidHysteresis { axis });
            }
        }
        Ok(())
    }

    /// Brick extent computed in u64, never in the narrower voxel-size type.
    /// For validated geometry the maximum is `u32::MAX * 32`, without overflow.
    pub fn brick_extent_mm(self) -> u64 {
        u64::from(self.mip0_voxel_size_mm) * u64::from(self.brick_size)
    }

    /// Ensures the actual focus brick stays inside an XZ window with one Y slice.
    /// Y cannot lag its raw coordinate; X/Z may lag by one brick only if the
    /// window has radius at least one. Does not validate planner budgets.
    pub fn validate_single_slice_coverage(
        self,
        ring_radius: u32,
    ) -> Result<(), WorldSdfFocusError> {
        self.validate()?;
        for (i, axis) in AXES.into_iter().enumerate() {
            if self.hysteresis_mm[i] != 0 && (i == 1 || ring_radius == 0) {
                return Err(WorldSdfFocusError::InvalidCoverage { axis });
            }
        }
        Ok(())
    }
}

const AXES: [ClipmapAxis; 3] = [ClipmapAxis::X, ClipmapAxis::Y, ClipmapAxis::Z];

/// A rejected focus mapping. No previous center is changed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldSdfFocusError {
    /// Voxel size is zero or brick size is outside 1..=32.
    InvalidGeometry,
    /// A deadband is at least one brick wide.
    InvalidHysteresis {
        /// Axis containing the invalid deadband.
        axis: ClipmapAxis,
    },
    /// A single-Y-slice window cannot cover the actual focus with this deadband.
    InvalidCoverage {
        /// Y with any margin, or X/Z with margin and zero window radius.
        axis: ClipmapAxis,
    },
    /// The raw world position maps outside the signed i32 brick domain.
    /// This is checked before applying hysteresis, so deadbands cannot hide it.
    CoordinateOverflow {
        /// Axis outside the representable domain.
        axis: ClipmapAxis,
    },
}

impl fmt::Display for WorldSdfFocusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry => f.write_str("invalid SDF focus voxel/brick geometry"),
            Self::InvalidHysteresis { axis } => write!(f, "invalid SDF focus deadband on {axis:?}"),
            Self::InvalidCoverage { axis } => {
                write!(f, "SDF focus deadband leaves window on {axis:?}")
            }
            Self::CoordinateOverflow { axis } => {
                write!(f, "SDF focus exceeds brick domain on {axis:?}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WorldSdfFocusError {}

/// A mapping or downstream publication failure; neither commits the new focus.
#[derive(Debug, PartialEq)]
pub enum WorldSdfFocusUpdateError<E> {
    /// Input coordinate mapping failed before the downstream callback.
    Focus(WorldSdfFocusError),
    /// Scheduler or GPU publication failed.
    Apply(E),
}

impl<E: fmt::Display> fmt::Display for WorldSdfFocusUpdateError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Focus(error) => error.fmt(f),
            Self::Apply(error) => write!(f, "SDF focus publication: {error}"),
        }
    }
}

#[cfg(feature = "std")]
impl<E: std::error::Error + 'static> std::error::Error for WorldSdfFocusUpdateError<E> {}

/// The committed center and optional downstream result of a focus update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfFocusUpdate<T> {
    /// Committed mip-zero center after this successful update.
    pub center: BrickCoord,
    /// None means the center was retained; no callback, planner or upload ran.
    pub applied: Option<T>,
}

/// Engine-neutral, allocation-free, O(1) world-mm focus controller.
///
/// A retained axis occupies `[center*extent-h, (center+1)*extent+h)` in
/// origin-relative millimeters. Axes leaving that interval independently snap
/// to floor-divided raw brick coordinates. The first update has no deadband.
/// All raw axes are checked before hysteresis; i128 subtraction handles even
/// opposite i64 extrema without overflow. Clipmap radius/budget validation
/// remains the downstream planner's responsibility.
pub struct WorldSdfFocusDriver {
    config: WorldSdfFocusConfig,
    center: Option<BrickCoord>,
}

impl WorldSdfFocusDriver {
    /// Validates even public configuration literals before storing the mapping.
    pub fn new(config: WorldSdfFocusConfig) -> Result<Self, WorldSdfFocusError> {
        config.validate()?;
        Ok(Self {
            config,
            center: None,
        })
    }

    /// Returns the immutable world/voxel mapping.
    pub fn config(&self) -> WorldSdfFocusConfig {
        self.config
    }

    /// Returns the last successfully published center, if initialized.
    pub fn center(&self) -> Option<BrickCoord> {
        self.center
    }

    fn target(&self, position_mm: [i64; 3]) -> Result<BrickCoord, WorldSdfFocusError> {
        let extent = i128::from(self.config.brick_extent_mm());
        let mut relative = [0_i128; 3];
        let mut target = [0_i32; 3];
        for i in 0..3 {
            relative[i] = i128::from(position_mm[i]) - i128::from(self.config.origin_mm[i]);
            target[i] = i32::try_from(relative[i].div_euclid(extent))
                .map_err(|_| WorldSdfFocusError::CoordinateOverflow { axis: AXES[i] })?;
        }
        if let Some(center) = self.center {
            for (i, current) in [center.x, center.y, center.z].into_iter().enumerate() {
                let margin = i128::from(self.config.hysteresis_mm[i]);
                let lower = i128::from(current) * extent - margin;
                let upper = (i128::from(current) + 1) * extent + margin;
                if (lower..upper).contains(&relative[i]) {
                    target[i] = current;
                }
            }
        }
        Ok(BrickCoord::new(target[0], target[1], target[2], 0))
    }

    /// Applies a changed center, committing focus only after callback success.
    ///
    /// The callback must itself preserve its old state on returned errors and
    /// must exclusively own its matching downstream planner. A panic also
    /// leaves focus unchanged, but cannot roll back callback-owned mutations.
    /// Use the GPU focus owner for enforced paired ownership. No callback runs
    /// on retained focus; pending dirty work must still be serviced separately.
    pub fn update_with<T, E>(
        &mut self,
        position_mm: [i64; 3],
        apply: impl FnOnce(BrickCoord) -> Result<T, E>,
    ) -> Result<WorldSdfFocusUpdate<T>, WorldSdfFocusUpdateError<E>> {
        let center = self
            .target(position_mm)
            .map_err(WorldSdfFocusUpdateError::Focus)?;
        let applied = if self.center == Some(center) {
            None
        } else {
            let output = apply(center).map_err(WorldSdfFocusUpdateError::Apply)?;
            self.center = Some(center);
            Some(output)
        };
        Ok(WorldSdfFocusUpdate { center, applied })
    }
}

#[cfg(test)]
mod tests;
