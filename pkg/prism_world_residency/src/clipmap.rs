//! Deterministic CPU layout planning for an SDF clipmap.
//!
//! [`CpuSdfClipmap`] owns only the integer layout contract for a toroidal
//! clipmap. Each mip is a fixed-size square in the X/Z plane. Its Y
//! coordinate is the floor-scaled Y coordinate of the base-level center.
//! Moving the center returns, for every mip, the coordinates that entered the
//! new square and replaces the current layout atomically.
//!
//! This module deliberately does not allocate GPU resources, upload data, or
//! sample an SDF. It is a `no_std` + `alloc` CPU planning primitive that a
//! renderer, baker, or residency service can consume.

use alloc::vec::Vec;
use core::fmt;
use core::mem::size_of;

use crate::BrickCoord;

/// The coordinate axis that overflowed while constructing a ring.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClipmapAxis {
    /// X coordinate in brick space.
    X,
    /// Y coordinate in brick space.
    Y,
    /// Z coordinate in brick space.
    Z,
}

/// Errors returned by the CPU clipmap layout planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdfClipmapError {
    /// The number of mip levels must be in the supported checked-shift range.
    InvalidMipCount,
    /// A zero-byte update budget cannot represent even the planner's outer
    /// result containers.
    InvalidCapacityBudget,
    /// A size, count, or byte computation overflowed.
    CapacityOverflow,
    /// The requested update would exceed the configured peak allocation
    /// budget. No layout state is changed when this is returned.
    CapacityBudgetExceeded {
        /// Bytes required by the prospective layout and returned update.
        required: usize,
        /// Configured peak allocation budget in bytes.
        budget: usize,
    },
    /// The update center must be expressed at base mip zero.
    InvalidCenterMip {
        /// Mip supplied by the caller.
        actual: u8,
    },
    /// A ring boundary could not be represented by an `i32` brick coordinate.
    CoordinateOverflow {
        /// Mip being constructed.
        mip: u8,
        /// Axis whose boundary or scaled center overflowed.
        axis: ClipmapAxis,
    },
}

impl fmt::Display for SdfClipmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMipCount => f.write_str("invalid SDF clipmap mip count"),
            Self::InvalidCapacityBudget => f.write_str("invalid SDF clipmap capacity budget"),
            Self::CapacityOverflow => f.write_str("SDF clipmap capacity arithmetic overflowed"),
            Self::CapacityBudgetExceeded { required, budget } => write!(
                f,
                "SDF clipmap update requires {required} bytes, budget is {budget} bytes"
            ),
            Self::InvalidCenterMip { actual } => {
                write!(f, "SDF clipmap center must use mip 0, got mip {actual}")
            }
            Self::CoordinateOverflow { mip, axis } => {
                write!(
                    f,
                    "SDF clipmap coordinate overflow at mip {mip} on {axis:?}"
                )
            }
        }
    }
}

/// Fixed geometry and peak allocation budget for one CPU SDF clipmap.
///
/// `ring_radius` is measured in each mip's own brick coordinates. Therefore
/// every mip has the same `(2 * ring_radius + 1)²` X/Z footprint, while a
/// movement of one base-level brick can cross several cells at coarse mips.
/// `max_update_bytes` bounds the prospective replacement layout and returned
/// entered-coordinate vectors. The already-owned old layout is not charged a
/// second time because it remains resident until a successful commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfClipmapConfig {
    /// Number of fixed mip layers.
    pub mip_levels: u8,
    /// Half-width and half-height of every square mip footprint.
    pub ring_radius: u32,
    /// Peak bytes allowed for one prospective layout plus its update result.
    pub max_update_bytes: usize,
}

impl SdfClipmapConfig {
    /// Creates a checked clipmap configuration.
    pub fn new(
        mip_levels: u8,
        ring_radius: u32,
        max_update_bytes: usize,
    ) -> Result<Self, SdfClipmapError> {
        let config = Self {
            mip_levels,
            ring_radius,
            max_update_bytes,
        };
        config.validate()?;
        Ok(config)
    }

    /// Returns the number of cells in one square mip footprint.
    pub fn cells_per_mip(&self) -> Result<usize, SdfClipmapError> {
        let side = self.side()?;
        side.checked_mul(side)
            .ok_or(SdfClipmapError::CapacityOverflow)
    }

    /// Returns the total number of layout cells across all mips.
    pub fn layout_cell_count(&self) -> Result<usize, SdfClipmapError> {
        self.cells_per_mip()?
            .checked_mul(usize::from(self.mip_levels))
            .ok_or(SdfClipmapError::CapacityOverflow)
    }

    fn validate(&self) -> Result<(), SdfClipmapError> {
        if self.mip_levels == 0 || self.mip_levels > MAX_MIP_LEVELS {
            return Err(SdfClipmapError::InvalidMipCount);
        }
        if self.max_update_bytes == 0 {
            return Err(SdfClipmapError::InvalidCapacityBudget);
        }
        let _ = self.side()?;
        let _ = self.cells_per_mip()?;
        let _ = self.layout_cell_count()?;
        Ok(())
    }

    fn side(&self) -> Result<usize, SdfClipmapError> {
        let side = u64::from(self.ring_radius)
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(SdfClipmapError::CapacityOverflow)?;
        usize::try_from(side).map_err(|_| SdfClipmapError::CapacityOverflow)
    }
}

/// Coordinates that entered a clipmap update, grouped by mip.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SdfClipmapUpdate {
    newly_entered: Vec<Vec<BrickCoord>>,
}

impl SdfClipmapUpdate {
    fn empty(mip_levels: u8) -> Result<Self, SdfClipmapError> {
        let mut newly_entered = Vec::new();
        newly_entered
            .try_reserve(usize::from(mip_levels))
            .map_err(|_| SdfClipmapError::CapacityOverflow)?;
        for _ in 0..mip_levels {
            newly_entered.push(Vec::new());
        }
        Ok(Self { newly_entered })
    }

    /// Returns the newly entered coordinates for every mip in ascending mip
    /// order. Each inner vector is sorted by [`BrickCoord`] order and contains
    /// no duplicates.
    pub fn layers(&self) -> &[Vec<BrickCoord>] {
        &self.newly_entered
    }

    /// Returns one mip's newly entered coordinates.
    pub fn layer(&self, mip: u8) -> Option<&[BrickCoord]> {
        self.newly_entered.get(usize::from(mip)).map(Vec::as_slice)
    }

    /// Consumes the update and returns its per-mip vectors.
    pub fn into_layers(self) -> Vec<Vec<BrickCoord>> {
        self.newly_entered
    }

    /// Returns whether no coordinate entered at any mip.
    pub fn is_empty(&self) -> bool {
        self.newly_entered.iter().all(Vec::is_empty)
    }

    /// Returns the number of coordinates across all mips.
    pub fn len(&self) -> usize {
        self.newly_entered.iter().map(Vec::len).sum()
    }
}

/// Deterministic CPU layout planner for fixed-size SDF clipmap rings.
///
/// The planner stores one sorted square X/Z layout per mip. Calling
/// [`Self::update_center`] computes the new layout and its set difference
/// before mutating this value, so checked errors and budget failures leave the
/// previous center and layout unchanged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuSdfClipmap {
    config: SdfClipmapConfig,
    center: Option<BrickCoord>,
    layout: Vec<Vec<BrickCoord>>,
}

impl CpuSdfClipmap {
    /// Creates an uninitialized planner. The first successful center update
    /// reports the complete layout as newly entered.
    pub fn new(config: SdfClipmapConfig) -> Result<Self, SdfClipmapError> {
        config.validate()?;
        let mut layout = Vec::new();
        layout
            .try_reserve(usize::from(config.mip_levels))
            .map_err(|_| SdfClipmapError::CapacityOverflow)?;
        Ok(Self {
            config,
            center: None,
            layout,
        })
    }

    /// Creates a planner and performs its first center update.
    pub fn with_center(
        config: SdfClipmapConfig,
        center: BrickCoord,
    ) -> Result<(Self, SdfClipmapUpdate), SdfClipmapError> {
        let mut clipmap = Self::new(config)?;
        let update = clipmap.update_center(center)?;
        Ok((clipmap, update))
    }

    /// Returns the fixed clipmap configuration.
    pub const fn config(&self) -> SdfClipmapConfig {
        self.config
    }

    /// Returns the last successfully committed base-level center.
    pub const fn center(&self) -> Option<BrickCoord> {
        self.center
    }

    /// Returns the currently committed layout grouped by ascending mip.
    ///
    /// Each layer is sorted by [`BrickCoord`] order and contains exactly the
    /// configured square footprint once the planner has been initialized.
    pub fn layout(&self) -> &[Vec<BrickCoord>] {
        &self.layout
    }

    /// Moves the base-level center and returns coordinates newly entering each
    /// mip's square ring.
    ///
    /// The input must be a mip-zero [`BrickCoord`]. Mip centers are obtained by
    /// mathematical floor division, including for negative coordinates. The
    /// operation is transactional: coordinate, capacity, or allocation errors
    /// do not change the previous layout.
    pub fn update_center(
        &mut self,
        center: BrickCoord,
    ) -> Result<SdfClipmapUpdate, SdfClipmapError> {
        validate_center(center)?;
        if self.center == Some(center) {
            return SdfClipmapUpdate::empty(self.config.mip_levels);
        }

        let geometry = Geometry::from_config(self.config)?;
        let mut entered_counts = Vec::new();
        entered_counts
            .try_reserve(usize::from(self.config.mip_levels))
            .map_err(|_| SdfClipmapError::CapacityOverflow)?;
        let mut entered_total = 0_usize;

        for mip in 0..self.config.mip_levels {
            let bounds = LayerBounds::new(center, mip, self.config.ring_radius, geometry.side)?;
            let old_layer = self.layout.get(usize::from(mip));
            let entered = match old_layer {
                Some(old_layer) => count_entered(bounds, mip, old_layer, geometry.side)?,
                None => geometry.cells_per_mip,
            };
            entered_total = entered_total
                .checked_add(entered)
                .ok_or(SdfClipmapError::CapacityOverflow)?;
            entered_counts.push(entered);
        }

        let required = update_bytes(geometry.layout_cells, entered_total, self.config.mip_levels)?;
        if required > self.config.max_update_bytes {
            return Err(SdfClipmapError::CapacityBudgetExceeded {
                required,
                budget: self.config.max_update_bytes,
            });
        }

        let mut new_layout = Vec::new();
        new_layout
            .try_reserve(usize::from(self.config.mip_levels))
            .map_err(|_| SdfClipmapError::CapacityOverflow)?;
        let mut newly_entered = Vec::new();
        newly_entered
            .try_reserve(usize::from(self.config.mip_levels))
            .map_err(|_| SdfClipmapError::CapacityOverflow)?;

        for mip in 0..self.config.mip_levels {
            let bounds = LayerBounds::new(center, mip, self.config.ring_radius, geometry.side)?;
            let layer = build_layer(bounds, mip, geometry.side, geometry.cells_per_mip)?;
            let old_layer = self.layout.get(usize::from(mip));
            let capacity = entered_counts[usize::from(mip)];
            let mut entered = Vec::new();
            entered
                .try_reserve(capacity)
                .map_err(|_| SdfClipmapError::CapacityOverflow)?;
            match old_layer {
                Some(old_layer) => {
                    for coord in &layer {
                        if old_layer.binary_search(coord).is_err() {
                            entered.push(*coord);
                        }
                    }
                }
                None => entered.extend(layer.iter().copied()),
            }
            debug_assert_eq!(entered.len(), capacity);
            newly_entered.push(entered);
            new_layout.push(layer);
        }

        self.center = Some(center);
        self.layout = new_layout;
        Ok(SdfClipmapUpdate { newly_entered })
    }
}

const MAX_MIP_LEVELS: u8 = 63;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Geometry {
    side: usize,
    cells_per_mip: usize,
    layout_cells: usize,
}

impl Geometry {
    fn from_config(config: SdfClipmapConfig) -> Result<Self, SdfClipmapError> {
        let side = config.side()?;
        let cells_per_mip = side
            .checked_mul(side)
            .ok_or(SdfClipmapError::CapacityOverflow)?;
        let layout_cells = cells_per_mip
            .checked_mul(usize::from(config.mip_levels))
            .ok_or(SdfClipmapError::CapacityOverflow)?;
        Ok(Self {
            side,
            cells_per_mip,
            layout_cells,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LayerBounds {
    min_x: i32,
    max_x: i32,
    y: i32,
    min_z: i32,
    max_z: i32,
}

impl LayerBounds {
    fn new(center: BrickCoord, mip: u8, radius: u32, side: usize) -> Result<Self, SdfClipmapError> {
        let center_x =
            floor_div_pow2(center.x, mip).ok_or(SdfClipmapError::CoordinateOverflow {
                mip,
                axis: ClipmapAxis::X,
            })?;
        let y = floor_div_pow2(center.y, mip).ok_or(SdfClipmapError::CoordinateOverflow {
            mip,
            axis: ClipmapAxis::Y,
        })?;
        let center_z =
            floor_div_pow2(center.z, mip).ok_or(SdfClipmapError::CoordinateOverflow {
                mip,
                axis: ClipmapAxis::Z,
            })?;
        let radius = i64::from(radius);
        let last = i64::try_from(side - 1).map_err(|_| SdfClipmapError::CapacityOverflow)?;
        let min_x =
            i64::from(center_x)
                .checked_sub(radius)
                .ok_or(SdfClipmapError::CoordinateOverflow {
                    mip,
                    axis: ClipmapAxis::X,
                })?;
        let max_x = min_x
            .checked_add(last)
            .ok_or(SdfClipmapError::CoordinateOverflow {
                mip,
                axis: ClipmapAxis::X,
            })?;
        let min_z =
            i64::from(center_z)
                .checked_sub(radius)
                .ok_or(SdfClipmapError::CoordinateOverflow {
                    mip,
                    axis: ClipmapAxis::Z,
                })?;
        let max_z = min_z
            .checked_add(last)
            .ok_or(SdfClipmapError::CoordinateOverflow {
                mip,
                axis: ClipmapAxis::Z,
            })?;
        Ok(Self {
            min_x: checked_i32(min_x, mip, ClipmapAxis::X)?,
            max_x: checked_i32(max_x, mip, ClipmapAxis::X)?,
            y,
            min_z: checked_i32(min_z, mip, ClipmapAxis::Z)?,
            max_z: checked_i32(max_z, mip, ClipmapAxis::Z)?,
        })
    }
}

fn validate_center(center: BrickCoord) -> Result<(), SdfClipmapError> {
    if center.mip == 0 {
        Ok(())
    } else {
        Err(SdfClipmapError::InvalidCenterMip { actual: center.mip })
    }
}

fn floor_div_pow2(value: i32, mip: u8) -> Option<i32> {
    let divisor = 1_i64.checked_shl(u32::from(mip))?;
    let value = i64::from(value);
    let quotient = value / divisor;
    let remainder = value % divisor;
    let floored = if remainder < 0 {
        quotient.checked_sub(1)?
    } else {
        quotient
    };
    i32::try_from(floored).ok()
}

fn checked_i32(value: i64, mip: u8, axis: ClipmapAxis) -> Result<i32, SdfClipmapError> {
    i32::try_from(value).map_err(|_| SdfClipmapError::CoordinateOverflow { mip, axis })
}

fn coord_at(
    bounds: LayerBounds,
    mip: u8,
    x_index: usize,
    z_index: usize,
) -> Result<BrickCoord, SdfClipmapError> {
    let x = i64::from(bounds.min_x)
        .checked_add(i64::try_from(x_index).map_err(|_| SdfClipmapError::CapacityOverflow)?)
        .ok_or(SdfClipmapError::CoordinateOverflow {
            mip,
            axis: ClipmapAxis::X,
        })?;
    let z = i64::from(bounds.min_z)
        .checked_add(i64::try_from(z_index).map_err(|_| SdfClipmapError::CapacityOverflow)?)
        .ok_or(SdfClipmapError::CoordinateOverflow {
            mip,
            axis: ClipmapAxis::Z,
        })?;
    Ok(BrickCoord::new(
        checked_i32(x, mip, ClipmapAxis::X)?,
        bounds.y,
        checked_i32(z, mip, ClipmapAxis::Z)?,
        mip,
    ))
}

fn build_layer(
    bounds: LayerBounds,
    mip: u8,
    side: usize,
    capacity: usize,
) -> Result<Vec<BrickCoord>, SdfClipmapError> {
    let mut layer = Vec::new();
    layer
        .try_reserve(capacity)
        .map_err(|_| SdfClipmapError::CapacityOverflow)?;
    for z_index in 0..side {
        for x_index in 0..side {
            layer.push(coord_at(bounds, mip, x_index, z_index)?);
        }
    }
    // `sort` is stable and makes the contract independent of loop traversal.
    layer.sort();
    Ok(layer)
}

fn count_entered(
    bounds: LayerBounds,
    mip: u8,
    old_layer: &[BrickCoord],
    side: usize,
) -> Result<usize, SdfClipmapError> {
    let mut count = 0_usize;
    for z_index in 0..side {
        for x_index in 0..side {
            let coord = coord_at(bounds, mip, x_index, z_index)?;
            if old_layer.binary_search(&coord).is_err() {
                count = count
                    .checked_add(1)
                    .ok_or(SdfClipmapError::CapacityOverflow)?;
            }
        }
    }
    Ok(count)
}

fn update_bytes(
    layout_cells: usize,
    entered_cells: usize,
    mip_levels: u8,
) -> Result<usize, SdfClipmapError> {
    let outer_bytes = usize::from(mip_levels)
        .checked_mul(size_of::<Vec<BrickCoord>>())
        .ok_or(SdfClipmapError::CapacityOverflow)?;
    let coordinate_bytes = layout_cells
        .checked_add(entered_cells)
        .and_then(|cells| cells.checked_mul(size_of::<BrickCoord>()))
        .ok_or(SdfClipmapError::CapacityOverflow)?;
    outer_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(coordinate_bytes))
        .ok_or(SdfClipmapError::CapacityOverflow)
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use alloc::vec;

    use super::*;

    fn config(mip_levels: u8, radius: u32) -> SdfClipmapConfig {
        SdfClipmapConfig::new(mip_levels, radius, 1_000_000).unwrap()
    }

    fn coords(layer: &[BrickCoord]) -> BTreeSet<BrickCoord> {
        layer.iter().copied().collect()
    }

    #[test]
    fn negative_centers_use_mathematical_floor_division() {
        let (clipmap, update) =
            CpuSdfClipmap::with_center(config(3, 0), BrickCoord::new(-3, -3, -3, 0)).unwrap();
        assert_eq!(clipmap.layout()[0], vec![BrickCoord::new(-3, -3, -3, 0)]);
        assert_eq!(clipmap.layout()[1], vec![BrickCoord::new(-2, -2, -2, 1)]);
        assert_eq!(clipmap.layout()[2], vec![BrickCoord::new(-1, -1, -1, 2)]);
        assert_eq!(update.len(), 3);
    }

    #[test]
    fn same_center_returns_empty_for_every_mip() {
        let (mut clipmap, first) =
            CpuSdfClipmap::with_center(config(2, 1), BrickCoord::new(4, 0, -2, 0)).unwrap();
        assert!(!first.is_empty());
        let before = clipmap.layout().to_vec();
        let update = clipmap.update_center(BrickCoord::new(4, 0, -2, 0)).unwrap();
        assert!(update.is_empty());
        assert_eq!(update.layers(), &[Vec::new(), Vec::new()]);
        assert_eq!(clipmap.layout(), before.as_slice());
    }

    #[test]
    fn mip_scaling_keeps_fixed_square_footprints() {
        let (clipmap, _) =
            CpuSdfClipmap::with_center(config(3, 1), BrickCoord::new(5, 7, 9, 0)).unwrap();
        assert_eq!(
            clipmap.layout().iter().map(Vec::len).collect::<Vec<_>>(),
            vec![9, 9, 9]
        );
        assert!(clipmap.layout()[1].iter().all(|coord| coord.y == 3));
        assert!(clipmap.layout()[2].iter().all(|coord| coord.y == 1));
        assert!(clipmap.layout()[1].contains(&BrickCoord::new(1, 3, 3, 1)));
        assert!(clipmap.layout()[2].contains(&BrickCoord::new(0, 1, 1, 2)));
    }

    #[test]
    fn stable_order_has_no_duplicates_and_large_displacement_is_bounded() {
        let (mut clipmap, _) = CpuSdfClipmap::with_center(
            config(2, 1),
            BrickCoord::new(i32::MIN + 1, 0, i32::MAX - 1, 0),
        )
        .unwrap();
        let update = clipmap
            .update_center(BrickCoord::new(i32::MAX - 1, 0, i32::MIN + 1, 0))
            .unwrap();
        for layer in update.layers() {
            assert!(layer.windows(2).all(|window| window[0] < window[1]));
            assert_eq!(layer.len(), coords(layer).len());
        }
        let repeat = clipmap
            .update_center(BrickCoord::new(i32::MAX - 1, 0, i32::MIN + 1, 0))
            .unwrap();
        assert!(repeat.is_empty());
    }

    #[test]
    fn boundary_ring_overflow_is_explicit() {
        let (mut clipmap, _) =
            CpuSdfClipmap::with_center(config(1, 0), BrickCoord::new(i32::MIN, 0, i32::MAX, 0))
                .unwrap();
        let before = clipmap.layout().to_vec();
        let error = clipmap
            .update_center(BrickCoord::new(i32::MIN, 0, i32::MAX, 0))
            .unwrap();
        assert!(error.is_empty());
        assert_eq!(clipmap.layout(), before.as_slice());

        let mut clipmap = CpuSdfClipmap::new(config(1, 1)).unwrap();
        let error = clipmap
            .update_center(BrickCoord::new(i32::MIN, 0, i32::MAX, 0))
            .unwrap_err();
        assert_eq!(
            error,
            SdfClipmapError::CoordinateOverflow {
                mip: 0,
                axis: ClipmapAxis::X
            }
        );
        assert!(clipmap.layout().is_empty());
    }

    #[test]
    fn budget_failure_does_not_change_layout() {
        let generous = config(2, 1);
        let required = update_bytes(18, 18, 2).unwrap();
        let constrained = SdfClipmapConfig::new(2, 1, required - 1).unwrap();
        let mut clipmap = CpuSdfClipmap::new(constrained).unwrap();
        let error = clipmap
            .update_center(BrickCoord::new(0, 0, 0, 0))
            .unwrap_err();
        assert_eq!(
            error,
            SdfClipmapError::CapacityBudgetExceeded {
                required,
                budget: required - 1
            }
        );
        assert_eq!(clipmap.center(), None);
        assert!(clipmap.layout().is_empty());

        let (mut clipmap, _) =
            CpuSdfClipmap::with_center(generous, BrickCoord::new(0, 0, 0, 0)).unwrap();
        let before_center = clipmap.center();
        let before_layout = clipmap.layout().to_vec();
        let too_small = SdfClipmapConfig::new(2, 1, 1).unwrap();
        clipmap.config = too_small;
        let error = clipmap
            .update_center(BrickCoord::new(10, 0, 10, 0))
            .unwrap_err();
        assert!(matches!(
            error,
            SdfClipmapError::CapacityBudgetExceeded { .. }
        ));
        assert_eq!(clipmap.center(), before_center);
        assert_eq!(clipmap.layout(), before_layout.as_slice());
    }

    #[test]
    fn invalid_config_and_center_are_explicit() {
        assert_eq!(
            SdfClipmapConfig::new(0, 1, 1).unwrap_err(),
            SdfClipmapError::InvalidMipCount
        );
        assert_eq!(
            SdfClipmapConfig::new(1, 1, 0).unwrap_err(),
            SdfClipmapError::InvalidCapacityBudget
        );
        let mut clipmap = CpuSdfClipmap::new(config(1, 0)).unwrap();
        assert_eq!(
            clipmap.update_center(BrickCoord::new(0, 0, 0, 1)),
            Err(SdfClipmapError::InvalidCenterMip { actual: 1 })
        );
    }
}
