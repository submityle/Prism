use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

/// Integer coordinate of one sparse SDF brick.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BrickCoord {
    /// Brick coordinate on the X axis.
    pub x: i32,
    /// Brick coordinate on the Y axis.
    pub y: i32,
    /// Brick coordinate on the Z axis.
    pub z: i32,
    /// Clipmap mip level.
    pub mip: u8,
}

impl BrickCoord {
    /// Creates a coordinate directly.
    pub const fn new(x: i32, y: i32, z: i32, mip: u8) -> Self {
        Self { x, y, z, mip }
    }
}

/// Errors from sparse SDF allocation and access.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdfError {
    /// Brick resolution must be non-zero and fit the bounded index arithmetic.
    InvalidBrickSize,
    /// Mip count must be non-zero.
    InvalidMipCount,
    /// A coordinate refers to a mip not owned by this grid.
    InvalidMip,
    /// The inserted sample count does not match the configured brick volume.
    SampleCount {
        /// Required number of samples.
        expected: usize,
        /// Number of samples supplied by the caller.
        actual: usize,
    },
    /// The dirty region minimum is greater than its maximum.
    InvalidRegion,
    /// The requested export exceeds its caller-provided bound.
    ExportBudgetExceeded {
        /// Export item being bounded.
        resource: SdfExportResource,
        /// Number of items required by the export.
        requested: usize,
        /// Maximum number of items allowed by the export budget.
        limit: usize,
    },
    /// An export offset or count does not fit the packed `u32` ABI.
    ExportIndexOverflow,
}

/// Export resource whose caller-provided bound was exceeded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdfExportResource {
    /// Allocated brick metadata records.
    Bricks,
    /// Quantized brick samples.
    Samples,
}

/// Caller-owned bounds for a deterministic sparse-brick export.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfExportBudget {
    /// Maximum number of allocated bricks to export.
    pub max_bricks: usize,
    /// Maximum number of packed samples to export.
    pub max_samples: usize,
}

impl SdfExportBudget {
    /// Creates an export budget from brick and sample limits.
    pub const fn new(max_bricks: usize, max_samples: usize) -> Self {
        Self {
            max_bricks,
            max_samples,
        }
    }
}

/// Packed metadata for one brick in [`SdfBrickExport`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfExportBrick {
    /// Coordinate of the exported brick.
    pub coord: BrickCoord,
    /// Start index into [`SdfBrickExport::samples`].
    pub sample_offset: u32,
    /// Number of samples owned by this brick.
    pub sample_count: u32,
}

/// Deterministic CPU export suitable for a future GPU upload adapter.
///
/// Metadata follows [`BrickCoord`] order from the grid's `BTreeMap`. Samples
/// retain the grid's z-major, y-next, x-fastest layout. This type contains no
/// device handle and does not claim that a GPU sampling path has run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SdfBrickExport {
    bricks: Vec<SdfExportBrick>,
    samples: Vec<i16>,
}

impl SdfBrickExport {
    /// Returns packed brick metadata in deterministic order.
    pub fn bricks(&self) -> &[SdfExportBrick] {
        &self.bricks
    }

    /// Returns packed quantized samples.
    pub fn samples(&self) -> &[i16] {
        &self.samples
    }

    /// Returns the number of exported bricks.
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }

    /// Returns the number of exported samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Returns the packed sample payload size in bytes.
    pub fn sample_bytes(&self) -> usize {
        self.samples.len() * core::mem::size_of::<i16>()
    }
}

/// Result of a read-only SDF consumer query.
///
/// `Value` means that the requested data is resident and the returned value is
/// valid. `Miss` means that the query was fully resident, but the requested
/// condition was not present. `Unknown` means that at least one brick needed
/// to answer the query is not resident; it must not be treated as an empty
/// field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdfQueryResult<T> {
    /// The query produced a resident value.
    Value(T),
    /// The query was fully known and did not match.
    Miss,
    /// The query cannot be decided because required data is missing.
    Unknown,
}

/// Conservative quantized bounds returned by a surface-region query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfBounds {
    /// Minimum stored sample in the queried brick coverage.
    pub min: i16,
    /// Maximum stored sample in the queried brick coverage.
    pub max: i16,
}

/// Read-only query contract shared by SDF consumers.
pub trait SdfQueryConsumer {
    /// Samples one voxel. A missing brick returns [`SdfQueryResult::Unknown`].
    fn query_sample(&self, voxel: [i32; 3], mip: u8) -> Result<SdfQueryResult<i16>, SdfError>;

    /// Tests whether the conservative SDF bounds for an inclusive region
    /// contain zero. A complete non-intersecting region returns `Miss`; a
    /// region with missing bricks returns `Unknown`.
    fn query_surface_region(
        &self,
        min_voxel: [i32; 3],
        max_voxel: [i32; 3],
        mip: u8,
    ) -> Result<SdfQueryResult<SdfBounds>, SdfError>;
}

/// A copyable, read-only view over one [`SparseSdfBrickGrid`].
///
/// Consumers can create independent views over the same grid without taking
/// ownership or mutating residency state. The view deliberately exposes only
/// query operations through [`SdfQueryConsumer`].
#[derive(Clone, Copy, Debug)]
pub struct SparseSdfQuery<'a> {
    grid: &'a SparseSdfBrickGrid,
}

impl<'a> SparseSdfQuery<'a> {
    /// Creates a read-only query view over a sparse grid.
    pub const fn new(grid: &'a SparseSdfBrickGrid) -> Self {
        Self { grid }
    }
}

impl SdfQueryConsumer for SparseSdfQuery<'_> {
    fn query_sample(&self, voxel: [i32; 3], mip: u8) -> Result<SdfQueryResult<i16>, SdfError> {
        Ok(match self.grid.sample(voxel, mip)? {
            Some(value) => SdfQueryResult::Value(value),
            None => SdfQueryResult::Unknown,
        })
    }

    fn query_surface_region(
        &self,
        min_voxel: [i32; 3],
        max_voxel: [i32; 3],
        mip: u8,
    ) -> Result<SdfQueryResult<SdfBounds>, SdfError> {
        self.grid.validate_mip(mip)?;
        if min_voxel[0] > max_voxel[0] || min_voxel[1] > max_voxel[1] || min_voxel[2] > max_voxel[2]
        {
            return Err(SdfError::InvalidRegion);
        }

        let size = i32::from(self.grid.brick_size);
        let min = [
            floor_div(min_voxel[0], size),
            floor_div(min_voxel[1], size),
            floor_div(min_voxel[2], size),
        ];
        let max = [
            floor_div(max_voxel[0], size),
            floor_div(max_voxel[1], size),
            floor_div(max_voxel[2], size),
        ];
        let mut bounds: Option<(i16, i16)> = None;
        let mut z = min[2];
        loop {
            let mut y = min[1];
            loop {
                let mut x = min[0];
                loop {
                    let coord = BrickCoord::new(x, y, z, mip);
                    let Some((local_min, local_max)) = self.grid.min_max(coord)? else {
                        return Ok(SdfQueryResult::Unknown);
                    };
                    bounds = Some(match bounds {
                        Some((current_min, current_max)) => {
                            (current_min.min(local_min), current_max.max(local_max))
                        }
                        None => (local_min, local_max),
                    });
                    if x == max[0] {
                        break;
                    }
                    x += 1;
                }
                if y == max[1] {
                    break;
                }
                y += 1;
            }
            if z == max[2] {
                break;
            }
            z += 1;
        }

        let Some((min, max)) = bounds else {
            return Ok(SdfQueryResult::Unknown);
        };
        let result = SdfBounds { min, max };
        if min <= 0 && max >= 0 {
            Ok(SdfQueryResult::Value(result))
        } else {
            Ok(SdfQueryResult::Miss)
        }
    }
}

impl fmt::Display for SdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBrickSize => f.write_str("invalid SDF brick size"),
            Self::InvalidMipCount => f.write_str("invalid SDF mip count"),
            Self::InvalidMip => f.write_str("SDF mip is outside the grid"),
            Self::SampleCount { .. } => f.write_str("SDF brick sample count is incorrect"),
            Self::InvalidRegion => f.write_str("SDF dirty region is inverted"),
            Self::ExportBudgetExceeded {
                resource,
                requested,
                limit,
            } => write!(
                f,
                "SDF export {resource:?} budget exceeded: requested {requested}, limit {limit}"
            ),
            Self::ExportIndexOverflow => f.write_str("SDF export index does not fit packed ABI"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SdfBrick {
    samples: Vec<i16>,
}

/// Sparse, clipmap-compatible signed-distance brick storage.
///
/// Bricks are keyed by integer coordinates and stored in a `BTreeMap`, which
/// gives deterministic iteration without relying on hash-map iteration order.
/// Dirty coordinates are retained even when a brick is not allocated yet, so a
/// baker can distinguish an empty brick from a brick that needs regeneration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseSdfBrickGrid {
    brick_size: u8,
    mip_levels: u8,
    bricks: BTreeMap<BrickCoord, SdfBrick>,
    dirty: BTreeSet<BrickCoord>,
}

impl SparseSdfBrickGrid {
    /// Creates a grid with cubic bricks and the given number of mips.
    pub fn new(brick_size: u8, mip_levels: u8) -> Result<Self, SdfError> {
        if brick_size == 0 || brick_size > 32 {
            return Err(SdfError::InvalidBrickSize);
        }
        if mip_levels == 0 {
            return Err(SdfError::InvalidMipCount);
        }
        Ok(Self {
            brick_size,
            mip_levels,
            bricks: BTreeMap::new(),
            dirty: BTreeSet::new(),
        })
    }

    /// Returns the configured edge length in voxels.
    pub const fn brick_size(&self) -> u8 {
        self.brick_size
    }

    /// Returns the number of mip levels.
    pub const fn mip_levels(&self) -> u8 {
        self.mip_levels
    }

    /// Returns the number of allocated bricks.
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }

    /// Returns the number of samples stored in every allocated brick.
    pub fn sample_count(&self) -> usize {
        self.sample_count_for_edge()
    }

    /// Returns allocated brick coordinates in deterministic map order.
    pub fn allocated_bricks(&self) -> Vec<BrickCoord> {
        self.bricks.keys().copied().collect()
    }

    /// Returns the samples for one allocated brick without copying them.
    pub fn brick_samples(&self, coord: BrickCoord) -> Result<Option<&[i16]>, SdfError> {
        self.validate_mip(coord.mip)?;
        Ok(self
            .bricks
            .get(&coord)
            .map(|brick| brick.samples.as_slice()))
    }

    /// Exports every allocated brick into bounded, deterministic packed data.
    ///
    /// The export performs all budget and packed-index checks before building
    /// output, so an error never returns a partial export. It is a CPU staging
    /// contract for an external adapter; this crate does not acquire an
    /// adapter, allocate GPU resources, or execute GPU sampling.
    pub fn export_allocated_bricks(
        &self,
        budget: SdfExportBudget,
    ) -> Result<SdfBrickExport, SdfError> {
        let brick_count = self.bricks.len();
        if brick_count > budget.max_bricks {
            return Err(SdfError::ExportBudgetExceeded {
                resource: SdfExportResource::Bricks,
                requested: brick_count,
                limit: budget.max_bricks,
            });
        }
        let sample_count = self
            .sample_count_for_edge()
            .checked_mul(brick_count)
            .ok_or(SdfError::ExportIndexOverflow)?;
        if sample_count > budget.max_samples {
            return Err(SdfError::ExportBudgetExceeded {
                resource: SdfExportResource::Samples,
                requested: sample_count,
                limit: budget.max_samples,
            });
        }
        let sample_count_u32 = u32::try_from(self.sample_count_for_edge())
            .map_err(|_| SdfError::ExportIndexOverflow)?;
        if u32::try_from(sample_count).is_err() {
            return Err(SdfError::ExportIndexOverflow);
        }

        let mut bricks = Vec::with_capacity(brick_count);
        let mut samples = Vec::with_capacity(sample_count);
        for (coord, brick) in &self.bricks {
            let sample_offset =
                u32::try_from(samples.len()).map_err(|_| SdfError::ExportIndexOverflow)?;
            bricks.push(SdfExportBrick {
                coord: *coord,
                sample_offset,
                sample_count: sample_count_u32,
            });
            samples.extend_from_slice(&brick.samples);
        }
        Ok(SdfBrickExport { bricks, samples })
    }

    /// Returns the number of dirty brick coordinates.
    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    /// Inserts or replaces a brick after validating its exact sample volume.
    pub fn insert_brick(&mut self, coord: BrickCoord, samples: Vec<i16>) -> Result<(), SdfError> {
        self.validate_mip(coord.mip)?;
        let expected = self.sample_count();
        if samples.len() != expected {
            return Err(SdfError::SampleCount {
                expected,
                actual: samples.len(),
            });
        }
        self.bricks.insert(coord, SdfBrick { samples });
        Ok(())
    }

    /// Removes an allocated brick and returns whether one existed.
    pub fn remove_brick(&mut self, coord: BrickCoord) -> Result<bool, SdfError> {
        self.validate_mip(coord.mip)?;
        let removed = self.bricks.remove(&coord).is_some();
        if removed {
            self.mark_parent_mips(coord);
        }
        Ok(removed)
    }

    /// Samples the nearest voxel at an integer world-voxel coordinate.
    pub fn sample(&self, voxel: [i32; 3], mip: u8) -> Result<Option<i16>, SdfError> {
        self.validate_mip(mip)?;
        let size = i32::from(self.brick_size);
        let bx = floor_div(voxel[0], size);
        let by = floor_div(voxel[1], size);
        let bz = floor_div(voxel[2], size);
        let lx = voxel[0] - bx * size;
        let ly = voxel[1] - by * size;
        let lz = voxel[2] - bz * size;
        let coord = BrickCoord::new(bx, by, bz, mip);
        let Some(brick) = self.bricks.get(&coord) else {
            return Ok(None);
        };
        let edge = usize::from(self.brick_size);
        let index = (usize::try_from(lz).unwrap_or(0) * edge + usize::try_from(ly).unwrap_or(0))
            * edge
            + usize::try_from(lx).unwrap_or(0);
        Ok(brick.samples.get(index).copied())
    }

    /// Marks every brick coordinate intersecting an inclusive voxel region.
    ///
    /// The returned count is the number of coordinates newly added to the
    /// dirty set. This includes not-yet-allocated bricks for incremental bake
    /// systems that allocate only non-empty results.
    pub fn mark_dirty_region(
        &mut self,
        min_voxel: [i32; 3],
        max_voxel: [i32; 3],
        mip: u8,
    ) -> Result<usize, SdfError> {
        self.validate_mip(mip)?;
        if min_voxel[0] > max_voxel[0] || min_voxel[1] > max_voxel[1] || min_voxel[2] > max_voxel[2]
        {
            return Err(SdfError::InvalidRegion);
        }
        let size = i32::from(self.brick_size);
        let min = [
            floor_div(min_voxel[0], size),
            floor_div(min_voxel[1], size),
            floor_div(min_voxel[2], size),
        ];
        let max = [
            floor_div(max_voxel[0], size),
            floor_div(max_voxel[1], size),
            floor_div(max_voxel[2], size),
        ];
        let mut added = 0;
        for z in min[2]..=max[2] {
            for y in min[1]..=max[1] {
                for x in min[0]..=max[0] {
                    added += self.mark_dirty_with_parents(BrickCoord::new(x, y, z, mip));
                }
            }
        }
        Ok(added)
    }

    /// Returns the exact quantized minimum and maximum stored in one brick.
    ///
    /// The result is conservative for culling and broad-phase queries because
    /// it never excludes a stored sample. An unallocated brick has no bounds.
    pub fn min_max(&self, coord: BrickCoord) -> Result<Option<(i16, i16)>, SdfError> {
        self.validate_mip(coord.mip)?;
        let Some(brick) = self.bricks.get(&coord) else {
            return Ok(None);
        };
        let mut samples = brick.samples.iter().copied();
        let Some(first) = samples.next() else {
            return Ok(None);
        };
        let mut min = first;
        let mut max = first;
        for sample in samples {
            min = min.min(sample);
            max = max.max(sample);
        }
        Ok(Some((min, max)))
    }

    /// Returns conservative bounds over all allocated bricks intersecting a
    /// voxel region at one mip.
    ///
    /// Missing bricks are ignored; callers must treat `None` as unknown rather
    /// than as an empty field. The query scans only allocated brick keys and
    /// therefore remains deterministic, at the cost of a sparse-map walk.
    pub fn region_min_max(
        &self,
        min_voxel: [i32; 3],
        max_voxel: [i32; 3],
        mip: u8,
    ) -> Result<Option<(i16, i16)>, SdfError> {
        self.validate_mip(mip)?;
        if min_voxel[0] > max_voxel[0] || min_voxel[1] > max_voxel[1] || min_voxel[2] > max_voxel[2]
        {
            return Err(SdfError::InvalidRegion);
        }
        let size = i32::from(self.brick_size);
        let min = [
            floor_div(min_voxel[0], size),
            floor_div(min_voxel[1], size),
            floor_div(min_voxel[2], size),
        ];
        let max = [
            floor_div(max_voxel[0], size),
            floor_div(max_voxel[1], size),
            floor_div(max_voxel[2], size),
        ];
        let mut result: Option<(i16, i16)> = None;
        for (coord, brick) in &self.bricks {
            if coord.mip != mip
                || coord.x < min[0]
                || coord.x > max[0]
                || coord.y < min[1]
                || coord.y > max[1]
                || coord.z < min[2]
                || coord.z > max[2]
            {
                continue;
            }
            let mut samples = brick.samples.iter().copied();
            let Some(first) = samples.next() else {
                continue;
            };
            let mut local_min = first;
            let mut local_max = first;
            for sample in samples {
                local_min = local_min.min(sample);
                local_max = local_max.max(sample);
            }
            result = Some(match result {
                Some((current_min, current_max)) => {
                    (current_min.min(local_min), current_max.max(local_max))
                }
                None => (local_min, local_max),
            });
        }
        Ok(result)
    }

    /// Returns dirty coordinates in deterministic order and clears the set.
    pub fn take_dirty(&mut self) -> Vec<BrickCoord> {
        let dirty = core::mem::take(&mut self.dirty);
        dirty.into_iter().collect()
    }

    /// Returns allocated bricks in a same-mip Chebyshev neighborhood.
    ///
    /// The center is included when allocated. Results are sorted by the
    /// `BrickCoord` order, making the output suitable for deterministic jobs.
    pub fn mip_neighborhood(
        &self,
        center: BrickCoord,
        radius: u8,
    ) -> Result<Vec<BrickCoord>, SdfError> {
        self.validate_mip(center.mip)?;
        let radius = i64::from(radius);
        let mut result = Vec::new();
        for coord in self.bricks.keys() {
            if coord.mip == center.mip
                && i64::from(coord.x).abs_diff(i64::from(center.x)) <= radius as u64
                && i64::from(coord.y).abs_diff(i64::from(center.y)) <= radius as u64
                && i64::from(coord.z).abs_diff(i64::from(center.z)) <= radius as u64
            {
                result.push(*coord);
            }
        }
        Ok(result)
    }

    fn sample_count_for_edge(&self) -> usize {
        let edge = usize::from(self.brick_size);
        edge * edge * edge
    }

    fn validate_mip(&self, mip: u8) -> Result<(), SdfError> {
        if mip >= self.mip_levels {
            Err(SdfError::InvalidMip)
        } else {
            Ok(())
        }
    }

    fn mark_dirty_with_parents(&mut self, coord: BrickCoord) -> usize {
        let mut added = 0;
        let mut current = coord;
        loop {
            if self.dirty.insert(current) {
                added += 1;
            }
            if current.mip + 1 >= self.mip_levels {
                break;
            }
            current = BrickCoord::new(
                floor_div(current.x, 2),
                floor_div(current.y, 2),
                floor_div(current.z, 2),
                current.mip + 1,
            );
        }
        added
    }

    fn mark_parent_mips(&mut self, coord: BrickCoord) {
        if coord.mip + 1 >= self.mip_levels {
            return;
        }
        let mut current = coord;
        while current.mip + 1 < self.mip_levels {
            current = BrickCoord::new(
                floor_div(current.x, 2),
                floor_div(current.y, 2),
                floor_div(current.z, 2),
                current.mip + 1,
            );
            self.dirty.insert(current);
        }
    }
}

fn floor_div(value: i32, divisor: i32) -> i32 {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder < 0 {
        quotient - 1
    } else {
        quotient
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn brick(samples: &[i16; 8]) -> Vec<i16> {
        samples.to_vec()
    }

    #[test]
    fn read_only_consumers_share_negative_and_boundary_samples() {
        let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
        let mut samples = [0_i16; 8];
        samples[0] = -9;
        samples[1] = 7;
        grid.insert_brick(BrickCoord::new(-1, 0, 0, 0), brick(&samples))
            .unwrap();

        let first = SparseSdfQuery::new(&grid);
        let second = SparseSdfQuery::new(&grid);
        assert_eq!(
            first.query_sample([-2, 0, 0], 0).unwrap(),
            SdfQueryResult::Value(-9)
        );
        assert_eq!(
            second.query_sample([-1, 0, 0], 0).unwrap(),
            SdfQueryResult::Value(7)
        );
        assert_eq!(
            first.query_sample([0, 0, 0], 0).unwrap(),
            SdfQueryResult::Unknown
        );
        assert_eq!(
            second.query_sample([i32::MIN, 0, 0], 0).unwrap(),
            SdfQueryResult::Unknown
        );
    }

    #[test]
    fn allocated_brick_access_is_sorted_and_borrowed() {
        let mut grid = SparseSdfBrickGrid::new(2, 2).unwrap();
        grid.insert_brick(BrickCoord::new(1, 0, 0, 0), vec![7; 8])
            .unwrap();
        grid.insert_brick(BrickCoord::new(-1, 0, 0, 0), vec![-3; 8])
            .unwrap();

        assert_eq!(grid.sample_count(), 8);
        assert_eq!(
            grid.allocated_bricks(),
            vec![BrickCoord::new(-1, 0, 0, 0), BrickCoord::new(1, 0, 0, 0)]
        );
        assert_eq!(
            grid.brick_samples(BrickCoord::new(-1, 0, 0, 0)).unwrap(),
            Some(&[-3; 8][..])
        );
    }

    #[test]
    fn range_query_distinguishes_value_miss_and_unknown() {
        let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
        grid.insert_brick(BrickCoord::new(-1, 0, 0, 0), vec![-2; 8])
            .unwrap();
        {
            let query = SparseSdfQuery::new(&grid);
            assert_eq!(
                query.query_surface_region([-2, 0, 0], [-1, 1, 1], 0),
                Ok(SdfQueryResult::Miss)
            );
            assert_eq!(
                query.query_surface_region([-2, 0, 0], [0, 1, 1], 0),
                Ok(SdfQueryResult::Unknown)
            );
        }

        grid.insert_brick(BrickCoord::new(0, 0, 0, 0), vec![3; 8])
            .unwrap();
        {
            let query = SparseSdfQuery::new(&grid);
            assert_eq!(
                query.query_surface_region([0, 0, 0], [1, 1, 1], 0),
                Ok(SdfQueryResult::Miss)
            );
        }
        grid.insert_brick(BrickCoord::new(-1, 0, 0, 0), vec![-3, 2, 2, 2, 2, 2, 2, 2])
            .unwrap();
        let query = SparseSdfQuery::new(&grid);
        assert_eq!(
            query.query_surface_region([-2, 0, 0], [1, 1, 1], 0),
            Ok(SdfQueryResult::Value(SdfBounds { min: -3, max: 3 }))
        );
    }

    #[test]
    fn range_query_rejects_inverted_regions_and_invalid_mips() {
        let grid = SparseSdfBrickGrid::new(2, 1).unwrap();
        let query = SparseSdfQuery::new(&grid);
        assert_eq!(
            query.query_surface_region([1, 0, 0], [0, 0, 0], 0),
            Err(SdfError::InvalidRegion)
        );
        assert_eq!(query.query_sample([0, 0, 0], 1), Err(SdfError::InvalidMip));
    }

    #[test]
    fn bounded_export_is_sorted_and_packs_samples_without_reordering() {
        let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
        grid.insert_brick(BrickCoord::new(2, -1, 0, 0), vec![20; 8])
            .unwrap();
        grid.insert_brick(
            BrickCoord::new(-1, 0, 0, 0),
            vec![-10, -9, -8, -7, -6, -5, -4, -3],
        )
        .unwrap();

        let export = grid
            .export_allocated_bricks(SdfExportBudget::new(2, 16))
            .unwrap();
        assert_eq!(export.brick_count(), 2);
        assert_eq!(export.sample_count(), 16);
        assert_eq!(export.sample_bytes(), 32);
        assert_eq!(export.bricks()[0].coord, BrickCoord::new(-1, 0, 0, 0));
        assert_eq!(export.bricks()[0].sample_offset, 0);
        assert_eq!(export.bricks()[1].coord, BrickCoord::new(2, -1, 0, 0));
        assert_eq!(export.bricks()[1].sample_offset, 8);
        assert_eq!(&export.samples()[..8], &[-10, -9, -8, -7, -6, -5, -4, -3]);
        assert_eq!(&export.samples()[8..], &[20; 8]);
    }

    #[test]
    fn bounded_export_rejects_bricks_or_samples_before_returning_partial_data() {
        let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
        grid.insert_brick(BrickCoord::new(0, 0, 0, 0), vec![1; 8])
            .unwrap();
        assert_eq!(
            grid.export_allocated_bricks(SdfExportBudget::new(0, 8)),
            Err(SdfError::ExportBudgetExceeded {
                resource: SdfExportResource::Bricks,
                requested: 1,
                limit: 0,
            })
        );
        assert_eq!(
            grid.export_allocated_bricks(SdfExportBudget::new(1, 7)),
            Err(SdfError::ExportBudgetExceeded {
                resource: SdfExportResource::Samples,
                requested: 8,
                limit: 7,
            })
        );
    }
}
