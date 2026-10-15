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
}

impl fmt::Display for SdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBrickSize => f.write_str("invalid SDF brick size"),
            Self::InvalidMipCount => f.write_str("invalid SDF mip count"),
            Self::InvalidMip => f.write_str("SDF mip is outside the grid"),
            Self::SampleCount { .. } => f.write_str("SDF brick sample count is incorrect"),
            Self::InvalidRegion => f.write_str("SDF dirty region is inverted"),
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

    fn sample_count(&self) -> usize {
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
