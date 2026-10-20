//! Bounded local rebaking into an exclusively borrowed published SDF snapshot.
//!
//! A voxel edit invalidates its intersecting bricks and their coarser parents.
//! Staging owns only those replacement samples; the resident map is never
//! cloned or scanned. Sampling errors and unwinding leave publication untouched.

extern crate alloc;

use alloc::vec::Vec;
use core::{fmt, mem::size_of};
use prism_world_residency::{BrickCoord, SdfError, SparseSdfBrickGrid};

use super::{
    SdfBridgeError, WorldSdfPhysicsBridge, WorldSdfTransform, WORLD_SDF_BRICK_METADATA_BYTES,
    WORLD_SDF_HEADER_BYTES, WORLD_SDF_SAMPLE_BYTES,
};

/// Inclusive edited voxel coverage at one mip, with all coarser parents implied.
///
/// The caller must include the source's full influence radius (and any sampling
/// halo). This is an invalidation region, not a geometric CSG operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfRebakeRegion {
    /// Minimum edited voxel, inclusive, at `mip`.
    pub min_voxel: [i32; 3],
    /// Maximum edited voxel, inclusive, at `mip`.
    pub max_voxel: [i32; 3],
    /// Finest mip invalidated by this edit. Finer mips are unaffected.
    pub mip: u8,
}

/// Independent work and staging limits for one atomic local update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfRebakeLimits {
    /// Maximum affected bricks, including all coarser mip parents.
    pub max_bricks: usize,
    /// Maximum source callback invocations across all affected bricks.
    pub max_samples: usize,
    /// Maximum logical staging bytes: descriptors plus replacement samples.
    ///
    /// Excludes the resident map, allocator overhead and source-owned memory;
    /// this is not an RSS or GPU memory bound.
    pub max_scratch_bytes: usize,
}

/// Resource bounded by [`SdfRebakeLimits`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdfRebakeResource {
    /// Affected bricks, including parents.
    Bricks,
    /// Quantized samples to evaluate.
    Samples,
    /// Logical staging bytes.
    ScratchBytes,
}

/// Exact position and physical units passed to a quantized field evaluator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfRebakePoint {
    /// Owning brick, including its mip.
    pub brick: BrickCoord,
    /// Voxel coordinate at this mip; i64 preserves full signed brick coverage.
    pub voxel: [i64; 3],
    /// Exact sample position in millimetres, before any float conversion.
    pub world_position_mm: [i128; 3],
    /// Physical voxel spacing at this mip, in millimetres.
    pub voxel_size_mm: i128,
    /// Millimetres represented by one returned signed-i16 unit.
    ///
    /// The evaluator must quantize using this scale at every mip. Values are
    /// stored verbatim: no clamping, normalization or distance rescaling occurs.
    pub distance_scale_mm: u32,
}

/// Costs and revision transition of a successfully published local update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfRebakeReport {
    /// Previously published revision.
    pub previous_revision: u64,
    /// Newly published revision.
    pub revision: u64,
    /// Replaced or newly inserted bricks, including coarser parents.
    pub rebaked_bricks: usize,
    /// Number of source callback invocations.
    pub sampled_voxels: usize,
    /// Bricks that were not resident before this update.
    pub inserted_bricks: usize,
    /// Logical bytes of replacement samples and staging descriptors.
    pub scratch_bytes: usize,
    /// Logical footprint of the complete published snapshot.
    pub logical_bytes: usize,
}

/// Failure before any resident samples or revision are changed.
#[derive(Debug, Eq, PartialEq)]
pub enum SdfRebakeError<E> {
    /// No snapshot has been published.
    NoSnapshot,
    /// Existing grid invalidations must be resolved by its owner first.
    DirtySnapshot,
    /// An edited minimum exceeds its maximum.
    InvalidRegion,
    /// A coordinate, revision, or final resident budget is invalid.
    Bridge(SdfBridgeError),
    /// Work or staging would exceed the operation's bound.
    BudgetExceeded {
        /// Resource that exceeded its limit.
        resource: SdfRebakeResource,
        /// Required count or bytes.
        required: usize,
        /// Allowed count or bytes.
        limit: usize,
    },
    /// A fallible replacement buffer allocation failed.
    AllocationFailed,
    /// The evaluator rejected a sample; its external side effects are not undone.
    Source {
        /// Owning brick of the failed sample.
        brick: BrickCoord,
        /// Exact voxel coordinate at the brick's mip.
        voxel: [i64; 3],
        /// Evaluator-specific error.
        error: E,
    },
}

impl<E: fmt::Display> fmt::Display for SdfRebakeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSnapshot => f.write_str("world SDF rebake has no published snapshot"),
            Self::DirtySnapshot => f.write_str("world SDF rebake requires a clean source grid"),
            Self::InvalidRegion => f.write_str("world SDF rebake region is inverted"),
            Self::Bridge(error) => error.fmt(f),
            Self::BudgetExceeded {
                resource,
                required,
                limit,
            } => write!(
                f,
                "world SDF rebake needs {required} {resource:?}; limit is {limit}"
            ),
            Self::AllocationFailed => f.write_str("world SDF rebake staging allocation failed"),
            Self::Source {
                brick,
                voxel,
                error,
            } => write!(
                f,
                "world SDF rebake sample {:?} at mip {} failed: {error}",
                voxel, brick.mip
            ),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for SdfRebakeError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Bridge(error) => Some(error),
            Self::Source { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl<E> From<SdfBridgeError> for SdfRebakeError<E> {
    fn from(error: SdfBridgeError) -> Self {
        Self::Bridge(error)
    }
}

struct Replacement {
    coord: BrickCoord,
    samples: Vec<i16>,
}

impl WorldSdfPhysicsBridge {
    /// Rebakes a local edit and all coarser parents, then publishes one revision.
    ///
    /// Every budget, coordinate, revision and allocation is checked before
    /// invoking `sample`. The evaluator must read one coherent source revision
    /// for the whole call and return the signed distance in the supplied units.
    /// Evaluation order is increasing mip, then brick z/y/x, then voxel z/y/x
    /// (x fastest). Each affected brick is evaluated once, including missing
    /// bricks, which become resident. Unaffected sample allocations are retained.
    ///
    /// Errors or evaluator unwinding preserve the previous data and revision.
    /// No evaluator is called during commit. Existing `BTreeMap` insertion uses
    /// the standard infallible allocator; process-aborting OOM is not recovered.
    /// This synchronous worker operation requires an exclusive borrow; it does
    /// not schedule jobs, update cell-host aggregate usage, or upload GPU data.
    ///
    /// ```
    /// use prism_world_physics_bridge::{SdfPhysicsBridgeBudget, SdfRebakeLimits,
    ///     SdfRebakeRegion, WorldSdfPhysicsBridge, WorldSdfSnapshot, WorldSdfTransform};
    /// use prism_world_residency::SparseSdfBrickGrid;
    /// let mut bridge = WorldSdfPhysicsBridge::new(
    ///     SdfPhysicsBridgeBudget::new(2, 16, 112).unwrap());
    /// bridge.publish(WorldSdfSnapshot::new(
    ///     WorldSdfTransform::new([0; 3], 10, 1).unwrap(), 1,
    ///     SparseSdfBrickGrid::new(2, 2).unwrap())).unwrap();
    /// let report = bridge.rebake_region(
    ///     SdfRebakeRegion { min_voxel: [0; 3], max_voxel: [0; 3], mip: 0 }, 2,
    ///     SdfRebakeLimits { max_bricks: 2, max_samples: 16, max_scratch_bytes: 4096 },
    ///     |point| i16::try_from(point.world_position_mm[1]),
    /// ).unwrap();
    /// assert_eq!(report.rebaked_bricks, 2);
    /// assert_eq!(bridge.current().unwrap().revision(), 2);
    /// ```
    pub fn rebake_region<E>(
        &mut self,
        region: SdfRebakeRegion,
        revision: u64,
        limits: SdfRebakeLimits,
        mut sample: impl FnMut(SdfRebakePoint) -> Result<i16, E>,
    ) -> Result<SdfRebakeReport, SdfRebakeError<E>> {
        let current = self.current.as_mut().ok_or(SdfRebakeError::NoSnapshot)?;
        if revision <= current.revision() {
            return Err(SdfBridgeError::StaleRevision {
                current: current.revision(),
                incoming: revision,
            }
            .into());
        }
        let grid = &current.source.grid;
        if grid.dirty_count() != 0 {
            return Err(SdfRebakeError::DirtySnapshot);
        }
        if region.mip >= grid.mip_levels() {
            return Err(SdfBridgeError::GridQuery(SdfError::InvalidMip).into());
        }
        if (0..3).any(|axis| region.min_voxel[axis] > region.max_voxel[axis]) {
            return Err(SdfRebakeError::InvalidRegion);
        }
        let transform = current.transform();
        let mut bricks = 0_usize;
        // Count level rectangles directly: no O(base-bricks * mip-count)
        // repeated parent enumeration, and no grid-wide key materialization.
        for_each_level(grid, region, |min, max, mip| {
            let mut level_count = 1_usize;
            for axis in 0..3 {
                let length = usize::try_from(i64::from(max[axis]) - i64::from(min[axis]) + 1)
                    .map_err(|_| SdfBridgeError::ArithmeticOverflow)?;
                level_count = level_count
                    .checked_mul(length)
                    .ok_or(SdfBridgeError::ArithmeticOverflow)?;
            }
            bricks = bricks
                .checked_add(level_count)
                .ok_or(SdfBridgeError::ArithmeticOverflow)?;
            // Positive spacing makes the two extrema sufficient for each axis.
            let edge = i64::from(grid.brick_size());
            for (coord, last) in [(min, 0), (max, edge - 1)] {
                let voxel = coord.map(|value| i64::from(value) * edge + last);
                point_at(
                    transform,
                    BrickCoord::new(coord[0], coord[1], coord[2], mip),
                    voxel,
                )?;
            }
            Ok::<_, SdfBridgeError>(())
        })?;
        check_limit(SdfRebakeResource::Bricks, bricks, limits.max_bricks)?;
        let samples = bricks
            .checked_mul(grid.sample_count())
            .ok_or(SdfBridgeError::ArithmeticOverflow)?;
        check_limit(SdfRebakeResource::Samples, samples, limits.max_samples)?;
        let scratch_bytes = bricks
            .checked_mul(size_of::<Replacement>())
            .and_then(|bytes| {
                samples
                    .checked_mul(WORLD_SDF_SAMPLE_BYTES)
                    .and_then(|n| bytes.checked_add(n))
            })
            .ok_or(SdfBridgeError::ArithmeticOverflow)?;
        check_limit(
            SdfRebakeResource::ScratchBytes,
            scratch_bytes,
            limits.max_scratch_bytes,
        )?;

        let mut inserted_bricks = 0_usize;
        for_each_brick(grid, region, |coord| {
            if grid
                .brick_samples(coord)
                .map_err(SdfBridgeError::GridQuery)?
                .is_none()
            {
                inserted_bricks += 1;
            }
            Ok::<_, SdfBridgeError>(())
        })?;
        let resident_bricks = grid
            .brick_count()
            .checked_add(inserted_bricks)
            .ok_or(SdfBridgeError::ArithmeticOverflow)?;
        if resident_bricks > self.budget.max_bricks {
            return Err(SdfBridgeError::BrickCapacityExceeded {
                required: resident_bricks,
                capacity: self.budget.max_bricks,
            }
            .into());
        }
        let resident_samples = resident_bricks
            .checked_mul(grid.sample_count())
            .ok_or(SdfBridgeError::ArithmeticOverflow)?;
        if resident_samples > self.budget.max_samples {
            return Err(SdfBridgeError::SampleCapacityExceeded {
                required: resident_samples,
                capacity: self.budget.max_samples,
            }
            .into());
        }
        let logical_bytes = resident_bricks
            .checked_mul(WORLD_SDF_BRICK_METADATA_BYTES)
            .and_then(|n| n.checked_add(WORLD_SDF_HEADER_BYTES))
            .and_then(|n| {
                resident_samples
                    .checked_mul(WORLD_SDF_SAMPLE_BYTES)
                    .and_then(|s| n.checked_add(s))
            })
            .ok_or(SdfBridgeError::ArithmeticOverflow)?;
        if logical_bytes > self.budget.max_bytes {
            return Err(SdfBridgeError::UploadBudgetExceeded {
                required: logical_bytes,
                limit: self.budget.max_bytes,
            }
            .into());
        }

        let mut replacements = Vec::new();
        replacements
            .try_reserve_exact(bricks)
            .map_err(|_| SdfRebakeError::AllocationFailed)?;
        for_each_brick(grid, region, |coord| {
            let mut samples = Vec::new();
            samples
                .try_reserve_exact(grid.sample_count())
                .map_err(|_| SdfRebakeError::AllocationFailed)?;
            replacements.push(Replacement { coord, samples });
            Ok::<_, SdfRebakeError<E>>(())
        })?;
        let edge = i64::from(grid.brick_size());
        for replacement in &mut replacements {
            let coord = replacement.coord;
            let base = [coord.x, coord.y, coord.z].map(|v| i64::from(v) * edge);
            for z in 0..edge {
                for y in 0..edge {
                    for x in 0..edge {
                        let point =
                            point_at(transform, coord, [base[0] + x, base[1] + y, base[2] + z])?;
                        replacement.samples.push(sample(point).map_err(|error| {
                            SdfRebakeError::Source {
                                brick: coord,
                                voxel: point.voxel,
                                error,
                            }
                        })?);
                    }
                }
            }
        }

        let previous_revision = current.revision();
        // All keys and exact volumes were constructed against this exclusively
        // borrowed grid. insert_brick's validation cannot reject any staged item.
        for replacement in replacements {
            current
                .source
                .grid
                .insert_brick(replacement.coord, replacement.samples)
                .map_err(SdfBridgeError::GridQuery)?;
        }
        current.source.revision = revision;
        current.logical_bytes = logical_bytes;
        Ok(SdfRebakeReport {
            previous_revision,
            revision,
            rebaked_bricks: bricks,
            sampled_voxels: samples,
            inserted_bricks,
            scratch_bytes,
            logical_bytes,
        })
    }
}

fn check_limit<E>(
    resource: SdfRebakeResource,
    required: usize,
    limit: usize,
) -> Result<(), SdfRebakeError<E>> {
    if required > limit {
        Err(SdfRebakeError::BudgetExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

fn point_at(
    transform: WorldSdfTransform,
    brick: BrickCoord,
    voxel: [i64; 3],
) -> Result<SdfRebakePoint, SdfBridgeError> {
    let voxel_size_mm = transform.mip_voxel_size_mm(brick.mip)?;
    let mut world_position_mm = [0_i128; 3];
    for axis in 0..3 {
        world_position_mm[axis] =
            affine_position(voxel[axis], voxel_size_mm, transform.origin_mm()[axis])
                .ok_or(SdfBridgeError::ArithmeticOverflow)?;
    }
    Ok(SdfRebakePoint {
        brick,
        voxel,
        world_position_mm,
        voxel_size_mm,
        distance_scale_mm: transform.distance_scale_mm(),
    })
}

// A negative origin can bring an overflowing positive product back into i128
// (and conversely for a negative product). Check the final affine coordinate,
// not only the intermediate multiplication. The usual path stays signed.
fn affine_position(voxel: i64, spacing: i128, origin: i64) -> Option<i128> {
    let voxel = i128::from(voxel);
    let origin = i128::from(origin);
    if let Some(product) = voxel.checked_mul(spacing) {
        return product.checked_add(origin);
    }
    let magnitude = voxel.unsigned_abs().checked_mul(spacing.unsigned_abs())?;
    if voxel > 0 && origin < 0 {
        i128::try_from(magnitude.checked_sub(origin.unsigned_abs())?).ok()
    } else if voxel < 0 && origin > 0 {
        let magnitude = magnitude.checked_sub(origin.unsigned_abs())?;
        // Subtract one before converting so i128::MIN remains representable.
        i128::try_from(magnitude.checked_sub(1)?)
            .ok()
            .map(|value| -value - 1)
    } else {
        None
    }
}

fn for_each_level<E>(
    grid: &SparseSdfBrickGrid,
    region: SdfRebakeRegion,
    mut visit: impl FnMut([i32; 3], [i32; 3], u8) -> Result<(), E>,
) -> Result<(), E> {
    let edge = i32::from(grid.brick_size());
    let mut min = region.min_voxel.map(|v| v.div_euclid(edge));
    let mut max = region.max_voxel.map(|v| v.div_euclid(edge));
    for mip in region.mip..grid.mip_levels() {
        visit(min, max, mip)?;
        min = min.map(|v| v.div_euclid(2));
        max = max.map(|v| v.div_euclid(2));
    }
    Ok(())
}

fn for_each_brick<E>(
    grid: &SparseSdfBrickGrid,
    region: SdfRebakeRegion,
    mut visit: impl FnMut(BrickCoord) -> Result<(), E>,
) -> Result<(), E> {
    for_each_level(grid, region, |min, max, mip| {
        for z in min[2]..=max[2] {
            for y in min[1]..=max[1] {
                for x in min[0]..=max[0] {
                    visit(BrickCoord::new(x, y, z, mip))?;
                }
            }
        }
        Ok(())
    })
}
