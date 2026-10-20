//! World sparse-SDF to physics-consumer bridge.
//!
//! The world residency crate owns quantized sparse-brick storage. This module
//! adds the missing physical interpretation at the consumer boundary:
//! explicit world origin, voxel spacing, distance scale, and a monotonic
//! source revision. The bridge only publishes fully budgeted snapshots and
//! keeps the previous snapshot when a candidate is rejected.

extern crate alloc;
use alloc::sync::Arc;

pub use prism_world_residency::{SdfBounds, SdfQueryResult};
use prism_world_residency::{SdfQueryConsumer, SparseSdfBrickGrid, SparseSdfQuery};

mod sampling;
pub use sampling::SdfDistanceGradient;
mod rebake;
pub use rebake::{
    SdfRebakeError, SdfRebakeLimits, SdfRebakePoint, SdfRebakeRegion, SdfRebakeReport,
    SdfRebakeResource, SdfRebakeUpdate, SdfRebakeWork,
};
#[cfg(feature = "tasks")]
pub use rebake::{SdfRebakeTask, SdfRebakeTaskError};

/// Logical bytes represented by the fixed SDF snapshot header.
pub const WORLD_SDF_HEADER_BYTES: usize = 48;
/// Logical bytes represented by one packed sparse-brick metadata record.
pub const WORLD_SDF_BRICK_METADATA_BYTES: usize = 16;
/// Logical bytes represented by one quantized SDF sample.
pub const WORLD_SDF_SAMPLE_BYTES: usize = 2;

/// World-space interpretation of integer sparse-SDF voxels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfTransform {
    /// World-space origin of voxel coordinate `[0, 0, 0]`, in millimetres.
    origin_mm: [i64; 3],
    /// Edge length of one mip-zero voxel, in millimetres.
    voxel_size_mm: u32,
    /// Physical millimetres represented by one quantized SDF unit.
    distance_scale_mm: u32,
}

impl WorldSdfTransform {
    /// Creates a transform with explicit positive unit scales.
    pub const fn new(
        origin_mm: [i64; 3],
        voxel_size_mm: u32,
        distance_scale_mm: u32,
    ) -> Result<Self, SdfBridgeError> {
        if voxel_size_mm == 0 || distance_scale_mm == 0 {
            return Err(SdfBridgeError::InvalidTransform);
        }
        Ok(Self {
            origin_mm,
            voxel_size_mm,
            distance_scale_mm,
        })
    }

    /// Returns the world-space position of voxel `[0, 0, 0]`, in millimetres.
    #[must_use]
    pub const fn origin_mm(self) -> [i64; 3] {
        self.origin_mm
    }

    /// Returns the mip-zero voxel edge length, in millimetres.
    #[must_use]
    pub const fn voxel_size_mm(self) -> u32 {
        self.voxel_size_mm
    }

    /// Returns the physical distance represented by one sample unit.
    #[must_use]
    pub const fn distance_scale_mm(self) -> u32 {
        self.distance_scale_mm
    }

    /// Maps a world-space millimetre position to the containing voxel at `mip`.
    ///
    /// Negative coordinates use mathematical floor division, matching sparse
    /// brick addressing. Coordinates outside the grid's signed-i32 ABI and
    /// mip scales that overflow checked arithmetic are rejected.
    pub fn voxel_of(
        self,
        world_position_mm: [i64; 3],
        mip: u8,
    ) -> Result<[i32; 3], SdfBridgeError> {
        let voxel_size = self.mip_voxel_size_mm(mip)?;
        let mut voxel = [0_i32; 3];
        for axis in 0..3 {
            let delta = i128::from(world_position_mm[axis]) - i128::from(self.origin_mm[axis]);
            let coordinate = delta.div_euclid(voxel_size);
            voxel[axis] =
                i32::try_from(coordinate).map_err(|_| SdfBridgeError::VoxelCoordinateOutOfRange)?;
        }
        Ok(voxel)
    }

    fn mip_voxel_size_mm(self, mip: u8) -> Result<i128, SdfBridgeError> {
        if mip >= 127 {
            return Err(SdfBridgeError::MipScaleOverflow);
        }
        let mip_scale = 1_i128
            .checked_shl(u32::from(mip))
            .ok_or(SdfBridgeError::MipScaleOverflow)?;
        i128::from(self.voxel_size_mm)
            .checked_mul(mip_scale)
            .ok_or(SdfBridgeError::MipScaleOverflow)
    }
}

/// An owned, versioned sparse-SDF world snapshot before consumer admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldSdfSnapshot {
    transform: WorldSdfTransform,
    revision: u64,
    grid: SparseSdfBrickGrid,
}

impl WorldSdfSnapshot {
    /// Creates a snapshot from a validated sparse grid and source revision.
    #[must_use]
    pub const fn new(
        transform: WorldSdfTransform,
        revision: u64,
        grid: SparseSdfBrickGrid,
    ) -> Self {
        Self {
            transform,
            revision,
            grid,
        }
    }

    /// Returns the world-space transform.
    #[must_use]
    pub const fn transform(&self) -> WorldSdfTransform {
        self.transform
    }

    /// Returns the source revision.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns the sparse grid without copying it.
    #[must_use]
    pub const fn grid(&self) -> &SparseSdfBrickGrid {
        &self.grid
    }

    /// Returns a read-only query view that preserves `Unknown` for missing data.
    #[must_use]
    pub const fn query(&self) -> SparseSdfQuery<'_> {
        SparseSdfQuery::new(&self.grid)
    }
}

/// Independent limits for one admitted physics SDF snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfPhysicsBridgeBudget {
    /// Maximum allocated bricks.
    pub max_bricks: usize,
    /// Maximum quantized samples across all allocated bricks.
    pub max_samples: usize,
    /// Maximum logical bytes for header, brick metadata, and samples.
    pub max_bytes: usize,
}

impl SdfPhysicsBridgeBudget {
    /// Creates a checked budget. Zero limits are valid for an empty snapshot.
    pub const fn new(
        max_bricks: usize,
        max_samples: usize,
        max_bytes: usize,
    ) -> Result<Self, SdfBridgeError> {
        if max_bricks == 0 && max_samples != 0 {
            return Err(SdfBridgeError::InvalidBudget);
        }
        Ok(Self {
            max_bricks,
            max_samples,
            max_bytes,
        })
    }
}

/// Failure raised before a new SDF snapshot replaces the current one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdfBridgeError {
    /// A world transform used a zero voxel or distance scale.
    InvalidTransform,
    /// A budget relationship is internally inconsistent.
    InvalidBudget,
    /// Mip-scaled voxel spacing could not be represented.
    MipScaleOverflow,
    /// A world position maps outside the signed-i32 voxel coordinate ABI.
    VoxelCoordinateOutOfRange,
    /// The sparse grid rejected a consumer query.
    GridQuery(prism_world_residency::SdfError),
    /// Checked logical-byte arithmetic overflowed.
    ArithmeticOverflow,
    /// The grid contains more bricks than the bridge admits.
    BrickCapacityExceeded {
        /// Number of bricks requested by the candidate.
        required: usize,
        /// Maximum admitted brick count.
        capacity: usize,
    },
    /// The grid contains more samples than the bridge admits.
    SampleCapacityExceeded {
        /// Number of samples requested by the candidate.
        required: usize,
        /// Maximum admitted sample count.
        capacity: usize,
    },
    /// The snapshot's logical upload footprint exceeds the bridge budget.
    UploadBudgetExceeded {
        /// Logical bytes requested by the candidate.
        required: usize,
        /// Maximum logical bytes admitted.
        limit: usize,
    },
    /// A replacement revision is not newer than the resident revision.
    StaleRevision {
        /// Resident source revision.
        current: u64,
        /// Candidate source revision.
        incoming: u64,
    },
}

impl core::fmt::Display for SdfBridgeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidTransform => formatter.write_str("world SDF transform is invalid"),
            Self::InvalidBudget => formatter.write_str("world SDF bridge budget is invalid"),
            Self::MipScaleOverflow => formatter.write_str("world SDF mip voxel size overflow"),
            Self::VoxelCoordinateOutOfRange => {
                formatter.write_str("world SDF voxel coordinate exceeds signed-i32 range")
            }
            Self::GridQuery(error) => write!(formatter, "world SDF grid query failed: {error}"),
            Self::ArithmeticOverflow => formatter.write_str("world SDF byte arithmetic overflow"),
            Self::BrickCapacityExceeded { required, capacity } => write!(
                formatter,
                "world SDF needs {required} bricks; capacity is {capacity}"
            ),
            Self::SampleCapacityExceeded { required, capacity } => write!(
                formatter,
                "world SDF needs {required} samples; capacity is {capacity}"
            ),
            Self::UploadBudgetExceeded { required, limit } => write!(
                formatter,
                "world SDF needs {required} logical bytes; limit is {limit}"
            ),
            Self::StaleRevision { current, incoming } => write!(
                formatter,
                "world SDF revision {incoming} is not newer than {current}"
            ),
        }
    }
}

impl std::error::Error for SdfBridgeError {}

/// A budget-admitted, immutable snapshot consumable by physics queries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SdfPhysicsSnapshot {
    source: WorldSdfSnapshot,
    logical_bytes: usize,
    publication: Arc<()>,
}

impl SdfPhysicsSnapshot {
    /// Returns the source transform.
    #[must_use]
    pub const fn transform(&self) -> WorldSdfTransform {
        self.source.transform
    }

    /// Returns the source revision.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.source.revision
    }

    /// Returns the source sparse grid.
    #[must_use]
    pub const fn grid(&self) -> &SparseSdfBrickGrid {
        &self.source.grid
    }

    /// Returns the logical consumer footprint.
    #[must_use]
    pub const fn logical_bytes(&self) -> usize {
        self.logical_bytes
    }

    /// Returns a read-only query view over resident samples.
    #[must_use]
    pub const fn query(&self) -> SparseSdfQuery<'_> {
        self.source.query()
    }

    /// Returns the underlying snapshot for diagnostics or another derived
    /// consumer, without exposing mutable state.
    #[must_use]
    pub const fn source(&self) -> &WorldSdfSnapshot {
        &self.source
    }

    /// Transfers the source snapshot without copying brick samples, for example
    /// after eviction to encode a rebuilt derived cache or admit another consumer.
    #[must_use]
    pub fn into_source(self) -> WorldSdfSnapshot {
        self.source
    }

    /// Queries quantized distance at a world-space millimetre position.
    ///
    /// Resident values are converted to signed millimetres using the explicit
    /// distance scale. Missing bricks remain `Unknown`; invalid mips and
    /// out-of-range coordinates return an error.
    pub fn query_world_distance_mm(
        &self,
        world_position_mm: [i64; 3],
        mip: u8,
    ) -> Result<SdfQueryResult<i64>, SdfBridgeError> {
        let voxel = self.transform().voxel_of(world_position_mm, mip)?;
        match self
            .query()
            .query_sample(voxel, mip)
            .map_err(SdfBridgeError::GridQuery)?
        {
            SdfQueryResult::Value(value) => Ok(SdfQueryResult::Value(
                i64::from(value) * i64::from(self.transform().distance_scale_mm),
            )),
            SdfQueryResult::Miss => Ok(SdfQueryResult::Miss),
            SdfQueryResult::Unknown => Ok(SdfQueryResult::Unknown),
        }
    }
}

/// Single-snapshot physics bridge with atomic replacement semantics.
#[derive(Debug)]
pub struct WorldSdfPhysicsBridge {
    budget: SdfPhysicsBridgeBudget,
    current: Option<SdfPhysicsSnapshot>,
}

impl Clone for WorldSdfPhysicsBridge {
    fn clone(&self) -> Self {
        let mut current = self.current.clone();
        if let Some(snapshot) = &mut current {
            // A cloned bridge is independently writable. Outstanding work from
            // its source must never be admitted by this separate publication.
            snapshot.publication = Arc::new(());
        }
        Self {
            budget: self.budget,
            current,
        }
    }
}

impl WorldSdfPhysicsBridge {
    /// Creates an empty bridge with fixed per-snapshot limits.
    #[must_use]
    pub const fn new(budget: SdfPhysicsBridgeBudget) -> Self {
        Self {
            budget,
            current: None,
        }
    }

    /// Returns the configured budget.
    #[must_use]
    pub const fn budget(&self) -> SdfPhysicsBridgeBudget {
        self.budget
    }

    /// Returns the current admitted snapshot.
    #[must_use]
    pub fn current(&self) -> Option<&SdfPhysicsSnapshot> {
        self.current.as_ref()
    }

    /// Validates and owns a candidate without changing the current snapshot.
    pub fn stage(&self, source: WorldSdfSnapshot) -> Result<SdfPhysicsSnapshot, SdfBridgeError> {
        let bricks = source.grid.brick_count();
        if bricks > self.budget.max_bricks {
            return Err(SdfBridgeError::BrickCapacityExceeded {
                required: bricks,
                capacity: self.budget.max_bricks,
            });
        }
        let samples = source
            .grid
            .sample_count()
            .checked_mul(bricks)
            .ok_or(SdfBridgeError::ArithmeticOverflow)?;
        if samples > self.budget.max_samples {
            return Err(SdfBridgeError::SampleCapacityExceeded {
                required: samples,
                capacity: self.budget.max_samples,
            });
        }
        let metadata_bytes = bricks
            .checked_mul(WORLD_SDF_BRICK_METADATA_BYTES)
            .ok_or(SdfBridgeError::ArithmeticOverflow)?;
        let sample_bytes = samples
            .checked_mul(WORLD_SDF_SAMPLE_BYTES)
            .ok_or(SdfBridgeError::ArithmeticOverflow)?;
        let bytes = WORLD_SDF_HEADER_BYTES
            .checked_add(metadata_bytes)
            .ok_or(SdfBridgeError::ArithmeticOverflow)?
            .checked_add(sample_bytes)
            .ok_or(SdfBridgeError::ArithmeticOverflow)?;
        if bytes > self.budget.max_bytes {
            return Err(SdfBridgeError::UploadBudgetExceeded {
                required: bytes,
                limit: self.budget.max_bytes,
            });
        }
        Ok(SdfPhysicsSnapshot {
            source,
            logical_bytes: bytes,
            publication: Arc::new(()),
        })
    }

    /// Replaces the current snapshot only after complete candidate validation.
    pub fn publish(
        &mut self,
        source: WorldSdfSnapshot,
    ) -> Result<&SdfPhysicsSnapshot, SdfBridgeError> {
        if let Some(current) = &self.current {
            if source.revision <= current.revision() {
                return Err(SdfBridgeError::StaleRevision {
                    current: current.revision(),
                    incoming: source.revision,
                });
            }
        }
        let staged = self.stage(source)?;
        self.current = Some(staged);
        Ok(self
            .current
            .as_ref()
            .expect("staged snapshot was published"))
    }

    /// Removes the disposable derived snapshot.
    pub fn evict(&mut self) -> Option<SdfPhysicsSnapshot> {
        self.current.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_world_residency::{BrickCoord, SdfQueryConsumer};
    use std::vec;

    fn source(revision: u64, value: i16) -> WorldSdfSnapshot {
        let mut grid = SparseSdfBrickGrid::new(2, 1).expect("valid grid");
        grid.insert_brick(BrickCoord::new(-1, 0, 0, 0), vec![value; 8])
            .expect("valid brick");
        WorldSdfSnapshot::new(
            WorldSdfTransform::new([100, 200, -300], 500, 2).expect("valid transform"),
            revision,
            grid,
        )
    }

    #[test]
    fn query_preserves_value_and_unknown_semantics() {
        let budget = SdfPhysicsBridgeBudget::new(1, 8, 96).expect("valid budget");
        let mut bridge = WorldSdfPhysicsBridge::new(budget);
        bridge.publish(source(1, -4)).expect("publish");
        let query = bridge.current().expect("current").query();
        assert_eq!(
            query.query_sample([-2, 0, 0], 0),
            Ok(SdfQueryResult::Value(-4))
        );
        assert_eq!(
            query.query_sample([0, 0, 0], 0),
            Ok(SdfQueryResult::Unknown)
        );
    }

    #[test]
    fn failed_stage_and_stale_revision_keep_old_snapshot() {
        let budget = SdfPhysicsBridgeBudget::new(1, 8, 96).expect("valid budget");
        let mut bridge = WorldSdfPhysicsBridge::new(budget);
        bridge.publish(source(2, -2)).expect("publish");
        let before = bridge.current().cloned().expect("current");
        assert_eq!(
            bridge.publish(source(2, 4)),
            Err(SdfBridgeError::StaleRevision {
                current: 2,
                incoming: 2
            })
        );
        let constrained = WorldSdfPhysicsBridge::new(
            SdfPhysicsBridgeBudget::new(0, 0, WORLD_SDF_HEADER_BYTES).expect("empty budget"),
        );
        assert_eq!(
            constrained.stage(source(1, 1)),
            Err(SdfBridgeError::BrickCapacityExceeded {
                required: 1,
                capacity: 0
            })
        );
        assert_eq!(bridge.current(), Some(&before));
    }

    #[test]
    fn transform_and_byte_budget_are_checked_before_publication() {
        assert_eq!(
            WorldSdfTransform::new([0, 0, 0], 0, 1),
            Err(SdfBridgeError::InvalidTransform)
        );
        let bridge = WorldSdfPhysicsBridge::new(
            SdfPhysicsBridgeBudget::new(1, 8, WORLD_SDF_HEADER_BYTES).expect("valid budget"),
        );
        assert_eq!(
            bridge.stage(source(1, 1)),
            Err(SdfBridgeError::UploadBudgetExceeded {
                required: 80,
                limit: WORLD_SDF_HEADER_BYTES
            })
        );
    }

    #[test]
    fn world_space_queries_use_floor_voxels_and_distance_scale() {
        let budget = SdfPhysicsBridgeBudget::new(1, 8, 96).expect("valid budget");
        let mut bridge = WorldSdfPhysicsBridge::new(budget);
        bridge.publish(source(1, -4)).expect("publish");
        let snapshot = bridge.current().expect("resident snapshot");
        assert_eq!(
            snapshot.query_world_distance_mm([-900, 200, -300], 0),
            Ok(SdfQueryResult::Value(-8))
        );
        assert_eq!(
            snapshot.query_world_distance_mm([100, 200, -300], 0),
            Ok(SdfQueryResult::Unknown)
        );
    }
}
