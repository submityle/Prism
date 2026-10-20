//! Loss-bounded bridge from baked world terrain to the physics heightfield ABI.
//!
//! `prism_world_terrain` stores signed millimetre samples and a planar embedding;
//! `prism_physics_gpu` consumes a regular `f32` heightfield in metres. This crate
//! owns that conversion boundary so callers do not duplicate units, revision,
//! capacity, or replacement policy. The authoritative terrain snapshot remains
//! in the world provider; the returned physics heightfield is disposable derived
//! state and can be rebuilt after eviction.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod sdf;

pub use sdf::{
    SdfBounds, SdfBridgeError, SdfDistanceGradient, SdfPhysicsBridgeBudget, SdfPhysicsSnapshot,
    SdfQueryResult, SdfRebakeError, SdfRebakeLimits, SdfRebakePoint, SdfRebakeRegion,
    SdfRebakeReport, SdfRebakeResource, WorldSdfPhysicsBridge, WorldSdfSnapshot, WorldSdfTransform,
    WORLD_SDF_BRICK_METADATA_BYTES, WORLD_SDF_HEADER_BYTES, WORLD_SDF_SAMPLE_BYTES,
};

use std::{fmt, vec::Vec};

use prism_physics_gpu::Heightfield;
use prism_world_terrain::{HeightfieldTile, PlanarHeightfieldEmbedding, TileKey};

/// Bytes represented by one physics GPU field header.
pub const PHYSICS_HEIGHTFIELD_HEADER_BYTES: usize = 32;
/// Bytes represented by one uploaded `f32` height sample.
pub const PHYSICS_HEIGHTFIELD_SAMPLE_BYTES: usize = 4;

/// Fixed limits for one derived physics heightfield snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldBridgeBudget {
    /// Maximum number of samples in the derived field.
    pub max_samples: usize,
    /// Maximum logical physics upload bytes, including the 32-byte field header.
    pub max_upload_bytes: usize,
}

impl HeightfieldBridgeBudget {
    /// Creates a budget after validating its byte arithmetic and nonzero limits.
    pub const fn new(
        max_samples: usize,
        max_upload_bytes: usize,
    ) -> Result<Self, HeightfieldBridgeError> {
        if max_samples == 0 || max_upload_bytes < PHYSICS_HEIGHTFIELD_HEADER_BYTES {
            return Err(HeightfieldBridgeError::InvalidBudget);
        }
        let Some(required) = max_samples.checked_mul(PHYSICS_HEIGHTFIELD_SAMPLE_BYTES) else {
            return Err(HeightfieldBridgeError::ArithmeticOverflow);
        };
        let Some(required) = required.checked_add(PHYSICS_HEIGHTFIELD_HEADER_BYTES) else {
            return Err(HeightfieldBridgeError::ArithmeticOverflow);
        };
        if required > max_upload_bytes {
            return Err(HeightfieldBridgeError::BudgetTooSmall {
                required,
                limit: max_upload_bytes,
            });
        }
        Ok(Self {
            max_samples,
            max_upload_bytes,
        })
    }
}

/// Aggregate limits for a resident collection of derived heightfields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldResidencyBudget {
    /// Maximum number of simultaneously resident tiles.
    pub max_tiles: usize,
    /// Maximum sum of samples across resident tiles.
    pub max_samples: usize,
    /// Maximum sum of logical upload bytes across resident tiles.
    pub max_upload_bytes: usize,
}

impl HeightfieldResidencyBudget {
    /// Creates aggregate limits with checked, nonzero capacities.
    pub const fn new(
        max_tiles: usize,
        max_samples: usize,
        max_upload_bytes: usize,
    ) -> Result<Self, HeightfieldBridgeError> {
        if max_tiles == 0 || max_samples == 0 || max_upload_bytes < PHYSICS_HEIGHTFIELD_HEADER_BYTES
        {
            return Err(HeightfieldBridgeError::InvalidBudget);
        }
        Ok(Self {
            max_tiles,
            max_samples,
            max_upload_bytes,
        })
    }
}

/// Identity and immutable derived data for one published world terrain tile.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicsHeightfieldSnapshot {
    tile: TileKey,
    revision: u64,
    embedding: PlanarHeightfieldEmbedding,
    heightfield: Heightfield,
    upload_bytes: usize,
}

impl PhysicsHeightfieldSnapshot {
    /// Source tile identity.
    #[must_use]
    pub const fn tile(&self) -> TileKey {
        self.tile
    }

    /// Source terrain revision.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Exact planar embedding used for the conversion.
    #[must_use]
    pub const fn embedding(&self) -> PlanarHeightfieldEmbedding {
        self.embedding
    }

    /// Disposable physics consumer representation.
    #[must_use]
    pub fn heightfield(&self) -> &Heightfield {
        &self.heightfield
    }

    /// Logical bytes consumed by the physics GPU uploader.
    #[must_use]
    pub const fn upload_bytes(&self) -> usize {
        self.upload_bytes
    }
}

/// Error raised before a bridge replacement mutates the current snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeightfieldBridgeError {
    /// A zero sample/byte budget was supplied.
    InvalidBudget,
    /// A checked size or unit conversion overflowed.
    ArithmeticOverflow,
    /// The configured byte limit cannot hold its sample limit.
    BudgetTooSmall {
        /// Required bytes for the requested sample limit.
        required: usize,
        /// Configured maximum bytes.
        limit: usize,
    },
    /// The baked tile sample count does not match its validated square spec.
    SampleCountMismatch {
        /// Expected square sample count.
        expected: usize,
        /// Actual sample count.
        actual: usize,
    },
    /// The tile needs more samples than this bridge admits.
    SampleCapacityExceeded {
        /// Number of samples in the tile.
        required: usize,
        /// Configured sample capacity.
        capacity: usize,
    },
    /// The converted physics upload exceeds the independent byte budget.
    UploadBudgetExceeded {
        /// Required logical upload bytes.
        required: usize,
        /// Configured maximum bytes.
        limit: usize,
    },
    /// The planar embedding has no positive spacing.
    InvalidSpacing,
    /// A millimetre value could not be represented as finite metres.
    NonFiniteConversion,
    /// A same-tile revision was replaced by an equal or older revision.
    StaleRevision {
        /// Existing resident revision.
        current: u64,
        /// Rejected incoming revision.
        incoming: u64,
    },
    /// The host allocation for the derived sample vector failed.
    AllocationFailed,
    /// The resident tile count would exceed the aggregate limit.
    ResidentTileCapacityExceeded {
        /// Number of resident tiles after the attempted publish.
        required: usize,
        /// Configured resident tile capacity.
        capacity: usize,
    },
    /// The resident sample sum would exceed the aggregate limit.
    ResidentSampleBudgetExceeded {
        /// Sample sum after the attempted publish.
        required: usize,
        /// Configured resident sample limit.
        limit: usize,
    },
    /// The resident upload-byte sum would exceed the aggregate limit.
    ResidentUploadBudgetExceeded {
        /// Upload-byte sum after the attempted publish.
        required: usize,
        /// Configured resident upload-byte limit.
        limit: usize,
    },
}

impl fmt::Display for HeightfieldBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBudget => f.write_str("heightfield bridge budget is invalid"),
            Self::ArithmeticOverflow => f.write_str("heightfield bridge arithmetic overflow"),
            Self::BudgetTooSmall { required, limit } => {
                write!(
                    f,
                    "heightfield bridge budget needs {required} bytes, limit {limit}"
                )
            }
            Self::SampleCountMismatch { expected, actual } => {
                write!(f, "heightfield has {actual} samples, expected {expected}")
            }
            Self::SampleCapacityExceeded { required, capacity } => {
                write!(
                    f,
                    "heightfield needs {required} samples, capacity {capacity}"
                )
            }
            Self::UploadBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "heightfield upload needs {required} bytes, limit {limit}"
                )
            }
            Self::InvalidSpacing => f.write_str("heightfield embedding spacing must be positive"),
            Self::NonFiniteConversion => {
                f.write_str("millimetre-to-metre heightfield conversion is non-finite")
            }
            Self::StaleRevision { current, incoming } => {
                write!(
                    f,
                    "stale heightfield revision {incoming}, current is {current}"
                )
            }
            Self::AllocationFailed => f.write_str("heightfield bridge allocation failed"),
            Self::ResidentTileCapacityExceeded { required, capacity } => {
                write!(
                    f,
                    "resident heightfield tiles need {required}, capacity {capacity}"
                )
            }
            Self::ResidentSampleBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "resident heightfield samples need {required}, limit {limit}"
                )
            }
            Self::ResidentUploadBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "resident heightfield upload needs {required} bytes, limit {limit}"
                )
            }
        }
    }
}

impl std::error::Error for HeightfieldBridgeError {}

/// Owns the currently published derived heightfield and replacement policy.
#[derive(Clone, Debug)]
pub struct WorldHeightfieldPhysicsBridge {
    budget: HeightfieldBridgeBudget,
    current: Option<PhysicsHeightfieldSnapshot>,
}

impl WorldHeightfieldPhysicsBridge {
    /// Creates an empty bridge with fixed sample and upload limits.
    pub const fn new(budget: HeightfieldBridgeBudget) -> Self {
        Self {
            budget,
            current: None,
        }
    }

    /// Returns the immutable bridge budget.
    #[must_use]
    pub const fn budget(&self) -> HeightfieldBridgeBudget {
        self.budget
    }

    /// Returns the currently published derived snapshot, if any.
    #[must_use]
    pub fn current(&self) -> Option<&PhysicsHeightfieldSnapshot> {
        self.current.as_ref()
    }

    /// Publishes a complete terrain snapshot atomically.
    ///
    /// All validation, conversion, and fallible allocation happen before the
    /// old snapshot is replaced. Equal or older revisions are rejected only
    /// when they target the same tile; publishing another tile is a normal
    /// replacement after its complete snapshot has been validated.
    pub fn publish(
        &mut self,
        tile: &HeightfieldTile,
        embedding: PlanarHeightfieldEmbedding,
    ) -> Result<&PhysicsHeightfieldSnapshot, HeightfieldBridgeError> {
        if let Some(current) = &self.current {
            if current.tile == tile.tile() && tile.revision() <= current.revision {
                return Err(HeightfieldBridgeError::StaleRevision {
                    current: current.revision,
                    incoming: tile.revision(),
                });
            }
        }
        let next = convert(tile, embedding, self.budget)?;
        self.current = Some(next);
        Ok(self.current.as_ref().expect("published snapshot exists"))
    }

    /// Removes the derived snapshot after the owning world cell is evicted.
    pub fn evict(&mut self) -> Option<PhysicsHeightfieldSnapshot> {
        self.current.take()
    }
}

/// Owns a deterministically ordered, bounded set of derived heightfields.
///
/// The vector is kept sorted by [`TileKey`] so physics consumers can derive
/// stable heightfield indices without a hash-map iteration order. Replacing a
/// tile never changes the number of resident entries; publishing a new tile
/// checks all aggregate budgets before mutating the set.
#[derive(Clone, Debug)]
pub struct WorldHeightfieldPhysicsResidency {
    per_tile_budget: HeightfieldBridgeBudget,
    budget: HeightfieldResidencyBudget,
    snapshots: Vec<PhysicsHeightfieldSnapshot>,
    resident_samples: usize,
    resident_upload_bytes: usize,
}

impl WorldHeightfieldPhysicsResidency {
    /// Creates an empty residency set with per-tile and aggregate limits.
    #[must_use]
    pub const fn new(
        per_tile_budget: HeightfieldBridgeBudget,
        budget: HeightfieldResidencyBudget,
    ) -> Self {
        Self {
            per_tile_budget,
            budget,
            snapshots: Vec::new(),
            resident_samples: 0,
            resident_upload_bytes: 0,
        }
    }

    /// Returns the per-tile conversion budget.
    #[must_use]
    pub fn per_tile_budget(&self) -> HeightfieldBridgeBudget {
        self.per_tile_budget
    }

    /// Returns the aggregate residency budget.
    #[must_use]
    pub fn budget(&self) -> HeightfieldResidencyBudget {
        self.budget
    }

    /// Returns resident snapshots in stable [`TileKey`] order.
    #[must_use]
    pub fn snapshots(&self) -> &[PhysicsHeightfieldSnapshot] {
        &self.snapshots
    }

    /// Returns the derived snapshot for one tile, if resident.
    #[must_use]
    pub fn snapshot(&self, tile: TileKey) -> Option<&PhysicsHeightfieldSnapshot> {
        self.snapshots
            .binary_search_by_key(&tile, PhysicsHeightfieldSnapshot::tile)
            .ok()
            .map(|index| &self.snapshots[index])
    }

    /// Returns the aggregate sample count currently resident.
    #[must_use]
    pub const fn resident_samples(&self) -> usize {
        self.resident_samples
    }

    /// Returns the aggregate logical upload bytes currently resident.
    #[must_use]
    pub const fn resident_upload_bytes(&self) -> usize {
        self.resident_upload_bytes
    }

    /// Publishes or replaces one tile after complete preflight.
    pub fn publish(
        &mut self,
        tile: &HeightfieldTile,
        embedding: PlanarHeightfieldEmbedding,
    ) -> Result<&PhysicsHeightfieldSnapshot, HeightfieldBridgeError> {
        let plan = conversion_plan(tile, embedding, self.per_tile_budget())?;
        let tile_key = tile.tile();
        let position = self
            .snapshots
            .binary_search_by_key(&tile_key, PhysicsHeightfieldSnapshot::tile);
        let (insert_index, old_revision, old_samples, old_upload_bytes) = match position {
            Ok(index) => {
                let current = &self.snapshots[index];
                (
                    index,
                    Some(current.revision()),
                    Some(
                        current.heightfield().rows() as usize
                            * current.heightfield().cols() as usize,
                    ),
                    Some(current.upload_bytes()),
                )
            }
            Err(index) => (index, None, None, None),
        };
        if let Some(current_revision) = old_revision {
            if tile.revision() <= current_revision {
                return Err(HeightfieldBridgeError::StaleRevision {
                    current: current_revision,
                    incoming: tile.revision(),
                });
            }
        }

        let old_samples = old_samples.unwrap_or(0);
        let old_upload_bytes = old_upload_bytes.unwrap_or(0);
        let resident_samples = self
            .resident_samples
            .checked_sub(old_samples)
            .and_then(|value| value.checked_add(plan.sample_count))
            .ok_or(HeightfieldBridgeError::ArithmeticOverflow)?;
        let resident_upload_bytes = self
            .resident_upload_bytes
            .checked_sub(old_upload_bytes)
            .and_then(|value| value.checked_add(plan.upload_bytes))
            .ok_or(HeightfieldBridgeError::ArithmeticOverflow)?;
        let budget = self.budget();
        if old_revision.is_none() {
            let required = self
                .snapshots
                .len()
                .checked_add(1)
                .ok_or(HeightfieldBridgeError::ArithmeticOverflow)?;
            if required > budget.max_tiles {
                return Err(HeightfieldBridgeError::ResidentTileCapacityExceeded {
                    required,
                    capacity: budget.max_tiles,
                });
            }
        }
        if resident_samples > budget.max_samples {
            return Err(HeightfieldBridgeError::ResidentSampleBudgetExceeded {
                required: resident_samples,
                limit: budget.max_samples,
            });
        }
        if resident_upload_bytes > budget.max_upload_bytes {
            return Err(HeightfieldBridgeError::ResidentUploadBudgetExceeded {
                required: resident_upload_bytes,
                limit: budget.max_upload_bytes,
            });
        }
        let next = convert_planned(tile, embedding, plan)?;
        if old_revision.is_none() {
            self.snapshots
                .try_reserve(1)
                .map_err(|_| HeightfieldBridgeError::AllocationFailed)?;
        }

        if old_revision.is_some() {
            self.snapshots[insert_index] = next;
            self.resident_samples = resident_samples;
            self.resident_upload_bytes = resident_upload_bytes;
            Ok(&self.snapshots[insert_index])
        } else {
            self.snapshots.insert(insert_index, next);
            self.resident_samples = resident_samples;
            self.resident_upload_bytes = resident_upload_bytes;
            Ok(&self.snapshots[insert_index])
        }
    }

    /// Evicts one tile and releases only its derived physics state.
    pub fn evict(&mut self, tile: TileKey) -> Option<PhysicsHeightfieldSnapshot> {
        let index = self
            .snapshots
            .binary_search_by_key(&tile, PhysicsHeightfieldSnapshot::tile)
            .ok()?;
        let snapshot = self.snapshots.remove(index);
        self.resident_samples -=
            snapshot.heightfield().rows() as usize * snapshot.heightfield().cols() as usize;
        self.resident_upload_bytes -= snapshot.upload_bytes();
        Some(snapshot)
    }
}

fn convert(
    tile: &HeightfieldTile,
    embedding: PlanarHeightfieldEmbedding,
    budget: HeightfieldBridgeBudget,
) -> Result<PhysicsHeightfieldSnapshot, HeightfieldBridgeError> {
    let plan = conversion_plan(tile, embedding, budget)?;
    convert_planned(tile, embedding, plan)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ConversionPlan {
    side: usize,
    sample_count: usize,
    upload_bytes: usize,
}

fn conversion_plan(
    tile: &HeightfieldTile,
    embedding: PlanarHeightfieldEmbedding,
    budget: HeightfieldBridgeBudget,
) -> Result<ConversionPlan, HeightfieldBridgeError> {
    if embedding.sample_spacing_mm == 0 {
        return Err(HeightfieldBridgeError::InvalidSpacing);
    }
    let side = usize::try_from(tile.spec().samples_per_edge())
        .map_err(|_| HeightfieldBridgeError::ArithmeticOverflow)?;
    let expected = side
        .checked_mul(side)
        .ok_or(HeightfieldBridgeError::ArithmeticOverflow)?;
    if tile.samples().len() != expected {
        return Err(HeightfieldBridgeError::SampleCountMismatch {
            expected,
            actual: tile.samples().len(),
        });
    }
    if expected > budget.max_samples {
        return Err(HeightfieldBridgeError::SampleCapacityExceeded {
            required: expected,
            capacity: budget.max_samples,
        });
    }
    let sample_bytes = expected
        .checked_mul(PHYSICS_HEIGHTFIELD_SAMPLE_BYTES)
        .ok_or(HeightfieldBridgeError::ArithmeticOverflow)?;
    let upload_bytes = PHYSICS_HEIGHTFIELD_HEADER_BYTES
        .checked_add(sample_bytes)
        .ok_or(HeightfieldBridgeError::ArithmeticOverflow)?;
    if upload_bytes > budget.max_upload_bytes {
        return Err(HeightfieldBridgeError::UploadBudgetExceeded {
            required: upload_bytes,
            limit: budget.max_upload_bytes,
        });
    }

    Ok(ConversionPlan {
        side,
        sample_count: expected,
        upload_bytes,
    })
}

fn convert_planned(
    tile: &HeightfieldTile,
    embedding: PlanarHeightfieldEmbedding,
    plan: ConversionPlan,
) -> Result<PhysicsHeightfieldSnapshot, HeightfieldBridgeError> {
    let metres = |millimetres: i128| {
        let value = millimetres as f64 * 0.001;
        let value = value as f32;
        value
            .is_finite()
            .then_some(value)
            .ok_or(HeightfieldBridgeError::NonFiniteConversion)
    };
    let origin = [
        metres(i128::from(embedding.sample_origin_mm[0]))?,
        metres(i128::from(embedding.sample_origin_mm[1]))?,
        metres(i128::from(embedding.sample_origin_mm[2]))?,
    ];
    let cell_size = metres(i128::from(embedding.sample_spacing_mm))?;
    let scale = i128::from(tile.spec().height_scale_mm());
    let mut heights = Vec::new();
    heights
        .try_reserve_exact(plan.sample_count)
        .map_err(|_| HeightfieldBridgeError::AllocationFailed)?;
    for &sample in tile.samples() {
        heights.push(metres(i128::from(sample) * scale)?);
    }
    Ok(PhysicsHeightfieldSnapshot {
        tile: tile.tile(),
        revision: tile.revision(),
        embedding,
        heightfield: Heightfield::new(
            u32::try_from(plan.side).map_err(|_| HeightfieldBridgeError::ArithmeticOverflow)?,
            u32::try_from(plan.side).map_err(|_| HeightfieldBridgeError::ArithmeticOverflow)?,
            cell_size,
            glam::Vec3::from_array(origin),
            heights,
        ),
        upload_bytes: plan.upload_bytes,
    })
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use glam::Vec3;
    use prism_physics_gpu::{cpu_sphere_heightfield_narrowphase, Particle};
    use prism_world_terrain::{
        BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec,
        TileKey,
    };

    use super::*;

    struct RampSampler;

    impl HeightSampler for RampSampler {
        type Error = Infallible;

        fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            Ok(HeightSampleValue::from_millimetres(
                i64::from(sample.u_q16() / 65_535) * 100 + i64::from(sample.v_q16() / 65_535) * 200,
            ))
        }
    }

    fn tile_at(tile_key: TileKey, revision: u64) -> HeightfieldTile {
        let mut tile = HeightfieldTile::bake(
            tile_key,
            HeightfieldSpec::new(3, 1).unwrap(),
            &RampSampler,
            BakeLimits {
                max_samples: 9,
                max_bytes: 18,
            },
        )
        .unwrap();
        if revision != 0 {
            tile.rebake(
                prism_world_terrain::DirtyRegion::new(tile.tile(), 0, 0, u16::MAX, u16::MAX)
                    .unwrap(),
                &RampSampler,
                BakeLimits {
                    max_samples: 9,
                    max_bytes: 18,
                },
            )
            .unwrap();
        }
        tile
    }

    fn tile(revision: u64) -> HeightfieldTile {
        tile_at(TileKey::root(CubeFace::PositiveY), revision)
    }

    fn embedding() -> PlanarHeightfieldEmbedding {
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [1_000, 2_000, -3_000],
            sample_spacing_mm: 500,
        }
    }

    #[test]
    fn publishes_mm_tile_as_meter_physics_heightfield_with_identity() {
        let budget = HeightfieldBridgeBudget::new(9, 68).unwrap();
        let mut bridge = WorldHeightfieldPhysicsBridge::new(budget);
        let snapshot = bridge.publish(&tile(0), embedding()).unwrap();

        assert_eq!(snapshot.tile(), TileKey::root(CubeFace::PositiveY));
        assert_eq!(snapshot.revision(), 0);
        assert_eq!(snapshot.heightfield().rows(), 3);
        assert_eq!(snapshot.heightfield().cols(), 3);
        assert_eq!(snapshot.heightfield().cell_size(), 0.5);
        assert_eq!(snapshot.heightfield().origin(), Vec3::new(1.0, 2.0, -3.0));
        assert_eq!(snapshot.upload_bytes(), 68);
    }

    #[test]
    fn converted_snapshot_is_consumable_by_physics_cpu_narrowphase() {
        let budget = HeightfieldBridgeBudget::new(9, 68).unwrap();
        let mut bridge = WorldHeightfieldPhysicsBridge::new(budget);
        let snapshot = bridge.publish(&tile(0), embedding()).unwrap();
        let spheres = [Particle::new(Vec3::new(1.25, 2.02, -2.75), 0.05)];
        let pairs = [prism_physics_gpu::HeightfieldSpherePair::new(0, 0)];

        let contacts =
            cpu_sphere_heightfield_narrowphase(&spheres, &[snapshot.heightfield().clone()], &pairs);

        assert!(contacts[0].is_some());
        assert!(contacts[0].unwrap().depth >= 0.0);
    }

    #[test]
    fn stale_same_tile_revision_and_failed_budget_keep_old_snapshot() {
        let budget = HeightfieldBridgeBudget::new(9, 68).unwrap();
        let mut bridge = WorldHeightfieldPhysicsBridge::new(budget);
        bridge.publish(&tile(0), embedding()).unwrap();
        let before = bridge.current().unwrap().clone();

        assert_eq!(
            bridge.publish(&tile(0), embedding()),
            Err(HeightfieldBridgeError::StaleRevision {
                current: 0,
                incoming: 0
            })
        );
        let small = HeightfieldBridgeBudget::new(4, 48).unwrap();
        let mut constrained = WorldHeightfieldPhysicsBridge::new(small);
        let error = constrained.publish(&tile(0), embedding()).unwrap_err();
        assert_eq!(
            error,
            HeightfieldBridgeError::SampleCapacityExceeded {
                required: 9,
                capacity: 4
            }
        );
        assert_eq!(constrained.current(), None);
        assert_eq!(bridge.current().unwrap(), &before);
    }

    #[test]
    fn eviction_removes_only_derived_state() {
        let budget = HeightfieldBridgeBudget::new(9, 68).unwrap();
        let mut bridge = WorldHeightfieldPhysicsBridge::new(budget);
        bridge.publish(&tile(0), embedding()).unwrap();
        assert!(bridge.evict().is_some());
        assert!(bridge.current().is_none());
    }

    #[test]
    fn residency_is_sorted_bounded_and_revision_aware() {
        let per_tile = HeightfieldBridgeBudget::new(9, 68).unwrap();
        let residency_budget = HeightfieldResidencyBudget::new(2, 18, 136).unwrap();
        let mut residency = WorldHeightfieldPhysicsResidency::new(per_tile, residency_budget);
        let positive_y = TileKey::root(CubeFace::PositiveY);
        let negative_y = TileKey::root(CubeFace::NegativeY);

        residency
            .publish(&tile_at(positive_y, 0), embedding())
            .unwrap();
        residency
            .publish(&tile_at(negative_y, 0), embedding())
            .unwrap();

        assert_eq!(residency.snapshots().len(), 2);
        assert!(residency.snapshots()[0].tile() < residency.snapshots()[1].tile());
        assert_eq!(residency.resident_samples(), 18);
        assert_eq!(residency.resident_upload_bytes(), 136);
        assert!(matches!(
            residency.publish(&tile_at(positive_y, 0), embedding()),
            Err(HeightfieldBridgeError::StaleRevision { .. })
        ));

        let evicted = residency.evict(negative_y).unwrap();
        assert_eq!(evicted.tile(), negative_y);
        assert_eq!(residency.snapshots().len(), 1);
        assert_eq!(residency.resident_samples(), 9);
        assert_eq!(residency.resident_upload_bytes(), 68);
    }

    #[test]
    fn residency_rejects_aggregate_capacity_without_mutation() {
        let per_tile = HeightfieldBridgeBudget::new(9, 68).unwrap();
        let residency_budget = HeightfieldResidencyBudget::new(1, 9, 68).unwrap();
        let mut residency = WorldHeightfieldPhysicsResidency::new(per_tile, residency_budget);
        let first = TileKey::root(CubeFace::PositiveY);
        let second = TileKey::root(CubeFace::NegativeY);
        residency.publish(&tile_at(first, 0), embedding()).unwrap();

        let error = residency
            .publish(&tile_at(second, 0), embedding())
            .unwrap_err();
        assert_eq!(
            error,
            HeightfieldBridgeError::ResidentTileCapacityExceeded {
                required: 2,
                capacity: 1,
            }
        );
        assert_eq!(residency.snapshots().len(), 1);
        assert!(residency.snapshot(first).is_some());
        assert!(residency.snapshot(second).is_none());
    }

    #[test]
    fn residency_rejects_sample_and_upload_aggregate_limits_before_publish() {
        let per_tile = HeightfieldBridgeBudget::new(9, 68).unwrap();
        let first = TileKey::root(CubeFace::PositiveY);
        let second = TileKey::root(CubeFace::NegativeY);

        let mut sample_limited = WorldHeightfieldPhysicsResidency::new(
            per_tile,
            HeightfieldResidencyBudget::new(2, 9, 136).unwrap(),
        );
        sample_limited
            .publish(&tile_at(first, 0), embedding())
            .unwrap();
        assert_eq!(
            sample_limited
                .publish(&tile_at(second, 0), embedding())
                .unwrap_err(),
            HeightfieldBridgeError::ResidentSampleBudgetExceeded {
                required: 18,
                limit: 9,
            }
        );
        assert_eq!(sample_limited.snapshots().len(), 1);

        let mut upload_limited = WorldHeightfieldPhysicsResidency::new(
            per_tile,
            HeightfieldResidencyBudget::new(2, 18, 68).unwrap(),
        );
        upload_limited
            .publish(&tile_at(first, 0), embedding())
            .unwrap();
        assert_eq!(
            upload_limited
                .publish(&tile_at(second, 0), embedding())
                .unwrap_err(),
            HeightfieldBridgeError::ResidentUploadBudgetExceeded {
                required: 136,
                limit: 68,
            }
        );
        assert_eq!(upload_limited.snapshots().len(), 1);
    }
}
