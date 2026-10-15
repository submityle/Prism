//! Content-neutral CPU deletion thinning for projected heightfield placements.
//!
//! A filtered cache depends on the complete source cache key, the complete
//! deletion set (including absent keys), and
//! [`HEIGHTFIELD_PLACEMENT_DELETION_VERSION`]. Source cache identity is never
//! rewritten. Placement identity includes its own version and the full canonical
//! source key; raw three-dimensional scatter deletion sets are not substitutes.
//!
//! Key Vec payload budgets include spare capacity, but exclude Vec/Arc headers,
//! allocator metadata, terrain/field snapshots and simultaneous old/new sets.
//! Vec reservations are fallible; Arc creation uses normal allocator OOM
//! handling rather than returning [`HeightfieldPlacementDeletionError::AllocationFailed`].
//! This module owns no persistence encoding, runtime, GPU or content policy.

use alloc::{sync::Arc, vec::Vec};
use core::{fmt, mem::size_of};

use crate::{
    HeightfieldPlacementError, HeightfieldPlacementKey, HeightfieldPlacementOutput,
    HEIGHTFIELD_PLACEMENT_VERSION,
};

/// Revision of full placement deletion identity and stable filtering semantics.
pub const HEIGHTFIELD_PLACEMENT_DELETION_VERSION: u32 = 1;

/// Limits on a supplied or newly allocated key Vec, excluding Arc metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldPlacementDeletionBudget {
    /// Maximum input count before deduplication, or result count after an edit.
    pub max_keys: usize,
    /// Maximum Vec capacity times `size_of::<HeightfieldPlacementKey>()`.
    pub max_key_bytes: usize,
}

/// Validated, sorted, unique full placement keys, immutably shared by value.
///
/// Cloning is O(1); equality compares full keys, not hashes or Arc addresses.
/// Construction reuses the supplied Vec and its spare capacity. Default is empty.
/// Keys need not be covered by any terrain or survive any placement recipe.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HeightfieldPlacementDeletionSet(Arc<Vec<HeightfieldPlacementKey>>);

/// Survivors plus the complete additional structural deletion cache dependency.
///
/// Consumers must also include [`HEIGHTFIELD_PLACEMENT_DELETION_VERSION`] in
/// their filtered-cache dependency; it is not a source placement version.
#[derive(Debug, Eq, PartialEq)]
pub struct DeletionFilteredHeightfieldPlacement {
    /// Original output with aligned rows removed, retaining both Vec allocations,
    /// the full source cache key and all upstream work counters unchanged.
    pub source: HeightfieldPlacementOutput,
    /// Complete immutable deletion input, including keys absent from this output.
    pub deletions: HeightfieldPlacementDeletionSet,
    /// Actual number of rows removed, not deletion-set cardinality.
    pub removed: usize,
}

/// Invalid identity, hard-budget violation or recoverable key Vec failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeightfieldPlacementDeletionError {
    /// Unsupported placement identity or failed raw source reconstruction.
    InvalidKey(HeightfieldPlacementError),
    /// Input count before deduplication or edited result count exceeds the limit.
    KeyBudgetExceeded {
        /// Required key count.
        required: usize,
        /// Maximum permitted key count.
        limit: usize,
    },
    /// Key Vec capacity payload, including spare capacity, exceeds the limit.
    KeyBytesBudgetExceeded {
        /// Required capacity payload bytes.
        required: usize,
        /// Maximum permitted payload bytes.
        limit: usize,
    },
    /// Key count or capacity byte multiplication overflows `usize`.
    ArithmeticOverflow,
    /// Fallible key Vec reservation failed, including an impossible layout.
    AllocationFailed,
    /// Placement points and author-field attributes have different lengths.
    MisalignedAttributes,
}

impl fmt::Display for HeightfieldPlacementDeletionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKey(error) => {
                write!(f, "invalid heightfield placement deletion key: {error}")
            }
            Self::KeyBudgetExceeded { required, limit } => write!(
                f,
                "heightfield placement deletions need {required} keys, limit {limit}"
            ),
            Self::KeyBytesBudgetExceeded { required, limit } => write!(
                f,
                "heightfield placement deletion Vec needs {required} bytes, limit {limit}"
            ),
            Self::ArithmeticOverflow => {
                f.write_str("heightfield placement deletion count or byte size overflow")
            }
            Self::AllocationFailed => {
                f.write_str("heightfield placement deletion key reservation failed")
            }
            Self::MisalignedAttributes => {
                f.write_str("heightfield placement points and attributes have different lengths")
            }
        }
    }
}

impl core::error::Error for HeightfieldPlacementDeletionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::InvalidKey(error) => Some(error),
            _ => None,
        }
    }
}

impl HeightfieldPlacementDeletionSet {
    /// Checks original length/capacity budgets and validates every full key,
    /// then sorts and deduplicates in place without another key buffer.
    ///
    /// Duplicate inputs count before deduplication; spare capacity is not shrunk.
    /// Validation checks placement version, canonical grid Y and raw source
    /// reconstruction only: it needs no terrain, projection or request coverage.
    pub fn new(
        mut keys: Vec<HeightfieldPlacementKey>,
        budget: HeightfieldPlacementDeletionBudget,
    ) -> Result<Self, HeightfieldPlacementDeletionError> {
        check_budget(keys.len(), keys.capacity(), budget)?;
        for &key in &keys {
            validate_key(key)?;
        }
        keys.sort_unstable();
        keys.dedup();
        Ok(Self(Arc::new(keys)))
    }

    /// Canonical complete keys in ascending structural order, without duplicates.
    pub fn keys(&self) -> &[HeightfieldPlacementKey] {
        self.0.as_slice()
    }

    /// Exact full-key membership via binary search, without geometry validation.
    pub fn is_deleted(&self, key: HeightfieldPlacementKey) -> bool {
        self.keys().binary_search(&key).is_ok()
    }

    /// Owned key Vec capacity payload, excluding Vec/Arc and allocator metadata.
    pub fn allocated_key_bytes(&self) -> usize {
        // Every path publishing a set has checked this multiplication.
        self.0.capacity() * size_of::<HeightfieldPlacementKey>()
    }

    /// Returns an immutable edited set, leaving this set and all clones intact.
    ///
    /// The key is validated even for a no-op. Unchanged membership returns an
    /// O(1) clone without allocation or reapplying budgets to existing data.
    /// Actual edits check the new result count, fallibly reserve a fresh Vec,
    /// check its actual capacity, and copy keys in canonical order. Budgets
    /// constrain the result, not peak memory while both snapshots coexist.
    pub fn with_deleted(
        &self,
        key: HeightfieldPlacementKey,
        deleted: bool,
        budget: HeightfieldPlacementDeletionBudget,
    ) -> Result<Self, HeightfieldPlacementDeletionError> {
        validate_key(key)?;
        let (index, len) = match (self.keys().binary_search(&key), deleted) {
            (Ok(_), true) | (Err(_), false) => return Ok(self.clone()),
            (Ok(index), false) => (index, self.0.len() - 1),
            (Err(index), true) => (
                index,
                self.0
                    .len()
                    .checked_add(1)
                    .ok_or(HeightfieldPlacementDeletionError::ArithmeticOverflow)?,
            ),
        };
        let mut keys = reserve_keys(len, budget)?;
        keys.extend_from_slice(&self.keys()[..index]);
        if deleted {
            keys.push(key);
            keys.extend_from_slice(&self.keys()[index..]);
        } else {
            keys.extend_from_slice(&self.keys()[index + 1..]);
        }
        Ok(Self(Arc::new(keys)))
    }
}

/// Stably removes exact full-placement-key matches in the original paired Vecs.
///
/// Validates alignment first, even for empty columns/deletions. The source is
/// consumed on success or failure. Both Vec pointers and capacities are retained,
/// even when all rows are removed. Geometry, samples, attributes, full source
/// cache and upstream work counters are unchanged for survivors. Deletion never
/// regenerates candidates or refills holes, so minimum distances remain valid.
///
/// O(1) for empty deletions, otherwise O(P log(D + 1)) time and O(1) extra space
/// for P rows and D deletion keys; no
/// output Vec allocation, terrain query or source-key revalidation is performed.
/// The returned deletion set is an O(1) shared clone of the complete input.
pub fn filter_heightfield_placement(
    mut source: HeightfieldPlacementOutput,
    deletions: &HeightfieldPlacementDeletionSet,
) -> Result<DeletionFilteredHeightfieldPlacement, HeightfieldPlacementDeletionError> {
    if source.points.len() != source.attributes.len() {
        return Err(HeightfieldPlacementDeletionError::MisalignedAttributes);
    }
    if deletions.keys().is_empty() {
        return Ok(DeletionFilteredHeightfieldPlacement {
            source,
            deletions: deletions.clone(),
            removed: 0,
        });
    }
    let original_len = source.points.len();
    let mut write = 0;
    for read in 0..original_len {
        if !deletions.is_deleted(source.points[read].key) {
            if read != write {
                source.points.swap(write, read);
                source.attributes.swap(write, read);
            }
            write += 1;
        }
    }
    source.points.truncate(write);
    source.attributes.truncate(write);
    Ok(DeletionFilteredHeightfieldPlacement {
        source,
        deletions: deletions.clone(),
        removed: original_len - write,
    })
}

fn validate_key(key: HeightfieldPlacementKey) -> Result<(), HeightfieldPlacementDeletionError> {
    if key.placement_version != HEIGHTFIELD_PLACEMENT_VERSION || key.source.grid.y != 0 {
        return Err(HeightfieldPlacementDeletionError::InvalidKey(
            HeightfieldPlacementError::InvalidKey,
        ));
    }
    key.source.rebuild().map_err(|error| {
        HeightfieldPlacementDeletionError::InvalidKey(HeightfieldPlacementError::Source(error))
    })?;
    Ok(())
}

fn key_bytes(capacity: usize) -> Result<usize, HeightfieldPlacementDeletionError> {
    capacity
        .checked_mul(size_of::<HeightfieldPlacementKey>())
        .ok_or(HeightfieldPlacementDeletionError::ArithmeticOverflow)
}

fn check_budget(
    len: usize,
    capacity: usize,
    budget: HeightfieldPlacementDeletionBudget,
) -> Result<(), HeightfieldPlacementDeletionError> {
    if len > budget.max_keys {
        return Err(HeightfieldPlacementDeletionError::KeyBudgetExceeded {
            required: len,
            limit: budget.max_keys,
        });
    }
    let required = key_bytes(capacity)?;
    if required > budget.max_key_bytes {
        return Err(HeightfieldPlacementDeletionError::KeyBytesBudgetExceeded {
            required,
            limit: budget.max_key_bytes,
        });
    }
    Ok(())
}

fn reserve_keys(
    len: usize,
    budget: HeightfieldPlacementDeletionBudget,
) -> Result<Vec<HeightfieldPlacementKey>, HeightfieldPlacementDeletionError> {
    check_budget(len, len, budget)?;
    let mut keys = Vec::new();
    keys.try_reserve_exact(len)
        .map_err(|_| HeightfieldPlacementDeletionError::AllocationFailed)?;
    // Actual capacity, not requested capacity, is charged before any copying.
    check_budget(len, keys.capacity(), budget)?;
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use alloc::{collections::BTreeMap, format, vec};
    use core::{convert::Infallible, error::Error};

    use super::*;
    use crate::{
        scatter_on_heightfield, BakeLimits, CubeFace, DirtyRegion, HeightSample, HeightSampleValue,
        HeightSampler, HeightfieldPlacementBudget, HeightfieldPlacementConfig,
        HeightfieldPlacementFilters, HeightfieldPlacementPoint, HeightfieldPlacementRegion,
        HeightfieldQueryError, HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding,
        PlanarHeightfieldSurface, TileKey,
    };
    use prism_world_residency::{
        DensityBrush, DensityBrushOp, DensityNoise, ScatterCandidateKey, ScatterDensityField,
        ScatterError, ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldSample, SectorCoord,
        SCATTER_KERNEL_VERSION,
    };

    const KEY_BYTES: usize = size_of::<HeightfieldPlacementKey>();
    const HEIGHTS: [i64; 9] = [0, 8, 16, -8, 0, 8, -16, -8, 0];

    fn budget() -> HeightfieldPlacementDeletionBudget {
        HeightfieldPlacementDeletionBudget {
            max_keys: usize::MAX,
            max_key_bytes: usize::MAX,
        }
    }

    fn zero_budget() -> HeightfieldPlacementDeletionBudget {
        HeightfieldPlacementDeletionBudget {
            max_keys: 0,
            max_key_bytes: 0,
        }
    }

    fn key(x: i64) -> HeightfieldPlacementKey {
        HeightfieldPlacementKey {
            placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
            source: ScatterCandidateKey {
                kernel_version: SCATTER_KERNEL_VERSION,
                seed: 42,
                version: 7,
                salt: 91,
                grid_spacing_mm: 100,
                grid: SectorCoord::new(x, 0, 0),
            },
        }
    }

    fn set(keys: &[HeightfieldPlacementKey]) -> HeightfieldPlacementDeletionSet {
        HeightfieldPlacementDeletionSet::new(keys.to_vec(), budget()).unwrap()
    }

    struct GridHeights<'a>(&'a [i64; 9]);

    impl HeightSampler for GridHeights<'_> {
        type Error = Infallible;

        fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            let index = |q: u16| ((u64::from(q) * 2 + 32_767) / 65_535) as usize;
            Ok(HeightSampleValue::from_millimetres(
                self.0[index(sample.v_q16()) * 3 + index(sample.u_q16())],
            ))
        }
    }

    fn bake_limits() -> BakeLimits {
        BakeLimits {
            max_samples: 9,
            max_bytes: 9 * size_of::<i16>(),
        }
    }

    fn surface(heights: &[i64; 9]) -> PlanarHeightfieldSurface {
        PlanarHeightfieldSurface::new(
            Arc::new(
                HeightfieldTile::bake(
                    TileKey::root(CubeFace::PositiveY),
                    HeightfieldSpec::new(3, 1).unwrap(),
                    &GridHeights(heights),
                    bake_limits(),
                )
                .unwrap(),
            ),
            PlanarHeightfieldEmbedding {
                sample_origin_mm: [-16, 40, -16],
                sample_spacing_mm: 16,
            },
        )
        .unwrap()
    }

    fn config() -> HeightfieldPlacementConfig {
        HeightfieldPlacementConfig {
            seed: 0x3a65_c807_9742_1eab,
            version: 7,
            salt: 0xf09c_1172_d538_208a,
            grid_spacing_mm: 3,
            min_distance_mm: 0,
            vertical_offset_mm: 7,
        }
    }

    fn region() -> HeightfieldPlacementRegion {
        HeightfieldPlacementRegion::new([-13, -11], [14, 13]).unwrap()
    }

    fn filters() -> HeightfieldPlacementFilters {
        HeightfieldPlacementFilters {
            surface_height_mm: [i64::MIN, i64::MAX],
            max_slope_rise: u32::MAX,
            max_slope_run: 1,
        }
    }

    fn field(painted: bool) -> ScatterDensityField {
        let brushes = if painted {
            vec![
                DensityBrush {
                    id: 9,
                    center_mm: [9, 47, 0],
                    inner_radius_mm: 10,
                    outer_radius_mm: 24,
                    opacity: u16::MAX,
                    density: DensityBrushOp::Add(-12_000),
                    species: Some(97),
                },
                DensityBrush {
                    id: 2,
                    center_mm: [-9, 47, 0],
                    inner_radius_mm: 10,
                    outer_radius_mm: 24,
                    opacity: u16::MAX,
                    density: DensityBrushOp::Add(4_000),
                    species: Some(23),
                },
            ]
        } else {
            Vec::new()
        };
        let limits = ScatterFieldBuildBudget {
            max_brushes: brushes.len(),
            max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
        };
        ScatterDensityField::new(
            ScatterFieldConfig {
                seed: 91,
                version: 3,
                base_density: if painted { 50_000 } else { u16::MAX },
                base_species: 11,
                noise: painted.then_some(DensityNoise {
                    cell_size_mm: 7,
                    amplitude: 12_000,
                    warp_cell_size_mm: 11,
                    warp_amplitude_mm: 3,
                }),
            },
            brushes,
            limits,
        )
        .unwrap()
    }

    fn generate(
        config: HeightfieldPlacementConfig,
        region: HeightfieldPlacementRegion,
        field: &ScatterDensityField,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldPlacementFilters,
    ) -> HeightfieldPlacementOutput {
        scatter_on_heightfield(
            config,
            region,
            field,
            surface,
            filters,
            HeightfieldPlacementBudget {
                max_candidate_evaluations: 1_000_000,
                max_sample_reads: 100_000,
                max_field_work: 1_000_000,
                max_output_points: 4_096,
                max_output_bytes: 4_096
                    * (size_of::<HeightfieldPlacementPoint>() + size_of::<ScatterFieldSample>()),
            },
        )
        .unwrap()
    }

    fn source(painted: bool) -> HeightfieldPlacementOutput {
        generate(
            config(),
            region(),
            &field(painted),
            &surface(&HEIGHTS),
            filters(),
        )
    }

    fn rows(
        source: &HeightfieldPlacementOutput,
    ) -> Vec<(HeightfieldPlacementPoint, ScatterFieldSample)> {
        source
            .points
            .iter()
            .copied()
            .zip(source.attributes.iter().copied())
            .collect()
    }

    fn layout(
        source: &HeightfieldPlacementOutput,
    ) -> (
        *const HeightfieldPlacementPoint,
        usize,
        *const ScatterFieldSample,
        usize,
    ) {
        (
            source.points.as_ptr(),
            source.points.capacity(),
            source.attributes.as_ptr(),
            source.attributes.capacity(),
        )
    }

    fn counters(source: &HeightfieldPlacementOutput) -> (u64, u64, u64) {
        (
            source.candidate_evaluations,
            source.surface_sample_reads,
            source.field_work,
        )
    }

    #[test]
    fn default_and_zero_budget_empty_construction() {
        assert_eq!(HEIGHTFIELD_PLACEMENT_DELETION_VERSION, 1);
        let empty = HeightfieldPlacementDeletionSet::new(Vec::new(), zero_budget()).unwrap();
        assert_eq!(empty, HeightfieldPlacementDeletionSet::default());
        assert!(empty.keys().is_empty());
        assert_eq!(empty.allocated_key_bytes(), 0);
        assert!(!empty.is_deleted(key(0)));
    }

    #[test]
    fn constructor_sorts_deduplicates_and_reuses_allocation_with_spare_capacity() {
        let mut keys = Vec::with_capacity(19);
        keys.extend([key(2), key(-1), key(2), key(0), key(-1)]);
        let pointer = keys.as_ptr();
        let bytes = keys.capacity() * KEY_BYTES;
        let actual = HeightfieldPlacementDeletionSet::new(
            keys,
            HeightfieldPlacementDeletionBudget {
                max_keys: 5,
                max_key_bytes: bytes,
            },
        )
        .unwrap();
        assert_eq!(actual.keys(), &[key(-1), key(0), key(2)]);
        assert_eq!(actual.keys().as_ptr(), pointer);
        assert_eq!(actual.allocated_key_bytes(), bytes);
        let independent = set(&[key(0), key(2), key(-1)]);
        assert_eq!(actual, independent);
        assert!(!Arc::ptr_eq(&actual.0, &independent.0));
    }

    #[test]
    fn constructor_budgets_duplicates_before_dedup_and_charges_empty_spare_capacity() {
        assert_eq!(
            HeightfieldPlacementDeletionSet::new(
                vec![key(0); 3],
                HeightfieldPlacementDeletionBudget {
                    max_keys: 1,
                    ..budget()
                }
            ),
            Err(HeightfieldPlacementDeletionError::KeyBudgetExceeded {
                required: 3,
                limit: 1
            })
        );
        for populated in [false, true] {
            let mut keys = Vec::with_capacity(17);
            if populated {
                keys.push(key(0));
            }
            let bytes = keys.capacity() * KEY_BYTES;
            assert_eq!(
                HeightfieldPlacementDeletionSet::new(
                    keys,
                    HeightfieldPlacementDeletionBudget {
                        max_key_bytes: bytes - 1,
                        ..budget()
                    }
                ),
                Err(HeightfieldPlacementDeletionError::KeyBytesBudgetExceeded {
                    required: bytes,
                    limit: bytes - 1
                })
            );
        }
    }

    #[test]
    fn membership_uses_placement_version_and_every_full_source_key_field() {
        let base = key(0);
        let mut variants = [base; 9];
        variants[0].placement_version += 1;
        variants[1].source.kernel_version += 1;
        variants[2].source.seed ^= 1_u64 << 63;
        variants[3].source.version += 1;
        variants[4].source.salt ^= 1_u64 << 63;
        variants[5].source.grid_spacing_mm += 1;
        variants[6].source.grid.x += 1;
        variants[7].source.grid.y += 1;
        variants[8].source.grid.z -= 1;
        let deleted = set(&[base]);
        assert!(deleted.is_deleted(base));
        for alternative in variants {
            assert!(!deleted.is_deleted(alternative));
        }
        let valid = [
            variants[2],
            variants[3],
            variants[4],
            variants[5],
            variants[6],
            variants[8],
        ];
        let all = set(&valid);
        assert_eq!(all.keys().len(), valid.len());
        for alternative in valid {
            assert!(all.is_deleted(alternative));
        }
    }

    #[test]
    fn constructor_and_edits_preserve_placement_and_source_validation_errors() {
        let mut invalid = [key(0); 7];
        invalid[0].placement_version = 0;
        invalid[1].placement_version += 1;
        invalid[2].source.grid.y = -1;
        invalid[3].source.grid.y = 1;
        invalid[4].source.kernel_version += 1;
        invalid[5].source.grid_spacing_mm = 0;
        invalid[6].source.grid.x = i64::MAX;
        let expected = [
            HeightfieldPlacementError::InvalidKey,
            HeightfieldPlacementError::InvalidKey,
            HeightfieldPlacementError::InvalidKey,
            HeightfieldPlacementError::InvalidKey,
            HeightfieldPlacementError::Source(ScatterError::UnsupportedKernelVersion),
            HeightfieldPlacementError::Source(ScatterError::InvalidGridSpacing),
            HeightfieldPlacementError::Source(ScatterError::CoordinateOverflow),
        ];
        let original = set(&[key(0)]);
        for (bad, reason) in invalid.into_iter().zip(expected) {
            let error = HeightfieldPlacementDeletionError::InvalidKey(reason);
            assert_eq!(
                HeightfieldPlacementDeletionSet::new(vec![bad, bad], budget()),
                Err(error)
            );
            for deleted in [false, true] {
                // Absent restore is a no-op, but must still validate before cloning.
                assert_eq!(
                    original.with_deleted(bad, deleted, zero_budget()),
                    Err(error)
                );
            }
        }
        let mut both_bad = invalid[5];
        both_bad.source.grid.y = 1;
        assert_eq!(
            validate_key(both_bad),
            Err(HeightfieldPlacementDeletionError::InvalidKey(
                HeightfieldPlacementError::InvalidKey
            ))
        );
        assert_eq!(original.keys(), &[key(0)]);
    }

    #[test]
    fn deletion_keys_require_no_terrain_coverage_and_keep_extreme_full_precision() {
        let mut extreme = key(0);
        extreme.source.grid_spacing_mm = 1;
        extreme.source.grid = SectorCoord::new(i64::MIN, 0, i64::MAX);
        let deleted = set(&[key(1_000_000), extreme]);
        assert!(deleted.is_deleted(extreme));
        assert_eq!(
            extreme.source.rebuild().unwrap().position_mm,
            [i64::MIN, 0, i64::MAX]
        );
        assert_eq!(
            key(1_000_000).rebuild(&surface(&HEIGHTS), 0),
            Err(HeightfieldPlacementError::Surface(
                HeightfieldQueryError::OutsideSurface
            ))
        );
        assert!(deleted.is_deleted(key(1_000_000)));
    }

    #[test]
    fn clones_and_noops_share_arc_and_ignore_tighter_budgets() {
        let original = set(&[key(0)]);
        for unchanged in [
            original.clone(),
            original.with_deleted(key(0), true, zero_budget()).unwrap(),
            original.with_deleted(key(1), false, zero_budget()).unwrap(),
        ] {
            assert!(Arc::ptr_eq(&original.0, &unchanged.0));
            assert_eq!(original, unchanged);
        }
    }

    #[test]
    fn edits_at_every_sorted_position_are_immutable() {
        let original = set(&[key(-2), key(0), key(2)]);
        let clone = original.clone();
        for inserted in [key(-3), key(-1), key(1), key(3)] {
            let changed = original.with_deleted(inserted, true, budget()).unwrap();
            assert!(!Arc::ptr_eq(&changed.0, &original.0));
            assert!(changed.keys().windows(2).all(|pair| pair[0] < pair[1]));
            assert!(changed.is_deleted(inserted));
            assert_eq!(
                changed.with_deleted(inserted, false, budget()).unwrap(),
                original
            );
        }
        for &removed in original.keys() {
            let changed = original.with_deleted(removed, false, budget()).unwrap();
            assert!(!changed.is_deleted(removed));
            assert_eq!(changed.keys().len(), 2);
            assert_eq!(
                changed.with_deleted(removed, true, budget()).unwrap(),
                original
            );
        }
        assert_eq!(original.keys(), &[key(-2), key(0), key(2)]);
        assert_eq!(clone, original);
        assert!(Arc::ptr_eq(&clone.0, &original.0));
    }

    #[test]
    fn actual_edits_enforce_result_count_and_byte_budgets_without_mutating_old_sets() {
        let original = set(&[key(0), key(2)]);
        let clone = original.clone();
        for (candidate, deleted, required, max_keys) in
            [(key(1), true, 3, 2), (key(0), false, 1, 0)]
        {
            assert_eq!(
                original.with_deleted(
                    candidate,
                    deleted,
                    HeightfieldPlacementDeletionBudget {
                        max_keys,
                        ..budget()
                    }
                ),
                Err(HeightfieldPlacementDeletionError::KeyBudgetExceeded {
                    required,
                    limit: max_keys
                })
            );
        }
        for (candidate, deleted, required) in
            [(key(1), true, 3 * KEY_BYTES), (key(0), false, KEY_BYTES)]
        {
            assert_eq!(
                original.with_deleted(
                    candidate,
                    deleted,
                    HeightfieldPlacementDeletionBudget {
                        max_key_bytes: required - 1,
                        ..budget()
                    }
                ),
                Err(HeightfieldPlacementDeletionError::KeyBytesBudgetExceeded {
                    required,
                    limit: required - 1
                })
            );
        }
        assert_eq!(original, clone);
        assert!(Arc::ptr_eq(&original.0, &clone.0));
        let mut keys = Vec::with_capacity(100);
        keys.extend([key(0), key(2)]);
        let roomy = HeightfieldPlacementDeletionSet::new(keys, budget()).unwrap();
        let changed = roomy
            .with_deleted(
                key(0),
                false,
                HeightfieldPlacementDeletionBudget {
                    max_keys: 1,
                    max_key_bytes: KEY_BYTES,
                },
            )
            .unwrap();
        assert_eq!(changed.keys(), &[key(2)]);
        assert_eq!(changed.allocated_key_bytes(), KEY_BYTES);
        assert!(roomy.allocated_key_bytes() >= 100 * KEY_BYTES);
        let empty = changed.with_deleted(key(2), false, zero_budget()).unwrap();
        assert_eq!(empty, HeightfieldPlacementDeletionSet::default());
        assert_eq!(empty.allocated_key_bytes(), 0);
    }

    #[test]
    fn arithmetic_capacity_budgets_and_impossible_reservation_are_checked() {
        assert_eq!(
            key_bytes(usize::MAX),
            Err(HeightfieldPlacementDeletionError::ArithmeticOverflow)
        );
        assert_eq!(
            reserve_keys(usize::MAX, budget()),
            Err(HeightfieldPlacementDeletionError::ArithmeticOverflow)
        );
        let impossible = isize::MAX as usize / KEY_BYTES + 1;
        assert_eq!(
            reserve_keys(impossible, budget()),
            Err(HeightfieldPlacementDeletionError::AllocationFailed)
        );
        assert_eq!(
            check_budget(
                1,
                2,
                HeightfieldPlacementDeletionBudget {
                    max_keys: 1,
                    max_key_bytes: KEY_BYTES
                }
            ),
            Err(HeightfieldPlacementDeletionError::KeyBytesBudgetExceeded {
                required: 2 * KEY_BYTES,
                limit: KEY_BYTES
            })
        );
    }

    #[test]
    fn errors_display_and_preserve_the_complete_source_chain() {
        let nested = HeightfieldPlacementDeletionError::InvalidKey(
            HeightfieldPlacementError::Source(ScatterError::InvalidGridSpacing),
        );
        assert_eq!(
            format!("{}", nested.source().unwrap()),
            format!(
                "{}",
                HeightfieldPlacementError::Source(ScatterError::InvalidGridSpacing)
            )
        );
        assert_eq!(
            format!("{}", nested.source().unwrap().source().unwrap()),
            format!("{}", ScatterError::InvalidGridSpacing)
        );
        assert!(nested
            .source()
            .unwrap()
            .source()
            .unwrap()
            .source()
            .is_none());
        for error in [
            nested,
            HeightfieldPlacementDeletionError::InvalidKey(HeightfieldPlacementError::InvalidKey),
            HeightfieldPlacementDeletionError::KeyBudgetExceeded {
                required: 2,
                limit: 1,
            },
            HeightfieldPlacementDeletionError::KeyBytesBudgetExceeded {
                required: 100,
                limit: 1,
            },
            HeightfieldPlacementDeletionError::ArithmeticOverflow,
            HeightfieldPlacementDeletionError::AllocationFailed,
            HeightfieldPlacementDeletionError::MisalignedAttributes,
        ] {
            assert!(!format!("{error}").is_empty());
            assert_eq!(
                error.source().is_some(),
                matches!(error, HeightfieldPlacementDeletionError::InvalidKey(_))
            );
        }
    }

    #[test]
    fn none_outside_all_and_partial_deletions_preserve_both_allocations_and_metadata() {
        for mode in 0..5 {
            let mut original = source(true);
            assert!(original.points.len() > 3);
            original.points.reserve(97);
            original.attributes.reserve(313);
            let before = layout(&original);
            let cache = original.cache_key.clone();
            let counts = counters(&original);
            assert!(counts.0 > 0 && counts.1 > 0 && counts.2 > 0);
            let last = original.points.len() - 1;
            let mut keys: Vec<_> = original
                .points
                .iter()
                .enumerate()
                .filter(|(i, _)| match mode {
                    2 => true,
                    3 => [0, last / 2, last].contains(i),
                    4 => i.is_multiple_of(2),
                    _ => false,
                })
                .map(|(_, point)| point.key)
                .collect();
            if mode == 1 {
                keys.push(key(1_000_000));
            }
            let deleted = set(&keys);
            let expected: Vec<_> = rows(&original)
                .into_iter()
                .filter(|(point, _)| !keys.contains(&point.key))
                .collect();
            let count = original.points.len();
            let result = filter_heightfield_placement(original, &deleted).unwrap();
            assert_eq!(rows(&result.source), expected);
            assert_eq!(result.removed, count - expected.len());
            assert_eq!(layout(&result.source), before);
            assert_eq!(result.source.cache_key, cache);
            assert_eq!(counters(&result.source), counts);
            assert!(Arc::ptr_eq(
                result.source.cache_key.surface.tile(),
                cache.surface.tile()
            ));
            assert!(Arc::ptr_eq(&result.deletions.0, &deleted.0));
        }
    }

    #[test]
    fn real_author_attributes_stay_paired_with_projected_geometry() {
        let original = source(true);
        assert!(original
            .attributes
            .windows(2)
            .any(|pair| pair[0].density != pair[1].density));
        assert!(original
            .attributes
            .windows(2)
            .any(|pair| pair[0].species != pair[1].species));
        let keys: Vec<_> = original
            .points
            .iter()
            .skip(1)
            .step_by(3)
            .map(|point| point.key)
            .collect();
        let expected: Vec<_> = rows(&original)
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % 3 != 1)
            .map(|(_, row)| row)
            .collect();
        let result = filter_heightfield_placement(original, &set(&keys)).unwrap();
        assert!(result.removed > 0);
        assert_eq!(rows(&result.source), expected);
        for (point, attribute) in rows(&result.source) {
            assert_eq!(
                result.source.cache_key.field.sample(point.position_mm),
                attribute
            );
            assert_eq!(
                point
                    .key
                    .rebuild(
                        &result.source.cache_key.surface,
                        config().vertical_offset_mm
                    )
                    .unwrap(),
                point
            );
            let raw = point.key.source.rebuild().unwrap();
            assert_ne!(raw.position_mm[1], point.position_mm[1]);
        }
    }

    #[test]
    fn empty_generated_output_keeps_spare_capacity_and_complete_deletions() {
        let empty_region = HeightfieldPlacementRegion::new([1000, 1000], [1000, 2000]).unwrap();
        let mut original = generate(
            config(),
            empty_region,
            &field(true),
            &surface(&HEIGHTS),
            filters(),
        );
        assert!(original.points.is_empty());
        original.points.reserve(17);
        original.attributes.reserve(31);
        let before = layout(&original);
        let cache = original.cache_key.clone();
        let counts = counters(&original);
        let deleted = set(&[key(1000)]);
        let result = filter_heightfield_placement(original, &deleted).unwrap();
        assert_eq!(result.removed, 0);
        assert_eq!(layout(&result.source), before);
        assert_eq!(result.source.cache_key, cache);
        assert_eq!(counters(&result.source), counts);
        assert_eq!(result.deletions, deleted);
    }

    #[test]
    fn misalignment_is_rejected_in_both_directions_before_empty_or_full_shortcuts() {
        for mode in 0..4 {
            for populated in [false, true] {
                let mut original = source(false);
                let deleted = if populated {
                    set(&original
                        .points
                        .iter()
                        .map(|point| point.key)
                        .collect::<Vec<_>>())
                } else {
                    HeightfieldPlacementDeletionSet::default()
                };
                match mode {
                    0 => {
                        original.points.pop();
                    }
                    1 => {
                        original.attributes.pop();
                    }
                    2 => original.points.clear(),
                    _ => original.attributes.clear(),
                }
                assert_eq!(
                    filter_heightfield_placement(original, &deleted),
                    Err(HeightfieldPlacementDeletionError::MisalignedAttributes)
                );
            }
        }
    }

    #[test]
    fn filtering_uses_full_placement_identity_without_revalidating_public_rows() {
        let base = source(false).points[0].key;
        let mut variants = [base; 9];
        variants[0].placement_version += 1;
        variants[1].source.kernel_version += 1;
        variants[2].source.seed ^= 1;
        variants[3].source.version += 1;
        variants[4].source.salt ^= 1;
        variants[5].source.grid_spacing_mm += 1;
        variants[6].source.grid.x += 1;
        variants[7].source.grid.y += 1;
        variants[8].source.grid.z -= 1;
        for alternative in variants {
            let mut original = source(false);
            original.points.truncate(1);
            original.attributes.truncate(1);
            original.points[0].key = alternative;
            let expected = rows(&original);
            let result = filter_heightfield_placement(original, &set(&[base])).unwrap();
            assert_eq!(result.removed, 0);
            assert_eq!(rows(&result.source), expected);
        }
    }

    #[test]
    fn removal_counts_rows_including_duplicate_keys_and_repeated_filtering_is_idempotent() {
        let mut original = source(false);
        let duplicate = original.points[0];
        let attribute = original.attributes[0];
        original.points.push(duplicate);
        original.attributes.push(attribute);
        let deleted = set(&[duplicate.key, duplicate.key, key(1000)]);
        let first = filter_heightfield_placement(original, &deleted).unwrap();
        assert_eq!(first.removed, 2);
        let before = layout(&first.source);
        let expected = rows(&first.source);
        let second = filter_heightfield_placement(first.source, &deleted).unwrap();
        assert_eq!(second.removed, 0);
        assert_eq!(rows(&second.source), expected);
        assert_eq!(layout(&second.source), before);
    }

    #[test]
    fn absent_keys_are_full_cache_dependencies_even_when_survivors_are_identical() {
        let first =
            filter_heightfield_placement(source(true), &HeightfieldPlacementDeletionSet::default())
                .unwrap();
        let second = filter_heightfield_placement(source(true), &set(&[key(1000)])).unwrap();
        assert_eq!(first.source, second.source);
        assert_eq!(first.removed, second.removed);
        assert_ne!(first.deletions, second.deletions);
        assert_ne!(first, second);
    }

    #[test]
    fn arbitrary_unaligned_xz_partitions_in_both_orders_match_the_whole_request() {
        let config = HeightfieldPlacementConfig {
            min_distance_mm: 3,
            ..config()
        };
        let terrain = surface(&HEIGHTS);
        let author = field(true);
        let limits = HeightfieldPlacementFilters {
            surface_height_mm: [32, 48],
            max_slope_rise: 1,
            max_slope_run: 1,
        };
        let original = generate(config, region(), &author, &terrain, limits);
        assert!(original.points.len() > 3);
        let mut keys: Vec<_> = original
            .points
            .iter()
            .step_by(3)
            .map(|point| point.key)
            .collect();
        keys.push(key(1000));
        let deleted = set(&keys);
        let whole = filter_heightfield_placement(original, &deleted).unwrap();
        let expected: BTreeMap<_, _> = rows(&whole.source)
            .into_iter()
            .map(|row| (row.0.key, row))
            .collect();
        let xs = [-13, -7, 2, 14];
        let zs = [-11, -4, 5, 13];
        for reverse in [false, true] {
            let mut union = BTreeMap::new();
            let mut removed = 0;
            for ordinal in 0..9 {
                let index = if reverse { 8 - ordinal } else { ordinal };
                let (x, z) = (index / 3, index % 3);
                let part_region =
                    HeightfieldPlacementRegion::new([xs[x], zs[z]], [xs[x + 1], zs[z + 1]])
                        .unwrap();
                let original = generate(config, part_region, &author, &terrain, limits);
                let before = layout(&original);
                let cache = original.cache_key.clone();
                let counts = counters(&original);
                let part = filter_heightfield_placement(original, &deleted).unwrap();
                assert_eq!(part.source.cache_key, cache);
                assert_eq!(counters(&part.source), counts);
                assert_eq!(layout(&part.source), before);
                assert_eq!(part.deletions, deleted);
                removed += part.removed;
                for row in rows(&part.source) {
                    assert!(union.insert(row.0.key, row).is_none());
                }
            }
            assert_eq!(removed, whole.removed);
            assert_eq!(union, expected);
        }
        for (i, point) in whole.source.points.iter().enumerate() {
            assert_eq!(
                point
                    .key
                    .rebuild(&terrain, config.vertical_offset_mm)
                    .unwrap(),
                *point
            );
            assert!(region().contains([point.position_mm[0], point.position_mm[2]]));
            for other in &whole.source.points[i + 1..] {
                let dx = i128::from(point.position_mm[0]) - i128::from(other.position_mm[0]);
                let dz = i128::from(point.position_mm[2]) - i128::from(other.position_mm[2]);
                assert!(dx * dx + dz * dz >= i128::from(config.min_distance_mm).pow(2));
            }
        }
    }

    #[test]
    fn namespace_changes_do_not_alias_but_offset_and_distance_do_not_change_identity() {
        let original = source(false);
        let deleted = set(&original
            .points
            .iter()
            .map(|point| point.key)
            .collect::<Vec<_>>());
        let mut configs = [config(); 4];
        configs[0].seed += 1;
        configs[1].version += 1;
        configs[2].salt += 1;
        configs[3].grid_spacing_mm += 1;
        for config in configs {
            let original = generate(
                config,
                region(),
                &field(false),
                &surface(&HEIGHTS),
                filters(),
            );
            assert!(!original.points.is_empty());
            let expected = rows(&original);
            let result = filter_heightfield_placement(original, &deleted).unwrap();
            assert_eq!(result.removed, 0);
            assert_eq!(rows(&result.source), expected);
        }
        for config in [
            HeightfieldPlacementConfig {
                vertical_offset_mm: -23,
                ..config()
            },
            HeightfieldPlacementConfig {
                min_distance_mm: 4,
                ..config()
            },
        ] {
            let original = generate(
                config,
                region(),
                &field(false),
                &surface(&HEIGHTS),
                filters(),
            );
            let count = original.points.len();
            assert!(count > 0);
            let result = filter_heightfield_placement(original, &deleted).unwrap();
            assert_eq!(result.removed, count);
            assert!(result.source.points.is_empty());
        }
    }

    #[test]
    fn complete_source_cache_fields_and_future_algorithm_revisions_are_preserved() {
        let baseline = source(false).cache_key;
        // Public descriptors are retained verbatim, not rebuilt or normalized.
        for variant in 0..18 {
            let mut original = source(false);
            match variant {
                0 => original.cache_key.placement_version += 1,
                1 => original.cache_key.source_kernel_version += 1,
                2 => original.cache_key.surface_version += 1,
                3 => original.cache_key.field_version += 1,
                4 => original.cache_key.config.seed += 1,
                5 => original.cache_key.config.version += 1,
                6 => original.cache_key.config.salt += 1,
                7 => original.cache_key.config.grid_spacing_mm += 1,
                8 => original.cache_key.config.min_distance_mm += 1,
                9 => original.cache_key.config.vertical_offset_mm += 1,
                10 => original.cache_key.region.min_xz_mm[0] -= 1,
                11 => original.cache_key.region.max_xz_mm[1] += 1,
                12 => original.cache_key.filters.surface_height_mm[0] += 1,
                13 => original.cache_key.filters.max_slope_rise -= 1,
                14 => original.cache_key.filters.max_slope_run += 1,
                15 => original.cache_key.surface = surface(&[0; 9]),
                16 => {
                    let old = &original.cache_key.surface;
                    original.cache_key.surface = PlanarHeightfieldSurface::new(
                        old.tile().clone(),
                        PlanarHeightfieldEmbedding {
                            sample_origin_mm: [-16, 41, -16],
                            ..old.embedding()
                        },
                    )
                    .unwrap();
                }
                _ => original.cache_key.field = field(true),
            }
            let expected = original.cache_key.clone();
            assert_ne!(expected, baseline);
            let counts = counters(&original);
            let deleted = set(&[original.points[0].key]);
            let result = filter_heightfield_placement(original, &deleted).unwrap();
            assert_eq!(result.source.cache_key, expected);
            assert_eq!(counters(&result.source), counts);
        }
    }

    #[test]
    fn unused_terrain_samples_still_invalidate_the_full_filtered_cache() {
        let a = surface(&[0; 9]);
        let equal = surface(&[0; 9]);
        assert!(!Arc::ptr_eq(a.tile(), equal.tile()));
        let mut changed = [0; 9];
        changed[8] = 9;
        let b = surface(&changed);
        let small = HeightfieldPlacementRegion::new([-13, -13], [-7, -7]).unwrap();
        let run = |terrain: &PlanarHeightfieldSurface| {
            filter_heightfield_placement(
                generate(config(), small, &field(false), terrain, filters()),
                &HeightfieldPlacementDeletionSet::default(),
            )
            .unwrap()
        };
        let first = run(&a);
        let same = run(&equal);
        let different = run(&b);
        assert_eq!(first, same);
        assert_eq!(rows(&first.source), rows(&different.source));
        assert_eq!(a.tile().revision(), b.tile().revision());
        assert_ne!(first.source.cache_key, different.source.cache_key);
    }

    #[test]
    fn retained_wrapper_is_unchanged_by_public_rebake_and_immutable_deletion_edit() {
        let terrain = surface(&HEIGHTS);
        let original = generate(config(), region(), &field(true), &terrain, filters());
        let deleted = set(&[original.points[0].key]);
        let expected = rows(&original)[1..].to_vec();
        let cache = original.cache_key.clone();
        let retained = filter_heightfield_placement(original, &deleted).unwrap();
        let mut owner = terrain.tile().as_ref().clone();
        let dirty = DirtyRegion::new(owner.tile(), 0, 0, u16::MAX, u16::MAX).unwrap();
        owner
            .rebake(dirty, &GridHeights(&[1; 9]), bake_limits())
            .unwrap();
        let published =
            PlanarHeightfieldSurface::new(Arc::new(owner), terrain.embedding()).unwrap();
        let edited = deleted.with_deleted(key(1000), true, budget()).unwrap();
        let next = filter_heightfield_placement(
            generate(config(), region(), &field(true), &published, filters()),
            &edited,
        )
        .unwrap();
        assert_ne!(next.source.cache_key, retained.source.cache_key);
        assert_ne!(next.deletions, retained.deletions);
        assert_eq!(
            next.source.cache_key.surface.tile().revision(),
            cache.surface.tile().revision() + 1
        );
        drop(terrain);
        drop(deleted);
        assert_eq!(retained.source.cache_key, cache);
        assert_eq!(rows(&retained.source), expected);
        assert_eq!(retained.deletions.keys().len(), 1);
        assert_eq!(retained.removed, 1);
    }
}
