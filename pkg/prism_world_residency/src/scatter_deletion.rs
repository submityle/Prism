//! Content-neutral scatter deletions identified by complete structural keys.
//!
//! Deletions are independent of request partitions, field recipes and thinning
//! radii. They never persist a per-point hash or reinterpret a source cache key.
//! A filtered cache depends on its source descriptor, [`SCATTER_DELETION_VERSION`]
//! and the complete [`ScatterDeletionSet`], including keys outside the request.
//! The wrappers retain that full set, not merely the keys removed in this call.
//!
//! Vec payload budgets include spare capacity, but exclude the Vec/Arc headers
//! and allocator metadata/rounding. Vec reservations are fallible; allocating
//! the Arc uses normal allocator OOM handling and cannot return
//! [`ScatterDeletionError::AllocationFailed`]. This is not an RSS budget or a
//! serialization format. No runtime, persistence, GPU or content policy is owned
//! here.

use alloc::{sync::Arc, vec::Vec};
use core::{fmt, mem::size_of};

use crate::{FieldScatterOutput, ScatterCandidateKey, ScatterError, ScatterOutput};

/// Revision of structural deletion-set and filtering semantics.
pub const SCATTER_DELETION_VERSION: u32 = 1;

/// Limits on a supplied or newly allocated key Vec, excluding Arc metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterDeletionBudget {
    /// Maximum input count before deduplication, or count after an actual edit.
    pub max_keys: usize,
    /// Maximum Vec capacity times `size_of::<ScatterCandidateKey>()`.
    pub max_key_bytes: usize,
}

/// Validated, sorted, unique full candidate keys, immutably shared by value.
///
/// Cloning is O(1); equality compares complete keys, never hashes or allocation
/// identity. Construction reuses the supplied Vec, including spare capacity.
/// Default is empty. Arc creation follows normal allocator OOM handling; only
/// key Vec reservations have a recoverable allocation error.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ScatterDeletionSet(Arc<Vec<ScatterCandidateKey>>);

/// Filtered source plus the complete additional structural cache dependency.
#[derive(Debug, Eq, PartialEq)]
pub struct DeletionFilteredScatter {
    /// Original output with deleted points removed; source cache key unchanged.
    pub source: ScatterOutput,
    /// Complete immutable deletion input, including keys absent from `source`.
    pub deletions: ScatterDeletionSet,
    /// Number of source points removed, not the deletion-set cardinality.
    pub removed: usize,
}

/// Filtered field output with aligned attributes and a full deletion dependency.
#[derive(Debug, Eq, PartialEq)]
pub struct DeletionFilteredFieldScatter {
    /// Original output with aligned rows removed; source cache key unchanged.
    pub source: FieldScatterOutput,
    /// Complete immutable deletion input, including keys absent from `source`.
    pub deletions: ScatterDeletionSet,
    /// Number of source rows removed, not the deletion-set cardinality.
    pub removed: usize,
}

/// Invalid structural input, hard-budget violation or key Vec allocation error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScatterDeletionError {
    /// A full key cannot be reconstructed by the current scatter kernel.
    InvalidKey(ScatterError),
    /// Supplied input count or edited result count exceeds the key limit.
    KeyBudgetExceeded {
        /// Required key count (before deduplication for construction).
        required: usize,
        /// Maximum permitted key count.
        limit: usize,
    },
    /// Key Vec capacity payload, including spare capacity, exceeds the limit.
    KeyBytesBudgetExceeded {
        /// Required key Vec payload bytes.
        required: usize,
        /// Maximum permitted payload bytes.
        limit: usize,
    },
    /// Key count or capacity byte multiplication overflows `usize`.
    ArithmeticOverflow,
    /// Fallible key Vec reservation failed, including a capacity layout error.
    AllocationFailed,
    /// Field points and attributes have different lengths.
    MisalignedAttributes,
}

impl fmt::Display for ScatterDeletionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKey(error) => write!(f, "invalid scatter deletion key: {error}"),
            Self::KeyBudgetExceeded { required, limit } => {
                write!(f, "scatter deletions need {required} keys, limit {limit}")
            }
            Self::KeyBytesBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "scatter deletion Vec needs {required} bytes, limit {limit}"
                )
            }
            Self::ArithmeticOverflow => f.write_str("scatter deletion count or byte size overflow"),
            Self::AllocationFailed => f.write_str("scatter deletion key reservation failed"),
            Self::MisalignedAttributes => {
                f.write_str("scatter field points and attributes have different lengths")
            }
        }
    }
}

impl core::error::Error for ScatterDeletionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::InvalidKey(error) => Some(error),
            _ => None,
        }
    }
}

impl ScatterDeletionSet {
    /// Checks original length/capacity budgets and rebuilds every key, then
    /// sorts and deduplicates in place without another key buffer.
    ///
    /// Duplicate keys still count toward the input budget; deduplication never
    /// hides an oversized input allocation. Spare capacity is not shrunk.
    pub fn new(
        mut keys: Vec<ScatterCandidateKey>,
        budget: ScatterDeletionBudget,
    ) -> Result<Self, ScatterDeletionError> {
        check_budget(keys.len(), keys.capacity(), budget)?;
        for key in &keys {
            key.rebuild().map_err(ScatterDeletionError::InvalidKey)?;
        }
        keys.sort_unstable();
        keys.dedup();
        Ok(Self(Arc::new(keys)))
    }

    /// Canonical full keys in ascending structural order, without duplicates.
    pub fn keys(&self) -> &[ScatterCandidateKey] {
        self.0.as_slice()
    }

    /// Tests exact full-key membership using binary search; no hash identity.
    pub fn is_deleted(&self, key: ScatterCandidateKey) -> bool {
        self.keys().binary_search(&key).is_ok()
    }

    /// Owned key Vec capacity payload, excluding Vec/Arc and allocator metadata.
    pub fn allocated_key_bytes(&self) -> usize {
        // All constructors check this multiplication before publishing a set.
        self.0.capacity() * size_of::<ScatterCandidateKey>()
    }

    /// Returns an immutable edited set, leaving this set and all clones intact.
    ///
    /// The key must rebuild even for a no-op. An unchanged membership returns
    /// an O(1) clone without allocation or reapplying budgets to existing data.
    /// An actual edit checks the new count and capacity budget, fallibly reserves
    /// a fresh Vec and copies keys in canonical order. The budget limits that
    /// result, not combined peak memory while the old and new sets coexist.
    pub fn with_deleted(
        &self,
        key: ScatterCandidateKey,
        deleted: bool,
        budget: ScatterDeletionBudget,
    ) -> Result<Self, ScatterDeletionError> {
        key.rebuild().map_err(ScatterDeletionError::InvalidKey)?;
        let (index, len) = match (self.keys().binary_search(&key), deleted) {
            (Ok(_), true) | (Err(_), false) => return Ok(self.clone()),
            (Ok(index), false) => (index, self.0.len() - 1),
            (Err(index), true) => (
                index,
                self.0
                    .len()
                    .checked_add(1)
                    .ok_or(ScatterDeletionError::ArithmeticOverflow)?,
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

    /// Stably removes deleted points in place, without a new point Vec or
    /// changing capacity, positions, source cache key or evaluation counts.
    ///
    /// Deletion is a subset operation, so existing minimum-distance guarantees
    /// are preserved; rejected source candidates are never regenerated/refilled.
    pub fn filter_source(&self, mut source: ScatterOutput) -> DeletionFilteredScatter {
        let original_len = source.points.len();
        source.points.retain(|point| !self.is_deleted(point.key));
        DeletionFilteredScatter {
            removed: original_len - source.points.len(),
            source,
            deletions: self.clone(),
        }
    }

    /// Stably compacts points and attributes together in their original Vecs.
    ///
    /// Rejects unequal lengths before filtering, even for an empty deletion set.
    /// Neither capacity, source cache descriptor nor work counters are changed.
    pub fn filter_field(
        &self,
        mut source: FieldScatterOutput,
    ) -> Result<DeletionFilteredFieldScatter, ScatterDeletionError> {
        if source.points.len() != source.attributes.len() {
            return Err(ScatterDeletionError::MisalignedAttributes);
        }
        let original_len = source.points.len();
        let mut write = 0;
        for read in 0..original_len {
            if !self.is_deleted(source.points[read].key) {
                if read != write {
                    source.points.swap(write, read);
                    source.attributes.swap(write, read);
                }
                write += 1;
            }
        }
        source.points.truncate(write);
        source.attributes.truncate(write);
        Ok(DeletionFilteredFieldScatter {
            source,
            deletions: self.clone(),
            removed: original_len - write,
        })
    }
}

fn key_bytes(capacity: usize) -> Result<usize, ScatterDeletionError> {
    capacity
        .checked_mul(size_of::<ScatterCandidateKey>())
        .ok_or(ScatterDeletionError::ArithmeticOverflow)
}

fn check_budget(
    len: usize,
    capacity: usize,
    budget: ScatterDeletionBudget,
) -> Result<(), ScatterDeletionError> {
    if len > budget.max_keys {
        return Err(ScatterDeletionError::KeyBudgetExceeded {
            required: len,
            limit: budget.max_keys,
        });
    }
    let required = key_bytes(capacity)?;
    if required > budget.max_key_bytes {
        return Err(ScatterDeletionError::KeyBytesBudgetExceeded {
            required,
            limit: budget.max_key_bytes,
        });
    }
    Ok(())
}

fn reserve_keys(
    len: usize,
    budget: ScatterDeletionBudget,
) -> Result<Vec<ScatterCandidateKey>, ScatterDeletionError> {
    check_budget(len, len, budget)?;
    let mut keys = Vec::new();
    keys.try_reserve_exact(len)
        .map_err(|_| ScatterDeletionError::AllocationFailed)?;
    // Allocators may provide more capacity than requested. Account for reality
    // before copying any keys; all subsequent pushes fit the reserved capacity.
    check_budget(len, keys.capacity(), budget)?;
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec};
    use core::error::Error;

    use super::*;
    use crate::{
        scatter, scatter_with_density_field, FieldScatterBudget, ScatterBudget, ScatterConfig,
        ScatterDensityField, ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldSample,
        ScatterPoint, ScatterRegion, SectorCoord, SCATTER_KERNEL_VERSION,
    };

    fn budget() -> ScatterDeletionBudget {
        ScatterDeletionBudget {
            max_keys: usize::MAX,
            max_key_bytes: usize::MAX,
        }
    }

    fn zero_budget() -> ScatterDeletionBudget {
        ScatterDeletionBudget {
            max_keys: 0,
            max_key_bytes: 0,
        }
    }

    fn key(x: i64) -> ScatterCandidateKey {
        ScatterCandidateKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            seed: 42,
            version: 7,
            salt: 91,
            grid_spacing_mm: 100,
            grid: SectorCoord::new(x, 0, 0),
        }
    }

    fn set(keys: &[ScatterCandidateKey]) -> ScatterDeletionSet {
        ScatterDeletionSet::new(keys.to_vec(), budget()).unwrap()
    }

    fn config() -> ScatterConfig {
        ScatterConfig {
            seed: key(0).seed,
            version: key(0).version,
            salt: key(0).salt,
            grid_spacing_mm: key(0).grid_spacing_mm,
            min_distance_mm: 0,
        }
    }

    fn scatter_budget() -> ScatterBudget {
        ScatterBudget {
            max_candidate_evaluations: 1_000_000,
            max_output_points: 10_000,
            max_output_bytes: 1_000_000,
        }
    }

    fn region(min: [i64; 3], max: [i64; 3]) -> ScatterRegion {
        ScatterRegion::new(min, max).unwrap()
    }

    fn source() -> ScatterOutput {
        scatter(
            config(),
            region([-300, 0, 0], [400, 100, 100]),
            scatter_budget(),
        )
        .unwrap()
    }

    fn field_source() -> FieldScatterOutput {
        let field = ScatterDensityField::new(
            ScatterFieldConfig {
                seed: 9,
                version: 1,
                base_density: u16::MAX,
                base_species: 73,
                noise: None,
            },
            Vec::new(),
            ScatterFieldBuildBudget {
                max_brushes: 0,
                max_brush_bytes: 0,
            },
        )
        .unwrap();
        scatter_with_density_field(
            config(),
            region([-300, 0, 0], [400, 100, 100]),
            scatter_budget(),
            &field,
            FieldScatterBudget {
                max_field_work: 10_000,
                max_output_bytes: 1_000_000,
            },
        )
        .unwrap()
    }

    #[test]
    fn default_and_zero_budget_construction_are_empty() {
        let empty = ScatterDeletionSet::new(Vec::new(), zero_budget()).unwrap();
        assert_eq!(empty, ScatterDeletionSet::default());
        assert!(empty.keys().is_empty());
        assert_eq!(empty.allocated_key_bytes(), 0);
        assert!(!empty.is_deleted(key(0)));
        assert_eq!(SCATTER_DELETION_VERSION, 1);
    }

    #[test]
    fn construction_sorts_deduplicates_and_keeps_original_allocation() {
        let mut keys = Vec::with_capacity(19);
        keys.extend([key(4), key(-1), key(0), key(4), key(-1)]);
        let pointer = keys.as_ptr();
        let capacity = keys.capacity();
        let actual = ScatterDeletionSet::new(keys, budget()).unwrap();
        assert_eq!(actual.keys(), &[key(-1), key(0), key(4)]);
        assert_eq!(actual.keys().as_ptr(), pointer);
        assert_eq!(
            actual.allocated_key_bytes(),
            capacity * size_of::<ScatterCandidateKey>()
        );
        assert_eq!(actual, set(&[key(0), key(4), key(-1)]));
    }

    #[test]
    fn membership_compares_every_full_key_field_independently() {
        let base = key(0);
        let variants = [
            ScatterCandidateKey {
                kernel_version: base.kernel_version + 1,
                ..base
            },
            ScatterCandidateKey {
                seed: base.seed + 1,
                ..base
            },
            ScatterCandidateKey {
                version: base.version + 1,
                ..base
            },
            ScatterCandidateKey {
                salt: base.salt + 1,
                ..base
            },
            ScatterCandidateKey {
                grid_spacing_mm: base.grid_spacing_mm + 1,
                ..base
            },
            ScatterCandidateKey {
                grid: SectorCoord::new(1, 0, 0),
                ..base
            },
            ScatterCandidateKey {
                grid: SectorCoord::new(0, 1, 0),
                ..base
            },
            ScatterCandidateKey {
                grid: SectorCoord::new(0, 0, 1),
                ..base
            },
        ];
        let deleted = set(&[base]);
        assert!(deleted.is_deleted(base));
        for variant in variants {
            assert!(
                !deleted.is_deleted(variant),
                "field distinction: {variant:?}"
            );
        }
        let valid = &variants[1..];
        let all = set(valid);
        assert_eq!(all.keys().len(), valid.len());
        assert!(!all.is_deleted(base));
        for variant in valid {
            assert!(all.is_deleted(*variant));
        }
    }

    #[test]
    fn constructor_counts_duplicates_before_deduplication() {
        assert_eq!(
            ScatterDeletionSet::new(
                vec![key(0); 3],
                ScatterDeletionBudget {
                    max_keys: 2,
                    ..budget()
                }
            ),
            Err(ScatterDeletionError::KeyBudgetExceeded {
                required: 3,
                limit: 2
            }),
        );
    }

    #[test]
    fn constructor_charges_spare_capacity_not_length() {
        let mut keys = Vec::with_capacity(32);
        keys.push(key(0));
        let required = keys.capacity() * size_of::<ScatterCandidateKey>();
        assert_eq!(
            ScatterDeletionSet::new(
                keys,
                ScatterDeletionBudget {
                    max_key_bytes: required - 1,
                    ..budget()
                }
            ),
            Err(ScatterDeletionError::KeyBytesBudgetExceeded {
                required,
                limit: required - 1
            }),
        );
    }

    #[test]
    fn empty_input_still_charges_retained_spare_capacity() {
        let keys = Vec::with_capacity(1);
        let bytes = keys.capacity() * size_of::<ScatterCandidateKey>();
        assert_eq!(
            ScatterDeletionSet::new(keys, zero_budget()),
            Err(ScatterDeletionError::KeyBytesBudgetExceeded {
                required: bytes,
                limit: 0
            }),
        );
    }

    #[test]
    fn exact_input_budgets_accept_original_capacity() {
        let keys = vec![key(1), key(1), key(0)];
        let bytes = keys.capacity() * size_of::<ScatterCandidateKey>();
        let actual = ScatterDeletionSet::new(
            keys,
            ScatterDeletionBudget {
                max_keys: 3,
                max_key_bytes: bytes,
            },
        )
        .unwrap();
        assert_eq!(actual.keys(), &[key(0), key(1)]);
        assert_eq!(actual.allocated_key_bytes(), bytes);
    }

    #[test]
    fn invalid_keys_are_rejected_by_construction_and_edits() {
        let invalid = [
            (
                ScatterCandidateKey {
                    kernel_version: 0,
                    ..key(0)
                },
                ScatterError::UnsupportedKernelVersion,
            ),
            (
                ScatterCandidateKey {
                    grid_spacing_mm: 0,
                    ..key(0)
                },
                ScatterError::InvalidGridSpacing,
            ),
            (key(i64::MAX), ScatterError::CoordinateOverflow),
        ];
        let original = set(&[key(0)]);
        for (bad, error) in invalid {
            // A duplicate invalid key must not be hidden by deduplication.
            assert_eq!(
                ScatterDeletionSet::new(vec![bad, bad], budget()),
                Err(ScatterDeletionError::InvalidKey(error))
            );
            for deleted in [false, true] {
                assert_eq!(
                    original.with_deleted(bad, deleted, budget()),
                    Err(ScatterDeletionError::InvalidKey(error))
                );
            }
        }
        assert_eq!(original.keys(), &[key(0)]);
    }

    #[test]
    fn clones_and_no_op_edits_share_the_same_arc_without_budgeting_again() {
        let original = set(&[key(0)]);
        let clone = original.clone();
        let same_delete = original.with_deleted(key(0), true, zero_budget()).unwrap();
        let same_restore = original.with_deleted(key(1), false, zero_budget()).unwrap();
        for unchanged in [clone, same_delete, same_restore] {
            assert!(Arc::ptr_eq(&original.0, &unchanged.0));
            assert_eq!(original, unchanged);
        }
    }

    #[test]
    fn insertion_and_removal_at_every_sorted_position_leave_old_sets_intact() {
        let original = set(&[key(-2), key(0), key(2)]);
        for inserted in [key(-3), key(-1), key(1), key(3)] {
            let changed = original.with_deleted(inserted, true, budget()).unwrap();
            assert!(!Arc::ptr_eq(&original.0, &changed.0));
            assert!(changed.is_deleted(inserted));
            assert!(changed.keys().windows(2).all(|pair| pair[0] < pair[1]));
            let restored = changed.with_deleted(inserted, false, budget()).unwrap();
            assert_eq!(restored, original);
            assert!(changed.is_deleted(inserted));
        }
        for removed in original.keys() {
            let changed = original.with_deleted(*removed, false, budget()).unwrap();
            assert!(!changed.is_deleted(*removed));
            assert_eq!(changed.keys().len(), 2);
            assert!(original.is_deleted(*removed));
        }
        assert_eq!(original.keys(), &[key(-2), key(0), key(2)]);
    }

    #[test]
    fn edits_check_count_and_byte_limits_and_preserve_old_shared_set_on_failure() {
        let original = set(&[key(0), key(2)]);
        let clone = original.clone();
        let bytes = size_of::<ScatterCandidateKey>();
        assert_eq!(
            original.with_deleted(
                key(1),
                true,
                ScatterDeletionBudget {
                    max_keys: 2,
                    ..budget()
                }
            ),
            Err(ScatterDeletionError::KeyBudgetExceeded {
                required: 3,
                limit: 2
            }),
        );
        assert_eq!(
            original.with_deleted(
                key(1),
                true,
                ScatterDeletionBudget {
                    max_key_bytes: 3 * bytes - 1,
                    ..budget()
                }
            ),
            Err(ScatterDeletionError::KeyBytesBudgetExceeded {
                required: 3 * bytes,
                limit: 3 * bytes - 1
            }),
        );
        assert_eq!(
            original.with_deleted(
                key(0),
                false,
                ScatterDeletionBudget {
                    max_key_bytes: bytes - 1,
                    ..budget()
                }
            ),
            Err(ScatterDeletionError::KeyBytesBudgetExceeded {
                required: bytes,
                limit: bytes - 1
            }),
        );
        assert_eq!(original, clone);
        assert!(Arc::ptr_eq(&original.0, &clone.0));
        assert_eq!(original.keys(), &[key(0), key(2)]);
    }

    #[test]
    fn editing_budgets_the_new_allocation_not_old_spare_capacity() {
        let mut keys = Vec::with_capacity(100);
        keys.extend([key(0), key(2)]);
        let original = ScatterDeletionSet::new(keys, budget()).unwrap();
        let changed = original
            .with_deleted(
                key(0),
                false,
                ScatterDeletionBudget {
                    max_keys: 1,
                    max_key_bytes: size_of::<ScatterCandidateKey>(),
                },
            )
            .unwrap();
        assert_eq!(changed.keys(), &[key(2)]);
        assert_eq!(
            changed.allocated_key_bytes(),
            size_of::<ScatterCandidateKey>()
        );
        assert!(original.allocated_key_bytes() >= 100 * size_of::<ScatterCandidateKey>());
        let empty = changed.with_deleted(key(2), false, zero_budget()).unwrap();
        assert_eq!(empty, ScatterDeletionSet::default());
        assert_eq!(empty.allocated_key_bytes(), 0);
    }

    #[test]
    fn checked_capacity_arithmetic_and_fallible_layout_errors_are_reported() {
        assert_eq!(
            key_bytes(usize::MAX),
            Err(ScatterDeletionError::ArithmeticOverflow)
        );
        // Fits usize bytes, but exceeds Vec's isize::MAX allocation limit.
        // This fails deterministically before contacting the allocator.
        let impossible = isize::MAX as usize / size_of::<ScatterCandidateKey>() + 1;
        assert_eq!(
            reserve_keys(impossible, budget()),
            Err(ScatterDeletionError::AllocationFailed)
        );
    }

    #[test]
    fn errors_have_display_and_only_invalid_key_has_an_error_source() {
        let errors = [
            ScatterDeletionError::InvalidKey(ScatterError::InvalidGridSpacing),
            ScatterDeletionError::KeyBudgetExceeded {
                required: 2,
                limit: 1,
            },
            ScatterDeletionError::KeyBytesBudgetExceeded {
                required: 80,
                limit: 1,
            },
            ScatterDeletionError::ArithmeticOverflow,
            ScatterDeletionError::AllocationFailed,
            ScatterDeletionError::MisalignedAttributes,
        ];
        for (index, error) in errors.into_iter().enumerate() {
            assert!(!format!("{error}").is_empty());
            assert_eq!(error.source().is_some(), index == 0);
        }
        assert_eq!(
            format!("{}", errors[0].source().unwrap()),
            format!("{}", ScatterError::InvalidGridSpacing),
        );
    }

    #[test]
    fn source_filter_is_stable_preserves_positions_capacity_pointer_and_descriptor() {
        let mut input = source();
        input.points.reserve(20);
        // Filtering must preserve caller order, even when not canonical grid order.
        input.points.reverse();
        let expected: Vec<_> = input
            .points
            .iter()
            .copied()
            .filter(|point| ![key(-3), key(0), key(3)].contains(&point.key))
            .collect();
        let pointer = input.points.as_ptr();
        let capacity = input.points.capacity();
        let cache_key = input.cache_key;
        let work = input.candidate_evaluations;
        let deletions = set(&[key(-3), key(0), key(3), key(100)]);
        let actual = deletions.filter_source(input);
        assert_eq!(actual.source.points, expected);
        assert_eq!(actual.removed, 3);
        assert_eq!(actual.source.points.as_ptr(), pointer);
        assert_eq!(actual.source.points.capacity(), capacity);
        assert_eq!(actual.source.cache_key, cache_key);
        assert_eq!(actual.source.candidate_evaluations, work);
        assert!(Arc::ptr_eq(&actual.deletions.0, &deletions.0));
    }

    #[test]
    fn source_empty_none_all_and_duplicate_rows_count_removed_points() {
        let unchanged = ScatterDeletionSet::default().filter_source(source());
        assert_eq!(unchanged.removed, 0);
        assert_eq!(unchanged.source, source());
        let mut input = source();
        input.points.push(input.points[0]);
        let len = input.points.len();
        let capacity = input.points.capacity();
        let keys: Vec<_> = input.points.iter().map(|point| point.key).collect();
        let actual = set(&keys).filter_source(input);
        assert!(actual.source.points.is_empty());
        assert_eq!(actual.removed, len);
        assert_eq!(actual.source.points.capacity(), capacity);
        let empty = actual.deletions.filter_source(actual.source);
        assert_eq!(empty.removed, 0);
        assert!(empty.source.points.is_empty());
    }

    #[test]
    fn field_filter_stably_preserves_alignment_and_both_vec_allocations() {
        let mut input = field_source();
        input.points.reverse();
        for (index, attribute) in input.attributes.iter_mut().enumerate() {
            *attribute = ScatterFieldSample {
                density: index as u16,
                species: 1000 + index as u32,
            };
        }
        input.points.reserve(21);
        input.attributes.reserve(37);
        let deleted = [key(-3), key(0), key(3)];
        let expected: Vec<_> = input
            .points
            .iter()
            .copied()
            .zip(input.attributes.iter().copied())
            .filter(|(point, _)| !deleted.contains(&point.key))
            .collect();
        let pointers = (input.points.as_ptr(), input.attributes.as_ptr());
        let capacities = (input.points.capacity(), input.attributes.capacity());
        let cache_key = input.cache_key.clone();
        let counters = (input.candidate_evaluations, input.field_work);
        let deletions = set(&deleted);
        let actual = deletions.filter_field(input).unwrap();
        assert_eq!(actual.removed, 3);
        assert_eq!(actual.source.points.len(), actual.source.attributes.len());
        let rows: Vec<_> = actual
            .source
            .points
            .iter()
            .copied()
            .zip(actual.source.attributes.iter().copied())
            .collect();
        assert_eq!(rows, expected);
        assert_eq!(
            (
                actual.source.points.as_ptr(),
                actual.source.attributes.as_ptr()
            ),
            pointers
        );
        assert_eq!(
            (
                actual.source.points.capacity(),
                actual.source.attributes.capacity()
            ),
            capacities
        );
        assert_eq!(actual.source.cache_key, cache_key);
        assert_eq!(
            (
                actual.source.candidate_evaluations,
                actual.source.field_work
            ),
            counters
        );
        assert!(Arc::ptr_eq(&actual.deletions.0, &deletions.0));
    }

    #[test]
    fn field_empty_none_and_all_deleted_preserve_capacity_and_counters() {
        let input = field_source();
        let unchanged = ScatterDeletionSet::default().filter_field(input).unwrap();
        assert_eq!(unchanged.removed, 0);
        assert_eq!(unchanged.source, field_source());
        let input = unchanged.source;
        let keys: Vec<_> = input.points.iter().map(|point| point.key).collect();
        let len = input.points.len();
        let capacities = (input.points.capacity(), input.attributes.capacity());
        let counters = (input.candidate_evaluations, input.field_work);
        let actual = set(&keys).filter_field(input).unwrap();
        assert_eq!(actual.removed, len);
        assert!(actual.source.points.is_empty());
        assert!(actual.source.attributes.is_empty());
        assert_eq!(
            (
                actual.source.points.capacity(),
                actual.source.attributes.capacity()
            ),
            capacities
        );
        assert_eq!(
            (
                actual.source.candidate_evaluations,
                actual.source.field_work
            ),
            counters
        );
        let empty = actual.deletions.filter_field(actual.source).unwrap();
        assert_eq!(empty.removed, 0);
    }

    #[test]
    fn field_rejects_both_directions_of_misalignment_even_without_deletions() {
        let deletions = ScatterDeletionSet::default();
        let mut short_attributes = field_source();
        short_attributes.attributes.pop();
        assert_eq!(
            deletions.filter_field(short_attributes),
            Err(ScatterDeletionError::MisalignedAttributes)
        );
        let mut long_attributes = field_source();
        long_attributes.attributes.push(ScatterFieldSample {
            density: 0,
            species: 0,
        });
        assert_eq!(
            deletions.filter_field(long_attributes),
            Err(ScatterDeletionError::MisalignedAttributes)
        );
        let mut empty_points = field_source();
        empty_points.points.clear();
        assert_eq!(
            deletions.filter_field(empty_points),
            Err(ScatterDeletionError::MisalignedAttributes)
        );
    }

    #[test]
    fn deletion_membership_is_independent_of_negative_and_positive_partitions() {
        let bounds = region([-350, -220, -190], [370, 230, 210]);
        let whole = scatter(config(), bounds, scatter_budget()).unwrap();
        let deleted_keys: Vec<_> = whole
            .points
            .iter()
            .step_by(3)
            .map(|point| point.key)
            .collect();
        assert!(!deleted_keys.is_empty());
        let deletions = set(&deleted_keys);
        let expected = deletions.filter_source(whole).source.points;
        let mut partitioned = Vec::new();
        for (left, right) in [(-350, -17), (-17, 0), (0, 143), (143, 370)] {
            let input = scatter(
                config(),
                region([left, -220, -190], [right, 230, 210]),
                scatter_budget(),
            )
            .unwrap();
            let filtered = deletions.filter_source(input);
            assert_eq!(filtered.deletions, deletions);
            partitioned.extend(filtered.source.points);
        }
        partitioned.sort_unstable_by_key(|point| point.key);
        assert_eq!(partitioned, expected);
    }

    #[test]
    fn seed_and_recipe_version_boundaries_do_not_alias_deletions() {
        let deletions = set(&[key(0)]);
        assert_eq!(deletions.filter_source(source()).removed, 1);
        for changed_config in [
            ScatterConfig {
                seed: config().seed + 1,
                ..config()
            },
            ScatterConfig {
                version: config().version + 1,
                ..config()
            },
            ScatterConfig {
                salt: config().salt + 1,
                ..config()
            },
        ] {
            let input = scatter(
                changed_config,
                region([-300, 0, 0], [400, 100, 100]),
                scatter_budget(),
            )
            .unwrap();
            let original_len = input.points.len();
            let actual = deletions.filter_source(input);
            assert_eq!(actual.removed, 0);
            assert_eq!(actual.source.points.len(), original_len);
        }
    }

    #[test]
    fn deletion_subset_preserves_source_thinning_and_never_refills_holes() {
        let config = ScatterConfig {
            min_distance_mm: 135,
            ..config()
        };
        let bounds = region([-400; 3], [400; 3]);
        let input = scatter(config, bounds, scatter_budget()).unwrap();
        assert!(input.points.len() > 3);
        let original = input.points.clone();
        let keys: Vec<_> = original.iter().step_by(2).map(|point| point.key).collect();
        let deletions = set(&keys);
        let actual = deletions.filter_source(input);
        assert_eq!(actual.removed, keys.len());
        assert_eq!(actual.source.points.len() + actual.removed, original.len());
        for (index, point) in actual.source.points.iter().enumerate() {
            assert!(original.contains(point));
            for other in &actual.source.points[index + 1..] {
                let squared: i128 = point
                    .position_mm
                    .iter()
                    .zip(other.position_mm)
                    .map(|(a, b)| (i128::from(*a) - i128::from(b)).pow(2))
                    .sum();
                assert!(squared >= i128::from(config.min_distance_mm).pow(2));
            }
        }
    }

    #[test]
    fn same_key_deletion_survives_source_distance_changes() {
        let mut key = key(0);
        // At the origin, the unthinned 100mm candidate is always request-owned.
        key.grid = SectorCoord::new(0, 0, 0);
        let deletions = set(&[key]);
        for distance in [0, 1] {
            let config = ScatterConfig {
                min_distance_mm: distance,
                ..config()
            };
            let input = scatter(config, region([-300; 3], [400; 3]), scatter_budget()).unwrap();
            assert!(input.points.iter().any(|point| point.key == key));
            assert_eq!(deletions.filter_source(input).removed, 1);
        }
    }

    #[test]
    fn source_cache_dependency_retains_full_deletions_even_for_absent_keys() {
        let first = set(&[key(100)]).filter_source(source());
        let second = set(&[key(101)]).filter_source(source());
        assert_eq!(first.source, second.source);
        assert_eq!(first.removed, 0);
        assert_eq!(second.removed, 0);
        assert_ne!(first.deletions, second.deletions);
        assert_ne!(first, second);
        let equal = set(&[key(100), key(100)]).filter_source(source());
        assert_eq!(first, equal);
    }

    #[test]
    fn field_cache_dependency_retains_full_deletions_and_field_recipe() {
        let first = set(&[key(100)]).filter_field(field_source()).unwrap();
        let second = set(&[key(101)]).filter_field(field_source()).unwrap();
        assert_eq!(first.source, second.source);
        assert_ne!(first, second);
        assert_eq!(
            first.source.cache_key.field_version,
            crate::SCATTER_FIELD_VERSION
        );
        assert_eq!(first.source.cache_key.field.config().base_species, 73);
        assert_eq!(first.deletions.keys(), &[key(100)]);
    }

    #[test]
    fn extreme_valid_world_keys_remain_full_precision() {
        let keys = [
            ScatterCandidateKey {
                grid_spacing_mm: 1,
                grid: SectorCoord::new(i64::MIN, 0, 0),
                ..key(0)
            },
            ScatterCandidateKey {
                grid_spacing_mm: 1,
                grid: SectorCoord::new(i64::MAX, 0, 0),
                ..key(0)
            },
        ];
        let deletions = set(&keys);
        for key in keys {
            assert!(deletions.is_deleted(key));
            let point: ScatterPoint = key.rebuild().unwrap();
            assert_eq!(point.position_mm[0], key.grid.x);
        }
        assert_eq!(deletions.keys(), &keys);
    }
}
