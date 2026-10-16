//! Content-neutral deletion filtering for planetary placement outputs.
//!
//! Deletion identity is the complete [`PlanetPlacementKey`]. The filtered
//! output keeps the original placement recipe and region, so deleting derived
//! rows never regenerates candidates or changes ownership.

use alloc::{sync::Arc, vec::Vec};
use core::{cmp::Ordering, fmt, mem::size_of};

use crate::{PlanetPlacementKey, PlanetPlacementOutput, PLANET_PLACEMENT_VERSION};

/// Revision of the planetary placement deletion contract.
pub const PLANET_PLACEMENT_DELETION_VERSION: u32 = 1;

/// Bounds for a supplied or edited deletion-key set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanetPlacementDeletionBudget {
    /// Maximum key count before deduplication or after an edit.
    pub max_keys: usize,
    /// Maximum key Vec payload bytes, including spare capacity.
    pub max_key_bytes: usize,
}

/// Sorted, unique, immutable complete planetary placement identities.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PlanetPlacementDeletionSet(Arc<Vec<PlanetPlacementKey>>);

/// Placement survivors plus the complete deletion dependency.
#[derive(Debug, PartialEq)]
pub struct DeletionFilteredPlanetPlacement {
    /// Source output with matching rows removed in place.
    pub source: PlanetPlacementOutput,
    /// Complete deletion input, including keys absent from the source.
    pub deletions: PlanetPlacementDeletionSet,
    /// Number of source rows removed.
    pub removed: usize,
}

/// Validation, budget, or allocation failure for planetary deletion state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanetPlacementDeletionError {
    /// A key does not belong to the planetary placement contract.
    InvalidKey,
    /// Key count exceeds the configured bound.
    KeyBudgetExceeded {
        /// Required key count.
        required: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Key Vec payload exceeds the configured byte bound.
    KeyBytesBudgetExceeded {
        /// Required payload bytes.
        required: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// A count-to-byte calculation overflowed.
    ArithmeticOverflow,
    /// A bounded key allocation failed.
    AllocationFailed,
}

impl fmt::Display for PlanetPlacementDeletionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKey => f.write_str("invalid planetary placement deletion key"),
            Self::KeyBudgetExceeded { required, limit } => {
                write!(f, "planetary deletion needs {required} keys, limit {limit}")
            }
            Self::KeyBytesBudgetExceeded { required, limit } => write!(
                f,
                "planetary deletion key payload needs {required} bytes, limit {limit}"
            ),
            Self::ArithmeticOverflow => f.write_str("planetary deletion size overflow"),
            Self::AllocationFailed => f.write_str("planetary deletion key allocation failed"),
        }
    }
}

impl core::error::Error for PlanetPlacementDeletionError {}

impl PlanetPlacementDeletionSet {
    /// Validates, sorts, and deduplicates complete keys in the supplied Vec.
    pub fn new(
        mut keys: Vec<PlanetPlacementKey>,
        budget: PlanetPlacementDeletionBudget,
    ) -> Result<Self, PlanetPlacementDeletionError> {
        check_budget(keys.len(), keys.capacity(), budget)?;
        if keys.iter().any(|key| !valid_key(*key)) {
            return Err(PlanetPlacementDeletionError::InvalidKey);
        }
        keys.sort_unstable_by(compare_keys);
        keys.dedup_by(|left, right| compare_keys(left, right) == Ordering::Equal);
        Ok(Self(Arc::new(keys)))
    }

    /// Returns canonical complete deletion identities.
    #[must_use]
    pub fn keys(&self) -> &[PlanetPlacementKey] {
        &self.0
    }

    /// Tests complete-key membership with binary search.
    #[must_use]
    pub fn is_deleted(&self, key: PlanetPlacementKey) -> bool {
        self.keys()
            .binary_search_by(|candidate| compare_keys(candidate, &key))
            .is_ok()
    }

    /// Returns the charged key payload bytes, including spare capacity.
    #[must_use]
    pub fn allocated_key_bytes(&self) -> usize {
        self.0.capacity() * size_of::<PlanetPlacementKey>()
    }

    /// Adds or removes one key without mutating this immutable set.
    pub fn with_deleted(
        &self,
        key: PlanetPlacementKey,
        deleted: bool,
        budget: PlanetPlacementDeletionBudget,
    ) -> Result<Self, PlanetPlacementDeletionError> {
        if !valid_key(key) {
            return Err(PlanetPlacementDeletionError::InvalidKey);
        }
        let found = self
            .keys()
            .binary_search_by(|candidate| compare_keys(candidate, &key));
        let (index, new_len) = match (found, deleted) {
            (Ok(_), true) | (Err(_), false) => return Ok(self.clone()),
            (Ok(index), false) => (index, self.0.len() - 1),
            (Err(index), true) => (
                index,
                self.0
                    .len()
                    .checked_add(1)
                    .ok_or(PlanetPlacementDeletionError::ArithmeticOverflow)?,
            ),
        };
        let mut next = reserve_keys(new_len, budget)?;
        next.extend_from_slice(&self.keys()[..index]);
        if deleted {
            next.push(key);
            next.extend_from_slice(&self.keys()[index..]);
        } else {
            next.extend_from_slice(&self.keys()[index + 1..]);
        }
        Ok(Self(Arc::new(next)))
    }
}

/// Removes exact complete-key matches while retaining source metadata and Vec
/// allocations. The input is consumed and no replacement output is allocated.
pub fn filter_planet_placement(
    mut source: PlanetPlacementOutput,
    deletions: &PlanetPlacementDeletionSet,
) -> Result<DeletionFilteredPlanetPlacement, PlanetPlacementDeletionError> {
    if deletions.keys().is_empty() {
        return Ok(DeletionFilteredPlanetPlacement {
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
            }
            write += 1;
        }
    }
    source.points.truncate(write);
    Ok(DeletionFilteredPlanetPlacement {
        source,
        deletions: deletions.clone(),
        removed: original_len - write,
    })
}

fn valid_key(key: PlanetPlacementKey) -> bool {
    key.placement_version == PLANET_PLACEMENT_VERSION
}

fn compare_keys(left: &PlanetPlacementKey, right: &PlanetPlacementKey) -> Ordering {
    (
        left.placement_version,
        left.seed,
        left.version,
        left.salt,
        left.tile.face().index(),
        left.tile.level(),
        left.tile.x(),
        left.tile.y(),
        left.cell[0],
        left.cell[1],
    )
        .cmp(&(
            right.placement_version,
            right.seed,
            right.version,
            right.salt,
            right.tile.face().index(),
            right.tile.level(),
            right.tile.x(),
            right.tile.y(),
            right.cell[0],
            right.cell[1],
        ))
}

fn key_bytes(capacity: usize) -> Result<usize, PlanetPlacementDeletionError> {
    capacity
        .checked_mul(size_of::<PlanetPlacementKey>())
        .ok_or(PlanetPlacementDeletionError::ArithmeticOverflow)
}

fn check_budget(
    len: usize,
    capacity: usize,
    budget: PlanetPlacementDeletionBudget,
) -> Result<(), PlanetPlacementDeletionError> {
    if len > budget.max_keys {
        return Err(PlanetPlacementDeletionError::KeyBudgetExceeded {
            required: len,
            limit: budget.max_keys,
        });
    }
    let required = key_bytes(capacity)?;
    if required > budget.max_key_bytes {
        return Err(PlanetPlacementDeletionError::KeyBytesBudgetExceeded {
            required,
            limit: budget.max_key_bytes,
        });
    }
    Ok(())
}

fn reserve_keys(
    len: usize,
    budget: PlanetPlacementDeletionBudget,
) -> Result<Vec<PlanetPlacementKey>, PlanetPlacementDeletionError> {
    check_budget(len, len, budget)?;
    let mut keys = Vec::new();
    keys.try_reserve_exact(len)
        .map_err(|_| PlanetPlacementDeletionError::AllocationFailed)?;
    check_budget(len, keys.capacity(), budget)?;
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{CubeFace, PlanetPlacementConfig, PlanetPlacementRegion, TileKey};

    fn key(cell: [u32; 2]) -> PlanetPlacementKey {
        PlanetPlacementKey {
            placement_version: PLANET_PLACEMENT_VERSION,
            seed: 7,
            version: 3,
            salt: 11,
            tile: TileKey::root(CubeFace::PositiveZ),
            cell,
        }
    }

    fn output() -> PlanetPlacementOutput {
        PlanetPlacementOutput {
            points: vec![
                crate::PlanetPlacementPoint {
                    key: key([0, 0]),
                    position_mm: [0, 1, 2],
                    normal: [0.0, 1.0, 0.0],
                    surface_height_numerator_mm: 4,
                    surface_height_denominator: 4,
                },
                crate::PlanetPlacementPoint {
                    key: key([1, 0]),
                    position_mm: [3, 4, 5],
                    normal: [0.0, 1.0, 0.0],
                    surface_height_numerator_mm: 8,
                    surface_height_denominator: 4,
                },
            ],
            config: PlanetPlacementConfig {
                seed: 7,
                version: 3,
                salt: 11,
                radius_mm: 1_000,
                origin_mm: [0; 3],
                vertical_offset_mm: 0,
                acceptance_threshold: u16::MAX,
            },
            region: PlanetPlacementRegion::new(TileKey::root(CubeFace::PositiveZ), [0, 0], [2, 1])
                .unwrap(),
            candidate_evaluations: 2,
        }
    }

    fn budget() -> PlanetPlacementDeletionBudget {
        PlanetPlacementDeletionBudget {
            max_keys: usize::MAX,
            max_key_bytes: usize::MAX,
        }
    }

    #[test]
    fn keys_are_sorted_and_deduplicated_by_complete_identity() {
        let set =
            PlanetPlacementDeletionSet::new(vec![key([1, 0]), key([0, 0]), key([1, 0])], budget())
                .unwrap();
        assert_eq!(set.keys(), &[key([0, 0]), key([1, 0])]);
        assert!(set.is_deleted(key([1, 0])));
    }

    #[test]
    fn filtering_preserves_recipe_region_and_capacity() {
        let source = output();
        let points_capacity = source.points.capacity();
        let config = source.config;
        let region = source.region;
        let deletions = PlanetPlacementDeletionSet::new(vec![key([0, 0])], budget()).unwrap();
        let filtered = filter_planet_placement(source, &deletions).unwrap();
        assert_eq!(filtered.removed, 1);
        assert_eq!(filtered.source.points.len(), 1);
        assert_eq!(filtered.source.points.capacity(), points_capacity);
        assert_eq!(filtered.source.config, config);
        assert_eq!(filtered.source.region, region);
    }

    #[test]
    fn invalid_key_and_budget_fail_before_publication() {
        let invalid = PlanetPlacementKey {
            placement_version: PLANET_PLACEMENT_VERSION + 1,
            ..key([0, 0])
        };
        assert_eq!(
            PlanetPlacementDeletionSet::new(vec![invalid], budget()),
            Err(PlanetPlacementDeletionError::InvalidKey)
        );
        assert!(matches!(
            PlanetPlacementDeletionSet::new(
                vec![key([0, 0])],
                PlanetPlacementDeletionBudget {
                    max_keys: 0,
                    max_key_bytes: usize::MAX,
                },
            ),
            Err(PlanetPlacementDeletionError::KeyBudgetExceeded { .. })
        ));
    }
}
