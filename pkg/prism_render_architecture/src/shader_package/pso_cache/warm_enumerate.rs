//! Bridges a package's [`PermutationSpace`] into deterministic warm requests.
//!
//! [`WarmSetPlanner`](super::WarmSetPlanner) accepts individual
//! [`PsoCacheKey`]s, but the authoritative source of *which* variants a package
//! can produce is its [`PermutationSpace`] (the mixed-radix feature switches)
//! crossed with the distinct render states it is drawn under. This module turns
//! that authoritative description into the concrete `(key, priority)` set a
//! planner should require, so the warm set is derived from the real permutation
//! source rather than hand-maintained key lists that can silently drift.
//!
//! A package is drawn under one or more fixed-function / render-target states
//! (each a [`PipelineStateHash`]); the same shader permutation compiles to a
//! distinct pipeline per state, so the enumerated variant set is the Cartesian
//! product `permutation_index × state`.
//!
//! Enumeration is explicitly budgeted: a permutation space can be astronomically
//! large, so [`PackageWarmSpec::enumerate`] refuses to materialize a product
//! that exceeds a caller-supplied cap instead of allocating without bound. All
//! output is deterministically ordered so golden tests reproduce exactly.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::shader_package::{PermutationSpace, ShaderPackageId};

use super::{PipelineStateHash, PsoCacheKey, WarmPriority, WarmSetPlanner};

/// Describes how one shader package's permutation space maps into warm
/// requests: the space itself, the render states it is drawn under, the base
/// priority for ordinary variants, and the subset of permutation indices that
/// must be treated as [`WarmPriority::Critical`].
#[derive(Clone, Debug)]
pub struct PackageWarmSpec {
    package: ShaderPackageId,
    space: PermutationSpace,
    states: Vec<PipelineStateHash>,
    base_priority: WarmPriority,
    critical: BTreeSet<u64>,
}

impl PackageWarmSpec {
    /// Creates a spec for `package` over `space`, drawn under a single default
    /// render state at [`WarmPriority::Normal`] with no critical overrides.
    #[must_use]
    pub fn new(package: ShaderPackageId, space: PermutationSpace) -> Self {
        Self {
            package,
            space,
            states: alloc::vec![PipelineStateHash::default()],
            base_priority: WarmPriority::Normal,
            critical: BTreeSet::new(),
        }
    }

    /// Sets the render states the package is drawn under.
    ///
    /// Duplicate hashes collapse and the set is kept sorted for deterministic
    /// enumeration. An empty iterator falls back to a single default state so a
    /// package always enumerates at least its permutation axis.
    #[must_use]
    pub fn with_states(mut self, states: impl IntoIterator<Item = PipelineStateHash>) -> Self {
        let unique: BTreeSet<PipelineStateHash> = states.into_iter().collect();
        self.states = if unique.is_empty() {
            alloc::vec![PipelineStateHash::default()]
        } else {
            unique.into_iter().collect()
        };
        self
    }

    /// Sets the base priority assigned to every non-critical variant.
    #[must_use]
    pub fn with_base_priority(mut self, priority: WarmPriority) -> Self {
        self.base_priority = priority;
        self
    }

    /// Marks permutation indices that must warm at [`WarmPriority::Critical`]
    /// (for example the default material variant that has to be resident before
    /// the first frame). Indices outside the space are ignored.
    #[must_use]
    pub fn mark_critical(mut self, indices: impl IntoIterator<Item = u64>) -> Self {
        let total = self.space.total_permutations();
        self.critical
            .extend(indices.into_iter().filter(|&index| index < total));
        self
    }

    /// The package this spec enumerates.
    #[must_use]
    pub fn package(&self) -> &ShaderPackageId {
        &self.package
    }

    /// The permutation space being enumerated.
    #[must_use]
    pub fn space(&self) -> &PermutationSpace {
        &self.space
    }

    /// The sorted, de-duplicated render states being enumerated.
    #[must_use]
    pub fn states(&self) -> &[PipelineStateHash] {
        &self.states
    }

    /// Total number of `(permutation, state)` variants this spec would emit.
    ///
    /// Saturates at [`u64::MAX`] rather than overflowing, matching
    /// [`PermutationSpace::total_permutations`].
    #[must_use]
    pub fn variant_count(&self) -> u64 {
        self.space
            .total_permutations()
            .saturating_mul(self.states.len() as u64)
    }

    /// Priority for a given permutation index: [`WarmPriority::Critical`] when
    /// the index was marked critical, otherwise the configured base priority.
    #[must_use]
    fn priority_for(&self, permutation_index: u64) -> WarmPriority {
        if self.critical.contains(&permutation_index) {
            WarmPriority::Critical
        } else {
            self.base_priority
        }
    }

    /// Materializes every `(key, priority)` warm request for this package.
    ///
    /// Output is ordered by permutation index then render state, so repeated
    /// runs produce byte-identical plans. Critical-marked permutations keep
    /// their position but carry [`WarmPriority::Critical`]; the planner is
    /// responsible for the final priority-first ordering.
    ///
    /// # Errors
    ///
    /// Returns [`WarmEnumerationError::ExceedsCap`] when [`Self::variant_count`]
    /// is greater than `cap`, so an unexpectedly huge permutation space cannot
    /// trigger an unbounded allocation.
    pub fn enumerate(
        &self,
        cap: u64,
    ) -> Result<Vec<(PsoCacheKey, WarmPriority)>, WarmEnumerationError> {
        let requested = self.variant_count();
        if requested > cap {
            return Err(WarmEnumerationError::ExceedsCap { requested, cap });
        }

        let total = self.space.total_permutations();
        let mut out = Vec::with_capacity(requested as usize);
        for permutation_index in 0..total {
            let priority = self.priority_for(permutation_index);
            for &state in &self.states {
                out.push((
                    PsoCacheKey::new(self.package.clone(), permutation_index, state),
                    priority,
                ));
            }
        }
        Ok(out)
    }
}

/// Why enumerating a [`PackageWarmSpec`] into warm requests failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WarmEnumerationError {
    /// The `(permutation × state)` product exceeds the caller's budget.
    ExceedsCap {
        /// Variants the spec would have produced.
        requested: u64,
        /// Budget the caller allowed.
        cap: u64,
    },
}

impl WarmSetPlanner {
    /// Requires every variant a [`PackageWarmSpec`] enumerates, deriving the
    /// warm set from the authoritative permutation source.
    ///
    /// Returns the number of distinct variants registered on success.
    ///
    /// # Errors
    ///
    /// Propagates [`WarmEnumerationError`] when the spec's variant count exceeds
    /// `cap` (see [`PackageWarmSpec::enumerate`]); the planner is left
    /// unchanged in that case.
    pub fn require_package(
        &mut self,
        spec: &PackageWarmSpec,
        cap: u64,
    ) -> Result<usize, WarmEnumerationError> {
        let items = spec.enumerate(cap)?;
        let count = items.len();
        self.require_all(items);
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pkg(name: &str) -> ShaderPackageId {
        ShaderPackageId::new(name)
    }

    #[test]
    fn single_state_enumerates_each_permutation_once() {
        let spec = PackageWarmSpec::new(pkg("mat"), PermutationSpace::new([2, 3]));
        assert_eq!(spec.variant_count(), 6);
        let items = spec.enumerate(1_000).unwrap();
        assert_eq!(items.len(), 6);
        // Default single state, all Normal, indices 0..6 in order.
        for (expected_index, (key, priority)) in items.iter().enumerate() {
            assert_eq!(key.permutation_index, expected_index as u64);
            assert_eq!(key.state, PipelineStateHash::default());
            assert_eq!(*priority, WarmPriority::Normal);
        }
    }

    #[test]
    fn states_multiply_the_variant_product() {
        let spec = PackageWarmSpec::new(pkg("mat"), PermutationSpace::new([2]))
            .with_states([PipelineStateHash(7), PipelineStateHash(3)]);
        // 2 permutations × 2 states.
        assert_eq!(spec.variant_count(), 4);
        let items = spec.enumerate(100).unwrap();
        assert_eq!(items.len(), 4);
        // States are sorted (3 before 7) and nested under each permutation.
        assert_eq!(items[0].0.permutation_index, 0);
        assert_eq!(items[0].0.state, PipelineStateHash(3));
        assert_eq!(items[1].0.state, PipelineStateHash(7));
        assert_eq!(items[2].0.permutation_index, 1);
        assert_eq!(items[2].0.state, PipelineStateHash(3));
    }

    #[test]
    fn duplicate_states_collapse() {
        let spec = PackageWarmSpec::new(pkg("mat"), PermutationSpace::new([3])).with_states([
            PipelineStateHash(1),
            PipelineStateHash(1),
            PipelineStateHash(1),
        ]);
        assert_eq!(spec.states().len(), 1);
        assert_eq!(spec.variant_count(), 3);
    }

    #[test]
    fn empty_states_fall_back_to_default() {
        let spec = PackageWarmSpec::new(pkg("mat"), PermutationSpace::new([4])).with_states([]);
        assert_eq!(spec.states(), [PipelineStateHash::default()]);
        assert_eq!(spec.variant_count(), 4);
    }

    #[test]
    fn critical_marks_only_listed_indices() {
        let spec = PackageWarmSpec::new(pkg("mat"), PermutationSpace::new([4]))
            .with_base_priority(WarmPriority::Background)
            .mark_critical([0, 2]);
        let items = spec.enumerate(100).unwrap();
        assert_eq!(items[0].1, WarmPriority::Critical);
        assert_eq!(items[1].1, WarmPriority::Background);
        assert_eq!(items[2].1, WarmPriority::Critical);
        assert_eq!(items[3].1, WarmPriority::Background);
    }

    #[test]
    fn critical_ignores_out_of_range_indices() {
        let spec =
            PackageWarmSpec::new(pkg("mat"), PermutationSpace::new([2])).mark_critical([0, 99]);
        let items = spec.enumerate(100).unwrap();
        assert_eq!(items[0].1, WarmPriority::Critical);
        assert_eq!(items[1].1, WarmPriority::Normal);
    }

    #[test]
    fn enumerate_rejects_when_product_exceeds_cap() {
        let spec = PackageWarmSpec::new(pkg("mat"), PermutationSpace::new([10, 10]));
        assert_eq!(spec.variant_count(), 100);
        let err = spec.enumerate(99).unwrap_err();
        assert_eq!(
            err,
            WarmEnumerationError::ExceedsCap {
                requested: 100,
                cap: 99,
            }
        );
    }

    #[test]
    fn require_package_feeds_planner_and_dedups_strongest() {
        let mut planner = WarmSetPlanner::new();
        let spec = PackageWarmSpec::new(pkg("mat"), PermutationSpace::new([3])).mark_critical([1]);
        let count = planner.require_package(&spec, 100).unwrap();
        assert_eq!(count, 3);
        assert_eq!(planner.len(), 3);
        // The critical permutation (index 1) warms first.
        let plan = planner.plan();
        assert_eq!(plan.entries()[0].key.permutation_index, 1);
        assert_eq!(plan.entries()[0].priority, WarmPriority::Critical);
    }

    #[test]
    fn require_package_over_cap_leaves_planner_unchanged() {
        let mut planner = WarmSetPlanner::new();
        let spec = PackageWarmSpec::new(pkg("mat"), PermutationSpace::new([5, 5]));
        assert!(planner.require_package(&spec, 10).is_err());
        assert!(planner.is_empty());
    }
}
