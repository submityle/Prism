//! Deterministic runtime HLOD selection over derived residency clusters.
//!
//! This module owns selection state and transition records only. HLOD assets,
//! scene truth, proxy decoding, GPU submission, and renderer graph ownership
//! remain with their respective callers.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

use prism_world_residency::{HlodCluster, HlodError, LodTransition};

/// Revision of the runtime HLOD selection contract.
pub const HLOD_RUNTIME_SELECTION_VERSION: u32 = 1;

/// Admission and output limits for one HLOD selection plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HlodSelectionBudget {
    /// Maximum number of clusters in the catalog.
    pub max_clusters: usize,
    /// Maximum number of distance requests in one plan.
    pub max_requests: usize,
    /// Maximum logical output bytes for one plan.
    pub max_output_bytes: usize,
}

impl HlodSelectionBudget {
    /// Creates a budget with checked output accounting.
    pub const fn new(
        max_clusters: usize,
        max_requests: usize,
        max_output_bytes: usize,
    ) -> Result<Self, HlodRuntimeError> {
        let required = match max_requests.checked_mul(core::mem::size_of::<HlodSelection>()) {
            Some(value) => value,
            None => return Err(HlodRuntimeError::ArithmeticOverflow),
        };
        if required > max_output_bytes {
            return Err(HlodRuntimeError::OutputBudgetExceeded {
                required,
                limit: max_output_bytes,
            });
        }
        Ok(Self {
            max_clusters,
            max_requests,
            max_output_bytes,
        })
    }
}

/// One distance sample submitted to the runtime HLOD selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HlodDistanceRequest {
    /// Stable HLOD cluster identity.
    pub cluster_id: u64,
    /// Integer distance in the cluster's declared world unit.
    pub distance: u32,
}

/// One selected representation and optional cross-fade transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HlodSelection {
    /// Stable HLOD cluster identity.
    pub cluster_id: u64,
    /// Selected representation level.
    pub level: usize,
    /// Derived asset key for the selected representation.
    pub asset_key: u64,
    /// Transition from the previously active representation, if changed.
    pub transition: Option<LodTransition>,
}

/// A deterministic, sorted HLOD selection result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HlodSelectionPlan {
    selections: Vec<HlodSelection>,
}

impl HlodSelectionPlan {
    /// Returns selections sorted by stable cluster identity.
    #[must_use]
    pub fn selections(&self) -> &[HlodSelection] {
        &self.selections
    }
}

/// Recoverable runtime HLOD selection errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HlodRuntimeError {
    /// A budget multiplication or byte calculation overflowed.
    ArithmeticOverflow,
    /// A cluster catalog insertion exceeded its configured capacity.
    ClusterCapacityExceeded {
        /// Configured maximum number of catalog clusters.
        limit: usize,
    },
    /// A selection request batch exceeded its configured count.
    RequestCapacityExceeded {
        /// Number of requests in the rejected plan.
        required: usize,
        /// Configured maximum number of requests.
        limit: usize,
    },
    /// The output reservation exceeded its configured byte budget.
    OutputBudgetExceeded {
        /// Logical bytes required by the rejected plan.
        required: usize,
        /// Configured logical output byte limit.
        limit: usize,
    },
    /// A cluster identity was inserted more than once.
    DuplicateCluster {
        /// Stable identity that was already present.
        cluster_id: u64,
    },
    /// A request named a cluster that is not in the catalog.
    UnknownCluster {
        /// Stable identity absent from the catalog.
        cluster_id: u64,
    },
    /// A request batch named a cluster more than once.
    DuplicateRequest {
        /// Stable identity repeated in one request batch.
        cluster_id: u64,
    },
    /// A temporary output reservation failed.
    AllocationFailed,
    /// The underlying HLOD declaration was invalid.
    Hlod(HlodError),
}

impl fmt::Display for HlodRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArithmeticOverflow => f.write_str("HLOD runtime arithmetic overflow"),
            Self::ClusterCapacityExceeded { limit } => {
                write!(f, "HLOD cluster capacity exceeded: limit {limit}")
            }
            Self::RequestCapacityExceeded { required, limit } => {
                write!(f, "HLOD requests {required} exceed limit {limit}")
            }
            Self::OutputBudgetExceeded { required, limit } => {
                write!(f, "HLOD output bytes {required} exceed limit {limit}")
            }
            Self::DuplicateCluster { cluster_id } => {
                write!(f, "duplicate HLOD cluster {cluster_id}")
            }
            Self::UnknownCluster { cluster_id } => write!(f, "unknown HLOD cluster {cluster_id}"),
            Self::DuplicateRequest { cluster_id } => {
                write!(f, "duplicate HLOD request {cluster_id}")
            }
            Self::AllocationFailed => f.write_str("HLOD selection allocation failed"),
            Self::Hlod(error) => write!(f, "HLOD declaration: {error}"),
        }
    }
}

impl core::error::Error for HlodRuntimeError {}

impl From<HlodError> for HlodRuntimeError {
    fn from(error: HlodError) -> Self {
        Self::Hlod(error)
    }
}

/// Runtime catalog and hysteresis state for derived HLOD clusters.
#[derive(Clone, Debug)]
pub struct HlodRuntime {
    budget: HlodSelectionBudget,
    clusters: BTreeMap<u64, HlodCluster>,
    active_levels: BTreeMap<u64, usize>,
}

impl HlodRuntime {
    /// Creates an empty selector with a fixed cluster and output budget.
    pub const fn new(budget: HlodSelectionBudget) -> Self {
        Self {
            budget,
            clusters: BTreeMap::new(),
            active_levels: BTreeMap::new(),
        }
    }

    /// Returns the configured budget.
    #[must_use]
    pub const fn budget(&self) -> HlodSelectionBudget {
        self.budget
    }

    /// Adds one validated cluster without replacing an existing identity.
    pub fn insert(&mut self, cluster: HlodCluster) -> Result<(), HlodRuntimeError> {
        if self.clusters.contains_key(&cluster.id()) {
            return Err(HlodRuntimeError::DuplicateCluster {
                cluster_id: cluster.id(),
            });
        }
        if self.clusters.len() >= self.budget.max_clusters {
            return Err(HlodRuntimeError::ClusterCapacityExceeded {
                limit: self.budget.max_clusters,
            });
        }
        self.clusters.insert(cluster.id(), cluster);
        Ok(())
    }

    /// Removes a cluster and any remembered hysteresis state.
    pub fn remove(&mut self, cluster_id: u64) -> Option<HlodCluster> {
        self.active_levels.remove(&cluster_id);
        self.clusters.remove(&cluster_id)
    }

    /// Returns the current active level, if the cluster has been selected.
    #[must_use]
    pub fn active_level(&self, cluster_id: u64) -> Option<usize> {
        self.active_levels.get(&cluster_id).copied()
    }

    /// Plans all requests atomically and commits hysteresis state only on success.
    pub fn plan(
        &mut self,
        requests: &[HlodDistanceRequest],
    ) -> Result<HlodSelectionPlan, HlodRuntimeError> {
        if requests.len() > self.budget.max_requests {
            return Err(HlodRuntimeError::RequestCapacityExceeded {
                required: requests.len(),
                limit: self.budget.max_requests,
            });
        }
        let required = requests
            .len()
            .checked_mul(core::mem::size_of::<HlodSelection>())
            .ok_or(HlodRuntimeError::ArithmeticOverflow)?;
        if required > self.budget.max_output_bytes {
            return Err(HlodRuntimeError::OutputBudgetExceeded {
                required,
                limit: self.budget.max_output_bytes,
            });
        }
        let mut ordered = requests.to_vec();
        ordered.sort_unstable_by_key(|request| request.cluster_id);
        let mut seen = BTreeSet::new();
        for request in &ordered {
            if !seen.insert(request.cluster_id) {
                return Err(HlodRuntimeError::DuplicateRequest {
                    cluster_id: request.cluster_id,
                });
            }
            if !self.clusters.contains_key(&request.cluster_id) {
                return Err(HlodRuntimeError::UnknownCluster {
                    cluster_id: request.cluster_id,
                });
            }
        }

        let mut selections = Vec::new();
        selections
            .try_reserve_exact(ordered.len())
            .map_err(|_| HlodRuntimeError::AllocationFailed)?;
        let mut next_levels = BTreeMap::new();
        for request in ordered {
            let cluster = self
                .clusters
                .get(&request.cluster_id)
                .expect("validated HLOD cluster identity");
            let current = self
                .active_levels
                .get(&request.cluster_id)
                .copied()
                .unwrap_or(0);
            let level = cluster.transition_target(request.distance, current)?;
            let transition = (level != current)
                .then(|| cluster.transition(request.distance, current, level))
                .transpose()?;
            let asset_key = cluster.levels()[level].asset_key;
            selections.push(HlodSelection {
                cluster_id: request.cluster_id,
                level,
                asset_key,
                transition,
            });
            next_levels.insert(request.cluster_id, level);
        }

        self.active_levels.extend(next_levels);
        Ok(HlodSelectionPlan { selections })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_world_residency::HlodLevel;

    fn budget() -> HlodSelectionBudget {
        HlodSelectionBudget::new(2, 2, 2 * core::mem::size_of::<HlodSelection>()).unwrap()
    }

    fn cluster(id: u64) -> HlodCluster {
        HlodCluster::new(
            id,
            vec![id * 10],
            vec![
                HlodLevel {
                    asset_key: id * 100 + 1,
                    enter_distance: 0,
                    transition_width: 10,
                },
                HlodLevel {
                    asset_key: id * 100 + 2,
                    enter_distance: 100,
                    transition_width: 10,
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn selection_is_sorted_and_hysteretic() {
        let mut runtime = HlodRuntime::new(budget());
        runtime.insert(cluster(2)).unwrap();
        runtime.insert(cluster(1)).unwrap();
        let first = runtime
            .plan(&[
                HlodDistanceRequest {
                    cluster_id: 2,
                    distance: 130,
                },
                HlodDistanceRequest {
                    cluster_id: 1,
                    distance: 0,
                },
            ])
            .unwrap();
        assert_eq!(
            first.selections(),
            &[
                HlodSelection {
                    cluster_id: 1,
                    level: 0,
                    asset_key: 101,
                    transition: None,
                },
                HlodSelection {
                    cluster_id: 2,
                    level: 1,
                    asset_key: 202,
                    transition: Some(LodTransition {
                        from_level: 0,
                        to_level: 1,
                        alpha_q16: 65_535,
                        dither_seed: cluster(2).dither_seed(0, 1).unwrap(),
                    }),
                },
            ]
        );
        let held = runtime
            .plan(&[HlodDistanceRequest {
                cluster_id: 2,
                distance: 95,
            }])
            .unwrap();
        assert_eq!(held.selections()[0].level, 1);
        assert!(held.selections()[0].transition.is_none());
    }

    #[test]
    fn invalid_plan_does_not_commit_partial_state() {
        let mut runtime = HlodRuntime::new(budget());
        runtime.insert(cluster(1)).unwrap();
        let error = runtime
            .plan(&[
                HlodDistanceRequest {
                    cluster_id: 1,
                    distance: 130,
                },
                HlodDistanceRequest {
                    cluster_id: 99,
                    distance: 0,
                },
            ])
            .unwrap_err();
        assert_eq!(error, HlodRuntimeError::UnknownCluster { cluster_id: 99 });
        assert_eq!(runtime.active_level(1), None);
    }

    #[test]
    fn duplicate_requests_and_catalog_entries_fail_closed() {
        let mut runtime = HlodRuntime::new(budget());
        runtime.insert(cluster(1)).unwrap();
        assert_eq!(
            runtime.insert(cluster(1)).unwrap_err(),
            HlodRuntimeError::DuplicateCluster { cluster_id: 1 }
        );
        assert_eq!(
            runtime
                .plan(&[
                    HlodDistanceRequest {
                        cluster_id: 1,
                        distance: 0,
                    },
                    HlodDistanceRequest {
                        cluster_id: 1,
                        distance: 1,
                    },
                ])
                .unwrap_err(),
            HlodRuntimeError::DuplicateRequest { cluster_id: 1 }
        );
    }
}
