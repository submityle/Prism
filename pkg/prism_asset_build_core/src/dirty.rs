//! Deterministic incremental invalidation for local build graphs.
//!
//! A [`DirtySet`] is the exact transitive successor closure of a set of
//! changed jobs. It contains no timestamps, filesystem paths, thread order, or
//! cache policy. Callers can therefore use it for plan-only diagnostics and as
//! the admission set for a later incremental executor without changing build
//! identity semantics.

use crate::graph::{BuildError, BuildGraph};
use crate::job::JobId;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

/// Deterministic set of jobs invalidated by direct changes and their graph
/// successors.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirtySet {
    direct: BTreeSet<JobId>,
    affected: BTreeSet<JobId>,
}

impl DirtySet {
    /// Computes a dirty set from a validated graph and direct changes.
    ///
    /// Every direct change must name a job in `graph`. Missing dependencies and
    /// cycles are rejected before propagation, so a successful result is safe
    /// to consume as a deterministic incremental plan.
    pub fn from_graph(
        graph: &BuildGraph,
        direct_changes: impl IntoIterator<Item = JobId>,
    ) -> Result<Self, DirtySetError> {
        graph.execution_order().map_err(DirtySetError::Graph)?;

        let direct: BTreeSet<_> = direct_changes.into_iter().collect();
        for &job in &direct {
            if graph.job(job).is_none() {
                return Err(DirtySetError::UnknownJob(job));
            }
        }

        let mut dependents: BTreeMap<JobId, Vec<JobId>> = BTreeMap::new();
        for (&job, node) in graph.jobs() {
            for &dependency in node.dependencies() {
                dependents.entry(dependency).or_default().push(job);
            }
        }
        for children in dependents.values_mut() {
            children.sort_unstable();
            children.dedup();
        }

        let mut affected = direct.clone();
        let mut queue: VecDeque<_> = direct.iter().copied().collect();
        while let Some(job) = queue.pop_front() {
            for &dependent in dependents.get(&job).into_iter().flatten() {
                if affected.insert(dependent) {
                    queue.push_back(dependent);
                }
            }
        }

        Ok(Self { direct, affected })
    }

    /// Returns the directly changed jobs, in ascending stable ID order.
    #[must_use]
    pub fn direct_jobs(&self) -> impl Iterator<Item = JobId> + '_ {
        self.direct.iter().copied()
    }

    /// Returns all invalidated jobs, in ascending stable ID order.
    #[must_use]
    pub fn jobs(&self) -> impl Iterator<Item = JobId> + '_ {
        self.affected.iter().copied()
    }

    /// Returns whether `job` is directly changed or transitively affected.
    #[must_use]
    pub fn contains(&self, job: JobId) -> bool {
        self.affected.contains(&job)
    }

    /// Number of invalidated jobs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.affected.len()
    }

    /// Returns whether no job was invalidated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.affected.is_empty()
    }
}

/// Failure while constructing a deterministic dirty set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirtySetError {
    /// The supplied change names no graph job.
    UnknownJob(JobId),
    /// The graph is not a valid DAG.
    Graph(BuildError),
}

impl fmt::Display for DirtySetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownJob(job) => write!(formatter, "dirty change names unknown job {}", job.0),
            Self::Graph(error) => write!(formatter, "cannot compute dirty set: {error}"),
        }
    }
}

impl std::error::Error for DirtySetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::UnknownJob(_) => None,
        }
    }
}
