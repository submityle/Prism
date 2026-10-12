use crate::job::{BuildJob, JobId, ProcessorError};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Outputs produced by a successful graph execution, keyed by stable job ID.
pub type BuildOutputs = BTreeMap<JobId, Vec<u8>>;

/// Validation or execution failure for a build graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BuildError {
    DuplicateJob(JobId),
    MissingDependency { job: JobId, dependency: JobId },
    Cycle { jobs: Vec<JobId> },
    ProcessorFailed { job: JobId, source: ProcessorError },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateJob(id) => write!(f, "duplicate build job {}", id.0),
            Self::MissingDependency { job, dependency } => write!(
                f,
                "build job {} depends on missing job {}",
                job.0, dependency.0
            ),
            Self::Cycle { jobs } => write!(f, "build graph contains a cycle involving {jobs:?}"),
            Self::ProcessorFailed { job, source } => {
                write!(f, "build job {} failed: {source}", job.0)
            }
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ProcessorFailed { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Deterministic, single-threaded DAG of local artifact-producing jobs.
///
/// The graph accepts jobs in any order. Missing references and cycles are
/// reported when an execution order is requested or the graph is executed.
/// Independent ready jobs are ordered by ascending `JobId`; this makes the
/// complete topological order independent of insertion order.
#[derive(Default)]
pub struct BuildGraph {
    jobs: BTreeMap<JobId, BuildJob>,
}

impl BuildGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a job, rejecting duplicate IDs immediately.
    pub fn add_job(&mut self, job: BuildJob) -> Result<(), BuildError> {
        let id = job.id();
        if self.jobs.contains_key(&id) {
            return Err(BuildError::DuplicateJob(id));
        }
        self.jobs.insert(id, job);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    pub(crate) fn job(&self, id: JobId) -> Option<&BuildJob> {
        self.jobs.get(&id)
    }

    /// Return the stable topological order, or a deterministic graph error.
    pub fn execution_order(&self) -> Result<Vec<JobId>, BuildError> {
        let mut indegree = BTreeMap::new();
        let mut dependents: BTreeMap<JobId, Vec<JobId>> = BTreeMap::new();

        for (&id, job) in &self.jobs {
            let mut degree = 0usize;
            for &dependency in job.dependencies() {
                if !self.jobs.contains_key(&dependency) {
                    return Err(BuildError::MissingDependency {
                        job: id,
                        dependency,
                    });
                }
                degree += 1;
                dependents.entry(dependency).or_default().push(id);
            }
            indegree.insert(id, degree);
        }

        let mut ready: BTreeSet<JobId> = indegree
            .iter()
            .filter_map(|(&id, &degree)| (degree == 0).then_some(id))
            .collect();
        let mut order = Vec::with_capacity(self.jobs.len());

        while let Some(id) = ready.pop_first() {
            order.push(id);
            if let Some(children) = dependents.get(&id) {
                for child in children {
                    let degree = indegree
                        .get_mut(child)
                        .expect("dependent job must be in indegree map");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(*child);
                    }
                }
            }
        }

        if order.len() != self.jobs.len() {
            let jobs = indegree
                .into_iter()
                .filter_map(|(id, degree)| (degree > 0).then_some(id))
                .collect();
            return Err(BuildError::Cycle { jobs });
        }
        Ok(order)
    }

    /// Execute each job serially in the deterministic topological order.
    /// Outputs from dependencies are passed in the dependent job's declared
    /// dependency order. A failure stops execution and returns no partial map.
    pub fn execute(&self) -> Result<BuildOutputs, BuildError> {
        let order = self.execution_order()?;
        let mut outputs = BuildOutputs::new();

        for id in order {
            let job = self.jobs.get(&id).expect("ordered job must exist");
            let inputs: Vec<&[u8]> = job
                .dependencies()
                .iter()
                .map(|dependency| {
                    outputs
                        .get(dependency)
                        .expect("dependencies precede their dependents")
                        .as_slice()
                })
                .collect();
            let bytes = job
                .process(&inputs)
                .map_err(|source| BuildError::ProcessorFailed { job: id, source })?;
            outputs.insert(id, bytes);
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::ProcessorError;

    fn job(id: u128, dependencies: &[u128], output: &'static [u8]) -> BuildJob {
        BuildJob::new(
            JobId(id),
            dependencies.iter().copied().map(JobId).collect::<Vec<_>>(),
            move |_: &[&[u8]]| Ok(output.to_vec()),
        )
    }

    #[test]
    fn topological_order_is_stable_across_insertion_orders() {
        let mut first = BuildGraph::new();
        first.add_job(job(4, &[1, 2], b"d")).unwrap();
        first.add_job(job(2, &[], b"b")).unwrap();
        first.add_job(job(3, &[1], b"c")).unwrap();
        first.add_job(job(1, &[], b"a")).unwrap();

        let mut second = BuildGraph::new();
        second.add_job(job(1, &[], b"a")).unwrap();
        second.add_job(job(3, &[1], b"c")).unwrap();
        second.add_job(job(2, &[], b"b")).unwrap();
        second.add_job(job(4, &[1, 2], b"d")).unwrap();

        let expected = vec![JobId(1), JobId(2), JobId(3), JobId(4)];
        assert_eq!(first.execution_order().unwrap(), expected);
        assert_eq!(second.execution_order().unwrap(), expected);
    }

    #[test]
    fn rejects_duplicate_job_ids() {
        let mut graph = BuildGraph::new();
        graph.add_job(job(7, &[], b"first")).unwrap();
        assert_eq!(
            graph.add_job(job(7, &[], b"second")),
            Err(BuildError::DuplicateJob(JobId(7)))
        );
    }

    #[test]
    fn reports_missing_dependencies() {
        let mut graph = BuildGraph::new();
        graph.add_job(job(3, &[9], b"unused")).unwrap();
        assert_eq!(
            graph.execution_order(),
            Err(BuildError::MissingDependency {
                job: JobId(3),
                dependency: JobId(9),
            })
        );
    }

    #[test]
    fn reports_cycles_with_stable_job_ids() {
        let mut graph = BuildGraph::new();
        graph.add_job(job(2, &[1], b"unused")).unwrap();
        graph.add_job(job(3, &[2], b"unused")).unwrap();
        graph.add_job(job(1, &[3], b"unused")).unwrap();
        assert_eq!(
            graph.execution_order(),
            Err(BuildError::Cycle {
                jobs: vec![JobId(1), JobId(2), JobId(3)]
            })
        );
    }

    #[test]
    fn executes_real_transforms_and_preserves_declared_input_order() {
        let mut graph = BuildGraph::new();
        graph.add_job(job(2, &[], b"left")).unwrap();
        graph.add_job(job(1, &[], b"right")).unwrap();
        graph
            .add_job(BuildJob::new(
                JobId(3),
                vec![JobId(2), JobId(1)],
                |inputs: &[&[u8]]| {
                    let mut output = inputs[0].to_vec();
                    output.extend_from_slice(b"+");
                    output.extend_from_slice(inputs[1]);
                    Ok(output)
                },
            ))
            .unwrap();

        let outputs = graph.execute().unwrap();
        assert_eq!(outputs[&JobId(3)], b"left+right");
    }

    #[test]
    fn processor_failure_identifies_the_job() {
        let mut graph = BuildGraph::new();
        graph
            .add_job(BuildJob::new(JobId(5), vec![], |_: &[&[u8]]| {
                Err(ProcessorError::new("invalid fixture"))
            }))
            .unwrap();
        assert_eq!(
            graph.execute(),
            Err(BuildError::ProcessorFailed {
                job: JobId(5),
                source: ProcessorError::new("invalid fixture"),
            })
        );
    }
}
