//! Ordered stage/schedule pipeline (design §15 "App 子应用流水线").
//!
//! A [`Pipeline`] is the lightweight hook an App uses to drive sub-application
//! pipelines on this pool: it runs an ordered list of **stages**, and within a
//! stage runs that stage's jobs in parallel. A stage does not begin until the
//! previous stage has fully completed, so stages observe each other's effects
//! in declaration order while still exploiting intra-stage parallelism.
//!
//! Stages run through [`TaskPool::scope`](crate::TaskPool::scope), so jobs may
//! borrow from the enclosing environment, the calling thread helps, and every
//! job is joined before the next stage (and before [`Pipeline::run`] returns).
//!
//! ```
//! # use prism_tasks::TaskPool;
//! use std::sync::Mutex;
//! let pool = TaskPool::with_threads(4);
//! let log = Mutex::new(Vec::new());
//! let mut pipeline = pool.pipeline();
//! pipeline.add_stage("extract").job(|| log.lock().unwrap().push("extract"));
//! pipeline.add_stage("simulate").job(|| log.lock().unwrap().push("simulate"));
//! pipeline.run();
//! assert_eq!(*log.lock().unwrap(), vec!["extract", "simulate"]);
//! ```

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::TaskPool;

#[cfg(doc)]
use crate::JobTrace;

/// A boxed stage job that may borrow from the pipeline's `'env` environment.
type StageJob<'env> = Box<dyn FnOnce() + Send + 'env>;

/// One stage of a [`Pipeline`]: a named group of jobs that run in parallel.
pub struct Stage<'env> {
    name: String,
    jobs: Vec<StageJob<'env>>,
}

impl<'env> Stage<'env> {
    /// Add a job to this stage. Jobs within a stage run in parallel.
    pub fn job<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce() + Send + 'env,
    {
        self.jobs.push(Box::new(f));
        self
    }

    /// This stage's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of jobs queued in this stage.
    #[must_use]
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }
}

/// An ordered pipeline of [`Stage`]s bound to a [`TaskPool`] (see
/// [`TaskPool::pipeline`]).
pub struct Pipeline<'p, 'env> {
    pool: &'p TaskPool,
    stages: Vec<Stage<'env>>,
}

impl<'p, 'env> Pipeline<'p, 'env> {
    /// Append a new stage and return a mutable handle for adding jobs to it.
    pub fn add_stage(&mut self, name: impl Into<String>) -> &mut Stage<'env> {
        self.stages.push(Stage {
            name: name.into(),
            jobs: Vec::new(),
        });
        self.stages
            .last_mut()
            .expect("a stage was just pushed")
    }

    /// Number of stages.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// The stage names in declaration order.
    #[must_use]
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(Stage::name).collect()
    }

    /// Run the pipeline: execute each stage in declaration order, running that
    /// stage's jobs in parallel and joining them before the next stage begins.
    pub fn run(self) {
        let Pipeline { pool, stages } = self;
        for stage in stages {
            if stage.jobs.is_empty() {
                continue;
            }
            pool.scope(|s| {
                for job in stage.jobs {
                    s.spawn(job);
                }
            });
        }
    }

    /// Like [`Pipeline::run`], but records a [`JobTrace`] span per stage job
    /// (span name = stage name), so the run shows up in the trace / chrome
    /// export with real per-worker timing.
    pub fn run_traced(self, trace: &crate::JobTrace) {
        let Pipeline { pool, stages } = self;
        for stage in stages {
            if stage.jobs.is_empty() {
                continue;
            }
            let name = stage.name;
            pool.scope(|s| {
                for job in stage.jobs {
                    let instrumented = trace.instrument(name.clone(), job);
                    s.spawn(instrumented);
                }
            });
        }
    }
}

impl TaskPool {
    /// Open an ordered [`Pipeline`] on this pool (design §15). Add stages with
    /// [`Pipeline::add_stage`] and jobs with [`Stage::job`], then
    /// [`Pipeline::run`].
    #[must_use]
    pub fn pipeline(&self) -> Pipeline<'_, '_> {
        Pipeline {
            pool: self,
            stages: Vec::new(),
        }
    }
}
