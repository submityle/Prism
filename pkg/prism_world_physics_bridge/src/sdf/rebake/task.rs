//! Optional shared-task-pool adapter; publication stays on the polling owner.

extern crate alloc;

use alloc::boxed::Box;
use core::{any::Any, fmt};
use prism_tasks::{CancelToken, JobHandle, TaskPool};

use super::{
    widen, SdfRebakeError, SdfRebakePoint, SdfRebakeReport, SdfRebakeUpdate, SdfRebakeWork,
    WorldSdfPhysicsBridge,
};

/// A sampling, publication, lifecycle, or captured worker-panic failure.
#[derive(Debug)]
pub enum SdfRebakeTaskError<E> {
    /// Sampling, cancellation, or base publication admission failed.
    Rebake(SdfRebakeError<E>),
    /// The evaluator unwound; the original payload remains available to inspect.
    Panicked(Box<dyn Any + Send + 'static>),
    /// This task's result was already collected by an earlier poll.
    AlreadyCollected,
}

impl<E: fmt::Display> fmt::Display for SdfRebakeTaskError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rebake(error) => error.fmt(f),
            Self::Panicked(_) => f.write_str("world SDF evaluator panicked"),
            Self::AlreadyCollected => f.write_str("world SDF task result already collected"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for SdfRebakeTaskError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Rebake(error) => Some(error),
            _ => None,
        }
    }
}

/// Pollable, cooperatively cancellable rebake on an existing task pool.
///
/// Dropping a pending task requests cancellation without waiting. An executing
/// evaluator cannot be interrupted; cancellation is checked between samples,
/// after the final sample, and immediately before commit. Dropping never
/// publishes data. Polling does not help execute queued jobs on the caller.
/// The task borrows its pool, so dropping a handle cannot trigger final pool
/// shutdown and implicitly join workers. Pool shutdown remains the owner's job.
#[must_use]
pub struct SdfRebakeTask<'pool, E> {
    job: Option<JobHandle<Result<SdfRebakeUpdate, SdfRebakeError<E>>>>,
    cancel: CancelToken,
    _pool: &'pool TaskPool,
}

impl<E> fmt::Debug for SdfRebakeTask<'_, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SdfRebakeTask")
            .field("job", &self.job)
            .field("cancelled", &self.cancel.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl SdfRebakeWork {
    /// Samples on the supplied pool, retaining publication on the polling owner.
    ///
    /// A child of the supplied cancellation scope isolates per-task cancellation.
    /// The closure must own a coherent composed source for this entire revision.
    /// With zero workers or the pool's single feature, sampling runs inline
    /// before this returns. This API creates no pool or additional thread.
    ///
    /// Poll [`SdfRebakeTask::try_commit`] while continuing to query the old snapshot.
    /// Preparation and spawning allocate; schedule them outside frame hot paths.
    ///
    /// ```
    /// use core::convert::Infallible;
    /// use prism_tasks::{CancelToken, TaskPool};
    /// use prism_world_physics_bridge::{SdfPhysicsBridgeBudget, SdfRebakeLimits,
    ///     SdfRebakeRegion, WorldSdfPhysicsBridge, WorldSdfSnapshot, WorldSdfTransform};
    /// use prism_world_residency::SparseSdfBrickGrid;
    /// let pool = TaskPool::with_threads(1);
    /// let mut bridge = WorldSdfPhysicsBridge::new(
    ///     SdfPhysicsBridgeBudget::new(2, 16, 112).unwrap());
    /// bridge.publish(WorldSdfSnapshot::new(
    ///     WorldSdfTransform::new([0; 3], 10, 1).unwrap(), 1,
    ///     SparseSdfBrickGrid::new(2, 2).unwrap())).unwrap();
    /// let work = bridge.prepare_rebake(
    ///     SdfRebakeRegion { min_voxel: [0; 3], max_voxel: [0; 3], mip: 0 }, 2,
    ///     SdfRebakeLimits { max_bricks: 2, max_samples: 16, max_scratch_bytes: 4096 },
    /// ).unwrap();
    /// let mut task = work.spawn(&pool, &CancelToken::new(), |_| Ok::<_, Infallible>(5));
    /// while !task.is_finished() { std::thread::yield_now(); }
    /// assert_eq!(bridge.current().unwrap().revision(), 1);
    /// assert_eq!(task.try_commit(&mut bridge).unwrap().unwrap().revision, 2);
    /// ```
    ///
    /// The pool owner must outlive its handles, so pool shutdown cannot turn
    /// a task poll or drop into a worker join:
    ///
    /// ```compile_fail,E0505
    /// use core::convert::Infallible;
    /// use prism_tasks::{CancelToken, TaskPool};
    /// use prism_world_physics_bridge::SdfRebakeWork;
    /// fn invalid_shutdown(pool: TaskPool, work: SdfRebakeWork) {
    ///     let task = work.spawn(&pool, &CancelToken::new(), |_| Ok::<_, Infallible>(0));
    ///     drop(pool);
    ///     drop(task);
    /// }
    /// ```
    pub fn spawn<'pool, E: Send + 'static>(
        self,
        pool: &'pool TaskPool,
        scope: &CancelToken,
        sample: impl FnMut(SdfRebakePoint) -> Result<i16, E> + Send + 'static,
    ) -> SdfRebakeTask<'pool, E> {
        let cancel = scope.child();
        let worker_cancel = cancel.clone();
        let job = pool
            .spawn_catch(move || self.evaluate_checked(sample, || worker_cancel.is_cancelled()));
        SdfRebakeTask {
            job: Some(job),
            cancel,
            _pool: pool,
        }
    }
}

impl<E> SdfRebakeTask<'_, E> {
    /// Whether computation finished, or its result has already been collected.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.job.as_ref().is_none_or(JobHandle::is_finished)
    }

    /// Requests cancellation for this task without cancelling sibling tasks.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Collects and commits a finished result, or returns None while computing.
    ///
    /// This never waits for unfinished work. A finished job uses its result-slot
    /// mutex once; no sampling runs on this caller. Success and errors both
    /// consume the result; subsequent polls return `AlreadyCollected`.
    ///
    /// Cancellation observed at the final check prevents publication, including
    /// cancellation after computation finished. Cancellation concurrent with or
    /// after that check does not revoke a commit already beginning. Base identity
    /// and revision are checked by the bridge before its first resident write.
    pub fn try_commit(
        &mut self,
        bridge: &mut WorldSdfPhysicsBridge,
    ) -> Result<Option<SdfRebakeReport>, SdfRebakeTaskError<E>> {
        let Some(job) = &self.job else {
            return Err(SdfRebakeTaskError::AlreadyCollected);
        };
        if !job.is_finished() {
            return Ok(None);
        }
        // The completion counter is terminal, so join_catch cannot help-run a job.
        // Taking the handle also prevents Drop from cancelling collected results.
        let Some(job) = self.job.take() else {
            return Err(SdfRebakeTaskError::AlreadyCollected);
        };
        let update = job
            .join_catch()
            .map_err(SdfRebakeTaskError::Panicked)?
            .map_err(SdfRebakeTaskError::Rebake)?;
        if self.cancel.is_cancelled() {
            return Err(SdfRebakeTaskError::Rebake(SdfRebakeError::Cancelled));
        }
        bridge
            .commit_rebake(update)
            .map(Some)
            .map_err(|error| SdfRebakeTaskError::Rebake(widen(error)))
    }
}

impl<E> Drop for SdfRebakeTask<'_, E> {
    fn drop(&mut self) {
        if self.job.is_some() {
            // The child token is private to this adapter and its worker. It has
            // no external waiter/waker callbacks, and cancellation never joins.
            self.cancel.cancel();
        }
    }
}
