//! Shared task-pool and frame-budgeted QoS wiring for `multi_thread`.

use prism_ecs::resource::Resource;
use prism_tasks::{FrameRunReport, FrameScheduler, Priority, TaskPool};
use std::sync::{Arc, Mutex, MutexGuard};

/// The `prism_tasks` pool a sub-app's schedules dispatch onto when the
/// `multi_thread` feature is on (design §8.2 conflict-graph executor).
///
/// Stored as a world resource so
/// [`SubApp::run_schedule`](crate::sub_app::SubApp::run_schedule) can fetch a
/// cheap clone and hand it to
/// [`World::run_schedule_parallel`](prism_ecs::world::World::run_schedule_parallel);
/// a world without it runs its schedules single-threaded. The pool is wrapped in
/// a newtype because the orphan rule forbids implementing the resource marker on
/// `TaskPool` here.
#[derive(Clone)]
pub struct TaskPoolResource(pub TaskPool);

impl Resource for TaskPoolResource {}

/// Persistent `prism_tasks` QoS queues driven once per app frame.
///
/// Systems may clone this resource and submit committed foreground work or
/// estimated-cost background work. [`SubApp`](crate::sub_app::SubApp) drains
/// every foreground lane at the end of the frame, then admits Background jobs
/// in FIFO order while [`FrameScheduleBudget`](crate::budget::FrameScheduleBudget)
/// still has headroom. Jobs that do not fit remain in the scheduler for the
/// next frame.
///
/// The scheduler is mutex-protected only while queue state is inspected or
/// changed. Admitted jobs execute after the lock is released, so a running job
/// may safely submit follow-up work to this same resource.
#[derive(Clone)]
pub struct TaskQosResource {
    scheduler: Arc<Mutex<FrameScheduler>>,
}

impl Resource for TaskQosResource {}

impl TaskQosResource {
    /// Create persistent QoS queues dispatching onto `pool`.
    #[must_use]
    pub fn new(pool: TaskPool) -> Self {
        Self {
            scheduler: Arc::new(Mutex::new(pool.frame_scheduler())),
        }
    }

    fn lock(&self) -> MutexGuard<'_, FrameScheduler> {
        self.scheduler
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn set_pool(&self, pool: TaskPool) {
        self.lock().set_pool(pool);
    }

    /// Queue a task on an explicit QoS lane.
    ///
    /// `est_nanos` gates only [`Priority::Background`]; committed lanes ignore
    /// the estimate and always run in the next frame drain.
    pub fn submit<F>(&self, lane: Priority, est_nanos: u64, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.lock().submit(lane, est_nanos, task);
    }

    /// Queue committed work on a foreground lane.
    ///
    /// # Panics
    ///
    /// Panics if `lane` is [`Priority::Background`]. Use
    /// [`submit_background`](Self::submit_background) for deferrable work.
    pub fn submit_foreground<F>(&self, lane: Priority, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.lock().submit_foreground(lane, task);
    }

    /// Queue deferrable Background work with an estimated cost in nanoseconds.
    pub fn submit_background<F>(&self, est_nanos: u64, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.lock().submit_background(est_nanos, task);
    }

    /// Number of tasks waiting across all QoS lanes.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.lock().pending()
    }

    /// Number of tasks waiting on `lane`.
    #[must_use]
    pub fn pending_in(&self, lane: Priority) -> usize {
        self.lock().pending_in(lane)
    }

    /// Whether every QoS lane is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Drain one app frame without holding the scheduler lock while jobs run.
    ///
    /// `headroom` is rechecked after each foreground batch and Background job.
    /// `None` means no frame policy is installed; all work drains without an
    /// estimated-cost cap. Otherwise both wall-clock headroom and the unspent
    /// estimated budget must admit each Background job. A running job cannot be
    /// preempted, so jobs should submit bounded chunks with realistic estimates.
    /// Follow-up submissions are observed without holding the lock during work.
    pub(crate) fn run_frame(&self, mut headroom: impl FnMut() -> Option<u64>) -> FrameRunReport {
        let mut total = FrameRunReport::default();
        let mut estimated_remaining = u64::MAX;

        loop {
            let remaining_nanos = headroom()
                .map(|wall_clock| wall_clock.min(estimated_remaining))
                .unwrap_or(u64::MAX);
            let batch = {
                let mut scheduler = self.lock();
                let foreground = scheduler.prepare_foreground();
                if foreground.is_empty() {
                    scheduler.prepare_next(remaining_nanos)
                } else {
                    foreground
                }
            };
            let admitted = batch.report();
            if batch.is_empty() {
                total.background_deferred = admitted.background_deferred;
                total.remaining_nanos_after = remaining_nanos;
                return total;
            }

            let report = batch.run();
            total.foreground_ran = total.foreground_ran.saturating_add(report.foreground_ran);
            total.background_ran = total.background_ran.saturating_add(report.background_ran);
            total.background_nanos_consumed = total
                .background_nanos_consumed
                .saturating_add(report.background_nanos_consumed);
            estimated_remaining = remaining_nanos.saturating_sub(report.background_nanos_consumed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[test]
    fn frame_drain_releases_scheduler_lock_before_running_follow_up_work() {
        let qos = TaskQosResource::new(TaskPool::with_threads(0));
        let hits = Arc::new(AtomicUsize::new(0));
        let submitter = qos.clone();
        let first_hits = Arc::clone(&hits);
        qos.submit_foreground(Priority::Critical, move || {
            first_hits.fetch_add(1, Ordering::Relaxed);
            let follow_up_hits = Arc::clone(&first_hits);
            submitter.submit_foreground(Priority::Normal, move || {
                follow_up_hits.fetch_add(1, Ordering::Relaxed);
            });
        });

        let report = qos.run_frame(|| Some(0));

        assert_eq!(report.foreground_ran, 2);
        assert_eq!(report.background_ran, 0);
        assert_eq!(hits.load(Ordering::Relaxed), 2);
        assert!(qos.is_empty());
    }

    #[test]
    fn background_carries_over_fifo_without_bypassing_an_expensive_head() {
        let qos = TaskQosResource::new(TaskPool::with_threads(0));
        let observed = Arc::new(Mutex::new(Vec::new()));
        for (cost, value) in [(5, 1), (1, 2), (0, 3)] {
            let observed = Arc::clone(&observed);
            qos.submit_background(cost, move || observed.lock().unwrap().push(value));
        }

        let first = qos.run_frame(|| Some(1));
        assert_eq!(first.background_ran, 0);
        assert_eq!(first.background_deferred, 3);
        assert_eq!(first.remaining_nanos_after, 1);
        assert!(observed.lock().unwrap().is_empty());

        let second = qos.run_frame(|| Some(5));
        assert_eq!(second.background_ran, 1);
        assert_eq!(second.background_deferred, 2);
        assert_eq!(second.background_nanos_consumed, 5);
        assert_eq!(second.remaining_nanos_after, 0);
        assert_eq!(*observed.lock().unwrap(), vec![1]);

        let third = qos.run_frame(|| Some(1));
        assert_eq!(third.background_ran, 2);
        assert_eq!(third.background_deferred, 0);
        assert_eq!(third.remaining_nanos_after, 0);
        assert_eq!(*observed.lock().unwrap(), vec![1, 2, 3]);
        assert!(qos.is_empty());
    }

    #[test]
    fn background_rechecks_headroom_after_foreground_and_each_job() {
        for foreground_exhausts_budget in [true, false] {
            let qos = TaskQosResource::new(TaskPool::with_threads(0));
            let available = Arc::new(AtomicUsize::new(10));
            let task_available = Arc::clone(&available);
            qos.submit_foreground(Priority::Critical, move || {
                if foreground_exhausts_budget {
                    task_available.store(0, Ordering::Relaxed);
                }
            });
            let task_available = Arc::clone(&available);
            qos.submit_background(1, move || {
                task_available.store(0, Ordering::Relaxed);
            });
            qos.submit_background(1, || {});

            let report = qos.run_frame(|| Some(available.load(Ordering::Relaxed) as u64));
            assert_eq!(report.foreground_ran, 1);
            let expected_background = usize::from(!foreground_exhausts_budget);
            assert_eq!(report.background_ran, expected_background);
            assert_eq!(report.background_deferred, 2 - expected_background);
            assert_eq!(report.remaining_nanos_after, 0);
        }
    }

    #[test]
    fn background_follow_up_does_not_escape_estimated_frame_limit() {
        let qos = TaskQosResource::new(TaskPool::with_threads(0));
        let submitter = qos.clone();
        let hits = Arc::new(AtomicUsize::new(0));
        let task_hits = Arc::clone(&hits);
        qos.submit_background(5, move || {
            task_hits.fetch_add(1, Ordering::Relaxed);
            submitter.submit_background(1, move || {
                task_hits.fetch_add(1, Ordering::Relaxed);
            });
        });

        let first = qos.run_frame(|| Some(5));
        assert_eq!(first.background_ran, 1);
        assert_eq!(first.background_deferred, 1);
        assert_eq!(hits.load(Ordering::Relaxed), 1);

        let second = qos.run_frame(|| Some(1));
        assert_eq!(second.background_ran, 1);
        assert_eq!(second.background_deferred, 0);
        assert_eq!(hits.load(Ordering::Relaxed), 2);
    }
}
