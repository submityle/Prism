//! Shared variable-frame scheduling budget (ECS design §24.1).
//!
//! Insert [`FrameScheduleBudget`] into a sub-app's world to consume one
//! [`FrameBudget`] across its whole update, including time spent in fixed
//! simulation and observer callbacks. Background systems in variable phases and
//! queued QoS Background tasks may defer; fixed simulation and explicit
//! `run_schedule` calls run completely. Systems resume on the next update using
//! their existing change detection window. This opt-in wall-clock policy is unsuitable for systems
//! whose execution must be identical between deterministic peers. Under
//! `multi_thread`, the frame tail drives `TaskQosResource`
//! using both wall-clock headroom and estimated Background costs. Running jobs
//! are not preempted; estimates should describe bounded chunks of work.

use prism_ecs::resource::Resource;
use prism_time::{Duration, FrameBudget};

/// QoS task activity observed during the most recent completed app frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TaskBudgetSnapshot {
    /// Committed tasks run from Critical through Low lanes.
    pub foreground_ran: usize,
    /// Background tasks admitted by the frame's remaining budget.
    pub background_ran: usize,
    /// Background tasks left queued for a later frame.
    pub background_deferred: usize,
    /// Estimated nanoseconds charged for admitted Background tasks.
    pub background_nanos_consumed: u64,
    /// Estimated task budget left after Background admission.
    pub scheduler_remaining_nanos: u64,
    /// Background tasks still queued after the frame drain completed.
    pub pending_background_tasks: usize,
}

/// Frame scheduling policy and results of the most recent sub-app update.
#[derive(Clone, Debug)]
pub struct FrameScheduleBudget {
    budget: FrameBudget,
    deferred: usize,
    elapsed: Duration,
    remaining: Duration,
    frame_index: u64,
    tasks: TaskBudgetSnapshot,
}

impl Resource for FrameScheduleBudget {}

impl FrameScheduleBudget {
    /// Share `budget` across all variable phases of each sub-app update.
    #[must_use]
    pub fn new(budget: FrameBudget) -> Self {
        Self {
            budget,
            deferred: 0,
            elapsed: Duration::ZERO,
            remaining: budget.target(),
            frame_index: 0,
            tasks: TaskBudgetSnapshot::default(),
        }
    }

    /// Current frame target. Changes take effect in the following phase.
    #[must_use]
    pub fn budget(&self) -> FrameBudget {
        self.budget
    }

    /// Adjust the target, for example after a display refresh-rate change.
    pub fn set_budget(&mut self, budget: FrameBudget) {
        self.budget = budget;
    }

    /// Background executions deferred during the most recent update.
    #[must_use]
    pub fn deferred_systems(&self) -> usize {
        self.deferred
    }

    /// Wall time of the most recent completed update, including fixed steps.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Actual wall-clock headroom left after the most recent completed update.
    #[must_use]
    pub fn remaining(&self) -> Duration {
        self.remaining
    }

    /// Number of updates completed under this policy.
    #[must_use]
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// QoS task admission and backlog from the most recent completed update.
    #[must_use]
    pub fn task_budget(&self) -> TaskBudgetSnapshot {
        self.tasks
    }

    /// Whether variable ECS work or Background tasks carried into a later frame.
    #[must_use]
    pub fn backpressured(&self) -> bool {
        self.deferred != 0 || self.tasks.pending_background_tasks != 0
    }

    pub(crate) fn begin(&mut self) {
        self.deferred = 0;
        self.tasks = TaskBudgetSnapshot::default();
    }
    pub(crate) fn record_deferred(&mut self, count: usize) {
        self.deferred += count;
    }
    #[cfg(feature = "multi_thread")]
    pub(crate) fn record_tasks(
        &mut self,
        report: prism_tasks::FrameRunReport,
        pending_background_tasks: usize,
    ) {
        self.tasks = TaskBudgetSnapshot {
            foreground_ran: report.foreground_ran,
            background_ran: report.background_ran,
            background_deferred: report.background_deferred,
            background_nanos_consumed: report.background_nanos_consumed,
            scheduler_remaining_nanos: report.remaining_nanos_after,
            pending_background_tasks,
        };
    }
    pub(crate) fn finish(&mut self, elapsed: Duration) {
        self.elapsed = elapsed;
        self.remaining = self.budget.headroom(elapsed).unwrap_or_default();
        self.frame_index = self.frame_index.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{App, First, Last, Update};
    use prism_ecs::schedule::{IntoSystemConfigs, Schedules, SystemLane};
    use prism_ecs::system::ResMut;

    #[derive(Default)]
    struct Counts {
        critical: usize,
        normal: usize,
        background: usize,
        fixed: usize,
    }
    impl Resource for Counts {}

    fn critical(mut counts: ResMut<Counts>) {
        counts.critical += 1;
    }
    fn normal(mut counts: ResMut<Counts>) {
        counts.normal += 1;
    }
    fn background(mut counts: ResMut<Counts>) {
        counts.background += 1;
    }
    fn fixed(mut counts: ResMut<Counts>) {
        counts.fixed += 1;
    }

    fn app() -> App {
        let mut app = App::new();
        app.insert_resource(Counts::default());
        app.insert_resource(FrameScheduleBudget::new(FrameBudget::new(Duration::ZERO)));
        app.add_systems(Update, critical.lane(SystemLane::Critical));
        app.add_systems(Update, normal);
        app.add_systems(Update, background.lane(SystemLane::Background));
        app
    }

    #[test]
    fn budget_defers_only_background_and_resumes_next_frame() {
        let mut app = app();
        app.update();
        let counts = app.world().resource::<Counts>();
        assert_eq!(
            (counts.critical, counts.normal, counts.background),
            (1, 1, 0)
        );
        assert_eq!(
            app.world()
                .resource::<FrameScheduleBudget>()
                .deferred_systems(),
            1
        );
        app.world_mut()
            .resource_mut::<FrameScheduleBudget>()
            .set_budget(FrameBudget::new(Duration::from_secs(3600)));
        app.update();
        let counts = app.world().resource::<Counts>();
        assert_eq!(
            (counts.critical, counts.normal, counts.background),
            (2, 2, 1)
        );
        assert_eq!(
            app.world()
                .resource::<FrameScheduleBudget>()
                .deferred_systems(),
            0
        );
    }

    #[test]
    fn budget_is_shared_across_phases_including_instrumented_updates() {
        let mut app = app();
        app.insert_resource(crate::FrameDiagnostics::default());
        app.world_mut()
            .resource_mut::<FrameScheduleBudget>()
            .set_budget(FrameBudget::new(Duration::from_millis(1)));
        app.add_systems(First, || std::thread::sleep(Duration::from_millis(4)));
        app.add_systems(Last, background.lane(SystemLane::Background));
        app.update();
        assert_eq!(app.world().resource::<Counts>().background, 0);
        let policy = app.world().resource::<FrameScheduleBudget>();
        assert_eq!(policy.deferred_systems(), 2);
        assert!(policy.elapsed() >= Duration::from_millis(4));
    }

    #[test]
    fn fixed_simulation_and_explicit_schedules_always_complete() {
        let mut app = app();
        app.set_time_update_strategy(crate::TimeUpdateStrategy::ManualDelta(
            Duration::from_millis(20),
        ));
        app.set_fixed_timestep(Duration::from_millis(10));
        app.add_systems(crate::FixedUpdate, fixed.lane(SystemLane::Background));
        app.update();
        assert_eq!(app.world().resource::<Counts>().fixed, 2);
        app.world_mut().run_schedule(Update);
        assert_eq!(app.world().resource::<Counts>().background, 1);
    }

    #[test]
    fn caught_panic_keeps_the_labeled_schedule_registered() {
        let mut app = App::new();
        app.add_systems(Update, || panic!("budget phase failure"));
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| app.update())).is_err());
        assert!(app.world().resource::<Schedules>().contains(Update));
    }

    #[cfg(feature = "multi_thread")]
    #[test]
    fn parallel_subapp_consumes_the_same_frame_policy() {
        let mut app = app();
        app.set_task_pool(prism_tasks::TaskPool::with_threads(2));
        app.update();
        let counts = app.world().resource::<Counts>();
        assert_eq!(
            (counts.critical, counts.normal, counts.background),
            (1, 1, 0)
        );
        assert_eq!(
            app.world()
                .resource::<FrameScheduleBudget>()
                .deferred_systems(),
            1
        );
    }

    #[cfg(feature = "multi_thread")]
    #[test]
    fn qos_zero_budget_keeps_foreground_and_recovers_background_fifo() {
        use prism_tasks::Priority;
        use std::sync::{Arc, Mutex};

        let mut app = App::new();
        app.set_task_pool(prism_tasks::TaskPool::with_threads(0));
        app.insert_resource(FrameScheduleBudget::new(FrameBudget::new(Duration::ZERO)));

        let observed = Arc::new(Mutex::new(Vec::new()));
        let qos = app.task_qos().expect("App installs QoS queues").clone();
        for (lane, value) in [(Priority::Normal, 20), (Priority::Critical, 10)] {
            let observed = Arc::clone(&observed);
            qos.submit_foreground(lane, move || observed.lock().unwrap().push(value));
        }
        let free_observed = Arc::clone(&observed);
        qos.submit_background(0, move || free_observed.lock().unwrap().push(0));
        for value in [1, 2] {
            let observed = Arc::clone(&observed);
            qos.submit_background(1, move || observed.lock().unwrap().push(value));
        }

        app.update();
        assert_eq!(*observed.lock().unwrap(), vec![10, 20, 0]);
        let policy = app.world().resource::<FrameScheduleBudget>();
        assert_eq!(policy.frame_index(), 1);
        assert_eq!(policy.remaining(), Duration::ZERO);
        assert!(policy.backpressured());
        assert_eq!(
            policy.task_budget(),
            TaskBudgetSnapshot {
                foreground_ran: 2,
                background_ran: 1,
                background_deferred: 2,
                background_nanos_consumed: 0,
                scheduler_remaining_nanos: 0,
                pending_background_tasks: 2,
            }
        );

        app.world_mut()
            .resource_mut::<FrameScheduleBudget>()
            .set_budget(FrameBudget::new(Duration::from_secs(3600)));
        app.update();

        assert_eq!(*observed.lock().unwrap(), vec![10, 20, 0, 1, 2]);
        let policy = app.world().resource::<FrameScheduleBudget>();
        assert_eq!(policy.frame_index(), 2);
        assert!(!policy.backpressured());
        assert_eq!(
            policy.remaining() + policy.elapsed(),
            policy.budget().target()
        );
        assert_eq!(policy.task_budget().foreground_ran, 0);
        assert_eq!(policy.task_budget().background_ran, 2);
        assert_eq!(policy.task_budget().background_deferred, 0);
        assert_eq!(policy.task_budget().background_nanos_consumed, 2);
        assert_eq!(policy.task_budget().pending_background_tasks, 0);
    }

    #[cfg(feature = "multi_thread")]
    #[test]
    fn pool_replacement_preserves_deferred_work_and_queue_handles() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let mut app = App::new();
        app.set_task_pool(prism_tasks::TaskPool::with_threads(0));
        app.insert_resource(FrameScheduleBudget::new(FrameBudget::new(Duration::ZERO)));
        let qos = app.task_qos().unwrap().clone();
        let hits = Arc::new(AtomicUsize::new(0));
        let task_hits = Arc::clone(&hits);
        qos.submit_background(1, move || {
            task_hits.fetch_add(1, Ordering::Relaxed);
        });
        app.update();
        assert_eq!(qos.pending(), 1);

        app.set_task_pool(prism_tasks::TaskPool::with_threads(0));
        let task_hits = Arc::clone(&hits);
        qos.submit_background(1, move || {
            task_hits.fetch_add(1, Ordering::Relaxed);
        });
        app.world_mut()
            .resource_mut::<FrameScheduleBudget>()
            .set_budget(FrameBudget::new(Duration::from_secs(3600)));
        app.update();

        assert_eq!(hits.load(Ordering::Relaxed), 2);
        assert!(qos.is_empty());
        assert!(app.task_qos().unwrap().is_empty());
    }

    #[cfg(feature = "multi_thread")]
    #[test]
    fn qos_without_frame_policy_drains_background_work() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let mut app = App::new();
        app.set_task_pool(prism_tasks::TaskPool::with_threads(0));
        let hits = Arc::new(AtomicUsize::new(0));
        let task_hits = Arc::clone(&hits);
        app.task_qos()
            .expect("App installs QoS queues")
            .submit_background(u64::MAX, move || {
                task_hits.fetch_add(1, Ordering::Relaxed);
            });

        app.update();

        assert_eq!(hits.load(Ordering::Relaxed), 1);
        assert!(app.task_qos().unwrap().is_empty());
    }
}
