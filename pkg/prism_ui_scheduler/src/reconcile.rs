//! Budgeted reconciliation of a windowed item list.
//!
//! This ties the pieces together into the §9.5 entry point: given a
//! [`VisibilityWindow`] and a per-index build step, [`ReconcileBatch`] enqueues
//! one unit per index onto the lane its visibility earns, then
//! [`update_budgeted`] drains them by urgency under a frame [`Deadline`].
//! Visible rows land first, the overscan band warms next, and far-off-screen
//! rows are only touched when the frame still has slack — producing a jank-free
//! incremental build out of ordinary closures.

use alloc::boxed::Box;

use crate::budget::{Clock, Deadline};
use crate::scheduler::{RunReport, Scheduler};
use crate::visibility::VisibilityWindow;
use crate::work::{StepOutcome, Work};

/// Builds a scheduler pre-loaded with one build unit per windowed index.
///
/// Each index in `0..count` is assigned a lane by
/// [`VisibilityWindow::lane_for`], so the resulting [`Scheduler`] drains
/// visible rows before overscan rows before idle rows.
#[derive(Debug)]
pub struct ReconcileBatch {
    window: VisibilityWindow,
    count: usize,
}

impl ReconcileBatch {
    /// Prepares a batch over `count` items partitioned by `window`.
    #[must_use]
    pub const fn new(window: VisibilityWindow, count: usize) -> ReconcileBatch {
        ReconcileBatch { window, count }
    }

    /// Enqueues one `build(index)` unit per index onto a fresh [`Scheduler`].
    ///
    /// `build` runs once per index when its unit is first stepped; the index
    /// order within a lane follows ascending index, so visible rows build
    /// top-to-bottom.
    #[must_use]
    pub fn into_scheduler<F>(self, build: F) -> Scheduler
    where
        F: FnMut(usize) + Clone + 'static,
    {
        let mut scheduler = Scheduler::new();
        for index in 0..self.count {
            let lane = self.window.lane_for(index);
            let mut build = build.clone();
            scheduler.enqueue(
                lane,
                Box::new(IndexBuild {
                    build: Some(move || build(index)),
                }),
            );
        }
        scheduler
    }
}

/// A one-shot build of a single windowed index.
struct IndexBuild<F> {
    build: Option<F>,
}

impl<F> Work for IndexBuild<F>
where
    F: FnOnce(),
{
    fn step(&mut self) -> StepOutcome {
        if let Some(build) = self.build.take() {
            build();
        }
        StepOutcome::Done
    }
}

/// Drives `scheduler` under `deadline`, returning the slice summary.
///
/// A thin, intention-revealing wrapper over [`Scheduler::run`] that names the
/// §9.5 operation ("update as much of the tree as the budget allows, then
/// yield"). Call it once per frame, re-using the same `scheduler` so unfinished
/// lower-lane work resumes next frame.
pub fn update_budgeted(
    scheduler: &mut Scheduler,
    clock: &impl Clock,
    deadline: Deadline,
) -> RunReport {
    scheduler.run(clock, deadline)
}
