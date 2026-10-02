//! Integration tests for the interruptible scheduler: preemption across lanes,
//! budget-driven yielding and resumption, visibility-to-lane mapping, and the
//! windowed `update_budgeted` reconciliation entry point.

extern crate alloc;

use alloc::rc::Rc;
use core::cell::RefCell;

use prism_ui_scheduler::{
    update_budgeted, Deadline, FrameBudget, Lane, LaneMask, ManualClock, ReconcileBatch, Scheduler,
    StepOutcome, StopReason, VisibilityWindow, Work,
};

/// A unit that appends its tag to a shared log on each step and finishes after
/// `steps` steps, letting tests observe interleaving order.
struct Logger {
    tag: &'static str,
    steps: u32,
    log: Rc<RefCell<Vec<&'static str>>>,
}

impl Work for Logger {
    fn step(&mut self) -> StepOutcome {
        self.log.borrow_mut().push(self.tag);
        self.steps -= 1;
        if self.steps == 0 {
            StepOutcome::Done
        } else {
            StepOutcome::More
        }
    }
}

/// A clock that advances a fixed amount every time it is read, so a run loop
/// consulting it between steps marches deterministically toward a deadline.
struct TickClock {
    cursor: core::cell::Cell<u64>,
    per_read: u64,
}

impl prism_ui_scheduler::Clock for TickClock {
    fn now_micros(&self) -> u64 {
        let now = self.cursor.get();
        self.cursor.set(now + self.per_read);
        now
    }
}

#[test]
fn higher_lane_drains_before_lower() {
    let clock = ManualClock::new(0);
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut sched = Scheduler::new();
    sched.schedule(
        Lane::Offscreen,
        Logger {
            tag: "off",
            steps: 1,
            log: log.clone(),
        },
    );
    sched.schedule(
        Lane::Visible,
        Logger {
            tag: "vis",
            steps: 1,
            log: log.clone(),
        },
    );
    sched.schedule(
        Lane::Input,
        Logger {
            tag: "in",
            steps: 1,
            log: log.clone(),
        },
    );
    let report = sched.run(&clock, Deadline::NEVER);
    assert_eq!(report.stop, StopReason::Drained);
    assert_eq!(report.completed, 3);
    assert_eq!(&*log.borrow(), &["in", "vis", "off"]);
}

#[test]
fn mid_run_high_lane_preempts_resuming_low_lane() {
    let clock = ManualClock::new(0);
    let log = Rc::new(RefCell::new(Vec::new()));
    let log2 = log.clone();
    sched_two_step(&clock, &log, log2);
}

fn sched_two_step(
    clock: &ManualClock,
    log: &Rc<RefCell<Vec<&'static str>>>,
    log2: Rc<RefCell<Vec<&'static str>>>,
) {
    let mut sched = Scheduler::new();
    sched.schedule(
        Lane::Visible,
        prism_ui_scheduler::FnWork::new({
            let mut remaining = 2u32;
            move || {
                log2.borrow_mut().push("vis");
                remaining -= 1;
                if remaining == 0 {
                    StepOutcome::Done
                } else {
                    StepOutcome::More
                }
            }
        }),
    );
    sched.schedule(
        Lane::Input,
        Logger {
            tag: "in",
            steps: 1,
            log: log.clone(),
        },
    );
    sched.run(clock, Deadline::NEVER);
    assert_eq!(&*log.borrow(), &["in", "vis", "vis"]);
}

#[test]
fn deadline_yields_then_resumes_next_frame() {
    let clock = TickClock {
        cursor: core::cell::Cell::new(0),
        per_read: 3,
    };
    let mut sched = Scheduler::new();
    let log = Rc::new(RefCell::new(Vec::new()));
    for _ in 0..10 {
        sched.schedule(
            Lane::Visible,
            Logger {
                tag: "x",
                steps: 1,
                log: log.clone(),
            },
        );
    }
    let budget = FrameBudget::from_micros(8);
    let first = sched.run(&clock, budget.deadline_from(0));
    assert_eq!(first.stop, StopReason::YieldedToDeadline);
    assert!(first.completed < 10);
    assert!(!sched.is_idle());
    let second = sched.run_to_completion(&clock);
    assert_eq!(second.stop, StopReason::Drained);
    assert_eq!(first.completed + second.completed, 10);
    assert!(sched.is_idle());
}

#[test]
fn visibility_window_assigns_lanes() {
    let window = VisibilityWindow::new(10..20, 5);
    assert_eq!(window.lane_for(15), Lane::Visible);
    assert_eq!(window.lane_for(9), Lane::Offscreen);
    assert_eq!(window.lane_for(23), Lane::Offscreen);
    assert_eq!(window.lane_for(2), Lane::Idle);
    assert_eq!(window.lane_for(40), Lane::Idle);
    assert_eq!(window.warm_range(), 5..25);
}

#[test]
fn reconcile_batch_builds_visible_rows_first() {
    let clock = ManualClock::new(0);
    let built = Rc::new(RefCell::new(Vec::new()));
    let window = VisibilityWindow::new(3..6, 1);
    let batch = ReconcileBatch::new(window, 10);
    let built2 = built.clone();
    let mut sched = batch.into_scheduler(move |index| built2.borrow_mut().push(index));
    let report = update_budgeted(&mut sched, &clock, Deadline::NEVER);
    assert_eq!(report.completed, 10);
    let order = built.borrow();
    assert_eq!(&order[0..3], &[3, 4, 5]);
    assert!(order[3] == 2 || order[3] == 6);
    assert!(order[4] == 2 || order[4] == 6);
}

#[test]
fn cancel_lane_discards_pending_units() {
    let clock = ManualClock::new(0);
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut sched = Scheduler::new();
    sched.schedule(
        Lane::Idle,
        Logger {
            tag: "a",
            steps: 1,
            log: log.clone(),
        },
    );
    sched.schedule(
        Lane::Idle,
        Logger {
            tag: "b",
            steps: 1,
            log: log.clone(),
        },
    );
    sched.schedule(
        Lane::Visible,
        Logger {
            tag: "v",
            steps: 1,
            log: log.clone(),
        },
    );
    sched.cancel_lane(Lane::Idle);
    assert_eq!(sched.lane_len(Lane::Idle), 0);
    sched.run_to_completion(&clock);
    assert_eq!(&*log.borrow(), &["v"]);
}

#[test]
fn pending_lanes_reflects_queue_state() {
    let mut sched = Scheduler::new();
    assert!(sched.pending_lanes().is_empty());
    sched.schedule(Lane::Animation, prism_ui_scheduler::OnceWork::new(|| {}));
    let mut expected = LaneMask::new();
    expected.insert(Lane::Animation);
    assert_eq!(sched.pending_lanes(), expected);
    assert_eq!(sched.pending_lanes().highest(), Some(Lane::Animation));
}
