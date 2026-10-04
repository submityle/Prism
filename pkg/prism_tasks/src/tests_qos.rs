//! §24.1 `QoS`-lane / frame-budget scheduler tests.
//!
//! Anti-vacuous contract: the deterministic lane core
//! ([`LaneQueues`](crate::qos::LaneQueues)) is checked against an *independent*
//! serial oracle (highest-lane-first, FIFO within a lane, background admitted
//! head-of-line while it fits the budget), covering lane ordering, budget
//! exhaustion / sufficiency, and the zero-budget / empty-queue boundaries. The
//! [`FrameScheduler`](crate::qos::FrameScheduler) façade is then exercised on a
//! real multi-threaded [`TaskPool`](crate::TaskPool) to prove foreground always
//! runs, background is gated, and deferred background carries to a later frame.

use alloc::vec;
use alloc::vec::Vec;
use alloc::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::qos::{admits_background, FrameStep, LanePlan, LaneQueues};
use crate::{Priority, TaskPool};

/// One submission: lane, estimated cost, and an identifying tag.
#[derive(Clone, Copy)]
struct Sub {
    lane: Priority,
    est: u64,
    tag: u32,
}

const fn sub(lane: Priority, est: u64, tag: u32) -> Sub {
    Sub { lane, est, tag }
}

/// Result of draining the production lane core.
struct Driven {
    served: Vec<(Priority, u32)>,
    remaining: u64,
    deferred_pending: usize,
}

/// Drive the real [`LaneQueues`] to completion for `remaining` headroom.
fn drive(subs: &[Sub], remaining: u64) -> Driven {
    let mut lanes: LaneQueues<u32> = LaneQueues::new();
    for s in subs {
        lanes.push(s.lane, s.est, s.tag);
    }
    let mut rem = remaining;
    let mut served = Vec::new();
    let mut deferred_pending = 0;
    loop {
        match lanes.next_step(&mut rem) {
            FrameStep::Ran { lane, payload, .. } => served.push((lane, payload)),
            FrameStep::DeferredBackground { pending, .. } => {
                deferred_pending = pending;
                break;
            }
            FrameStep::Idle => break,
        }
    }
    Driven {
        served,
        remaining: rem,
        deferred_pending,
    }
}

/// Independent oracle: foreground highest-first + FIFO within a lane, then
/// background in submission order admitted head-of-line while it fits.
fn oracle(subs: &[Sub], remaining: u64) -> Driven {
    let mut served = Vec::new();
    for lane in [
        Priority::Critical,
        Priority::High,
        Priority::Normal,
        Priority::Low,
    ] {
        for s in subs.iter().filter(|s| s.lane == lane) {
            served.push((lane, s.tag));
        }
    }
    let mut rem = remaining;
    let mut deferred = false;
    let mut deferred_pending = 0;
    for s in subs.iter().filter(|s| s.lane == Priority::Background) {
        if deferred {
            deferred_pending += 1;
            continue;
        }
        if admits_background(rem, s.est) {
            rem -= s.est;
            served.push((Priority::Background, s.tag));
        } else {
            deferred = true;
            deferred_pending += 1;
        }
    }
    Driven {
        served,
        remaining: rem,
        deferred_pending,
    }
}

fn assert_matches_oracle(subs: &[Sub], remaining: u64) -> Driven {
    let got = drive(subs, remaining);
    let want = oracle(subs, remaining);
    assert_eq!(got.served, want.served, "served sequence");
    assert_eq!(got.remaining, want.remaining, "remaining headroom");
    assert_eq!(
        got.deferred_pending, want.deferred_pending,
        "deferred background count"
    );
    got
}

#[test]
fn lanes_serve_highest_priority_first_fifo_within_lane() {
    // Interleaved submissions across every lane, with multiple items per lane
    // to pin down FIFO ordering. Huge budget so background never defers.
    let subs = [
        sub(Priority::Low, 0, 10),
        sub(Priority::Background, 1, 20),
        sub(Priority::Normal, 0, 30),
        sub(Priority::Critical, 0, 40),
        sub(Priority::Normal, 0, 31),
        sub(Priority::High, 0, 50),
        sub(Priority::Background, 1, 21),
        sub(Priority::Critical, 0, 41),
        sub(Priority::Low, 0, 11),
        sub(Priority::High, 0, 51),
    ];
    let got = assert_matches_oracle(&subs, 1_000_000);
    // Spell out the expected order explicitly as a second, hand-written oracle.
    assert_eq!(
        got.served,
        vec![
            (Priority::Critical, 40),
            (Priority::Critical, 41),
            (Priority::High, 50),
            (Priority::High, 51),
            (Priority::Normal, 30),
            (Priority::Normal, 31),
            (Priority::Low, 10),
            (Priority::Low, 11),
            (Priority::Background, 20),
            (Priority::Background, 21),
        ]
    );
    assert_eq!(got.deferred_pending, 0);
}

#[test]
fn background_defers_when_budget_exhausted() {
    // 5 ns of headroom, background items cost 3 ns each: only the first fits;
    // the rest defer head-of-line even though a later cheaper one exists.
    let subs = [
        sub(Priority::Critical, 0, 1),
        sub(Priority::Background, 3, 100),
        sub(Priority::Background, 3, 101),
        sub(Priority::Background, 1, 102),
    ];
    let got = assert_matches_oracle(&subs, 5);
    assert_eq!(
        got.served,
        vec![(Priority::Critical, 1), (Priority::Background, 100)]
    );
    assert_eq!(got.remaining, 2, "3 of 5 ns spent");
    assert_eq!(got.deferred_pending, 2, "two background items slip");
}

#[test]
fn background_admitted_when_budget_sufficient() {
    let subs = [
        sub(Priority::Normal, 0, 1),
        sub(Priority::Background, 4, 100),
        sub(Priority::Background, 4, 101),
        sub(Priority::Background, 4, 102),
    ];
    let got = assert_matches_oracle(&subs, 12);
    assert_eq!(
        got.served,
        vec![
            (Priority::Normal, 1),
            (Priority::Background, 100),
            (Priority::Background, 101),
            (Priority::Background, 102),
        ]
    );
    assert_eq!(got.remaining, 0, "all 12 ns spent");
    assert_eq!(got.deferred_pending, 0);
}

#[test]
fn zero_budget_runs_free_background_but_defers_costly() {
    // A zero-cost background item is admitted even at zero budget; the next,
    // positive-cost item defers.
    let subs = [
        sub(Priority::Background, 0, 1),
        sub(Priority::Background, 1, 2),
    ];
    let got = assert_matches_oracle(&subs, 0);
    assert_eq!(got.served, vec![(Priority::Background, 1)]);
    assert_eq!(got.remaining, 0);
    assert_eq!(got.deferred_pending, 1);
}

#[test]
fn zero_budget_does_not_block_foreground() {
    // Foreground is committed work: it runs this frame even with no background
    // headroom at all.
    let subs = [
        sub(Priority::Critical, 0, 1),
        sub(Priority::Low, 0, 2),
        sub(Priority::Background, 1, 3),
    ];
    let got = assert_matches_oracle(&subs, 0);
    assert_eq!(
        got.served,
        vec![(Priority::Critical, 1), (Priority::Low, 2)]
    );
    assert_eq!(got.deferred_pending, 1);
}

#[test]
fn empty_queue_is_idle() {
    let mut lanes: LaneQueues<u32> = LaneQueues::new();
    assert!(lanes.is_empty());
    assert_eq!(lanes.len(), 0);
    assert_eq!(lanes.peek_plan(100), LanePlan::Idle);
    let mut rem = 100;
    assert!(matches!(lanes.next_step(&mut rem), FrameStep::Idle));
    assert_eq!(rem, 100, "idle step never spends budget");
}

#[test]
fn peek_plan_predicts_next_step() {
    let mut lanes: LaneQueues<u32> = LaneQueues::new();
    lanes.push(Priority::Normal, 0, 1);
    lanes.push(Priority::Background, 10, 2);
    // Foreground pending -> plan to run Normal.
    assert_eq!(
        lanes.peek_plan(5),
        LanePlan::Run {
            lane: Priority::Normal,
            est_nanos: 0
        }
    );
    let mut rem = 5;
    assert!(matches!(
        lanes.next_step(&mut rem),
        FrameStep::Ran {
            lane: Priority::Normal,
            ..
        }
    ));
    // Now only a 10 ns background item remains with 5 ns of budget -> defer.
    assert_eq!(
        lanes.peek_plan(rem),
        LanePlan::DeferBackground {
            pending: 1,
            head_est_nanos: 10
        }
    );
}

#[test]
fn frame_scheduler_runs_foreground_and_gates_background() {
    let pool = TaskPool::with_threads(4);
    let mut sched = pool.frame_scheduler();

    let fg = Arc::new(AtomicUsize::new(0));
    let bg = Arc::new(AtomicUsize::new(0));

    for _ in 0..3 {
        let fg = Arc::clone(&fg);
        sched.submit_foreground(Priority::Normal, move || {
            fg.fetch_add(1, Ordering::Relaxed);
        });
    }
    // Three 4 ns background jobs; frame 1 only has room for two.
    for _ in 0..3 {
        let bg = Arc::clone(&bg);
        sched.submit_background(4, move || {
            bg.fetch_add(1, Ordering::Relaxed);
        });
    }
    assert_eq!(sched.pending(), 6);

    let report = sched.run_frame(9);
    assert_eq!(report.foreground_ran, 3);
    assert_eq!(report.background_ran, 2);
    assert_eq!(report.background_deferred, 1);
    assert_eq!(report.background_nanos_consumed, 8);
    assert_eq!(report.remaining_nanos_after, 1);
    assert_eq!(fg.load(Ordering::Relaxed), 3);
    assert_eq!(bg.load(Ordering::Relaxed), 2);
    assert_eq!(sched.pending(), 1, "one background job carried over");
    assert_eq!(sched.pending_in(Priority::Background), 1);

    // Frame 2: ample budget drains the carried-over background job.
    let report2 = sched.run_frame(100);
    assert_eq!(report2.foreground_ran, 0);
    assert_eq!(report2.background_ran, 1);
    assert_eq!(report2.background_deferred, 0);
    assert_eq!(bg.load(Ordering::Relaxed), 3);
    assert!(sched.is_empty());
}

#[test]
fn frame_scheduler_single_threaded_fallback_runs_inline() {
    let pool = TaskPool::with_threads(0);
    assert!(pool.is_single_threaded());
    let mut sched = pool.frame_scheduler();
    let hits = Arc::new(AtomicUsize::new(0));
    {
        let hits = Arc::clone(&hits);
        sched.submit_foreground(Priority::Critical, move || {
            hits.fetch_add(1, Ordering::Relaxed);
        });
    }
    {
        let hits = Arc::clone(&hits);
        sched.submit_background(0, move || {
            hits.fetch_add(1, Ordering::Relaxed);
        });
    }
    let report = sched.run_frame(0);
    assert_eq!(report.foreground_ran, 1);
    assert_eq!(report.background_ran, 1, "zero-cost background still runs");
    assert_eq!(hits.load(Ordering::Relaxed), 2);
    assert!(sched.is_empty());
}
