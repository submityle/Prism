//! §24.2 thread-class separation (compute / I-O / main-thread) tests.
//!
//! Anti-vacuous contract: the deterministic routing core
//! ([`ClassRouter`](crate::ClassRouter)) is checked against an *independent*
//! serial oracle (compute-first, then I/O, then main; FIFO within a class;
//! compute / I/O admitted up to their per-wave budget) covering the routing
//! table, class isolation, the zero / exact / over-budget boundaries, and
//! deferral carry-over. The [`ThreadClassPool`](crate::ThreadClassPool) façade
//! is then exercised on a real multi-threaded [`TaskPool`](crate::TaskPool) +
//! [`NamedThreads`](crate::NamedThreads) to prove compute runs on the pool, I/O
//! runs off-pool, main-thread work waits for the pump, and the single-threaded
//! fallback runs compute / I/O inline in router order.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::{
    admits, route, ClassRouter, ExecLane, LaneBudget, NamedThreads, RoutePlan, RouteStep, TaskPool,
    ThreadClassPool, WorkClass,
};

/// One submission: a class and an identifying tag.
#[derive(Clone, Copy)]
struct Sub {
    class: WorkClass,
    tag: u32,
}

const fn sub(class: WorkClass, tag: u32) -> Sub {
    Sub { class, tag }
}

/// Result of a full router drain: the dispatch sequence plus the deferred
/// compute / I/O counts reported at `Idle`.
struct Drained {
    seq: Vec<(ExecLane, u32)>,
    compute_deferred: usize,
    io_deferred: usize,
}

/// Drive the real [`ClassRouter`] to completion under `budget`.
fn drive(subs: &[Sub], budget: LaneBudget) -> Drained {
    let mut router: ClassRouter<u32> = ClassRouter::new();
    for s in subs {
        router.push(s.class, s.tag);
    }
    let mut budget = budget;
    let mut seq = Vec::new();
    loop {
        match router.next_step(&mut budget) {
            RouteStep::Dispatch { lane, payload, .. } => seq.push((lane, payload)),
            RouteStep::Idle {
                compute_deferred,
                io_deferred,
            } => {
                return Drained {
                    seq,
                    compute_deferred,
                    io_deferred,
                }
            }
        }
    }
}

/// Independent oracle: compute-first then I/O then main, FIFO within a class,
/// compute / I/O admitted up to the per-wave budget.
fn oracle(subs: &[Sub], budget: LaneBudget) -> Drained {
    let tags = |c: WorkClass| -> Vec<u32> {
        subs.iter()
            .filter(|s| s.class == c)
            .map(|s| s.tag)
            .collect()
    };
    let compute = tags(WorkClass::Compute);
    let io = tags(WorkClass::Io);
    let main = tags(WorkClass::MainThread);
    let ca = compute.len().min(budget.compute_slots);
    let ia = io.len().min(budget.io_slots);
    let mut seq = Vec::new();
    for &t in &compute[..ca] {
        seq.push((ExecLane::ComputePool, t));
    }
    for &t in &io[..ia] {
        seq.push((ExecLane::IoPool, t));
    }
    for &t in &main {
        seq.push((ExecLane::MainQueue, t));
    }
    Drained {
        seq,
        compute_deferred: compute.len() - ca,
        io_deferred: io.len() - ia,
    }
}

fn assert_matches_oracle(subs: &[Sub], budget: LaneBudget) {
    let got = drive(subs, budget);
    let want = oracle(subs, budget);
    assert_eq!(got.seq, want.seq, "dispatch sequence");
    assert_eq!(
        got.compute_deferred, want.compute_deferred,
        "compute deferred"
    );
    assert_eq!(got.io_deferred, want.io_deferred, "io deferred");
}

// ----------------------------------------------------------------------------
// Pure routing core (deterministic, oracle-checked).
// ----------------------------------------------------------------------------

#[test]
fn routing_table_is_total_and_fixed() {
    assert_eq!(route(WorkClass::Compute), ExecLane::ComputePool);
    assert_eq!(route(WorkClass::Io), ExecLane::IoPool);
    assert_eq!(route(WorkClass::MainThread), ExecLane::MainQueue);
    // The convenience wrapper agrees, and indices are the stable lane order.
    assert_eq!(WorkClass::Compute.exec_lane(), ExecLane::ComputePool);
    assert_eq!(WorkClass::Io.exec_lane(), ExecLane::IoPool);
    assert_eq!(WorkClass::MainThread.exec_lane(), ExecLane::MainQueue);
    assert_eq!(WorkClass::Compute.index(), 0);
    assert_eq!(WorkClass::Io.index(), 1);
    assert_eq!(WorkClass::MainThread.index(), 2);
}

#[test]
fn unbounded_drain_routes_every_class_fifo() {
    let subs = [
        sub(WorkClass::Io, 10),
        sub(WorkClass::Compute, 1),
        sub(WorkClass::MainThread, 100),
        sub(WorkClass::Compute, 2),
        sub(WorkClass::Io, 11),
        sub(WorkClass::MainThread, 101),
        sub(WorkClass::Compute, 3),
    ];
    assert_matches_oracle(&subs, LaneBudget::unbounded());
    // Spelled out: compute (FIFO), then I/O (FIFO), then main (FIFO).
    let got = drive(&subs, LaneBudget::unbounded());
    assert_eq!(
        got.seq,
        vec![
            (ExecLane::ComputePool, 1),
            (ExecLane::ComputePool, 2),
            (ExecLane::ComputePool, 3),
            (ExecLane::IoPool, 10),
            (ExecLane::IoPool, 11),
            (ExecLane::MainQueue, 100),
            (ExecLane::MainQueue, 101),
        ]
    );
}

#[test]
fn io_saturation_does_not_starve_compute_or_main() {
    // io_slots = 0 models a fully-saturated / stalled I/O lane. Compute and
    // main must dispatch in full regardless; every I/O job is deferred.
    let subs = [
        sub(WorkClass::Io, 10),
        sub(WorkClass::Compute, 1),
        sub(WorkClass::Io, 11),
        sub(WorkClass::MainThread, 100),
        sub(WorkClass::Io, 12),
        sub(WorkClass::Compute, 2),
    ];
    let budget = LaneBudget::new(usize::MAX, 0);
    assert_matches_oracle(&subs, budget);
    let got = drive(&subs, budget);
    assert_eq!(
        got.seq,
        vec![
            (ExecLane::ComputePool, 1),
            (ExecLane::ComputePool, 2),
            (ExecLane::MainQueue, 100),
        ]
    );
    assert_eq!(got.io_deferred, 3);
    assert_eq!(got.compute_deferred, 0);
}

#[test]
fn compute_cap_defers_the_overflow() {
    // Exactly-cap and over-cap compute boundaries, with I/O unaffected.
    let subs = [
        sub(WorkClass::Compute, 1),
        sub(WorkClass::Compute, 2),
        sub(WorkClass::Compute, 3),
        sub(WorkClass::Io, 10),
    ];
    // Exact cap: all three admitted, none deferred.
    assert_matches_oracle(&subs, LaneBudget::new(3, usize::MAX));
    // Under cap: one deferred.
    let budget = LaneBudget::new(2, usize::MAX);
    assert_matches_oracle(&subs, budget);
    let got = drive(&subs, budget);
    assert_eq!(got.compute_deferred, 1);
    assert_eq!(got.io_deferred, 0);
    assert_eq!(
        got.seq,
        vec![
            (ExecLane::ComputePool, 1),
            (ExecLane::ComputePool, 2),
            (ExecLane::IoPool, 10),
        ]
    );
}

#[test]
fn empty_router_is_idle() {
    let got = drive(&[], LaneBudget::unbounded());
    assert!(got.seq.is_empty());
    assert_eq!(got.compute_deferred, 0);
    assert_eq!(got.io_deferred, 0);

    let mut router: ClassRouter<u32> = ClassRouter::new();
    assert!(router.is_empty());
    let mut budget = LaneBudget::unbounded();
    assert!(matches!(
        router.next_step(&mut budget),
        RouteStep::Idle {
            compute_deferred: 0,
            io_deferred: 0
        }
    ));
}

#[test]
fn admits_respects_slots_and_main_is_unbounded() {
    let budget = LaneBudget::new(1, 0);
    assert!(admits(&budget, WorkClass::Compute));
    assert!(!admits(&budget, WorkClass::Io));
    assert!(admits(&budget, WorkClass::MainThread));
    // Main is always admissible even at a zero compute/io budget.
    let zero = LaneBudget::new(0, 0);
    assert!(admits(&zero, WorkClass::MainThread));
    assert!(!admits(&zero, WorkClass::Compute));
}

#[test]
fn peek_plan_predicts_next_step() {
    let mut router: ClassRouter<u32> = ClassRouter::new();
    router.push(WorkClass::Io, 10);
    router.push(WorkClass::Compute, 1);
    router.push(WorkClass::MainThread, 100);

    // With compute available, peek and step both pick compute first.
    let mut budget = LaneBudget::unbounded();
    assert_eq!(
        router.peek_plan(&budget),
        RoutePlan::Dispatch {
            class: WorkClass::Compute,
            lane: ExecLane::ComputePool,
        }
    );
    // Step through and confirm peek matches each decision until idle.
    loop {
        let plan = router.peek_plan(&budget);
        match router.next_step(&mut budget) {
            RouteStep::Dispatch { class, lane, .. } => {
                assert_eq!(plan, RoutePlan::Dispatch { class, lane });
            }
            RouteStep::Idle {
                compute_deferred,
                io_deferred,
            } => {
                assert_eq!(
                    plan,
                    RoutePlan::Idle {
                        compute_deferred,
                        io_deferred
                    }
                );
                break;
            }
        }
    }
}

#[test]
fn deferred_work_carries_over_to_the_next_wave() {
    // A capped wave defers overflow; a later unbounded wave drains it, FIFO.
    let mut router: ClassRouter<u32> = ClassRouter::new();
    for tag in [1u32, 2, 3, 4] {
        router.push(WorkClass::Compute, tag);
    }

    let mut wave1 = LaneBudget::new(2, usize::MAX);
    let mut first = Vec::new();
    loop {
        match router.next_step(&mut wave1) {
            RouteStep::Dispatch { payload, .. } => first.push(payload),
            RouteStep::Idle {
                compute_deferred, ..
            } => {
                assert_eq!(compute_deferred, 2);
                break;
            }
        }
    }
    assert_eq!(first, vec![1, 2]);

    let mut wave2 = LaneBudget::unbounded();
    let mut second = Vec::new();
    while let RouteStep::Dispatch { payload, .. } = router.next_step(&mut wave2) {
        second.push(payload);
    }
    assert_eq!(second, vec![3, 4]);
    assert!(router.is_empty());
}

// ----------------------------------------------------------------------------
// Threaded façade over a real pool + named lanes.
// ----------------------------------------------------------------------------

#[test]
fn facade_routes_to_pool_io_and_main_pump() {
    let pool = TaskPool::with_threads(4);
    let named = NamedThreads::new();
    let mut tc = pool.thread_class_pool(&named);
    assert!(!tc.is_inline());

    let compute_ran = Arc::new(AtomicUsize::new(0));
    let io_ran = Arc::new(AtomicUsize::new(0));
    let main_order: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));

    for _ in 0..8 {
        let c = Arc::clone(&compute_ran);
        tc.submit_compute(move || {
            c.fetch_add(1, Ordering::Relaxed);
        });
    }
    for _ in 0..5 {
        let i = Arc::clone(&io_ran);
        tc.submit_io(move || {
            i.fetch_add(1, Ordering::Relaxed);
        });
    }
    for tag in [100u32, 101, 102] {
        let m = Arc::clone(&main_order);
        tc.submit_main(move || m.lock().unwrap().push(tag));
    }

    let report = tc.dispatch();
    assert_eq!(report.compute_dispatched, 8);
    assert_eq!(report.io_dispatched, 5);
    assert_eq!(report.main_queued, 3);
    assert_eq!(report.deferred(), 0);

    tc.wait();
    assert_eq!(compute_ran.load(Ordering::Relaxed), 8);
    assert_eq!(io_ran.load(Ordering::Relaxed), 5);
    // Main-thread work must NOT run until the main thread pumps it.
    assert!(main_order.lock().unwrap().is_empty());

    let pumped = tc.pump_main();
    assert_eq!(pumped, 3);
    assert_eq!(*main_order.lock().unwrap(), vec![100, 101, 102]);
    assert!(tc.is_empty());
}

#[test]
fn facade_isolation_defers_io_but_runs_compute() {
    // io_slots = 0: a saturated I/O lane must not block compute from running.
    let pool = TaskPool::with_threads(3);
    let named = NamedThreads::new();
    let mut tc = ThreadClassPool::with_budget(pool, named, LaneBudget::new(usize::MAX, 0));

    let compute_ran = Arc::new(AtomicUsize::new(0));
    let io_ran = Arc::new(AtomicUsize::new(0));
    for _ in 0..6 {
        let c = Arc::clone(&compute_ran);
        tc.submit_compute(move || {
            c.fetch_add(1, Ordering::Relaxed);
        });
    }
    for _ in 0..4 {
        let i = Arc::clone(&io_ran);
        tc.submit_io(move || {
            i.fetch_add(1, Ordering::Relaxed);
        });
    }

    let report = tc.dispatch_blocking();
    assert_eq!(report.compute_dispatched, 6);
    assert_eq!(report.io_dispatched, 0);
    assert_eq!(report.io_deferred, 4);
    assert_eq!(compute_ran.load(Ordering::Relaxed), 6);
    assert_eq!(io_ran.load(Ordering::Relaxed), 0);

    // Lift the budget: the deferred I/O now drains.
    tc.set_budget(LaneBudget::unbounded());
    let report = tc.dispatch_blocking();
    assert_eq!(report.io_dispatched, 4);
    assert_eq!(io_ran.load(Ordering::Relaxed), 4);
    assert!(tc.is_empty());
}

#[test]
fn single_threaded_fallback_runs_compute_io_inline() {
    // A single-threaded pool yields the inline fallback: compute + I/O run
    // inline on dispatch in router-drain order, main waits for the pump.
    let pool = TaskPool::with_threads(0);
    let named = NamedThreads::new();
    let mut tc = pool.thread_class_pool(&named);
    assert!(tc.is_inline());

    let order: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
    let main_order: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));

    // Interleave submissions; compute drains before I/O before main.
    let o = Arc::clone(&order);
    tc.submit_io(move || o.lock().unwrap().push(10));
    let o = Arc::clone(&order);
    tc.submit_compute(move || o.lock().unwrap().push(1));
    let o = Arc::clone(&order);
    tc.submit_io(move || o.lock().unwrap().push(11));
    let o = Arc::clone(&order);
    tc.submit_compute(move || o.lock().unwrap().push(2));
    let m = Arc::clone(&main_order);
    tc.submit_main(move || m.lock().unwrap().push(100));

    let report = tc.dispatch();
    assert_eq!(report.compute_dispatched, 2);
    assert_eq!(report.io_dispatched, 2);
    assert_eq!(report.main_queued, 1);
    // Inline execution already happened for compute + I/O, in router order.
    assert_eq!(*order.lock().unwrap(), vec![1, 2, 10, 11]);
    // wait() is a no-op in the fallback and must not hang.
    tc.wait();
    assert!(main_order.lock().unwrap().is_empty());

    let pumped = tc.pump_main();
    assert_eq!(pumped, 1);
    assert_eq!(*main_order.lock().unwrap(), vec![100]);
}

#[test]
fn facade_budget_caps_per_wave_dispatch() {
    // The façade honours a per-wave compute cap and carries the overflow.
    let pool = TaskPool::with_threads(2);
    let named = NamedThreads::new();
    let mut tc = ThreadClassPool::with_budget(pool, named, LaneBudget::new(3, usize::MAX));

    let compute_ran = Arc::new(AtomicUsize::new(0));
    for _ in 0..7 {
        let c = Arc::clone(&compute_ran);
        tc.submit_compute(move || {
            c.fetch_add(1, Ordering::Relaxed);
        });
    }

    let r1 = tc.dispatch_blocking();
    assert_eq!(r1.compute_dispatched, 3);
    assert_eq!(r1.compute_deferred, 4);
    assert_eq!(compute_ran.load(Ordering::Relaxed), 3);

    let r2 = tc.dispatch_blocking();
    assert_eq!(r2.compute_dispatched, 3);
    assert_eq!(r2.compute_deferred, 1);

    let r3 = tc.dispatch_blocking();
    assert_eq!(r3.compute_dispatched, 1);
    assert_eq!(r3.compute_deferred, 0);
    assert_eq!(compute_ran.load(Ordering::Relaxed), 7);
    assert!(tc.is_empty());
}
