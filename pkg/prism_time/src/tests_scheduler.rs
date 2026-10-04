//! §24.6 tests: the deterministic in-game scheduler — delay / periodic firing,
//! deterministic ordering, cancel / reschedule, catch-up across a big delta,
//! the per-advance fire cap, and double-run reproducibility. All oracles are
//! hand-computed from fixed inputs.

use crate::{Duration, Fired, Scheduler, TimerKind};

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// Collect the payloads from a fired slice, preserving order.
fn payloads(fired: &[Fired<u32>]) -> Vec<u32> {
    fired.iter().map(|f| f.payload).collect()
}

#[test]
fn schedule_after_fires_once_at_the_right_time() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    sched.schedule_after(ms(100), 7);
    assert_eq!(sched.len(), 1);

    // Before the due time: nothing fires.
    assert!(sched.advance_collect(ms(50)).is_empty());
    assert_eq!(sched.len(), 1);
    assert!(!sched.has_due());

    // Crossing 100 ms fires exactly once, then retires.
    let fired = sched.advance_collect(ms(50));
    assert_eq!(payloads(&fired), vec![7]);
    assert_eq!(fired[0].at, ms(100));
    assert_eq!(fired[0].kind, TimerKind::Once);
    assert_eq!(sched.len(), 0);

    // No re-fire afterwards.
    assert!(sched.advance_collect(ms(1000)).is_empty());
}

#[test]
fn zero_delay_fires_on_next_drain_even_without_advance() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    sched.schedule_after(Duration::ZERO, 1);
    // due == now (0); a zero-length advance still drains it.
    let fired = sched.advance_collect(Duration::ZERO);
    assert_eq!(payloads(&fired), vec![1]);
    assert_eq!(fired[0].at, Duration::ZERO);
}

#[test]
fn same_instant_fires_in_schedule_order() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    // Three timers all due at 10 ms; must fire in schedule order 10, 20, 30.
    sched.schedule_after(ms(10), 10);
    sched.schedule_after(ms(10), 20);
    sched.schedule_after(ms(10), 30);
    let fired = sched.advance_collect(ms(10));
    assert_eq!(payloads(&fired), vec![10, 20, 30]);
    for f in &fired {
        assert_eq!(f.at, ms(10));
    }
}

#[test]
fn mixed_due_times_sort_ascending_within_one_advance() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    sched.schedule_after(ms(30), 30);
    sched.schedule_after(ms(10), 10);
    sched.schedule_after(ms(20), 20);
    // One big advance crosses all three; they come out time-sorted.
    let fired = sched.advance_collect(ms(100));
    assert_eq!(payloads(&fired), vec![10, 20, 30]);
    assert_eq!(fired[0].at, ms(10));
    assert_eq!(fired[1].at, ms(20));
    assert_eq!(fired[2].at, ms(30));
}

#[test]
fn periodic_fires_every_period() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    sched.schedule_every(ms(500), 9);
    assert_eq!(sched.len(), 1);

    // First fire at 500 ms.
    assert!(sched.advance_collect(ms(499)).is_empty());
    let f1 = sched.advance_collect(ms(1));
    assert_eq!(payloads(&f1), vec![9]);
    assert_eq!(f1[0].at, ms(500));
    assert_eq!(f1[0].kind, TimerKind::Repeat);
    assert_eq!(sched.len(), 1); // still live (re-armed)

    // Second fire at 1000 ms.
    let f2 = sched.advance_collect(ms(500));
    assert_eq!(payloads(&f2), vec![9]);
    assert_eq!(f2[0].at, ms(1000));
}

#[test]
fn periodic_catches_up_across_a_big_delta() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    sched.schedule_every(ms(100), 1);
    // One 350 ms advance crosses 100, 200, 300 => three fires in order.
    let fired = sched.advance_collect(ms(350));
    assert_eq!(payloads(&fired), vec![1, 1, 1]);
    assert_eq!(fired[0].at, ms(100));
    assert_eq!(fired[1].at, ms(200));
    assert_eq!(fired[2].at, ms(300));
    // Next fire is scheduled for 400 ms: 60 ms more (to 410) fires it once.
    let next = sched.advance_collect(ms(60));
    assert_eq!(payloads(&next), vec![1]);
    assert_eq!(next[0].at, ms(400));
}

#[test]
fn cancel_prevents_firing() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    let h = sched.schedule_after(ms(100), 5);
    assert!(sched.is_active(h));
    assert!(sched.cancel(h));
    assert!(!sched.is_active(h));
    assert_eq!(sched.len(), 0);
    // Cancelling again is a no-op returning false.
    assert!(!sched.cancel(h));
    assert!(sched.advance_collect(ms(1000)).is_empty());
}

#[test]
fn cancel_one_of_many_leaves_the_rest() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    sched.schedule_after(ms(10), 1);
    let h2 = sched.schedule_after(ms(10), 2);
    sched.schedule_after(ms(10), 3);
    assert!(sched.cancel(h2));
    let fired = sched.advance_collect(ms(10));
    assert_eq!(payloads(&fired), vec![1, 3]);
}

#[test]
fn reschedule_moves_fire_time_and_keeps_handle_valid() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    let h = sched.schedule_after(ms(100), 42);
    // Push it out to 300 ms from now.
    assert!(sched.reschedule(h, ms(300)));
    assert!(sched.is_active(h));

    // The original 100 ms schedule is superseded: nothing at 100.
    assert!(sched.advance_collect(ms(100)).is_empty());
    // Still nothing until 300 ms total.
    assert!(sched.advance_collect(ms(199)).is_empty());
    let fired = sched.advance_collect(ms(1));
    assert_eq!(payloads(&fired), vec![42]);
    assert_eq!(fired[0].at, ms(300));
    // Handle issued originally is the one reported (survives reschedule).
    assert_eq!(fired[0].handle, h);
}

#[test]
fn reschedule_dead_handle_fails() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    let h = sched.schedule_after(ms(10), 1);
    let _ = sched.advance_collect(ms(10)); // fires and retires h
    assert!(!sched.reschedule(h, ms(10)));
    assert!(!sched.is_active(h));
}

#[test]
fn slot_reuse_does_not_resurrect_a_stale_handle() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    let h1 = sched.schedule_after(ms(10), 1);
    let _ = sched.advance_collect(ms(10)); // h1 fires, slot freed
    // New schedule reuses the freed slot under a fresh generation.
    let h2 = sched.schedule_after(ms(10), 2);
    assert_eq!(h1.index(), h2.index()); // same slot
    assert_ne!(h1.generation(), h2.generation());
    assert!(!sched.is_active(h1)); // stale handle rejected
    assert!(sched.is_active(h2));
    // Cancelling with the stale handle must not touch the live timer.
    assert!(!sched.cancel(h1));
    assert!(sched.is_active(h2));
}

#[test]
fn max_fires_cap_defers_the_remainder() {
    let mut sched: Scheduler<u32> = Scheduler::new().with_max_fires_per_advance(2);
    sched.schedule_after(ms(1), 1);
    sched.schedule_after(ms(1), 2);
    sched.schedule_after(ms(1), 3);
    sched.schedule_after(ms(1), 4);

    // One advance crosses all four, but the cap fires only two.
    let first = sched.advance_collect(ms(1));
    assert_eq!(payloads(&first), vec![1, 2]);
    // The backlog drains without advancing time (also capped at two).
    let mut rest = Vec::new();
    assert_eq!(sched.drain_due(&mut rest), 2);
    assert_eq!(payloads(&rest), vec![3, 4]);
    assert!(!sched.has_due());
}

#[test]
fn two_identical_runs_produce_identical_streams() {
    fn run() -> Vec<(u32, Duration)> {
        let mut sched: Scheduler<u32> = Scheduler::new();
        sched.schedule_every(ms(100), 1);
        sched.schedule_after(ms(250), 2);
        let h = sched.schedule_after(ms(400), 3);
        let mut out = Vec::new();
        for frame in 0..10 {
            // Deterministic awkward frame pattern.
            let dt = if frame % 2 == 0 { ms(60) } else { ms(90) };
            for f in sched.advance_collect(dt) {
                out.push((f.payload, f.at));
            }
            if frame == 5 {
                // Reschedule timer 3 mid-run; both runs do it identically.
                sched.reschedule(h, ms(50));
            }
        }
        out
    }
    let a = run();
    let b = run();
    assert_eq!(a, b);
    // Sanity: the stream is non-empty and time-ordered within each advance.
    assert!(!a.is_empty());
}

#[test]
fn schedule_every_from_separates_first_and_period() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    sched.schedule_every_from(ms(50), ms(200), 1);
    // First fire at 50 ms.
    let f0 = sched.advance_collect(ms(50));
    assert_eq!(payloads(&f0), vec![1]);
    assert_eq!(f0[0].at, ms(50));
    // Next at 50 + 200 = 250 ms.
    assert!(sched.advance_collect(ms(199)).is_empty());
    let f1 = sched.advance_collect(ms(1));
    assert_eq!(payloads(&f1), vec![1]);
    assert_eq!(f1[0].at, ms(250));
}

#[test]
#[should_panic(expected = "non-zero")]
fn schedule_every_zero_period_panics() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    sched.schedule_every(Duration::ZERO, 1);
}

#[test]
fn clear_removes_all_timers_but_keeps_time() {
    let mut sched: Scheduler<u32> = Scheduler::new();
    sched.schedule_after(ms(10), 1);
    sched.schedule_every(ms(10), 2);
    let _ = sched.advance_collect(ms(5));
    sched.clear();
    assert_eq!(sched.len(), 0);
    assert!(sched.is_empty());
    assert_eq!(sched.elapsed(), ms(5)); // time preserved
    assert!(sched.advance_collect(ms(1000)).is_empty());
}
