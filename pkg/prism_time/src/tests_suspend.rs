//! §24.7 tests: suspend / resume time-source correction. Every oracle is
//! hand-computed from fixed wall timestamps; the correction is pure integer
//! nanosecond arithmetic, so each expectation is exact.

use crate::{Duration, SuspendableClock};

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

// --- plain running advance -------------------------------------------------

#[test]
fn running_advance_accumulates_wall_deltas() {
    let mut c = SuspendableClock::new();
    // First observation only sets the baseline.
    assert_eq!(c.advance_to(ms(0)), ms(0));
    assert_eq!(c.elapsed(), ms(0));
    // Subsequent observations fold in the running delta.
    assert_eq!(c.advance_to(ms(10)), ms(10));
    assert_eq!(c.advance_to(ms(25)), ms(15));
    assert_eq!(c.elapsed(), ms(25));
    assert_eq!(c.elapsed_nanos(), 25_000_000);
    // Tick view: 25 ms / 5 ms = 5 ticks.
    assert_eq!(c.ticks(ms(5)), 5);
    assert_eq!(c.ticks(Duration::ZERO), 0);
    assert!(!c.is_suspended());
}

#[test]
fn backward_timestamp_saturates_to_zero_delta() {
    let mut c = SuspendableClock::new();
    c.advance_to(ms(0));
    assert_eq!(c.advance_to(ms(10)), ms(10));
    // Non-monotonic backward timestamp: no negative delta, no change.
    assert_eq!(c.advance_to(ms(5)), ms(0));
    assert_eq!(c.elapsed(), ms(10));
}

// --- suspend / resume ------------------------------------------------------

#[test]
fn suspend_resume_excludes_the_gap_and_resume_has_no_spike() {
    let mut c = SuspendableClock::new();
    c.advance_to(ms(0));
    assert_eq!(c.advance_to(ms(10)), ms(10));

    // Suspend 2 ms later: the running portion up to the suspend instant counts.
    assert_eq!(c.suspend(ms(12)), ms(2));
    assert_eq!(c.elapsed(), ms(12));
    assert!(c.is_suspended());

    // Timestamps observed while suspended do not advance simulated time.
    assert_eq!(c.advance_to(ms(100)), ms(0));
    assert_eq!(c.elapsed(), ms(12));

    // Resume far later: the suspended span is excluded (reported, not charged).
    assert_eq!(c.resume(ms(500)), ms(488));
    assert_eq!(c.total_suspended(), ms(488));
    assert_eq!(c.elapsed(), ms(12));
    assert!(!c.is_suspended());

    // The first post-resume frame is a normal small delta, not a 488 ms spike.
    assert_eq!(c.advance_to(ms(510)), ms(10));
    assert_eq!(c.elapsed(), ms(22));
}

#[test]
fn suspend_before_first_observation_sets_baseline() {
    let mut c = SuspendableClock::new();
    // Suspend with no prior timestamp: establishes the baseline, zero applied.
    assert_eq!(c.suspend(ms(5)), ms(0));
    assert!(c.is_suspended());
    assert_eq!(c.resume(ms(20)), ms(15));
    assert_eq!(c.total_suspended(), ms(15));
    assert_eq!(c.advance_to(ms(25)), ms(5));
    assert_eq!(c.elapsed(), ms(5));
}

#[test]
fn double_suspend_and_stray_resume_are_noops() {
    let mut c = SuspendableClock::new();
    c.advance_to(ms(0));
    c.advance_to(ms(10));
    assert_eq!(c.suspend(ms(10)), ms(0));
    // A second suspend while suspended does nothing.
    assert_eq!(c.suspend(ms(50)), ms(0));
    assert_eq!(c.resume(ms(30)), ms(20));
    // A resume while already running does nothing.
    assert_eq!(c.resume(ms(40)), ms(0));
    assert_eq!(c.elapsed(), ms(10));
}

// --- max catch-up clamp ----------------------------------------------------

#[test]
fn max_delta_clamp_discards_excess() {
    let mut c = SuspendableClock::new().with_max_delta(ms(100));
    assert_eq!(c.max_delta(), Some(ms(100)));
    c.advance_to(ms(0));
    // Under the cap: folded in whole.
    assert_eq!(c.advance_to(ms(50)), ms(50));
    // Over the cap: only 100 ms of the 250 ms step counts; 150 ms discarded.
    assert_eq!(c.advance_to(ms(300)), ms(100));
    assert_eq!(c.elapsed(), ms(150));
    assert_eq!(c.total_clamped(), ms(150));
}

#[test]
fn zero_max_delta_freezes_running_advance() {
    let mut c = SuspendableClock::new().with_max_delta(Duration::ZERO);
    c.advance_to(ms(0));
    assert_eq!(c.advance_to(ms(10)), ms(0));
    assert_eq!(c.elapsed(), ms(0));
    assert_eq!(c.total_clamped(), ms(10));
}

#[test]
fn max_delta_can_be_set_and_cleared() {
    let mut c = SuspendableClock::new();
    assert_eq!(c.max_delta(), None);
    c.set_max_delta(Some(ms(20)));
    c.advance_to(ms(0));
    assert_eq!(c.advance_to(ms(100)), ms(20)); // clamped
    c.set_max_delta(None);
    assert_eq!(c.advance_to(ms(300)), ms(200)); // unclamped
}

// --- reset + determinism ---------------------------------------------------

#[test]
fn reset_restores_initial_state_keeping_clamp() {
    let mut c = SuspendableClock::new().with_max_delta(ms(100));
    c.advance_to(ms(0));
    c.advance_to(ms(50));
    c.suspend(ms(60));
    c.resume(ms(200));
    c.reset();
    assert_eq!(c.elapsed(), ms(0));
    assert!(!c.is_suspended());
    assert_eq!(c.total_suspended(), ms(0));
    assert_eq!(c.total_clamped(), ms(0));
    assert_eq!(c.max_delta(), Some(ms(100))); // clamp preserved
    // Baseline re-established cleanly after reset.
    assert_eq!(c.advance_to(ms(1000)), ms(0));
    assert_eq!(c.advance_to(ms(1010)), ms(10));
}

#[test]
fn two_runs_produce_identical_simulated_timelines() {
    // (op, wall_ms): op 0 = advance, 1 = suspend, 2 = resume.
    let seq: [(u8, u64); 10] = [
        (0, 0),
        (0, 16),
        (0, 33),
        (1, 40),
        (0, 1000),
        (2, 5000),
        (0, 5016),
        (0, 10016), // large step, clamped
        (0, 10032),
        (0, 10048),
    ];
    let run = || {
        let mut c = SuspendableClock::new().with_max_delta(ms(100));
        for &(op, w) in &seq {
            match op {
                0 => {
                    c.advance_to(ms(w));
                }
                1 => {
                    c.suspend(ms(w));
                }
                _ => {
                    c.resume(ms(w));
                }
            }
        }
        (c.elapsed_nanos(), c.total_suspended(), c.total_clamped())
    };
    assert_eq!(run(), run());
}
