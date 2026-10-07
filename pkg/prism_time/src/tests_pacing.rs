//! §24.2/§24.3 — tests for frame pacing, the Reflex-form begin gate, and VRR.

use crate::pacing::{BeginDecision, FramePacer, ReflexGate, VrrPresent, VrrWindow};

const NS_60HZ: u64 = 16_666_666;

#[test]
fn first_present_only_anchors() {
    let mut p = FramePacer::new(NS_60HZ);
    let info = p.on_present(1_000);
    assert_eq!(info.interval_ns, 0);
    assert_eq!(info.jitter_ns, 0);
    assert!(!info.resynced);
    assert_eq!(info.next_present_ns, 1_000 + NS_60HZ);
    assert_eq!(p.presents(), 1);
}

#[test]
fn steady_cadence_has_zero_jitter_and_no_resync() {
    let mut p = FramePacer::new(NS_60HZ);
    let mut t = 0u64;
    p.on_present(t);
    for _ in 0..120 {
        t += NS_60HZ;
        let info = p.on_present(t);
        assert_eq!(info.jitter_ns, 0);
        assert!(!info.resynced);
        assert_eq!(info.interval_ns, NS_60HZ);
    }
    assert_eq!(p.resyncs(), 0);
    assert_eq!(p.max_jitter_ns(), 0);
    assert_eq!(p.smoothed_interval_ns(), NS_60HZ);
}

#[test]
fn small_jitter_is_absorbed_without_moving_prediction() {
    // Presents wobble +/- 1ms around the cadence but never past the half-frame
    // resync threshold: the predicted next-present stays on the steady grid.
    let mut p = FramePacer::new(NS_60HZ);
    let base = 1_000_000u64;
    p.on_present(base);
    let wobble: [i64; 6] = [
        1_000_000, -1_000_000, 500_000, -500_000, 1_000_000, -1_000_000,
    ];
    for (i, w) in wobble.iter().enumerate() {
        let ideal = base + NS_60HZ * (i as u64 + 1);
        let actual = (ideal as i64 + w) as u64;
        let info = p.on_present(actual);
        assert!(!info.resynced, "small jitter must not resync");
        // Prediction stays locked to the ideal grid, not the jittered sample.
        assert_eq!(info.next_present_ns, ideal + NS_60HZ);
    }
    assert_eq!(p.resyncs(), 0);
}

#[test]
fn large_deviation_hard_resyncs() {
    let mut p = FramePacer::new(NS_60HZ);
    p.on_present(0);
    // A huge hitch (whole frame late) exceeds the half-frame threshold.
    let late = NS_60HZ + NS_60HZ; // one full frame late vs predicted (NS_60HZ)
    let info = p.on_present(late);
    assert!(info.resynced);
    assert!(info.jitter_ns > 0);
    assert_eq!(info.next_present_ns, late + NS_60HZ);
    assert_eq!(p.resyncs(), 1);
}

#[test]
fn non_monotonic_timestamp_resyncs_and_zero_interval() {
    let mut p = FramePacer::new(NS_60HZ);
    p.on_present(10_000_000);
    let info = p.on_present(9_000_000); // goes backwards
    assert_eq!(info.interval_ns, 0);
    assert!(info.resynced);
    assert_eq!(info.next_present_ns, 9_000_000 + NS_60HZ);
}

#[test]
fn from_hz_matches_period() {
    let p = FramePacer::from_hz(60);
    assert_eq!(p.target_interval_ns(), 1_000_000_000 / 60);
    let p0 = FramePacer::from_hz(0); // clamped to 1 Hz
    assert_eq!(p0.target_interval_ns(), 1_000_000_000);
}

#[test]
fn reset_clears_history() {
    let mut p = FramePacer::new(NS_60HZ);
    p.on_present(0);
    p.on_present(NS_60HZ * 3); // forces resync
    p.reset();
    assert_eq!(p.presents(), 0);
    assert_eq!(p.resyncs(), 0);
    assert_eq!(p.max_jitter_ns(), 0);
    assert_eq!(p.smoothed_interval_ns(), NS_60HZ);
}

#[test]
fn pacer_is_deterministic_across_runs() {
    let run = || {
        let mut p = FramePacer::new(NS_60HZ).with_smoothing(1, 4);
        let samples = [0u64, 16_000_000, 33_600_000, 50_000_000, 70_000_000];
        let mut out = (0u64, 0i64, 0u64);
        for s in samples {
            let i = p.on_present(s);
            out = (i.next_present_ns, i.jitter_ns, i.smoothed_ns);
        }
        (out, p.resyncs(), p.max_jitter_ns())
    };
    assert_eq!(run(), run());
}

// ---- ReflexGate -------------------------------------------------------------

#[test]
fn reflex_queue_full_blocks() {
    let gate = ReflexGate::new(2, 0);
    assert_eq!(gate.decide(0, 1_000, 100, 2), BeginDecision::QueueFull);
    assert_eq!(gate.decide(0, 1_000, 100, 3), BeginDecision::QueueFull);
}

#[test]
fn reflex_waits_until_just_in_time() {
    // deadline 1000, work 100, margin 50 -> latest begin = 850.
    let gate = ReflexGate::new(3, 50);
    assert_eq!(gate.latest_begin_ns(1_000, 100), 850);
    assert_eq!(
        gate.decide(800, 1_000, 100, 1),
        BeginDecision::Wait { wait_ns: 50 }
    );
    assert_eq!(gate.decide(850, 1_000, 100, 1), BeginDecision::Begin);
    assert_eq!(gate.decide(900, 1_000, 100, 1), BeginDecision::Begin);
}

#[test]
fn reflex_begins_immediately_when_deadline_is_tight() {
    let gate = ReflexGate::new(3, 0);
    // Work already exceeds the time to deadline -> latest begin saturates to 0.
    assert_eq!(gate.latest_begin_ns(500, 800), 0);
    assert_eq!(gate.decide(100, 500, 800, 0), BeginDecision::Begin);
}

#[test]
fn reflex_zero_max_clamps_to_one() {
    let gate = ReflexGate::new(0, 0);
    assert_eq!(gate.max_frames_in_flight(), 1);
    assert_eq!(gate.decide(0, 1_000, 100, 1), BeginDecision::QueueFull);
}

// ---- VrrWindow --------------------------------------------------------------

#[test]
fn vrr_from_hz_orders_bounds() {
    let w = VrrWindow::from_hz(144, 48); // intentionally reversed
    assert_eq!(w.min_interval_ns(), 1_000_000_000 / 144);
    assert_eq!(w.max_interval_ns(), 1_000_000_000 / 48);
    assert!(w.min_interval_ns() < w.max_interval_ns());
}

#[test]
fn vrr_in_range_passthrough() {
    let w = VrrWindow::from_hz(48, 144);
    let mid = (w.min_interval_ns() + w.max_interval_ns()) / 2;
    assert_eq!(w.classify(mid), VrrPresent::InRange { interval_ns: mid });
    assert!(w.contains(mid));
}

#[test]
fn vrr_fast_frame_is_clamped() {
    let w = VrrWindow::from_hz(48, 144);
    let too_fast = w.min_interval_ns() / 2;
    assert_eq!(
        w.classify(too_fast),
        VrrPresent::ClampedFast {
            interval_ns: w.min_interval_ns()
        }
    );
    assert!(!w.contains(too_fast));
}

#[test]
fn vrr_slow_frame_triggers_lfc_within_window() {
    let w = VrrWindow::from_hz(48, 144); // max_interval ~= 20.83ms
                                         // 30ms is slower than the 48Hz floor -> LFC should duplicate.
    let slow = 30_000_000u64;
    match w.classify(slow) {
        VrrPresent::Lfc {
            multiplier,
            sub_interval_ns,
        } => {
            assert_eq!(multiplier, 2);
            assert_eq!(sub_interval_ns, slow / 2);
            // The sub-interval must fall back inside the refresh window.
            assert!(w.contains(sub_interval_ns));
        }
        other => panic!("expected LFC, got {other:?}"),
    }
}

#[test]
fn vrr_very_slow_frame_uses_higher_multiplier() {
    let w = VrrWindow::from_hz(48, 144); // max_interval ~= 20.83ms
    let very_slow = 50_000_000u64; // needs x3 to fit under ~20.83ms
    match w.classify(very_slow) {
        VrrPresent::Lfc {
            multiplier,
            sub_interval_ns,
        } => {
            assert_eq!(multiplier, 3);
            assert!(sub_interval_ns <= w.max_interval_ns());
            assert!(sub_interval_ns >= w.min_interval_ns());
        }
        other => panic!("expected LFC, got {other:?}"),
    }
}

#[test]
fn vrr_earliest_present_enforces_max_refresh() {
    let w = VrrWindow::from_hz(48, 144);
    let last = 1_000_000u64;
    // Ready sooner than the min interval -> held to last + min_interval.
    let early = last + w.min_interval_ns() / 2;
    assert_eq!(
        w.earliest_present_ns(early, last),
        last + w.min_interval_ns()
    );
    // Ready later than the floor -> presented when ready.
    let late = last + w.min_interval_ns() * 3;
    assert_eq!(w.earliest_present_ns(late, last), late);
}

#[test]
fn vrr_effective_interval_accessor() {
    let in_range = VrrPresent::InRange { interval_ns: 10 };
    assert_eq!(in_range.effective_interval_ns(), 10);
    let clamped = VrrPresent::ClampedFast { interval_ns: 7 };
    assert_eq!(clamped.effective_interval_ns(), 7);
    let lfc = VrrPresent::Lfc {
        multiplier: 2,
        sub_interval_ns: 15,
    };
    assert_eq!(lfc.effective_interval_ns(), 15);
}

#[test]
fn vrr_new_clamps_degenerate_window() {
    let w = VrrWindow::new(0, 0);
    assert_eq!(w.min_interval_ns(), 1);
    assert_eq!(w.max_interval_ns(), 1);
    let w2 = VrrWindow::new(100, 50); // max < min
    assert_eq!(w2.min_interval_ns(), 100);
    assert_eq!(w2.max_interval_ns(), 100);
}
