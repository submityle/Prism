//! M6 tests for the hitch detector.

use crate::hitch::{HitchConfig, HitchDetector};

#[test]
fn absolute_budget_flags_over_budget_frames() {
    let config = HitchConfig {
        budget_nanos: 16_666_667, // 60 FPS
        window: 32,
        percentile: 0.95,
        spike_ratio: 0.0, // disable baseline path; test the budget path alone
        min_samples: 4,
    };
    let mut det = HitchDetector::new(config);

    // A comfortable frame is silent.
    assert!(det.record_frame_at(8_000_000, 1_000).is_none());
    assert_eq!(det.hitch_count(), 0);

    // A frame over budget is flagged with over_budget set.
    let event = det
        .record_frame_at(40_000_000, 2_000)
        .expect("over-budget frame is a hitch");
    assert!(event.over_budget);
    assert!(!event.over_baseline);
    assert_eq!(event.duration_nanos, 40_000_000);
    assert_eq!(event.timestamp_nanos, 2_000);
    assert_eq!(det.hitch_count(), 1);
    assert_eq!(det.worst_nanos(), 40_000_000);
    assert_eq!(det.frames_observed(), 2);
}

#[test]
fn baseline_spike_is_measured_over_prior_frames_only() {
    let config = HitchConfig {
        budget_nanos: 0, // disable absolute budget; isolate the spike path
        window: 16,
        percentile: 0.5, // median baseline
        spike_ratio: 3.0,
        min_samples: 4,
    };
    let mut det = HitchDetector::new(config);

    // Prime a stable ~10ms baseline. Below min_samples the baseline is 0 and
    // nothing can be flagged yet.
    for i in 0..4 {
        assert!(det.record_frame_at(10_000_000, i).is_none());
    }
    assert_eq!(det.baseline_nanos(), 10_000_000);

    // A 50ms frame (> 3x the 10ms median) is a spike. The baseline is computed
    // over the *prior* frames, so the slow frame cannot mask itself.
    let event = det
        .record_frame_at(50_000_000, 100)
        .expect("5x baseline frame is a hitch");
    assert!(event.over_baseline);
    assert!(!event.over_budget);
    assert_eq!(event.baseline_nanos, 10_000_000);
}

#[test]
fn window_eviction_keeps_history_bounded() {
    let config = HitchConfig {
        budget_nanos: 0,
        window: 4,
        percentile: 1.0, // max of the retained window
        spike_ratio: 0.0,
        min_samples: 1,
    };
    let mut det = HitchDetector::new(config);
    for d in [100_u64, 200, 300, 400] {
        det.record_frame_at(d, 0);
    }
    assert_eq!(det.baseline_nanos(), 400);
    // Evict the window with small frames; the old 400 max must roll off.
    for _ in 0..4 {
        det.record_frame_at(50, 0);
    }
    assert_eq!(det.baseline_nanos(), 50);
}

#[test]
fn window_is_clamped_to_at_least_one() {
    let config = HitchConfig {
        window: 0,
        ..HitchConfig::fps_60()
    };
    let det = HitchDetector::new(config);
    assert_eq!(det.config().window, 1, "window must never be zero");
}

#[test]
fn presets_match_their_frame_rates() {
    assert_eq!(HitchConfig::fps_60().budget_nanos, 16_667_000);
    assert_eq!(HitchConfig::fps_120().budget_nanos, 8_333_000);
}
