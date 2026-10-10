//! §24.1 tests: performance-budget contract, scheduler feedback, regression
//! detection, and hotspot attribution. Pure arithmetic; no clock.

use crate::budget::{
    hotspot_diff, Baseline, BudgetRegistry, Hotspot, RegressionConfig, RegressionTracker,
};

#[test]
fn declare_and_overwrite_budget_preserves_order() {
    let mut reg = BudgetRegistry::fps_60();
    reg.declare("render", 8_000_000)
        .declare("physics", 2_000_000)
        .declare("gameplay", 3_000_000);
    assert_eq!(reg.len(), 3);
    assert_eq!(reg.budget_of("render"), Some(8_000_000));
    assert_eq!(reg.frame_budget_nanos(), 16_667_000);

    // Overwrite replaces in place, order unchanged.
    reg.declare("render", 7_000_000);
    assert_eq!(reg.len(), 3);
    assert_eq!(reg.budget_of("render"), Some(7_000_000));
    assert_eq!(reg.budget_of("missing"), None);
}

#[test]
fn single_category_over_and_under_budget() {
    let mut reg = BudgetRegistry::fps_60();
    reg.declare("render", 8_000_000);

    let under = reg.evaluate("render", 6_000_000).unwrap();
    assert!(!under.over_budget);
    assert_eq!(under.overspend_nanos, 0);
    assert!((under.utilization() - 0.75).abs() < 1e-9);

    let over = reg.evaluate("render", 10_000_000).unwrap();
    assert!(over.over_budget);
    assert_eq!(over.overspend_nanos, 2_000_000);
    assert!((over.utilization() - 1.25).abs() < 1e-9);

    assert!(reg.evaluate("absent", 1).is_none());
}

#[test]
fn frame_report_rolls_up_and_feeds_scheduler_headroom() {
    let mut reg = BudgetRegistry::fps_60();
    reg.declare("render", 8_000_000)
        .declare("physics", 2_000_000)
        .declare("gameplay", 3_000_000);

    // render over, physics/gameplay under; total under the 16.667ms frame.
    let report = reg.evaluate_frame(&[
        ("render", 9_000_000),
        ("physics", 1_500_000),
        ("gameplay", 2_000_000),
    ]);
    assert_eq!(report.total_measured_nanos, 12_500_000);
    assert!(!report.over_frame);
    assert_eq!(report.remaining_background_nanos, 16_667_000 - 12_500_000);

    // Exactly one overspender, and it is render.
    let overspenders: Vec<_> = report.overspenders().collect();
    assert_eq!(overspenders.len(), 1);
    assert_eq!(overspenders[0].category, "render");
    assert_eq!(overspenders[0].overspend_nanos, 1_000_000);

    // Declaration order preserved in statuses.
    assert_eq!(report.statuses[0].category, "render");
    assert_eq!(report.statuses[1].category, "physics");
    assert_eq!(report.statuses[2].category, "gameplay");
}

#[test]
fn frame_over_budget_zeroes_background_headroom() {
    let mut reg = BudgetRegistry::fps_120(); // 8.333ms frame
    reg.declare("render", 6_000_000);

    let report = reg.evaluate_frame(&[("render", 7_000_000), ("physics", 3_000_000)]);
    assert_eq!(report.total_measured_nanos, 10_000_000);
    assert!(report.over_frame);
    // No headroom for background work when the frame is blown.
    assert_eq!(report.remaining_background_nanos, 0);

    // physics was undeclared -> zero budget -> reported after declared ones.
    let physics = report
        .statuses
        .iter()
        .find(|s| s.category == "physics")
        .unwrap();
    assert_eq!(physics.budget_nanos, 0);
    assert!(!physics.over_budget); // zero budget never flips the red flag
    assert_eq!(physics.overspend_nanos, 3_000_000); // but overspend == measured
    assert_eq!(physics.utilization(), 0.0); // zero budget -> ratio 0
}

#[test]
fn declared_but_unmeasured_category_is_within_budget() {
    let mut reg = BudgetRegistry::fps_60();
    reg.declare("audio", 1_000_000);
    let report = reg.evaluate_frame(&[("render", 5_000_000)]);
    let audio = report
        .statuses
        .iter()
        .find(|s| s.category == "audio")
        .unwrap();
    assert_eq!(audio.measured_nanos, 0);
    assert!(!audio.over_budget);
    // render still contributes to the summed foreground work + headroom.
    assert_eq!(report.total_measured_nanos, 5_000_000);
}

#[test]
fn baseline_from_samples_matches_nearest_rank() {
    // 1..=100 => p50 nearest-rank = 50, p99 = 99.
    let samples: Vec<u64> = (1..=100).collect();
    let base = Baseline::from_samples(&samples);
    assert_eq!(base.p50_nanos, 50);
    assert_eq!(base.p99_nanos, 99);
    // Empty => zeros.
    assert_eq!(Baseline::from_samples(&[]).p50_nanos, 0);
}

#[test]
fn regression_fires_only_past_threshold_and_above_floor() {
    let mut tracker = RegressionTracker::new(RegressionConfig {
        threshold_ratio: 1.05, // +5%
        min_baseline_nanos: 50_000,
    });
    tracker.set_baseline(
        "render.submit",
        Baseline {
            p50_nanos: 1_000_000,
            p99_nanos: 2_000_000,
        },
    );

    // +4% on both: within threshold, silent.
    assert!(tracker
        .check("render.submit", 1_040_000, 2_080_000, Some("abc1234"))
        .is_none());

    // +6% on p99 only: fires, attributed, p50 clean.
    let alert = tracker
        .check("render.submit", 1_020_000, 2_120_000, Some("abc1234"))
        .expect("p99 regression should fire");
    assert!(!alert.p50_regressed);
    assert!(alert.p99_regressed);
    assert_eq!(alert.commit.as_deref(), Some("abc1234"));
    assert!((alert.p99_ratio - 1.06).abs() < 1e-9);

    // Unknown key: no baseline, no alert.
    assert!(tracker
        .check("unknown", 9_999_999, 9_999_999, None)
        .is_none());
}

#[test]
fn regression_ignores_micro_spans_below_noise_floor() {
    let mut tracker = RegressionTracker::new(RegressionConfig::default());
    // Baseline below the 50us floor: even a huge relative jump is ignored.
    tracker.set_baseline(
        "tiny",
        Baseline {
            p50_nanos: 10_000,
            p99_nanos: 20_000,
        },
    );
    assert!(tracker.check("tiny", 1_000_000, 2_000_000, None).is_none());
}

#[test]
fn hotspot_diff_sorts_by_cost_and_flags_regressions() {
    let baseline = [
        Hotspot::new("shade", 4_000_000),
        Hotspot::new("cull", 1_000_000),
        Hotspot::new("skin", 500_000),
    ];
    let current = [
        Hotspot::new("skin", 520_000),       // +4% within threshold
        Hotspot::new("cull", 1_500_000),     // +50% regression
        Hotspot::new("shade", 4_050_000),    // +1.25% clean
        Hotspot::new("newcomer", 2_000_000), // appeared from nothing
    ];

    let diff = hotspot_diff(&current, &baseline, 1.10); // +10% threshold

    // Sorted by current self time descending.
    let order: Vec<&str> = diff.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(order, ["shade", "newcomer", "cull", "skin"]);

    let cull = diff.iter().find(|d| d.name == "cull").unwrap();
    assert!(cull.is_regression);
    assert_eq!(cull.delta_nanos, 500_000);
    assert!((cull.ratio - 1.5).abs() < 1e-9);

    let skin = diff.iter().find(|d| d.name == "skin").unwrap();
    assert!(!skin.is_regression);

    let shade = diff.iter().find(|d| d.name == "shade").unwrap();
    assert!(!shade.is_regression);

    // A newly-appeared hotspot (zero baseline) with real cost is a regression.
    let newcomer = diff.iter().find(|d| d.name == "newcomer").unwrap();
    assert!(newcomer.is_regression);
    assert_eq!(newcomer.baseline_nanos, 0);
    assert_eq!(newcomer.ratio, 0.0);
}
