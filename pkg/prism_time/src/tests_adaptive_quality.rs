//! §24.4 tests: frame-budget-driven adaptive quality. Every oracle is
//! hand-computed from fixed `(frame, budget)` inputs; the control law is pure
//! integer / ppm arithmetic, so each expectation is exact.

use crate::{
    utilization_ppm, AdaptiveQualityConfig, AdaptiveQualityController, Duration, QualityAdjustment,
};

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

// --- utilization_ppm -------------------------------------------------------

#[test]
fn utilization_is_exact_fraction_of_budget() {
    // 16 ms budget. util_ppm = frame_ns * 1e6 / budget_ns.
    assert_eq!(utilization_ppm(ms(16), ms(16)), 1_000_000); // 100%
    assert_eq!(utilization_ppm(ms(8), ms(16)), 500_000); // 50%
    assert_eq!(utilization_ppm(ms(12), ms(16)), 750_000); // 75%
    assert_eq!(utilization_ppm(ms(24), ms(16)), 1_500_000); // 150%
    assert_eq!(utilization_ppm(ms(32), ms(16)), 2_000_000); // 200%
    // 13 ms = 812_500 ppm, inside the default 80%..100% dead-band.
    assert_eq!(utilization_ppm(ms(13), ms(16)), 812_500);
}

#[test]
fn utilization_zero_budget_is_zero() {
    assert_eq!(utilization_ppm(ms(16), Duration::ZERO), 0);
}

// --- hysteresis + patience -------------------------------------------------

#[test]
fn fast_downgrade_after_patience() {
    let cfg = AdaptiveQualityConfig::new(0, 4); // patience 2 down, cooldown 8
    let mut ctl = AdaptiveQualityController::at_max(cfg);
    assert_eq!(ctl.level(), 4);

    // One over-budget frame is not enough (patience 2).
    assert_eq!(ctl.observe(ms(16), ms(16)), QualityAdjustment::Hold);
    assert_eq!(ctl.over_streak(), 1);
    assert_eq!(ctl.level(), 4);

    // Second consecutive over-budget frame downgrades.
    assert_eq!(
        ctl.observe(ms(16), ms(16)),
        QualityAdjustment::Downgrade { from: 4, to: 3 }
    );
    assert_eq!(ctl.level(), 3);
    assert_eq!(ctl.cooldown_remaining(), 8);
    assert_eq!(ctl.over_streak(), 0);
}

#[test]
fn dead_band_frames_hold_and_reset_streaks() {
    let cfg = AdaptiveQualityConfig::new(0, 4).with_patience(2, 4).with_cooldown(0);
    let mut ctl = AdaptiveQualityController::new(cfg, 2);

    // 13 ms @ 16 ms budget = 812_500 ppm: neutral, nothing moves.
    for _ in 0..10 {
        assert_eq!(ctl.observe(ms(13), ms(16)), QualityAdjustment::Hold);
        assert_eq!(ctl.level(), 2);
        assert_eq!(ctl.over_streak(), 0);
        assert_eq!(ctl.under_streak(), 0);
    }
}

#[test]
fn slow_upgrade_requires_full_patience() {
    let cfg = AdaptiveQualityConfig::new(0, 4).with_patience(2, 4).with_cooldown(0);
    let mut ctl = AdaptiveQualityController::new(cfg, 1);

    // 8 ms @ 16 ms = 500_000 ppm: comfortably under budget.
    assert_eq!(ctl.observe(ms(8), ms(16)), QualityAdjustment::Hold); // streak 1
    assert_eq!(ctl.observe(ms(8), ms(16)), QualityAdjustment::Hold); // streak 2
    assert_eq!(ctl.observe(ms(8), ms(16)), QualityAdjustment::Hold); // streak 3
    assert_eq!(
        ctl.observe(ms(8), ms(16)),
        QualityAdjustment::Upgrade { from: 1, to: 2 }
    ); // streak 4 -> up
    assert_eq!(ctl.level(), 2);
    assert_eq!(ctl.under_streak(), 0);
}

#[test]
fn neutral_frame_breaks_upgrade_streak() {
    let cfg = AdaptiveQualityConfig::new(0, 4).with_patience(2, 3).with_cooldown(0);
    let mut ctl = AdaptiveQualityController::new(cfg, 1);
    assert_eq!(ctl.observe(ms(8), ms(16)), QualityAdjustment::Hold); // under 1
    assert_eq!(ctl.observe(ms(8), ms(16)), QualityAdjustment::Hold); // under 2
    // A neutral frame resets the under streak.
    assert_eq!(ctl.observe(ms(13), ms(16)), QualityAdjustment::Hold);
    assert_eq!(ctl.under_streak(), 0);
    // Must build the full streak again.
    assert_eq!(ctl.observe(ms(8), ms(16)), QualityAdjustment::Hold); // 1
    assert_eq!(ctl.observe(ms(8), ms(16)), QualityAdjustment::Hold); // 2
    assert_eq!(
        ctl.observe(ms(8), ms(16)),
        QualityAdjustment::Upgrade { from: 1, to: 2 }
    );
}

// --- severe overrun --------------------------------------------------------

#[test]
fn severe_overrun_drops_multiple_levels_and_bypasses_cooldown() {
    let cfg = AdaptiveQualityConfig::new(0, 4); // severe 150%, step 2, cooldown 8
    let mut ctl = AdaptiveQualityController::at_max(cfg);

    // Trigger a normal downgrade first so a cooldown is active.
    assert_eq!(ctl.observe(ms(16), ms(16)), QualityAdjustment::Hold);
    assert_eq!(
        ctl.observe(ms(16), ms(16)),
        QualityAdjustment::Downgrade { from: 4, to: 3 }
    );
    assert_eq!(ctl.cooldown_remaining(), 8);

    // A severe frame (24 ms = 150%) bypasses the cooldown and drops 2 levels.
    assert_eq!(
        ctl.observe(ms(24), ms(16)),
        QualityAdjustment::Downgrade { from: 3, to: 1 }
    );
    assert_eq!(ctl.level(), 1);
    assert_eq!(ctl.cooldown_remaining(), 8);
}

#[test]
fn severe_clamps_at_min_level() {
    let cfg = AdaptiveQualityConfig::new(0, 4).with_severe(1_500_000, 2);
    let mut ctl = AdaptiveQualityController::new(cfg, 1);
    // Step 2 from level 1 saturates at the floor (0).
    assert_eq!(
        ctl.observe(ms(24), ms(16)),
        QualityAdjustment::Downgrade { from: 1, to: 0 }
    );
    // Already at min: a severe frame holds (nowhere lower to go).
    assert_eq!(ctl.observe(ms(24), ms(16)), QualityAdjustment::Hold);
    assert!(ctl.is_at_min());
}

// --- cooldown --------------------------------------------------------------

#[test]
fn cooldown_suppresses_normal_adjustments_for_exact_window() {
    let cfg = AdaptiveQualityConfig::new(0, 4).with_patience(1, 1).with_cooldown(3);
    let mut ctl = AdaptiveQualityController::at_max(cfg);

    // Patience 1: first over frame downgrades, arming a 3-frame cooldown.
    assert_eq!(
        ctl.observe(ms(16), ms(16)),
        QualityAdjustment::Downgrade { from: 4, to: 3 }
    );
    assert_eq!(ctl.cooldown_remaining(), 3);

    // Exactly 3 held frames despite being over budget each time.
    assert_eq!(ctl.observe(ms(16), ms(16)), QualityAdjustment::Hold);
    assert_eq!(ctl.cooldown_remaining(), 2);
    assert_eq!(ctl.observe(ms(16), ms(16)), QualityAdjustment::Hold);
    assert_eq!(ctl.cooldown_remaining(), 1);
    assert_eq!(ctl.observe(ms(16), ms(16)), QualityAdjustment::Hold);
    assert_eq!(ctl.cooldown_remaining(), 0);

    // Cooldown elapsed: the next over-budget frame downgrades again.
    assert_eq!(
        ctl.observe(ms(16), ms(16)),
        QualityAdjustment::Downgrade { from: 3, to: 2 }
    );
}

// --- config clamping + helpers ---------------------------------------------

#[test]
fn config_is_reclamped_into_consistent_state() {
    // max < min, upgrade > downgrade, severe < downgrade, zero step/patience.
    let cfg = AdaptiveQualityConfig::new(5, 2)
        .with_hysteresis(900_000, 500_000)
        .with_severe(100_000, 0)
        .with_patience(0, 0);
    let ctl = AdaptiveQualityController::new(cfg, 3);
    let c = ctl.config();
    assert_eq!(c.min_level, 5);
    assert_eq!(c.max_level, 5); // raised to min
    assert_eq!(c.upgrade_ppm, 500_000); // lowered to downgrade
    assert_eq!(c.downgrade_ppm, 500_000);
    assert_eq!(c.severe_ppm, 500_000); // raised to downgrade
    assert_eq!(c.severe_step, 1);
    assert_eq!(c.downgrade_patience, 1);
    assert_eq!(c.upgrade_patience, 1);
    // start_level clamped into [5, 5].
    assert_eq!(ctl.level(), 5);
}

#[test]
fn set_level_and_reset_clear_state() {
    let cfg = AdaptiveQualityConfig::new(0, 4).with_patience(2, 2).with_cooldown(5);
    let mut ctl = AdaptiveQualityController::at_max(cfg);
    assert_eq!(ctl.observe(ms(16), ms(16)), QualityAdjustment::Hold);
    ctl.set_level(2);
    assert_eq!(ctl.level(), 2);
    assert_eq!(ctl.over_streak(), 0);
    assert_eq!(ctl.cooldown_remaining(), 0);
    ctl.set_level(99); // clamps to max
    assert_eq!(ctl.level(), 4);
    assert!(ctl.is_at_max());
    ctl.reset(0);
    assert!(ctl.is_at_min());
}

#[test]
fn adjustment_helpers_report_change_and_delta() {
    let up = QualityAdjustment::Upgrade { from: 1, to: 2 };
    let down = QualityAdjustment::Downgrade { from: 3, to: 1 };
    assert!(up.changed() && down.changed());
    assert!(!QualityAdjustment::Hold.changed());
    assert_eq!(up.delta(), 1);
    assert_eq!(down.delta(), -2);
    assert_eq!(QualityAdjustment::Hold.delta(), 0);
    assert_eq!(up.to_level(), Some(2));
    assert_eq!(QualityAdjustment::Hold.to_level(), None);
}

#[test]
fn zero_budget_holds_without_state_change() {
    let mut ctl = AdaptiveQualityController::default();
    let before = ctl;
    assert_eq!(ctl.observe(ms(16), Duration::ZERO), QualityAdjustment::Hold);
    assert_eq!(ctl, before);
}

// --- determinism -----------------------------------------------------------

#[test]
fn two_runs_produce_identical_trajectories() {
    // A fixed, varied workload: over, severe, under, neutral frames.
    let seq: [(u64, u64); 12] = [
        (16, 16),
        (16, 16),
        (24, 16),
        (8, 16),
        (8, 16),
        (13, 16),
        (8, 16),
        (8, 16),
        (8, 16),
        (8, 16),
        (32, 16),
        (16, 16),
    ];
    let cfg = AdaptiveQualityConfig::new(0, 4).with_patience(2, 3).with_cooldown(2);
    let mut a = AdaptiveQualityController::at_max(cfg);
    let mut b = AdaptiveQualityController::at_max(cfg);
    for &(f, bud) in &seq {
        let da = a.observe(ms(f), ms(bud));
        let db = b.observe(ms(f), ms(bud));
        assert_eq!(da, db);
        assert_eq!(a.level(), b.level());
    }
}
