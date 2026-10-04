//! End-to-end integration tests for the CPU-budget governor closed loop.
//!
//! These wire the governor's independent stages together exactly as an engine
//! frame loop would and drive them over a sequence of simulated audio blocks:
//! a [`BudgetTracker`] smooths each block's measured cost ratio into a load,
//! a [`Hysteresis`] controller turns that load into a dwell-gated raise / hold
//! / lower decision, the decision steps a [`QualityTier`] along a
//! [`QualityLadder`], and the active [`PowerProfile`] caps the emitted tier.
//! The resulting [`GovernorReport`] is asserted block by block.
//!
//! The per-module unit tests cover each stage in isolation (one `observe`, one
//! `update`, one `clamp`). What is only observable end to end -- and therefore
//! only covered here -- is the *loop behaviour*: that sustained overload
//! actually lowers quality after (and only after) the dwell window, that
//! recovery climbs back, that the dead band holds steady without flapping,
//! that a power profile caps the ceiling with the right [`DegradeReason`], and
//! that extreme inputs never push the tier out of range.
//!
//! # Provenance
//! Original work. Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, or Google Resonance Audio source or derived code, and no
//! AI/ML. The CPU-budget-driven quality-scaling idea is drawn only as a
//! concept from public descriptions of performance governors; the driver below
//! and the implementation under test are independent.
//!
//! # Relationship
//! Covers §32 (performance-adaptive governance and audio LOD) as a black-box
//! public-API contract over the `prism_audio_governor::governor` stages
//! (`budget` -> `hysteresis` -> `lod` -> `power` -> `report`). The small
//! `ClosedLoop` driver mirrors how a host wires the stages per block.

use prism_audio_core::math::Sample;

use prism_audio_governor::governor::budget::BudgetTracker;
use prism_audio_governor::governor::hysteresis::Hysteresis;
use prism_audio_governor::governor::lod::{QualityLadder, QualityTier};
use prism_audio_governor::governor::power::{PowerConstraints, PowerProfile};
use prism_audio_governor::governor::report::{DegradeReason, GovernorReport};

/// A per-block governor driver that wires the independent stages into the §32
/// closed loop. The loop keeps its own uncapped tier wish; the active power
/// profile only caps the emitted tier, so the loop's intent and the platform
/// ceiling stay distinguishable in the report.
struct ClosedLoop {
    tracker: BudgetTracker,
    hysteresis: Hysteresis,
    ladder: QualityLadder,
    constraints: PowerConstraints,
    /// The loop's own desired tier, in `[0, ladder.max]`, before the power cap.
    wish: QualityTier,
}

impl ClosedLoop {
    /// Builds a loop with no load smoothing (so a block's load equals its cost
    /// ratio and the dwell logic is exactly countable), a `[0.6, 0.85]` dead
    /// band, and a three-block dwell. The loop starts at full richness.
    fn new(profile: PowerProfile) -> Self {
        let ladder = QualityLadder::standard();
        let wish = ladder.max_tier();
        Self {
            tracker: BudgetTracker::new(0.0),
            hysteresis: Hysteresis::new(0.6, 0.85, 3),
            constraints: profile.constraints(),
            ladder,
            wish,
        }
    }

    /// Advances the loop by one block of measured `cost_ratio` and returns the
    /// resulting report.
    fn step(&mut self, cost_ratio: Sample) -> GovernorReport {
        let load = self.tracker.observe(cost_ratio);
        let delta = self.hysteresis.update(load).delta();

        // Step the loop's own wish within the ladder's valid range.
        let max = i32::from(self.ladder.max_tier().0);
        let next = (i32::from(self.wish.0) + delta).clamp(0, max);
        self.wish = QualityTier(next as u8);

        // The power profile caps only the emitted tier.
        let emitted = self.constraints.clamp_tier(self.wish);
        let reason = if emitted.0 >= self.ladder.max_tier().0 {
            DegradeReason::None
        } else if emitted.0 < self.wish.0 {
            DegradeReason::PowerCap
        } else {
            DegradeReason::CpuBudget
        };

        GovernorReport::new(emitted, load, reason, self.ladder.profile(emitted))
    }

    fn tier(&mut self, cost_ratio: Sample) -> QualityTier {
        self.step(cost_ratio).tier
    }
}

#[test]
fn sustained_overload_lowers_quality_only_after_the_dwell_window() {
    let mut loop_ = ClosedLoop::new(PowerProfile::Desktop);
    let top = QualityLadder::standard().max_tier();

    // Two overloaded blocks must not move the tier: the dwell is three.
    assert_eq!(loop_.tier(1.5), top, "block 1 holds");
    assert_eq!(loop_.tier(1.5), top, "block 2 holds");
    // The third consecutive hot block fires the first Lower.
    assert_eq!(loop_.tier(1.5), QualityTier(top.0 - 1), "block 3 steps down");
}

#[test]
fn relentless_overload_walks_all_the_way_down_to_the_floor() {
    let mut loop_ = ClosedLoop::new(PowerProfile::Desktop);
    // Far more hot blocks than tiers; the loop must bottom out at tier 0 and
    // stay there without underflowing.
    let top = QualityLadder::standard().max_tier();
    let mut last = top;
    for _ in 0..60 {
        let report = loop_.step(2.0);
        assert!(report.tier.0 <= last.0, "tier is monotonically non-increasing");
        // Desktop never power-caps, so the reason is purely load-driven: it is
        // `None` only while the loop still sits at full richness (inside the
        // dwell window), and `CpuBudget` for every block it has stepped down.
        let expected = if report.tier.0 == top.0 {
            DegradeReason::None
        } else {
            DegradeReason::CpuBudget
        };
        assert_eq!(report.reason, expected);
        last = report.tier;
    }
    assert_eq!(last, QualityTier(0), "relentless overload pins the floor");
}

#[test]
fn recovery_climbs_back_up_after_the_dwell_window() {
    let mut loop_ = ClosedLoop::new(PowerProfile::Desktop);
    // Drive to the floor first.
    for _ in 0..60 {
        loop_.step(2.0);
    }
    assert_eq!(loop_.tier(2.0), QualityTier(0));

    // Now go idle. Two cool blocks hold, the third raises one tier.
    assert_eq!(loop_.tier(0.1), QualityTier(0), "cool block 1 holds");
    assert_eq!(loop_.tier(0.1), QualityTier(0), "cool block 2 holds");
    assert_eq!(loop_.tier(0.1), QualityTier(1), "cool block 3 raises");
}

#[test]
fn dead_band_load_holds_the_tier_perfectly_steady() {
    let mut loop_ = ClosedLoop::new(PowerProfile::Desktop);
    // Step down once so we are not pinned at the ceiling.
    for _ in 0..3 {
        loop_.step(1.5);
    }
    let settled = loop_.tier(0.7); // 0.7 is inside the [0.6, 0.85] dead band
    for _ in 0..40 {
        assert_eq!(
            loop_.tier(0.7),
            settled,
            "a load parked in the dead band must never move the tier"
        );
    }
}

#[test]
fn power_profile_caps_the_ceiling_with_a_powercap_reason() {
    let mut loop_ = ClosedLoop::new(PowerProfile::MobileLow);
    let cap = PowerProfile::MobileLow.constraints().max_quality_tier;

    // Idle forever: the loop's wish climbs to the ladder max, but the emitted
    // tier is pinned at the power cap and the reason names the cap.
    let mut report = loop_.step(0.1);
    for _ in 0..40 {
        report = loop_.step(0.1);
    }
    assert_eq!(report.tier, cap, "emitted tier is clamped to the power cap");
    assert_eq!(report.reason, DegradeReason::PowerCap);
    assert!(report.tier.0 < QualityLadder::standard().max_tier().0);
}

#[test]
fn idle_desktop_rests_at_full_quality_with_no_degrade_reason() {
    let mut loop_ = ClosedLoop::new(PowerProfile::Desktop);
    let mut report = loop_.step(0.1);
    for _ in 0..20 {
        report = loop_.step(0.1);
    }
    assert_eq!(report.tier, QualityLadder::standard().max_tier());
    assert_eq!(report.reason, DegradeReason::None);
    // At full quality nothing is saved.
    assert!(report.savings.mean().abs() < 1e-6);
}

#[test]
fn richer_tiers_cost_more_keep_more_voices_and_save_less() {
    let ladder = QualityLadder::standard();
    let top = ladder.max_tier().0;
    for t in 1..=top {
        let lo = ladder.profile(QualityTier(t - 1));
        let hi = ladder.profile(QualityTier(t));
        assert!(
            hi.aggregate_cost() >= lo.aggregate_cost(),
            "aggregate cost must not decrease as tier rises ({} -> {})",
            t - 1,
            t
        );
        assert!(
            ladder.quality_fraction(QualityTier(t)) > ladder.quality_fraction(QualityTier(t - 1)),
            "quality fraction must strictly increase with tier"
        );
        assert!(
            hi.virtualization_threshold <= lo.virtualization_threshold,
            "a richer tier must keep at least as many voices audible"
        );
    }
}

#[test]
fn extreme_and_non_finite_inputs_never_leave_the_valid_range() {
    let mut loop_ = ClosedLoop::new(PowerProfile::Console);
    let cap = PowerProfile::Console.constraints().max_quality_tier.0;
    let garbage = [
        Sample::INFINITY,
        Sample::NEG_INFINITY,
        Sample::NAN,
        -5.0,
        1e30,
        0.0,
        9999.0,
    ];
    for (i, &c) in garbage.iter().cycle().take(200).enumerate() {
        let report = loop_.step(c);
        assert!(report.tier.0 <= cap, "tier {} exceeded cap on step {i}", report.tier.0);
        assert!(report.load.is_finite(), "load stayed finite on step {i}");
        assert!(report.profile.aggregate_cost().is_finite());
    }
}
