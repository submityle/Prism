//! Performance-adaptive quality governor: the CPU-budget closed loop.
//!
//! This module assembles the governor from its parts: a
//! [`budget::BudgetTracker`] smooths per-block cost telemetry into a load, a
//! [`hysteresis::Hysteresis`] controller turns that load into raise/lower
//! decisions at block boundaries, a [`lod::QualityLadder`] maps the resulting
//! tier onto a concrete [`lod::LodProfile`], and a [`power::PowerConstraints`]
//! caps the tier for the active platform. The governor publishes a
//! [`report::GovernorReport`] every step for the profiler.
//!
//! The governor **only tunes parameters, never graph topology** (design section
//! 32): it chooses an LOD profile; applying that profile by retuning existing
//! nodes is the audio graph's job and is inherently click-free. Decisions are
//! made at block boundaries, off the real-time DSP hot path.
//!
//! # Submodules
//!
//! - [`budget`] -- CPU-budget telemetry smoothing.
//! - [`hysteresis`] -- dual-threshold, dwell-gated transition policy.
//! - [`lod`] -- audio LOD dimensions and the quality ladder.
//! - [`importance`] -- per-voice effective-importance scoring.
//! - [`power`] -- platform power-profile ceilings.
//! - [`report`] -- the observable snapshot.
//!
//! # Determinism
//!
//! Every component is deterministic, so a fixed telemetry sequence drives a
//! fixed tier trajectory, which is what makes the governor golden-testable.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Implements design section 32. Its virtualisation threshold and the
//! effective importance from [`importance`] feed the masking-aware
//! virtualisation and clustering of design section 33 (see [`crate::masking`]
//! and [`crate::clustering`]). It consumes telemetry from design section 21 and
//! publishes to the profiler of design section 26.

pub mod budget;
pub mod hysteresis;
pub mod importance;
pub mod lod;
pub mod power;
pub mod report;

use prism_audio_core::math::Sample;

use self::budget::BudgetTracker;
use self::hysteresis::{Hysteresis, Transition};
use self::lod::{LodProfile, QualityLadder, QualityTier};
use self::power::{PowerConstraints, PowerProfile};
use self::report::{DegradeReason, GovernorReport};

/// Tunable construction parameters for a [`QualityGovernor`].
#[derive(Debug, Clone)]
pub struct GovernorConfig {
    /// The quality ladder the governor walks.
    pub ladder: QualityLadder,
    /// Smoothing coefficient for the budget tracker (weight of the previous
    /// estimate), in `[0, 1)`.
    pub load_smoothing: Sample,
    /// Load at or below which quality may be raised.
    pub raise_at: Sample,
    /// Load at or above which quality is lowered.
    pub lower_at: Sample,
    /// Consecutive qualifying blocks required before a transition fires.
    pub hold_blocks: u32,
    /// Platform power profile capping the tier.
    pub power: PowerProfile,
    /// The tier the governor starts at (clamped to the ladder and power cap).
    pub start_tier: QualityTier,
}

impl Default for GovernorConfig {
    /// A balanced default: the standard five-rung ladder, moderate smoothing, a
    /// `[0.6, 0.9]` dead band with a three-block dwell, desktop power, starting
    /// at the top tier.
    fn default() -> Self {
        let ladder = QualityLadder::standard();
        let start_tier = ladder.max_tier();
        Self {
            ladder,
            load_smoothing: 0.7,
            raise_at: 0.6,
            lower_at: 0.9,
            hold_blocks: 3,
            power: PowerProfile::Desktop,
            start_tier,
        }
    }
}

/// The CPU-budget closed-loop quality governor (design section 32).
///
/// Call [`observe_block`](QualityGovernor::observe_block) once per block
/// boundary with the fraction of the budget the last block consumed; the
/// governor updates its tier under hysteresis and the power cap and returns the
/// published [`GovernorReport`].
#[derive(Debug, Clone)]
pub struct QualityGovernor {
    ladder: QualityLadder,
    tracker: BudgetTracker,
    hysteresis: Hysteresis,
    constraints: PowerConstraints,
    tier: QualityTier,
    reason: DegradeReason,
}

impl QualityGovernor {
    /// Builds a governor from a [`GovernorConfig`].
    #[must_use]
    pub fn new(config: GovernorConfig) -> Self {
        let constraints = config.power.constraints();
        let tracker = BudgetTracker::new(config.load_smoothing);
        let hysteresis = Hysteresis::new(config.raise_at, config.lower_at, config.hold_blocks);
        let tier = clamp_tier(&config.ladder, &constraints, config.start_tier);
        let reason = initial_reason(&config.ladder, &constraints, tier);
        Self {
            ladder: config.ladder,
            tracker,
            hysteresis,
            constraints,
            tier,
            reason,
        }
    }

    /// Builds a governor with [`GovernorConfig::default`].
    #[must_use]
    pub fn standard() -> Self {
        Self::new(GovernorConfig::default())
    }

    /// Observes one block's budget consumption ratio (render time / budget
    /// time) and advances the quality tier at this block boundary.
    ///
    /// Returns the fresh [`GovernorReport`]. The input is smoothed, run through
    /// the hysteresis policy, applied as a one-tier step, and clamped to both
    /// the ladder range and the power ceiling.
    pub fn observe_block(&mut self, block_cost_ratio: Sample) -> GovernorReport {
        let load = self.tracker.observe(block_cost_ratio);
        let transition = self.hysteresis.update(load);
        self.apply_transition(transition);
        self.report()
    }

    /// Applies a single hysteresis [`Transition`] to the current tier, clamped
    /// to the ladder and power ceiling, and recomputes the degrade reason.
    fn apply_transition(&mut self, transition: Transition) {
        let max_by_power = self.constraints.max_quality_tier.0;
        let max_by_ladder = self.ladder.max_tier().0;
        let ceiling = max_by_power.min(max_by_ladder);

        let next = match transition {
            Transition::Raise => self.tier.0.saturating_add(1),
            Transition::Lower => self.tier.0.saturating_sub(1),
            Transition::Hold => self.tier.0,
        };
        self.tier = QualityTier(next.min(ceiling));
        self.recompute_reason(ceiling);
    }

    /// Determines why the tier is where it is, for observability.
    fn recompute_reason(&mut self, ceiling: u8) {
        let ladder_top = self.ladder.max_tier().0;
        self.reason = if self.tier.0 >= ladder_top {
            // At the richest rung the ladder offers; nothing held back.
            DegradeReason::None
        } else if ceiling < ladder_top && self.tier.0 >= ceiling {
            // The power cap, not the loop, is the binding limit.
            DegradeReason::PowerCap
        } else {
            // The ladder could go higher but the budget loop keeps us down.
            DegradeReason::CpuBudget
        };
    }

    /// Switches the active platform power profile and re-clamps the tier.
    pub fn set_power_profile(&mut self, power: PowerProfile) {
        self.constraints = power.constraints();
        let ceiling = self
            .constraints
            .max_quality_tier
            .0
            .min(self.ladder.max_tier().0);
        self.tier = QualityTier(self.tier.0.min(ceiling));
        self.recompute_reason(ceiling);
    }

    /// Returns the current quality tier.
    #[must_use]
    #[inline]
    pub fn tier(&self) -> QualityTier {
        self.tier
    }

    /// Returns the LOD profile the current tier resolves to.
    #[must_use]
    #[inline]
    pub fn profile(&self) -> LodProfile {
        self.ladder.profile(self.tier)
    }

    /// Returns the current smoothed CPU-budget load.
    #[must_use]
    #[inline]
    pub fn load(&self) -> Sample {
        self.tracker.load()
    }

    /// Returns the active platform constraints.
    #[must_use]
    #[inline]
    pub fn constraints(&self) -> PowerConstraints {
        self.constraints
    }

    /// Returns the effective physical-voice ceiling: the power profile's cap.
    #[must_use]
    #[inline]
    pub fn max_physical_voices(&self) -> u32 {
        self.constraints.max_physical_voices
    }

    /// Builds and returns the current observable report.
    #[must_use]
    pub fn report(&self) -> GovernorReport {
        GovernorReport::new(self.tier, self.tracker.load(), self.reason, self.profile())
    }
}

/// Clamps a tier to both the ladder range and the power ceiling.
fn clamp_tier(
    ladder: &QualityLadder,
    constraints: &PowerConstraints,
    tier: QualityTier,
) -> QualityTier {
    let ceiling = constraints.max_quality_tier.0.min(ladder.max_tier().0);
    QualityTier(tier.0.min(ceiling))
}

/// Computes the degrade reason for a freshly constructed governor.
fn initial_reason(
    ladder: &QualityLadder,
    constraints: &PowerConstraints,
    tier: QualityTier,
) -> DegradeReason {
    let ladder_top = ladder.max_tier().0;
    let ceiling = constraints.max_quality_tier.0.min(ladder_top);
    if tier.0 >= ladder_top {
        DegradeReason::None
    } else if ceiling < ladder_top && tier.0 >= ceiling {
        DegradeReason::PowerCap
    } else {
        DegradeReason::CpuBudget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    fn config_fast() -> GovernorConfig {
        GovernorConfig {
            load_smoothing: 0.0, // follow input exactly for deterministic tests
            raise_at: 0.6,
            lower_at: 0.9,
            hold_blocks: 2,
            ..GovernorConfig::default()
        }
    }

    #[test]
    fn starts_at_requested_tier() {
        let g = QualityGovernor::new(GovernorConfig {
            start_tier: QualityTier(2),
            ..config_fast()
        });
        assert_eq!(g.tier(), QualityTier(2));
    }

    #[test]
    fn sustained_overload_lowers_quality() {
        let mut g = QualityGovernor::new(config_fast());
        assert_eq!(g.tier(), QualityTier(4));
        g.observe_block(1.2);
        let r = g.observe_block(1.2);
        assert_eq!(r.tier, QualityTier(3));
        assert_eq!(r.reason, DegradeReason::CpuBudget);
    }

    #[test]
    fn headroom_raises_quality_back() {
        let mut g = QualityGovernor::new(GovernorConfig {
            start_tier: QualityTier(1),
            ..config_fast()
        });
        g.observe_block(0.1);
        let r = g.observe_block(0.1);
        assert_eq!(r.tier, QualityTier(2));
    }

    #[test]
    fn tier_never_exceeds_power_cap() {
        let mut g = QualityGovernor::new(GovernorConfig {
            power: PowerProfile::MobileLow, // caps at tier 2
            start_tier: QualityTier(0),
            ..config_fast()
        });
        // Lots of headroom: try to climb well past the cap.
        for _ in 0..20 {
            g.observe_block(0.0);
        }
        assert_eq!(g.tier(), QualityTier(2));
        assert_eq!(g.report().reason, DegradeReason::PowerCap);
    }

    #[test]
    fn tier_floors_at_zero() {
        let mut g = QualityGovernor::new(GovernorConfig {
            start_tier: QualityTier(0),
            ..config_fast()
        });
        for _ in 0..20 {
            g.observe_block(2.0);
        }
        assert_eq!(g.tier(), QualityTier(0));
    }

    #[test]
    fn dead_band_is_stable() {
        let mut g = QualityGovernor::new(config_fast());
        let start = g.tier();
        for _ in 0..50 {
            g.observe_block(0.75); // inside [0.6, 0.9]
        }
        assert_eq!(g.tier(), start);
    }

    #[test]
    fn changing_power_profile_reclamps() {
        let mut g = QualityGovernor::new(config_fast());
        assert_eq!(g.tier(), QualityTier(4));
        g.set_power_profile(PowerProfile::MobileLow);
        assert_eq!(g.tier(), QualityTier(2));
        assert_eq!(g.report().reason, DegradeReason::PowerCap);
    }

    #[test]
    fn report_tracks_load() {
        let mut g = QualityGovernor::new(config_fast());
        let r = g.observe_block(0.5);
        assert!((r.load - 0.5).abs() < EPS);
    }

    #[test]
    fn profile_matches_tier() {
        let g = QualityGovernor::standard();
        assert_eq!(g.profile(), QualityLadder::standard().profile(g.tier()));
    }

    #[test]
    fn voice_ceiling_follows_power() {
        let g = QualityGovernor::new(GovernorConfig {
            power: PowerProfile::MobileHigh,
            ..config_fast()
        });
        assert_eq!(g.max_physical_voices(), 96);
    }
}
