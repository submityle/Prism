//! Observability: the snapshot the governor publishes every block boundary.
//!
//! Design section 32 requires the governor to be observable -- the current
//! tier, *why* it degraded, and how much each LOD dimension is saving must be
//! visible to the profiler (design section 26) so sound designers and engineers
//! can tune. This module defines the plain-data [`GovernorReport`] the governor
//! emits and the [`DegradeReason`] enumerating why the current tier is below
//! the ceiling.
//!
//! Per-dimension savings are expressed as `1.0 - relative_cost`: a dimension
//! running at full quality saves `0.0`, a dimension at its cheapest saves close
//! to `1.0`.
//!
//! # Determinism
//!
//! Everything here is derived by pure functions from a [`LodProfile`], so the
//! report is reproducible given the same tier.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Summarises the [`crate::governor::lod::LodProfile`] selected by
//! [`crate::governor::QualityGovernor`] and the load from
//! [`crate::governor::budget`]; it is the structure the telemetry ring (design
//! section 21) forwards to the profiler.

use prism_audio_core::math::Sample;

use crate::governor::lod::{LodProfile, QualityTier};

/// Why the governor is sitting below the maximum quality tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DegradeReason {
    /// The governor is at the highest tier allowed; nothing is being held back.
    None,
    /// Sustained CPU-budget pressure lowered the tier (the closed loop).
    CpuBudget,
    /// The active platform power profile caps the tier below the loop's wish.
    PowerCap,
}

/// Normalised per-dimension savings, each in `[0, 1]`, where `0.0` means the
/// dimension is at full quality and higher means more CPU is being saved.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LodSavings {
    /// Savings from reduced oversampling.
    pub oversampling: Sample,
    /// Savings from reduced reverb quality.
    pub reverb: Sample,
    /// Savings from cheaper spatialisation.
    pub spatial: Sample,
    /// Savings from a coarser modulation/automation rate.
    pub modulation: Sample,
}

impl LodSavings {
    /// Derives the savings implied by a profile relative to full quality.
    #[must_use]
    pub fn from_profile(profile: &LodProfile) -> Self {
        Self {
            oversampling: 1.0 - profile.oversampling.relative_cost(),
            reverb: 1.0 - profile.reverb.relative_cost(),
            spatial: 1.0 - profile.spatial.relative_cost(),
            modulation: 1.0 - profile.modulation.relative_cost(),
        }
    }

    /// Returns the mean savings across the four dimensions, a single headline
    /// number for the profiler.
    #[must_use]
    pub fn mean(&self) -> Sample {
        (self.oversampling + self.reverb + self.spatial + self.modulation) / 4.0
    }
}

/// A complete snapshot of the governor's state at a block boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GovernorReport {
    /// The quality tier currently in force.
    pub tier: QualityTier,
    /// The smoothed CPU-budget load that drove the latest decision.
    pub load: Sample,
    /// Why the tier is below maximum (or [`DegradeReason::None`]).
    pub reason: DegradeReason,
    /// The full LOD profile the tier resolves to.
    pub profile: LodProfile,
    /// Per-dimension savings versus full quality.
    pub savings: LodSavings,
}

impl GovernorReport {
    /// Builds a report from the current tier, load, reason, and profile,
    /// deriving the savings automatically.
    #[must_use]
    pub fn new(
        tier: QualityTier,
        load: Sample,
        reason: DegradeReason,
        profile: LodProfile,
    ) -> Self {
        Self {
            tier,
            load,
            reason,
            profile,
            savings: LodSavings::from_profile(&profile),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::governor::lod::{OversamplingTier, QualityLadder, ReverbQuality, SpatialMode};

    const EPS: Sample = 1e-6;

    #[test]
    fn full_quality_saves_nothing() {
        let profile = QualityLadder::standard().profile(QualityTier(4));
        let s = LodSavings::from_profile(&profile);
        // Tier 4 uses X4 oversampling, High reverb, HRTF, per-sample mod.
        assert!(s.oversampling.abs() < EPS);
        assert!(s.reverb.abs() < EPS);
        assert!(s.spatial.abs() < EPS);
        assert!(s.modulation.abs() < EPS);
    }

    #[test]
    fn cheapest_tier_saves_a_lot() {
        let profile = QualityLadder::standard().profile(QualityTier(0));
        let s = LodSavings::from_profile(&profile);
        assert!(s.oversampling > 0.5);
        assert!(s.spatial > 0.5);
        assert!(s.mean() > 0.5);
    }

    #[test]
    fn report_derives_savings() {
        let profile = LodProfile {
            oversampling: OversamplingTier::X1,
            reverb: ReverbQuality::Low,
            spatial: SpatialMode::Stereo,
            hoa_order: 0,
            modulation: crate::governor::lod::ModulationRate::PerBlock,
            virtualization_threshold: 0.5,
        };
        let report = GovernorReport::new(QualityTier(0), 0.95, DegradeReason::CpuBudget, profile);
        assert_eq!(report.reason, DegradeReason::CpuBudget);
        assert!((report.load - 0.95).abs() < EPS);
        assert!(report.savings.spatial > 0.5);
    }
}
