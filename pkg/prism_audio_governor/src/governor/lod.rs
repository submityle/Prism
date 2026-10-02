//! Audio level-of-detail: the quality dimensions the governor scales.
//!
//! Just as a renderer swaps high-poly meshes for billboards at distance, the
//! audio engine scales *quality dimensions* with available CPU budget. This
//! module defines those dimensions as explicit enums and bundles a snapshot of
//! them into a [`LodProfile`]. A [`QualityLadder`] maps a discrete quality tier
//! (`0` = cheapest, higher = richer) onto a profile, and the governor simply
//! walks that ladder up and down.
//!
//! The dimensions, mirroring design section 32, are:
//!
//! - **Oversampling** for nonlinear/waveshaping stages: `4x` -> `2x` -> `1x`.
//! - **Reverb quality**: convolution partition size, feedback-delay-network
//!   line count, early-reflection taps -- bundled into coarse tiers.
//! - **Spatialisation**: HRTF binaural for near/important sources degrading to
//!   cheap VBAP/stereo panning, with an Ambisonic (HOA) order that drops with
//!   budget.
//! - **Modulation/automation rate**: per-sample control for the richest tier,
//!   per-block control to save cycles on secondary voices.
//! - **Virtualisation threshold**: the effective-importance level below which a
//!   voice is culled to virtual (design section 25); it rises as budget
//!   tightens so more low-contribution voices go silent.
//!
//! Only parameters are described here -- never graph topology. The governor
//! applies a profile by retuning existing nodes, which is click-free; topology
//! hot-swaps are left to explicit recompilation.
//!
//! # Determinism
//!
//! Profiles are plain data and the ladder is a precomputed table, so tier ->
//! profile lookups are exact and reproducible.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! [`QualityTier`] is stepped by [`crate::governor::QualityGovernor`] under the
//! hysteresis policy of [`crate::governor::hysteresis`]; the resulting
//! [`LodProfile`] carries the virtualisation threshold compared against the
//! effective importance from [`crate::governor::importance`], and its cost
//! estimate feeds [`crate::governor::report`].

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use prism_audio_core::math::{Sample, lerp};
use prism_audio_core::voice::Importance;

/// Anti-aliasing oversampling factor for nonlinear stages (waveshapers,
/// saturators). Higher factors push aliasing products further out of band at
/// proportionally higher cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum OversamplingTier {
    /// No oversampling: process at the base rate (cheapest).
    X1,
    /// Double-rate oversampling.
    X2,
    /// Quadruple-rate oversampling (richest, used for near/important voices).
    X4,
}

impl OversamplingTier {
    /// Returns the integer oversampling factor (`1`, `2`, or `4`).
    #[must_use]
    #[inline]
    pub const fn factor(self) -> u32 {
        match self {
            OversamplingTier::X1 => 1,
            OversamplingTier::X2 => 2,
            OversamplingTier::X4 => 4,
        }
    }

    /// Relative CPU cost of this tier in `[0, 1]`, normalised so `X4` is `1.0`.
    #[must_use]
    #[inline]
    pub fn relative_cost(self) -> Sample {
        self.factor() as Sample / OversamplingTier::X4.factor() as Sample
    }
}

/// Coarse reverb-quality tier bundling convolution partition size, feedback
/// delay line count, and early-reflection tap budget into one knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ReverbQuality {
    /// Shared, cheap tail (few delay lines, no discrete early reflections).
    /// Used for distant or secondary sources that route to a common send.
    Low,
    /// Moderate line count and a handful of early reflections.
    Medium,
    /// Full-quality convolution/feedback-delay-network tail with dense early
    /// reflections for near, important sources.
    High,
}

impl ReverbQuality {
    /// Representative feedback-delay-network line count for this tier.
    #[must_use]
    #[inline]
    pub const fn delay_lines(self) -> u32 {
        match self {
            ReverbQuality::Low => 4,
            ReverbQuality::Medium => 8,
            ReverbQuality::High => 16,
        }
    }

    /// Representative discrete early-reflection tap count for this tier.
    #[must_use]
    #[inline]
    pub const fn early_reflection_taps(self) -> u32 {
        match self {
            ReverbQuality::Low => 0,
            ReverbQuality::Medium => 6,
            ReverbQuality::High => 18,
        }
    }

    /// Relative CPU cost of this tier in `[0, 1]`, normalised so `High` is
    /// `1.0`.
    #[must_use]
    #[inline]
    pub fn relative_cost(self) -> Sample {
        match self {
            ReverbQuality::Low => 0.25,
            ReverbQuality::Medium => 0.55,
            ReverbQuality::High => 1.0,
        }
    }
}

/// Spatialisation technique for a source, from expensive binaural down to cheap
/// amplitude panning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SpatialMode {
    /// Plain stereo amplitude panning (cheapest, no elevation cue).
    Stereo,
    /// Vector-base amplitude panning across the active speaker layout.
    Vbap,
    /// Head-related-transfer-function binaural convolution (richest).
    Hrtf,
}

impl SpatialMode {
    /// Relative CPU cost of this mode in `[0, 1]`, normalised so `Hrtf` is
    /// `1.0`.
    #[must_use]
    #[inline]
    pub fn relative_cost(self) -> Sample {
        match self {
            SpatialMode::Stereo => 0.1,
            SpatialMode::Vbap => 0.35,
            SpatialMode::Hrtf => 1.0,
        }
    }
}

/// Control-rate granularity for modulation and automation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ModulationRate {
    /// One control update per block (cheapest; adequate for secondary voices).
    PerBlock,
    /// One control update per sample (richest; smoothest modulation).
    PerSample,
}

impl ModulationRate {
    /// Relative CPU cost of this rate in `[0, 1]`, normalised so `PerSample` is
    /// `1.0`. The per-block figure assumes a representative 64-sample block.
    #[must_use]
    #[inline]
    pub fn relative_cost(self) -> Sample {
        match self {
            ModulationRate::PerBlock => 1.0 / 64.0,
            ModulationRate::PerSample => 1.0,
        }
    }
}

/// Highest Ambisonic order considered rich enough to never be exceeded by the
/// ladder. Third order is a common production ceiling for interactive audio.
pub const MAX_HOA_ORDER: u8 = 3;

/// A full snapshot of every audio LOD dimension at one quality tier.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LodProfile {
    /// Oversampling factor for nonlinear stages.
    pub oversampling: OversamplingTier,
    /// Reverb-quality tier.
    pub reverb: ReverbQuality,
    /// Spatialisation technique.
    pub spatial: SpatialMode,
    /// Higher-order-Ambisonics order (`0` = none, up to [`MAX_HOA_ORDER`]).
    pub hoa_order: u8,
    /// Modulation/automation control rate.
    pub modulation: ModulationRate,
    /// Effective-importance level below which a voice is virtualised. Rises as
    /// budget tightens so more low-contribution voices are culled.
    pub virtualization_threshold: Importance,
}

impl LodProfile {
    /// Returns the number of Ambisonic channels implied by [`hoa_order`], i.e.
    /// `(order + 1)^2`.
    ///
    /// [`hoa_order`]: LodProfile::hoa_order
    #[must_use]
    #[inline]
    pub fn ambisonic_channels(&self) -> u32 {
        let n = self.hoa_order as u32 + 1;
        n * n
    }

    /// Returns a coarse aggregate relative cost in `[0, 1]`, the mean of the
    /// per-dimension relative costs. Used by the profiler to visualise how much
    /// headroom the current tier buys. Higher means more expensive.
    #[must_use]
    pub fn aggregate_cost(&self) -> Sample {
        let hoa = self.hoa_order as Sample / MAX_HOA_ORDER as Sample;
        let sum = self.oversampling.relative_cost()
            + self.reverb.relative_cost()
            + self.spatial.relative_cost()
            + self.modulation.relative_cost()
            + hoa;
        sum / 5.0
    }
}

/// An ordered table of [`LodProfile`]s indexed by quality tier.
///
/// Tier `0` is the cheapest survivable configuration; the highest index is the
/// richest. The governor holds a current tier and steps it within `[0, len-1]`.
/// The table is built once, off the audio thread.
#[derive(Debug, Clone)]
pub struct QualityLadder {
    rungs: Vec<LodProfile>,
}

/// A quality tier: an index into a [`QualityLadder`]. Higher is richer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct QualityTier(pub u8);

impl QualityLadder {
    /// Builds a ladder from an explicit list of profiles, ordered cheapest
    /// first.
    ///
    /// Returns `None` if the list is empty, since a ladder must have at least
    /// one rung for the governor to rest on.
    #[must_use]
    pub fn from_profiles(rungs: Vec<LodProfile>) -> Option<Self> {
        if rungs.is_empty() {
            None
        } else {
            Some(Self { rungs })
        }
    }

    /// Builds the default five-rung ladder spanning the full quality range,
    /// from an all-cheap tier 0 to an all-rich tier 4.
    ///
    /// The virtualisation threshold falls monotonically as quality rises, so
    /// richer tiers keep more voices audible.
    #[must_use]
    pub fn standard() -> Self {
        let rungs = vec![
            // Tier 0: survival mode -- everything cheap, cull aggressively.
            LodProfile {
                oversampling: OversamplingTier::X1,
                reverb: ReverbQuality::Low,
                spatial: SpatialMode::Stereo,
                hoa_order: 0,
                modulation: ModulationRate::PerBlock,
                virtualization_threshold: 0.5,
            },
            // Tier 1.
            LodProfile {
                oversampling: OversamplingTier::X1,
                reverb: ReverbQuality::Low,
                spatial: SpatialMode::Vbap,
                hoa_order: 1,
                modulation: ModulationRate::PerBlock,
                virtualization_threshold: 0.35,
            },
            // Tier 2.
            LodProfile {
                oversampling: OversamplingTier::X2,
                reverb: ReverbQuality::Medium,
                spatial: SpatialMode::Vbap,
                hoa_order: 2,
                modulation: ModulationRate::PerBlock,
                virtualization_threshold: 0.2,
            },
            // Tier 3.
            LodProfile {
                oversampling: OversamplingTier::X2,
                reverb: ReverbQuality::High,
                spatial: SpatialMode::Hrtf,
                hoa_order: 2,
                modulation: ModulationRate::PerSample,
                virtualization_threshold: 0.1,
            },
            // Tier 4: reference quality -- keep nearly everything audible.
            LodProfile {
                oversampling: OversamplingTier::X4,
                reverb: ReverbQuality::High,
                spatial: SpatialMode::Hrtf,
                hoa_order: 3,
                modulation: ModulationRate::PerSample,
                virtualization_threshold: 0.03,
            },
        ];
        Self { rungs }
    }

    /// Returns the number of rungs (always at least `1`).
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.rungs.len()
    }

    /// Returns `false`; a ladder always has at least one rung. Present so the
    /// type satisfies the `len`/`is_empty` convention.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Returns the highest valid tier index.
    #[must_use]
    #[inline]
    pub fn max_tier(&self) -> QualityTier {
        QualityTier((self.rungs.len() - 1) as u8)
    }

    /// Clamps an arbitrary tier into the ladder's valid range.
    #[must_use]
    #[inline]
    pub fn clamp_tier(&self, tier: QualityTier) -> QualityTier {
        let hi = (self.rungs.len() - 1) as u8;
        QualityTier(tier.0.min(hi))
    }

    /// Returns the profile at a tier, clamping out-of-range indices to the
    /// nearest valid rung so the lookup is always defined.
    #[must_use]
    #[inline]
    pub fn profile(&self, tier: QualityTier) -> LodProfile {
        let idx = self.clamp_tier(tier).0 as usize;
        self.rungs[idx]
    }

    /// Returns the fraction of maximum quality represented by a tier, in
    /// `[0, 1]`, where the top rung is `1.0`. Interpolates linearly across the
    /// rung indices for smooth profiler readouts.
    #[must_use]
    pub fn quality_fraction(&self, tier: QualityTier) -> Sample {
        if self.rungs.len() == 1 {
            return 1.0;
        }
        let idx = self.clamp_tier(tier).0 as Sample;
        let top = (self.rungs.len() - 1) as Sample;
        lerp(0.0, 1.0, idx / top)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn oversampling_factors() {
        assert_eq!(OversamplingTier::X1.factor(), 1);
        assert_eq!(OversamplingTier::X2.factor(), 2);
        assert_eq!(OversamplingTier::X4.factor(), 4);
        assert!((OversamplingTier::X4.relative_cost() - 1.0).abs() < EPS);
        assert!(OversamplingTier::X1.relative_cost() < OversamplingTier::X2.relative_cost());
    }

    #[test]
    fn reverb_cost_is_monotonic() {
        assert!(ReverbQuality::Low.relative_cost() < ReverbQuality::Medium.relative_cost());
        assert!(ReverbQuality::Medium.relative_cost() < ReverbQuality::High.relative_cost());
        assert!(ReverbQuality::Low.delay_lines() < ReverbQuality::High.delay_lines());
    }

    #[test]
    fn spatial_cost_is_ordered() {
        assert!(SpatialMode::Stereo.relative_cost() < SpatialMode::Vbap.relative_cost());
        assert!(SpatialMode::Vbap.relative_cost() < SpatialMode::Hrtf.relative_cost());
    }

    #[test]
    fn modulation_rate_cost() {
        assert!(ModulationRate::PerBlock.relative_cost() < ModulationRate::PerSample.relative_cost());
    }

    #[test]
    fn ambisonic_channel_count() {
        let mut p = QualityLadder::standard().profile(QualityTier(4));
        assert_eq!(p.ambisonic_channels(), 16); // (3+1)^2
        p.hoa_order = 1;
        assert_eq!(p.ambisonic_channels(), 4);
        p.hoa_order = 0;
        assert_eq!(p.ambisonic_channels(), 1);
    }

    #[test]
    fn standard_ladder_quality_increases_with_tier() {
        let ladder = QualityLadder::standard();
        assert_eq!(ladder.len(), 5);
        assert!(!ladder.is_empty());
        let mut prev = ladder.profile(QualityTier(0)).aggregate_cost();
        for t in 1..=4u8 {
            let cost = ladder.profile(QualityTier(t)).aggregate_cost();
            assert!(cost > prev - EPS, "tier {t} cost {cost} prev {prev}");
            prev = cost;
        }
    }

    #[test]
    fn virtualization_threshold_falls_with_quality() {
        let ladder = QualityLadder::standard();
        let lo = ladder.profile(QualityTier(0)).virtualization_threshold;
        let hi = ladder.profile(QualityTier(4)).virtualization_threshold;
        assert!(hi < lo);
    }

    #[test]
    fn tier_clamping() {
        let ladder = QualityLadder::standard();
        assert_eq!(ladder.max_tier(), QualityTier(4));
        assert_eq!(ladder.clamp_tier(QualityTier(99)), QualityTier(4));
        // Out-of-range lookups resolve to the top rung rather than panicking.
        let p = ladder.profile(QualityTier(200));
        assert_eq!(p, ladder.profile(QualityTier(4)));
    }

    #[test]
    fn quality_fraction_spans_unit_interval() {
        let ladder = QualityLadder::standard();
        assert!(ladder.quality_fraction(QualityTier(0)).abs() < EPS);
        assert!((ladder.quality_fraction(QualityTier(4)) - 1.0).abs() < EPS);
        assert!((ladder.quality_fraction(QualityTier(2)) - 0.5).abs() < EPS);
    }

    #[test]
    fn empty_ladder_rejected() {
        assert!(QualityLadder::from_profiles(Vec::new()).is_none());
        let one = vec![QualityLadder::standard().profile(QualityTier(0))];
        let ladder = QualityLadder::from_profiles(one).unwrap();
        assert!((ladder.quality_fraction(QualityTier(0)) - 1.0).abs() < EPS);
    }
}
