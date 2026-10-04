//! Pitch-shift anti-aliasing level-of-detail: how the governor scales the
//! quality of *variable-rate sample playback* (transposed one-shots, wavetable
//! voices) with CPU budget.
//!
//! The sibling [`crate::governor::lod`] module already scales an
//! `OversamplingTier` for *nonlinear* stages (waveshapers, saturators). That is
//! a different axis from the aliasing introduced by **resampling a sample at a
//! playback ratio other than `1.0`**: reading a buffer faster than it was
//! recorded scales its spectrum up, folding any content above `Nyquist / ratio`
//! back into the audible band. This module makes the policy for suppressing
//! that fold-back an explicit, budget-driven ladder with two orthogonal knobs:
//!
//! - **Interpolation grade** ([`AntiAliasInterp`]): linear (2-tap) for the
//!   cheapest voices, a short windowed-sinc kernel for the middle, and a long
//!   high-order sinc for near/important voices. Richer kernels have a steeper
//!   stop-band, so they both interpolate downward ratios more faithfully and
//!   reject the images of upward ratios further.
//! - **Wavetable mip bias** (`mip_bias`): for sources that ship pre-decimated
//!   band-limited copies (one octave of bandwidth halved per mip), the governor
//!   can bias *up* the chosen mip under budget pressure, trading brightness for
//!   a cheap, alias-free result when a long kernel is unaffordable.
//!
//! Each discrete [`QualityTier`] the governor walks maps onto a
//! [`ResampleLodProfile`] via the [`ResampleLodLadder`]. Like the main LOD
//! ladder this module only describes *parameters* a voice should request from
//! the resampler; it never selects a concrete resampler implementation or
//! touches graph topology. The host maps a profile onto the matching
//! `prism_audio_resample` grade (`Linear` / `Sinc` / `HighOrderSinc`).
//!
//! # Determinism
//!
//! Profiles are plain data, the ladder is a precomputed table, and mip
//! selection is an integer octave count derived by repeated doubling (no
//! transcendental `log2`), so every tier -> profile and (ratio, profile) -> mip
//! lookup is exact and reproducible on every platform.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The underlying
//! facts -- that upward transposition folds content above `Nyquist / ratio`,
//! that one mip octave halves bandwidth, and that longer sinc kernels have
//! steeper stop-bands -- are classical multirate signal-processing results.
//!
//! # Relationship
//!
//! Implements the design-section-42 open item "variable-rate `SamplePlayer`
//! anti-aliasing tiers (oversample factor vs wavetable mip) and their default
//! mapping onto the section-32 audio LOD". Keyed by the same
//! [`QualityTier`](crate::governor::lod::QualityTier) that
//! [`crate::governor::QualityGovernor`] steps under the
//! [`crate::governor::hysteresis`] policy, and complements the nonlinear-stage
//! `OversamplingTier` of [`crate::governor::lod`] rather than duplicating it.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use prism_audio_core::math::Sample;

use crate::governor::lod::QualityTier;

/// Largest mip bias the ladder or a caller may request. A bias of `4` darkens
/// playback by up to four octaves of pre-decimation, which is already past the
/// point of audible dullness and serves purely as a safety clamp.
pub const MAX_MIP_BIAS: u8 = 4;

/// Largest integer oversampling factor a resample LOD profile may request for
/// the variable-rate reader. Kept in lockstep with the richest nonlinear tier.
pub const MAX_RESAMPLE_OVERSAMPLE: u8 = 4;

/// Interpolation grade a variable-rate reader should request from the
/// resampler. Ordered cheapest-to-richest; the ordering is meaningful and
/// relied upon by the ladder's monotonicity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum AntiAliasInterp {
    /// Two-tap linear interpolation. Cheapest, but a gentle stop-band that both
    /// dulls downward ratios and lets upward images through, so it is paired
    /// with a positive `mip_bias` at the low tiers.
    Linear,
    /// Short windowed-sinc kernel: a good balance of stop-band rejection and
    /// cost for the middle tiers.
    Sinc,
    /// Long high-order windowed-sinc kernel: the steepest stop-band, reserved
    /// for near and important voices where transposition artefacts would be
    /// audible.
    HighOrderSinc,
}

impl Default for AntiAliasInterp {
    /// A safe middle ground: a short sinc kernel.
    #[inline]
    fn default() -> Self {
        AntiAliasInterp::Sinc
    }
}

impl AntiAliasInterp {
    /// Integer rank ordering the grades cheapest (`0`) to richest (`2`).
    #[must_use]
    #[inline]
    pub const fn rank(self) -> u8 {
        match self {
            AntiAliasInterp::Linear => 0,
            AntiAliasInterp::Sinc => 1,
            AntiAliasInterp::HighOrderSinc => 2,
        }
    }

    /// Representative filter length (tap count) of this grade. Used only to
    /// derive a relative cost; the resampler owns the real kernel.
    #[must_use]
    #[inline]
    pub const fn taps(self) -> u32 {
        match self {
            AntiAliasInterp::Linear => 2,
            AntiAliasInterp::Sinc => 16,
            AntiAliasInterp::HighOrderSinc => 64,
        }
    }

    /// Relative CPU cost in `(0, 1]`, normalised so `HighOrderSinc` is `1.0`.
    #[must_use]
    #[inline]
    pub fn relative_cost(self) -> Sample {
        self.taps() as Sample / AntiAliasInterp::HighOrderSinc.taps() as Sample
    }
}

/// The resample quality dimensions the governor scales for one tier.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ResampleLodProfile {
    /// Interpolation grade to request for variable-rate reads.
    pub interp: AntiAliasInterp,
    /// Extra octaves of mip pre-decimation to bias into [`required_mip`] for
    /// sources that ship band-limited mips. `0` picks the exact alias-free mip;
    /// higher values pick a darker, cheaper mip under budget pressure. Clamped
    /// to [`MAX_MIP_BIAS`].
    pub mip_bias: u8,
    /// Integer oversampling factor for the reader itself (`1` = none). Lets a
    /// rich tier push the resampler's own images further out before the final
    /// decimation. Clamped to [`MAX_RESAMPLE_OVERSAMPLE`].
    pub max_oversample: u8,
}

impl Default for ResampleLodProfile {
    /// A balanced default: short sinc, no mip bias, no reader oversampling.
    #[inline]
    fn default() -> Self {
        ResampleLodProfile {
            interp: AntiAliasInterp::Sinc,
            mip_bias: 0,
            max_oversample: 1,
        }
    }
}

impl ResampleLodProfile {
    /// Builds a profile with its fields clamped to their valid ranges
    /// (`mip_bias` to [`MAX_MIP_BIAS`], `max_oversample` to at least `1` and at
    /// most [`MAX_RESAMPLE_OVERSAMPLE`]).
    #[must_use]
    #[inline]
    pub fn new(interp: AntiAliasInterp, mip_bias: u8, max_oversample: u8) -> Self {
        ResampleLodProfile {
            interp,
            mip_bias: mip_bias.min(MAX_MIP_BIAS),
            max_oversample: max_oversample.clamp(1, MAX_RESAMPLE_OVERSAMPLE),
        }
    }

    /// Relative CPU cost of this profile in `(0, 4]`: the interpolation cost
    /// scaled by the reader oversampling factor. `mip_bias` does not add cost
    /// (a darker mip is strictly cheaper to read), so it is excluded here.
    #[must_use]
    #[inline]
    pub fn aggregate_cost(&self) -> Sample {
        self.interp.relative_cost() * Sample::from(self.max_oversample)
    }
}

/// A precomputed table mapping a discrete [`QualityTier`] (rung index) onto a
/// [`ResampleLodProfile`]. Rung `0` is cheapest; higher rungs are richer.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ResampleLodLadder {
    rungs: Vec<ResampleLodProfile>,
}

impl ResampleLodLadder {
    /// Builds a ladder from explicit rungs (cheapest first). Returns `None` if
    /// `rungs` is empty, since an empty ladder has no profile to serve.
    #[must_use]
    pub fn from_profiles(rungs: Vec<ResampleLodProfile>) -> Option<Self> {
        if rungs.is_empty() {
            None
        } else {
            Some(ResampleLodLadder { rungs })
        }
    }

    /// The standard five-rung ladder aligned with
    /// [`crate::governor::lod::QualityLadder::standard`]: linear + a safety mip
    /// bias at the bottom, climbing through a short sinc to a high-order sinc
    /// with reader oversampling at the top. Quality is non-decreasing and
    /// `aggregate_cost` is non-decreasing across the rungs.
    #[must_use]
    pub fn standard() -> Self {
        ResampleLodLadder {
            rungs: vec![
                // Tier 0: cheapest -- linear read, lean on a darker mip to hide
                // images cheaply.
                ResampleLodProfile::new(AntiAliasInterp::Linear, 1, 1),
                // Tier 1: still linear, one octave of mip safety.
                ResampleLodProfile::new(AntiAliasInterp::Linear, 1, 1),
                // Tier 2: short sinc, exact mip.
                ResampleLodProfile::new(AntiAliasInterp::Sinc, 0, 1),
                // Tier 3: short sinc with 2x reader oversampling.
                ResampleLodProfile::new(AntiAliasInterp::Sinc, 0, 2),
                // Tier 4: richest -- high-order sinc with 2x oversampling.
                ResampleLodProfile::new(AntiAliasInterp::HighOrderSinc, 0, 2),
            ],
        }
    }

    /// Number of rungs. Always at least `1` for a ladder built via the
    /// constructors.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.rungs.len()
    }

    /// Whether the ladder has no rungs. Only reachable on a default-constructed
    /// value; the public constructors reject empty ladders.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rungs.is_empty()
    }

    /// The highest valid tier index (`len - 1`), saturating to `0` on an empty
    /// ladder.
    #[must_use]
    #[inline]
    pub fn max_tier(&self) -> QualityTier {
        QualityTier(self.rungs.len().saturating_sub(1) as u8)
    }

    /// Clamps an arbitrary tier into the ladder's valid range.
    #[must_use]
    #[inline]
    pub fn clamp_tier(&self, tier: QualityTier) -> QualityTier {
        QualityTier(tier.0.min(self.max_tier().0))
    }

    /// The profile for `tier`, clamped into range. Returns the richest rung for
    /// out-of-range tiers so a mis-sized tier never panics or under-filters.
    #[must_use]
    pub fn profile(&self, tier: QualityTier) -> ResampleLodProfile {
        let idx = self.clamp_tier(tier).0 as usize;
        self.rungs.get(idx).copied().unwrap_or_default()
    }
}

impl Default for ResampleLodLadder {
    /// The standard five-rung ladder.
    #[inline]
    fn default() -> Self {
        ResampleLodLadder::standard()
    }
}

/// Guard: ratios whose absolute difference from `1.0` is below this are treated
/// as unity (bit-exact passthrough, no resampling).
const UNITY_EPS: Sample = 1.0e-6;

/// Returns the mip index a variable-rate read should use for `pitch_ratio`
/// under `profile`, given a source with `num_mips` band-limited mips (mip `0`
/// = full bandwidth, each higher mip halving bandwidth / one octave).
///
/// Upward ratios (`> 1`) scale the source spectrum up and would fold content
/// above `Nyquist / ratio` back into band; the smallest mip whose halved
/// bandwidth clears that fold-back is `ceil(log2(ratio))` octaves, computed
/// here by repeated doubling for exactness. The profile's `mip_bias` is added
/// on top for an extra cheap safety margin. Downward ratios (`<= 1`) do not
/// alias, so mip `0` (full bandwidth) is used with no bias. The result is
/// clamped to the available mips. A non-finite or non-positive ratio is treated
/// as unity.
#[must_use]
pub fn required_mip(num_mips: usize, pitch_ratio: Sample, profile: &ResampleLodProfile) -> usize {
    if num_mips == 0 {
        return 0;
    }
    let ratio = if pitch_ratio.is_finite() && pitch_ratio > 0.0 {
        pitch_ratio
    } else {
        1.0
    };
    let top = num_mips - 1;
    if ratio <= 1.0 + UNITY_EPS {
        return 0;
    }
    // Smallest octave count whose doubling reaches or exceeds the ratio.
    let mut octaves: usize = 0;
    let mut bound: Sample = 1.0;
    while bound < ratio && octaves < 32 {
        bound *= 2.0;
        octaves += 1;
    }
    (octaves + profile.mip_bias as usize).min(top)
}

/// Returns the interpolation grade a variable-rate read should request for
/// `pitch_ratio` under `profile`. At (near-)unity ratio the read is a bit-exact
/// passthrough, so the cheapest [`AntiAliasInterp::Linear`] grade is returned
/// regardless of tier; otherwise the profile's grade is used.
#[must_use]
pub fn recommended_interp(pitch_ratio: Sample, profile: &ResampleLodProfile) -> AntiAliasInterp {
    if pitch_ratio.is_finite() && (pitch_ratio - 1.0).abs() <= UNITY_EPS {
        AntiAliasInterp::Linear
    } else {
        profile.interp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interp_rank_is_ordered() {
        assert!(AntiAliasInterp::Linear < AntiAliasInterp::Sinc);
        assert!(AntiAliasInterp::Sinc < AntiAliasInterp::HighOrderSinc);
        assert_eq!(AntiAliasInterp::Linear.rank(), 0);
        assert_eq!(AntiAliasInterp::HighOrderSinc.rank(), 2);
    }

    #[test]
    fn interp_cost_is_ordered_and_normalised() {
        let lin = AntiAliasInterp::Linear.relative_cost();
        let sinc = AntiAliasInterp::Sinc.relative_cost();
        let hi = AntiAliasInterp::HighOrderSinc.relative_cost();
        assert!(lin < sinc && sinc < hi);
        assert!((hi - 1.0).abs() < 1.0e-6);
        assert!(lin > 0.0);
    }

    #[test]
    fn profile_new_clamps_fields() {
        let p = ResampleLodProfile::new(AntiAliasInterp::Linear, 99, 99);
        assert_eq!(p.mip_bias, MAX_MIP_BIAS);
        assert_eq!(p.max_oversample, MAX_RESAMPLE_OVERSAMPLE);
        let q = ResampleLodProfile::new(AntiAliasInterp::Sinc, 0, 0);
        assert_eq!(q.max_oversample, 1);
    }

    #[test]
    fn standard_ladder_shape() {
        let ladder = ResampleLodLadder::standard();
        assert_eq!(ladder.len(), 5);
        assert!(!ladder.is_empty());
        assert_eq!(ladder.max_tier(), QualityTier(4));
    }

    #[test]
    fn standard_ladder_is_monotonic() {
        let ladder = ResampleLodLadder::standard();
        let mut prev = ladder.profile(QualityTier(0));
        for t in 1..=4u8 {
            let cur = ladder.profile(QualityTier(t));
            assert!(cur.interp.rank() >= prev.interp.rank());
            assert!(cur.mip_bias <= prev.mip_bias);
            assert!(cur.max_oversample >= prev.max_oversample);
            assert!(cur.aggregate_cost() >= prev.aggregate_cost());
            prev = cur;
        }
    }

    #[test]
    fn clamp_and_profile_handle_out_of_range() {
        let ladder = ResampleLodLadder::standard();
        assert_eq!(ladder.clamp_tier(QualityTier(200)), QualityTier(4));
        assert_eq!(ladder.profile(QualityTier(200)), ladder.profile(QualityTier(4)));
    }

    #[test]
    fn from_profiles_rejects_empty() {
        assert!(ResampleLodLadder::from_profiles(Vec::new()).is_none());
        let one = ResampleLodLadder::from_profiles(vec![ResampleLodProfile::default()]);
        assert!(one.is_some());
    }

    #[test]
    fn required_mip_unity_is_zero() {
        let p = ResampleLodProfile::new(AntiAliasInterp::Sinc, 0, 1);
        assert_eq!(required_mip(8, 1.0, &p), 0);
    }

    #[test]
    fn required_mip_octaves_are_ceil_log2() {
        let p = ResampleLodProfile::new(AntiAliasInterp::Sinc, 0, 1);
        assert_eq!(required_mip(8, 2.0, &p), 1);
        assert_eq!(required_mip(8, 4.0, &p), 2);
        assert_eq!(required_mip(8, 8.0, &p), 3);
        // 1.5x upshift is within one octave -> conservative ceil picks mip 1.
        assert_eq!(required_mip(8, 1.5, &p), 1);
        // Just over two octaves -> mip 3.
        assert_eq!(required_mip(8, 4.1, &p), 3);
    }

    #[test]
    fn required_mip_applies_bias_upward_only() {
        let biased = ResampleLodProfile::new(AntiAliasInterp::Linear, 2, 1);
        // Upward: octaves (1 for 2.0) + bias 2 = 3.
        assert_eq!(required_mip(8, 2.0, &biased), 3);
        // Downward: no aliasing, no bias.
        assert_eq!(required_mip(8, 0.5, &biased), 0);
        assert_eq!(required_mip(8, 0.25, &biased), 0);
    }

    #[test]
    fn required_mip_clamps_to_available_mips() {
        let p = ResampleLodProfile::new(AntiAliasInterp::Sinc, 0, 1);
        // 16x = 4 octaves, but only 3 mips exist (indices 0..=2).
        assert_eq!(required_mip(3, 16.0, &p), 2);
        // Single mip -> always 0.
        assert_eq!(required_mip(1, 100.0, &p), 0);
        // Zero mips -> guarded to 0.
        assert_eq!(required_mip(0, 100.0, &p), 0);
    }

    #[test]
    fn required_mip_sanitises_bad_ratio() {
        let p = ResampleLodProfile::new(AntiAliasInterp::Sinc, 0, 1);
        assert_eq!(required_mip(8, Sample::NAN, &p), 0);
        assert_eq!(required_mip(8, Sample::INFINITY, &p), 0);
        assert_eq!(required_mip(8, -2.0, &p), 0);
        assert_eq!(required_mip(8, 0.0, &p), 0);
    }

    #[test]
    fn recommended_interp_unity_is_linear() {
        let p = ResampleLodProfile::new(AntiAliasInterp::HighOrderSinc, 0, 2);
        assert_eq!(recommended_interp(1.0, &p), AntiAliasInterp::Linear);
        assert_eq!(recommended_interp(1.000_000_1, &p), AntiAliasInterp::Linear);
    }

    #[test]
    fn recommended_interp_nonunity_uses_profile() {
        let p = ResampleLodProfile::new(AntiAliasInterp::HighOrderSinc, 0, 2);
        assert_eq!(recommended_interp(2.0, &p), AntiAliasInterp::HighOrderSinc);
        assert_eq!(recommended_interp(0.5, &p), AntiAliasInterp::HighOrderSinc);
    }

    #[test]
    fn default_profile_and_ladder() {
        assert_eq!(ResampleLodProfile::default().interp, AntiAliasInterp::Sinc);
        assert_eq!(ResampleLodLadder::default(), ResampleLodLadder::standard());
    }
}
