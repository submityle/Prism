//! Continuous LOD cross-fade to eliminate hair popping.
//!
//! The LOD ladder in [`super::lod`] snaps a groom to a single tier at a hard
//! coverage threshold. Snapping the whole groom at once *pops*: strands or
//! cards appear or vanish in one frame, which reads as a flicker (design §4,
//! §11 "无 LOD pop"). This module spreads each switch across a small *coverage
//! band* around the threshold and dissolves between the two tiers with a
//! deterministic per-strand screen-door dither, so the transition is
//! imperceptible and stable frame to frame.
//!
//! The math is pure and deterministic (design §9): coverage plus thresholds map
//! to a [`HairLodTransition`], and a strand index plus a blend factor map to a
//! keep/drop decision via [`strand_survives_dither`], reusing the same
//! `splitmix64`-style hash as guide interpolation so the two stages agree on
//! per-strand identity. Nothing here allocates or samples a real random source.
//!
//! A transition also honors [`HairGroup::native_form`]: a card-authored groom
//! never cross-fades *into* strands it does not own, exactly as
//! [`super::lod::resolve_hair_lod`] clamps the discrete tier.

use super::interpolation::hash_to_unit;
use super::lod::HairLodThresholds;
use super::HairGroup;
use super::HairLodTier;

/// A resolved cross-fade between two adjacent LOD tiers.
///
/// When `from == to` there is no active transition and `blend` is `0`: the
/// groom renders wholly as that single tier. Otherwise `from` is the finer tier
/// being dissolved out and `to` is the coarser tier being faded in, with
/// `blend` running `0..=1` from fully-`from` to fully-`to`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairLodTransition {
    /// The finer tier (higher detail) being faded out.
    pub from: HairLodTier,
    /// The coarser tier (lower detail) being faded in.
    pub to: HairLodTier,
    /// Cross-fade position in `0..=1`: `0` is fully `from`, `1` is fully `to`.
    pub blend: f32,
}

impl HairLodTransition {
    /// A settled (non-transitioning) result at a single tier.
    #[must_use]
    pub fn settled(tier: HairLodTier) -> Self {
        Self {
            from: tier,
            to: tier,
            blend: 0.0,
        }
    }

    /// Returns `true` while a genuine cross-fade between two distinct tiers is
    /// in progress (the blend is strictly inside the open interval `(0, 1)`).
    #[must_use]
    pub fn is_cross_fading(self) -> bool {
        self.from != self.to && self.blend > 0.0 && self.blend < 1.0
    }

    /// The opacity weight of the coarser proxy (`to`) during the fade, in
    /// `0..=1`. Callers fade the card/mesh proxy in with this and fade the
    /// finer representation out with `1 - proxy_alpha()`.
    #[must_use]
    pub fn proxy_alpha(self) -> f32 {
        if self.from == self.to {
            0.0
        } else {
            self.blend
        }
    }
}

/// Resolves the cross-fade for a groom at a given screen coverage.
///
/// `band` is the half-width, in coverage units, of the dissolve region on each
/// side of a threshold; a non-positive `band` disables cross-fading and this
/// degrades to a hard tier pick. When `coverage` falls within `band` of the
/// nearest crossed threshold, the result cross-fades between the finer tier
/// (just above the threshold) and the coarser tier (just below it); otherwise
/// it is [`HairLodTransition::settled`] at the single selected tier.
///
/// Both tiers are clamped no finer than [`HairGroup::native_form`]. When that
/// clamp collapses the two tiers to the same value (for example a card-authored
/// groom near the strands/reduced boundary), the result settles with no visible
/// fade, so authoring intent always wins over the coverage band.
#[must_use]
pub fn resolve_hair_lod_transition(
    group: HairGroup,
    coverage: f32,
    thresholds: HairLodThresholds,
    band: f32,
) -> HairLodTransition {
    let native = group.native_form;
    let settled = |tier: HairLodTier| HairLodTransition::settled(tier.coarser_of(native));

    if band <= 0.0 || !band.is_finite() || !coverage.is_finite() {
        return settled(super::lod::select_hair_lod_tier(coverage, thresholds));
    }

    // Each threshold separates a finer tier (at/above it) from a coarser one
    // (below it). Find the nearest threshold whose band the coverage is inside.
    let boundaries = [
        (
            thresholds.reduced_strands_below,
            HairLodTier::Strands,
            HairLodTier::ReducedStrands,
        ),
        (
            thresholds.cards_below,
            HairLodTier::ReducedStrands,
            HairLodTier::Cards,
        ),
        (thresholds.mesh_below, HairLodTier::Cards, HairLodTier::Mesh),
    ];

    let mut best: Option<(f32, HairLodTier, HairLodTier, f32)> = None;
    for (threshold, finer, coarser) in boundaries {
        if !threshold.is_finite() {
            continue;
        }
        let distance = (coverage - threshold).abs();
        if distance >= band {
            continue;
        }
        // `blend` runs 0 at the finer edge (threshold + band) to 1 at the
        // coarser edge (threshold - band).
        let blend = ((threshold + band - coverage) / (2.0 * band)).clamp(0.0, 1.0);
        let is_closer = match best {
            Some((best_distance, ..)) => distance < best_distance,
            None => true,
        };
        if is_closer {
            best = Some((distance, finer, coarser, blend));
        }
    }

    match best {
        Some((_, finer, coarser, blend)) => {
            let from = finer.coarser_of(native);
            let to = coarser.coarser_of(native);
            if from == to {
                // native_form collapsed the band: no visible fade.
                HairLodTransition::settled(from)
            } else {
                HairLodTransition { from, to, blend }
            }
        }
        None => settled(super::lod::select_hair_lod_tier(coverage, thresholds)),
    }
}

/// Deterministic screen-door decision: does strand `strand_index` still draw as
/// the finer tier at cross-fade position `blend`?
///
/// A stable per-strand hash in `[0, 1)` is compared against `blend`: the strand
/// keeps drawing while its hash is at least `blend`. At `blend == 0` every
/// strand survives (fully finer); as `blend` rises toward `1` a growing,
/// deterministic subset drops out, so the finer representation dissolves evenly
/// rather than popping. `seed` scopes the pattern per groom so different grooms
/// dissolve independently. Out-of-range `blend` saturates.
#[must_use]
pub fn strand_survives_dither(seed: u32, strand_index: u32, blend: f32) -> bool {
    let b = blend.clamp(0.0, 1.0);
    // hash in [0,1); survive while hash >= blend so the kept fraction is
    // (1 - blend). At blend == 1 no strand survives (hash is always < 1).
    hash_to_unit(seed, strand_index) >= b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deformation::DeformationHandle;
    use crate::hair::HairGroupHandle;

    fn thresholds() -> HairLodThresholds {
        HairLodThresholds {
            reduced_strands_below: 0.6,
            cards_below: 0.3,
            mesh_below: 0.1,
        }
    }

    fn strand_group() -> HairGroup {
        HairGroup {
            handle: HairGroupHandle(1),
            guide_strand_count: 64,
            max_render_strands: 4096,
            segments_per_strand: 16,
            deformation: DeformationHandle(1),
            native_form: HairLodTier::Strands,
        }
    }

    #[test]
    fn far_from_threshold_settles_at_single_tier() {
        let t = resolve_hair_lod_transition(strand_group(), 0.8, thresholds(), 0.05);
        assert_eq!(t, HairLodTransition::settled(HairLodTier::Strands));
        assert!(!t.is_cross_fading());
        assert!((t.proxy_alpha() - 0.0).abs() < 1.0e-6);
    }

    #[test]
    fn exactly_on_threshold_is_half_blended() {
        // Coverage == reduced_strands_below (0.6) sits at the band center.
        let t = resolve_hair_lod_transition(strand_group(), 0.6, thresholds(), 0.1);
        assert_eq!(t.from, HairLodTier::Strands);
        assert_eq!(t.to, HairLodTier::ReducedStrands);
        assert!((t.blend - 0.5).abs() < 1.0e-6);
        assert!(t.is_cross_fading());
    }

    #[test]
    fn blend_runs_from_zero_at_finer_edge_to_one_at_coarser_edge() {
        let band = 0.1;
        // Just inside the finer edge of the band: blend ~ 0 (mostly `from`).
        let near_finer = 0.6 + band - 0.001;
        let a = resolve_hair_lod_transition(strand_group(), near_finer, thresholds(), band);
        assert_eq!(a.from, HairLodTier::Strands);
        assert_eq!(a.to, HairLodTier::ReducedStrands);
        assert!(a.blend < 0.01, "blend {} not near 0", a.blend);
        // Just inside the coarser edge of the band: blend ~ 1 (mostly `to`).
        let near_coarser = 0.6 - band + 0.001;
        let b = resolve_hair_lod_transition(strand_group(), near_coarser, thresholds(), band);
        assert!(b.blend > 0.99, "blend {} not near 1", b.blend);
        // At and beyond the band edge the groom settles at the coarser tier.
        let at_edge = resolve_hair_lod_transition(strand_group(), 0.6 - band, thresholds(), band);
        assert_eq!(
            at_edge,
            HairLodTransition::settled(HairLodTier::ReducedStrands)
        );
    }

    #[test]
    fn zero_band_degrades_to_hard_pick() {
        let t = resolve_hair_lod_transition(strand_group(), 0.6, thresholds(), 0.0);
        assert_eq!(t, HairLodTransition::settled(HairLodTier::Strands));
    }

    #[test]
    fn native_form_card_never_fades_into_strands() {
        let mut group = strand_group();
        group.native_form = HairLodTier::Cards;
        // Near the strands/reduced boundary a card-native groom must not fade
        // between strands it does not own; it settles at Cards.
        let t = resolve_hair_lod_transition(group, 0.6, thresholds(), 0.1);
        assert_eq!(t, HairLodTransition::settled(HairLodTier::Cards));
        assert!(!t.is_cross_fading());
    }

    #[test]
    fn native_form_card_still_fades_cards_to_mesh() {
        let mut group = strand_group();
        group.native_form = HairLodTier::Cards;
        // The cards/mesh boundary is coarser than Cards, so the fade survives.
        let t = resolve_hair_lod_transition(group, 0.1, thresholds(), 0.05);
        assert_eq!(t.from, HairLodTier::Cards);
        assert_eq!(t.to, HairLodTier::Mesh);
        assert!(t.is_cross_fading());
    }

    #[test]
    fn dither_keeps_all_at_zero_blend_and_none_at_full_blend() {
        for index in 0..256u32 {
            assert!(strand_survives_dither(7, index, 0.0));
            assert!(!strand_survives_dither(7, index, 1.0));
        }
    }

    #[test]
    fn dither_kept_fraction_tracks_blend() {
        // About (1 - blend) of strands should survive at an intermediate blend.
        let total = 10_000u32;
        let blend = 0.25;
        let kept = (0..total)
            .filter(|&i| strand_survives_dither(42, i, blend))
            .count();
        let expected = ((1.0 - blend) * total as f32) as usize;
        let tolerance = (total / 40) as usize; // 2.5% slack for hash noise.
        assert!(
            kept.abs_diff(expected) <= tolerance,
            "kept {kept}, expected ~{expected}"
        );
    }

    #[test]
    fn dither_is_deterministic() {
        for index in 0..64u32 {
            let a = strand_survives_dither(3, index, 0.5);
            let b = strand_survives_dither(3, index, 0.5);
            assert_eq!(a, b);
        }
    }
}
