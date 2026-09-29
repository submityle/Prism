//! Streaming-feedback to priority scoring for texture pages.
//!
//! The `GPU` feedback pass reports, per visible page, which mip a view wants and
//! how important that surface is on screen. This module turns each such report
//! into a single fixed-point priority the residency table and scheduler order
//! pages by. The score deliberately uses only integer and fixed-point
//! arithmetic: no `log`/`exp`/`pow` (the mip gap is already an integer count of
//! levels) and no floating-point comparisons, so the ordering is exact,
//! `no_std`-friendly, and bit-for-bit deterministic across machines.
//!
//! Three signals combine additively into the score:
//!
//! * mip error — how many mip levels coarser the finest resident data is than
//!   what the view wants. A page with no resident data at all is charged a
//!   fixed [`MISSING_PAGE_MIP_PENALTY`] so a blank surface always outranks a
//!   merely slightly-blurry one.
//! * screen importance — a caller-supplied fixed-point weight (`0..=`
//!   [`MAX_SCREEN_IMPORTANCE`]) standing in for projected screen coverage.
//! * semantic weight — per [`TextureSemantic`] bias so albedo and normals win
//!   the pool over masks when detail budget is scarce.
//!
//! mip error is scaled by [`MIP_URGENCY`] so that closing a one-level gap is
//! worth roughly as much as a high-importance high-semantic surface, letting
//! badly-undersampled pages preempt merely valuable ones.

use super::{TexturePageKey, TextureSemantic};

/// Largest screen-importance value a caller may supply; higher inputs saturate.
pub const MAX_SCREEN_IMPORTANCE: u16 = 1000;

/// Fixed-point weight added to the score per mip level of shortfall.
pub const MIP_URGENCY: u64 = 10_000;

/// mip-error charge for a page with no resident data of any level.
///
/// Chosen larger than any realistic mip chain so a fully-missing page always
/// sorts above a page that merely needs a finer level.
pub const MISSING_PAGE_MIP_PENALTY: u32 = 24;

/// Per-[`TextureSemantic`] fixed-point priority weights.
///
/// Higher weights bias the residency pool toward the channels that matter most
/// perceptually. Defaults rank base color and normals highest, then the
/// packed roughness/metal/`AO` and `HDR` channels, with height and coverage
/// masks lowest since coarse data there is least objectionable.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SemanticWeights {
    /// Weight for [`TextureSemantic::Color`].
    pub color: u32,
    /// Weight for [`TextureSemantic::Normal`].
    pub normal: u32,
    /// Weight for [`TextureSemantic::RoughnessMetalAo`].
    pub roughness_metal_ao: u32,
    /// Weight for [`TextureSemantic::Height`].
    pub height: u32,
    /// Weight for [`TextureSemantic::Mask`].
    pub mask: u32,
    /// Weight for [`TextureSemantic::Hdr`].
    pub hdr: u32,
}

impl SemanticWeights {
    /// The default perceptual weighting used when a caller supplies none.
    pub const DEFAULT: Self = Self {
        color: 100,
        normal: 90,
        hdr: 85,
        roughness_metal_ao: 70,
        height: 55,
        mask: 35,
    };

    /// The weight assigned to `semantic`.
    #[must_use]
    pub const fn weight(&self, semantic: TextureSemantic) -> u32 {
        match semantic {
            TextureSemantic::Color => self.color,
            TextureSemantic::Normal => self.normal,
            TextureSemantic::RoughnessMetalAo => self.roughness_metal_ao,
            TextureSemantic::Height => self.height,
            TextureSemantic::Mask => self.mask,
            TextureSemantic::Hdr => self.hdr,
        }
    }
}

impl Default for SemanticWeights {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// One page's streaming demand as reported by the feedback pass.
///
/// `desired_mip` is the finest mip level the view wants (0 is finest);
/// `resident_mip` is the finest level currently backed by physical storage, or
/// `None` when nothing is resident. `screen_importance` is a caller fixed-point
/// coverage weight and `frame` is the reporting frame index for `LRU` tie-breaks.
/// `byte_cost` is the physical size the page occupies once resident, forwarded
/// unchanged to the budget scheduler.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PageDemand {
    /// Which page this demand refers to.
    pub key: TexturePageKey,
    /// Perceptual channel this page belongs to.
    pub semantic: TextureSemantic,
    /// Finest mip level the view wants (0 = finest).
    pub desired_mip: u8,
    /// Finest mip level currently resident, or `None` if nothing is resident.
    pub resident_mip: Option<u8>,
    /// Fixed-point screen coverage/importance, saturated at [`MAX_SCREEN_IMPORTANCE`].
    pub screen_importance: u16,
    /// Physical byte cost once resident.
    pub byte_cost: u64,
    /// Frame index this demand was reported.
    pub frame: u64,
}

impl PageDemand {
    /// Number of mip levels by which resident detail falls short of desired.
    ///
    /// Zero when the resident level is at least as fine as desired; a missing
    /// page is charged [`MISSING_PAGE_MIP_PENALTY`]. Because both mips are
    /// integers this needs no logarithm.
    #[must_use]
    pub fn mip_error(&self) -> u32 {
        match self.resident_mip {
            None => MISSING_PAGE_MIP_PENALTY,
            Some(resident) => u32::from(resident.saturating_sub(self.desired_mip)),
        }
    }

    /// Screen importance clamped to the representable range.
    #[must_use]
    pub fn clamped_importance(&self) -> u16 {
        self.screen_importance.min(MAX_SCREEN_IMPORTANCE)
    }

    /// The fixed-point streaming priority for this demand.
    ///
    /// `mip_error * MIP_URGENCY + semantic_weight * screen_importance`, all in
    /// `u64` so it neither overflows for realistic inputs nor relies on any
    /// floating-point comparison. Higher means load sooner and evict later.
    #[must_use]
    pub fn priority(&self, weights: &SemanticWeights) -> u64 {
        let urgency = u64::from(self.mip_error()) * MIP_URGENCY;
        let value = u64::from(weights.weight(self.semantic)) * u64::from(self.clamped_importance());
        urgency + value
    }

    /// Pushes this demand into a residency table at its computed priority.
    ///
    /// Convenience wrapper over [`super::residency::TextureResidencyTable::request`]
    /// that scores the demand with `weights` first.
    pub fn apply_to(
        &self,
        table: &mut super::residency::TextureResidencyTable,
        weights: &SemanticWeights,
    ) {
        table.request(self.key, self.priority(weights), self.byte_cost, self.frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demand(
        semantic: TextureSemantic,
        desired: u8,
        resident: Option<u8>,
        imp: u16,
    ) -> PageDemand {
        PageDemand {
            key: TexturePageKey {
                texture: 1,
                mip: desired,
                layer: 0,
                x: 0,
                y: 0,
            },
            semantic,
            desired_mip: desired,
            resident_mip: resident,
            screen_importance: imp,
            byte_cost: 4096,
            frame: 1,
        }
    }

    #[test]
    fn mip_error_is_zero_when_resident_is_fine_enough() {
        assert_eq!(
            demand(TextureSemantic::Color, 2, Some(2), 500).mip_error(),
            0
        );
        assert_eq!(
            demand(TextureSemantic::Color, 2, Some(1), 500).mip_error(),
            0
        );
    }

    #[test]
    fn mip_error_counts_levels_of_shortfall() {
        assert_eq!(
            demand(TextureSemantic::Color, 1, Some(4), 500).mip_error(),
            3
        );
    }

    #[test]
    fn missing_page_uses_fixed_penalty() {
        assert_eq!(
            demand(TextureSemantic::Color, 0, None, 500).mip_error(),
            MISSING_PAGE_MIP_PENALTY
        );
    }

    #[test]
    fn importance_saturates_at_max() {
        let d = demand(TextureSemantic::Color, 0, Some(0), 50_000);
        assert_eq!(d.clamped_importance(), MAX_SCREEN_IMPORTANCE);
    }

    #[test]
    fn color_outranks_mask_at_equal_importance_and_error() {
        let w = SemanticWeights::DEFAULT;
        let color = demand(TextureSemantic::Color, 0, Some(0), 500).priority(&w);
        let mask = demand(TextureSemantic::Mask, 0, Some(0), 500).priority(&w);
        assert!(color > mask, "color {color} should outrank mask {mask}");
    }

    #[test]
    fn semantic_default_order_is_monotone() {
        let w = SemanticWeights::DEFAULT;
        assert!(w.weight(TextureSemantic::Color) >= w.weight(TextureSemantic::Normal));
        assert!(w.weight(TextureSemantic::Normal) >= w.weight(TextureSemantic::Hdr));
        assert!(w.weight(TextureSemantic::Hdr) >= w.weight(TextureSemantic::RoughnessMetalAo));
        assert!(w.weight(TextureSemantic::RoughnessMetalAo) >= w.weight(TextureSemantic::Height));
        assert!(w.weight(TextureSemantic::Height) >= w.weight(TextureSemantic::Mask));
    }

    #[test]
    fn missing_page_preempts_satisfied_high_value_page() {
        let w = SemanticWeights::DEFAULT;
        // A fully-missing mask page (charged MISSING_PAGE_MIP_PENALTY mips) must
        // outrank a fully-satisfied top-importance color page: a blank surface
        // is worse than a slightly-coarse one regardless of semantic value.
        let missing_mask = demand(TextureSemantic::Mask, 0, None, 1000).priority(&w);
        let satisfied_color = demand(TextureSemantic::Color, 0, Some(0), 1000).priority(&w);
        assert!(missing_mask > satisfied_color);
    }

    #[test]
    fn larger_mip_gap_raises_priority_within_a_semantic() {
        let w = SemanticWeights::DEFAULT;
        let small_gap = demand(TextureSemantic::Color, 0, Some(1), 500).priority(&w);
        let big_gap = demand(TextureSemantic::Color, 0, Some(5), 500).priority(&w);
        assert!(big_gap > small_gap);
    }

    #[test]
    fn priority_is_deterministic_for_equal_inputs() {
        let w = SemanticWeights::DEFAULT;
        let a = demand(TextureSemantic::Normal, 1, Some(4), 750).priority(&w);
        let b = demand(TextureSemantic::Normal, 1, Some(4), 750).priority(&w);
        assert_eq!(a, b);
    }

    #[test]
    fn apply_to_scores_and_requests() {
        let w = SemanticWeights::DEFAULT;
        let d = demand(TextureSemantic::Color, 0, None, 500);
        let mut table = super::super::residency::TextureResidencyTable::new();
        d.apply_to(&mut table, &w);
        let rec = table.record(d.key).expect("requested");
        assert_eq!(rec.priority, d.priority(&w));
        assert_eq!(rec.byte_cost, 4096);
    }
}
