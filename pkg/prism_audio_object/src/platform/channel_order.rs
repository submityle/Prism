//! Canonical channel orderings for bed delivery to platform renderers.
//!
//! The engine builds beds in a single canonical order (see
//! [`crate::bed::BedLayout`]): front pair, centre, LFE, then surrounds and
//! height speakers. Platform renderers and file transports do not all agree on
//! where the LFE sits or in which order the remaining speakers appear, so a bed
//! delivery must be permuted into the ordering the target expects. This module
//! enumerates the orderings the bridge supports and resolves, for a given
//! layout, the permutation from platform slot to canonical channel index.
//!
//! The two orderings here are the common building blocks: the engine-canonical
//! film order with the LFE in slot 4, and the variant that moves the LFE to the
//! final slot (used by several consumer and file pipelines). A platform/device
//! layer that needs an exact vendor slot map applies it downstream; this module
//! provides the deterministic, label-driven reorderings the control-rate bridge
//! needs.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Supports design section 16 (platform spatial backend bridge). Consumed by
//! [`crate::platform::delivery`] when it emits a bed payload so the per-channel
//! rows are presented in the platform's expected order.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::bed::BedLayout;

/// A channel ordering convention for bed delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ChannelOrder {
    /// The engine's native film order (identity): front pair, centre, LFE,
    /// then surrounds and height speakers, exactly as [`BedLayout::channels`]
    /// emits them.
    EngineCanonical,
    /// The engine order with the LFE channel moved to the final slot; all
    /// other channels keep their canonical relative order.
    LfeLast,
}

impl ChannelOrder {
    /// Resolves the permutation from platform slot to canonical channel index
    /// for `layout`.
    ///
    /// The returned vector has one entry per channel of `layout`; entry `i`
    /// holds the canonical index of the channel that occupies platform slot
    /// `i`. [`ChannelOrder::EngineCanonical`] returns the identity permutation
    /// `0..channel_count`.
    #[must_use]
    pub fn permutation(self, layout: BedLayout) -> Vec<usize> {
        let count = layout.channel_count();
        match self {
            ChannelOrder::EngineCanonical => (0..count).collect(),
            ChannelOrder::LfeLast => match layout.lfe_index() {
                None => (0..count).collect(),
                Some(lfe) => {
                    let mut order: Vec<usize> = (0..count).filter(|&i| i != lfe).collect();
                    order.push(lfe);
                    order
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_is_identity() {
        let perm = ChannelOrder::EngineCanonical.permutation(BedLayout::Surround7_1_4);
        let expected: Vec<usize> = (0..BedLayout::Surround7_1_4.channel_count()).collect();
        assert_eq!(perm, expected);
    }

    #[test]
    fn lfe_last_moves_lfe_to_end_and_is_a_permutation() {
        let layout = BedLayout::Surround5_1_4;
        let lfe = layout.lfe_index().expect("5.1.4 has an LFE");
        let perm = ChannelOrder::LfeLast.permutation(layout);

        assert_eq!(perm.len(), layout.channel_count());
        assert_eq!(*perm.last().expect("non-empty"), lfe);

        // It is a genuine permutation: every canonical index appears once.
        let mut seen = perm.clone();
        seen.sort_unstable();
        let identity: Vec<usize> = (0..layout.channel_count()).collect();
        assert_eq!(seen, identity);

        // Relative order of non-LFE channels is preserved.
        let non_lfe: Vec<usize> = perm.iter().copied().filter(|&i| i != lfe).collect();
        let expected: Vec<usize> = (0..layout.channel_count()).filter(|&i| i != lfe).collect();
        assert_eq!(non_lfe, expected);
    }

    #[test]
    fn lfe_last_on_lfe_less_layout_is_identity() {
        // Stereo has no LFE, so LfeLast collapses to the identity order.
        let perm = ChannelOrder::LfeLast.permutation(BedLayout::Stereo);
        assert_eq!(perm, [0, 1]);
    }
}
