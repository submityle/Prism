//! The quality level that keys the pages of a paged Patch.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The idea of
//! authoring several quality variants of one graph and selecting one at
//! load/run time is a classical scalability technique; this is an independent
//! data model for it.
//!
//! # Relationship
//! Supports design section 11 (the Patch procedural graph) and the design
//! section 42 open item on "`MetaSound` Pages-style tiered compilation coupled to
//! the section 32 quality governor". [`QualityLevel`] is the authoring-layer key
//! that a host maps the governor's runtime quality tier onto. This crate never
//! depends on `prism_audio_governor`, keeping the authoring layer (M5) strictly
//! below the budget governor (M7).

/// A discrete quality level selecting one page of a [`super::PagedPatch`].
///
/// Levels are a total order: `0` is the cheapest survivable rendering and
/// higher values request progressively richer graphs. The value mirrors the
/// index space of the governor's quality tier so a host can map one onto the
/// other without this crate depending on the governor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct QualityLevel(pub u8);

impl QualityLevel {
    /// The cheapest survivable level; every paged Patch must define a page here.
    pub const LOWEST: QualityLevel = QualityLevel(0);

    /// Builds a level from its raw index.
    #[must_use]
    pub const fn new(index: u8) -> Self {
        QualityLevel(index)
    }

    /// Returns the raw index of this level.
    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Returns the level one step richer, saturating at [`u8::MAX`].
    #[must_use]
    pub const fn richer(self) -> Self {
        QualityLevel(self.0.saturating_add(1))
    }

    /// Returns the level one step cheaper, saturating at [`QualityLevel::LOWEST`].
    #[must_use]
    pub const fn cheaper(self) -> Self {
        QualityLevel(self.0.saturating_sub(1))
    }
}

impl Default for QualityLevel {
    fn default() -> Self {
        QualityLevel::LOWEST
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_is_by_index() {
        assert!(QualityLevel::new(0) < QualityLevel::new(3));
        assert_eq!(QualityLevel::LOWEST, QualityLevel::default());
    }

    #[test]
    fn step_helpers_saturate() {
        assert_eq!(QualityLevel::new(0).cheaper(), QualityLevel::new(0));
        assert_eq!(QualityLevel::new(u8::MAX).richer(), QualityLevel::new(u8::MAX));
        assert_eq!(QualityLevel::new(2).richer(), QualityLevel::new(3));
        assert_eq!(QualityLevel::new(2).cheaper(), QualityLevel::new(1));
    }
}
