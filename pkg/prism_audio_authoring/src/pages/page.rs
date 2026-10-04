//! A single page: one quality variant of a Patch graph.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! A [`PatchPage`] pairs a [`QualityLevel`] floor with a full
//! [`crate::patch::PatchDescription`]. A [`super::PagedPatch`] holds an ordered
//! set of these and resolves one per requested level. Part of the design
//! section 42 "`MetaSound` Pages-style tiered compilation" item layered on the
//! design section 11 Patch model.

use alloc::string::String;

use crate::patch::PatchDescription;

use super::tier::QualityLevel;

/// One quality variant of a Patch, active from `min_level` upward.
///
/// A page becomes a candidate when the requested [`QualityLevel`] is at least
/// `min_level`; selection among candidates picks the one with the highest
/// `min_level` (see [`super::PagedPatch::resolve`]).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PatchPage {
    /// The lowest quality level at which this page is a candidate.
    min_level: QualityLevel,
    /// A human-readable label for tooling and telemetry (for example "mobile").
    label: String,
    /// The Patch graph compiled when this page is active.
    description: PatchDescription,
}

impl PatchPage {
    /// Builds a page active from `min_level` upward.
    #[must_use]
    pub fn new(min_level: QualityLevel, label: String, description: PatchDescription) -> Self {
        Self {
            min_level,
            label,
            description,
        }
    }

    /// Returns the lowest quality level at which this page is a candidate.
    #[must_use]
    pub fn min_level(&self) -> QualityLevel {
        self.min_level
    }

    /// Returns the human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the Patch graph for this page.
    #[must_use]
    pub fn description(&self) -> &PatchDescription {
        &self.description
    }

    /// Returns `true` when `requested` is at least this page's `min_level`.
    #[must_use]
    pub fn activates_at(&self, requested: QualityLevel) -> bool {
        requested >= self.min_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn activation_is_inclusive_from_min_level() {
        let page = PatchPage::new(
            QualityLevel::new(2),
            "high".to_string(),
            PatchDescription::new(),
        );
        assert!(!page.activates_at(QualityLevel::new(1)));
        assert!(page.activates_at(QualityLevel::new(2)));
        assert!(page.activates_at(QualityLevel::new(5)));
        assert_eq!(page.label(), "high");
        assert_eq!(page.min_level(), QualityLevel::new(2));
    }
}
