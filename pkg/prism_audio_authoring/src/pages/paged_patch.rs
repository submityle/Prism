//! An ordered set of pages forming one scalable Patch asset.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. Authoring several
//! quality variants of one graph and resolving one per quality tier is a
//! classical scalability technique; this is an independent data model.
//!
//! # Relationship
//! A [`PagedPatch`] is the design section 42 "`MetaSound` Pages-style tiered
//! compilation" counterpart to a single [`crate::patch::PatchDescription`]: it
//! stores one [`PatchPage`] per quality variant and resolves exactly one for a
//! requested [`QualityLevel`]. A host compiles the resolved page through the
//! ordinary design section 11 compiler; [`super::PageSelector`] decides *when*
//! a resolution change is worth a recompile.

use alloc::vec::Vec;

use super::error::PagesError;
use super::page::PatchPage;
use super::tier::QualityLevel;

/// A scalable Patch: one [`PatchPage`] per quality variant.
///
/// Pages are stored sorted ascending by [`PatchPage::min_level`]. A valid
/// paged Patch always has a base page at [`QualityLevel::LOWEST`], so
/// [`PagedPatch::resolve`] is total over every requested level.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PagedPatch {
    /// Pages sorted ascending by `min_level`, with no two sharing a level.
    pages: Vec<PatchPage>,
}

impl PagedPatch {
    /// Builds a paged Patch from the supplied pages.
    ///
    /// The pages may arrive in any order; they are sorted ascending by
    /// `min_level`. Fails with [`PagesError::Empty`] when no page is given,
    /// [`PagesError::DuplicateLevel`] when two pages share a level, and
    /// [`PagesError::MissingBasePage`] when none sits at
    /// [`QualityLevel::LOWEST`].
    pub fn new(mut pages: Vec<PatchPage>) -> Result<Self, PagesError> {
        if pages.is_empty() {
            return Err(PagesError::Empty);
        }
        pages.sort_by_key(PatchPage::min_level);
        for window in pages.windows(2) {
            if window[0].min_level() == window[1].min_level() {
                return Err(PagesError::DuplicateLevel(window[0].min_level()));
            }
        }
        if pages[0].min_level() != QualityLevel::LOWEST {
            return Err(PagesError::MissingBasePage);
        }
        Ok(Self { pages })
    }

    /// Returns the pages in ascending `min_level` order.
    #[must_use]
    pub fn pages(&self) -> &[PatchPage] {
        &self.pages
    }

    /// Returns the number of pages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Returns `true` when the paged Patch holds no pages.
    ///
    /// A validly constructed [`PagedPatch`] is never empty; this exists for
    /// API completeness alongside [`PagedPatch::len`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Returns the base page at [`QualityLevel::LOWEST`].
    #[must_use]
    pub fn base(&self) -> &PatchPage {
        &self.pages[0]
    }

    /// Returns the index of the page serving `requested`.
    ///
    /// This is the page with the highest `min_level` not exceeding `requested`.
    /// Because a valid paged Patch has a base page at
    /// [`QualityLevel::LOWEST`], a match always exists.
    #[must_use]
    pub fn resolve_index(&self, requested: QualityLevel) -> usize {
        let mut chosen = 0;
        for (index, page) in self.pages.iter().enumerate() {
            if page.activates_at(requested) {
                chosen = index;
            } else {
                break;
            }
        }
        chosen
    }

    /// Returns the page serving `requested` (see [`PagedPatch::resolve_index`]).
    #[must_use]
    pub fn resolve(&self, requested: QualityLevel) -> &PatchPage {
        &self.pages[self.resolve_index(requested)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::PatchDescription;
    use alloc::string::ToString;
    use alloc::vec;

    fn page(level: u8, label: &str) -> PatchPage {
        PatchPage::new(
            QualityLevel::new(level),
            label.to_string(),
            PatchDescription::new(),
        )
    }

    #[test]
    fn rejects_empty() {
        assert_eq!(PagedPatch::new(vec![]).unwrap_err(), PagesError::Empty);
    }

    #[test]
    fn rejects_missing_base() {
        let err = PagedPatch::new(vec![page(1, "mid")]).unwrap_err();
        assert_eq!(err, PagesError::MissingBasePage);
    }

    #[test]
    fn rejects_duplicate_level() {
        let err = PagedPatch::new(vec![page(0, "a"), page(0, "b")]).unwrap_err();
        assert_eq!(err, PagesError::DuplicateLevel(QualityLevel::LOWEST));
    }

    #[test]
    fn sorts_and_resolves_fallback() {
        let paged = PagedPatch::new(vec![page(4, "ultra"), page(0, "base"), page(2, "high")])
            .expect("valid");
        assert_eq!(paged.base().label(), "base");
        // Below any raised floor falls back to base.
        assert_eq!(paged.resolve(QualityLevel::new(1)).label(), "base");
        // Exact and between floors pick the highest not exceeding the request.
        assert_eq!(paged.resolve(QualityLevel::new(2)).label(), "high");
        assert_eq!(paged.resolve(QualityLevel::new(3)).label(), "high");
        assert_eq!(paged.resolve(QualityLevel::new(9)).label(), "ultra");
        assert_eq!(paged.len(), 3);
        assert!(!paged.is_empty());
    }
}
