//! Error type for building and resolving a paged Patch.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Reports the failure modes of [`super::PagedPatch`] construction, part of the
//! design section 42 "`MetaSound` Pages-style tiered compilation" item built on
//! the design section 11 Patch model.

use super::tier::QualityLevel;

/// A paged-Patch construction failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PagesError {
    /// No pages were supplied; a paged Patch needs at least a base page.
    Empty,
    /// No page is defined at [`QualityLevel::LOWEST`], so the cheapest tier has
    /// no survivable rendering to fall back to.
    MissingBasePage,
    /// Two pages declared the same [`QualityLevel`], which is ambiguous.
    DuplicateLevel(QualityLevel),
}

impl core::fmt::Display for PagesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PagesError::Empty => write!(f, "a paged patch needs at least one page"),
            PagesError::MissingBasePage => {
                write!(f, "a paged patch needs a base page at quality level 0")
            }
            PagesError::DuplicateLevel(level) => {
                write!(f, "two pages share quality level {}", level.index())
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PagesError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn messages_are_distinct() {
        let a = format!("{}", PagesError::Empty);
        let b = format!("{}", PagesError::MissingBasePage);
        let c = format!("{}", PagesError::DuplicateLevel(QualityLevel::new(2)));
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert!(c.contains('2'));
    }
}
