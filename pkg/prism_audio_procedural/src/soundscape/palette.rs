//! A soundscape palette: the set of elements active for one environment state.
//!
//! A palette is simply the collection of [`SoundscapeElement`]s that make up a
//! particular ambience (a forest at dawn, a city street, a cave). The scheduler
//! walks a palette each block and scatters its elements around the listener
//! according to their individual rules. The palette owns its elements in a
//! heap-allocated list built once at setup; it exposes only read access and a
//! builder so the real-time scheduler never mutates it.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the palette of design section 37; a bag of
//! [`crate::soundscape::element::SoundscapeElement`]s consumed by
//! [`crate::soundscape::scheduler::SoundscapeScheduler`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::soundscape::element::SoundscapeElement;

/// An ordered collection of ambience elements for a single soundscape state.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SoundscapePalette {
    elements: Vec<SoundscapeElement>,
}

impl SoundscapePalette {
    /// Creates an empty palette.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    /// Creates an empty palette with room for `capacity` elements preallocated.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            elements: Vec::with_capacity(capacity),
        }
    }

    /// Appends an element and returns the palette, for fluent construction.
    #[inline]
    #[must_use]
    pub fn with_element(mut self, element: SoundscapeElement) -> Self {
        self.elements.push(element);
        self
    }

    /// Appends an element in place.
    #[inline]
    pub fn push(&mut self, element: SoundscapeElement) {
        self.elements.push(element);
    }

    /// Returns the elements as a slice.
    #[inline]
    #[must_use]
    pub fn elements(&self) -> &[SoundscapeElement] {
        &self.elements
    }

    /// Returns the number of elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Returns `true` when the palette has no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Removes all elements, keeping the allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.elements.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_and_reads_back() {
        let palette = SoundscapePalette::new()
            .with_element(SoundscapeElement::new(1))
            .with_element(SoundscapeElement::new(2));
        assert_eq!(palette.len(), 2);
        assert!(!palette.is_empty());
        assert_eq!(palette.elements()[0].id(), 1);
        assert_eq!(palette.elements()[1].id(), 2);
    }

    #[test]
    fn empty_palette_reports_empty() {
        let palette = SoundscapePalette::new();
        assert!(palette.is_empty());
        assert_eq!(palette.len(), 0);
    }

    #[test]
    fn push_and_clear() {
        let mut palette = SoundscapePalette::with_capacity(4);
        palette.push(SoundscapeElement::new(7));
        assert_eq!(palette.len(), 1);
        palette.clear();
        assert!(palette.is_empty());
    }
}
