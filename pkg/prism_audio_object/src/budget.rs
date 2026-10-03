//! The hardware renderable-object budget model.
//!
//! Platform object renderers (Atmos, Tempest, Windows Sonic) expose a finite
//! number of simultaneously renderable objects. [`ObjectBudget`] captures that
//! ceiling. When a scene presents more objects than the budget allows, the
//! surplus is reduced by [`crate::clustering`] (energy-preserving merge) before
//! being handed to the renderer. This type only models the capacity; it does
//! not itself perform any merging.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the hardware object budget of design section 44.2 (linked to the
//! source-clustering idea of section 33). Drives [`crate::clustering`] and
//! [`crate::render`].

/// A hardware object-count ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectBudget {
    /// Maximum number of objects the target renderer can present at once.
    max_objects: usize,
}

impl ObjectBudget {
    /// Creates a budget allowing `max_objects` renderable objects.
    ///
    /// A budget of zero means the renderer cannot present discrete objects at
    /// all; callers should fold everything into the bed in that case.
    #[must_use]
    pub const fn new(max_objects: usize) -> Self {
        Self { max_objects }
    }

    /// Returns the maximum renderable object count.
    #[must_use]
    pub const fn max_objects(self) -> usize {
        self.max_objects
    }

    /// Returns whether `count` objects fit within this budget.
    #[must_use]
    pub const fn fits(self, count: usize) -> bool {
        count <= self.max_objects
    }

    /// Returns how many objects are over budget for a scene of `count`
    /// objects (zero when within budget).
    #[must_use]
    pub const fn overflow(self, count: usize) -> usize {
        count.saturating_sub(self.max_objects)
    }

    /// Returns the number of output clusters a scene of `count` objects should
    /// be reduced to: `count` when it fits, otherwise the ceiling.
    #[must_use]
    pub const fn target_clusters(self, count: usize) -> usize {
        if count <= self.max_objects {
            count
        } else {
            self.max_objects
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_and_overflow_are_consistent() {
        let b = ObjectBudget::new(4);
        assert!(b.fits(4));
        assert!(!b.fits(5));
        assert_eq!(b.overflow(4), 0);
        assert_eq!(b.overflow(7), 3);
    }

    #[test]
    fn target_clusters_caps_at_budget() {
        let b = ObjectBudget::new(3);
        assert_eq!(b.target_clusters(2), 2);
        assert_eq!(b.target_clusters(3), 3);
        assert_eq!(b.target_clusters(10), 3);
    }

    #[test]
    fn zero_budget_folds_everything() {
        let b = ObjectBudget::new(0);
        assert!(!b.fits(1));
        assert_eq!(b.target_clusters(5), 0);
        assert_eq!(b.max_objects(), 0);
    }
}
