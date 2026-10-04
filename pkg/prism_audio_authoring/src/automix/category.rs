//! Mix categories: a priority-ordered axis orthogonal to the bus tree.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! CRIWARE, or Google Resonance Audio source or derived code; no AI/ML. The
//! category/priority concept is a widely used mixing idiom; this is an
//! independent classical implementation.
//!
//! # Relationship
//! Implements the "category axis" of design section 46.8 (declarative
//! auto-mixing and category loudness governance). Categories name coarse
//! content classes (dialogue, UI, weapons, ambience, music, ...) that duck one
//! another by declarative rule (see [`super::rule`]). This module owns only the
//! category registry; the ducking relationships live in [`super::ruleset`] and
//! are compiled onto the design section 12 modulation matrix by
//! [`super::compiler`].

use alloc::string::String;
use alloc::vec::Vec;

/// A stable handle to a [`Category`] registered in a [`CategorySet`].
///
/// The wrapped index is the category's slot in the owning set and is also the
/// position used by the compiler to address per-category reduction buses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CategoryId(pub usize);

/// A named mix category with a loudness priority.
///
/// `priority` orders importance: a higher value means the category is more
/// important and is more likely to duck others rather than be ducked. The
/// priority is advisory metadata consumed by rule authoring and governance; the
/// actual ducking is expressed by explicit [`super::rule::DuckingRule`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Category {
    /// Human-readable category name, unique within its [`CategorySet`].
    name: String,
    /// Loudness priority; larger is more important.
    priority: i32,
}

impl Category {
    /// Creates a category with the given `name` and `priority`.
    #[must_use]
    pub fn new(name: impl Into<String>, priority: i32) -> Self {
        Self {
            name: name.into(),
            priority,
        }
    }

    /// Returns the category name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the loudness priority; larger is more important.
    #[must_use]
    pub fn priority(&self) -> i32 {
        self.priority
    }
}

/// An ordered registry of mix [`Category`]s.
///
/// Categories are addressed by [`CategoryId`], whose index equals the insertion
/// slot. Names must be unique; re-adding an existing name returns its existing
/// id so authoring code can resolve categories idempotently.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CategorySet {
    /// Categories indexed by [`CategoryId`].
    categories: Vec<Category>,
}

impl CategorySet {
    /// Creates an empty category set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            categories: Vec::new(),
        }
    }

    /// Registers a category by `name` and `priority`, returning its id.
    ///
    /// If a category with the same name already exists its id is returned and
    /// the stored priority is left unchanged, keeping registration idempotent.
    pub fn add(&mut self, name: impl Into<String>, priority: i32) -> CategoryId {
        let name = name.into();
        if let Some(id) = self.find(&name) {
            return id;
        }
        let id = CategoryId(self.categories.len());
        self.categories.push(Category::new(name, priority));
        id
    }

    /// Returns the id of the category named `name`, if present.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<CategoryId> {
        self.categories
            .iter()
            .position(|c| c.name() == name)
            .map(CategoryId)
    }

    /// Returns the category for `id`, if the id is in range.
    #[must_use]
    pub fn get(&self, id: CategoryId) -> Option<&Category> {
        self.categories.get(id.0)
    }

    /// Returns the priority of `id`, or `None` if the id is out of range.
    #[must_use]
    pub fn priority(&self, id: CategoryId) -> Option<i32> {
        self.get(id).map(Category::priority)
    }

    /// Returns the number of registered categories.
    #[must_use]
    pub fn len(&self) -> usize {
        self.categories.len()
    }

    /// Returns `true` if no categories are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.categories.is_empty()
    }

    /// Returns `true` if `id` addresses a registered category.
    #[must_use]
    pub fn contains(&self, id: CategoryId) -> bool {
        id.0 < self.categories.len()
    }

    /// Iterates over `(CategoryId, &Category)` pairs in registration order.
    pub fn iter(&self) -> impl Iterator<Item = (CategoryId, &Category)> {
        self.categories
            .iter()
            .enumerate()
            .map(|(i, c)| (CategoryId(i), c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_is_idempotent_by_name() {
        let mut set = CategorySet::new();
        let a = set.add("dialogue", 100);
        let b = set.add("dialogue", 50);
        assert_eq!(a, b);
        assert_eq!(set.len(), 1);
        // Priority of the first registration is retained.
        assert_eq!(set.priority(a), Some(100));
    }

    #[test]
    fn ids_are_insertion_order() {
        let mut set = CategorySet::new();
        let a = set.add("music", 10);
        let b = set.add("ui", 20);
        assert_eq!(a, CategoryId(0));
        assert_eq!(b, CategoryId(1));
        assert_eq!(set.find("ui"), Some(b));
        assert!(set.contains(b));
        assert!(!set.contains(CategoryId(2)));
    }

    #[test]
    fn iter_yields_ids_and_categories() {
        let mut set = CategorySet::new();
        set.add("a", 1);
        set.add("b", 2);
        let collected: Vec<_> = set.iter().map(|(id, c)| (id, c.priority())).collect();
        assert_eq!(collected, [(CategoryId(0), 1), (CategoryId(1), 2)]);
    }
}
