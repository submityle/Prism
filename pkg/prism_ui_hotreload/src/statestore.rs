//! Per-node state keyed by stable [`NodePath`], with reload application.
//!
//! A [`StateStore`] holds arbitrary application state `T` for individual nodes,
//! addressed by their [`NodePath`]. When a [`ReloadPlan`] is applied, state for
//! removed and recreated nodes is dropped, state for preserved nodes is kept,
//! and added nodes are left untouched for the caller to populate afterwards.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::identity::NodePath;
use crate::plan::ReloadPlan;

/// A summary of what [`StateStore::apply_plan`] did to the store.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReloadReport {
    /// Number of preserved nodes reported by the plan.
    pub preserved: usize,
    /// Number of state entries actually dropped from the store.
    pub dropped: usize,
    /// Number of added nodes reported by the plan.
    pub added: usize,
}

/// A map from [`NodePath`] to per-node state of type `T`.
///
/// The store is ordered, so iteration over [`StateStore::paths`] is
/// deterministic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateStore<T> {
    /// The backing path-to-state map.
    entries: BTreeMap<NodePath, T>,
}

impl<T> Default for StateStore<T> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }
}

impl<T> StateStore<T> {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces the state at `path`, returning any previous value.
    pub fn insert(&mut self, path: NodePath, value: T) -> Option<T> {
        self.entries.insert(path, value)
    }

    /// Returns a reference to the state at `path`, if any.
    #[must_use]
    pub fn get(&self, path: &NodePath) -> Option<&T> {
        self.entries.get(path)
    }

    /// Returns a mutable reference to the state at `path`, if any.
    pub fn get_mut(&mut self, path: &NodePath) -> Option<&mut T> {
        self.entries.get_mut(path)
    }

    /// Removes and returns the state at `path`, if any.
    pub fn remove(&mut self, path: &NodePath) -> Option<T> {
        self.entries.remove(path)
    }

    /// Returns `true` when `path` has stored state.
    #[must_use]
    pub fn contains(&self, path: &NodePath) -> bool {
        self.entries.contains_key(path)
    }

    /// Returns the number of stored entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the store holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the stored paths in sorted order.
    #[must_use]
    pub fn paths(&self) -> Vec<&NodePath> {
        self.entries.keys().collect()
    }

    /// Removes every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Applies a [`ReloadPlan`]: drops state for removed and recreated nodes,
    /// keeps preserved state, and leaves added nodes untouched.
    ///
    /// Returns a [`ReloadReport`] with the plan's preserved and added counts
    /// plus the number of entries actually dropped from this store.
    pub fn apply_plan(&mut self, plan: &ReloadPlan) -> ReloadReport {
        let mut dropped = 0;
        for path in plan.removed() {
            if self.entries.remove(path).is_some() {
                dropped += 1;
            }
        }
        for item in plan.recreated() {
            if self.entries.remove(&item.path).is_some() {
                dropped += 1;
            }
        }
        ReloadReport {
            preserved: plan.preserved().len(),
            dropped,
            added: plan.added().len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StateStore;
    use crate::identity::paths_of;
    use crate::plan::plan;
    use prism_ui::Element;

    fn path_at(root: &Element, index: usize) -> crate::identity::NodePath {
        paths_of(root)[index].0.clone()
    }

    #[test]
    fn insert_get_remove_roundtrip() {
        let tree = Element::box_().child(Element::text("a"));
        let path = path_at(&tree, 1);
        let mut store: StateStore<i32> = StateStore::new();
        assert!(store.is_empty());
        assert_eq!(store.insert(path.clone(), 10), None);
        assert_eq!(store.get(&path), Some(&10));
        assert_eq!(store.insert(path.clone(), 20), Some(10));
        assert_eq!(store.len(), 1);
        assert!(store.contains(&path));
        assert_eq!(store.remove(&path), Some(20));
        assert!(!store.contains(&path));
    }

    #[test]
    fn get_mut_edits_in_place() {
        let tree = Element::box_();
        let path = path_at(&tree, 0);
        let mut store: StateStore<i32> = StateStore::new();
        store.insert(path.clone(), 1);
        *store.get_mut(&path).unwrap() += 41;
        assert_eq!(store.get(&path), Some(&42));
        assert!(store
            .get_mut(&path_at(&Element::text("other"), 0))
            .is_none());
    }

    #[test]
    fn paths_are_sorted_and_clear_empties() {
        let tree = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let p0 = path_at(&tree, 0);
        let p1 = path_at(&tree, 1);
        let mut store: StateStore<u8> = StateStore::new();
        store.insert(p1.clone(), 2);
        store.insert(p0.clone(), 1);
        let paths = store.paths();
        assert_eq!(paths.len(), 2);
        assert!(paths[0] < paths[1]);
        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn apply_plan_drops_removed_state() {
        let old = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let new = Element::box_().child(Element::text("a"));
        let removed_path = path_at(&old, 2);
        let kept_path = path_at(&old, 1);
        let mut store: StateStore<&str> = StateStore::new();
        store.insert(removed_path.clone(), "doomed");
        store.insert(kept_path.clone(), "safe");
        let report = store.apply_plan(&plan(&old, &new));
        assert_eq!(report.dropped, 1);
        assert_eq!(report.preserved, 2);
        assert_eq!(report.added, 0);
        assert!(!store.contains(&removed_path));
        assert!(store.contains(&kept_path));
    }

    #[test]
    fn apply_plan_drops_recreated_state() {
        let old = Element::box_().child(Element::box_().key_str("slot"));
        let new = Element::box_().child(Element::text("t").key_str("slot"));
        let slot_path = path_at(&old, 1);
        let mut store: StateStore<u32> = StateStore::new();
        store.insert(slot_path.clone(), 99);
        let report = store.apply_plan(&plan(&old, &new));
        assert_eq!(report.dropped, 1);
        assert!(!store.contains(&slot_path));
    }

    #[test]
    fn apply_plan_counts_added_and_keeps_preserved() {
        let old = Element::box_().child(Element::text("a"));
        let new = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let kept = path_at(&old, 1);
        let mut store: StateStore<&str> = StateStore::new();
        store.insert(kept.clone(), "keep");
        let report = store.apply_plan(&plan(&old, &new));
        assert_eq!(report.added, 1);
        assert_eq!(report.dropped, 0);
        assert!(store.contains(&kept));
    }
}
