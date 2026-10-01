//! The [`HotReloader`]: a tree plus state that survives reloads.
//!
//! A reloader owns the current element tree and a [`StateStore`] of per-node
//! state. Reloading with a new tree diffs old against new, drops state for
//! nodes that were removed or recreated, keeps state for preserved nodes, and
//! then adopts the new tree as current.

use prism_ui::Element;

use crate::plan::{plan, ReloadPlan};
use crate::statestore::{ReloadReport, StateStore};

/// Owns the current element tree and the per-node state that persists across
/// reloads.
#[derive(Clone, Debug, PartialEq)]
pub struct HotReloader<T> {
    /// The tree currently mounted.
    current: Element,
    /// Per-node state keyed by stable path.
    state: StateStore<T>,
}

impl<T> HotReloader<T> {
    /// Creates a reloader for `initial` with an empty state store.
    #[must_use]
    pub fn new(initial: Element) -> Self {
        Self {
            current: initial,
            state: StateStore::new(),
        }
    }

    /// Returns the currently mounted tree.
    #[must_use]
    pub fn current(&self) -> &Element {
        &self.current
    }

    /// Returns the per-node state store.
    #[must_use]
    pub fn state(&self) -> &StateStore<T> {
        &self.state
    }

    /// Returns the per-node state store mutably, for seeding node state.
    pub fn state_mut(&mut self) -> &mut StateStore<T> {
        &mut self.state
    }

    /// Computes the plan that [`HotReloader::reload`] would apply for `new`,
    /// without mutating anything.
    #[must_use]
    pub fn plan_for(&self, new: &Element) -> ReloadPlan {
        plan(&self.current, new)
    }

    /// Reloads with `new_source`: diffs it against the current tree, applies the
    /// plan to the state store, adopts the new tree, and returns the report.
    pub fn reload(&mut self, new_source: Element) -> ReloadReport {
        let plan = plan(&self.current, &new_source);
        let report = self.state.apply_plan(&plan);
        self.current = new_source;
        report
    }
}

#[cfg(test)]
mod tests {
    use super::HotReloader;
    use crate::identity::paths_of;
    use prism_ui::Element;

    fn path_at(root: &Element, index: usize) -> crate::identity::NodePath {
        paths_of(root)[index].0.clone()
    }

    #[test]
    fn new_starts_with_tree_and_empty_state() {
        let tree = Element::box_().child(Element::text("a"));
        let reloader: HotReloader<i32> = HotReloader::new(tree.clone());
        assert_eq!(reloader.current(), &tree);
        assert!(reloader.state().is_empty());
    }

    #[test]
    fn reload_adopts_new_tree() {
        let old = Element::box_().child(Element::text("a"));
        let new = Element::box_().child(Element::text("b"));
        let mut reloader: HotReloader<i32> = HotReloader::new(old);
        reloader.reload(new.clone());
        assert_eq!(reloader.current(), &new);
    }

    #[test]
    fn reload_preserves_and_drops_state() {
        let old = Element::box_()
            .child(Element::text("keep").key_str("k"))
            .child(Element::text("gone").key_str("g"));
        let keep_path = path_at(&old, 1);
        let gone_path = path_at(&old, 2);
        let mut reloader: HotReloader<&str> = HotReloader::new(old.clone());
        reloader.state_mut().insert(keep_path.clone(), "keep-state");
        reloader.state_mut().insert(gone_path.clone(), "gone-state");

        let new = Element::box_().child(Element::text("keep").key_str("k"));
        let report = reloader.reload(new);
        assert_eq!(report.dropped, 1);
        assert!(reloader.state().contains(&keep_path));
        assert!(!reloader.state().contains(&gone_path));
        assert_eq!(reloader.state().get(&keep_path), Some(&"keep-state"));
    }

    #[test]
    fn plan_for_does_not_mutate() {
        let old = Element::box_().child(Element::text("a"));
        let new = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let reloader: HotReloader<i32> = HotReloader::new(old.clone());
        let plan = reloader.plan_for(&new);
        assert_eq!(plan.counts().added, 1);
        // Current tree is unchanged.
        assert_eq!(reloader.current(), &old);
    }
}
