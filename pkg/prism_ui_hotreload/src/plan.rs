//! Diffing an old element tree against a new one into a [`ReloadPlan`].
//!
//! A plan classifies every logical node (identified by its [`NodePath`]) into
//! one of four outcomes:
//!
//! * [`NodeChange::Preserved`] — the node exists in both trees with the same
//!   [`ElementKind`], so its per-node state may be kept.
//! * [`NodeChange::Recreated`] — the node exists in both trees under the same
//!   path but its kind changed, so its state must be discarded. This happens
//!   for keyed nodes whose kind changed; a positional node that changes kind
//!   gets a different path and is instead reported as a removal plus an
//!   addition.
//! * [`NodeChange::Removed`] — the node existed only in the old tree.
//! * [`NodeChange::Added`] — the node exists only in the new tree.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::{Element, ElementKind};

use crate::identity::{paths_of, NodePath};

/// A node that must be recreated because its kind changed under a stable path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recreated {
    /// The shared path of the node in both trees.
    pub path: NodePath,
    /// A human-readable explanation of why the node was recreated.
    pub reason: String,
}

/// The classification of a single logical node when diffing two trees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeChange {
    /// Present in both trees with an unchanged kind; state may be preserved.
    Preserved(NodePath),
    /// Present only in the new tree.
    Added(NodePath),
    /// Present only in the old tree.
    Removed(NodePath),
    /// Present in both trees but with a changed kind; state is invalidated.
    Recreated {
        /// The shared path of the node.
        path: NodePath,
        /// Why the node was recreated.
        reason: String,
    },
}

/// Aggregate counts for a [`ReloadPlan`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanCounts {
    /// Number of preserved nodes.
    pub preserved: usize,
    /// Number of added nodes.
    pub added: usize,
    /// Number of removed nodes.
    pub removed: usize,
    /// Number of recreated nodes.
    pub recreated: usize,
}

/// The result of diffing an old element tree against a new one.
///
/// Each category is stored separately and in deterministic path order. Use
/// [`ReloadPlan::changes`] for a single flattened list of [`NodeChange`]s.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReloadPlan {
    /// Nodes present in both trees with an unchanged kind.
    preserved: Vec<NodePath>,
    /// Nodes present only in the new tree.
    added: Vec<NodePath>,
    /// Nodes present only in the old tree.
    removed: Vec<NodePath>,
    /// Nodes present under a stable path but with a changed kind.
    recreated: Vec<Recreated>,
}

impl ReloadPlan {
    /// Returns the preserved node paths.
    #[must_use]
    pub fn preserved(&self) -> &[NodePath] {
        &self.preserved
    }

    /// Returns the added node paths.
    #[must_use]
    pub fn added(&self) -> &[NodePath] {
        &self.added
    }

    /// Returns the removed node paths.
    #[must_use]
    pub fn removed(&self) -> &[NodePath] {
        &self.removed
    }

    /// Returns the recreated nodes.
    #[must_use]
    pub fn recreated(&self) -> &[Recreated] {
        &self.recreated
    }

    /// Returns the per-category counts.
    #[must_use]
    pub fn counts(&self) -> PlanCounts {
        PlanCounts {
            preserved: self.preserved.len(),
            added: self.added.len(),
            removed: self.removed.len(),
            recreated: self.recreated.len(),
        }
    }

    /// Returns `true` when the two trees were structurally identical, i.e. only
    /// preserved nodes and nothing added, removed, or recreated.
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.recreated.is_empty()
    }

    /// Flattens the plan into a single ordered list of [`NodeChange`]s.
    ///
    /// The order is preserved, then added, then removed, then recreated; each
    /// group is in path order.
    #[must_use]
    pub fn changes(&self) -> Vec<NodeChange> {
        let mut out = Vec::new();
        for path in &self.preserved {
            out.push(NodeChange::Preserved(path.clone()));
        }
        for path in &self.added {
            out.push(NodeChange::Added(path.clone()));
        }
        for path in &self.removed {
            out.push(NodeChange::Removed(path.clone()));
        }
        for item in &self.recreated {
            out.push(NodeChange::Recreated {
                path: item.path.clone(),
                reason: item.reason.clone(),
            });
        }
        out
    }

    /// Renders a stable, human-readable multi-line report of the plan.
    #[must_use]
    pub fn report(&self) -> String {
        let counts = self.counts();
        let mut out = String::new();
        out.push_str(&format!(
            "reload plan: {} preserved, {} added, {} removed, {} recreated\n",
            counts.preserved, counts.added, counts.removed, counts.recreated
        ));
        for path in &self.preserved {
            out.push_str(&format!("  = {path}\n"));
        }
        for path in &self.added {
            out.push_str(&format!("  + {path}\n"));
        }
        for path in &self.removed {
            out.push_str(&format!("  - {path}\n"));
        }
        for item in &self.recreated {
            out.push_str(&format!("  ~ {} ({})\n", item.path, item.reason));
        }
        out
    }
}

/// Returns a readable name for an [`ElementKind`], used in recreate reasons.
fn kind_label(kind: &ElementKind) -> String {
    match kind {
        ElementKind::Box => String::from("box"),
        ElementKind::Text => String::from("text"),
        ElementKind::Custom(name) => format!("custom({name})"),
    }
}

/// Diffs `old` against `new`, producing a [`ReloadPlan`].
///
/// Nodes are matched by [`NodePath`]. A path present in both trees is preserved
/// when the two nodes share a kind, or recreated when the kind changed. Paths
/// present on only one side are removed or added accordingly.
#[must_use]
pub fn plan(old: &Element, new: &Element) -> ReloadPlan {
    let old_nodes: BTreeMap<NodePath, &Element> = paths_of(old).into_iter().collect();
    let new_nodes: BTreeMap<NodePath, &Element> = paths_of(new).into_iter().collect();

    let mut result = ReloadPlan::default();

    for (path, old_el) in &old_nodes {
        match new_nodes.get(path) {
            Some(new_el) => {
                if old_el.kind() == new_el.kind() {
                    result.preserved.push(path.clone());
                } else {
                    result.recreated.push(Recreated {
                        path: path.clone(),
                        reason: format!(
                            "kind changed {} -> {}",
                            kind_label(old_el.kind()),
                            kind_label(new_el.kind())
                        ),
                    });
                }
            }
            None => result.removed.push(path.clone()),
        }
    }

    for path in new_nodes.keys() {
        if !old_nodes.contains_key(path) {
            result.added.push(path.clone());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::{plan, NodeChange};
    use prism_ui::Element;

    #[test]
    fn identical_trees_are_all_preserved() {
        let tree = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let plan = plan(&tree, &tree);
        assert!(plan.is_noop());
        assert_eq!(plan.counts().preserved, 3);
        assert_eq!(plan.counts().added, 0);
        assert_eq!(plan.counts().removed, 0);
        assert_eq!(plan.counts().recreated, 0);
    }

    #[test]
    fn appended_child_is_added() {
        let old = Element::box_().child(Element::text("a"));
        let new = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let plan = plan(&old, &new);
        assert_eq!(plan.counts().added, 1);
        assert_eq!(plan.counts().preserved, 2);
        assert!(!plan.is_noop());
    }

    #[test]
    fn dropped_child_is_removed() {
        let old = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let new = Element::box_().child(Element::text("a"));
        let plan = plan(&old, &new);
        assert_eq!(plan.counts().removed, 1);
        assert_eq!(plan.counts().preserved, 2);
    }

    #[test]
    fn keyed_kind_change_is_recreated() {
        let old = Element::box_().child(Element::box_().key_str("slot"));
        let new = Element::box_().child(Element::text("now text").key_str("slot"));
        let plan = plan(&old, &new);
        assert_eq!(plan.counts().recreated, 1);
        assert_eq!(plan.counts().removed, 0);
        assert_eq!(plan.counts().added, 0);
        assert!(plan.recreated()[0].reason.contains("box -> text"));
    }

    #[test]
    fn positional_kind_change_is_remove_plus_add() {
        let old = Element::box_().child(Element::box_());
        let new = Element::box_().child(Element::text("x"));
        let plan = plan(&old, &new);
        // Positional identity includes kind, so the paths differ.
        assert_eq!(plan.counts().removed, 1);
        assert_eq!(plan.counts().added, 1);
        assert_eq!(plan.counts().recreated, 0);
    }

    #[test]
    fn changes_flattens_every_category() {
        let old = Element::box_()
            .child(Element::text("keep"))
            .child(Element::box_().key_str("slot"))
            .child(Element::text("gone").key_str("g"));
        let new = Element::box_()
            .child(Element::text("keep"))
            .child(Element::text("changed").key_str("slot"))
            .child(Element::text("new").key_str("n"));
        let plan = plan(&old, &new);
        let changes = plan.changes();
        assert_eq!(
            changes.len(),
            plan.counts().preserved
                + plan.counts().added
                + plan.counts().removed
                + plan.counts().recreated
        );
        assert!(changes
            .iter()
            .any(|c| matches!(c, NodeChange::Recreated { .. })));
        assert!(changes.iter().any(|c| matches!(c, NodeChange::Added(_))));
        assert!(changes.iter().any(|c| matches!(c, NodeChange::Removed(_))));
        assert!(changes
            .iter()
            .any(|c| matches!(c, NodeChange::Preserved(_))));
    }

    #[test]
    fn report_mentions_counts_and_is_deterministic() {
        let old = Element::box_().child(Element::text("a"));
        let new = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let plan = plan(&old, &new);
        let report = plan.report();
        assert!(report.contains("1 added"));
        assert_eq!(report, plan.report());
    }
}
