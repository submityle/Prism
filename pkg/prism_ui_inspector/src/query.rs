//! Filtering a [`TreeSnapshot`] for nodes that match a predicate.
//!
//! A [`Query`] is a small, composable filter built fluently: it can require a
//! specific kind label, require one or more style classes to be present, and
//! require the text content to contain a substring. All configured conditions
//! must hold for a node to match. Queries walk the tree in depth-first preorder,
//! so results and their [`NodePath`]s are deterministic.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui_devtools::{SnapshotNode, TreeSnapshot};

use crate::path::NodePath;

/// A composable predicate over [`SnapshotNode`]s.
///
/// Build one with [`Query::new`] and refine it with [`Query::kind`],
/// [`Query::with_class`], and [`Query::text_contains`]. An empty query matches
/// every node.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Query {
    kind: Option<String>,
    classes: Vec<String>,
    text_contains: Option<String>,
}

impl Query {
    /// Creates an empty query that matches every node.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requires the node's kind label to equal `kind` exactly.
    #[must_use]
    pub fn kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = Some(kind.into());
        self
    }

    /// Requires the node to carry the style class `class`.
    ///
    /// Repeated calls accumulate; every requested class must be present.
    #[must_use]
    pub fn with_class(mut self, class: impl Into<String>) -> Self {
        self.classes.push(class.into());
        self
    }

    /// Requires the node's text content to contain `needle` as a substring.
    #[must_use]
    pub fn text_contains(mut self, needle: impl Into<String>) -> Self {
        self.text_contains = Some(needle.into());
        self
    }

    /// Returns `true` when `node` satisfies every configured condition.
    #[must_use]
    pub fn matches(&self, node: &SnapshotNode) -> bool {
        if let Some(kind) = &self.kind
            && node.kind != *kind
        {
            return false;
        }
        for class in &self.classes {
            if !node.classes.contains(class) {
                return false;
            }
        }
        if let Some(needle) = &self.text_contains {
            match &node.text {
                Some(text) if text.contains(needle.as_str()) => {}
                _ => return false,
            }
        }
        true
    }

    /// Returns every matching node paired with its [`NodePath`], in preorder.
    #[must_use]
    pub fn find_all<'a>(&self, tree: &'a TreeSnapshot) -> Vec<(NodePath, &'a SnapshotNode)> {
        let mut out = Vec::new();
        let mut current = Vec::new();
        self.walk(&tree.root, &mut current, &mut out);
        out
    }

    /// Returns the first matching node in preorder, if any.
    #[must_use]
    pub fn find_first<'a>(&self, tree: &'a TreeSnapshot) -> Option<(NodePath, &'a SnapshotNode)> {
        self.find_all(tree).into_iter().next()
    }

    /// Returns the number of matching nodes.
    #[must_use]
    pub fn count(&self, tree: &TreeSnapshot) -> usize {
        self.find_all(tree).len()
    }

    /// Depth-first helper collecting matches with their paths.
    fn walk<'a>(
        &self,
        node: &'a SnapshotNode,
        current: &mut Vec<usize>,
        out: &mut Vec<(NodePath, &'a SnapshotNode)>,
    ) {
        if self.matches(node) {
            out.push((NodePath::from_indices(current.iter().copied()), node));
        }
        for (index, child) in node.children.iter().enumerate() {
            current.push(index);
            self.walk(child, current, out);
            current.pop();
        }
    }
}

/// Finds every node carrying the style class `class`.
///
/// Convenience wrapper over [`Query::with_class`] plus [`Query::find_all`].
#[must_use]
pub fn find_by_class<'a>(
    tree: &'a TreeSnapshot,
    class: impl Into<String>,
) -> Vec<(NodePath, &'a SnapshotNode)> {
    Query::new().with_class(class).find_all(tree)
}

/// Finds every node whose kind label equals `kind`.
///
/// Convenience wrapper over [`Query::kind`] plus [`Query::find_all`].
#[must_use]
pub fn find_by_kind<'a>(
    tree: &'a TreeSnapshot,
    kind: impl Into<String>,
) -> Vec<(NodePath, &'a SnapshotNode)> {
    Query::new().kind(kind).find_all(tree)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    extern crate alloc;

    use prism_ui::Element;
    use prism_ui_devtools::snapshot;

    use super::{find_by_class, find_by_kind, Query};
    use crate::path::NodePath;

    fn sample() -> prism_ui_devtools::TreeSnapshot {
        let view = Element::box_()
            .class("app")
            .child(
                Element::box_()
                    .class("card")
                    .child(Element::text("hello world")),
            )
            .child(Element::box_().class("card").class("raised"))
            .child(Element::text("footer"));
        snapshot(&view)
    }

    #[test]
    fn empty_query_matches_all() {
        let tree = sample();
        assert_eq!(Query::new().count(&tree), tree.node_count());
    }

    #[test]
    fn kind_filter() {
        let tree = sample();
        let boxes = Query::new().kind("Box").find_all(&tree);
        assert_eq!(boxes.len(), 3);
        let texts = Query::new().kind("Text").count(&tree);
        assert_eq!(texts, 2);
    }

    #[test]
    fn class_filter() {
        let tree = sample();
        let cards = Query::new().with_class("card").find_all(&tree);
        assert_eq!(cards.len(), 2);
    }

    #[test]
    fn multiple_classes_require_all() {
        let tree = sample();
        let raised = Query::new().with_class("card").with_class("raised");
        assert_eq!(raised.count(&tree), 1);
    }

    #[test]
    fn text_contains_filter() {
        let tree = sample();
        let hits = Query::new().text_contains("world").find_all(&tree);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].1.text.as_deref(), Some("hello world"));
    }

    #[test]
    fn text_contains_requires_text_node() {
        let tree = sample();
        // A Box has no text, so a text filter never matches it.
        let hits = Query::new().kind("Box").text_contains("x").count(&tree);
        assert_eq!(hits, 0);
    }

    #[test]
    fn combined_conditions() {
        let tree = sample();
        let q = Query::new().kind("Box").with_class("card");
        assert_eq!(q.count(&tree), 2);
    }

    #[test]
    fn find_first_returns_preorder_match() {
        let tree = sample();
        let first = Query::new().with_class("card").find_first(&tree).unwrap();
        assert_eq!(first.0, NodePath::from_indices([0]));
    }

    #[test]
    fn find_first_none_when_no_match() {
        let tree = sample();
        assert!(Query::new().kind("Canvas").find_first(&tree).is_none());
    }

    #[test]
    fn paths_are_correct() {
        let tree = sample();
        let cards = find_by_class(&tree, "card");
        let paths: Vec<_> = cards.iter().map(|(p, _)| p.clone()).collect();
        assert_eq!(
            paths,
            [NodePath::from_indices([0]), NodePath::from_indices([1])]
        );
    }

    #[test]
    fn free_function_find_by_kind() {
        let tree = sample();
        let texts = find_by_kind(&tree, "Text");
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[0].0, NodePath::from_indices([0, 0]));
        assert_eq!(texts[1].0, NodePath::from_indices([2]));
    }
}
