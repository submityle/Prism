//! Owned snapshots of a live [`Element`] tree.
//!
//! The runtime's [`Element`] borrows from the frame that built it and is not
//! convenient to keep around for inspection or diffing. [`snapshot`] walks an
//! [`Element`] into a fully owned [`TreeSnapshot`] of [`SnapshotNode`]s that can
//! be stored, compared, and rendered deterministically — the foundation for the
//! snapshot tests tooling in this crate relies on.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use prism_ui::{Element, ElementKind};

/// A single owned node captured from an [`Element`] subtree.
///
/// Every field is owned, so a `SnapshotNode` outlives the frame that produced
/// the source [`Element`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotNode {
    /// The node's kind label: `"Box"`, `"Text"`, or the custom name for
    /// [`ElementKind::Custom`].
    pub kind: String,
    /// The node's text content, if it is a text run.
    pub text: Option<String>,
    /// The node's style class names, in application order.
    pub classes: Vec<String>,
    /// The node's children, in order.
    pub children: Vec<SnapshotNode>,
}

impl SnapshotNode {
    /// Total number of nodes in this subtree, including `self`.
    #[must_use]
    pub fn node_count(&self) -> usize {
        let mut total = 1;
        for child in &self.children {
            total += child.node_count();
        }
        total
    }

    /// Depth of this subtree. A leaf node has depth `1`.
    #[must_use]
    pub fn depth(&self) -> usize {
        let mut deepest = 0;
        for child in &self.children {
            let d = child.depth();
            if d > deepest {
                deepest = d;
            }
        }
        deepest + 1
    }
}

/// An owned, deterministic snapshot of an [`Element`] tree.
///
/// Produced by [`snapshot`]. Holds the captured root [`SnapshotNode`] and
/// exposes whole-tree [`TreeSnapshot::node_count`] and [`TreeSnapshot::depth`]
/// queries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeSnapshot {
    /// The captured root node.
    pub root: SnapshotNode,
}

impl TreeSnapshot {
    /// Total number of nodes in the tree, including the root.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }

    /// Depth of the tree. A lone root has depth `1`.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.root.depth()
    }
}

/// Returns the kind label for an [`ElementKind`].
fn kind_label(kind: &ElementKind) -> String {
    match kind {
        ElementKind::Box => "Box".to_string(),
        ElementKind::Text => "Text".to_string(),
        ElementKind::Custom(name) => name.clone(),
    }
}

/// Captures a [`SnapshotNode`] for `element` and its subtree.
fn capture(element: &Element) -> SnapshotNode {
    let children = element.child_elements().iter().map(capture).collect();
    SnapshotNode {
        kind: kind_label(element.kind()),
        text: element.text_content().map(ToString::to_string),
        classes: element.class_names().to_vec(),
        children,
    }
}

/// Walks an [`Element`] tree into an owned [`TreeSnapshot`].
///
/// The traversal is depth-first and preserves sibling order, so the resulting
/// snapshot — and anything derived from it — is deterministic.
#[must_use]
pub fn snapshot(element: &Element) -> TreeSnapshot {
    TreeSnapshot {
        root: capture(element),
    }
}
