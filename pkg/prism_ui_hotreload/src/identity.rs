//! Stable logical identity for element-tree nodes.
//!
//! Hot reload needs to decide whether a node in the new tree is "the same
//! node" as one in the old tree, so that per-node state can survive the swap.
//! That decision is made by [`NodePath`]: the chain of [`NodeIdent`]s from the
//! root down to a node. Two nodes in two different trees are considered the
//! same logical node exactly when their paths are equal.
//!
//! A node's identity segment is derived deterministically:
//!
//! * If the node carries an explicit [`Key`] it is identified by
//!   [`NodeIdent::Keyed`], independent of its position or kind. This lets a
//!   keyed list item keep its state even when it moves or changes kind.
//! * Otherwise it is identified by [`NodeIdent::Positional`], combining its
//!   sibling index with its [`ElementKind`]. A positional node that changes
//!   kind therefore gets a different identity.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;

use prism_ui::{Element, ElementKind, Key};

/// One segment of a [`NodePath`]: how a single node is identified relative to
/// its parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeIdent {
    /// The node carried an explicit [`Key`]; identity ignores position.
    Keyed(Key),
    /// The node had no explicit key; identity is its sibling index paired with
    /// its [`ElementKind`].
    Positional {
        /// Zero-based index among the parent's children (the root uses `0`).
        index: usize,
        /// The node's element kind at the time of capture.
        kind: ElementKind,
    },
}

impl NodeIdent {
    /// Builds a keyed identity segment.
    #[must_use]
    pub fn keyed(key: Key) -> Self {
        NodeIdent::Keyed(key)
    }

    /// Builds a positional identity segment from a sibling index and kind.
    #[must_use]
    pub fn positional(index: usize, kind: ElementKind) -> Self {
        NodeIdent::Positional { index, kind }
    }

    /// Returns `true` when this segment is a keyed identity.
    #[must_use]
    pub fn is_keyed(&self) -> bool {
        matches!(self, NodeIdent::Keyed(_))
    }
}

/// Returns a total-order sort key for an [`ElementKind`].
///
/// [`ElementKind`] is not itself ordered, so this maps it to a comparable
/// `(tag, name)` pair: boxes sort before texts before custom kinds, and custom
/// kinds are ordered by name.
fn kind_order(kind: &ElementKind) -> (u8, &str) {
    match kind {
        ElementKind::Box => (0, ""),
        ElementKind::Text => (1, ""),
        ElementKind::Custom(name) => (2, name.as_str()),
    }
}

impl Ord for NodeIdent {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (NodeIdent::Keyed(a), NodeIdent::Keyed(b)) => a.cmp(b),
            (NodeIdent::Keyed(_), NodeIdent::Positional { .. }) => Ordering::Less,
            (NodeIdent::Positional { .. }, NodeIdent::Keyed(_)) => Ordering::Greater,
            (
                NodeIdent::Positional {
                    index: ia,
                    kind: ka,
                },
                NodeIdent::Positional {
                    index: ib,
                    kind: kb,
                },
            ) => ia.cmp(ib).then_with(|| kind_order(ka).cmp(&kind_order(kb))),
        }
    }
}

impl PartialOrd for NodeIdent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for NodeIdent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeIdent::Keyed(Key::Index(i)) => write!(f, "@{i}"),
            NodeIdent::Keyed(Key::Int(i)) => write!(f, "#{i}"),
            NodeIdent::Keyed(Key::Str(s)) => write!(f, "#{s}"),
            NodeIdent::Positional { index, kind } => {
                write!(f, "{}[{index}]", kind_name(kind))
            }
        }
    }
}

/// Returns the human-readable name of an [`ElementKind`] for display.
fn kind_name(kind: &ElementKind) -> &str {
    match kind {
        ElementKind::Box => "box",
        ElementKind::Text => "text",
        ElementKind::Custom(name) => name.as_str(),
    }
}

/// The chain of [`NodeIdent`]s from the root to a node.
///
/// A path uniquely identifies a logical node within a tree. Equality of paths
/// across two trees is the signal that a node is "the same" for the purpose of
/// state preservation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodePath(pub Vec<NodeIdent>);

impl NodePath {
    /// Builds a path from its ordered segments.
    #[must_use]
    pub fn new(segments: Vec<NodeIdent>) -> Self {
        NodePath(segments)
    }

    /// Returns the ordered identity segments, root first.
    #[must_use]
    pub fn segments(&self) -> &[NodeIdent] {
        &self.0
    }

    /// Returns the number of segments in the path (its depth from the root).
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when the path has no segments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for NodePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "/");
        }
        for segment in &self.0 {
            write!(f, "/{segment}")?;
        }
        Ok(())
    }
}

/// Derives the identity segment for a node at the given sibling index.
fn ident_of(index: usize, element: &Element) -> NodeIdent {
    match element.explicit_key() {
        Some(key) => NodeIdent::Keyed(key.clone()),
        None => NodeIdent::Positional {
            index,
            kind: element.kind().clone(),
        },
    }
}

/// Walks `element` depth-first in preorder, pushing each node's path.
fn collect<'a>(
    path: &mut Vec<NodeIdent>,
    element: &'a Element,
    out: &mut Vec<(NodePath, &'a Element)>,
) {
    out.push((NodePath(path.clone()), element));
    for (index, child) in element.child_elements().iter().enumerate() {
        path.push(ident_of(index, child));
        collect(path, child, out);
        path.pop();
    }
}

/// Returns every node of `root` paired with its [`NodePath`], in a
/// deterministic depth-first preorder.
///
/// The root is assigned sibling index `0`. The traversal order depends only on
/// the tree structure, so the same tree always yields the same sequence.
#[must_use]
pub fn paths_of(root: &Element) -> Vec<(NodePath, &Element)> {
    let mut out = Vec::new();
    let mut path = Vec::new();
    path.push(ident_of(0, root));
    collect(&mut path, root, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::{ident_of, kind_order, paths_of, NodeIdent, NodePath};
    use alloc::string::ToString;
    use alloc::vec;
    use prism_ui::{Element, ElementKind, Key};

    #[test]
    fn keyed_identity_ignores_position_and_kind() {
        let a = Element::box_().key_str("same");
        let b = Element::text("x").key_str("same");
        assert_eq!(ident_of(3, &a), ident_of(7, &b));
        assert!(ident_of(0, &a).is_keyed());
    }

    #[test]
    fn positional_identity_includes_index_and_kind() {
        let text = Element::text("x");
        let boxed = Element::box_();
        assert_ne!(ident_of(0, &text), ident_of(1, &text));
        assert_ne!(ident_of(0, &text), ident_of(0, &boxed));
        assert!(!ident_of(0, &text).is_keyed());
    }

    #[test]
    fn paths_are_preorder_and_stable() {
        let tree = Element::box_()
            .child(Element::text("a"))
            .child(Element::box_().child(Element::text("b")));
        let first = paths_of(&tree);
        let second = paths_of(&tree);
        assert_eq!(first.len(), 4);
        assert_eq!(first, second);
        // Root path has one segment; the deepest node has three.
        assert_eq!(first[0].0.len(), 1);
        assert_eq!(first.last().unwrap().0.len(), 3);
    }

    #[test]
    fn keyed_child_path_is_structural() {
        let tree = Element::box_().child(Element::text("hi").key_int(42));
        let paths = paths_of(&tree);
        let child = &paths[1].0;
        assert_eq!(child.segments().len(), 2);
        assert_eq!(child.segments()[1], NodeIdent::Keyed(Key::Int(42)));
    }

    #[test]
    fn display_is_readable() {
        let path = NodePath::new(vec![
            NodeIdent::positional(0, ElementKind::Box),
            NodeIdent::Keyed(Key::Str("row".to_string())),
            NodeIdent::positional(2, ElementKind::Text),
        ]);
        assert_eq!(path.to_string(), "/box[0]/#row/text[2]");
        assert_eq!(NodePath::new(vec![]).to_string(), "/");
    }

    #[test]
    fn ordering_separates_keyed_and_positional() {
        let keyed = NodeIdent::Keyed(Key::Int(1));
        let positional = NodeIdent::positional(0, ElementKind::Box);
        assert!(keyed < positional);
        // Positional ordering follows (index, kind).
        assert!(
            NodeIdent::positional(0, ElementKind::Box)
                < NodeIdent::positional(0, ElementKind::Text)
        );
        assert_eq!(kind_order(&ElementKind::Box), (0, ""));
    }

    #[test]
    fn paths_can_key_a_sorted_map() {
        use alloc::collections::BTreeMap;
        let tree = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let mut map: BTreeMap<NodePath, usize> = BTreeMap::new();
        for (i, (path, _)) in paths_of(&tree).into_iter().enumerate() {
            map.insert(path, i);
        }
        assert_eq!(map.len(), 3);
    }
}
