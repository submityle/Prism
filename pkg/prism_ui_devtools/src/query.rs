//! Addressing and search over a captured [`TreeSnapshot`].
//!
//! A UI inspector needs two primitives on top of a snapshot: *address a node*
//! by a stable location, and *search* for nodes matching a predicate. This
//! module provides both over the owned [`SnapshotNode`]/[`TreeSnapshot`] types
//! from [`crate::snapshot`], without allocating during addressing.
//!
//! A **node path** is a slice of child indices describing how to descend from a
//! starting node. The empty path `[]` denotes the starting node itself, `[0]`
//! its first child, `[1, 2]` the third child of its second child, and so on —
//! the same depth-first, sibling-ordered convention [`crate::snapshot`] and
//! [`crate::render_tree`] use, so paths are stable and deterministic.
//!
//! Search walks the tree in **pre-order** (a node is visited before its
//! children, and children in order), so [`TreeSnapshot::find`] returns the
//! first match in reading order and [`TreeSnapshot::find_all`] returns every
//! match with its path, ordered by that same pre-order.
//!
//! # Example
//!
//! ```
//! use prism_ui::Element;
//! use prism_ui_devtools::snapshot;
//!
//! let view = Element::box_()
//!     .child(Element::text("a"))
//!     .child(Element::box_().child(Element::text("b")));
//! let snap = snapshot(&view);
//!
//! // Address the nested text node by its child-index path.
//! let nested = snap.node_at(&[1, 0]).unwrap();
//! assert_eq!(nested.text.as_deref(), Some("b"));
//!
//! // Find every text node in pre-order.
//! let texts = snap.find_all(|node| node.kind == "Text");
//! assert_eq!(texts, [vec![0], vec![1, 0]]);
//! ```

use alloc::vec::Vec;

use crate::snapshot::{SnapshotNode, TreeSnapshot};

impl SnapshotNode {
    /// Returns the descendant reached by following `path` from this node, or
    /// [`None`] when any index along the way is out of range.
    ///
    /// The empty path returns this node. Each element of `path` selects a child
    /// by its zero-based index, descending one level per element.
    #[must_use]
    pub fn node_at(&self, path: &[usize]) -> Option<&SnapshotNode> {
        match path.split_first() {
            None => Some(self),
            Some((&first, rest)) => self.children.get(first)?.node_at(rest),
        }
    }

    /// Appends the paths of every node in this subtree for which `predicate`
    /// returns `true`, in pre-order, extending `prefix` as it descends.
    ///
    /// Returns `true` as soon as a match is found when `first_only` is set, so
    /// callers can short-circuit. `prefix` is always restored to its original
    /// length before returning.
    fn visit<F: Fn(&SnapshotNode) -> bool>(
        &self,
        predicate: &F,
        first_only: bool,
        prefix: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) -> bool {
        if predicate(self) {
            out.push(prefix.clone());
            if first_only {
                return true;
            }
        }
        for (index, child) in self.children.iter().enumerate() {
            prefix.push(index);
            let done = child.visit(predicate, first_only, prefix, out);
            prefix.pop();
            if done {
                return true;
            }
        }
        false
    }
}

impl TreeSnapshot {
    /// Returns the node addressed by `path` from the root, or [`None`] when any
    /// index is out of range.
    ///
    /// The empty path returns the root node; see [`SnapshotNode::node_at`].
    #[must_use]
    pub fn node_at(&self, path: &[usize]) -> Option<&SnapshotNode> {
        self.root.node_at(path)
    }

    /// Returns the path to the first node (in pre-order) for which `predicate`
    /// returns `true`, or [`None`] when nothing matches.
    ///
    /// The returned path is suitable for [`TreeSnapshot::node_at`].
    #[must_use]
    pub fn find<F: Fn(&SnapshotNode) -> bool>(&self, predicate: F) -> Option<Vec<usize>> {
        let mut prefix = Vec::new();
        let mut out = Vec::new();
        self.root.visit(&predicate, true, &mut prefix, &mut out);
        out.into_iter().next()
    }

    /// Returns the paths to every node (in pre-order) for which `predicate`
    /// returns `true`.
    ///
    /// Each returned path is suitable for [`TreeSnapshot::node_at`]. The result
    /// is empty when nothing matches.
    #[must_use]
    pub fn find_all<F: Fn(&SnapshotNode) -> bool>(&self, predicate: F) -> Vec<Vec<usize>> {
        let mut prefix = Vec::new();
        let mut out = Vec::new();
        self.root.visit(&predicate, false, &mut prefix, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;

    use crate::snapshot::{SnapshotNode, TreeSnapshot};

    fn leaf(kind: &str) -> SnapshotNode {
        SnapshotNode {
            kind: kind.to_string(),
            text: None,
            classes: Vec::new(),
            children: Vec::new(),
        }
    }

    fn branch(kind: &str, children: Vec<SnapshotNode>) -> SnapshotNode {
        SnapshotNode {
            kind: kind.to_string(),
            text: None,
            classes: Vec::new(),
            children,
        }
    }

    // Box
    //   Text (a)
    //   Box
    //     Text (b)
    //     Text (c)
    fn sample() -> TreeSnapshot {
        TreeSnapshot {
            root: branch(
                "Box",
                alloc::vec![
                    leaf("Text"),
                    branch("Box", alloc::vec![leaf("Text"), leaf("Text")]),
                ],
            ),
        }
    }

    #[test]
    fn node_at_addresses_descendants() {
        let tree = sample();
        // Empty path is the root.
        assert_eq!(tree.node_at(&[]).unwrap().kind, "Box");
        assert_eq!(tree.node_at(&[0]).unwrap().kind, "Text");
        assert_eq!(tree.node_at(&[1]).unwrap().kind, "Box");
        assert_eq!(tree.node_at(&[1, 0]).unwrap().kind, "Text");
        assert_eq!(tree.node_at(&[1, 1]).unwrap().kind, "Text");
    }

    #[test]
    fn node_at_out_of_range_is_none() {
        let tree = sample();
        assert!(tree.node_at(&[2]).is_none());
        assert!(tree.node_at(&[0, 0]).is_none()); // leaf has no children
        assert!(tree.node_at(&[1, 2]).is_none());
    }

    #[test]
    fn find_returns_first_in_preorder() {
        let tree = sample();
        // The first "Text" in pre-order is the root's first child.
        assert_eq!(tree.find(|n| n.kind == "Text"), Some(alloc::vec![0]));
        // The first "Box" is the root itself (empty path).
        assert_eq!(tree.find(|n| n.kind == "Box"), Some(Vec::new()));
        assert!(tree.find(|n| n.kind == "Canvas").is_none());
    }

    #[test]
    fn find_all_lists_matches_in_preorder() {
        let tree = sample();
        assert_eq!(
            tree.find_all(|n| n.kind == "Text"),
            [alloc::vec![0], alloc::vec![1, 0], alloc::vec![1, 1]],
        );
        assert_eq!(
            tree.find_all(|n| n.kind == "Box"),
            [Vec::new(), alloc::vec![1]],
        );
        assert!(tree.find_all(|n| n.kind == "Canvas").is_empty());
    }

    #[test]
    fn found_paths_resolve_back_to_matching_nodes() {
        let tree = sample();
        for path in tree.find_all(|n| n.kind == "Text") {
            assert_eq!(tree.node_at(&path).unwrap().kind, "Text");
        }
    }

    // SplitMix64 for deterministic random tree shapes.
    fn next_rand(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn random_node(state: &mut u64, depth: usize) -> SnapshotNode {
        // Two kinds, roughly balanced.
        let kind = if next_rand(state).is_multiple_of(2) { "Box" } else { "Text" };
        let child_count = if depth == 0 {
            0
        } else {
            (next_rand(state) % 4) as usize
        };
        let mut children = Vec::new();
        for _ in 0..child_count {
            children.push(random_node(state, depth - 1));
        }
        branch(kind, children)
    }

    // Independent pre-order enumeration of (path, kind), used as the oracle.
    fn enumerate(node: &SnapshotNode, prefix: &mut Vec<usize>, out: &mut Vec<(Vec<usize>, String)>) {
        out.push((prefix.clone(), node.kind.clone()));
        for (index, child) in node.children.iter().enumerate() {
            prefix.push(index);
            enumerate(child, prefix, out);
            prefix.pop();
        }
    }

    #[test]
    fn search_matches_independent_preorder_enumeration() {
        let mut state = 0xDEAD_BEEF_0BAD_F00Du64;
        for _ in 0..200 {
            let tree = TreeSnapshot {
                root: random_node(&mut state, 5),
            };

            let mut all = Vec::new();
            enumerate(&tree.root, &mut Vec::new(), &mut all);

            for kind in ["Box", "Text"] {
                let expected: Vec<Vec<usize>> = all
                    .iter()
                    .filter(|(_, k)| k == kind)
                    .map(|(path, _)| path.clone())
                    .collect();
                let found = tree.find_all(|n| n.kind == kind);
                assert_eq!(found, expected);

                // `find` yields exactly the first `find_all` entry.
                assert_eq!(tree.find(|n| n.kind == kind), expected.first().cloned());

                // Every found path addresses a node of the requested kind.
                for path in &found {
                    assert_eq!(tree.node_at(path).unwrap().kind, *kind);
                }
            }
        }
    }
}
