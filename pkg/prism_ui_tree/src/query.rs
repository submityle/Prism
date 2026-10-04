//! Structural relationship queries over a [`Tree`]: depth, ancestor walks,
//! root resolution and the lowest common ancestor.
//!
//! These are read-only navigations layered on the parent links the tree
//! already maintains. They let higher Loom layers answer questions like "is
//! this focus target inside that overlay subtree?" or "what is the nearest
//! shared container of these two nodes?" without materialising paths by hand.
//!
//! All walks rely on the acyclic parent invariant the tree enforces: every
//! node has at most one parent and `append_child` detaches a node from its
//! previous parent first, so a parent chain can never loop.

use crate::arena::NodeId;
use crate::tree::Tree;

/// Iterator over the proper ancestors of a node, from the nearest parent up to
/// (and including) the root. Yields nothing for a root or missing node.
///
/// Created by [`Tree::ancestors`].
#[derive(Debug)]
pub struct Ancestors<'a, K, T> {
    tree: &'a Tree<K, T>,
    next: Option<NodeId>,
}

impl<K, T> Iterator for Ancestors<'_, K, T> {
    type Item = NodeId;

    #[inline]
    fn next(&mut self) -> Option<NodeId> {
        let current = self.next?;
        self.next = self.tree.parent(current);
        Some(current)
    }
}

impl<K, T> Tree<K, T> {
    /// Returns an iterator over the proper ancestors of `id`, nearest first.
    ///
    /// The node itself is not yielded. A root or missing node yields an empty
    /// iterator.
    #[inline]
    pub fn ancestors(&self, id: NodeId) -> Ancestors<'_, K, T> {
        Ancestors {
            tree: self,
            next: self.parent(id),
        }
    }

    /// The depth of `id`: the number of ancestors between it and its root.
    ///
    /// A root (detached) node has depth `0`. Returns `None` for a missing node.
    #[inline]
    pub fn depth(&self, id: NodeId) -> Option<usize> {
        if !self.contains(id) {
            return None;
        }
        Some(self.ancestors(id).count())
    }

    /// The topmost ancestor of `id` (its root), or `id` itself when it is
    /// already a root. Returns `None` for a missing node.
    pub fn root_of(&self, id: NodeId) -> Option<NodeId> {
        if !self.contains(id) {
            return None;
        }
        let mut current = id;
        while let Some(parent) = self.parent(current) {
            current = parent;
        }
        Some(current)
    }

    /// Whether `ancestor` is a *proper* ancestor of `descendant` (strictly
    /// above it in the same tree). A node is never its own ancestor, and a
    /// missing node is never related.
    pub fn is_ancestor(&self, ancestor: NodeId, descendant: NodeId) -> bool {
        if ancestor == descendant {
            return false;
        }
        self.ancestors(descendant).any(|node| node == ancestor)
    }

    /// The lowest (deepest) common ancestor of `a` and `b`, counting each node
    /// as an ancestor of itself.
    ///
    /// Returns `Some(a)` when `a == b`, the shallower node when one is an
    /// ancestor of the other, and `None` when either node is missing or the
    /// two live in different trees (no shared root).
    pub fn lowest_common_ancestor(&self, a: NodeId, b: NodeId) -> Option<NodeId> {
        let mut depth_a = self.depth(a)?;
        let mut depth_b = self.depth(b)?;
        let mut node_a = a;
        let mut node_b = b;

        // Lift the deeper node until both sit at the same depth.
        while depth_a > depth_b {
            node_a = self.parent(node_a)?;
            depth_a -= 1;
        }
        while depth_b > depth_a {
            node_b = self.parent(node_b)?;
            depth_b -= 1;
        }

        // Climb in lockstep until the paths meet. Reaching a root without
        // meeting means the nodes belong to different trees.
        while node_a != node_b {
            node_a = self.parent(node_a)?;
            node_b = self.parent(node_b)?;
        }
        Some(node_a)
    }
}

#[cfg(test)]
mod tests {
    use crate::Tree;
    use crate::arena::NodeId;
    use alloc::vec::Vec;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, hi: usize) -> usize {
            (self.next_u64() % hi as u64) as usize
        }
    }

    /// Independent oracle: the root-to-node path (root first, `id` last),
    /// built by collecting proper ancestors and reversing.
    fn root_path(tree: &Tree<u32, u32>, id: NodeId) -> Vec<NodeId> {
        let mut path: Vec<NodeId> = tree.ancestors(id).collect();
        path.reverse();
        path.push(id);
        path
    }

    /// Independent LCA oracle via the longest common prefix of the two root
    /// paths.
    fn lca_via_paths(tree: &Tree<u32, u32>, a: NodeId, b: NodeId) -> Option<NodeId> {
        let pa = root_path(tree, a);
        let pb = root_path(tree, b);
        let mut last = None;
        for (x, y) in pa.iter().zip(pb.iter()) {
            if x == y {
                last = Some(*x);
            } else {
                break;
            }
        }
        last
    }

    #[test]
    fn fixed_small_tree() {
        //        r
        //      /   \
        //     a     b
        //    / \     \
        //   c   d     e
        let mut tree: Tree<u32, u32> = Tree::new();
        let r = tree.create(None, 0);
        let a = tree.create(None, 1);
        let b = tree.create(None, 2);
        let c = tree.create(None, 3);
        let d = tree.create(None, 4);
        let e = tree.create(None, 5);
        tree.append_child(r, a);
        tree.append_child(r, b);
        tree.append_child(a, c);
        tree.append_child(a, d);
        tree.append_child(b, e);

        assert_eq!(tree.depth(r), Some(0));
        assert_eq!(tree.depth(a), Some(1));
        assert_eq!(tree.depth(c), Some(2));
        assert_eq!(tree.depth(e), Some(2));

        assert_eq!(tree.root_of(c), Some(r));
        assert_eq!(tree.root_of(r), Some(r));

        let anc: Vec<NodeId> = tree.ancestors(c).collect();
        assert_eq!(anc, [a, r]);
        assert_eq!(tree.ancestors(r).count(), 0);

        assert!(tree.is_ancestor(a, c));
        assert!(tree.is_ancestor(r, e));
        assert!(!tree.is_ancestor(c, a));
        assert!(!tree.is_ancestor(a, a));
        assert!(!tree.is_ancestor(a, e));

        assert_eq!(tree.lowest_common_ancestor(c, d), Some(a));
        assert_eq!(tree.lowest_common_ancestor(c, e), Some(r));
        assert_eq!(tree.lowest_common_ancestor(a, c), Some(a));
        assert_eq!(tree.lowest_common_ancestor(c, c), Some(c));
        assert_eq!(tree.lowest_common_ancestor(r, e), Some(r));
    }

    #[test]
    fn missing_and_cross_tree() {
        let mut tree: Tree<u32, u32> = Tree::new();
        let r1 = tree.create(None, 0);
        let a = tree.create(None, 1);
        tree.append_child(r1, a);
        // Second, independent tree.
        let r2 = tree.create(None, 2);
        let b = tree.create(None, 3);
        tree.append_child(r2, b);

        // Different trees share no root; the path oracle agrees.
        assert_eq!(tree.lowest_common_ancestor(a, b), None);
        assert_eq!(tree.lowest_common_ancestor(r1, r2), None);
        assert_eq!(lca_via_paths(&tree, a, b), tree.lowest_common_ancestor(a, b));

        // Missing node after removal.
        tree.remove_subtree(r2);
        assert_eq!(tree.depth(b), None);
        assert_eq!(tree.root_of(b), None);
        assert_eq!(tree.lowest_common_ancestor(a, b), None);
        assert!(!tree.is_ancestor(r2, b));
    }

    #[test]
    fn randomized_against_path_oracle() {
        let mut rng = SplitMix64(0x0A11_CE5E_u64 ^ 0x5EED);
        for _ in 0..1000 {
            let mut tree: Tree<u32, u32> = Tree::new();
            let count = 1 + rng.below(40);
            // Build a forest: each node either becomes a new root or a child of
            // a previously created node.
            let mut nodes: Vec<NodeId> = Vec::with_capacity(count);
            for i in 0..count {
                let id = tree.create(None, i as u32);
                if !nodes.is_empty() && rng.below(5) != 0 {
                    let parent = nodes[rng.below(nodes.len())];
                    tree.append_child(parent, id);
                }
                nodes.push(id);
            }

            // Check many random pairs against the independent path oracle.
            for _ in 0..24 {
                let a = nodes[rng.below(nodes.len())];
                let b = nodes[rng.below(nodes.len())];
                let got = tree.lowest_common_ancestor(a, b);
                let expected = lca_via_paths(&tree, a, b);
                assert_eq!(got, expected, "lca mismatch for {a:?},{b:?}");

                if let Some(lca) = got {
                    // The LCA is an ancestor-or-self of both endpoints.
                    assert!(lca == a || tree.is_ancestor(lca, a));
                    assert!(lca == b || tree.is_ancestor(lca, b));
                    // And it is no deeper than either endpoint.
                    let depth_lca = tree.depth(lca).unwrap();
                    assert!(depth_lca <= tree.depth(a).unwrap());
                    assert!(depth_lca <= tree.depth(b).unwrap());
                }

                // Cross-check is_ancestor against the LCA definition.
                let strict = got == Some(a) && a != b;
                assert_eq!(tree.is_ancestor(a, b), strict);

                // depth equals ancestor-path length and matches the oracle.
                assert_eq!(tree.depth(a), Some(tree.ancestors(a).count()));
                assert_eq!(tree.depth(a), Some(root_path(&tree, a).len() - 1));

                // root_of is the first node on the root path.
                assert_eq!(tree.root_of(a), Some(root_path(&tree, a)[0]));
            }
        }
    }
}
