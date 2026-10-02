//! Incremental layout: recompute only the dirty subtrees.
//!
//! The full solver ([`LayoutTree::compute_layout`]) always re-solves the
//! entire tree. The incremental entry point
//! [`LayoutTree::compute_layout_incremental`] instead re-solves only the
//! maximal dirty subtrees, so steady-state cost tracks the amount of change
//! rather than the total node count (the §9.3 cost contract).
//!
//! # How dirty roots are found
//!
//! [`LayoutTree::mark_needs_layout`] propagates the geometry dirty bit up the
//! parent chain, stopping at the nearest relayout boundary. As a result every
//! dirty region forms a connected subtree whose topmost node is either the
//! document root or a relayout boundary. Those topmost nodes are the *dirty
//! roots*; re-solving each of them in isolation reproduces the exact geometry
//! a full solve would have produced, because a boundary's own size is fixed by
//! its own constraints and therefore unaffected by its parent.
//!
//! ```
//! use prism_ui_layout::{
//!     AvailableSpace, Dimension, LayoutStyle, LayoutTree, Size,
//! };
//!
//! let mut tree = LayoutTree::new();
//! // A content-measured leaf: it is *not* a boundary itself, so marking it
//! // dirty propagates up to its fixed-size wrapper.
//! let leaf = tree.new_leaf_with_measure(
//!     LayoutStyle::default(),
//!     |_known: Size<Option<f32>>, _avail: Size<AvailableSpace>| Size::new(40.0, 40.0),
//! );
//! // A fixed-size wrapper is a relayout boundary.
//! let boundary = tree.new_node(
//!     LayoutStyle {
//!         size: Size::new(Dimension::Points(80.0), Dimension::Points(80.0)),
//!         ..LayoutStyle::default()
//!     },
//!     &[leaf],
//! );
//! let sibling = tree.new_leaf(LayoutStyle {
//!     size: Size::new(Dimension::Points(10.0), Dimension::Points(10.0)),
//!     ..LayoutStyle::default()
//! });
//! let root = tree.new_node(LayoutStyle::default(), &[boundary, sibling]);
//!
//! let viewport =
//!     Size::new(AvailableSpace::Definite(200.0), AvailableSpace::Definite(200.0));
//! tree.compute_layout_incremental(root, viewport); // first frame: full solve
//!
//! // Change only the inner leaf; propagation stops at `boundary`.
//! tree.mark_needs_layout(leaf);
//! tree.compute_layout_incremental(root, viewport);
//!
//! assert!(tree.was_relayouted(leaf));
//! assert!(tree.was_relayouted(boundary));
//! assert!(!tree.was_relayouted(sibling)); // untouched subtree preserved
//! ```

use alloc::vec::Vec;

use crate::dirty::is_relayout_boundary;
use crate::geometry::{AvailableSpace, Size};
use crate::tree::{LayoutTree, NodeId};

impl LayoutTree {
    /// Recomputes layout, touching only the dirty subtrees.
    ///
    /// The first call (when the whole tree is still dirty) performs a full
    /// solve. Subsequent calls re-solve only the maximal dirty subtrees found
    /// via [`LayoutTree::mark_needs_layout`] propagation. Use
    /// [`LayoutTree::was_relayouted`] to inspect which nodes were recomputed.
    pub fn compute_layout_incremental(&mut self, root: NodeId, viewport: Size<AvailableSpace>) {
        self.begin_pass();

        let roots = self.dirty_roots(root);
        for dirty_root in roots {
            if dirty_root == root {
                self.resolve_subtree(root, viewport);
            } else {
                let available = self.boundary_available(dirty_root);
                self.resolve_subtree(dirty_root, available);
            }
        }

        self.clear_all_needs_layout();
    }

    /// Collects the maximal dirty subtree roots reachable from `root`.
    ///
    /// A node is a dirty root when it needs layout and its parent does not
    /// (or it has no parent).
    fn dirty_roots(&self, root: NodeId) -> Vec<NodeId> {
        let mut roots = Vec::new();
        for node in self.node_ids() {
            if !self.dirty_flags(node).needs_layout() {
                continue;
            }
            let is_root = match self.parent(node) {
                None => true,
                Some(parent) => !self.dirty_flags(parent).needs_layout(),
            };
            if is_root {
                roots.push(node);
            }
        }
        // Guarantee the document root is solved on the first (fully dirty)
        // frame even if arena order would place a descendant earlier.
        if roots.is_empty() && self.dirty_flags(root).needs_layout() {
            roots.push(root);
        }
        roots
    }

    /// Returns the available space a relayout boundary is solved against:
    /// exactly its own definite size.
    fn boundary_available(&self, node: NodeId) -> Size<AvailableSpace> {
        let style = self.style(node);
        debug_assert!(is_relayout_boundary(style));
        let width = style.size.width.resolve(None).unwrap_or(0.0);
        let height = style.size.height.resolve(None).unwrap_or(0.0);
        Size::new(
            AvailableSpace::Definite(width),
            AvailableSpace::Definite(height),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::{AvailableSpace, Dimension, Size};
    use crate::style::LayoutStyle;
    use crate::tree::{LayoutTree, NodeId};

    fn viewport() -> Size<AvailableSpace> {
        Size::new(
            AvailableSpace::Definite(200.0),
            AvailableSpace::Definite(200.0),
        )
    }

    fn fixed(w: f32, h: f32) -> LayoutStyle {
        LayoutStyle {
            size: Size::new(Dimension::Points(w), Dimension::Points(h)),
            ..LayoutStyle::default()
        }
    }

    /// Builds root -> [boundary -> leaf, sibling].
    ///
    /// The inner leaf is content-measured (not a boundary itself) so that
    /// marking it dirty propagates up to the fixed-size `boundary` wrapper,
    /// where propagation then stops.
    fn build() -> (LayoutTree, NodeId, NodeId, NodeId, NodeId) {
        let mut tree = LayoutTree::new();
        let leaf = tree.new_leaf_with_measure(
            LayoutStyle::default(),
            |_k: Size<Option<f32>>, _a: Size<AvailableSpace>| Size::new(40.0, 40.0),
        );
        let boundary = tree.new_node(fixed(80.0, 80.0), &[leaf]);
        let sibling = tree.new_leaf(fixed(10.0, 10.0));
        let root = tree.new_node(LayoutStyle::default(), &[boundary, sibling]);
        (tree, root, boundary, leaf, sibling)
    }

    #[test]
    fn first_frame_solves_everything() {
        let (mut tree, root, boundary, leaf, sibling) = build();
        tree.compute_layout_incremental(root, viewport());
        assert!(tree.was_relayouted(root));
        assert!(tree.was_relayouted(boundary));
        assert!(tree.was_relayouted(leaf));
        assert!(tree.was_relayouted(sibling));
    }

    #[test]
    fn local_change_only_touches_its_subtree() {
        let (mut tree, root, boundary, leaf, sibling) = build();
        tree.compute_layout_incremental(root, viewport());

        tree.mark_needs_layout(leaf);
        tree.compute_layout_incremental(root, viewport());

        assert!(tree.was_relayouted(leaf));
        assert!(tree.was_relayouted(boundary));
        assert!(!tree.was_relayouted(root));
        assert!(!tree.was_relayouted(sibling));
    }

    #[test]
    fn boundary_stops_upward_propagation() {
        let (mut tree, root, boundary, leaf, _sibling) = build();
        tree.compute_layout_incremental(root, viewport());

        tree.mark_needs_layout(leaf);
        assert!(tree.dirty_flags(leaf).needs_layout());
        assert!(tree.dirty_flags(boundary).needs_layout());
        assert!(!tree.dirty_flags(root).needs_layout());
    }

    #[test]
    fn needs_paint_does_not_trigger_relayout() {
        let (mut tree, root, _boundary, leaf, _sibling) = build();
        tree.compute_layout_incremental(root, viewport());

        tree.mark_needs_paint(leaf);
        assert!(tree.dirty_flags(leaf).needs_paint());
        assert!(!tree.dirty_flags(leaf).needs_layout());

        tree.compute_layout_incremental(root, viewport());
        // Nothing was dirty for layout, so no node is recomputed.
        assert!(!tree.was_relayouted(leaf));
        assert!(!tree.was_relayouted(root));
    }

    #[test]
    fn preserved_subtree_keeps_its_geometry() {
        let (mut tree, root, _boundary, leaf, sibling) = build();
        tree.compute_layout_incremental(root, viewport());
        let before = *tree.layout(sibling);

        tree.mark_needs_layout(leaf);
        tree.compute_layout_incremental(root, viewport());

        assert_eq!(*tree.layout(sibling), before);
    }

    #[test]
    fn boundary_subtree_root_keeps_position() {
        let (mut tree, root, boundary, leaf, _sibling) = build();
        tree.compute_layout_incremental(root, viewport());
        let boundary_before = *tree.layout(boundary);

        tree.mark_needs_layout(leaf);
        tree.compute_layout_incremental(root, viewport());

        // The boundary's position within its parent is preserved even though
        // its subtree was re-solved in isolation.
        assert_eq!(tree.layout(boundary).location, boundary_before.location);
        assert_eq!(tree.layout(boundary).size, boundary_before.size);
    }

    #[test]
    fn measure_cache_skips_identical_remeasure() {
        let mut tree = LayoutTree::new();
        let leaf = tree.new_leaf_with_measure(
            LayoutStyle::default(),
            |_k: Size<Option<f32>>, _a: Size<AvailableSpace>| Size::new(30.0, 15.0),
        );
        let root = tree.new_node(LayoutStyle::default(), &[leaf]);

        tree.compute_layout_incremental(root, viewport());
        let after_first = tree.measure_calls(leaf);
        assert!(after_first >= 1);

        // Re-solving with identical constraints must hit the cache.
        tree.mark_needs_layout(root);
        tree.compute_layout_incremental(root, viewport());
        assert_eq!(tree.measure_calls(leaf), after_first);
    }

    #[test]
    fn dirtying_a_leaf_invalidates_its_measure_cache() {
        let mut tree = LayoutTree::new();
        let leaf = tree.new_leaf_with_measure(
            LayoutStyle::default(),
            |_k: Size<Option<f32>>, _a: Size<AvailableSpace>| Size::new(30.0, 15.0),
        );
        let root = tree.new_node(LayoutStyle::default(), &[leaf]);

        tree.compute_layout_incremental(root, viewport());
        let after_first = tree.measure_calls(leaf);

        tree.mark_needs_layout(leaf);
        tree.compute_layout_incremental(root, viewport());
        assert!(tree.measure_calls(leaf) > after_first);
    }
}
