//! Spatial hit testing over a tree of layout rectangles.
//!
//! `prism_ui`'s retained [`Element`](prism_ui::Element) tree carries no layout
//! geometry on its own, so hit testing operates on a parallel [`HitNode`] tree
//! that pairs each interactive node with its computed [`Rect`], a `z-index`,
//! and a `pointer-events` mode. Callers build this tree from the retained tree
//! plus layout results.
//!
//! # Model
//!
//! * A node is a hit candidate only when `point` falls inside its own
//!   rectangle, and descent into children only happens for nodes that contain
//!   the point (a clipped model). Children drawn outside their parent's bounds
//!   are therefore not hit; this keeps the traversal deterministic.
//! * Among sibling children the topmost is tried first, ordered by `z-index`
//!   descending with later document order winning ties, matching painters'
//!   order where the last-painted sibling is on top.
//! * A node whose `pointer-events` mode is [`PointerEvents::None`] is
//!   transparent to hits: it is never returned as a target and never appears
//!   in the path, but its descendants are still tested so events pass through
//!   to whatever lies beneath.
//!
//! The returned path runs from the root down to the target.

use alloc::vec;
use alloc::vec::Vec;

use crate::event::NodeId;
use crate::geometry::{rect_contains, Point, Rect};

/// How a node participates in hit testing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PointerEvents {
    /// The node can be hit and appears in the hit path.
    #[default]
    Auto,
    /// The node is transparent to hits; its descendants are still tested.
    None,
}

/// A node in the hit-test tree.
///
/// Build one of these per interactive node from the retained tree and layout
/// results, then pass the root to [`hit_test`].
#[derive(Clone, Debug, PartialEq)]
pub struct HitNode {
    id: NodeId,
    rect: Rect,
    z_index: i32,
    pointer_events: PointerEvents,
    children: Vec<HitNode>,
}

impl HitNode {
    /// Creates a leaf node with `z-index` zero and `pointer-events: auto`.
    pub fn new(id: NodeId, rect: Rect) -> Self {
        Self {
            id,
            rect,
            z_index: 0,
            pointer_events: PointerEvents::Auto,
            children: Vec::new(),
        }
    }

    /// Returns a copy with the `z-index` set to `z`.
    pub fn with_z_index(mut self, z: i32) -> Self {
        self.z_index = z;
        self
    }

    /// Returns a copy with the `pointer-events` mode set to `mode`.
    pub fn with_pointer_events(mut self, mode: PointerEvents) -> Self {
        self.pointer_events = mode;
        self
    }

    /// Appends a child, returning the updated node (builder style).
    pub fn child(mut self, child: HitNode) -> Self {
        self.children.push(child);
        self
    }

    /// Appends several children, returning the updated node.
    pub fn children<I: IntoIterator<Item = HitNode>>(mut self, children: I) -> Self {
        self.children.extend(children);
        self
    }

    /// Returns this node's identifier.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Returns this node's rectangle.
    pub fn rect(&self) -> Rect {
        self.rect
    }

    /// Returns this node's `z-index`.
    pub fn z_index(&self) -> i32 {
        self.z_index
    }

    /// Returns this node's `pointer-events` mode.
    pub fn pointer_events(&self) -> PointerEvents {
        self.pointer_events
    }

    /// Returns this node's children.
    pub fn child_nodes(&self) -> &[HitNode] {
        &self.children
    }

    /// Returns the indices of children ordered topmost-first.
    fn topmost_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.children.len()).collect();
        order.sort_by(|&a, &b| {
            let za = self.children[a].z_index;
            let zb = self.children[b].z_index;
            // Higher z-index is on top; for equal z the later sibling wins.
            zb.cmp(&za).then_with(|| b.cmp(&a))
        });
        order
    }

    /// Recursively computes the hit path rooted at this node.
    fn hit(&self, point: Point<f32>) -> Option<Vec<NodeId>> {
        if !rect_contains(&self.rect, point) {
            return None;
        }
        for index in self.topmost_order() {
            if let Some(sub) = self.children[index].hit(point) {
                return Some(self.prepend(sub));
            }
        }
        match self.pointer_events {
            PointerEvents::Auto => Some(vec![self.id]),
            PointerEvents::None => None,
        }
    }

    /// Prepends this node to `tail` unless it is transparent to hits.
    fn prepend(&self, mut tail: Vec<NodeId>) -> Vec<NodeId> {
        match self.pointer_events {
            PointerEvents::Auto => {
                let mut path = Vec::with_capacity(tail.len() + 1);
                path.push(self.id);
                path.append(&mut tail);
                path
            }
            PointerEvents::None => tail,
        }
    }
}

/// Returns the root-to-target hit path for `point`, or an empty vector when
/// nothing is hit.
///
/// See the [module documentation](self) for the exact hit-testing model.
pub fn hit_test(root: &HitNode, point: Point<f32>) -> Vec<NodeId> {
    root.hit(point).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Size;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::new(Point::new(x, y), Size::new(w, h))
    }

    fn node(id: u64, r: Rect) -> HitNode {
        HitNode::new(NodeId::new(id), r)
    }

    #[test]
    fn miss_returns_empty_path() {
        let root = node(1, rect(0.0, 0.0, 10.0, 10.0));
        assert!(hit_test(&root, Point::new(50.0, 50.0)).is_empty());
    }

    #[test]
    fn nested_hit_returns_full_path() {
        let root = node(1, rect(0.0, 0.0, 100.0, 100.0)).child(
            node(2, rect(10.0, 10.0, 30.0, 30.0)).child(node(3, rect(15.0, 15.0, 10.0, 10.0))),
        );
        let path = hit_test(&root, Point::new(18.0, 18.0));
        assert_eq!(path, vec![NodeId::new(1), NodeId::new(2), NodeId::new(3)]);
    }

    #[test]
    fn z_order_prefers_topmost_sibling() {
        // Two overlapping siblings; the one with higher z-index wins.
        let low = node(2, rect(0.0, 0.0, 50.0, 50.0)).with_z_index(1);
        let high = node(3, rect(0.0, 0.0, 50.0, 50.0)).with_z_index(5);
        let root = node(1, rect(0.0, 0.0, 100.0, 100.0)).children([low, high]);
        let path = hit_test(&root, Point::new(10.0, 10.0));
        assert_eq!(path, vec![NodeId::new(1), NodeId::new(3)]);
    }

    #[test]
    fn equal_z_prefers_later_sibling() {
        let first = node(2, rect(0.0, 0.0, 50.0, 50.0));
        let second = node(3, rect(0.0, 0.0, 50.0, 50.0));
        let root = node(1, rect(0.0, 0.0, 100.0, 100.0)).children([first, second]);
        let path = hit_test(&root, Point::new(10.0, 10.0));
        assert_eq!(path, vec![NodeId::new(1), NodeId::new(3)]);
    }

    #[test]
    fn pointer_events_none_passes_through() {
        // A transparent overlay sits on top but lets the hit fall through to
        // the sibling beneath it, and never appears in the path.
        let overlay = node(2, rect(0.0, 0.0, 50.0, 50.0))
            .with_z_index(10)
            .with_pointer_events(PointerEvents::None);
        let beneath = node(3, rect(0.0, 0.0, 50.0, 50.0)).with_z_index(0);
        let root = node(1, rect(0.0, 0.0, 100.0, 100.0)).children([beneath, overlay]);
        let path = hit_test(&root, Point::new(10.0, 10.0));
        assert_eq!(path, vec![NodeId::new(1), NodeId::new(3)]);
    }

    #[test]
    fn pointer_events_none_still_tests_children() {
        let child = node(3, rect(5.0, 5.0, 10.0, 10.0));
        let transparent = node(2, rect(0.0, 0.0, 50.0, 50.0))
            .with_pointer_events(PointerEvents::None)
            .child(child);
        let root = node(1, rect(0.0, 0.0, 100.0, 100.0)).child(transparent);
        let path = hit_test(&root, Point::new(8.0, 8.0));
        // Node 2 is skipped, but its child is still reachable.
        assert_eq!(path, vec![NodeId::new(1), NodeId::new(3)]);
    }

    #[test]
    fn accessors_report_builder_values() {
        let n = node(1, rect(0.0, 0.0, 10.0, 10.0))
            .with_z_index(4)
            .with_pointer_events(PointerEvents::None)
            .child(node(2, rect(1.0, 1.0, 2.0, 2.0)));
        assert_eq!(n.id(), NodeId::new(1));
        assert_eq!(n.z_index(), 4);
        assert_eq!(n.pointer_events(), PointerEvents::None);
        assert_eq!(n.child_nodes().len(), 1);
        assert_eq!(n.rect().right(), 10.0);
    }
}
