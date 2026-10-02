//! Focus traversal with a wrapping tab order.
//!
//! A [`FocusRing`] holds the set of focusable nodes together with their
//! `tabindex`. [`FocusRing::focus_next`] and [`FocusRing::focus_prev`] move the focus
//! forward and backward, wrapping around the ends.
//!
//! # Ordering policy
//!
//! Nodes are ordered by `tabindex` ascending, and ties are broken by insertion
//! (document) order. Unlike the web platform, this does not special-case a
//! `tabindex` of `0` to sort after positive values; a plain ascending order is
//! used because it is simpler and fully deterministic. Insert nodes in document
//! order and give them equal `tabindex` to reproduce natural document order.

use alloc::vec::Vec;

use crate::event::NodeId;

/// A focusable node and its `tabindex`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FocusEntry {
    id: NodeId,
    tabindex: i32,
    insertion: usize,
}

/// An ordered ring of focusable nodes supporting wrapping traversal.
#[derive(Default)]
pub struct FocusRing {
    entries: Vec<FocusEntry>,
    current: Option<NodeId>,
}

impl FocusRing {
    /// Creates an empty focus ring.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a focusable node with the given `tabindex`.
    ///
    /// Insertion order is remembered and used to break `tabindex` ties.
    pub fn add(&mut self, id: NodeId, tabindex: i32) {
        let insertion = self.entries.len();
        self.entries.push(FocusEntry {
            id,
            tabindex,
            insertion,
        });
    }

    /// Returns the number of focusable nodes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when there are no focusable nodes.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the currently focused node, if any.
    pub fn current(&self) -> Option<NodeId> {
        self.current
    }

    /// Returns the node ids in traversal order.
    pub fn order(&self) -> Vec<NodeId> {
        let mut sorted = self.entries.clone();
        sorted.sort_by(|a, b| {
            a.tabindex
                .cmp(&b.tabindex)
                .then(a.insertion.cmp(&b.insertion))
        });
        let mut ids = Vec::with_capacity(sorted.len());
        for entry in sorted {
            ids.push(entry.id);
        }
        ids
    }

    /// Sets focus to `id` when it is part of the ring, returning whether it
    /// was found.
    pub fn focus(&mut self, id: NodeId) -> bool {
        if self.entries.iter().any(|e| e.id == id) {
            self.current = Some(id);
            true
        } else {
            false
        }
    }

    /// Clears the current focus.
    pub fn blur(&mut self) {
        self.current = None;
    }

    /// Moves focus to the next node, wrapping to the first, and returns it.
    pub fn focus_next(&mut self) -> Option<NodeId> {
        self.step(1)
    }

    /// Moves focus to the previous node, wrapping to the last, and returns it.
    pub fn focus_prev(&mut self) -> Option<NodeId> {
        self.step(-1)
    }

    /// Advances the focus by `delta` positions in traversal order.
    fn step(&mut self, delta: isize) -> Option<NodeId> {
        let order = self.order();
        if order.is_empty() {
            self.current = None;
            return None;
        }
        let len = order.len();
        let next_index = match self
            .current
            .and_then(|id| order.iter().position(|&o| o == id))
        {
            Some(pos) => {
                let pos = pos as isize + delta;
                // Wrap into range without using the remainder of a negative.
                let len_i = len as isize;
                (((pos % len_i) + len_i) % len_i) as usize
            }
            None => {
                if delta >= 0 {
                    0
                } else {
                    len - 1
                }
            }
        };
        let id = order[next_index];
        self.current = Some(id);
        Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_sorts_by_tabindex_then_insertion() {
        let mut ring = FocusRing::new();
        ring.add(NodeId::new(1), 2);
        ring.add(NodeId::new(2), 1);
        ring.add(NodeId::new(3), 1);
        assert_eq!(
            ring.order(),
            [NodeId::new(2), NodeId::new(3), NodeId::new(1)]
        );
    }

    #[test]
    fn next_wraps_around() {
        let mut ring = FocusRing::new();
        ring.add(NodeId::new(1), 0);
        ring.add(NodeId::new(2), 0);
        assert_eq!(ring.focus_next(), Some(NodeId::new(1)));
        assert_eq!(ring.focus_next(), Some(NodeId::new(2)));
        assert_eq!(ring.focus_next(), Some(NodeId::new(1)));
    }

    #[test]
    fn prev_wraps_from_start_to_end() {
        let mut ring = FocusRing::new();
        ring.add(NodeId::new(1), 0);
        ring.add(NodeId::new(2), 0);
        ring.add(NodeId::new(3), 0);
        assert_eq!(ring.focus_prev(), Some(NodeId::new(3)));
        assert_eq!(ring.focus_prev(), Some(NodeId::new(2)));
    }

    #[test]
    fn focus_requires_membership() {
        let mut ring = FocusRing::new();
        ring.add(NodeId::new(1), 0);
        assert!(ring.focus(NodeId::new(1)));
        assert_eq!(ring.current(), Some(NodeId::new(1)));
        assert!(!ring.focus(NodeId::new(9)));
        assert_eq!(ring.current(), Some(NodeId::new(1)));
        ring.blur();
        assert_eq!(ring.current(), None);
    }

    #[test]
    fn empty_ring_has_no_focus() {
        let mut ring = FocusRing::new();
        assert!(ring.is_empty());
        assert_eq!(ring.focus_next(), None);
        assert_eq!(ring.len(), 0);
    }
}
