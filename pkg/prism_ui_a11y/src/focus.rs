//! Sequential focus (Tab) order computation.

use alloc::vec::Vec;

use prism_ui::Key;

use crate::tree::A11yTree;

/// The computed sequential focus order for a tree.
///
/// The order follows the web platform's tabbing rules:
///
/// 1. Nodes with a **positive** `tab_index` come first, ordered by ascending
///    `tab_index`; ties keep insertion order (a stable sort).
/// 2. Then nodes with `tab_index == 0`, in insertion order.
///
/// Nodes that are not focusable, have a **negative** `tab_index`, or are
/// disabled or hidden are skipped entirely. [`next`](FocusOrder::next) and
/// [`prev`](FocusOrder::prev) wrap around the ends.
#[derive(Clone, Debug, Default)]
pub struct FocusOrder {
    order: Vec<Key>,
}

impl FocusOrder {
    /// Computes the focus order for `tree`.
    #[must_use]
    pub fn compute(tree: &A11yTree) -> Self {
        let mut positive: Vec<(i32, usize, Key)> = Vec::new();
        let mut zero: Vec<Key> = Vec::new();

        for (position, node) in tree.iter().enumerate() {
            if !node.is_tab_stop() {
                continue;
            }
            if node.tab_index > 0 {
                positive.push((node.tab_index, position, node.key.clone()));
            } else {
                // `is_tab_stop` already rejected negative indices, so this is 0.
                zero.push(node.key.clone());
            }
        }

        // Stable by construction, but sort explicitly on (tab_index, position)
        // so the contract holds regardless of the sort's stability guarantee.
        positive.sort_by_key(|&(tab_index, position, _)| (tab_index, position));

        let mut order = Vec::with_capacity(positive.len() + zero.len());
        order.extend(positive.into_iter().map(|(_, _, key)| key));
        order.extend(zero);
        Self { order }
    }

    /// The ordered focusable keys.
    #[must_use]
    pub fn keys(&self) -> &[Key] {
        &self.order
    }

    /// The number of focusable stops.
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Returns `true` when there are no focusable stops.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The first focusable key, if any.
    #[must_use]
    pub fn first(&self) -> Option<&Key> {
        self.order.first()
    }

    /// The last focusable key, if any.
    #[must_use]
    pub fn last(&self) -> Option<&Key> {
        self.order.last()
    }

    /// The index of `key` within the order, if it is a focusable stop.
    #[must_use]
    pub fn position(&self, key: &Key) -> Option<usize> {
        self.order.iter().position(|k| k == key)
    }

    /// The next key after `current`, wrapping to [`first`](FocusOrder::first).
    ///
    /// If `current` is not in the order, returns the first key (so a fresh Tab
    /// press lands somewhere sensible). Returns `None` only for an empty order.
    #[must_use]
    pub fn next(&self, current: &Key) -> Option<&Key> {
        if self.order.is_empty() {
            return None;
        }
        match self.position(current) {
            Some(i) => {
                let n = self.order.len();
                self.order.get((i + 1) % n)
            }
            None => self.first(),
        }
    }

    /// The previous key before `current`, wrapping to
    /// [`last`](FocusOrder::last).
    ///
    /// If `current` is not in the order, returns the last key. Returns `None`
    /// only for an empty order.
    #[must_use]
    pub fn prev(&self, current: &Key) -> Option<&Key> {
        if self.order.is_empty() {
            return None;
        }
        match self.position(current) {
            Some(i) => {
                let n = self.order.len();
                self.order.get((i + n - 1) % n)
            }
            None => self.last(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FocusOrder;
    use crate::node::A11yNode;
    use crate::role::Role;
    use crate::state::AriaState;
    use crate::tree::A11yTree;
    use prism_ui::Key;

    fn stop(key: i64, tab_index: i32) -> A11yNode {
        A11yNode::builder(Key::Int(key), Role::Button)
            .tab_index(tab_index)
            .build()
    }

    #[test]
    fn positive_before_zero_and_ascending() {
        let mut tree = A11yTree::new();
        tree.insert(stop(1, 0));
        tree.insert(stop(2, 3));
        tree.insert(stop(3, 1));
        tree.insert(stop(4, 0));

        let order = FocusOrder::compute(&tree);
        let keys = order.keys().to_vec();
        assert_eq!(keys, [Key::Int(3), Key::Int(2), Key::Int(1), Key::Int(4)]);
    }

    #[test]
    fn equal_positive_keep_insertion_order() {
        let mut tree = A11yTree::new();
        tree.insert(stop(10, 2));
        tree.insert(stop(20, 2));
        tree.insert(stop(30, 2));
        let order = FocusOrder::compute(&tree);
        assert_eq!(order.keys(), [Key::Int(10), Key::Int(20), Key::Int(30)]);
    }

    #[test]
    fn skips_disabled_hidden_negative_and_unfocusable() {
        let mut tree = A11yTree::new();
        tree.insert(stop(1, 0));
        tree.insert(
            A11yNode::builder(Key::Int(2), Role::Button)
                .state(AriaState::new().disabled(true))
                .build(),
        );
        tree.insert(
            A11yNode::builder(Key::Int(3), Role::Button)
                .state(AriaState::new().hidden(true))
                .build(),
        );
        tree.insert(stop(4, -1));
        tree.insert(A11yNode::builder(Key::Int(5), Role::Group).build());
        tree.insert(stop(6, 0));

        let order = FocusOrder::compute(&tree);
        assert_eq!(order.keys(), [Key::Int(1), Key::Int(6)]);
    }

    #[test]
    fn next_prev_wrap_around() {
        let mut tree = A11yTree::new();
        tree.insert(stop(1, 0));
        tree.insert(stop(2, 0));
        tree.insert(stop(3, 0));
        let order = FocusOrder::compute(&tree);

        assert_eq!(order.next(&Key::Int(1)), Some(&Key::Int(2)));
        assert_eq!(order.next(&Key::Int(3)), Some(&Key::Int(1)));
        assert_eq!(order.prev(&Key::Int(1)), Some(&Key::Int(3)));
        assert_eq!(order.prev(&Key::Int(2)), Some(&Key::Int(1)));
    }

    #[test]
    fn first_last_and_unknown_current() {
        let mut tree = A11yTree::new();
        tree.insert(stop(1, 0));
        tree.insert(stop(2, 0));
        let order = FocusOrder::compute(&tree);

        assert_eq!(order.first(), Some(&Key::Int(1)));
        assert_eq!(order.last(), Some(&Key::Int(2)));
        // Unknown current jumps to the ends.
        assert_eq!(order.next(&Key::Int(99)), Some(&Key::Int(1)));
        assert_eq!(order.prev(&Key::Int(99)), Some(&Key::Int(2)));
    }

    #[test]
    fn empty_order_has_no_navigation() {
        let tree = A11yTree::new();
        let order = FocusOrder::compute(&tree);
        assert!(order.is_empty());
        assert_eq!(order.first(), None);
        assert_eq!(order.next(&Key::Int(1)), None);
        assert_eq!(order.prev(&Key::Int(1)), None);
    }
}
